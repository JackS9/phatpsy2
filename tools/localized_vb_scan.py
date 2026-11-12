#!/usr/bin/env python3
"""Option B, first test: exact-integral localized perfect-pairing VB for homonuclear and
heteronuclear diatomics, with strictly one-center, symmetry-adapted orbitals optimized
under the exact energy.  References in the identical exact STO basis: RHF, UHF (PySCF
with custom integrals) and separated-atom ROHF.

Usage: PYTHONPATH=. .venv/bin/python tools/localized_vb_scan.py N N DZ 1.9,2.0744,2.3,2.6,3.0
"""
from __future__ import annotations

import contextlib
import io
import sys
import time

import numpy as np

from phatpsy.localized_vb import (AOIntegrals, LocalizedVBModel, OrbitalSpec, atom_basis,
                                  diatomic_ao_integrals, optimize, orthonormal_mo_guess,
                                  pyscf_mean_field, symmetry_mask)

HEV = 27.211386245988
VALENCE = {  # element -> (n_inactive_sigma, unpaired sigma?, unpaired pi_x?, unpaired pi_y?) for PP bonding
    "N": dict(inactive_sigma=2, pairs=("sigma", "pi_x", "pi_y")),
    "C": dict(inactive_sigma=2, pairs=("sigma", "pi_x", "pi_y")),   # C(3P)->valence state for CO triple bond
    "O": dict(inactive_sigma=2, pairs=("sigma", "pi_x", "pi_y")),
    "F": dict(inactive_sigma=2, inactive_pi=("pi_x", "pi_y"), pairs=("sigma",)),
    "H": dict(inactive_sigma=0, pairs=("sigma",)),
    "Li": dict(inactive_sigma=1, pairs=("sigma",)),
}
NELEC = {"H": 1, "Li": 3, "C": 6, "N": 7, "O": 8, "F": 9}


def atom_reference(ints: AOIntegrals, atom: int, nelec: int, mult_unpaired: int) -> float:
    idx = np.where(ints.atom_of == atom)[0]
    sub = np.ix_(idx, idx)
    V = ints.V_A if atom == 0 else ints.V_B
    h = ints.T[sub] + V[sub]
    eri = ints.eri[np.ix_(idx, idx, idx, idx)]
    a = AOIntegrals([ints.basis[i] for i in idx], np.zeros(len(idx), int), ints.S[sub], ints.T[sub],
                    V[sub], np.zeros_like(h), eri, (1.0, 0.0), 1e9)
    na = (nelec + mult_unpaired) // 2
    nb = nelec - na
    mf = pyscf_mean_field(a, (na, nb), "ROHF" if na != nb else "RHF", h=h)
    return float(mf.e_tot)


def initial_orbitals(model: LocalizedVBModel, rng_seed: int = 0):
    """Hydrogenic-ish start: inactive 1s = tightest s, lone pair = diffuse s - pz mix,
    bond sigma = diffuse s + pz toward partner, pi = diffuse p."""
    ints = model.ints
    orbs = []
    for s in model.specs:
        mask = model.masks[[i for i, t in enumerate(model.specs) if t is s][0]]
        v = np.zeros(ints.nao)
        fs = [ints.basis[i] for i in mask]
        toward = 1.0 if s.atom == 0 else -1.0     # A at lower z
        if s.sym == "sigma":
            ls = [k for k, f in zip(mask, fs) if f.l == 0]
            lp = [k for k, f in zip(mask, fs) if f.l == 1]
            if s.role == "inactive" and s.pair == -1:          # core
                k = max(ls, key=lambda i: ints.basis[i].zeta); v[k] = 1.0
            else:
                ks = [k for k in ls if ints.basis[k].n >= 2] or ls
                for k in ks:
                    v[k] = 0.5
                for k in lp:
                    v[k] = (0.6 if s.role != "inactive" else -0.6) * toward
        else:
            for k, f in zip(mask, fs):
                if f.l == 1:
                    v[k] = 0.7
        orbs.append(v)
    return orbs


def build_model(ints, el_a, el_b, mirror):
    specs = []
    for atom, el in ((0, el_a), (1, el_b)):
        val = VALENCE[el]
        for k in range(val["inactive_sigma"]):
            specs.append(OrbitalSpec(atom, "sigma", "inactive", pair=-(k + 1)))
        for k, sym in enumerate(val.get("inactive_pi", ())):
            specs.append(OrbitalSpec(atom, sym, "inactive", pair=-(10 + k)))
        for k, sym in enumerate(val["pairs"]):
            specs.append(OrbitalSpec(atom, sym, "pair-a" if atom == 0 else "pair-b", pair=k))
    return LocalizedVBModel(ints, specs, mirror=mirror)


def run_point(el_a, el_b, basis, R, x0=None, methods=("pp", "covalent", "cas")):
    bA = atom_basis((0.0, 0.0, 0.0), el_a, basis)
    bB = atom_basis((0.0, 0.0, R), el_b, basis)
    za, zb = float(NELEC[el_a]), float(NELEC[el_b])
    t = time.time()
    ints = diatomic_ao_integrals(bA, bB, za, zb)
    t_int = time.time() - t
    mirror = el_a == el_b
    out = dict(R=R, t_int=t_int, ints=ints, E={}, x={})
    x = None if x0 is None else x0.get(methods[0])
    for meth in methods:
        model = build_model(ints, el_a, el_b, mirror)
        model.method = meth
        if x is None:
            x = model.pack(initial_orbitals(model))
        t = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            res = optimize(model, x, maxiter=600, gtol=1e-6)
        x = res.x
        out["E"][meth] = float(res.fun)
        out["x"][meth] = res.x
        print(f"   R={R:6.3f} {meth:9s} E={res.fun:.6f} nit={res.nit} ok={res.success} ({time.time()-t:.0f}s)", flush=True)
    nel = NELEC[el_a] + NELEC[el_b]
    with contextlib.redirect_stdout(io.StringIO()):
        rhf = pyscf_mean_field(ints, (nel // 2, nel - nel // 2), "RHF")
        from pyscf import mcscf
        nact = 2 * len(VALENCE[el_a]["pairs"])
        mc = mcscf.CASSCF(rhf, nact, nact)
        mc.conv_tol = 1e-9
        # Seed CASSCF with the localized inactive/active spans so that it cannot land on a
        # different active space (e.g. a pi lone pair for HF) than the localized VB.
        model = build_model(ints, el_a, el_b, mirror)
        orbs = model.orbitals(out["x"][methods[-1]])
        ina = [o for o, s in zip(orbs, model.specs) if s.role == "inactive"]
        inactive, pairs = model.split(orbs)
        Xa = np.array([a for a, _ in pairs] + [b for _, b in pairs]).T
        guess = orthonormal_mo_guess(inactive, Xa, ints.S)
        mc.kernel(mo_coeff=guess)
    out["E"]["rhf"] = float(rhf.e_tot)
    out["E"]["casscf"] = float(mc.e_tot)
    return out


def main():
    el_a, el_b, basis = sys.argv[1], sys.argv[2], sys.argv[3]
    Rs = [float(r) for r in sys.argv[4].split(",")]
    methods = tuple(sys.argv[5].split(",")) if len(sys.argv) > 5 else ("pp", "covalent", "cas")
    print(f"=== localized VB (one-center orbitals, exact integrals): {el_a}{el_b} {basis} ===")
    x0 = None
    rows = []
    for R in Rs:
        r = run_point(el_a, el_b, basis, R, x0, methods)
        x0 = r["x"]
        rows.append(r)
    ints = rows[0]["ints"]
    unp = {"N": 3, "C": 2, "O": 2, "F": 1, "H": 1, "Li": 1}
    e_atoms = atom_reference(ints, 0, NELEC[el_a], unp[el_a]) + atom_reference(ints, 1, NELEC[el_b], unp[el_b])
    print(f"separated atoms (ROHF ground states, same basis): {e_atoms:.6f}")
    keys = list(methods) + ["rhf", "casscf"]
    print("   R      " + "  ".join(f"{k:>10s}" for k in keys) + "    (binding energy, eV; + = bound)")
    for r in rows:
        print(f"  {r['R']:6.3f}  " + "  ".join(f"{(e_atoms - r['E'][k]) * HEV:10.3f}" for k in keys))
    print("   R      " + "  ".join(f"{k:>10s}" for k in keys) + "    (total energy, Ha)")
    for r in rows:
        print(f"  {r['R']:6.3f}  " + "  ".join(f"{r['E'][k]:10.5f}" for k in keys))


if __name__ == "__main__":
    main()

