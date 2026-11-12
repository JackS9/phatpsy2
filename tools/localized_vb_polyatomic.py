#!/usr/bin/env python3
"""Exact-integral localized-orbital VB for small polyatomics (bonded region only).

Wavefunction: doubly occupied one-center inactive orbitals (cores, lone pairs) and one
spin-coupled pair per bond (hybrid on the heavy atom + orbital on the partner), with the
CI taken over the span of the one-center bond orbitals (covalent + ionic structures).
Every orbital is expanded only in its own atom's STOs; exact multi-center integrals from
phatpsy.multicenter_integrals; orbitals optimized with the analytic gradient.

Usage: PYTHONPATH=. .venv/bin/python tools/localized_vb_polyatomic.py H2O ADF-TZP [R_OH] [theta_deg]
"""
from __future__ import annotations

import contextlib
import io
import sys
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from phatpsy.localized_vb import (atom_basis, cas_energy_and_gradient, orthonormal_mo_guess,
                                  pyscf_mean_field)
from phatpsy.multicenter_integrals import MolecularAOIntegrals, molecular_ao_integrals

HEV = 27.211386245988


def atomic_orbitals(ints, atom, nelec, unpaired):
    """ROHF orbitals of an isolated atom in its own STO block (natural intuitive AOs)."""
    from phatpsy.localized_vb import AOIntegrals
    idx = np.where(ints.atom_of == atom)[0]
    sub = np.ix_(idx, idx)
    # bare one-center Hamiltonian: kinetic + own-nucleus attraction (from the basis itself)
    from phatpsy.two_center_integrals import ProlateGrid
    Z = ints.charges[atom]
    c = ints.coords[atom]
    far = c + np.array([0.0, 0.0, 3.0])
    g = ProlateGrid(c, far, zeta_min=min(ints.basis[i].zeta for i in idx), n_eta=48, n_phi=16)
    from phatpsy.multicenter_integrals import _eval, _kin
    chi = np.array([_eval(ints.basis[i], g.X, g.Y, g.Z) for i in idx])
    tchi = np.array([_kin(ints.basis[i], g.X, g.Y, g.Z) for i in idx])
    w = g.weights
    T = np.einsum("pijk,qijk,ijk->pq", chi, tchi, w); T = 0.5 * (T + T.T)
    Vn = np.einsum("pijk,qijk,ijk->pq", chi, chi, w * (-Z / g.r_a))
    h = T + Vn
    a = AOIntegrals([ints.basis[i] for i in idx], np.zeros(len(idx), int), ints.S[sub], T, Vn,
                    np.zeros_like(h), ints.eri[np.ix_(idx, idx, idx, idx)], (1.0, 0.0), 1e9)
    na = (nelec + unpaired) // 2
    nb = nelec - na
    with contextlib.redirect_stdout(io.StringIO()):
        mf = pyscf_mean_field(a, (na, nb), "ROHF" if na != nb else "RHF", h=h)
    return idx, mf


def p_vector(ints, atom, mo_p_coef_by_shell, direction):
    """Directed valence p orbital along a unit vector, using the real-harmonic phase
    convention of the Fortran core / evaluate_real_sto (p_x, p_y carry -1)."""
    v = np.zeros(ints.nao)
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    for k, f in enumerate(ints.basis):
        if ints.atom_of[k] != atom or f.l != 1:
            continue
        c = mo_p_coef_by_shell.get((f.n, f.zeta), 0.0)
        comp = {1: -d[0], -1: -d[1], 0: d[2]}[f.m]
        v[k] = c * comp
    return v


@dataclass
class Spec:
    atom: int
    role: str      # "inactive" / "active"


class PolyVB:
    def __init__(self, ints, specs, nelec_act):
        self.ints, self.specs, self.nelec_act = ints, specs, nelec_act
        self.masks = [np.where(ints.atom_of == s.atom)[0] for s in specs]
        self.off = np.cumsum([0] + [len(m) for m in self.masks])

    def orbitals(self, x):
        out = []
        for k, m in enumerate(self.masks):
            v = np.zeros(self.ints.nao)
            v[m] = x[self.off[k]:self.off[k + 1]]
            out.append(v)
        return out

    def pack(self, orbs):
        return np.concatenate([o[m] for o, m in zip(orbs, self.masks)])

    def split(self, orbs):
        Xi = np.array([o for o, s in zip(orbs, self.specs) if s.role == "inactive"]).T
        Xa = np.array([o for o, s in zip(orbs, self.specs) if s.role == "active"]).T
        return Xi, Xa

    def fg(self, x):
        orbs = self.orbitals(x)
        Xi, Xa = self.split(orbs)
        e, Gi, Ga = cas_energy_and_gradient(Xi, Xa, self.ints, self.nelec_act)
        gi = iter(Gi.T); ga = iter(Ga.T)
        grads = [next(gi) if s.role == "inactive" else next(ga) for s in self.specs]
        return e, np.concatenate([g[m] for g, m in zip(grads, self.masks)])


def water(basis_name="ADF-TZP", R=1.8088, theta=104.52):
    th = np.radians(theta) / 2
    O = np.zeros(3)
    H1 = np.array([0.0, R * np.sin(th), R * np.cos(th)])
    H2 = np.array([0.0, -R * np.sin(th), R * np.cos(th)])
    atoms = [(8.0, O, atom_basis(O, "O", basis_name)), (1.0, H1, atom_basis(H1, "H", basis_name)),
             (1.0, H2, atom_basis(H2, "H", basis_name))]
    return atoms


def main():
    basis_name = sys.argv[2] if len(sys.argv) > 2 else "ADF-TZP"
    R = float(sys.argv[3]) if len(sys.argv) > 3 else 1.8088
    theta = float(sys.argv[4]) if len(sys.argv) > 4 else 104.52
    atoms = water(basis_name, R, theta)
    t = time.time()
    ints = molecular_ao_integrals(atoms)
    print(f"H2O {basis_name} R={R} theta={theta}: integrals {time.time()-t:.0f}s, nao={ints.nao}", flush=True)

    # Intuitive start: atomic ROHF orbitals; sp3-like O hybrids toward each H
    io_, mfO = atomic_orbitals(ints, 0, 8, 2)
    CO = np.zeros((ints.nao, mfO.mo_coeff.shape[1]))
    CO[io_] = mfO.mo_coeff
    # identify 1s (lowest), 2s (second) and the radial shape of 2p from the O ROHF orbitals
    c1s, c2s = CO[:, 0], CO[:, 1]
    p_mo = CO[:, 2]
    pshell = {}
    for k in io_:
        f = ints.basis[k]
        if f.l == 1:
            pshell[(f.n, f.zeta)] = pshell.get((f.n, f.zeta), 0.0) + abs(p_mo[k])
    norm = np.sqrt(sum(v * v for v in pshell.values()))
    pshell = {k: v / norm for k, v in pshell.items()}
    O, H1, H2 = (np.asarray(a[1]) for a in atoms)
    d1, d2 = (H1 - O) / np.linalg.norm(H1 - O), (H2 - O) / np.linalg.norm(H2 - O)
    bis = -(d1 + d2) / np.linalg.norm(d1 + d2)
    nrm = np.cross(d1, d2); nrm /= np.linalg.norm(nrm)
    hyb = lambda d, s: s * c2s + np.sqrt(1 - s * s) * p_vector(ints, 0, pshell, d)
    b1, b2 = hyb(d1, 0.5), hyb(d2, 0.5)
    lp_s = hyb(bis, 0.6)                        # in-plane s-rich lone pair
    lp_p = p_vector(ints, 0, pshell, nrm)       # out-of-plane p lone pair
    hs = []
    for ia in (1, 2):
        ih, mfH = atomic_orbitals(ints, ia, 1, 1)
        v = np.zeros(ints.nao); v[ih] = mfH.mo_coeff[:, 0]; hs.append(v)
    specs = [Spec(0, "inactive"), Spec(0, "inactive"), Spec(0, "inactive"),
             Spec(0, "active"), Spec(0, "active"), Spec(1, "active"), Spec(2, "active")]
    orbs = [c1s, lp_s, lp_p, b1, b2, hs[0], hs[1]]
    model = PolyVB(ints, specs, (2, 2))
    x0 = model.pack(orbs)
    t = time.time()
    e0, _ = model.fg(x0)
    res = minimize(model.fg, x0, jac=True, method="BFGS", options={"maxiter": 800, "gtol": 1e-6})
    print(f"  localized VB  start E={e0:.6f}  optimized E={res.fun:.6f}  nit={res.nit} ok={res.success} "
          f"({time.time()-t:.0f}s)", flush=True)

    with contextlib.redirect_stdout(io.StringIO()):
        rhf = pyscf_mean_field(ints, (5, 5), "RHF")
        from pyscf import mcscf
        Xi, Xa = model.split(model.orbitals(res.x))
        mc = mcscf.CASSCF(rhf, 4, 4)
        mc.conv_tol = 1e-9
        mc.kernel(mo_coeff=orthonormal_mo_guess(Xi, Xa, ints.S))
    _, mfH1 = atomic_orbitals(ints, 1, 1, 1)
    e_at = mfO.e_tot + 2 * mfH1.e_tot
    print(f"  atoms (ROHF O 3P + 2 H, same basis): {e_at:.6f}")
    for name, e in (("localized VB", res.fun), ("RHF", rhf.e_tot), ("CASSCF(4,4)", mc.e_tot)):
        print(f"  {name:13s} E={e:.6f}  atomization D_e={(e_at - e) * HEV:7.3f} eV")
    print("  experiment D_e = 10.08 eV (232.6 kcal/mol, incl. ZPE-corrected)")
    np.save("/tmp/h2o_lvb_x.npy", res.x)


if __name__ == "__main__":
    main()
