#!/usr/bin/env python3
# Usage: PYTHONPATH=. .venv/bin/python tools/crosscheck_determinant_pyscf.py
"""Independent cross-check of determinant_energy: rebuild the same PHATPSY localized
determinant in PySCF using a many-Gaussian least-squares fit of every STO."""
import contextlib, io, math, sys
import numpy as np
from pyscf import gto, scf
from phatpsy.builder import build_diatomic
from phatpsy.driver import PhatpsyDriver
from phatpsy.two_center_integrals import ProlateGrid
from phatpsy.vb_correction import occupied_spin_orbitals, determinant_energy

BOHR = 0.52917721092


def fit_sto(n, l, zeta, K=14):
    """Fit r^(n-1-l) exp(-zeta r) by sum_k c_k exp(-a_k r^2) (r^l carried by the GTO);
    validated: 1s/2s/2p kinetic and nuclear-attraction integrals to ~1e-5."""
    a = zeta ** 2 * np.geomspace(0.02, 300.0, K)
    r = np.geomspace(1e-4, 40.0 / zeta, 4000)
    w = r ** (2 + 2 * l) * np.gradient(r)
    f = r ** (n - 1 - l) * np.exp(-zeta * r)
    G = np.exp(-np.outer(r ** 2, a))
    sw = np.sqrt(w)
    c = np.linalg.lstsq(G * sw[:, None], f * sw, rcond=1e-14)[0]
    return a, c


def pyscf_mol(sym, X, shells_by_atom):
    basis = {}
    for tag, shells in shells_by_atom.items():
        bl = []
        for (n, l, z) in shells:
            a, c = fit_sto(n, l, z)
            # PySCF coefficients multiply normalized primitives: convert from raw.
            norm = np.array([gto.gto_norm(l, ai) for ai in a])
            bl.append([l] + [[float(ai), float(ci / ni)] for ai, ci, ni in zip(a, c, norm)])
        basis[tag] = bl
    return gto.M(atom=[[f"{sym[0]}1", tuple(X[:, 0])], [f"{sym[1]}2", tuple(X[:, 1])]],
                 basis=basis, unit="Bohr", spin=0, charge=0, cart=False, verbose=0)


def main(a, b, R, basis_name):
    d = PhatpsyDriver(quiet=True)
    deck = build_diatomic(a, b, bond_length=R, basis=basis_name, seed_hybrids=True)
    deck.control.convergence = 1e-8; deck.control.max_cycles = 300
    with contextlib.redirect_stdout(io.StringIO()):
        st = d.run_deck(deck)
    sos = occupied_spin_orbitals(st)
    X = np.asarray(st["atom_coordinates"]); Z = np.asarray(st["atom_charges"])[0]
    g = ProlateGrid(X[:, 0], X[:, 1], zeta_min=1.0, n_eta=64)
    mine = determinant_energy(g, sos, float(Z[0]), float(Z[1]))["E"]

    # Shell lists per atom in PHATPSY order
    shells = {}
    for atom, sym in ((1, a), (2, b)):
        bas = next(so["basis"] for so in sos if so["atom"] == atom)
        seen = []
        for f in bas:
            key = (f.n, f.l, f.zeta)
            if key not in seen:
                seen.append(key)
        shells[f"{sym}{atom}"] = seen
    # Geometry: PHATPSY puts the atoms where it likes; rotate so A->B is +z.
    mol = pyscf_mol((a, b), X, shells)
    # AO map: PHATPSY (shell, ml=-l..l) -> PySCF ao index; PySCF p order (x,y,z)
    labels = mol.ao_labels(fmt=False)
    def pyscf_index(atom, shell_i, l, ml):
        # PySCF lists shells per atom in input order; within p: x(0),y(1),z(2)
        offs = mol.offset_ao_by_atom()[atom - 1][2]
        k = offs
        for (n, ll, z) in shells[list(shells)[atom - 1]][:shell_i]:
            k += 2 * ll + 1
        if l == 0:
            return k, 1.0
        if l == 1:
            comp = {1: 0, -1: 1, 0: 2}[ml]
            return k + comp, (-1.0 if ml != 0 else 1.0)   # Condon-Shortley phase
        raise NotImplementedError
    nao = mol.nao_nr()
    C = {0: [], 1: []}
    for so in sos:
        v = np.zeros(nao)
        sh = shells[list(shells)[so["atom"] - 1]]
        for f, c in zip(so["basis"], so["coef"]):
            si = sh.index((f.n, f.l, f.zeta))
            k, ph = pyscf_index(so["atom"], si, f.l, f.m)
            v[k] += ph * c
        C[so["spin"]].append(v)
    S = mol.intor("int1e_ovlp")
    dms = []
    for s in (0, 1):
        Cs = np.array(C[s]).T
        M = Cs.T @ S @ Cs
        dms.append(Cs @ np.linalg.inv(M) @ Cs.T)
    mf = scf.UHF(mol)
    e_pyscf = mf.energy_tot(dm=np.array(dms))
    print(f"{a}{b} {basis_name} R={R}: PHATPSY={st.total_energy:.6f}  E_det(prolate)={mine:.6f}  "
          f"E_det(PySCF, GTO-fit)={e_pyscf:.6f}  diff={mine - e_pyscf:.2e}  N={np.trace((dms[0]+dms[1])@S):.5f}")


if __name__ == "__main__":
    main("H", "H", 1.4, "DZP" if len(sys.argv) < 2 else sys.argv[1])
    main("N", "N", 2.0744, "DZ")
    main("N", "N", 4.0, "DZ")
