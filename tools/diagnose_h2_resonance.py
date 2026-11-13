#!/usr/bin/env python3
"""
Stage 0 diagnostic for the two-center resonance plan (branch two-center-wip).

Section A -- Gate G0: reproduce classic H2 valence-bond results with the prolate
            spheroidal STO integral engine and no adjustable parameters:
              Heitler-London / Sugiura (zeta = 1), Wang (zeta optimized),
              Weinbaum / Coulson-Fischer (covalent + ionic, zeta optimized).
Section B -- Decompose PHATPSY's own H2 curve: for PHATPSY-converged localized
            orbitals at each R, compare
              E_PHATPSY (single-center functional, model potential)
              E_HP      (Hartree product, exact two-center Coulomb)
              E_HL      (Heitler-London spin coupling of the same orbitals)
              E_CAS     (Weinbaum/Coulson-Fischer CI in span{a, b})

Usage:
    PYTHONPATH=. .venv/bin/python tools/diagnose_h2_resonance.py [--section A|B|all]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math

import numpy as np
from scipy.optimize import minimize_scalar

from phatpsy.two_center_integrals import STO, ProlateGrid
from phatpsy.vb_correction import localized_occupied_orbitals, pair_energies_from_orbitals

HARTREE_EV = 27.211386245988
HARTREE_CM = 219474.6313632
MU_H2 = 1836.15267343 * 1.00782503223 / 2.0  # reduced mass of H2 in m_e
E_EXACT_H2 = -1.174475714  # Kolos-Wolniewicz / Pachucki, R = 1.4011 Bohr
R_EXACT = 1.4011


def bare_pair(R: float, zeta: float):
    A, B = (0.0, 0.0, 0.0), (0.0, 0.0, R)
    g = ProlateGrid(A, B, zeta_min=zeta)
    return pair_energies_from_orbitals(
        g, [STO(A, 1, 0, 0, zeta)], np.array([1.0]), [STO(B, 1, 0, 0, zeta)], np.array([1.0]), 1.0, 1.0, lmax=20
    )


def curve_min(f, lo=1.0, hi=2.4):
    r = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-5})
    h = 0.01
    k = (f(r.x + h) - 2 * f(r.x) + f(r.x - h)) / (h * h)
    we = math.sqrt(max(k, 0.0) / MU_H2) * HARTREE_CM
    return r.x, r.fun, we


def section_a():
    print("=" * 96)
    print("Section A (Gate G0): classic H2 valence-bond results, no adjustable parameters")
    print("=" * 96)
    rows = []
    # Heitler-London, zeta = 1
    Re, E, we = curve_min(lambda R: bare_pair(R, 1.0).E_HL)
    rows.append(("Heitler-London (Sugiura), zeta=1", 1.0, Re, E, we, "R_e 1.64, D_e 3.14-3.16 eV"))
    # Wang: zeta optimized at each R
    def wang(R):
        return minimize_scalar(lambda z: bare_pair(R, z).E_HL, bounds=(0.9, 1.5), method="bounded",
                               options={"xatol": 1e-6}).fun
    Re, E, we = curve_min(wang, 1.2, 1.7)
    zw = minimize_scalar(lambda z: bare_pair(Re, z).E_HL, bounds=(0.9, 1.5), method="bounded").x
    rows.append(("Wang (HL, zeta optimized)", zw, Re, E, we, "R_e 1.41, zeta 1.166, D_e 3.76-3.78 eV"))
    # Weinbaum / Coulson-Fischer: CAS(2,2), zeta optimized
    def weinbaum(R):
        return minimize_scalar(lambda z: bare_pair(R, z).E_CAS, bounds=(0.9, 1.5), method="bounded",
                               options={"xatol": 1e-6}).fun
    Re, E, we = curve_min(weinbaum, 1.2, 1.7)
    zb = minimize_scalar(lambda z: bare_pair(Re, z).E_CAS, bounds=(0.9, 1.5), method="bounded").x
    rows.append(("Weinbaum / Coulson-Fischer", zb, Re, E, we, "R_e 1.42, zeta 1.19, D_e 4.00-4.03 eV"))
    # MO (Coulson) for completeness: CAS restricted to sigma_g^2 is RHF in this space
    print(f"{'method':38s} {'zeta':>6s} {'R_e':>7s} {'E_min':>11s} {'D_e(eV)':>8s} {'w_e(cm-1)':>10s}   literature")
    for name, z, Re, E, we, lit in rows:
        print(f"{name:38s} {z:6.3f} {Re:7.4f} {E:11.6f} {(-1.0 - E) * HARTREE_EV:8.3f} {we:10.1f}   {lit}")
    print(f"{'Exact (Kolos-Wolniewicz/Pachucki)':38s} {'':6s} {R_EXACT:7.4f} {E_EXACT_H2:11.6f} "
          f"{(-1.0 - E_EXACT_H2) * HARTREE_EV:8.3f} {4401.2:10.1f}")
    return rows


def phatpsy_pair(R: float, basis: str):
    from phatpsy.builder import build_diatomic
    from phatpsy.driver import PhatpsyDriver

    deck = build_diatomic("H", "H", bond_length=R, basis=basis)
    deck.control.convergence = 1e-9
    deck.control.max_cycles = 300
    with contextlib.redirect_stdout(io.StringIO()):
        st = PhatpsyDriver(quiet=True).run_deck(deck)
    orbs = localized_occupied_orbitals(st)
    (ba, ca, _), (bb, cb, _) = orbs[1], orbs[2]
    X = np.asarray(st["atom_coordinates"])
    zmin = min(f.zeta for f in ba + bb)
    g = ProlateGrid(X[:, 0], X[:, 1], zeta_min=zmin)
    pe = pair_energies_from_orbitals(g, ba, ca, bb, cb, 1.0, 1.0, lmax=24)
    return float(st.total_energy), pe, bool(st.converged)


def section_b(bases=("DZ", "DZP", "ADF-TZP"), Rs=None):
    print()
    print("=" * 96)
    print("Section B: PHATPSY H2 -- single-center functional vs exact pair energies of the same orbitals")
    print("=" * 96)
    if Rs is None:
        Rs = np.array([1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2])
    summary = {}
    for basis in bases:
        print(f"\n--- basis {basis} ---")
        print(f"{'R':>5s} {'conv':>4s} {'E_PHATPSY':>11s} {'E_HP(exact)':>11s} {'E_HL':>11s} {'E_CAS':>11s} "
              f"{'S':>7s} {'w_dbl':>6s} {'T_aa':>7s}")
        data = []
        for R in Rs:
            e_ph, pe, conv = phatpsy_pair(float(R), basis)
            data.append((R, e_ph, pe.E_HP, pe.E_HL, pe.E_CAS))
            print(f"{R:5.2f} {('y' if conv else 'n'):>4s} {e_ph:11.6f} {pe.E_HP:11.6f} {pe.E_HL:11.6f} {pe.E_CAS:11.6f} "
                  f"{pe.S:7.4f} {pe.ionic_weight:6.3f} {pe.components['T_aa']:7.4f}")
        data = np.array(data)
        summary[basis] = data
        print(f"{'curve':12s} {'R_e':>7s} {'E_min':>11s} {'D_e(eV)':>8s}")
        for j, name in ((1, "PHATPSY"), (2, "HP(exact)"), (3, "HL"), (4, "CAS")):
            c = np.polyfit(data[:, 0], data[:, j], 6)
            rr = np.linspace(data[0, 0], data[-1, 0], 4001)
            ee = np.polyval(c, rr)
            k = int(np.argmin(ee))
            flag = "" if 0 < k < len(rr) - 1 else "  (edge: no interior minimum)"
            print(f"{name:12s} {rr[k]:7.4f} {ee[k]:11.6f} {(-1.0 - ee[k]) * HARTREE_EV:8.3f}{flag}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all", choices=["A", "B", "all"])
    args = ap.parse_args()
    if args.section in ("A", "all"):
        section_a()
    if args.section in ("B", "all"):
        section_b()
