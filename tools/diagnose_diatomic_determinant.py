#!/usr/bin/env python3
# Usage: PYTHONPATH=. .venv/bin/python tools/diagnose_diatomic_determinant.py [A B BASIS]
"""Stage 3 measurement: exact energy of PHATPSY's own localized determinant for diatomics,
decomposed into intra-atomic, classical electrostatic and Pauli (exchange+orthogonality)
parts, versus the PHATPSY functional, along the bond coordinate."""
import contextlib, io, sys, time
import numpy as np
from phatpsy.builder import build_molecule
from phatpsy.driver import PhatpsyDriver
from phatpsy.two_center_integrals import ProlateGrid
from phatpsy.vb_correction import occupied_spin_orbitals, interatomic_decomposition

HEV = 27.211386245988
d = PhatpsyDriver(quiet=True)


def run(a, b, R, basis):
    deck = build_molecule([(a, [0.0, 0.0, 0.0]), (b, [0.0, 0.0, R])], basis=basis,
                          spins=["up", "down"], seed_hybrids=True)
    deck.control.convergence = 1e-8
    deck.control.max_cycles = 300
    with contextlib.redirect_stdout(io.StringIO()):
        st = d.run_deck(deck)
    sos = occupied_spin_orbitals(st)
    X = np.asarray(st["atom_coordinates"]); Z = np.asarray(st["atom_charges"])[0]
    g = ProlateGrid(X[:, 0], X[:, 1], zeta_min=0.6, n_eta=64)
    dec = interatomic_decomposition(g, sos, float(Z[0]), float(Z[1]))
    return float(st.total_energy), dec, bool(st.converged)


systems = [(sys.argv[1], sys.argv[2], sys.argv[3])] if len(sys.argv) > 3 else [
    ("N", "N", "DZ"), ("N", "N", "DZP"), ("F", "H", "DZ"), ("Li", "H", "DZ"), ("C", "O", "DZ")]
Rs = {"NN": [1.9, 2.0744, 2.3, 2.6, 3.0, 4.0, 6.0], "FH": [1.5, 1.7328, 2.0, 2.5, 3.5, 6.0],
      "LiH": [2.6, 3.015, 3.5, 4.5, 6.0, 8.0], "CO": [1.9, 2.132, 2.4, 2.8, 3.5, 6.0]}
for a, b, basis in systems:
    key = a + b
    print(f"\n=== {key} {basis} (seeded, spins up/down) ===")
    print(f"{'R':>6s} {'E_PHATPSY':>12s} {'E_det':>12s} {'E_HP':>12s} {'E_elst':>9s} {'E_pauli':>9s} "
          f"{'gap(eV)':>8s} conv")
    rows = []
    for R in Rs[key]:
        try:
            t = time.time()
            e, dec, conv = run(a, b, R, basis)
            rows.append((R, e, dec["E_det"], dec["E_HP"]))
            print(f"{R:6.3f} {e:12.6f} {dec['E_det']:12.6f} {dec['E_HP']:12.6f} {dec['E_elst']:9.4f} "
                  f"{dec['E_pauli']:9.4f} {(e - dec['E_det']) * HEV:8.2f} {conv} ({time.time()-t:.0f}s)")
        except Exception as ex:
            print(f"{R:6.3f} ERROR {type(ex).__name__}: {ex}")
        sys.stdout.flush()
    if len(rows) > 2:
        rows = np.array(rows)
        ref = rows[-1]
        print("binding relative to largest R (eV):  R, PHATPSY, exact-det, Hartree-product")
        for r in rows:
            print(f"   {r[0]:6.3f} {(r[1]-ref[1])*HEV:9.3f} {(r[2]-ref[2])*HEV:9.3f} {(r[3]-ref[3])*HEV:9.3f}")
