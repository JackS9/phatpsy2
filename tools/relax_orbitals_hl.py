#!/usr/bin/env python3
"""Stage 2(b): optimize strictly one-center sigma orbitals a (on A) and mirror b
(on B) under E_HL and E_CAS in the PHATPSY DZP and ADF-TZP hydrogen bases.

Usage:
    PYTHONPATH=. .venv/bin/python tools/relax_orbitals_hl.py [DZP] [ADF-TZP]
"""
from __future__ import annotations

import sys
import time
from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize

from phatpsy.basis import get_element_basis
from phatpsy.two_center_integrals import STO, ProlateGrid
from phatpsy.vb_correction import PairEnergies, pair_energies_from_orbitals

HEV = 27.211386245988


def sigma_basis(
    basis_name: str, center: Tuple[float, float, float], toward_sign: float
) -> Tuple[List[STO], np.ndarray]:
    """Return the m=0 (sigma) STOs on ``center`` for Hydrogen in ``basis_name``,
    along with the mirror parity signs (+1 for l=0, ``toward_sign`` for l=1)."""
    ed = get_element_basis("H", basis_name)
    fs: List[STO] = []
    signs: List[float] = []
    for f in ed.functions:
        if f.l == 0:
            fs.append(STO(center, f.n, 0, 0, f.zeta))
            signs.append(1.0)
        elif f.l == 1:
            fs.append(STO(center, f.n, 1, 0, f.zeta))
            signs.append(toward_sign)
    return fs, np.array(signs)


def energy(
    c: np.ndarray, R: float, basis_name: str, which: str
) -> Tuple[float, PairEnergies]:
    """Evaluate ``which`` ('E_HL' or 'E_CAS') at bond length ``R`` for one-center
    sigma orbital coefficients ``c``."""
    A, B = (0.0, 0.0, 0.0), (0.0, 0.0, float(R))
    fa, sa = sigma_basis(basis_name, A, +1.0)
    fb, sb = sigma_basis(basis_name, B, -1.0)
    zmin = min(f.zeta for f in fa)
    g = ProlateGrid(A, B, zeta_min=zmin, n_eta=40)
    pe = pair_energies_from_orbitals(g, fa, c * sa, fb, c * sb, 1.0, 1.0, lmax=20)
    return float(getattr(pe, which)), pe


def main() -> None:
    bases = sys.argv[1:] or ["DZP", "ADF-TZP"]
    for basis_name in bases:
        fa, _ = sigma_basis(basis_name, (0.0, 0.0, 0.0), 1.0)
        print(
            f"\n=== {basis_name}: sigma functions "
            f"{[(f.n, f.l, f.zeta) for f in fa]} ===",
            flush=True,
        )
        for which in ("E_HL", "E_CAS"):
            rows = []
            c0 = np.zeros(len(fa))
            c0[[i for i, f in enumerate(fa) if f.l == 0][0]] = 1.0
            for R in (1.3, 1.4, 1.5, 1.6):
                t = time.time()
                r = minimize(
                    lambda c: energy(c, R, basis_name, which)[0],
                    c0,
                    method="BFGS",
                    options={"gtol": 1e-7, "eps": 1e-6},
                )
                c0 = r.x
                e, pe = energy(r.x, R, basis_name, which)
                rows.append((R, e))
                c_unit = np.round(r.x / np.linalg.norm(r.x), 4)
                print(
                    f"{which:6s} R={R:.2f} E={e:.6f} T_aa={pe.components['T_aa']:.4f} "
                    f"S={pe.S:.4f} c={c_unit} ({time.time() - t:.0f}s)",
                    flush=True,
                )
            rows_arr = np.array(rows)
            p = np.polyfit(rows_arr[:, 0], rows_arr[:, 1], 3)
            rr = np.linspace(1.3, 1.6, 3001)
            ee = np.polyval(p, rr)
            k = int(np.argmin(ee))
            print(
                f"  -> {which} min: R_e={rr[k]:.4f}  E={ee[k]:.6f}  "
                f"D_e={(-1.0 - ee[k]) * HEV:.3f} eV",
                flush=True,
            )


if __name__ == "__main__":
    main()

