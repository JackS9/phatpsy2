#!/usr/bin/env python3
"""Locality study for exact-integral localized VB (branch two-center-wip).

For H2O (DZP) and CO / N2 (DZP), compares orbital-domain choices against delocalized
CASSCF in the identical exact STO basis:
  * strict one-center   every orbital uses only its own atom's STOs
  * option 2            strict one-center, lone pairs moved into the active space (H2O)
  * option 1            bond-local lone pairs (own atom + bonded neighbours)
  * option 1 + H        additionally bond-local H orbitals (H2O)
and reports the gauge-invariant own-atom character of each optimized subspace
(phatpsy.localized_vb.own_atom_character).

Requires /tmp/h2o_lvb_x.npy from tools/localized_vb_polyatomic.py (H2O DZP start).
Usage: PYTHONPATH=. .venv/bin/python tools/locality_analysis.py
"""
import contextlib
import io
import pickle
import sys

import numpy as np

sys.path.insert(0, "tools")
import localized_vb_polyatomic as P  # noqa: E402
from localized_vb_scan import build_model, run_point  # noqa: E402
from phatpsy.localized_vb import MaskedVB, orthonormal_mo_guess, own_atom_character, pyscf_mean_field  # noqa: E402
from phatpsy.multicenter_integrals import molecular_ao_integrals  # noqa: E402

HEV = 27.211386245988


def subspace_report(ints, model, x, owners):
    Xi, Xa = model.spaces(model.orbitals(x))
    parts = []
    for space, W in (("inactive", Xi), ("active", Xa)):
        for a in sorted(set(owners)):
            k = sum(1 for r, o in zip(model.roles, owners) if r == space and o == a)
            if k and W.shape[1]:
                parts.append(f"{space}/atom{a} " + " ".join(f"{c:.3f}" for c in own_atom_character(ints, W, a, k)))
    return " | ".join(parts)


def h2o():
    try:
        ints = pickle.load(open("/tmp/h2o_ints_DZP.pkl", "rb"))
    except Exception:
        ints = molecular_ao_integrals(P.water("DZP"))
        pickle.dump(ints, open("/tmp/h2o_ints_DZP.pkl", "wb"))
    specs = [P.Spec(0, "inactive")] * 3 + [P.Spec(0, "active")] * 2 + [P.Spec(1, "active"), P.Spec(2, "active")]
    orbs = P.PolyVB(ints, specs, (2, 2)).orbitals(np.load("/tmp/h2o_lvb_x.npy"))
    roles = [s.role for s in specs]
    owners = [s.atom for s in specs]
    one = [np.where(ints.atom_of == a)[0] for a in owners]
    allao = np.arange(ints.nao)
    with contextlib.redirect_stdout(io.StringIO()):
        rhf = pyscf_mean_field(ints, (5, 5), "RHF")
    _, mfO = P.atomic_orbitals(ints, 0, 8, 2)
    _, mfH = P.atomic_orbitals(ints, 1, 1, 1)
    e_at = mfO.e_tot + 2 * mfH.e_tot
    rows = {}
    for name, free, rl, nel in (("strict one-center", [], roles, (2, 2)),
                                ("option 2: lone pairs active", [], ["inactive"] + ["active"] * 6, (4, 4)),
                                ("option 1: bond-local lone pairs", [1, 2], roles, (2, 2)),
                                ("option 1 + bond-local H", [1, 2, 5, 6], roles, (2, 2))):
        m = MaskedVB(ints, rl, [allao if k in free else one[k] for k in range(7)], nel)
        r = m.optimize(m.pack(orbs))
        rows[name] = (r.fun, nel, m, r.x)
    from pyscf import mcscf
    refs = {}
    for nel, nact in (((2, 2), 4), ((4, 4), 6)):
        m = rows["strict one-center" if nel == (2, 2) else "option 2: lone pairs active"][2]
        x = rows["strict one-center" if nel == (2, 2) else "option 2: lone pairs active"][3]
        Xi, Xa = m.spaces(m.orbitals(x))
        with contextlib.redirect_stdout(io.StringIO()):
            mc = mcscf.CASSCF(rhf, nact, sum(nel))
            mc.conv_tol = 1e-9
            mc.kernel(mo_coeff=orthonormal_mo_guess(Xi, Xa, ints.S))
        refs[nel] = mc.e_tot
    print("H2O DZP (R_OH 1.8088, 104.52 deg)")
    label = {(2, 2): "CASSCF(4,4)", (4, 4): "CASSCF(8,6)"}
    for name, (e, nel, m, x) in rows.items():
        print(f"  {name:34s} E={e:.6f} D_e={(e_at - e) * HEV:6.3f} eV  cost vs {label[nel]}"
              f"={(e - refs[nel]) * HEV:6.3f} eV")
        print(f"      own-atom character: {subspace_report(ints, m, x, owners)}")
    print(f"  CASSCF(4,4) E={refs[(2, 2)]:.6f} D_e={(e_at - refs[(2, 2)]) * HEV:6.3f}   "
          f"CASSCF(8,6) E={refs[(4, 4)]:.6f}   RHF D_e={(e_at - rhf.e_tot) * HEV:6.3f}   expt D_e=10.08 eV")


def diatomics():
    for a, b, R in (("C", "O", 2.132), ("N", "N", 2.0744)):
        with contextlib.redirect_stdout(io.StringIO()):
            out = run_point(a, b, "DZP", R, None, ("cas",))
        ints = out["ints"]
        dm = build_model(ints, a, b, a == b)
        orbs = dm.orbitals(out["x"]["cas"])
        roles = [s.role if s.role == "inactive" else "active" for s in dm.specs]
        owners = [s.atom for s in dm.specs]
        mvals = np.array([f.m for f in ints.basis])
        lone = [k for k, s in enumerate(dm.specs) if s.role == "inactive" and s.pair == -2]
        print(f"{a}{b} DZP R={R}  CASSCF(6,6)={out['E']['casscf']:.6f}")
        for name, bl in (("strict one-center", False), ("option 1: bond-local lone pairs", True)):
            masks = [np.asarray(mk) for mk in dm.masks]
            if bl:
                for k in lone:
                    masks[k] = np.where(mvals == 0)[0]
            m = MaskedVB(ints, roles, masks, (3, 3))
            r = m.optimize(m.pack(orbs))
            print(f"  {name:34s} E={r.fun:.6f} cost={(r.fun - out['E']['casscf']) * HEV:6.3f} eV")


if __name__ == "__main__":
    h2o()
    diatomics()

