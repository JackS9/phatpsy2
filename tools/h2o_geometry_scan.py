#!/usr/bin/env python3
"""Stage 10 (Step 1): First-Principles H2O Equilibrium Geometry & Vibrational Scan.

Computes the H2O bond-angle (theta_HOH) and symmetric-stretch (R_OH) potential energy
surfaces in exact multi-center STO bases (DZ and DZP) with ZERO empirical parameters
(no w_HH = 0.40 Phillips-Kleinman attenuation weight), comparing:

  1. Strict one-center Localized VB (`strict_1c`):
     Every orbital (O: 1s, lp_sigma, lp_pi, b1, b2; H1: h1; H2: h2) is expanded
     strictly in its own atom's STOs (own-atom character = 1.000), with CAS(4,4)
     valence spin-coupling over the two O-H bonds (b1, b2, h1, h2).
  2. Frozen-Core Embedded Localized VB (`embedded_1s_e1` and `embedded_1s_e2`):
     The O 1s^2 core is frozen at the isolated O(^3P) atomic ROHF 1s orbital and
     embedded via exact Phillips-Kleinman back-projection (Level E1) and
     charge-conserving unprojected-ERI embedding (Level E2), leaving an 8-electron
     strictly one-center valence VB problem.
  3. Bond-Local Lone Pairs (`option1_bond_local_lp`):
     The two O lone pairs are allowed onto the bonded H atoms while the 4 active
     bonding orbitals remain strictly one-center.
  4. Bond-Local Lone Pairs + H (`option1_plus_h`):
     Both the O lone pairs and H active orbitals are bond-local while O 1s, b1, b2
     remain strictly one-center.
  5. Delocalized MO References (`RHF` and `CASSCF(4,4)`):
     Evaluated in the identical exact multi-center STO basis.

Usage:
  PYTHONPATH=. .venv/bin/python tools/h2o_geometry_scan.py [DZ|DZP|ALL]
"""
from __future__ import annotations

import contextlib
import io
import os
import pickle
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, "tools")
import localized_vb_polyatomic as P  # noqa: E402
from phatpsy.embedding import embed_from_full_integrals  # noqa: E402
from phatpsy.localized_vb import (  # noqa: E402
    MaskedVB,
    orthonormal_mo_guess,
    own_atom_character,
    pyscf_mean_field,
)
from phatpsy.multicenter_integrals import MolecularAOIntegrals, molecular_ao_integrals  # noqa: E402

HEV = 27.211386245988
HARTREE_TO_CM = 219474.63136320
AMU_TO_ME = 1822.888486209
M_H = 1.00782503223 * AMU_TO_ME
M_O = 15.99491461957 * AMU_TO_ME

# Experimental reference values (Herzberg / NIST CCCBDB)
EXPT_R_OH = 1.8088       # Bohr (0.9572 A)
EXPT_THETA = 104.52      # deg
EXPT_OMEGA1 = 3832.0     # cm^-1 (harmonic symmetric stretch; nu_1 = 3657 cm^-1)
EXPT_OMEGA2 = 1648.0     # cm^-1 (harmonic bend; nu_2 = 1595 cm^-1)
EXPT_DE = 10.08          # eV (atomization energy D_e; D_0 = 9.51 eV)


def get_water_integrals(basis_name: str, R: float, theta: float, cache_dir: str = "/tmp") -> MolecularAOIntegrals:
    """Load or compute exact multi-center STO integrals for H2O(basis_name, R, theta)."""
    fname = os.path.join(cache_dir, f"h2o_ints_{basis_name}_R{R:.4f}_th{theta:.2f}.pkl")
    if os.path.exists(fname):
        try:
            with open(fname, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    # Reuse legacy equilibrium cache if matching
    if basis_name == "DZP" and abs(R - 1.8088) < 1e-4 and abs(theta - 104.52) < 1e-2:
        legacy_pkl = "/tmp/h2o_ints_DZP.pkl"
        if os.path.exists(legacy_pkl):
            try:
                with open(legacy_pkl, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
    t0 = time.time()
    atoms = P.water(basis_name, R=R, theta=theta)
    ints = molecular_ao_integrals(atoms)
    dt = time.time() - t0
    print(f"    [integrals] {basis_name} R={R:.4f} theta={theta:.2f} deg -> {dt:.1f}s (nao={ints.nao})", flush=True)
    try:
        with open(fname, "wb") as f:
            pickle.dump(ints, f)
    except Exception:
        pass
    return ints


def initial_h2o_orbitals(ints: MolecularAOIntegrals, R: float, theta: float):
    """Build intuitive sp3-like initial localized orbitals on O, H1, H2."""
    io_, mfO = P.atomic_orbitals(ints, 0, 8, 2)
    CO = np.zeros((ints.nao, mfO.mo_coeff.shape[1]))
    CO[io_] = mfO.mo_coeff
    c1s, c2s = CO[:, 0], CO[:, 1]
    p_mo = CO[:, 2]
    pshell = {}
    for k in io_:
        f = ints.basis[k]
        if f.l == 1:
            pshell[(f.n, f.zeta)] = pshell.get((f.n, f.zeta), 0.0) + abs(p_mo[k])
    norm = np.sqrt(sum(v * v for v in pshell.values()))
    pshell = {k: v / norm for k, v in pshell.items()}

    O_xyz, H1_xyz, H2_xyz = ints.coords[0], ints.coords[1], ints.coords[2]
    d1 = (H1_xyz - O_xyz) / np.linalg.norm(H1_xyz - O_xyz)
    d2 = (H2_xyz - O_xyz) / np.linalg.norm(H2_xyz - O_xyz)
    bis = -(d1 + d2) / np.linalg.norm(d1 + d2)
    nrm = np.cross(d1, d2)
    nrm /= np.linalg.norm(nrm)

    def hyb(d, s):
        return s * c2s + np.sqrt(1.0 - s * s) * P.p_vector(ints, 0, pshell, d)

    b1 = hyb(d1, 0.5)
    b2 = hyb(d2, 0.5)
    lp_s = hyb(bis, 0.6)
    lp_p = P.p_vector(ints, 0, pshell, nrm)

    hs = []
    for ia in (1, 2):
        ih, mfH = P.atomic_orbitals(ints, ia, 1, 1)
        v = np.zeros(ints.nao)
        v[ih] = mfH.mo_coeff[:, 0]
        hs.append(v)

    orbs = [c1s, lp_s, lp_p, b1, b2, hs[0], hs[1]]
    return orbs, mfO, mfH


def evaluate_h2o_point(
    ints: MolecularAOIntegrals,
    R: float,
    theta: float,
    prev_orbs: Dict[str, List[np.ndarray]] | None = None,
) -> Tuple[Dict[str, float], Dict[str, List[np.ndarray]], float, Dict[str, float]]:
    """Evaluate all H2O VB, embedded-VB, and MO methods at a single geometry (R, theta)."""
    orbs0, mfO, mfH = initial_h2o_orbitals(ints, R, theta)
    e_atoms = float(mfO.e_tot + 2.0 * mfH.e_tot)

    roles_all = ["inactive", "inactive", "inactive", "active", "active", "active", "active"]
    owners_all = [0, 0, 0, 0, 0, 1, 2]
    one_masks = [np.where(ints.atom_of == a)[0] for a in owners_all]
    all_ao = np.arange(ints.nao)

    energies: Dict[str, float] = {}
    opt_orbs: Dict[str, List[np.ndarray]] = {}
    char_info: Dict[str, float] = {}

    # 1. Strict one-center Localized VB
    m_strict = MaskedVB(ints, roles_all, one_masks, (2, 2))
    seed_strict = prev_orbs["strict_1c"] if (prev_orbs and "strict_1c" in prev_orbs) else orbs0
    r_strict = m_strict.optimize(m_strict.pack(seed_strict), maxiter=600, gtol=1e-6)
    # Also check fresh start if seeded to avoid local minimum drift
    if prev_orbs and "strict_1c" in prev_orbs:
        r_fresh = m_strict.optimize(m_strict.pack(orbs0), maxiter=600, gtol=1e-6)
        if r_fresh.fun < r_strict.fun:
            r_strict = r_fresh
    energies["strict_1c"] = float(r_strict.fun)
    opt_orbs["strict_1c"] = m_strict.orbitals(r_strict.x)

    # 2. Option 1: Bond-local lone pairs (lone pairs 1, 2 allowed on O + H1 + H2)
    masks_opt1 = [all_ao if k in (1, 2) else one_masks[k] for k in range(7)]
    m_opt1 = MaskedVB(ints, roles_all, masks_opt1, (2, 2))
    r_opt1 = m_opt1.optimize(m_opt1.pack(opt_orbs["strict_1c"]), maxiter=600, gtol=1e-6)
    energies["option1_bond_local_lp"] = float(r_opt1.fun)
    opt_orbs["option1_bond_local_lp"] = m_opt1.orbitals(r_opt1.x)
    Xi_1, _ = m_opt1.spaces(opt_orbs["option1_bond_local_lp"])
    char_info["lp_own_atom_min"] = float(np.min(own_atom_character(ints, Xi_1, 0, 3)))

    # 3. Option 1 + H: Bond-local lone pairs and H orbitals
    masks_opt1h = [all_ao if k in (1, 2, 5, 6) else one_masks[k] for k in range(7)]
    m_opt1h = MaskedVB(ints, roles_all, masks_opt1h, (2, 2))
    r_opt1h = m_opt1h.optimize(m_opt1h.pack(opt_orbs["option1_bond_local_lp"]), maxiter=600, gtol=1e-6)
    energies["option1_plus_h"] = float(r_opt1h.fun)
    opt_orbs["option1_plus_h"] = m_opt1h.orbitals(r_opt1h.x)

    # 4. Frozen-Core Embedded Localized VB (O 1s^2 frozen at atomic ROHF 1s)
    C_core = orbs0[0][:, None]  # (nao, 1) O 1s orbital
    roles_val = ["inactive", "inactive", "active", "active", "active", "active"]
    masks_val = one_masks[1:]
    val_seed = opt_orbs["strict_1c"][1:]
    for mode_key, mode_name in (("embedded_1s_e1", "exact_backprojected"),
                                ("embedded_1s_e2", "unprojected_eri")):
        emb_ints = embed_from_full_integrals(ints, C_core, all_ao, mode=mode_name)
        m_emb = MaskedVB(emb_ints, roles_val, masks_val, (2, 2))
        r_emb = m_emb.optimize(m_emb.pack(val_seed), maxiter=600, gtol=1e-6)
        energies[mode_key] = float(r_emb.fun)
        opt_orbs[mode_key] = m_emb.orbitals(r_emb.x)

    # 5. Delocalized RHF and CASSCF(4,4)
    from pyscf import mcscf
    with contextlib.redirect_stdout(io.StringIO()):
        rhf = pyscf_mean_field(ints, (5, 5), "RHF")
        energies["rhf"] = float(rhf.e_tot)
        Xi_s, Xa_s = m_strict.spaces(opt_orbs["strict_1c"])
        mc = mcscf.CASSCF(rhf, 4, 4)
        mc.conv_tol = 1e-9
        mc.kernel(mo_coeff=orthonormal_mo_guess(Xi_s, Xa_s, ints.S))
        energies["casscf_44"] = float(mc.e_tot)

    return energies, opt_orbs, e_atoms, char_info


def fit_minimum_quadratic(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float, float]:
    """Fit a degree-3 polynomial (or quadratic around minimum) to (xs, ys) and return (x_min, y_min, d2y/dx2)."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    imin = int(np.argmin(y))
    # Use up to 5 points centered on imin for a local cubic/quadratic fit
    lo = max(0, min(imin - 2, len(x) - 5))
    hi = min(len(x), lo + 5)
    x_sub, y_sub = x[lo:hi], y[lo:hi]
    deg = 3 if len(x_sub) >= 4 else 2
    x0 = float(x[imin])
    dx = x_sub - x0
    coefs = np.polyfit(dx, y_sub, deg=deg)
    dcoefs = np.polyder(coefs, 1)
    ddcoefs = np.polyder(coefs, 2)
    roots = np.roots(dcoefs)
    real_roots = [float(r.real) for r in roots if abs(r.imag) < 1e-7]
    span = float(x_sub[-1] - x_sub[0])
    cand = [r for r in real_roots if abs(r) <= 0.6 * span and np.polyval(ddcoefs, r) > 0]
    if cand:
        best_dx = min(cand, key=lambda r: abs(r))
    else:
        # Fallback to local 3-point quadratic around imin
        lo2 = max(0, min(imin - 1, len(x) - 3))
        dx2 = x[lo2:lo2 + 3] - x0
        c2 = np.polyfit(dx2, y[lo2:lo2 + 3], deg=2)
        best_dx = float(-c2[1] / (2.0 * c2[0])) if c2[0] > 0 else 0.0
        coefs = c2
        ddcoefs = np.polyder(c2, 2)
    x_min = x0 + best_dx
    y_min = float(np.polyval(coefs, best_dx))
    curv = float(np.polyval(ddcoefs, best_dx))
    return x_min, y_min, curv


def bend_frequency_cm1(k_theta_rad2: float, R_bohr: float, theta_deg: float) -> float:
    """Harmonic H-O-H bending frequency (cm^-1) from d^2E/dtheta^2 (Ha/rad^2)."""
    if k_theta_rad2 <= 0.0:
        return 0.0
    th = np.radians(theta_deg)
    G_tt = (2.0 / (R_bohr * R_bohr)) * (1.0 / M_H + (1.0 - np.cos(th)) / M_O)
    return float(np.sqrt(k_theta_rad2 * G_tt) * HARTREE_TO_CM)


def sym_stretch_frequency_cm1(d2E_dR2: float, theta_deg: float) -> float:
    """Harmonic symmetric O-H stretch frequency omega_1 (cm^-1) from d^2E/dR_OH^2 (Ha/Bohr^2).

    Since R_1 = R_2 = R_OH, S_1 = (dR_1 + dR_2)/sqrt(2) = sqrt(2) dR_OH, so
    F_11 = d^2E/dS_1^2 = 0.5 * d^2E/dR_OH^2.
    """
    if d2E_dR2 <= 0.0:
        return 0.0
    th = np.radians(theta_deg)
    F_11 = 0.5 * d2E_dR2
    G_11 = 1.0 / M_H + (1.0 + np.cos(th)) / M_O
    return float(np.sqrt(F_11 * G_11) * HARTREE_TO_CM)


METHODS = [
    ("strict_1c", "Strict 1c VB"),
    ("embedded_1s_e1", "Embedded 1s (Exact PK, E1)"),
    ("embedded_1s_e2", "Embedded 1s (Unproj ERI, E2)"),
    ("option1_bond_local_lp", "Option 1 (Bond-Local LP)"),
    ("option1_plus_h", "Option 1+H (Bond-Local LP+H)"),
    ("casscf_44", "Delocalized CASSCF(4,4)"),
    ("rhf", "Delocalized RHF"),
]


def run_basis_scan(basis_name: str, thetas: Sequence[float], Rs: Sequence[float]):
    print("=" * 96)
    print(f"H2O FIRST-PRINCIPLES GEOMETRY SCAN — Basis: {basis_name} (w_HH empirical tuning = NONE)")
    print("=" * 96)

    # --- 1. Bond-angle scan at R_OH = EXPT_R_OH ---
    print(f"\n[1] Bond-Angle Scan theta_HOH at R_OH = {EXPT_R_OH:.4f} Bohr:")
    angle_data: Dict[str, List[float]] = {k: [] for k, _ in METHODS}
    prev_orbs = None
    e_atoms_ref = 0.0
    for th in thetas:
        ints = get_water_integrals(basis_name, EXPT_R_OH, th)
        ens, prev_orbs, e_atoms_ref, _ = evaluate_h2o_point(ints, EXPT_R_OH, th, prev_orbs)
        for k, _ in METHODS:
            angle_data[k].append(ens[k])
        print(
            f"  theta={th:6.2f} deg | Strict1c={ens['strict_1c']:.6f}  EmbE2={ens['embedded_1s_e2']:.6f}  "
            f"Opt1(LP)={ens['option1_bond_local_lp']:.6f}  CASSCF={ens['casscf_44']:.6f}  RHF={ens['rhf']:.6f}",
            flush=True,
        )

    # --- 2. Symmetric O-H bond-length scan at theta_HOH = EXPT_THETA ---
    print(f"\n[2] Symmetric Stretch Scan R_OH at theta_HOH = {EXPT_THETA:.2f} deg:")
    stretch_data: Dict[str, List[float]] = {k: [] for k, _ in METHODS}
    prev_orbs = None
    for R in Rs:
        ints = get_water_integrals(basis_name, R, EXPT_THETA)
        ens, prev_orbs, e_atoms_ref, _ = evaluate_h2o_point(ints, R, EXPT_THETA, prev_orbs)
        for k, _ in METHODS:
            stretch_data[k].append(ens[k])
        print(
            f"  R_OH ={R:6.4f} Bohr | Strict1c={ens['strict_1c']:.6f}  EmbE2={ens['embedded_1s_e2']:.6f}  "
            f"Opt1(LP)={ens['option1_bond_local_lp']:.6f}  CASSCF={ens['casscf_44']:.6f}  RHF={ens['rhf']:.6f}",
            flush=True,
        )

    # --- 3. Summary table of equilibrium geometry, frequencies, and D_e ---
    print(f"\n[3] Summary of First-Principles H2O Observables ({basis_name}):")
    hdr = (
        f"  {'Method':30s} | {'theta_e (deg)':>13s} | {'R_e (Bohr)':>10s} | "
        f"{'omega_2 (cm-1)':>14s} | {'omega_1 (cm-1)':>14s} | {'D_e (eV)':>9s} | {'E_min (Ha)':>12s}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    summary: Dict[str, Dict[str, float]] = {}
    thetas_rad = np.radians(np.asarray(thetas, dtype=float))
    for key, label in METHODS:
        th_min_rad, e_th_min, k_tt = fit_minimum_quadratic(thetas_rad, angle_data[key])
        th_min_deg = float(np.degrees(th_min_rad))
        R_min, e_R_min, d2E_dR2 = fit_minimum_quadratic(Rs, stretch_data[key])
        omega2 = bend_frequency_cm1(k_tt, R_min, th_min_deg)
        omega1 = sym_stretch_frequency_cm1(d2E_dR2, th_min_deg)
        De_eV = (e_atoms_ref - e_R_min) * HEV
        summary[key] = {
            "theta_e": th_min_deg,
            "R_e": R_min,
            "k_theta": k_tt,
            "k_R": d2E_dR2,
            "omega_2": omega2,
            "omega_1": omega1,
            "D_e": De_eV,
            "E_min": e_R_min,
        }
        print(
            f"  {label:30s} | {th_min_deg:13.2f} | {R_min:10.4f} | "
            f"{omega2:14.1f} | {omega1:14.1f} | {De_eV:9.3f} | {e_R_min:12.6f}"
        )
    print(
        f"  {'Experiment (Herzberg / NIST)':30s} | {EXPT_THETA:13.2f} | {EXPT_R_OH:10.4f} | "
        f"{EXPT_OMEGA2:14.1f} | {EXPT_OMEGA1:14.1f} | {EXPT_DE:9.2f} | {'---':>12s}"
    )
    return {
        "basis": basis_name,
        "thetas": list(thetas),
        "Rs": list(Rs),
        "angle_data": angle_data,
        "stretch_data": stretch_data,
        "summary": summary,
        "e_atoms": e_atoms_ref,
    }


def main():
    mode = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"
    thetas_dz = [90.0, 95.0, 100.0, 104.52, 110.0, 115.0, 120.0]
    Rs_dz = [1.65, 1.73, 1.8088, 1.89, 1.98, 2.08, 2.18]

    thetas_dzp = [94.0, 99.0, 104.52, 110.0, 116.0]
    Rs_dzp = [1.68, 1.74, 1.8088, 1.88, 1.96]

    thetas_tzp = [94.0, 99.0, 104.52, 110.0, 116.0]
    Rs_tzp = [1.68, 1.74, 1.8088, 1.88, 1.96]

    results = {}
    if mode in ("DZ", "ALL"):
        results["DZ"] = run_basis_scan("DZ", thetas_dz, Rs_dz)
    if mode in ("DZP", "ALL"):
        results["DZP"] = run_basis_scan("DZP", thetas_dzp, Rs_dzp)
    if mode in ("ADF-TZP", "TZP", "ALL"):
        results["ADF-TZP"] = run_basis_scan("ADF-TZP", thetas_tzp, Rs_tzp)

    with open("/tmp/h2o_geometry_scan_results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()

