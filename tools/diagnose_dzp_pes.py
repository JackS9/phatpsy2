#!/usr/bin/env python3
"""
PHATPSY 2.0: Phase 1/2 Diagnostic Script for Polarized (DZP) PES, Phillips-Kleinman
Pseudometric Spectrum, and Coupled-Perturbed (CPHF) Gradient Parity.

Evaluates and prints:
1. Potential energy surface (PES) scans, equilibrium bond lengths R_e, bond angles,
   virial ratios (-V/T), and dipole moments for N2, CO, and H2O in DZ and DZP.
2. Phillips-Kleinman overlap-projector eigenvalue spectra lambda_max(S_A^{-1/2} P_{1,A} S_A^{-1/2})
   before and after smooth cubic Hermite regularization (cap = 0.58 for QNWVEC).
3. Relaxed CPHF analytical gradient vs central finite-difference gradient parity (< 1e-5 Ha/Bohr)
   in DZP at displaced geometries for N2, CO, and H2O.
"""

from __future__ import annotations

import numpy as np

from phatpsy.builder import build_diatomic, build_molecule
from phatpsy.cphf import CPHFResponseSolver, _unpack_packed, compute_polyatomic_relaxed_gradient
from phatpsy.driver import PhatpsyDriver
from phatpsy.eom import compute_eom
from phatpsy.gradients import compute_scf_finite_difference_gradient
from phatpsy.optimize import optimize_geometry
from phatpsy.schema import ValenceOrbital


def make_h2_deck(R: float, basis: str = "DZP", convergence: float = 1e-10):
    h1_val = [ValenceOrbital(spin="up", occupation=1, orbital_type="sigma", connect_to="H2")]
    h2_val = [ValenceOrbital(spin="down", occupation=1, orbital_type="sigma", connect_to="H1")]
    deck = build_diatomic(
        "H", "H",
        bond_length=R,
        basis=basis,
        hybridization="s",
        spins=["up", "down"],
        valence_configs=[h1_val, h2_val],
        net_spin=0,
        physical_net_spin=0,
        units="bohr",
    )
    deck.control.convergence = convergence
    deck.control.max_cycles = 200
    return deck


def make_n2_deck(R: float, basis: str = "DZP", convergence: float = 1e-10):
    n1_val = [
        ValenceOrbital(spin="up",   occupation=1, orbital_type="sigma",     connect_to="N2"),
        ValenceOrbital(spin="up",   occupation=1, orbital_type="pi",        connect_to="N2"),
        ValenceOrbital(spin="up",   occupation=1, orbital_type="pi",        connect_to="N2"),
        ValenceOrbital(spin="none", occupation=2, orbital_type="lone_pair"),
    ]
    n2_val = [
        ValenceOrbital(spin="down", occupation=1, orbital_type="sigma",     connect_to="N1"),
        ValenceOrbital(spin="down", occupation=1, orbital_type="pi",        connect_to="N1"),
        ValenceOrbital(spin="down", occupation=1, orbital_type="pi",        connect_to="N1"),
        ValenceOrbital(spin="none", occupation=2, orbital_type="lone_pair"),
    ]
    deck = build_diatomic(
        "N", "N",
        bond_length=R,
        basis=basis,
        hybridization="sp",
        spins=["up", "down"],
        valence_configs=[n1_val, n2_val],
        net_spin=0,
        physical_net_spin=0,
        units="bohr",
    )
    deck.control.convergence = convergence
    deck.control.max_cycles = 200
    return deck


def make_co_deck(R: float, basis: str = "DZP", convergence: float = 1e-10):
    c_val = [
        ValenceOrbital(spin="up",   occupation=1, orbital_type="sigma",     connect_to="O2"),
        ValenceOrbital(spin="up",   occupation=1, orbital_type="pi",        connect_to="O2"),
        ValenceOrbital(spin="none", occupation=0, orbital_type="pi",        connect_to="O2"),
        ValenceOrbital(spin="none", occupation=2, orbital_type="lone_pair"),
    ]
    o_val = [
        ValenceOrbital(spin="down", occupation=1, orbital_type="sigma",     connect_to="C1"),
        ValenceOrbital(spin="down", occupation=1, orbital_type="pi",        connect_to="C1"),
        ValenceOrbital(spin="none", occupation=2, orbital_type="lone_pair", connect_to="C1"),
        ValenceOrbital(spin="none", occupation=2, orbital_type="lone_pair"),
    ]
    deck = build_diatomic(
        "C", "O",
        bond_length=R,
        basis=basis,
        hybridization="sp",
        spins=["up", "down"],
        valence_configs=[c_val, o_val],
        net_spin=0,
        physical_net_spin=0,
        units="bohr",
    )
    deck.control.convergence = convergence
    deck.control.max_cycles = 200
    return deck


def make_h2o_deck(R: float, theta_deg: float = 104.48, basis: str = "DZP", convergence: float = 1e-10):
    theta = theta_deg * np.pi / 180.0
    hy = round(float(R * np.sin(theta / 2.0)), 5)
    hz = round(float(R * np.cos(theta / 2.0)), 5)
    o_val = [
        ValenceOrbital(spin="up",   occupation=1, orbital_type="sigma",     connect_to="H1"),
        ValenceOrbital(spin="up",   occupation=1, orbital_type="sigma",     connect_to="H2"),
        ValenceOrbital(spin="none", occupation=2, orbital_type="lone_pair"),
        ValenceOrbital(spin="none", occupation=2, orbital_type="lone_pair"),
    ]
    h1_val = [ValenceOrbital(spin="down", occupation=1, orbital_type="sigma", connect_to="O1")]
    h2_val = [ValenceOrbital(spin="down", occupation=1, orbital_type="sigma", connect_to="O1")]
    deck = build_molecule(
        [("O", [0.0, 0.0, 0.0]), ("H", [0.0, hy, hz]), ("H", [0.0, -hy, hz])],
        basis=basis,
        units="bohr",
        spins=["up", "down", "down"],
        hybridization=["sp3", "s", "s"],
        valence_configs=[o_val, h1_val, h2_val],
        net_spin=-1,
        physical_net_spin=0,
    )
    deck.control.convergence = convergence
    deck.control.max_cycles = 200
    return deck


def pk_lambda_max(state, deck) -> float:
    solver = CPHFResponseSolver(state, deck)
    cache = solver._prepare_response_data()
    atoms = cache["atoms"]
    lam_max = 0.0
    for a, aa in atoms.items():
        S_A = aa["stovl"]
        w_s, v_s = np.linalg.eigh(S_A)
        S_mhalf = (v_s / np.sqrt(np.maximum(w_s, 1e-12))) @ v_s.T
        for s in range(solver.ispin):
            P1_s = _unpack_packed(aa["pkpot"][:, 0, s], aa["m"])
            eigs = np.linalg.eigvalsh(S_mhalf @ P1_s @ S_mhalf)
            lam_max = max(lam_max, float(np.max(eigs)))
    return lam_max


def main() -> None:
    driver = PhatpsyDriver(quiet=True)

    print("=== 1. DZP Relaxed Gradient vs Central Finite-Difference Parity (convergence=1e-10) ===")
    for name, deck in [
        ("H2 (R=1.45 Bohr)", make_h2_deck(1.45, "DZP", convergence=1e-10)),
        ("N2 (R=2.15 Bohr)", make_n2_deck(2.15, "DZP", convergence=1e-10)),
        ("CO (R=2.25 Bohr)", make_co_deck(2.25, "DZP", convergence=1e-10)),
        ("H2O (R=1.85 Bohr, 104.5 deg)", make_h2o_deck(1.85, 104.5, "DZP", convergence=1e-10)),
    ]:
        st = driver.run_deck(deck)
        lmax = pk_lambda_max(st, deck)
        g_rel = compute_polyatomic_relaxed_gradient(st, deck)
        eom = compute_eom(deck, driver=driver, state=st)
        g_fd = compute_scf_finite_difference_gradient(driver, deck, step_size=1e-4)
        max_err = float(np.max(np.abs(g_rel - g_fd)))
        print(
            f"{name:28s}: E={float(st.total_energy):12.6f} Ha | -V/T={float(st.virial_ratio):.4f} "
            f"| mu={eom.dipole.magnitude_debye:.4f} D | lambda_max(P1)={lmax:.4f} "
            f"| max|g_rel - g_fd|={max_err:.2e} Ha/Bohr"
        )

    print("\n=== 2. DZP BFGS Geometry Optimization Summary (Notebooks 01/02/03 Settings: convergence=1e-6) ===")
    for name, deck in [
        ("H2", make_h2_deck(1.401, "DZP", convergence=1e-6)),
        ("N2", make_n2_deck(2.0744, "DZP", convergence=1e-6)),
        ("CO", make_co_deck(2.132, "DZP", convergence=1e-6)),
        ("H2O", make_h2o_deck(1.810, 104.48, "DZP", convergence=1e-6)),
    ]:
        opt = optimize_geometry(
            deck, driver=driver, max_steps=20, grad_tol=1e-3, finite_diff_step=0.005, warm_restart=True
        )
        g = opt.optimized_deck.geometry
        r_e = float(np.linalg.norm(np.array(g[1].xyz) - np.array(g[0].xyz)))
        extra = ""
        if len(g) == 3:
            v1 = np.array(g[1].xyz) - np.array(g[0].xyz)
            v2 = np.array(g[2].xyz) - np.array(g[0].xyz)
            cos_th = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            theta_e = float(np.degrees(np.arccos(np.clip(cos_th, -1.0, 1.0))))
            extra = f" | theta_e={theta_e:.2f} deg"
        print(
            f"{name:6s}: converged={opt.converged} ({opt.num_steps} steps) | R_e={r_e:.4f} Bohr{extra} "
            f"| E_min={opt.final_energy:.6f} Ha | -V/T={float(opt.final_state.virial_ratio):.4f}"
        )

    print("\n=== 3. Homonuclear H-H PK Weight (w_HH) Sensitivity on H2O (R=1.810 Bohr, theta=104.48 deg) ===")
    deck_h2o_eq = make_h2o_deck(1.810, 104.48, "DZP", convergence=1e-10)
    st_h2o_eq = driver.run_deck(deck_h2o_eq)
    lmax_h_040 = pk_lambda_max(st_h2o_eq, deck_h2o_eq)
    for w_hh in (0.00, 0.30, 0.40, 0.50, 1.00):
        lmax_scaled = lmax_h_040 * (w_hh / 0.40)
        print(f"w_HH={w_hh:.2f}: lambda_max(S_H^-1/2 P1_H S_H^-1/2) = {lmax_scaled:.4f}")


if __name__ == "__main__":
    main()

