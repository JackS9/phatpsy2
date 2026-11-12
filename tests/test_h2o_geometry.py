"""Tests for Stage 10 (Step 1): First-principles H2O equilibrium geometry and rotational invariance."""
from __future__ import annotations

import numpy as np

from phatpsy.embedding import embed_from_full_integrals
from phatpsy.localized_vb import MaskedVB, atom_basis, pyscf_mean_field
from phatpsy.multicenter_integrals import molecular_ao_integrals
from tools.h2o_geometry_scan import evaluate_h2o_point, fit_minimum_quadratic, get_water_integrals


def test_h2o_rotational_invariance_dz():
    """Exact multi-center STO integrals and energies on H2O are invariant under 3D frame rotation."""
    R = 1.8088
    theta = 104.52
    th = np.radians(theta) / 2.0
    O_xyz = np.zeros(3)

    # Frame A: bisector along +z in yz-plane
    H1_a = np.array([0.0, R * np.sin(th), R * np.cos(th)])
    H2_a = np.array([0.0, -R * np.sin(th), R * np.cos(th)])
    atoms_a = [
        (8.0, O_xyz, atom_basis(O_xyz, "O", "DZ")),
        (1.0, H1_a, atom_basis(H1_a, "H", "DZ")),
        (1.0, H2_a, atom_basis(H2_a, "H", "DZ")),
    ]

    # Frame B: H1 along +z, H2 in xz-plane at angle theta
    rad = np.radians(theta)
    H1_b = np.array([0.0, 0.0, R])
    H2_b = np.array([R * np.sin(rad), 0.0, R * np.cos(rad)])
    atoms_b = [
        (8.0, O_xyz, atom_basis(O_xyz, "O", "DZ")),
        (1.0, H1_b, atom_basis(H1_b, "H", "DZ")),
        (1.0, H2_b, atom_basis(H2_b, "H", "DZ")),
    ]

    ints_a = molecular_ao_integrals(atoms_a)
    ints_b = molecular_ao_integrals(atoms_b)

    mf_a = pyscf_mean_field(ints_a, (5, 5), "RHF")
    mf_b = pyscf_mean_field(ints_b, (5, 5), "RHF")
    assert abs(mf_a.e_tot - mf_b.e_tot) < 1e-11


def test_h2o_first_principles_bond_angle_and_core_embedding_dz():
    """H2O equilibrium bond angle ~105 deg emerges without empirical w_HH tuning, and E1/E2 embedding match."""
    R = 1.8088
    thetas = [95.0, 104.52, 115.0]
    e_strict = []
    e_emb_e1 = []
    e_emb_e2 = []
    e_opt1 = []

    prev = None
    for th in thetas:
        ints = get_water_integrals("DZ", R, th)
        ens, prev, _, _ = evaluate_h2o_point(ints, R, th, prev)
        e_strict.append(ens["strict_1c"])
        e_emb_e1.append(ens["embedded_1s_e1"])
        e_emb_e2.append(ens["embedded_1s_e2"])
        e_opt1.append(ens["option1_bond_local_lp"])
        # Level E1 and Level E2 must agree to < 5 microHartree
        assert abs(ens["embedded_1s_e1"] - ens["embedded_1s_e2"]) < 5e-6
        # Frozen O 1s^2 core must stay within 0.3 mHa of all-electron Strict 1c VB
        assert abs(ens["embedded_1s_e1"] - ens["strict_1c"]) < 3e-4

    # Interior point (104.52 deg) must be strictly lower in energy than both 95 deg and 115 deg
    for curve in (e_strict, e_emb_e1, e_emb_e2, e_opt1):
        assert curve[1] < curve[0] - 1e-3
        assert curve[1] < curve[2] - 1e-3

    th_rad = np.radians(thetas)
    th_strict_deg = float(np.degrees(fit_minimum_quadratic(th_rad, e_strict)[0]))
    th_emb_deg = float(np.degrees(fit_minimum_quadratic(th_rad, e_emb_e2)[0]))
    assert 103.0 <= th_strict_deg <= 108.0
    assert abs(th_strict_deg - th_emb_deg) < 0.1

