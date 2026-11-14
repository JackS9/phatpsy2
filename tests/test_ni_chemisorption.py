"""
Stage 11 regression tests:
  1. Exact 1-center Ni(3d^8 4s^2, ^3F) ROHF and [Ar]-core Phillips-Kleinman embedding
     in MaskedVB across SZ and DZP.
  2. Exact joint environment overlap metric (S_BB)^-1 on the 8-neighbor Ni(100) slab:
     proves that lmin(S_tilde) is negative (-2.04) under naive pairwise projection
     and strictly positive (+0.078) under exact (S_BB)^-1 with pk_scale = 1.0 identically.
  3. Top-site CO chemisorption on Ni(100): variational Blyholder domain ordering
     E(Regime 2: +sigma + pi) < E(Regime 1: +sigma) < E(Regime 0: strict fragment domains)
     and synergistic enhancement of pi back-donation on the Ni(100) slab vs bare Ni1.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import eigh

from phatpsy.localized_vb import MaskedVB
from phatpsy.multicenter_integrals import ProlateGrid
from phatpsy.two_center_integrals import STO
from phatpsy.embedding import (
    FrozenEnvironmentAtom,
    embed_from_full_integrals,
    rotate_z_sto_matrix,
    solve_atomic_ni_rohf,
    build_active_ni_co_integrals,
    embed_collinear_in_c4v_slab,
)
from tools.ni_slab_and_chemisorption import (
    analyze_phatpsy_embedded_states,
    solve_nico_blyholder_regimes,
)

HARTREE_TO_EV = 27.211386245988


def test_phatpsy_embedded_states_core_level_shifts():
    """Verify surface vs bulk core-level shifts and 3d crystal-field splitting reduction
    in the converged pk_scale=1.0 PHATPSY embedded states."""
    summary = analyze_phatpsy_embedded_states()
    bulk = summary["Bulk (Ni13, Oh, CN=12)"]
    sub = summary["Subsurface (Ni100, C4v, CN=12)"]
    surf = summary["Surface (Ni100, C4v, CN=8)"]

    # Surface Ni(100) (CN=8) has a positive core-level shift relative to Bulk Ni13 (CN=12)
    de_2p_surf = (surf["e2p"] - bulk["e2p"]) * HARTREE_TO_EV
    de_3p_surf = (surf["e3p"] - bulk["e3p"]) * HARTREE_TO_EV
    assert 2.0 < de_2p_surf < 3.5
    assert 2.0 < de_3p_surf < 3.5

    # Subsurface Ni(100) (CN=12) has a small negative core-level shift relative to Bulk Ni13
    de_2p_sub = (sub["e2p"] - bulk["e2p"]) * HARTREE_TO_EV
    assert -1.0 < de_2p_sub < 0.0

    # Surface 3d crystal-field splitting is reduced by >3x relative to Bulk (missing 4 top neighbors)
    assert surf["e3d_split_ev"] < 0.35 * bulk["e3d_split_ev"]
    assert abs(surf["net_spin"] - 2.0) < 0.05


@pytest.fixture(scope="module")
def ni_dzp_ar_embedded():
    """Shared 1-center Ni(3d^8 4s^2, ^3F) ROHF and [Ar]-core exact backprojected embedding."""
    b, S, T, V, eri, E_rohf, C_core, C_val = solve_atomic_ni_rohf("DZP")
    n = len(b)
    val_idx = [i for i, f in enumerate(b) if (f.n == 3 and f.l == 2) or (f.n == 4)]

    class _FullNi:
        nao = n
        basis = b
        atom_of = np.zeros(n, dtype=int)

        def jk_matrices(self):
            return (
                eri.reshape(n * n, n * n),
                np.ascontiguousarray(eri.transpose(0, 3, 2, 1)).reshape(n * n, n * n),
            )

    _FullNi.S = S
    _FullNi.h = T + V
    _FullNi.eri = eri
    _FullNi.vnn = 0.0

    emb = embed_from_full_integrals(_FullNi(), C_core, val_idx, mode="exact_backprojected")
    return E_rohf, C_val, val_idx, emb


def test_atomic_ni_rohf_and_ar_core_embedding(ni_dzp_ar_embedded):
    """Verify 1-center Ni(3d^8 4s^2, ^3F) ROHF and [Ar]-core PK embedding accuracy (< 1 mHa)."""
    E_rohf, C_val, val_idx, emb = ni_dzp_ar_embedded
    assert E_rohf == pytest.approx(-1506.407900, abs=1e-4)

    m_v = np.array([f.m for f in emb.basis])
    l_v = np.array([f.l for f in emb.basis])
    roles = ["inactive"] * 4 + ["active"] * 2
    masks = [
        np.where((l_v == 0) & (m_v == 0))[0],
        np.where((l_v == 2) & (m_v == 2))[0],
        np.where((l_v == 2) & (m_v == 1))[0],
        np.where((l_v == 2) & (m_v == -1))[0],
        np.where((l_v == 2) & (m_v == 0))[0],
        np.where((l_v == 2) & (m_v == -2))[0],
    ]
    mvb = MaskedVB(emb, roles, masks, (2, 0))
    C_v_sub = C_val[val_idx, :]
    res = mvb.optimize(mvb.pack([C_v_sub[:, k] for k in range(6)]), maxiter=50, gtol=1e-7)
    assert abs(res.fun - E_rohf) < 1.0e-3


def test_ni100_slab_joint_metric_positivity(ni_dzp_ar_embedded):
    """Fast Stage 11 check: exact (S_BB)^-1 on the 8-neighbor Ni(100) slab eliminates the negative
    naive pseudometric eigenvalue without empirical pk_scale damping."""
    _, _, _, emb = ni_dzp_ar_embedded
    b_0 = emb.basis
    surf_rep = FrozenEnvironmentAtom.from_embedded_state("data/ni_surf_embedded.json", (4.709, 0.0, 0.0))
    sub_rep = FrozenEnvironmentAtom.from_embedded_state("data/ni_sub_embedded.json", (2.3545, 2.3545, -3.330))

    env_atoms = []
    for rep in [surf_rep, sub_rep]:
        w_occ = rep.occ_weights if rep.occ_weights is not None else np.ones(rep.C_occ.shape[1])
        x0, y0, z0 = rep.center
        for k in range(4):
            phi = k * (0.5 * math.pi)
            c, s = math.cos(phi), math.sin(phi)
            xyz_k = (c * x0 - s * y0, s * x0 + c * y0, z0)
            U_E = rotate_z_sto_matrix(rep.basis, phi)
            b_k = [STO(xyz_k, f.n, f.l, f.m, f.zeta, f.norm()) for f in rep.basis]
            env_atoms.append(FrozenEnvironmentAtom(rep.z_nuc, xyz_k, b_k, U_E @ rep.C_occ, w_occ.copy()))

    n_orb_per = surf_rep.C_occ.shape[1]
    S_BB_orb = np.eye(8 * n_orb_per)
    pair_cache = {}
    for i in range(8):
        Ei = env_atoms[i]
        wi = Ei.occ_weights if Ei.occ_weights is not None else np.ones(n_orb_per)
        for j in range(i):
            Ej = env_atoms[j]
            wj = Ej.occ_weights if Ej.occ_weights is not None else np.ones(n_orb_per)
            if np.linalg.norm(np.asarray(Ei.center) - np.asarray(Ej.center)) > 7.5:
                continue
            key = tuple(round(float(Ei.center[a] - Ej.center[a]), 5) for a in range(3))
            if key not in pair_cache:
                g_ij = ProlateGrid(Ei.center, Ej.center, zeta_min=0.95, n_eta=16, n_phi=8)
                chi_i = np.array([g_ij.evaluate(f) for f in Ei.basis])
                chi_j = np.array([g_ij.evaluate(f) for f in Ej.basis])
                pair_cache[key] = np.einsum("pijk,qijk,ijk->pq", chi_i, chi_j, g_ij.weights)
            s_ij = (Ei.C_occ.T @ pair_cache[key] @ Ej.C_occ) * np.sqrt(np.outer(wi, wj))
            sl_i = slice(i * n_orb_per, (i + 1) * n_orb_per)
            sl_j = slice(j * n_orb_per, (j + 1) * n_orb_per)
            S_BB_orb[sl_i, sl_j] = s_ij
            S_BB_orb[sl_j, sl_i] = s_ij.T

    U_0_list = [rotate_z_sto_matrix(b_0, k * (0.5 * math.pi)) for k in range(4)]
    S_0B_blocks = []
    for rep in [surf_rep, sub_rep]:
        w_occ = rep.occ_weights if rep.occ_weights is not None else np.ones(n_orb_per)
        g_p = ProlateGrid((0.0, 0.0, 0.0), rep.center, zeta_min=0.95, n_eta=16, n_phi=8)
        chi_0 = np.array([g_p.evaluate(f) for f in b_0])
        chi_E = np.array([g_p.evaluate(f) for f in rep.basis])
        S_0_orb = np.einsum("pijk,qijk,ijk->pq", chi_0, chi_E, g_p.weights) @ rep.C_occ
        for k in range(4):
            S_0B_blocks.append((U_0_list[k] @ S_0_orb) * np.sqrt(w_occ)[None, :])

    S_0B_orb = np.hstack(S_0B_blocks)
    dS_00_slab = S_0B_orb @ np.linalg.solve(S_BB_orb, S_0B_orb.T)
    S_00_naive = emb.S - S_0B_orb @ S_0B_orb.T
    lmin_nv = float(np.min(eigh(0.5 * (S_00_naive + S_00_naive.T), emb.S_bare, eigvals_only=True)))
    S_00_slab = 0.5 * ((emb.S - dS_00_slab) + (emb.S - dS_00_slab).T)
    lmin_ex = float(np.min(eigh(S_00_slab, emb.S_bare, eigvals_only=True)))

    assert lmin_nv < -1.5
    assert lmin_ex > 0.05


@pytest.mark.slow
def test_ni100_slab_metric_and_blyholder_synergism():
    """Verify that:
      1. The 8-neighbor Ni(100) slab has lmin(S_tilde_naive) < -1.5 under naive pairwise
         projection, and lmin(S_tilde_exact) > +0.05 under the exact (S_BB)^-1 metric
         with zero empirical pk_scale damping (pk_scale = 1.0).
      2. Top-site CO chemisorption at R_NiC = 3.59 Bohr obeys strict variational ordering
         E(Regime 2) < E(Regime 1) < E(Regime 0) and exhibits enhanced pi back-donation
         on the Ni(100) slab relative to a bare Ni1 atom."""
    R_NiC = 3.59
    R_CO = 2.132
    mol_bare, C_val_Ni_sub = build_active_ni_co_integrals(
        R_NiC, R_CO, "DZP", n_eta=20, n_phi=10, lmax=10
    )
    b0, b1, b2, _ = solve_nico_blyholder_regimes(mol_bare, C_val_Ni_sub, maxiter=60)
    assert b1 < b0 - 0.01
    assert b2 < b1 - 0.01

    surf_rep = FrozenEnvironmentAtom.from_embedded_state("data/ni_surf_embedded.json", (4.709, 0.0, 0.0))
    sub_rep = FrozenEnvironmentAtom.from_embedded_state("data/ni_sub_embedded.json", (2.3545, 2.3545, -3.330))
    mol_slab, lmin_ex, lmin_nv = embed_collinear_in_c4v_slab(
        mol_bare,
        [10.0, 6.0, 8.0],
        [(0.0, 0.0, 0.0), (0.0, 0.0, R_NiC), (0.0, 0.0, R_NiC + R_CO)],
        surf_rep,
        sub_rep,
        n_eta=18,
        n_phi=8,
        lmax_exch=8,
    )
    assert lmin_nv < -1.5
    assert lmin_ex > 0.05

    s0, s1, s2, _ = solve_nico_blyholder_regimes(mol_slab, C_val_Ni_sub, maxiter=60)
    assert s1 < s0 - 0.004
    assert s2 < s1 - 0.02

    # Slab embedding suppresses excessive 4s sigma over-binding and enhances 3d_pi -> 2pi* backbonding:
    dE_sig_bare = (b1 - b0) * HARTREE_TO_EV
    dE_pi_bare = (b2 - b1) * HARTREE_TO_EV
    dE_sig_slab = (s1 - s0) * HARTREE_TO_EV
    dE_pi_slab = (s2 - s1) * HARTREE_TO_EV
    assert abs(dE_sig_slab) < abs(dE_sig_bare)
    assert abs(dE_pi_slab) > 1.25 * abs(dE_pi_bare)

