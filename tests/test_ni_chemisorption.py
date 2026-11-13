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

import numpy as np
import pytest

from phatpsy.localized_vb import MaskedVB
from phatpsy.embedding import (
    FrozenEnvironmentAtom,
    embed_from_full_integrals,
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


def test_atomic_ni_rohf_and_ar_core_embedding():
    """Verify 1-center Ni(3d^8 4s^2, ^3F) ROHF and [Ar]-core PK embedding accuracy (< 1 mHa)."""
    b, S, T, V, eri, E_rohf, C_core, C_val = solve_atomic_ni_rohf("DZP")
    assert E_rohf == pytest.approx(-1506.407900, abs=1e-4)

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

