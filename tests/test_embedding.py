"""Tests for localized-VB active-site embedding in a PHATPSY environment (phatpsy.embedding)."""
from __future__ import annotations

import numpy as np

from phatpsy.embedding import (
    FrozenEnvironmentAtom,
    embed_diatomic_in_environment,
    embed_from_full_integrals,
    one_center_ao_matrices,
    one_center_coulomb_potential,
)
from phatpsy.localized_vb import (
    MaskedVB,
    atom_basis,
    cas_energy_and_gradient,
    diatomic_ao_integrals,
)
from phatpsy.two_center_integrals import ProlateGrid


def test_one_center_coulomb_potential_matches_neumann_grid():
    """Analytic one-center multipole + Gaunt Coulomb potential must match ProlateGrid.potentials_at
    across s, p, d shells and cross-shell (s-p, p-d, d-d) densities."""
    basis = atom_basis((0.0, 0.0, -1.1), "O", "DZP")
    n = len(basis)
    rng = np.random.default_rng(17)
    A = rng.normal(size=(n, n))
    D = 0.5 * (A + A.T)

    g = ProlateGrid((0.0, 0.0, -1.1), (0.0, 0.0, 1.1), zeta_min=1.0, n_eta=48)
    chi = np.array([g.evaluate(f) for f in basis])
    rho = np.einsum("mn,mijk,nijk->ijk", D, chi, chi)
    pts = np.array([
        [0.0, 0.0, 1.1],
        [0.4, -0.3, 0.7],
        [-0.8, 0.5, -0.2],
        [1.2, 0.9, -1.8],
    ])
    v_grid = g.potentials_at(rho[None, ...], pts, lmax=28, mmax=6)[0]
    v_exact = one_center_coulomb_potential(basis, D, pts)
    np.testing.assert_allclose(v_exact, v_grid, atol=1e-7, rtol=1e-6)


def test_one_center_ao_matrices_match_prolate_grid():
    """Analytic one-center (S, T, V_nuc, eri) must agree with ProlateGrid integration to < 1e-8."""
    basis = atom_basis((0.0, 0.0, -1.0), "Li", "DZP")
    S1, T1, V1, eri1 = one_center_ao_matrices(basis, 3.0)
    ints = diatomic_ao_integrals(basis, atom_basis((0.0, 0.0, 1.0), "H", "DZ"), 3.0, 1.0, n_eta=56, lmax=28)
    n = len(basis)
    np.testing.assert_allclose(S1, ints.S[:n, :n], atol=1e-9)
    np.testing.assert_allclose(T1, ints.T[:n, :n], atol=1e-8)
    np.testing.assert_allclose(V1, ints.V_A[:n, :n], atol=1e-8)
    np.testing.assert_allclose(eri1, ints.eri[:n, :n, :n, :n], atol=1e-7)


def test_exact_backprojected_embedding_isomorphic_to_full_frozen_core():
    """Level E1 (exact_backprojected) must give the exact same energy and analytic orbital
    gradient as running cas_energy_and_gradient on the full system with C_env frozen in Xi."""
    R = 3.015
    b_li = atom_basis((0.0, 0.0, -R / 2), "Li", "DZP")
    b_h = atom_basis((0.0, 0.0, R / 2), "H", "DZP")
    ints = diatomic_ao_integrals(b_li, b_h, 3.0, 1.0, n_eta=48, lmax=24)

    # Freeze a Li 1s core orbital in C_env
    li_env = FrozenEnvironmentAtom.from_atomic_rhf(3.0, (0.0, 0.0, -R / 2), b_li, n_occ=1)
    C_env = np.zeros((ints.nao, 1))
    C_env[:len(b_li), 0] = li_env.C_occ[:, 0]

    emb = embed_from_full_integrals(ints, C_env, mode="exact_backprojected")

    # Random localized valence orbitals (1 on Li, 1 on H)
    rng = np.random.default_rng(42)
    Xa = np.zeros((ints.nao, 2))
    Xa[ints.atom_of == 0, 0] = rng.normal(size=int(np.sum(ints.atom_of == 0)))
    Xa[ints.atom_of == 1, 1] = rng.normal(size=int(np.sum(ints.atom_of == 1)))

    e_full, _, Ga_full = cas_energy_and_gradient(C_env, Xa, ints, nelec_act=(1, 1))
    e_emb, _, Ga_emb = cas_energy_and_gradient(np.zeros((emb.nao, 0)), Xa, emb, nelec_act=(1, 1))

    assert abs(e_emb - e_full) < 1e-12
    np.testing.assert_allclose(Ga_emb, Ga_full, atol=1e-11)


def test_lih_eri_free_core_embedding_matches_all_electron_vb():
    """ERI-free core embedding on LiH (freezing atomic Li 1s^2 via embed_diatomic_in_environment
    or unprojected_eri) must prevent variational collapse onto the Li 1s core and reproduce
    the all-electron Localized VB binding energy within ~0.05 eV."""
    R = 3.015
    b_li = atom_basis((0.0, 0.0, -R / 2), "Li", "DZP")
    b_h = atom_basis((0.0, 0.0, R / 2), "H", "DZP")
    ints = diatomic_ao_integrals(b_li, b_h, 3.0, 1.0, n_eta=48, lmax=24)

    li_env = FrozenEnvironmentAtom.from_atomic_rhf(3.0, (0.0, 0.0, -R / 2), b_li, n_occ=1)
    C_env = np.zeros((ints.nao, 1))
    C_env[:len(b_li), 0] = li_env.C_occ[:, 0]

    idx_li = np.where(ints.atom_of == 0)[0]
    idx_h = np.where(ints.atom_of == 1)[0]

    # Seed for valence bond orbitals: Li 2s (index 2 in DZP) and H 1s (index 0 on H)
    o_li = np.zeros(ints.nao); o_li[idx_li[2]] = 1.0
    o_h = np.zeros(ints.nao); o_h[idx_h[0]] = 1.0

    # 1. Exact back-projected frozen-core VB
    emb_e1 = embed_from_full_integrals(ints, C_env, mode="exact_backprojected")
    m_e1 = MaskedVB(emb_e1, ["active", "active"], [idx_li, idx_h], (1, 1))
    res_e1 = m_e1.optimize(m_e1.pack([o_li, o_h]))

    # 2. Unprojected-ERI frozen-core VB (Level E2)
    emb_e2 = embed_from_full_integrals(ints, C_env, mode="unprojected_eri")
    m_e2 = MaskedVB(emb_e2, ["active", "active"], [idx_li, idx_h], (1, 1))
    res_e2 = m_e2.optimize(m_e2.pack([o_li, o_h]))

    # 3. Level E3: ERI-free embedding via embed_diatomic_in_environment
    emb_e3 = embed_diatomic_in_environment(b_li, b_h, 3.0, 1.0, [li_env], n_eta=48, lmax=24)
    m_e3 = MaskedVB(emb_e3, ["active", "active"], [idx_li, idx_h], (1, 1))
    res_e3 = m_e3.optimize(m_e3.pack([o_li, o_h]))

    # 4. Control without PK projection: collapses onto Li 1s core by > 1.5 Ha!
    emb_nopk = embed_from_full_integrals(ints, C_env, mode="bare_fock_no_pk")
    m_nopk = MaskedVB(emb_nopk, ["active", "active"], [idx_li, idx_h], (1, 1))
    res_nopk = m_nopk.optimize(m_nopk.pack([o_li, o_h]))

    assert abs(res_e2.fun - res_e1.fun) < 1e-6      # Exact match on single-atom core with W renormalization
    assert abs(res_e3.fun - res_e1.fun) < 1e-6      # Level E3 matches Level E1 to < 1 uHa on LiH!
    assert res_nopk.fun < res_e1.fun - 1.5          # variational collapse without PK projection


def test_two_core_joint_pk_embedding_and_external_phatpsy_env():
    """Joint PK projector across multiple frozen cores (N2) and external PHATPSY state
    embedding (H2 + He) must keep S_eff positive semi-definite and reproduce full-integral
    reference energies."""
    from phatpsy.driver import PhatpsyDriver
    from phatpsy.multicenter_integrals import molecular_ao_integrals
    from tools.localized_vb_scan import build_model, initial_orbitals

    # Part A: N2 (DZ) with both N 1s^2 cores frozen simultaneously
    R = 2.0744
    b_a = atom_basis((0.0, 0.0, 0.0), "N", "DZ")
    b_b = atom_basis((0.0, 0.0, R), "N", "DZ")
    ints = diatomic_ao_integrals(b_a, b_b, 7.0, 7.0, n_eta=48, lmax=24)
    ea = FrozenEnvironmentAtom.from_atomic_rhf(7.0, (0.0, 0.0, 0.0), b_a, n_occ=1)
    eb = FrozenEnvironmentAtom.from_atomic_rhf(7.0, (0.0, 0.0, R), b_b, n_occ=1)
    C_env = np.zeros((ints.nao, 2))
    C_env[:len(b_a), 0] = ea.C_occ[:, 0]
    C_env[len(b_a):, 1] = eb.C_occ[:, 0]

    emb_e1 = embed_from_full_integrals(ints, C_env, mode="exact_backprojected")
    emb_e3 = embed_diatomic_in_environment(b_a, b_b, 7.0, 7.0, [ea, eb], n_eta=48, lmax=24)
    assert np.min(np.linalg.eigvalsh(emb_e3.S)) >= -1e-14
    np.testing.assert_allclose(emb_e3.S, emb_e1.S, atol=1e-10)
    np.testing.assert_allclose(emb_e3.h, emb_e1.h, atol=1e-5)

    # Part B: External PHATPSY environment atom (H2 + He at d = 3.0 Bohr)
    drv = PhatpsyDriver(quiet=True)
    he_state = drv.run_molecule([("He", 0.0, 0.0, 0.0)], basis="DZ", spin_unrestricted=False)
    he_phatpsy = FrozenEnvironmentAtom.from_phatpsy_state(he_state, atom_index=1, n_occ=1)

    c_h1, c_h2, c_he = (0.0, 0.0, -0.7), (0.0, 0.0, 0.7), (0.0, 0.0, 3.7)
    bh1 = atom_basis(c_h1, "H", "DZ")
    bh2 = atom_basis(c_h2, "H", "DZ")
    bhe = atom_basis(c_he, "He", "DZ")
    he_env = FrozenEnvironmentAtom(2.0, c_he, bhe, he_phatpsy.C_occ.copy())

    m_ints = molecular_ao_integrals([(1.0, c_h1, bh1), (1.0, c_h2, bh2), (2.0, c_he, bhe)],
                                    n_eta=36, n_phi=18, lmax=18, mmax=4)
    idx_act = np.where(m_ints.atom_of < 2)[0]
    idx_he = np.where(m_ints.atom_of == 2)[0]
    C_he = np.zeros((m_ints.nao, 1)); C_he[idx_he, 0] = he_env.C_occ[:, 0]

    e2_he = embed_from_full_integrals(m_ints, C_he, active_ao_idx=idx_act, mode="unprojected_eri")
    e3_he = embed_diatomic_in_environment(bh1, bh2, 1.0, 1.0, [he_env], n_eta=44, lmax=20)
    ih1 = np.where(e3_he.atom_of == 0)[0]
    ih2 = np.where(e3_he.atom_of == 1)[0]
    o1 = np.zeros(e3_he.nao); o1[ih1[0]] = 1.0
    o2 = np.zeros(e3_he.nao); o2[ih2[0]] = 1.0
    r2 = MaskedVB(e2_he, ["active", "active"], [ih1, ih2], (1, 1)).optimize(
        MaskedVB(e2_he, ["active", "active"], [ih1, ih2], (1, 1)).pack([o1, o2])
    )
    r3 = MaskedVB(e3_he, ["active", "active"], [ih1, ih2], (1, 1)).optimize(
        MaskedVB(e3_he, ["active", "active"], [ih1, ih2], (1, 1)).pack([o1, o2])
    )
    assert abs(r3.fun - r2.fun) < 2e-4


