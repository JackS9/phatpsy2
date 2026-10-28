# tests/test_modelpot.py
"""Unit tests for modernized modelpot module (Group 3)."""
import math
import numpy as np
import pytest
from _phatpsy_core import modelpot


def test_genfac_factorials():
    """Verify factorial table generation matches math.factorial."""
    nfact = 12
    fact = modelpot.genfac(nfact)
    assert len(fact) == nfact
    for n in range(nfact):
        expected = float(math.factorial(n))
        assert fact[n] == pytest.approx(expected, rel=1e-14)


def test_genbc_binomial_coefficients():
    """Verify Pascal's triangle binomial coefficients match math.comb."""
    nrows = 7
    # index_arr(n) = n*(n-1)/2 (1-based), so index_arr[i] for row i (0-based)
    index_arr = np.array([(i * (i - 1)) // 2 for i in range(1, nrows + 1)], dtype=np.int32)
    nbinom = (nrows * (nrows + 1)) // 2
    binom = modelpot.genbc(nbinom, index_arr)

    # In Fortran, row n (1-based, order n-1):
    # binom(index(n)+m) for m=1..n corresponds to comb(n-1, m-1)
    for n in range(1, nrows + 1):
        idx = (n * (n - 1)) // 2
        for m in range(1, n + 1):
            val = binom[idx + m - 1]
            expected = float(math.comb(n - 1, m - 1))
            assert val == pytest.approx(expected, rel=1e-14)


def test_gennrm_spherical_harmonic_normalization():
    """Verify spherical harmonic normalization constants against analytical formulas."""
    lmax = 2
    nfact = 2 * lmax + 2
    fact = modelpot.genfac(nfact)
    nplm = ((lmax + 1) * (lmax + 2)) // 2
    plmnrm = modelpot.gennrm_default(fact, nplm, lmax, True)

    # Analytical real spherical harmonic normalization:
    # N(0,0) = sqrt(1 / (4*pi))
    # N(1,0) = sqrt(3 / (4*pi))
    # N(1,1) = sqrt(3 / (4*pi))
    pi = math.pi
    expected_n00 = math.sqrt(1.0 / (4.0 * pi))
    expected_n10 = math.sqrt(3.0 / (4.0 * pi))
    expected_n11 = math.sqrt(3.0 / (4.0 * pi))

    assert plmnrm[0] == pytest.approx(expected_n00, rel=1e-12)
    assert plmnrm[1] == pytest.approx(expected_n10, rel=1e-12)
    assert plmnrm[2] == pytest.approx(expected_n11, rel=1e-12)


def test_intnrm_intermediate_normalization():
    """Verify intermediate normalization <i|(1-P)|i> = 1."""
    nbas = 2
    nvec = 1
    n2bas = (nbas * (nbas + 1)) // 2

    # Vector initially normalized to 1 in standard sense: evec = [1/sqrt(2), 1/sqrt(2)]
    evec = np.array([[1.0 / math.sqrt(2.0)], [1.0 / math.sqrt(2.0)]], dtype=np.float64, order="F")

    # Projection operator P packed: P11=0.2, P12=0.0, P22=0.2 (P has eigenvalue 0.2)
    # <evec|P|evec> = 0.5*0.2 + 0.5*0.2 = 0.2
    # So <evec|(1-P)|evec> = 1 - 0.2 = 0.8
    projec = np.array([0.2, 0.0, 0.2], dtype=np.float64)

    evec_norm, scrat = modelpot.intnrm(evec, projec)

    # After normalization: evec should be scaled by 1 / sqrt(0.8)
    expected_scale = 1.0 / math.sqrt(0.8)
    expected_evec = (1.0 / math.sqrt(2.0)) * expected_scale
    assert evec_norm[0, 0] == pytest.approx(expected_evec, rel=1e-12)
    assert evec_norm[1, 0] == pytest.approx(expected_evec, rel=1e-12)

    # Check that <evec_norm|(1-P)|evec_norm> == 1.0
    v = evec_norm[:, 0]
    p_mat = np.array([[0.2, 0.0], [0.0, 0.2]])
    overlap_1_minus_p = v @ (np.eye(2) - p_mat) @ v
    assert overlap_1_minus_p == pytest.approx(1.0, rel=1e-12)


def test_dampv_matrix_scaling():
    """Verify dampv scales potential matrices by vdamp factor."""
    msto = 2
    m2sto = 3
    nvterm = 2
    ispin = 2
    vdamp = 0.75

    epot2 = np.ones((m2sto, ispin), dtype=np.float64, order="F")
    pkpot = np.ones((m2sto, 4, ispin), dtype=np.float64, order="F")
    zpot2 = np.ones(m2sto, dtype=np.float64)
    vpot2 = np.ones(nvterm, dtype=np.float64)

    epot2, pkpot, zpot2, vpot2 = modelpot.dampv(epot2, pkpot, zpot2, vpot2, msto, vdamp)

    assert np.allclose(epot2, 0.75)
    # pkpot terms 1:2 (0 and 1 in 0-indexed) are damped, terms 2:3 are untouched
    assert np.allclose(pkpot[:, 0:2, :], 0.75)
    assert np.allclose(pkpot[:, 2:4, :], 1.0)
    assert np.allclose(zpot2, 0.75)
    assert np.allclose(vpot2, 0.75)


def test_constrained_model_potential_fit():
    """Verify Phase 3.2: Exact equality-constrained model potential fitting."""
    from phatpsy.builder import build_diatomic
    from phatpsy.driver import PhatpsyDriver

    driver = PhatpsyDriver(quiet=True)

    # 1. Standard N2 calculation with legacy penalty fit
    deck_leg = build_diatomic("N", "N", 2.068, basis="DZ", spin_flip=True, units="bohr")
    deck_leg.control.use_constrained_fit = False
    deck_leg.control.pk_scale = 0.0
    res_leg = driver.run_deck(deck_leg)

    # 2. Modern N2 calculation with exact equality-constrained fit (QCSTFT=T)
    deck_cst = build_diatomic("N", "N", 2.068, basis="DZ", spin_flip=True, units="bohr")
    deck_cst.control.use_constrained_fit = True
    deck_cst.control.pk_scale = 0.0
    res_cst = driver.run_deck(deck_cst)

    assert res_leg.converged is True
    assert res_cst.converged is True

    # Physical virial ratio bounds: -V/T ~ 2.0 (allow 0.06 for cross-BLAS variance)
    assert abs(res_cst.virial_ratio - 2.0) <= 0.06

    # Energy difference is small and consistent with eliminating soft penalty cusp distortion
    delta_e = abs(res_cst.total_energy - res_leg.total_energy)
    assert delta_e < 0.03  # within 0.03 Ha of legacy baseline


def test_dynamic_exponent_scaling():
    """Verify Phase 3.1: Charge-dependent model potential exponent scaling formula."""
    vexp_base = np.array([0.0, 1.5, 3.2, 5.8], dtype=np.float64)
    znuc = 7.0  # Nitrogen

    # 1. Neutral atom: q = 0.0 -> identical base exponents
    vexp_neutral = modelpot.scale_screening_exponents(vexp_base, znuc, 0.0)
    np.testing.assert_allclose(vexp_neutral, vexp_base, atol=1e-14)

    # 2. Cation: q = +1.0 -> screening contracts (exponents increase)
    expected_scale_cat = (1.0 + 1.0 / 7.0) ** (1.0 / 3.0)
    expected_cat = np.copy(vexp_base) * expected_scale_cat
    expected_cat[0] = 0.0
    vexp_cat = modelpot.scale_screening_exponents(vexp_base, znuc, 1.0)
    np.testing.assert_allclose(vexp_cat, expected_cat, rtol=1e-12)
    assert np.all(vexp_cat[1:] > vexp_base[1:])

    # 3. Anion: q = -1.0 -> screening expands (exponents decrease)
    expected_scale_an = (1.0 - 1.0 / 7.0) ** (1.0 / 3.0)
    expected_an = np.copy(vexp_base) * expected_scale_an
    expected_an[0] = 0.0
    vexp_an = modelpot.scale_screening_exponents(vexp_base, znuc, -1.0)
    np.testing.assert_allclose(vexp_an, expected_an, rtol=1e-12)
    assert np.all(vexp_an[1:] < vexp_base[1:])


def test_dynamic_exponents_deck():
    """Verify Phase 3.1: Full SCF calculation with QDYNEX dynamic exponent optimization."""
    from phatpsy.builder import build_diatomic
    from phatpsy.driver import PhatpsyDriver

    driver = PhatpsyDriver(quiet=True)
    deck = build_diatomic("N", "N", 2.068, basis="DZ", spin_flip=True, units="bohr")
    deck.control.use_dynamic_exponents = True
    deck.control.max_cycles = 10

    stdin_text = deck.to_stdin_text()
    assert "QDYNEX=T" in stdin_text

    res = driver.run_deck(deck)
    assert res.converged is True
    assert abs(res.virial_ratio - 2.0) < 0.05
    assert math.isclose(res.total_energy, -109.03, abs_tol=0.2)
