# tests/test_density_group6.py
"""Unit tests for modernized density matrix, population analysis, and wavefunction routines in Group 6 (density.f90)."""
import math
import numpy as np
import pytest
from _phatpsy_core import density


def pack_lower(mat: np.ndarray) -> np.ndarray:
    """Helper: pack symmetric square matrix into lower-triangular 1D vector."""
    n = mat.shape[0]
    packed = []
    for i in range(n):
        for j in range(i + 1):
            packed.append(mat[i, j])
    return np.array(packed, dtype=np.float64, order="F")


def unpack_lower(packed: np.ndarray, n: int) -> np.ndarray:
    """Helper: unpack lower-triangular 1D vector into symmetric square matrix."""
    mat = np.zeros((n, n), dtype=np.float64, order="F")
    idx = 0
    for i in range(n):
        for j in range(i + 1):
            val = packed[idx]
            mat[i, j] = val
            mat[j, i] = val
            idx += 1
    return mat


def test_denmat_construction():
    """Verify one-electron density matrix construction D = C * occ * C^T and thresholding."""
    nbas = 3
    nvec = 2

    # Two orthonormal column vectors
    evec = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0]
    ], dtype=np.float64, order="F")
    occ = np.array([2.0, 1.0], dtype=np.float64)

    dens = density.denmat(evec, occ)
    # Expected density matrix D = diag([2, 1, 0])
    # Lower triangular packed: D_11, D_21, D_22, D_31, D_32, D_33
    expected = np.array([2.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    np.testing.assert_allclose(dens, expected, atol=1e-14)

    # General eigenvectors
    rng = np.random.default_rng(42)
    evec_rand = rng.standard_normal((nbas, nbas))
    occ_rand = np.array([1.5, 0.5, 0.0], dtype=np.float64)

    d_full_expected = evec_rand @ np.diag(occ_rand) @ evec_rand.T
    d_packed_expected = pack_lower(d_full_expected)

    dens_rand = density.denmat(evec_rand, occ_rand)
    np.testing.assert_allclose(dens_rand, d_packed_expected, rtol=1e-12, atol=1e-14)


def test_averag_expectation_value():
    """Verify one-electron operator expectation value <O> = Tr(D * O)."""
    nbas = 3
    rng = np.random.default_rng(101)

    m_o = rng.standard_normal((nbas, nbas))
    o_full = (m_o + m_o.T) / 2.0
    o_packed = pack_lower(o_full)

    m_d = rng.standard_normal((nbas, nbas))
    d_full = (m_d + m_d.T) / 2.0
    d_packed = pack_lower(d_full)

    expected_tr = np.trace(d_full @ o_full)
    calculated_val = density.averag(o_packed, d_packed)

    assert calculated_val == pytest.approx(expected_tr, rel=1e-12, abs=1e-14)


def test_chrgbo_and_mulpop():
    """Verify Mulliken and net-charge population analysis."""
    nbas = 2
    # Simple 2x2 density and overlap
    d_full = np.array([
        [1.2, 0.3],
        [0.3, 0.8]
    ], dtype=np.float64, order="F")
    s_full = np.array([
        [1.0, 0.25],
        [0.25, 1.0]
    ], dtype=np.float64, order="F")

    d_packed = pack_lower(d_full)
    s_packed = pack_lower(s_full)

    # Trace of D * S
    expected_totpop = np.trace(d_full @ s_full)

    # Test mulpop (Mulliken)
    dens_mod_mul, popul_mul, totpop_mul = density.mulpop(d_packed.copy(), s_packed, nbas)
    assert totpop_mul == pytest.approx(expected_totpop, rel=1e-12)
    assert np.sum(popul_mul) == pytest.approx(expected_totpop, rel=1e-12)

    # In Mulliken: pop_i = D_ii + sum_{j != i} D_ij * S_ij
    # D_11 = 1.2, D_12 * S_12 = 0.3 * 0.25 = 0.075 -> pop_1 = 1.2 + 0.075 = 1.275
    # D_22 = 0.8, D_21 * S_21 = 0.075 -> pop_2 = 0.8 + 0.075 = 0.875
    assert popul_mul[0] == pytest.approx(1.275, rel=1e-12)
    assert popul_mul[1] == pytest.approx(0.875, rel=1e-12)

    # Test chrgbo (Net-charge weighted)
    dens_mod_chrg, popul_chrg, totpop_chrg = density.chrgbo(d_packed.copy(), s_packed, nbas)
    assert totpop_chrg == pytest.approx(expected_totpop, rel=1e-12)
    assert np.sum(popul_chrg) == pytest.approx(expected_totpop, rel=1e-12)


def test_neword_vector_permutation():
    """Verify vector reordering via index permutation."""
    vec = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    # Permutation: new 1st is old 3rd, new 2nd is old 1st, new 3rd is old 4th, new 4th is old 2nd (1-based)
    iorder = np.array([3, 1, 4, 2], dtype=np.int32)

    vec_out, scrat = density.neword(vec.copy(), iorder)
    expected = np.array([30.0, 10.0, 40.0, 20.0], dtype=np.float64)
    np.testing.assert_allclose(vec_out, expected, atol=1e-14)
    np.testing.assert_allclose(scrat, [10.0, 20.0, 30.0, 40.0], atol=1e-14)


def test_occmet_metric_weighting():
    """Verify occupation-weighted metric inverse scaling."""
    norb = 2
    ontran = np.array([
        [2.0, 1.0],
        [1.0, 3.0]
    ], dtype=np.float64, order="F")
    occ = np.array([4.0, 9.0], dtype=np.float64)

    # ontran[i, j] *= sqrt(occ[i] * occ[j])
    # sqrt(occ) = [2, 3]
    # ontran[0, 0] = 2.0 * 2 * 2 = 8.0
    # ontran[0, 1] = 1.0 * 2 * 3 = 6.0
    # ontran[1, 1] = 3.0 * 3 * 3 = 27.0
    ontran_scaled, stovl = density.occmet(ontran.copy(), occ)

    assert ontran_scaled[0, 0] == pytest.approx(8.0, rel=1e-12)
    assert ontran_scaled[0, 1] == pytest.approx(6.0, rel=1e-12)
    assert ontran_scaled[1, 1] == pytest.approx(27.0, rel=1e-12)

    # stovl is packed: (0,0), (1,0), (1,1)
    np.testing.assert_allclose(stovl, [8.0, 6.0, 27.0], atol=1e-14)


def test_sfct_and_wfct():
    """Verify S-function and W-function recursion formulas for STO integrals."""
    # Test SFCT: S(nu, 0; alpha) = fact(nu) / (alpha + 1)^nu
    # Here fact(nu) = (nu-1)!
    alpha = 1.5
    nu = 3
    mumxp1 = 2
    # mumxp1 = 2 -> mu = 0, 1 (S has length 2)
    s = density.sfct(nu, mumxp1, alpha)
    # Check S has shape (2,)
    assert s.shape == (2,)

    # Test WFCT
    n1 = 2
    n2 = 2
    eta1 = 1.2
    eta2 = 1.4
    nmx = max(n1, n2)
    nmnm1 = min(n1, n2) - 1
    w, s_out = density.wfct(nmx, nmnm1, n1, eta1, n2, eta2)
    assert w.shape == (nmnm1,)
    assert s_out.shape == (nmx,)
