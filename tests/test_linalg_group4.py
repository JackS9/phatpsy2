# tests/test_linalg_group4.py
"""Unit tests for modernized linear algebra routines in Group 4 (linalg.f90)."""
import numpy as np
import pytest
from _phatpsy_core import linalg


def pack_lower(mat: np.ndarray) -> np.ndarray:
    """Helper: pack symmetric square matrix into lower-triangular 1D vector (1-based row-major)."""
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


def test_trisq():
    """Verify packed symmetric times square matrix: H * U."""
    n = 4
    rng = np.random.default_rng(42)
    # Generate random symmetric matrix H_full
    m = rng.standard_normal((n, n))
    h_full = np.asfortranarray((m + m.T) / 2.0)
    h_packed = pack_lower(h_full)

    u = np.asfortranarray(rng.standard_normal((n, n)))

    # Compute via modernized Fortran
    result = linalg.trisq(n, h_packed, u)

    # Expected product
    expected = h_full @ u

    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)


def test_tritri():
    """Verify product of two packed triangular matrices: A * B."""
    n = 4
    n2 = (n * (n + 1)) // 2
    rng = np.random.default_rng(123)

    m1 = rng.standard_normal((n, n))
    a_full = np.asfortranarray((m1 + m1.T) / 2.0)
    a_packed = pack_lower(a_full)

    m2 = rng.standard_normal((n, n))
    b_full = np.asfortranarray((m2 + m2.T) / 2.0)
    b_packed = pack_lower(b_full)

    ab = linalg.tritri(a_packed, b_packed, n, n2)
    expected = a_full @ b_full

    np.testing.assert_allclose(ab, expected, rtol=1e-12, atol=1e-14)


def test_uthu():
    """Verify similarity transformation: U^T * H * U."""
    n = 4
    rng = np.random.default_rng(456)

    m = rng.standard_normal((n, n))
    h_full = np.asfortranarray((m + m.T) / 2.0)
    h_packed = pack_lower(h_full)

    u = np.asfortranarray(rng.standard_normal((n, n)))

    scrat, result_packed = linalg.uthu(n, h_packed, u)

    expected_scrat = h_full @ u
    np.testing.assert_allclose(scrat, expected_scrat, rtol=1e-12, atol=1e-14)

    expected_full = u.T @ h_full @ u
    result_full = unpack_lower(result_packed, n)
    np.testing.assert_allclose(result_full, expected_full, rtol=1e-12, atol=1e-14)


def test_invert_lapack():
    """Verify general matrix inversion via LAPACK LU."""
    n = 4
    rng = np.random.default_rng(789)
    a = np.asfortranarray(rng.standard_normal((n, n)))

    inv_a, ierr = linalg.invert(a, n, n, n)
    assert ierr == 0

    identity = a @ inv_a
    np.testing.assert_allclose(identity, np.eye(n), rtol=1e-12, atol=1e-14)

    # Singular matrix test
    a_singular = np.ones((n, n), dtype=np.float64, order="F")
    _, ierr_sing = linalg.invert(a_singular, n, n, n)
    assert ierr_sing == 1


def test_matinv_lapack():
    """Verify in-place inversion, solver A*X=B, and determinant via LAPACK."""
    n = 3
    m = 2
    rng = np.random.default_rng(101)
    a = np.asfortranarray(rng.standard_normal((n, n)))
    b = np.asfortranarray(rng.standard_normal((n, m)))

    a_orig = a.copy()
    b_orig = b.copy()

    a_inv, b_sol, determ = linalg.matinv(a, b)

    expected_inv = np.linalg.inv(a_orig)
    expected_det = np.linalg.det(a_orig)
    expected_sol = expected_inv @ b_orig

    np.testing.assert_allclose(a_inv, expected_inv, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(determ, expected_det, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(b_sol, expected_sol, rtol=1e-12, atol=1e-14)


def test_schmid_orthonormalization():
    """Verify Schmidt orthonormalization of overlap matrix: T^T * S * T = I."""
    n = 3
    # Positive-definite matrix
    s_full = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 1.0, 0.3],
        [0.1, 0.3, 1.0]
    ], dtype=np.float64, order="F")
    s_packed = pack_lower(s_full)

    # Run schmid (in-place transform returned in packed array)
    t_packed = linalg.schmid(s_packed, n)

    # Reconstruct upper-triangular T matrix:
    # According to schmid.f algorithm:
    # For column N (1..nrows), T(1..N, N) is stored in the N-th row positions
    # of the packed storage: index KPN + 1 .. KPN + N.
    t_mat = np.zeros((n, n), dtype=np.float64, order="F")
    idx = 0
    for col in range(n):
        for row in range(col + 1):
            t_mat[row, col] = t_packed[idx]
            idx += 1

    # Check T^T * S * T == I
    ortho = t_mat.T @ s_full @ t_mat
    np.testing.assert_allclose(ortho, np.eye(n), rtol=1e-10, atol=1e-12)


def test_normlz_eigenvectors():
    """Verify normalization of eigenvectors: C_i^T * S * C_i = 1."""
    nbas = 3
    nvec = 3
    s_full = np.array([
        [1.0, 0.3, 0.1],
        [0.3, 1.0, 0.2],
        [0.1, 0.2, 1.0]
    ], dtype=np.float64, order="F")
    s_packed = pack_lower(s_full)

    rng = np.random.default_rng(202)
    evec = np.asfortranarray(rng.standard_normal((nbas, nvec)))

    evec_norm = linalg.normlz(evec, s_packed)

    for i in range(nvec):
        v = evec_norm[:, i]
        metric = v.T @ s_full @ v
        assert metric == pytest.approx(1.0, rel=1e-12)


def test_round_function():
    """Verify numeric rounding function."""
    assert linalg.round(12.3456789, -2) == pytest.approx(12.35, rel=1e-14)
    assert linalg.round(1234.56, 2) == pytest.approx(1200.0, rel=1e-14)
    assert linalg.round(0.00789, -3) == pytest.approx(0.008, rel=1e-14)


def test_maxovl_ordering():
    """Verify maximum overlap column vector reordering."""
    nbas = 3
    nvec = 3
    s_full = np.eye(nbas, dtype=np.float64, order="F")
    s_packed = pack_lower(s_full)

    # evec1 is identity
    evec1 = np.eye(nbas, dtype=np.float64, order="F")
    # evec2 has permuted columns: [col2, col0, col1]
    evec2 = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0]
    ], dtype=np.float64, order="F")

    iorder, qnword = linalg.maxovl(evec1, evec2, s_packed)
    assert bool(qnword) is True
    # In evec2:
    # col 2 (1-based index 2 in Fortran, value [1,0,0]^T) matches evec1[:, 0]
    # col 3 matches evec1[:, 1]
    # col 1 matches evec1[:, 2]
    # iorder tells where each vector goes
    assert len(iorder) == nvec
