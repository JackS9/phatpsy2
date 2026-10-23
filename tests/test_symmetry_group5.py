# tests/test_symmetry_group5.py
"""Unit tests for modernized symmetry operations and projection kernels in Group 5 (symmetry.f90)."""
import math
import numpy as np
import pytest
from _phatpsy_core import symmetry


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


def test_genrot_euler():
    """Verify Euler angle 3D rotation matrix orthogonality and determinant."""
    alpha = 0.3
    beta = 0.5
    gamma = 0.7

    rot = symmetry.genrot(alpha, beta, gamma)

    # Orthogonality: R^T * R = I
    np.testing.assert_allclose(rot.T @ rot, np.eye(3), rtol=1e-12, atol=1e-14)
    # Proper rotation: det(R) = +1
    assert np.linalg.det(rot) == pytest.approx(1.0, rel=1e-12)

    # 90-degree rotation around z-axis (alpha=pi/2, beta=0, gamma=0)
    # In PHATPSY convention, coordinates transform as xyz_new = rot^T @ xyz_old
    rot_z90 = symmetry.genrot(math.pi / 2.0, 0.0, 0.0)
    v = np.array([1.0, 0.0, 0.0])
    v_rot = rot_z90.T @ v
    np.testing.assert_allclose(v_rot, [0.0, 1.0, 0.0], atol=1e-12)


def test_sympak_packing():
    """Verify sympak packing of symmetric and non-symmetric matrices."""
    # Symmetric matrix
    a_sym = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 5.0],
        [3.0, 5.0, 6.0]
    ], dtype=np.float64, order="F")

    packed = symmetry.sympak(a_sym)
    expected = np.array([1.0, 2.0, 4.0, 3.0, 5.0, 6.0], dtype=np.float64)
    np.testing.assert_allclose(packed, expected, rtol=1e-14)

    # Non-symmetric matrix (average with transpose)
    a_asym = np.array([
        [1.0, 1.0, 2.0],
        [3.0, 4.0, 4.0],
        [4.0, 6.0, 6.0]
    ], dtype=np.float64, order="F")
    packed_asym = symmetry.sympak(a_asym)
    expected_asym = np.array([1.0, 2.0, 4.0, 3.0, 5.0, 6.0], dtype=np.float64)
    np.testing.assert_allclose(packed_asym, expected_asym, rtol=1e-14)


def test_sympro_projection_algebra():
    """Verify sympro modifications of symmetric matrix with projection matrix."""
    n = 3
    rng = np.random.default_rng(77)

    m = rng.standard_normal((n, n))
    a_full = (m + m.T) / 2.0
    a_packed = pack_lower(a_full)

    b = rng.standard_normal((n, n))

    # Test icode = 1: A + 1/2(AB + B^T A)
    a_mod, scrat = symmetry.sympro(a_packed.copy(), b, 1)
    expected_full = a_full + 0.5 * (a_full @ b + b.T @ a_full)
    np.testing.assert_allclose(unpack_lower(a_mod, n), expected_full, rtol=1e-12, atol=1e-14)

    # Test icode = -1: A - 1/2(AB + B^T A)
    a_mod_neg, _ = symmetry.sympro(a_packed.copy(), b, -1)
    expected_full_neg = a_full - 0.5 * (a_full @ b + b.T @ a_full)
    np.testing.assert_allclose(unpack_lower(a_mod_neg, n), expected_full_neg, rtol=1e-12, atol=1e-14)

    # Test icode = 2: A + 1/2(BA + AB^T)
    a_mod_2, _ = symmetry.sympro(a_packed.copy(), b, 2)
    expected_full_2 = a_full + 0.5 * (b @ a_full + a_full @ b.T)
    np.testing.assert_allclose(unpack_lower(a_mod_2, n), expected_full_2, rtol=1e-12, atol=1e-14)

    # Test icode = 3: A + lower_tri(AB)
    a_mod_3, _ = symmetry.sympro(a_packed.copy(), b, 3)
    ab = a_full @ b
    expected_full_3 = a_full.copy()
    for i in range(n):
        for j in range(i + 1):
            expected_full_3[i, j] += ab[i, j]
            expected_full_3[j, i] = expected_full_3[i, j]
    np.testing.assert_allclose(unpack_lower(a_mod_3, n), expected_full_3, rtol=1e-12, atol=1e-14)


def test_sympr1_onsite_fock():
    """Verify on-site pseudopotential Fock contribution: F = P*R*S + S*R*P - P*R*P."""
    n = 3
    rng = np.random.default_rng(88)

    m_p = rng.standard_normal((n, n))
    m_r = rng.standard_normal((n, n))
    m_s = rng.standard_normal((n, n))

    p_full = (m_p + m_p.T) / 2.0
    r_full = (m_r + m_r.T) / 2.0
    s_full = (m_s + m_s.T) / 2.0

    p_packed = pack_lower(p_full)
    r_packed = pack_lower(r_full)
    s_packed = pack_lower(s_full)

    f_packed, scrat = symmetry.sympr1(p_packed, r_packed, s_packed, n)

    expected_full = p_full @ r_full @ s_full + s_full @ r_full @ p_full - p_full @ r_full @ p_full
    expected_packed = pack_lower(expected_full)

    np.testing.assert_allclose(f_packed, expected_packed, rtol=1e-12, atol=1e-14)


def test_sympr2_offsite_pseudopotential():
    """Verify off-site pseudopotential generation: R' = R*S*P + P*S*R - P*S*R*S*P."""
    n = 3
    rng = np.random.default_rng(99)

    m_p = rng.standard_normal((n, n))
    m_r = rng.standard_normal((n, n))
    m_s = rng.standard_normal((n, n))

    p_full = (m_p + m_p.T) / 2.0
    r_full = (m_r + m_r.T) / 2.0
    s_full = (m_s + m_s.T) / 2.0

    p_packed = pack_lower(p_full)
    r_packed = pack_lower(r_full)
    s_packed = pack_lower(s_full)

    r_out, scrat1, scrat2 = symmetry.sympr2(p_packed, r_packed.copy(), s_packed, n)

    rsp = r_full @ s_full @ p_full
    psr = p_full @ s_full @ r_full
    psrsp = p_full @ s_full @ r_full @ s_full @ p_full
    expected_full = rsp + psr - psrsp
    expected_packed = pack_lower(expected_full)

    np.testing.assert_allclose(r_out, expected_packed, rtol=1e-12, atol=1e-14)
