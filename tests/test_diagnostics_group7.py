# tests/test_diagnostics_group7.py
"""Unit tests for Group 7 routines: linalg transformations (cdct, ctsc, pcct, insert, insrtd)
and diagnostics utilities (noble, mapndx)."""
import numpy as np
import pytest
from _phatpsy_core import linalg, diagnostics


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


def test_cdct_similarity_transform():
    """Verify cdct: DD = C * D * C^T for diagonal D and rectangular/square C."""
    nbas = 4
    nvec = 3
    rng = np.random.default_rng(123)
    c = np.asfortranarray(rng.standard_normal((nbas, nvec)))
    d = np.asfortranarray(rng.standard_normal(nvec))

    # Analytical full matrix
    dd_full = c @ np.diag(d) @ c.T
    expected_packed = pack_lower(dd_full)

    dd_result = linalg.cdct(c, d)
    np.testing.assert_allclose(dd_result, expected_packed, rtol=1e-12, atol=1e-14)


def test_ctsc_similarity_transform():
    """Verify ctsc: SS = C^T * S * C for packed symmetric S and matrix C."""
    nbas = 4
    nvec = 3
    rng = np.random.default_rng(456)
    c = np.asfortranarray(rng.standard_normal((nbas, nvec)))

    # Generate random symmetric matrix S
    s_full = rng.standard_normal((nbas, nbas))
    s_full = 0.5 * (s_full + s_full.T)
    s_packed = pack_lower(s_full)

    # Analytical result SS = C^T * S * C
    ss_full = c.T @ s_full @ c
    expected_packed = pack_lower(ss_full)

    scrat, ss_result = linalg.ctsc(c, s_packed)
    np.testing.assert_allclose(ss_result, expected_packed, rtol=1e-12, atol=1e-14)


def test_pcct_projection_inverse():
    """Verify pcct: PSINV = P * C * C^T for packed P and matrix C."""
    nbas = 4
    nvec = 4
    rng = np.random.default_rng(789)
    c = np.asfortranarray(rng.standard_normal((nbas, nvec)))

    # Generate random symmetric matrix P
    p_full = rng.standard_normal((nbas, nbas))
    p_full = 0.5 * (p_full + p_full.T)
    p_packed = pack_lower(p_full)

    # Expected: P * C * C^T
    expected_psinv = p_full @ c @ c.T

    psinv = linalg.pcct(p_packed, c)
    np.testing.assert_allclose(psinv, expected_psinv, rtol=1e-12, atol=1e-14)


def test_insert_rectangular_block():
    """Verify insert: placing a submatrix into packed symmetric matrix."""
    # Suppose we have a 4x4 packed matrix (size 10)
    supmat = np.zeros(10, dtype=np.float64, order="F")
    # Submatrix 2x2 to insert at row 3, col 1 (1-based: i=3, j=1)
    # Rows 3..4, cols 1..2
    block = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], dtype=np.float64, order="F")

    # In 1-based indexing for packed lower triangle:
    # row 3: (3,1) is index (3*2)/2 + 1 = 4, (3,2) is index 5
    # row 4: (4,1) is index (4*3)/2 + 1 = 7, (4,2) is index 8
    # 0-based: indices 3, 4, 6, 7
    linalg.insert(block, supmat, 3, 1)

    expected = np.zeros(10, dtype=np.float64)
    expected[3] = 1.0  # (3, 1)
    expected[4] = 2.0  # (3, 2)
    expected[6] = 3.0  # (4, 1)
    expected[7] = 4.0  # (4, 2)

    np.testing.assert_allclose(supmat, expected, atol=1e-14)


def test_insrtd_diagonal_block():
    """Verify insrtd: placing a packed symmetric diagonal block into packed matrix."""
    supmat = np.zeros(10, dtype=np.float64, order="F")
    # 2x2 packed block: 3 elements (11, 21, 22)
    block = np.array([10.0, 20.0, 30.0], dtype=np.float64, order="F")

    # Insert starting at diagonal atom/block i=2 (1-based)
    # In insrtd: ij = (i*(i+1))/2 = (2*3)/2 = 3 (1-based, index 2 in 0-based)
    # k=1: kl=1 -> supmat(3) = block(1) = 10.0
    # k=2: kl=2,3 -> supmat(5) = block(2) = 20.0, supmat(6) = block(3) = 30.0
    linalg.insrtd(block, supmat, 2, 2)

    expected = np.zeros(10, dtype=np.float64)
    expected[2] = 10.0
    expected[4] = 20.0
    expected[5] = 30.0

    np.testing.assert_allclose(supmat, expected, atol=1e-14)


def test_noble_core_orbitals():
    """Verify noble: core orbital counts for atoms across the periodic table."""
    # H (1), He (2): 0
    assert diagnostics.noble(1) == 0
    assert diagnostics.noble(2) == 0
    # Li (3) .. Ne (10): 1 ([He] core)
    assert diagnostics.noble(3) == 1
    assert diagnostics.noble(6) == 1
    assert diagnostics.noble(10) == 1
    # Na (11) .. Ar (18): 5 ([Ne] core: 1s, 2s, 2p_x, 2p_y, 2p_z)
    assert diagnostics.noble(11) == 5
    assert diagnostics.noble(18) == 5
    # K (19) .. Kr (36): 9 ([Ar] core)
    assert diagnostics.noble(19) == 9
    assert diagnostics.noble(36) == 9
    # Rb (37) .. Xe (54): 18 ([Kr] core)
    assert diagnostics.noble(37) == 18
    assert diagnostics.noble(54) == 18
    # Cs (55) .. Rn (86): 27 ([Xe] core)
    assert diagnostics.noble(55) == 27
    assert diagnostics.noble(86) == 27
    # Fr (87)+: 43 ([Rn] core)
    assert diagnostics.noble(87) == 43


def test_mapndx_grid_coordinates():
    """Verify mapndx: cartesian projection coordinate to grid index conversion."""
    mapdim = 50
    # Center is at (50 + 1) / 2 = 25
    center = (mapdim + 1) // 2

    # At x = 0.0: xmod = 0.5 -> int(0.5) = 0 -> idx = 25
    assert diagnostics.mapndx(0.0, mapdim) == center
    # At x = 1.0: xmod = 1.5 -> int(1.5) = 1 -> idx = 26
    assert diagnostics.mapndx(1.0, mapdim) == center + 1
    # At x = -1.0: xmod = -1.5 -> int(-1.5) = -1 -> idx = 24
    assert diagnostics.mapndx(-1.0, mapdim) == center - 1
