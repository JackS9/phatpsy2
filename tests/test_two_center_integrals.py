"""
Tests for the exploratory two-center STO integral engine (phatpsy/two_center_integrals.py).

All reference values are independent closed forms or the Fortran core kernels; nothing
here is fitted.
  - Sugiura (1927) / Coulson closed forms for 1s(zeta=1) STOs on two centers:
    S, T_ab, V, Coulomb J=(aa|bb), exchange K=(ab|ab), hybrid (aa|ab), and (aa|aa)=5/8.
  - Neumann expansion of 1/r12 versus direct evaluation.
  - Real-harmonic sign convention versus Fortran gensab/rotsab for an off-axis
    O(DZP)--H(DZP) pair (p, d functions with m < 0 included).
"""
import math

import numpy as np
import pytest
from scipy.special import exp1

from phatpsy.two_center_integrals import (
    ProlateGrid,
    STO,
    eri_tensor,
    one_electron_matrices,
)

EULER_GAMMA = 0.5772156649015329


def _sugiura(R):
    S = math.exp(-R) * (1 + R + R * R / 3)
    Sp = math.exp(R) * (1 - R + R * R / 3)
    Ei = lambda x: -exp1(-x)
    J = 1 / R - math.exp(-2 * R) * (1 / R + 11 / 8 + 3 * R / 4 + R * R / 6)
    hyb = math.exp(-R) * (R + 1 / 8 + 5 / (16 * R)) + math.exp(-3 * R) * (-1 / 8 - 5 / (16 * R))
    K = 0.2 * (
        -math.exp(-2 * R) * (-25 / 8 + 23 * R / 4 + 3 * R * R + R ** 3 / 3)
        + 6 / R * (S * S * (EULER_GAMMA + math.log(R)) + Sp * Sp * Ei(-4 * R) - 2 * S * Sp * Ei(-2 * R))
    )
    Vaa = -1.0 - (1 / R - math.exp(-2 * R) * (1 + 1 / R))
    Vab = -2 * math.exp(-R) * (1 + R)
    Tab = -0.5 * S + math.exp(-R) * (1 + R)
    return dict(S=S, J=J, K=K, hyb=hyb, Vaa=Vaa, Vab=Vab, Tab=Tab)


@pytest.mark.parametrize("R", [0.8, 1.4, 2.5, 4.0])
def test_sugiura_closed_forms(R):
    A, B = (0.0, 0.0, 0.0), (0.0, 0.0, R)
    g = ProlateGrid(A, B, zeta_min=1.0)
    S, T, V, chi = one_electron_matrices(g, [STO(A, 1, 0, 0, 1.0), STO(B, 1, 0, 0, 1.0)], 1.0, 1.0)
    ref = _sugiura(R)
    assert S[0, 0] == pytest.approx(1.0, abs=1e-12)
    assert S[0, 1] == pytest.approx(ref["S"], abs=1e-12)
    assert T[0, 0] == pytest.approx(0.5, abs=1e-12)
    assert T[0, 1] == pytest.approx(ref["Tab"], abs=1e-12)
    assert V[0, 0] == pytest.approx(ref["Vaa"], abs=1e-12)
    assert V[0, 1] == pytest.approx(ref["Vab"], abs=1e-12)
    eri = eri_tensor(g, chi, lmax=30)
    assert eri[0, 0, 0, 0] == pytest.approx(0.625, abs=1e-11)
    assert eri[0, 0, 1, 1] == pytest.approx(ref["J"], abs=1e-11)
    assert eri[0, 0, 0, 1] == pytest.approx(ref["hyb"], abs=1e-11)
    assert eri[0, 1, 0, 1] == pytest.approx(ref["K"], abs=1e-11)


def test_neumann_expansion_matches_direct_inverse_distance():
    g = ProlateGrid((0.0, 0.0, 0.0), (0.0, 0.0, 1.4))
    assert g.neumann_selftest(lmax=60) < 1e-5


def test_coulomb_stable_at_high_angular_truncation():
    R = 1.4
    A, B = (0.0, 0.0, 0.0), (0.0, 0.0, R)
    g = ProlateGrid(A, B, zeta_min=1.0)
    a, b = g.evaluate(STO(A, 1, 0, 0, 1.0)), g.evaluate(STO(B, 1, 0, 0, 1.0))
    ref = _sugiura(R)
    for lmax in (20, 50, 70):
        assert g.coulomb(a * b, a * b, lmax=lmax) == pytest.approx(ref["K"], abs=1e-11)


@pytest.mark.parametrize(
    "sym_b,xb",
    [("H", [0.3, 1.43, 1.11]), ("O", [-0.9, 0.4, 1.3]), ("O", [1.1, -0.7, -0.8])],
)
def test_real_harmonic_signs_match_fortran_overlap_off_axis(sym_b, xb):
    """Generic off-axis O(DZP)-X(DZP) overlap block agrees with the Fortran gensab/rotsab
    kernels for every s, p, d function (fixes the Condon-Shortley phase convention)."""
    import _phatpsy_core
    from phatpsy.basis import get_element_basis
    from phatpsy.gradients import DiatomicFrame, compute_diatomic_overlap_matrix_and_gradient

    # Fortran angular/factorial tables are populated by the block-data defaults.
    _phatpsy_core.state.init_block_data_and_defaults()

    xa = np.array([0.0, 0.0, 0.0])
    xb = np.array(xb, dtype=float)
    shells, basis = {}, []
    for tag, sym, xyz in (("A", "O", xa), ("B", sym_b, xb)):
        ed = get_element_basis(sym, "DZP")
        sh = [(f.n, f.l, f.zeta) for f in ed.functions]
        shells[tag] = sh
        for n, l, z in sh:
            for m in range(-l, l + 1):
                basis.append(STO(tuple(xyz), n, l, m, z))
    norm = lambda n, z: (2 * z) ** (n + 0.5) / math.sqrt(math.factorial(2 * n))
    arrs = {}
    for tag in ("A", "B"):
        sh = shells[tag]
        arrs[tag] = (
            np.array([s[0] for s in sh], dtype=np.int32),
            np.array([s[1] for s in sh], dtype=np.int32),
            np.array([s[2] for s in sh]),
            np.array([norm(s[0], s[2]) for s in sh]),
        )
    s_f, _, _ = compute_diatomic_overlap_matrix_and_gradient(*arrs["A"], *arrs["B"], DiatomicFrame(xa, xb))
    g = ProlateGrid(xa, xb, zeta_min=0.5)
    S, _, _, _ = one_electron_matrices(g, basis, 8.0, 1.0)
    nA = s_f.shape[0]
    assert np.max(np.abs(S[:nA, nA:] - s_f)) < 1e-9
