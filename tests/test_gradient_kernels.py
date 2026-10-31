"""
Unit tests for Phase 11.3: Closed-Form Analytical Two-Center Integral Radial Recurrences (twocenter.f90).
Verifies exact analytical derivative recurrences for d_gensab (overlap radial derivative) and
d_genvab (model potential radial derivative) against independent 2-point central difference baselines.

NOTE (Anti-Circularity Invariant): d_gensab and d_genvab in twocenter.f90 are true closed-form analytical
recurrences evaluated via auxiliary function parameter derivatives (dA_n/dalpha = -A_{n+1},
dB_m/dbeta = -B_{m+1}) and geometric prefactor power rules. These tests verify analytical derivatives
against independent finite differences computed from gensab and genvab.
"""
import math
import numpy as np
import pytest

import _phatpsy_core


def compute_slater_norm(n: int, eta: float) -> float:
    """Slater orbital normalization constant (2*zeta)**(n + 0.5) / sqrt((2n)!)."""
    return (2.0 * eta) ** (n + 0.5) / math.sqrt(math.factorial(2 * n))


class TestTwoCenterIntegralDerivatives:
    """Non-circular verification for Phase 11.3 analytical radial derivative recurrences."""


    @pytest.mark.parametrize("rab", [1.0, 1.5, 2.068, 3.0, 4.5])
    def test_d_gensab_vs_finite_difference_homonuclear(self, rab):
        """
        Verify d_gensab analytical recurrence matches 2-point central difference of gensab
        for multi-shell s and p STO orbitals (N2 valence-like basis).
        """
        # 1s, 2s, 2p basis
        na = np.array([1, 2, 2], dtype=np.int32)
        la = np.array([0, 0, 1], dtype=np.int32)
        etaa = np.array([6.665, 1.956, 1.956], dtype=np.float64)
        anorm = np.array([compute_slater_norm(n, e) for n, e in zip(na, etaa)], dtype=np.float64)

        nb = np.array([1, 2, 2], dtype=np.int32)
        lb = np.array([0, 0, 1], dtype=np.int32)
        etab = np.array([6.665, 1.956, 1.956], dtype=np.float64)
        bnorm = np.array([compute_slater_norm(n, e) for n, e in zip(nb, etab)], dtype=np.float64)

        llmxp1 = 5  # LMAX = 2 -> 2*2 + 1 = 5

        # Analytical closed-form derivative kernel (Phase 11.3)
        d_ovrlap = _phatpsy_core.twocenter.compute_d_gensab(
            llmxp1, na, la, etaa, anorm, nb, lb, etab, bnorm, rab
        )

        # Central finite difference of gensab
        h = 1e-5
        s_plus = _phatpsy_core.twocenter.compute_gensab(
            llmxp1, na, la, etaa, anorm, nb, lb, etab, bnorm, rab + h
        )
        s_minus = _phatpsy_core.twocenter.compute_gensab(
            llmxp1, na, la, etaa, anorm, nb, lb, etab, bnorm, rab - h
        )
        d_ovrlap_fd = (s_plus - s_minus) / (2.0 * h)

        assert np.max(np.abs(d_ovrlap)) > 1e-4, f"Trivial zero kernel output at R={rab}"
        diff = np.max(np.abs(d_ovrlap - d_ovrlap_fd))
        assert diff < 1e-6, f"d_gensab mismatch at R={rab}: max diff={diff:.3e}"

    @pytest.mark.parametrize("rab", [1.2, 2.132, 3.5])
    def test_d_gensab_vs_finite_difference_heteronuclear(self, rab):
        """
        Verify d_gensab for heteronuclear diatomic (C-O type exponents).
        """
        # Atom A (Carbon): 1s, 2s, 2p
        na = np.array([1, 2, 2], dtype=np.int32)
        la = np.array([0, 0, 1], dtype=np.int32)
        etaa = np.array([5.672, 1.608, 1.567], dtype=np.float64)
        anorm = np.array([compute_slater_norm(n, e) for n, e in zip(na, etaa)], dtype=np.float64)

        # Atom B (Oxygen): 1s, 2s, 2p
        nb = np.array([1, 2, 2], dtype=np.int32)
        lb = np.array([0, 0, 1], dtype=np.int32)
        etab = np.array([7.657, 2.245, 2.226], dtype=np.float64)
        bnorm = np.array([compute_slater_norm(n, e) for n, e in zip(nb, etab)], dtype=np.float64)

        llmxp1 = 5

        d_ovrlap = _phatpsy_core.twocenter.compute_d_gensab(
            llmxp1, na, la, etaa, anorm, nb, lb, etab, bnorm, rab
        )

        h = 1e-5
        s_plus = _phatpsy_core.twocenter.compute_gensab(
            llmxp1, na, la, etaa, anorm, nb, lb, etab, bnorm, rab + h
        )
        s_minus = _phatpsy_core.twocenter.compute_gensab(
            llmxp1, na, la, etaa, anorm, nb, lb, etab, bnorm, rab - h
        )
        d_ovrlap_fd = (s_plus - s_minus) / (2.0 * h)

        assert np.max(np.abs(d_ovrlap)) > 1e-4, f"Trivial zero kernel output at R={rab}"
        diff = np.max(np.abs(d_ovrlap - d_ovrlap_fd))
        assert diff < 1e-6, f"Heteronuclear d_gensab mismatch at R={rab}: max diff={diff:.3e}"

    @pytest.mark.parametrize("rab", [1.5, 2.2, 3.8])
    def test_d_genvab_vs_finite_difference(self, rab):
        """
        Verify d_genvab matches numerical finite differences of genvab
        for multi-term model potential matrix elements.
        """
        n = np.array([1, 2, 2], dtype=np.int32)
        l = np.array([0, 0, 1], dtype=np.int32)
        eta = np.array([5.0, 1.8, 1.8], dtype=np.float64)
        anorm = np.array([compute_slater_norm(ni, ei) for ni, ei in zip(n, eta)], dtype=np.float64)

        # Two model potential expansion terms: s-type and p-type
        lv = np.array([0, 1], dtype=np.int32)
        mv = np.array([0, 0], dtype=np.int32)
        vexp = np.array([2.5, 1.8], dtype=np.float64)

        msto = 1 + 1 + 3  # 1 (s) + 1 (s) + 3 (p) = 5
        m2sto = (msto * (msto + 1)) // 2  # 15

        d_stvab = _phatpsy_core.twocenter.compute_d_genvab(
            m2sto, n, l, eta, rab, lv, mv, vexp, anorm, True
        )

        h = 1e-5
        v_plus = _phatpsy_core.twocenter.compute_genvab(
            m2sto, n, l, eta, rab + h, lv, mv, vexp, anorm, True
        )
        v_minus = _phatpsy_core.twocenter.compute_genvab(
            m2sto, n, l, eta, rab - h, lv, mv, vexp, anorm, True
        )
        d_stvab_fd = (v_plus - v_minus) / (2.0 * h)

        assert np.max(np.abs(d_stvab)) > 1e-4, f"Trivial zero kernel output at R={rab}"
        diff = np.max(np.abs(d_stvab - d_stvab_fd))
        assert diff < 1e-6, f"d_genvab mismatch at R={rab}: max diff={diff:.3e}"
