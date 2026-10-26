# tests/test_exchange_dft.py
"""
Unit and integration tests for Modern Density-Functional Exchange Kernels (Phase 3.3).
Tests:
- Slater-Dirac Local Density Exchange (LDA)
- Becke 88 (B88) Generalized Gradient Approximation
- Perdew-Burke-Ernzerhof (PBE) Generalized Gradient Approximation
- Analytical STO exchange integration on hydrogen 1s
- Multi-exponential least squares fitting
- Deck execution with exchange_kernel="slater" and exchange_kernel="pbe"
"""

import math
import numpy as np
import pytest

from phatpsy.exchange_dft import (
    eval_slater_exchange,
    eval_becke88_exchange,
    eval_pbe_exchange,
    eval_exchange_potential,
    fit_exchange_potential,
    CX_LDA,
    BETA_B88,
    KAPPA_PBE,
    MU_PBE
)
from phatpsy.schema import CalculationDeck, ControlConfig, AtomConfig
from phatpsy.driver import PhatpsyDriver


def test_slater_exchange_pointwise():
    """Verify Slater-Dirac LDA potential against analytical formulas."""
    # Zero density floor
    vx_0, eps_0 = eval_slater_exchange(np.array([0.0]))
    assert vx_0[0] == 0.0
    assert eps_0[0] == 0.0

    # Test uniform electron gas densities
    rhos = np.array([0.001, 0.05, 0.2, 1.0, 5.0])
    
    # Kohn-Sham Dirac alpha = 2/3
    vx_ks, eps_ks = eval_slater_exchange(rhos, alpha=2.0 / 3.0)
    expected_vx_ks = - (3.0 / np.pi * rhos) ** (1.0 / 3.0)
    expected_eps_ks = - CX_LDA * (rhos ** (1.0 / 3.0))
    np.testing.assert_allclose(vx_ks, expected_vx_ks, rtol=1e-12)
    np.testing.assert_allclose(eps_ks, expected_eps_ks, rtol=1e-12)

    # Slater X-alpha with alpha = 1.0
    vx_slater, _ = eval_slater_exchange(rhos, alpha=1.0)
    expected_vx_slater = - 1.5 * (3.0 / np.pi * rhos) ** (1.0 / 3.0)
    np.testing.assert_allclose(vx_slater, expected_vx_slater, rtol=1e-12)


def test_becke88_exchange():
    """Verify Becke 88 GGA exchange enhancement factor and asymptotic behavior."""
    rho = np.array([0.1, 0.1, 0.1])
    # When grad_rho = 0, F_x should be identically 1.0
    grad_0 = np.array([0.0, 0.0, 0.0])
    vx, eps_x, fx = eval_becke88_exchange(rho, grad_0)
    np.testing.assert_allclose(fx, [1.0, 1.0, 1.0], atol=1e-15)

    # For non-zero gradient, F_x must be strictly > 1.0
    grad_nonzero = np.array([0.01, 0.05, 0.2])
    vx_gga, eps_gga, fx_gga = eval_becke88_exchange(rho, grad_nonzero)
    assert np.all(fx_gga > 1.0)
    # Check enhancement increases monotonically with gradient
    assert fx_gga[0] < fx_gga[1] < fx_gga[2]


def test_pbe_exchange():
    """Verify PBE GGA exchange enhancement and Lieb-Oxford bound limit."""
    rho = np.array([0.2, 0.2, 0.2])
    # When grad_rho = 0, F_x = 1.0
    vx_0, eps_0, fx_0 = eval_pbe_exchange(rho, np.zeros(3))
    np.testing.assert_allclose(fx_0, [1.0, 1.0, 1.0], atol=1e-15)

    # Test large gradient limit: F_x -> 1 + kappa = 1.804
    huge_grad = np.array([1e6])
    _, _, fx_inf = eval_pbe_exchange(np.array([0.01]), huge_grad)
    np.testing.assert_allclose(fx_inf[0], 1.0 + KAPPA_PBE, rtol=1e-4)


def test_fortran_python_pointwise_parity():
    """Verify that Fortran exchange_dft module produces bitwise identical pointwise potentials."""
    try:
        from _phatpsy_core import exchange_dft as f_exch
    except ImportError:
        pytest.skip("_phatpsy_core not available")

    rhos = [0.001, 0.02, 0.15, 0.8, 3.5]
    d_rhos = [0.0005, 0.01, 0.08, 0.5, 2.0]

    for rho, drho in zip(rhos, d_rhos):
        # 1: Slater
        vx_py = eval_exchange_potential(np.array([rho]), np.array([drho]), kernel="slater", alpha=2.0 / 3.0)[0]
        vx_f, fx_f = f_exch.eval_exchange_potential_point(rho, drho, 1, 2.0 / 3.0)
        np.testing.assert_allclose(vx_f, vx_py, rtol=1e-12)
        assert fx_f == 1.0

        # 2: Becke88
        vx_py_b88 = eval_exchange_potential(np.array([rho]), np.array([drho]), kernel="becke88", alpha=2.0 / 3.0)[0]
        vx_f_b88, fx_f_b88 = f_exch.eval_exchange_potential_point(rho, drho, 2, 2.0 / 3.0)
        np.testing.assert_allclose(vx_f_b88, vx_py_b88, rtol=1e-12)

        # 3: PBE
        vx_py_pbe = eval_exchange_potential(np.array([rho]), np.array([drho]), kernel="pbe", alpha=2.0 / 3.0)[0]
        vx_f_pbe, fx_f_pbe = f_exch.eval_exchange_potential_point(rho, drho, 3, 2.0 / 3.0)
        np.testing.assert_allclose(vx_f_pbe, vx_py_pbe, rtol=1e-12)


def test_hydrogen_1s_analytical_exchange():
    """
    Analytical benchmark:
    For H 1s orbital with zeta=1.0: R(r) = 2 exp(-r).
    rho(r) = R(r)^2 / (4*pi) = (1/pi) exp(-2r).
    Slater-Dirac potential (alpha=2/3): v_x(r) = - (3/pi * rho)^(1/3) = - (3/pi^2)^(1/3) exp(-2r/3).
    Integral int_0^inf R(r)^2 v_x(r) r^2 dr
      = - 4 (3/pi^2)^(1/3) int_0^inf r^2 exp(-8r/3) dr
      = - 4 (3/pi^2)^(1/3) * (2 / (8/3)^3)
      = - 0.283655337424... Ha.
    """
    # Use mapped Gauss-Legendre quadrature
    n_points = 64
    t, w = np.polynomial.legendre.leggauss(n_points)
    r0 = 1.0
    r = r0 * (1.0 + t) / (1.0 - t)
    dr_dt = 2.0 * r0 / ((1.0 - t) ** 2)
    weights = w * dr_dt

    R = 2.0 * np.exp(-r)
    rho = (1.0 / (4.0 * np.pi)) * (R ** 2)
    vx = eval_exchange_potential(rho, kernel="slater", alpha=2.0 / 3.0)

    num_integral = np.sum(weights * (R ** 2) * vx * (r ** 2))
    analytical = - 4.0 * ((3.0 / (np.pi ** 2)) ** (1.0 / 3.0)) * (2.0 / ((8.0 / 3.0) ** 3))

    np.testing.assert_allclose(num_integral, analytical, rtol=1e-11)


def test_fit_exchange_exponential():
    """Verify multi-exponential least-squares fitting of radial exchange potential."""
    r = np.linspace(0.01, 10.0, 200)
    vexp = np.array([0.5, 1.5, 3.0])
    true_coefs = np.array([-0.3, -0.6, -0.1])
    # Synthesize test potential from known basis
    vx = true_coefs[0] * np.exp(-vexp[0] * r) + \
         true_coefs[1] * np.exp(-vexp[1] * r) + \
         true_coefs[2] * np.exp(-vexp[2] * r)

    fit_coefs, rms = fit_exchange_potential(r, vx, vexp)
    np.testing.assert_allclose(fit_coefs, true_coefs, atol=1e-12)
    assert rms < 1e-12


def test_exchange_dft_deck_run(tmp_path):
    """
    Integration deck test:
    Verify that a calculation deck with exchange_kernel='slater' and 'pbe'
    completes SCF iterations cleanly and relaxations produce physical energies.
    """
    deck = CalculationDeck(
        title="N2 Molecule with Slater-Dirac Exchange Kernel",
        control=ControlConfig(
            max_cycles=3,
            convergence=1e-3,
            pk_scale=1.0,
            use_pseudometric=True,
            exchange_kernel="slater",
            exchange_alpha=2.0 / 3.0
        ),
        geometry=[
            AtomConfig(element="N", id="N1", xyz=[0.0, 0.0, -1.037]),
            AtomConfig(element="N", id="N2", xyz=[0.0, 0.0,  1.037]),
        ]
    )

    deck_text = deck.to_stdin_text()
    assert "QDFTEX=T" in deck_text
    assert "EXCH_TYPE=1" in deck_text
    assert "EXCH_ALPHA=0.666667" in deck_text

    driver = PhatpsyDriver(quiet=True)
    result = driver.run_deck(deck)
    assert result.converged
    assert result.total_energy is not None
    # Verify reasonable physical total energy for N2
    assert -115.0 < result.total_energy < -100.0


def test_pbe_deck_run(tmp_path):
    """Verify that a calculation deck with exchange_kernel='pbe' runs successfully."""
    deck = CalculationDeck(
        title="N2 Molecule with PBE Exchange Kernel",
        control=ControlConfig(
            max_cycles=3,
            convergence=1e-3,
            pk_scale=1.0,
            use_pseudometric=True,
            exchange_kernel="pbe"
        ),
        geometry=[
            AtomConfig(element="N", id="N1", xyz=[0.0, 0.0, -1.037]),
            AtomConfig(element="N", id="N2", xyz=[0.0, 0.0,  1.037]),
        ]
    )

    deck_text = deck.to_stdin_text()
    assert "QDFTEX=T" in deck_text
    assert "EXCH_TYPE=3" in deck_text

    driver = PhatpsyDriver(quiet=True)
    result = driver.run_deck(deck)
    assert result.converged
    assert result.total_energy is not None
    assert -115.0 < result.total_energy < -100.0
