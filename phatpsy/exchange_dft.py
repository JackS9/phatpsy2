# phatpsy/exchange_dft.py
"""
Modern Density-Functional Exchange Kernels for PHATPSY 2.0.

Implements local and gradient-corrected exchange functionals:
- Slater-Dirac Local Density Exchange (LDA, alpha=2/3 for Kohn-Sham Dirac, alpha=1.0 for Slater X-alpha)
- Becke 88 (B88) Generalized Gradient Approximation (GGA)
- Perdew-Burke-Ernzerhof (PBE) Generalized Gradient Approximation (GGA)

Provides:
- Pointwise and volumetric potential and energy density evaluations
- Multi-exponential least-squares fitting against screening bases
- Direct integration with PHATPSY atomic basis evaluations
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# Physical constants and functional parameters in atomic units
PI = math.pi
RHO_FLOOR = 1.0e-15
BETA_B88 = 0.0042
KAPPA_PBE = 0.804
MU_PBE = 0.2195149727645171
CX_LDA = 0.75 * (3.0 / PI) ** (1.0 / 3.0)

# Perdew-Wang 92 (PW92) LDA Correlation parameters
A_PW92 = 0.031090690869654895
ALPHA1_PW92 = 0.21370
BETA1_PW92 = 7.5957
BETA2_PW92 = 3.5876
BETA3_PW92 = 1.6382
BETA4_PW92 = 0.49294

# Perdew-Burke-Ernzerhof (PBE) GGA Correlation parameters
BETA_PBE_C = 0.066725
GAMMA_PBE_C = 0.031090690869654895


def eval_slater_exchange(
    rho: np.ndarray,
    alpha: float = 2.0 / 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate Slater-Dirac local density exchange potential and energy density.

    Parameters:
        rho: Electron density array (Bohr^-3)
        alpha: Slater alpha parameter (default 2/3 for Kohn-Sham Dirac, 1.0 for Slater)

    Returns:
        (vx, eps_x):
            vx: Local exchange potential (Hartree)
            eps_x: Exchange energy density per electron (Hartree)
    """
    rho_arr = np.asarray(rho, dtype=np.float64)
    rho_eff = np.maximum(rho_arr, RHO_FLOOR)
    mask = rho_arr > RHO_FLOOR

    # Energy density per electron: eps_x = -C_x * rho^(1/3)
    eps_x = np.where(mask, -CX_LDA * (rho_eff ** (1.0 / 3.0)), 0.0)

    # Local potential: v_x = -alpha * (3/2) * (3/pi * rho)^(1/3)
    # When alpha = 2/3, v_x = - (3/pi * rho)^(1/3) = (4/3) * eps_x
    vx = np.where(mask, -alpha * 1.5 * ((3.0 / PI * rho_eff) ** (1.0 / 3.0)), 0.0)

    return vx, eps_x


def eval_becke88_exchange(
    rho: np.ndarray,
    grad_rho: np.ndarray,
    alpha: float = 2.0 / 3.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate Becke 88 (B88) GGA exchange potential, energy density, and enhancement factor.

    Parameters:
        rho: Electron density array (Bohr^-3)
        grad_rho: Gradient magnitude |grad rho| array (Bohr^-4)
        alpha: Underlying LDA alpha parameter (default 2/3)

    Returns:
        (vx, eps_x, fx):
            vx: Local exchange potential (Hartree)
            eps_x: Exchange energy density per electron (Hartree)
            fx: Enhancement factor F_x >= 1.0
    """
    rho_arr = np.asarray(rho, dtype=np.float64)
    grad_arr = np.asarray(grad_rho, dtype=np.float64)
    rho_eff = np.maximum(rho_arr, RHO_FLOOR)
    mask = rho_arr > RHO_FLOOR

    vx_lda, eps_lda = eval_slater_exchange(rho_arr, alpha=alpha)

    # Reduced density gradient x = |grad rho| / rho^(4/3)
    x = np.where(mask, np.abs(grad_arr) / (rho_eff ** (4.0 / 3.0)), 0.0)
    asinh_x = np.arcsinh(x)

    # Enhancement factor: F_x = 1 + (beta * x^2 / C_x) / (1 + 6 * beta * x * asinh(x))
    denom = 1.0 + 6.0 * BETA_B88 * x * asinh_x
    fx = np.where(mask, 1.0 + (BETA_B88 * x * x / CX_LDA) / denom, 1.0)

    vx = vx_lda * fx
    eps_x = eps_lda * fx

    return vx, eps_x, fx


def eval_pbe_exchange(
    rho: np.ndarray,
    grad_rho: np.ndarray,
    alpha: float = 2.0 / 3.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate Perdew-Burke-Ernzerhof (PBE) GGA exchange potential, energy density, and enhancement factor.

    Parameters:
        rho: Electron density array (Bohr^-3)
        grad_rho: Gradient magnitude |grad rho| array (Bohr^-4)
        alpha: Underlying LDA alpha parameter (default 2/3)

    Returns:
        (vx, eps_x, fx):
            vx: Local exchange potential (Hartree)
            eps_x: Exchange energy density per electron (Hartree)
            fx: Enhancement factor F_x >= 1.0
    """
    rho_arr = np.asarray(rho, dtype=np.float64)
    grad_arr = np.asarray(grad_rho, dtype=np.float64)
    rho_eff = np.maximum(rho_arr, RHO_FLOOR)
    mask = rho_arr > RHO_FLOOR

    vx_lda, eps_lda = eval_slater_exchange(rho_arr, alpha=alpha)

    # Fermi momentum k_F = (3 * pi^2 * rho)^(1/3)
    k_f = (3.0 * PI * PI * rho_eff) ** (1.0 / 3.0)
    s = np.where(mask, np.abs(grad_arr) / (2.0 * k_f * rho_eff), 0.0)

    # Enhancement factor: F_x(s) = 1 + kappa - kappa / (1 + mu * s^2 / kappa)
    fx = np.where(mask, 1.0 + KAPPA_PBE - KAPPA_PBE / (1.0 + MU_PBE * s * s / KAPPA_PBE), 1.0)

    vx = vx_lda * fx
    eps_x = eps_lda * fx

    return vx, eps_x, fx


def eval_exchange_potential(
    rho: np.ndarray,
    grad_rho: Optional[np.ndarray] = None,
    kernel: str = "slater",
    alpha: float = 2.0 / 3.0
) -> np.ndarray:
    """
    Unified entry point for evaluating exchange potentials across all supported kernels.

    Supported kernels:
    - 'slater' / 'lda' / 'dirac': Slater-Dirac local exchange
    - 'becke88' / 'b88': Becke 88 GGA exchange
    - 'pbe': Perdew-Burke-Ernzerhof GGA exchange
    """
    k_norm = kernel.lower().strip()
    if grad_rho is None:
        grad_rho = np.zeros_like(rho)

    if k_norm in ("slater", "lda", "dirac"):
        vx, _ = eval_slater_exchange(rho, alpha=alpha)
        return vx
    elif k_norm in ("becke88", "b88"):
        vx, _, _ = eval_becke88_exchange(rho, grad_rho, alpha=alpha)
        return vx
    elif k_norm == "pbe":
        vx, _, _ = eval_pbe_exchange(rho, grad_rho, alpha=alpha)
        return vx
    else:
        raise ValueError(f"Unknown exchange kernel: '{kernel}'. Choose 'slater', 'becke88', or 'pbe'.")


def eval_pw92_correlation(
    rho: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate Perdew-Wang 92 (PW92) LDA correlation potential and energy density.

    Parameters:
        rho: Electron density array (Bohr^-3)

    Returns:
        (vc, eps_c):
            vc: Local correlation potential (Hartree)
            eps_c: Correlation energy density per electron (Hartree)
    """
    rho_arr = np.asarray(rho, dtype=np.float64)
    rho_eff = np.maximum(rho_arr, RHO_FLOOR)
    mask = rho_arr > RHO_FLOOR

    rs = (3.0 / (4.0 * PI * rho_eff)) ** (1.0 / 3.0)
    rs_sqrt = np.sqrt(rs)

    q_val = 2.0 * A_PW92 * (
        BETA1_PW92 * rs_sqrt
        + BETA2_PW92 * rs
        + BETA3_PW92 * rs * rs_sqrt
        + BETA4_PW92 * rs * rs
    )
    q_prime = 2.0 * A_PW92 * (
        0.5 * BETA1_PW92 / rs_sqrt
        + BETA2_PW92
        + 1.5 * BETA3_PW92 * rs_sqrt
        + 2.0 * BETA4_PW92 * rs
    )
    ln_term = np.log(1.0 + 1.0 / q_val)

    eps_c_unif = -2.0 * A_PW92 * (1.0 + ALPHA1_PW92 * rs) * ln_term
    deps_drs = (
        -2.0 * A_PW92 * ALPHA1_PW92 * ln_term
        + 2.0 * A_PW92 * (1.0 + ALPHA1_PW92 * rs) * q_prime / (q_val * (1.0 + q_val))
    )
    vc_unif = eps_c_unif - (rs / 3.0) * deps_drs

    vc = np.where(mask, vc_unif, 0.0)
    eps_c = np.where(mask, eps_c_unif, 0.0)
    return vc, eps_c


def eval_pbe_correlation(
    rho: np.ndarray,
    grad_rho: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate Perdew-Burke-Ernzerhof (PBE) GGA correlation potential, energy density, and H term.

    Parameters:
        rho: Electron density array (Bohr^-3)
        grad_rho: Gradient magnitude |grad rho| array (Bohr^-4)

    Returns:
        (vc, eps_c, H):
            vc: Local correlation potential (Hartree)
            eps_c: Correlation energy density per electron (Hartree)
            H: Gradient correction term H >= 0
    """
    rho_arr = np.asarray(rho, dtype=np.float64)
    grad_arr = np.asarray(grad_rho, dtype=np.float64)
    rho_eff = np.maximum(rho_arr, RHO_FLOOR)
    mask = rho_arr > RHO_FLOOR

    rs = (3.0 / (4.0 * PI * rho_eff)) ** (1.0 / 3.0)
    rs_sqrt = np.sqrt(rs)

    q_val = 2.0 * A_PW92 * (
        BETA1_PW92 * rs_sqrt
        + BETA2_PW92 * rs
        + BETA3_PW92 * rs * rs_sqrt
        + BETA4_PW92 * rs * rs
    )
    q_prime = 2.0 * A_PW92 * (
        0.5 * BETA1_PW92 / rs_sqrt
        + BETA2_PW92
        + 1.5 * BETA3_PW92 * rs_sqrt
        + 2.0 * BETA4_PW92 * rs
    )
    ln_term = np.log(1.0 + 1.0 / q_val)

    eps_c_unif = -2.0 * A_PW92 * (1.0 + ALPHA1_PW92 * rs) * ln_term
    deps_drs = (
        -2.0 * A_PW92 * ALPHA1_PW92 * ln_term
        + 2.0 * A_PW92 * (1.0 + ALPHA1_PW92 * rs) * q_prime / (q_val * (1.0 + q_val))
    )
    vc_unif = eps_c_unif - (rs / 3.0) * deps_drs

    # Dimensionless density gradient t = |grad rho| / (2 * k_s * rho)
    k_f = (3.0 * PI * PI * rho_eff) ** (1.0 / 3.0)
    k_s = np.sqrt(4.0 * k_f / PI)
    t = np.abs(grad_arr) / (2.0 * k_s * rho_eff)

    exp_term = np.exp(-eps_c_unif / GAMMA_PBE_C)
    a_param = (BETA_PBE_C / GAMMA_PBE_C) / np.maximum(exp_term - 1.0, 1.0e-12)

    t2 = t * t
    t4 = t2 * t2
    u_denom = 1.0 + a_param * t2 + a_param * a_param * t4
    u_val = (1.0 + a_param * t2) / u_denom
    w_val = (BETA_PBE_C / GAMMA_PBE_C) * t2 * u_val
    h_val = GAMMA_PBE_C * np.log(1.0 + w_val)

    # Functional derivatives
    dh_dw = GAMMA_PBE_C / (1.0 + w_val)
    du_dt = (-2.0 * a_param**2 * t * t2 - 4.0 * a_param**3 * t * t4) / (u_denom**2)
    dw_dt = (BETA_PBE_C / GAMMA_PBE_C) * (2.0 * t * u_val + t2 * du_dt)

    du_da = -a_param * t4 * (2.0 + a_param * t2) / (u_denom**2)
    dw_da = (BETA_PBE_C / GAMMA_PBE_C) * t2 * du_da
    da_drs = (a_param**2 / BETA_PBE_C) * exp_term * deps_drs

    dh_dt = dh_dw * dw_dt
    dh_drs = dh_dw * dw_da * da_drs

    vc_pbe = vc_unif + h_val - (rs / 3.0) * dh_drs - (7.0 / 6.0) * t * dh_dt

    vc = np.where(mask, vc_pbe, 0.0)
    eps_c = np.where(mask, eps_c_unif + h_val, 0.0)
    H = np.where(mask, h_val, 0.0)
    return vc, eps_c, H


def eval_correlation_potential(
    rho: np.ndarray,
    grad_rho: Optional[np.ndarray] = None,
    kernel: str = "pbe",
) -> np.ndarray:
    """
    Unified entry point for evaluating correlation potentials across all supported kernels.

    Supported kernels:
    - 'pw92' / 'lda': Perdew-Wang 92 LDA correlation
    - 'pbe' / 'gga': Perdew-Burke-Ernzerhof GGA correlation
    """
    k_norm = kernel.lower().strip()
    if grad_rho is None:
        grad_rho = np.zeros_like(rho)

    if k_norm in ("pw92", "lda"):
        vc, _ = eval_pw92_correlation(rho)
        return vc
    elif k_norm in ("pbe", "gga"):
        vc, _, _ = eval_pbe_correlation(rho, grad_rho)
        return vc
    elif k_norm in ("none", ""):
        return np.zeros_like(rho)
    else:
        raise ValueError(f"Unknown correlation kernel: '{kernel}'. Choose 'pw92' or 'pbe'.")


def eval_xc_potential(
    rho: np.ndarray,
    grad_rho: Optional[np.ndarray] = None,
    exchange_kernel: str = "pbe",
    correlation_kernel: str = "pbe",
    alpha: float = 2.0 / 3.0,
) -> np.ndarray:
    """
    Evaluate combined exchange-correlation potential v_xc(r) = v_x(r) + v_c(r).
    """
    vx = eval_exchange_potential(rho, grad_rho=grad_rho, kernel=exchange_kernel, alpha=alpha)
    if correlation_kernel and correlation_kernel.lower() not in ("none", ""):
        vc = eval_correlation_potential(rho, grad_rho=grad_rho, kernel=correlation_kernel)
        return vx + vc
    return vx


def fit_exchange_potential(
    r_grid: np.ndarray,
    vx_grid: np.ndarray,
    vexp: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float]:
    """
    Fit radial exchange potential to multi-exponential screening basis:
        v_x(r) ~ sum_k c_k^X exp(-zeta_k r)

    Parameters:
        r_grid: Radial points r_p (Bohr)
        vx_grid: Potential values v_x(r_p) (Hartree)
        vexp: Screening exponents {zeta_k} (Bohr^-1)
        weights: Radial integration metric/weights W_p * r_p^2 (default: uniform)

    Returns:
        (vcoef, rms_residual):
            vcoef: Expansion coefficients {c_k^X} (Hartree)
            rms_residual: Weighted root-mean-square residual (Hartree)
    """
    r_arr = np.asarray(r_grid, dtype=np.float64)
    v_arr = np.asarray(vx_grid, dtype=np.float64)
    zeta = np.asarray(vexp, dtype=np.float64)
    n_terms = len(zeta)

    if weights is None:
        w_eff = r_arr ** 2
    else:
        w_eff = np.asarray(weights, dtype=np.float64) * (r_arr ** 2)

    # Design matrix: A_p,k = exp(-zeta_k * r_p) * sqrt(w_eff_p)
    sqrt_w = np.sqrt(np.maximum(w_eff, 0.0))
    X = np.exp(-np.outer(r_arr, zeta)) * sqrt_w[:, np.newaxis]
    y = v_arr * sqrt_w

    coefs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    v_fit = np.exp(-np.outer(r_arr, zeta)) @ coefs
    sum_w = np.sum(w_eff)
    if sum_w > 0:
        rms = float(np.sqrt(np.sum(w_eff * (v_arr - v_fit) ** 2) / sum_w))
    else:
        rms = float(np.sqrt(np.mean((v_arr - v_fit) ** 2)))

    return coefs, rms
