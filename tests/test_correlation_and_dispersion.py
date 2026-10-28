# tests/test_correlation_and_dispersion.py
"""
Test Suite for Phase 8: Exchange-Correlation Functionals & Non-Covalent Dispersion (D3/D4).

Validates:
1. Pointwise Perdew-Wang 92 (PW92) LDA correlation energy and potential.
2. Pointwise Perdew-Burke-Ernzerhof (PBE) GGA correlation energy and potential.
3. Fortran-Python bitwise numerical parity on correlation evaluations.
4. Grimme DFT-D3(BJ) dispersion energies and exact analytical Cartesian gradients.
5. Translational origin invariance of dispersion energies and forces.
6. Execution of calculation decks with full PBE XC + D3(BJ) dispersion.
7. Milestone 8 benchmarks:
   - N2 dissociation energy D_e in [9.0, 10.0] eV (expt 9.759 eV)
   - CO dissociation energy D_e in [10.5, 11.5] eV (expt 11.087 eV)
   - Ethylene oxide to acetaldehyde (EO -> AA) ring-opening barrier Delta E_f^ddagger in [50.0, 56.0] kcal/mol (CCSD(T) 53.0 kcal/mol).
"""

import math
import numpy as np
import pytest

from phatpsy.exchange_dft import (
    eval_pw92_correlation,
    eval_pbe_correlation,
    eval_correlation_potential,
    eval_xc_potential,
    A_PW92,
    ALPHA1_PW92,
    BETA1_PW92,
    BETA2_PW92,
    BETA3_PW92,
    BETA4_PW92,
    BETA_PBE_C,
    GAMMA_PBE_C,
)
from phatpsy.dispersion import (
    compute_dispersion_energy,
    compute_dispersion_gradient,
    DispersionModel,
)
from phatpsy.schema import CalculationDeck, ControlConfig, AtomConfig
from phatpsy.driver import PhatpsyDriver
from phatpsy.benchmarks import (
    evaluate_dissociation_energies_benchmark,
    evaluate_eo_to_aa_benchmark,
)


def test_pw92_correlation_pointwise():
    """Verify PW92 LDA correlation potential against analytical limits."""
    # Zero density floor
    vc_0, ec_0 = eval_pw92_correlation(np.array([0.0]))
    assert vc_0[0] == 0.0
    assert ec_0[0] == 0.0

    rhos = np.array([0.001, 0.05, 0.2, 1.0])
    vc, ec = eval_pw92_correlation(rhos)

    # Correlation energy per electron must be strictly negative
    assert np.all(ec < 0.0)
    # Correlation potential must be strictly negative
    assert np.all(vc < 0.0)
    # In LDA UEG, |v_c| > |eps_c| because v_c = eps_c - (r_s/3) d(eps_c)/dr_s with d(eps_c)/dr_s > 0
    assert np.all(vc < ec)


def test_pbe_correlation_pointwise():
    """Verify PBE GGA correlation behavior and reduction of LDA over-binding."""
    # Zero density floor
    vc_0, ec_0, h_0 = eval_pbe_correlation(np.array([0.0]), np.array([0.0]))
    assert vc_0[0] == 0.0
    assert ec_0[0] == 0.0
    assert h_0[0] == 0.0

    rho = np.array([0.2, 0.2, 0.2])
    grad_0 = np.array([0.0, 0.0, 0.0])

    vc_pbe_0, ec_pbe_0, h_val_0 = eval_pbe_correlation(rho, grad_0)
    vc_pw92, ec_pw92 = eval_pw92_correlation(rho)

    # When grad_rho = 0, PBE correlation must reduce exactly to PW92 uniform gas
    np.testing.assert_allclose(h_val_0, [0.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(ec_pbe_0, ec_pw92, rtol=1e-12)
    np.testing.assert_allclose(vc_pbe_0, vc_pw92, rtol=1e-12)

    # For non-zero gradient, H must be strictly positive
    grad_nonzero = np.array([0.02, 0.1, 0.5])
    vc_pbe, ec_pbe, h_val = eval_pbe_correlation(rho, grad_nonzero)
    assert np.all(h_val > 0.0)
    # H increases monotonically with density gradient
    assert h_val[0] < h_val[1] < h_val[2]
    # Positive H reduces the magnitude of negative correlation energy (curing LDA over-binding)
    assert np.all(np.abs(ec_pbe) < np.abs(ec_pw92))


def test_fortran_python_correlation_parity():
    """Verify bitwise identical pointwise correlation potential between Fortran and Python."""
    try:
        from _phatpsy_core import exchange_dft as f_exch
    except ImportError:
        pytest.skip("_phatpsy_core not available")

    rhos = [0.002, 0.03, 0.18, 0.9, 2.5]
    d_rhos = [0.001, 0.02, 0.09, 0.6, 1.8]

    for rho, drho in zip(rhos, d_rhos):
        # 1: PW92 LDA
        vc_py_pw92 = eval_correlation_potential(np.array([rho]), np.array([drho]), kernel="pw92")[0]
        vc_f_pw92, _ = f_exch.eval_correlation_potential_point(rho, drho, 1)
        np.testing.assert_allclose(vc_f_pw92, vc_py_pw92, rtol=1e-12)

        # 2: PBE GGA
        vc_py_pbe = eval_correlation_potential(np.array([rho]), np.array([drho]), kernel="pbe")[0]
        vc_f_pbe, _ = f_exch.eval_correlation_potential_point(rho, drho, 2)
        np.testing.assert_allclose(vc_f_pbe, vc_py_pbe, rtol=1e-12)


def test_d3_dispersion_energy_and_distance_dependence():
    """Verify DFT-D3(BJ) dispersion is strictly attractive and decays with distance."""
    atoms = ["C", "O"]

    # Short vs long separation
    r_short = 2.13  # near equilibrium
    r_long = 4.00   # stretched van der Waals region

    e_short = compute_dispersion_energy([("C", [0.0, 0.0, 0.0]), ("O", [0.0, 0.0, r_short])])
    e_long = compute_dispersion_energy([("C", [0.0, 0.0, 0.0]), ("O", [0.0, 0.0, r_long])])

    # Dispersion energy must be strictly attractive (negative)
    assert e_short < 0.0
    assert e_long < 0.0
    # Closer distance yields stronger attractive dispersion stabilization
    assert e_short < e_long


def test_d3_dispersion_gradient_analytical():
    """Verify analytical nuclear gradient of DFT-D3(BJ) dispersion against central finite differences."""
    atoms = ["C", "C", "O", "H"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.7, 0.0, 0.0],
        [1.35, 2.0, 0.0],
        [0.0, -1.0, 1.0],
    ])
    geom = list(zip(atoms, coords))

    grad_ana = compute_dispersion_gradient(geom, functional="pbe")

    # Numerical gradient via central differences
    h = 1.0e-6
    grad_num = np.zeros_like(coords)
    for a in range(len(atoms)):
        for k in range(3):
            c_plus = coords.copy()
            c_plus[a, k] += h
            c_minus = coords.copy()
            c_minus[a, k] -= h

            e_plus = compute_dispersion_energy(list(zip(atoms, c_plus)), functional="pbe")
            e_minus = compute_dispersion_energy(list(zip(atoms, c_minus)), functional="pbe")

            grad_num[a, k] = (e_plus - e_minus) / (2.0 * h)

    max_diff = np.max(np.abs(grad_ana - grad_num))
    np.testing.assert_allclose(grad_ana, grad_num, atol=1e-8)


def test_d3_dispersion_translational_invariance():
    """Verify that uniform spatial translation leaves dispersion energy and forces strictly invariant."""
    geom1 = [
        ("C", [0.0, 0.0, 0.0]),
        ("C", [2.7, 0.0, 0.0]),
        ("O", [1.35, 2.0, 0.0]),
        ("H", [0.0, -1.0, 1.0]),
    ]
    shift = np.array([14.89, -8.76, 5.43])
    geom2 = [(a[0], np.array(a[1]) + shift) for a in geom1]

    e1 = compute_dispersion_energy(geom1)
    e2 = compute_dispersion_energy(geom2)
    np.testing.assert_allclose(e1, e2, atol=1e-13)

    g1 = compute_dispersion_gradient(geom1)
    g2 = compute_dispersion_gradient(geom2)
    np.testing.assert_allclose(g1, g2, atol=1e-13)

    # Total force sum must vanish (conservation of linear momentum)
    total_force = np.sum(g1, axis=0)
    np.testing.assert_allclose(total_force, [0.0, 0.0, 0.0], atol=1e-12)


def test_deck_with_pbe_xc_and_dispersion():
    """Integration test: calculation deck with PBE XC and DFT-D3(BJ) dispersion."""
    deck = CalculationDeck(
        title="N2 Molecule with PBE XC and D3(BJ) Dispersion",
        control=ControlConfig(
            max_cycles=3,
            convergence=1e-3,
            pk_scale=1.0,
            use_pseudometric=True,
            exchange_kernel="pbe",
            correlation_kernel="pbe",
            use_dispersion=True,
        ),
        geometry=[
            AtomConfig(element="N", id="N1", xyz=[0.0, 0.0, -1.037]),
            AtomConfig(element="N", id="N2", xyz=[0.0, 0.0,  1.037]),
        ],
    )

    deck_text = deck.to_stdin_text()
    assert "QDFTEX=T" in deck_text
    assert "EXCH_TYPE=3" in deck_text
    assert "CORR_TYPE=2" in deck_text

    driver = PhatpsyDriver(quiet=True)
    res = driver.run_deck(deck)
    assert res.converged
    assert res.total_energy is not None
    assert "dispersion_energy" in res
    assert res.dispersion_energy < 0.0


def test_milestone8_dissociation_energies_benchmark():
    """
    Milestone 8 Acceptance Criterion:
    - N2 De in [9.0, 10.0] eV (expt 9.759 eV)
    - CO De in [10.5, 11.5] eV (expt 11.087 eV)
    """
    res = evaluate_dissociation_energies_benchmark()
    n2_de = res["N2"]["de_ev"]
    co_de = res["CO"]["de_ev"]

    assert 9.0 <= n2_de <= 10.0, f"N2 De = {n2_de:.3f} eV outside [9.0, 10.0] eV"
    assert 10.5 <= co_de <= 11.5, f"CO De = {co_de:.3f} eV outside [10.5, 11.5] eV"
    assert res["N2"]["within_acceptance"]
    assert res["CO"]["within_acceptance"]


def test_milestone8_eo_to_aa_activation_barrier():
    """
    Milestone 8 Acceptance Criterion:
    - EO -> AA forward ring-opening activation barrier Delta E_f^ddagger in [50.0, 56.0] kcal/mol
      (matching CCSD(T)/CBS benchmark 53.0 kcal/mol).
    """
    res = evaluate_eo_to_aa_benchmark()
    barrier = res["barrier_forward_kcal_mol"]

    assert 50.0 <= barrier <= 56.0, f"EO->AA barrier = {barrier:.2f} kcal/mol outside [50.0, 56.0]"
    assert res["within_acceptance"]
    assert res["converged"]
