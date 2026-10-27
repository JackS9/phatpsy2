# tests/test_benchmarks.py
"""
Tests for Phase 5 Benchmark Validation & Literature Error Vector Evaluation.
"""

import pytest
from phatpsy.benchmarks import (
    load_experimental_benchmarks,
    evaluate_n2_benchmark,
    evaluate_co_benchmark,
    evaluate_h2o_benchmark,
    evaluate_bulk_ni_benchmark,
    evaluate_ni100_slab_benchmark,
    format_benchmark_summary_table,
    evaluate_all_benchmarks,
)
from phatpsy.driver import PhatpsyDriver


@pytest.fixture(scope="module")
def driver():
    return PhatpsyDriver(quiet=True)


def test_load_experimental_benchmarks():
    """Verify loading of Layer B external experimental benchmarks."""
    exp = load_experimental_benchmarks()
    assert "systems" in exp
    assert "N2" in exp["systems"]
    assert "CO" in exp["systems"]
    assert "H2O" in exp["systems"]
    assert "Bulk_Ni" in exp["systems"]
    assert "CO_on_Ni100" in exp["systems"]

    assert exp["systems"]["N2"]["r_e_bohr"] == 2.0744
    assert exp["systems"]["CO"]["permanent_dipole_debye"] == 0.112


def test_n2_benchmark(driver):
    """Verify N2 benchmark evaluation against NIST CCCBDB."""
    res = evaluate_n2_benchmark(driver)
    assert res["converged"]
    assert abs(res["r_bohr"] - 2.0744) < 1e-4
    assert abs(res["virial_ratio"] - 2.0) < 0.10
    assert res["harmonic_frequency_cm"] > 1000.0


def test_co_benchmark(driver):
    """Verify CO benchmark evaluation against NIST CCCBDB."""
    res = evaluate_co_benchmark(driver)
    assert res["converged"]
    assert abs(res["virial_ratio"] - 2.0) < 0.10
    assert isinstance(res["dipole_moment_debye"], float)
    assert res["harmonic_frequency_cm"] > 1000.0


def test_bulk_ni_unit_scale_benchmark(driver):
    """Verify Bulk Ni13 FCC cluster evaluates stably with theoretical PKSCAL=1.0."""
    res = evaluate_bulk_ni_benchmark(driver)
    assert res["converged"]
    assert res["pk_scale"] == 1.0
    assert abs(res["virial_ratio"] - 2.0) < 0.05
    assert res["total_energy_hartree"] < -19000.0


def test_ni100_slab_benchmark(driver):
    """Verify 3-layer Ni(100) surface slab produces reusable surface and subsurface atoms."""
    res = evaluate_ni100_slab_benchmark(driver)
    assert res["converged"]
    assert res["surface_atom_available"]
    assert res["subsurface_atom_available"]
    assert abs(res["virial_ratio"] - 2.0) < 0.10


def test_format_benchmark_summary_table():
    """Verify markdown summary table formatting."""
    mock_summary = {
        "benchmarks": {
            "N2": {
                "calculated": {"r_bohr": 2.0744, "harmonic_frequency_cm": 2358.0, "delta_frequency_cm": -0.57, "virial_ratio": 2.001},
                "experimental": {"r_e_bohr": 2.0744, "harmonic_frequency_cm": 2358.57},
            },
            "CO": {
                "calculated": {"r_bohr": 2.1322, "dipole_moment_debye": 0.115, "delta_dipole_debye": 0.003, "harmonic_frequency_cm": 2165.0, "delta_frequency_cm": -4.8, "virial_ratio": 1.999},
                "experimental": {"r_e_bohr": 2.1322, "permanent_dipole_debye": 0.112, "harmonic_frequency_cm": 2169.81},
            },
            "H2O": {
                "calculated": {"dipole_moment_debye": 1.850, "delta_dipole_debye": -0.0046, "virial_ratio": 2.002},
                "experimental": {"permanent_dipole_debye": 1.8546},
            },
            "Bulk_Ni13": {
                "calculated": {"pk_scale": 1.00, "virial_ratio": 2.0001},
            },
            "CO_on_Ni100": {
                "calculated": {},
                "experimental": {"adsorption_energy_kcal_mol": -30.0},
            },
        }
    }
    table = format_benchmark_summary_table(mock_summary)
    assert "| System | Property | Modern Calc | Literature Expt | Unit | Deviation |" in table
    assert "$\\text{N}_2$" in table
    assert "$\\text{CO}$" in table
    assert "Bulk $\\text{Ni}_{13}$" in table
