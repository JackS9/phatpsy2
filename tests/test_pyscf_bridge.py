# tests/test_pyscf_bridge.py
"""
Tests for Phase 10: Automated Multi-Level Computational Parity Suite against PySCF.
Validates PySCF bridge runners, L2 error vector norm metric, and fallback behavior.
"""

import math
import pytest
from pathlib import Path

from phatpsy.builder import build_diatomic
from phatpsy.driver import PhatpsyDriver
from phatpsy.benchmarks import compute_l2_error_norm
from tools.benchmark_pyscf_bridge import (
    HAS_PYSCF,
    run_single_pyscf_system,
    run_system_parity,
    run_pyscf_parity_suite,
    ParitySuiteResult,
    ParityComparisonRecord,
)


@pytest.fixture(scope="module")
def driver():
    return PhatpsyDriver(quiet=True)


def test_pyscf_availability_flag():
    """Verify PySCF availability detection."""
    assert isinstance(HAS_PYSCF, bool)


@pytest.mark.skipif(not HAS_PYSCF, reason="PySCF is not installed")
def test_single_system_pyscf_calculation():
    """Verify PySCF execution for a small diatomic system (N2)."""
    deck = build_diatomic("N", "N", 2.068, basis="DZ", spin_flip=True, units="bohr")
    pyscf_res = run_single_pyscf_system(deck, basis="sto-3g")

    rhf = pyscf_res["rhf"]
    assert rhf is not None
    assert rhf["converged"]
    assert rhf["total_energy"] < -100.0
    # Inversion symmetry: dipole must be strictly zero
    assert abs(rhf["dipole_magnitude_debye"]) < 1e-4

    pbe = pyscf_res["pbe"]
    assert pbe is not None
    assert pbe["converged"]
    assert pbe["total_energy"] < -100.0


@pytest.mark.skipif(not HAS_PYSCF, reason="PySCF is not installed")
def test_system_parity_n2(driver):
    """Verify run_system_parity on N2 comparing PHATPSY and PySCF."""
    deck = build_diatomic("N", "N", 2.068, basis="DZ", spin_flip=True, units="bohr")
    rec = run_system_parity(deck, formula="N2", driver=driver, basis_pyscf="sto-3g")

    assert rec.formula == "N2"
    assert rec.phatpsy["converged"]
    assert rec.pyscf_rhf["converged"]

    # Centrosymmetric N2 must have dipole near zero in both codes
    assert rec.phatpsy["dipole_magnitude_debye"] == 0.0
    assert rec.pyscf_rhf["dipole_magnitude_debye"] < 1e-3
    assert rec.delta_dipole_debye < 1e-3

    # Energy difference percent should be physical (< 15%)
    assert rec.delta_energy_percent is not None
    assert abs(rec.delta_energy_percent) < 15.0


def test_compute_l2_error_norm_exact():
    """Verify L2 error norm is zero when calculated equals reference."""
    summary = {
        "benchmarks": {
            "N2": {
                "calculated": {"r_bohr": 2.0744, "harmonic_frequency_cm": 2358.57, "virial_ratio": 2.0},
                "experimental": {"r_e_bohr": 2.0744, "harmonic_frequency_cm": 2358.57},
            },
            "CO": {
                "calculated": {"r_bohr": 2.1322, "harmonic_frequency_cm": 2169.81, "virial_ratio": 2.0},
                "experimental": {"r_e_bohr": 2.1322, "harmonic_frequency_cm": 2169.81},
            },
        }
    }

    res = compute_l2_error_norm(summary, threshold=0.15)
    assert res["passed"]
    assert abs(res["l2_norm"]) < 1e-8


def test_compute_l2_error_norm_synthetic():
    """Verify mathematical correctness of ||epsilon||_2 formula with known errors."""
    # Two components with known relative errors: 0.03 and 0.04
    # Equal weights (0.5 each): sqrt(0.5 * 0.03^2 + 0.5 * 0.04^2) = sqrt(0.5 * 0.0009 + 0.5 * 0.0016)
    # = sqrt(0.00125) = 0.035355...
    summary = {
        "components": {
            "prop_A": {"relative_error": 0.03},
            "prop_B": {"relative_error": 0.04},
        }
    }

    res = compute_l2_error_norm(summary, threshold=0.15)
    expected_norm = math.sqrt(0.5 * (0.03 ** 2) + 0.5 * (0.04 ** 2))
    assert abs(res["l2_norm"] - expected_norm) < 1e-6
    assert res["passed"]

    # Test threshold rejection
    strict_res = compute_l2_error_norm(summary, threshold=0.02)
    assert not strict_res["passed"]


def test_compute_l2_error_norm_pyscf_parity():
    """Verify L2 error norm calculation from PySCF parity dictionary."""
    mock_parity = {
        "systems": {
            "N2": {"delta_energy_percent": 0.55, "phatpsy": {"total_energy": -108.36}, "pyscf_rhf": {"total_energy": -108.95}},
            "H2O": {"delta_energy_percent": 0.79, "phatpsy": {"total_energy": -76.62}, "pyscf_rhf": {"total_energy": -76.02}},
        }
    }

    res = compute_l2_error_norm(mock_parity, threshold=0.15)
    assert res["passed"]
    # (0.0055^2 + 0.0079^2)/2 = (0.00003025 + 0.00006241)/2 = 0.00004633 -> sqrt ~ 0.0068
    assert res["l2_norm"] < 0.02


def test_parity_suite_serialization_and_markdown():
    """Verify ParitySuiteResult serialization and report generation."""
    rec = ParityComparisonRecord(
        system_name="Test Dinitrogen",
        formula="N2",
        geometry_bohr=[{"element": "N", "xyz": [0, 0, 1]}, {"element": "N", "xyz": [0, 0, -1]}],
        phatpsy={"total_energy": -108.0, "virial_ratio": 2.0, "dipole_magnitude_debye": 0.0},
        pyscf_rhf={"method": "RHF", "total_energy": -108.5, "dipole_magnitude_debye": 0.0},
        pyscf_pbe={"method": "PBE", "total_energy": -109.0},
        pyscf_ccsd=None,
        delta_energy_hartree=0.5,
        delta_energy_percent=0.46,
        delta_dipole_debye=0.0,
        density_rmsd_au=0.04,
    )

    suite = ParitySuiteResult(
        timestamp="2026-10-30T00:00:00",
        pyscf_available=True,
        pyscf_version="2.14.0",
        basis_set="cc-pvdz",
        systems={"N2": rec},
        mean_energy_deviation_percent=0.46,
        max_energy_deviation_percent=0.46,
        passed=True,
        l2_error_norm={"l2_norm": 0.0046, "threshold": 0.15, "passed": True},
    )

    d = suite.to_dict()
    assert d["pyscf_available"] is True
    assert "N2" in d["systems"]
    assert d["l2_error_norm"]["l2_norm"] == 0.0046

    md = suite.generate_markdown_report()
    assert "# PHATPSY 2.0 vs PySCF High-Level ab initio Parity Report" in md
    assert "Composite $L_2$ Error Vector Norm Metric" in md
    assert "N2" in md
