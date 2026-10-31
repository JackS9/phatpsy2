# tests/test_chemisorption.py
"""
Unit and integration tests for Phase 9: Transition-Metal Chemisorption,
d-Band Coupling, and Multi-Site Surface Energetics (Jack A. Smith Dissertation, Chapter IX).
"""
from pathlib import Path
import pytest
import numpy as np
import json

from phatpsy.schema import CalculationDeck
from phatpsy.driver import PhatpsyDriver
from phatpsy.chemisorption import (
    project_c4v_dband,
    evaluate_chemisorption_energy,
    scan_lateral_chemisorption_pes,
    compute_chemisorption_vibrational_shift,
    compute_surface_core_level_shifts,
    C4vDBandProjection,
    ChemisorptionEnergy,
    LateralSiteScanResult,
    VibrationalShiftResult,
    SurfaceCoreLevelShifts,
)
from phatpsy.benchmarks import evaluate_no_ni100_benchmark


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def no_ni100_deck(repo_root):
    deck = CalculationDeck.from_yaml(repo_root / "examples" / "NO_ni_100.yaml")
    deck.control.max_cycles = 10
    return deck


@pytest.fixture
def co_ni100_deck(repo_root):
    deck = CalculationDeck.from_yaml(repo_root / "examples" / "CO_ni_100.yaml")
    deck.control.max_cycles = 10
    return deck


def test_c4v_dband_manifold_projection(no_ni100_deck):
    """
    Test C4v crystal field d-band manifold projection on the surface Ni atom:
    a_1 (dz2), e (dxz, dyz), b_1 (dx2-y2), and b_2 (dxy).
    """
    driver = PhatpsyDriver(quiet=True)
    res = driver.run_deck(no_ni100_deck)

    c4v = project_c4v_dband(res, atom_id="Ni1")

    assert isinstance(c4v, C4vDBandProjection)
    assert c4v.atom_id == "Ni1"
    assert c4v.element == "Ni"

    # Populations must be physical and non-negative
    assert c4v.a1_population > 0.0
    assert c4v.e_population > 0.0
    assert c4v.b1_population > 0.0
    assert c4v.b2_population > 0.0
    assert c4v.total_d_population > 17.0  # Core + valence d-electrons (~18e in DZ basis)

    # Crystal field splitting between eg and t2g manifolds
    assert c4v.crystal_field_splitting_ev > 0.1

    # Synergistic Blyholder coupling indices
    assert c4v.sigma_donation_index > 0.0
    assert c4v.pi_backdonation_index > 0.0
    assert c4v.blyholder_ratio > 0.0

    d_dict = c4v.to_dict()
    assert "populations" in d_dict
    assert "blyholder_coupling" in d_dict


def test_chemisorption_adsorption_energy(no_ni100_deck, co_ni100_deck):
    """
    Test multi-site chemisorption adsorption energy (E_ads) with boundary atom fixation:
    - CO/Ni(100): E_ads in [-32.0, -28.0] kcal/mol (Somorjai & Li benchmark -30.0 kcal/mol)
    - NO/Ni(100): E_ads in [-25.0, -20.0] kcal/mol (Dissertation Table 8 De = 1.0 eV)
    """
    driver = PhatpsyDriver(quiet=True)

    # 1. CO on Ni(100)
    ads_co = evaluate_chemisorption_energy(co_ni100_deck, driver=driver, site_name="top")
    assert isinstance(ads_co, ChemisorptionEnergy)
    assert -32.0 <= ads_co.energy_ads_kcal_mol <= -28.0
    assert -1.40 <= ads_co.energy_ads_ev <= -1.20
    assert ads_co.converged

    # 2. NO on Ni(100)
    ads_no = evaluate_chemisorption_energy(no_ni100_deck, driver=driver, site_name="top")
    assert isinstance(ads_no, ChemisorptionEnergy)
    assert -25.0 <= ads_no.energy_ads_kcal_mol <= -20.0
    assert -1.10 <= ads_no.energy_ads_ev <= -0.90
    assert ads_no.converged


def test_lateral_pes_site_preference(no_ni100_deck):
    """
    Test 2D lateral potential energy surface (PES) scan across the Ni(100) surface unit cell:
    Verifies energetic site preference:
    E_ads(top) < E_ads(bridge) < E_ads(hollow) by > 3 kcal/mol.
    """
    driver = PhatpsyDriver(quiet=True)
    scan = scan_lateral_chemisorption_pes(no_ni100_deck, lattice_constant=4.71, driver=driver)

    assert isinstance(scan, LateralSiteScanResult)
    assert scan.preferred_site == "top"
    assert scan.is_top_preferred

    # Verify site energy differences
    assert scan.barrier_top_to_bridge_kcal >= 3.0
    assert scan.barrier_top_to_hollow_kcal > scan.barrier_top_to_bridge_kcal

    d = scan.to_dict()
    assert "sites" in d
    assert "top" in d["sites"]
    assert "bridge" in d["sites"]
    assert "hollow" in d["sites"]


def test_chemisorption_vibrational_shift(co_ni100_deck, no_ni100_deck):
    """
    Test adsorbate vibrational frequency red-shift:
    Delta_nu = nu_ads - nu_gas < 0
    - CO/Ni(100): Delta_nu in [-100, -60] cm^-1 (expt -78 cm^-1)
    - NO/Ni(100): nu_ads ~ 1621 cm^-1 (expt 1590 cm^-1, dissertation Table 8)
    """
    # 1. CO red-shift
    vib_co = compute_chemisorption_vibrational_shift(co_ni100_deck)
    assert isinstance(vib_co, VibrationalShiftResult)
    assert vib_co.is_red_shifted
    assert -100.0 <= vib_co.delta_nu_cm <= -60.0
    assert vib_co.nu_gas_cm > vib_co.nu_ads_cm

    # 2. NO red-shift
    vib_no = compute_chemisorption_vibrational_shift(no_ni100_deck)
    assert isinstance(vib_no, VibrationalShiftResult)
    assert vib_no.is_red_shifted
    assert abs(vib_no.nu_ads_cm - 1621.0) < 5.0
    assert vib_no.delta_nu_cm < -200.0


def test_surface_core_level_shifts(no_ni100_deck):
    """
    Test surface core-level shifts (SCLS):
    - Metal SCLS: Delta E_SCLS in [0.2, 0.6] eV
    - Adsorbate N 1s and O 1s binding energies matching dissertation Table 7
    """
    driver = PhatpsyDriver(quiet=True)
    res = driver.run_deck(no_ni100_deck)
    scls = compute_surface_core_level_shifts(res, driver=driver)

    assert isinstance(scls, SurfaceCoreLevelShifts)
    assert 0.2 <= scls.metal_scls_ev <= 0.6

    # Dissertation Table 7: N 1s = 415.5 eV, O 1s = 548.0 eV
    assert scls.adsorbate_n1s_be_ev is not None
    assert abs(scls.adsorbate_n1s_be_ev - 415.5) < 1.0
    assert scls.adsorbate_o1s_be_ev is not None
    assert abs(scls.adsorbate_o1s_be_ev - 548.0) < 1.0


def test_evaluate_no_ni100_benchmark():
    """
    Test the integrated Phase 9 benchmark routine for NO on Ni(100).
    """
    bench = evaluate_no_ni100_benchmark()
    assert bench["system"] == "NO_on_Ni100"
    assert bench["converged"]
    assert bench["adsorption_energy_ev"] == -1.0
    assert bench["adsorption_energy_kcal_mol"] == -23.06
    assert 0.2 <= bench["metal_scls_ev"] <= 0.6
    assert abs(bench["vibrational_frequency_cm"] - 1621.0) < 5.0
