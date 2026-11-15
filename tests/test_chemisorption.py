# tests/test_chemisorption.py
"""
Non-circular tests for transition-metal chemisorption analysis routines (Phase 9 / Phase 18):
C4v d-band projection, chemisorption energy decomposition, lateral PES scans,
vibrational shifts from PES curvature, and Koopmans core-level eigenvalue extraction.
"""
import copy
from pathlib import Path
import numpy as np
import pytest
from pyscf import gto, scf

from phatpsy.schema import CalculationDeck
from phatpsy.driver import PhatpsyDriver
from phatpsy.state import CalculationState
from phatpsy.chemisorption import (
    HARTREE_TO_KCAL_MOL,
    HARTREE_TO_EV,
    AMU_TO_ME,
    HARTREE_TO_CM,
    AU_FORCE_CONST_TO_MDYNE_A,
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
from phatpsy.vibrations import ATOMIC_MASSES_AMU
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
    return CalculationDeck.from_yaml(repo_root / "examples" / "CO_ni_100.yaml")


def test_c4v_dband_manifold_projection(no_ni100_deck):
    """
    Verify C4v crystal-field d-band projection extracts genuine orbital populations
    and raises when d-population data is absent (no fabricated fallback constants).
    """
    driver = PhatpsyDriver(quiet=True)
    res = driver.run_deck(no_ni100_deck)

    c4v = project_c4v_dband(res, atom_id="Ni1")
    assert isinstance(c4v, C4vDBandProjection)
    assert c4v.atom_id == "Ni1"
    assert c4v.element == "Ni"

    # Populations must be physical and conserve the d-shell sum
    assert c4v.a1_population > 0.0
    assert c4v.e_population > 0.0
    assert c4v.b1_population > 0.0
    assert c4v.b2_population > 0.0
    assert c4v.total_d_population == pytest.approx(
        c4v.a1_population + c4v.e_population + c4v.b1_population + c4v.b2_population,
        abs=1e-10,
    )
    assert c4v.crystal_field_splitting_ev >= 0.0
    assert c4v.sigma_donation_index >= 0.0
    assert c4v.pi_backdonation_index >= 0.0

    # Empty state must raise rather than falling back to hardcoded d-populations
    empty_state = CalculationState(
        {
            "total_energy": -1.0,
            "basis_info": {"iatomx": [1], "lx": [2], "mlx": [0]},
            "atom_populations": np.array([[np.nan, np.nan]]),
        }
    )
    with pytest.raises(ValueError, match="No valid d-orbital population data"):
        project_c4v_dband(empty_state, atom_id="Ni1")


def test_chemisorption_adsorption_energy(co_ni100_deck, no_ni100_deck):
    """
    Verify evaluate_chemisorption_energy computes genuine energy differences
    E_ads = E(slab+ads) - E(clean) - E(gas) without hardcoded empirical overrides.
    """
    driver = PhatpsyDriver(quiet=True)

    ads_co = evaluate_chemisorption_energy(co_ni100_deck, driver=driver, site_name="top")
    assert isinstance(ads_co, ChemisorptionEnergy)
    assert ads_co.converged
    expected_ha = ads_co.energy_slab_ads - ads_co.energy_clean_slab - ads_co.energy_gas_ads
    assert ads_co.energy_ads_hartree == pytest.approx(expected_ha, abs=1e-10)
    assert ads_co.energy_ads_kcal_mol == pytest.approx(expected_ha * HARTREE_TO_KCAL_MOL, abs=1e-8)
    assert ads_co.energy_ads_ev == pytest.approx(expected_ha * HARTREE_TO_EV, abs=1e-8)
    assert abs(ads_co.energy_ads_kcal_mol - (-30.0)) > 1.0

    ads_no = evaluate_chemisorption_energy(no_ni100_deck, driver=driver, site_name="top")
    expected_no_ha = ads_no.energy_slab_ads - ads_no.energy_clean_slab - ads_no.energy_gas_ads
    assert ads_no.energy_ads_hartree == pytest.approx(expected_no_ha, abs=1e-10)
    assert abs(ads_no.energy_ads_ev - (-1.0)) > 0.1


def test_lateral_pes_site_preference(co_ni100_deck):
    """
    Verify scan_lateral_chemisorption_pes runs genuinely displaced geometries
    rather than adding hardcoded +4.0 / +8.0 kcal/mol offsets.
    """
    driver = PhatpsyDriver(quiet=True)
    scan = scan_lateral_chemisorption_pes(
        co_ni100_deck,
        sites=["top", "bridge"],
        lattice_constant=4.71,
        driver=driver,
    )

    assert isinstance(scan, LateralSiteScanResult)
    assert set(scan.sites.keys()) == {"top", "bridge"}
    assert abs(scan.barrier_top_to_bridge_kcal - 4.0) > 1e-2

    # Direct verification: displace C1, O1 manually to bridge (a/2, 0) and run_deck
    d_bridge = copy.deepcopy(co_ni100_deck)
    for at in d_bridge.geometry:
        if at.element in ("C", "O"):
            at.xyz[0] += 4.71 / 2.0
            at.equivalent_to = None
            at.symmetry_op = None
            at.symmetry_axis = None
    ads_top = evaluate_chemisorption_energy(co_ni100_deck, driver=driver, site_name="top")
    ads_bridge = evaluate_chemisorption_energy(d_bridge, driver=driver, site_name="bridge")
    direct_delta_kcal = ads_bridge.energy_ads_kcal_mol - ads_top.energy_ads_kcal_mol
    assert scan.barrier_top_to_bridge_kcal == pytest.approx(direct_delta_kcal, abs=1e-3)

    expected_pref = min(scan.sites, key=lambda k: scan.sites[k]["e_ads_kcal_mol"])
    assert scan.preferred_site == expected_pref
    assert scan.is_top_preferred == (expected_pref == "top")


def test_chemisorption_vibrational_shift(co_ni100_deck, no_ni100_deck, repo_root):
    """
    Verify compute_chemisorption_vibrational_shift computes gas and adsorbed stretch
    frequencies directly from PES curvature and satisfies reduced-mass scaling.
    """
    driver = PhatpsyDriver(quiet=True)
    vib_co = compute_chemisorption_vibrational_shift(co_ni100_deck, driver=driver)

    assert isinstance(vib_co, VibrationalShiftResult)
    assert abs(vib_co.nu_gas_cm - 2169.24) > 5.0
    assert 1900.0 < vib_co.nu_gas_cm < 2600.0  # Uncorrelated SCF CO gas stretch oracle
    assert vib_co.delta_nu_cm == pytest.approx(vib_co.nu_ads_cm - vib_co.nu_gas_cm, abs=1e-6)
    assert vib_co.is_red_shifted == (vib_co.delta_nu_cm < 0.0)

    mu_co = (ATOMIC_MASSES_AMU["C"] * ATOMIC_MASSES_AMU["O"]) / (ATOMIC_MASSES_AMU["C"] + ATOMIC_MASSES_AMU["O"])
    expected_k_gas = ((vib_co.nu_gas_cm / HARTREE_TO_CM) ** 2) * (mu_co * AMU_TO_ME) * AU_FORCE_CONST_TO_MDYNE_A
    assert vib_co.force_constant_gas_mdyne_a == pytest.approx(expected_k_gas, rel=1e-4)

    # Also verify NO gas frequency comes from NO PES curvature rather than 1904.0 / 1621.0 literals
    res_no = driver.run_deck(no_ni100_deck)
    vib_no = compute_chemisorption_vibrational_shift(res_no, isolated_ads=repo_root / "examples" / "NO.yaml", driver=driver)
    assert abs(vib_no.nu_gas_cm - 1904.0) > 10.0
    assert abs(vib_no.nu_ads_cm - 1621.0) > 10.0
    assert 1800.0 < vib_no.nu_gas_cm < 2600.0


def test_surface_core_level_shifts(no_ni100_deck, repo_root):
    """
    Verify compute_surface_core_level_shifts extracts Koopmans core orbital eigenvalues
    matching extracted embedded-atom orbitals and independent PySCF gas STO-6G reference.
    """
    driver = PhatpsyDriver(quiet=True)
    res = driver.run_deck(no_ni100_deck)
    gas_no_deck = CalculationDeck.from_yaml(repo_root / "examples" / "NO.yaml")
    res_gas = driver.run_deck(gas_no_deck)

    scls = compute_surface_core_level_shifts(res, driver=driver, gas_state=res_gas)

    assert isinstance(scls, SurfaceCoreLevelShifts)
    assert scls.method_label == "koopmans_orbital_energy"
    assert scls.expected_relaxation_offset_ev == pytest.approx(15.0)
    assert scls.metal_scls_ev == pytest.approx(scls.surface_core_level_ev - scls.bulk_core_level_ev, abs=1e-6)

    # Verify old hardcoded dissertation/magic literals are gone
    assert abs(scls.adsorbate_n1s_be_ev - 415.5) > 0.5
    assert abs(scls.adsorbate_o1s_be_ev - 548.0) > 0.5
    assert abs(scls.metal_scls_ev - 0.38) > 0.05

    # Direct check against extracted N1/O1 1s eigenvalues in embedded_atoms
    n1_e = min(0.5 * (o["energy_alpha"] + o["energy_beta"]) for o in res.embedded_atoms["N1"].orbitals)
    o1_e = min(0.5 * (o["energy_alpha"] + o["energy_beta"]) for o in res.embedded_atoms["O1"].orbitals)
    assert scls.adsorbate_n1s_be_ev == pytest.approx(-n1_e * HARTREE_TO_EV, abs=1e-3)
    assert scls.adsorbate_o1s_be_ev == pytest.approx(-o1_e * HARTREE_TO_EV, abs=1e-3)

    # Independent PySCF UHF/STO-6G gas NO Koopmans oracle (within 5 eV for DZ vs STO-6G basis)
    mol = gto.M(atom="N 0 0 0; O 0 0 2.1747", basis="sto-6g", unit="Bohr", spin=1, verbose=0)
    mf = scf.UHF(mol).run()
    pyscf_o1s_ev = -0.5 * (mf.mo_energy[0][0] + mf.mo_energy[1][0]) * HARTREE_TO_EV
    pyscf_n1s_ev = -0.5 * (mf.mo_energy[0][1] + mf.mo_energy[1][1]) * HARTREE_TO_EV
    assert abs(scls.koopmans_gas_1s_be_ev["N1"] - pyscf_n1s_ev) < 5.0
    assert abs(scls.koopmans_gas_1s_be_ev["O1"] - pyscf_o1s_ev) < 5.0


def test_evaluate_no_ni100_benchmark():
    """
    Test the integrated benchmark routine for NO on Ni(100) computes live observables.
    (Quarantined in BASELINE_QUARANTINE while legacy default phatpsy_scf fails to converge.)
    """
    bench = evaluate_no_ni100_benchmark()
    assert bench["system"] == "NO_on_Ni100"
    assert bench["core_level_method"] == "koopmans_orbital_energy"
    assert abs(bench["adsorption_energy_ev"] - (-1.0)) > 0.1
    assert abs(bench["adsorption_energy_kcal_mol"] - (-23.06)) > 1.0
    assert abs(bench["vibrational_frequency_cm"] - 1621.0) > 10.0
    assert bench["converged"]


def test_co_ni100_chemisorption_and_c4v_dband(co_ni100_deck):
    """
    Test SCF convergence, virial invariant, C4v d-band projection, and
    non-circular adsorption energy identity on CO_ni_100.yaml.
    """
    driver = PhatpsyDriver(quiet=True)
    res = driver.run_deck(co_ni100_deck)
    assert res.converged
    assert abs(res.virial_ratio - 2.0) <= 0.05

    c4v = project_c4v_dband(res, atom_id="Ni1")
    assert c4v.total_d_population > 17.0
    assert c4v.crystal_field_splitting_ev >= 0.0
    assert c4v.sigma_donation_index >= 0.0
    assert c4v.pi_backdonation_index >= 0.0

    ads_co = evaluate_chemisorption_energy(res, driver=driver, site_name="top")
    assert ads_co.converged
    assert ads_co.energy_ads_hartree == pytest.approx(
        res.total_energy - ads_co.energy_clean_slab - ads_co.energy_gas_ads,
        abs=1e-9,
    )
    assert abs(ads_co.energy_ads_kcal_mol - (-30.0)) > 1.0
    assert abs(ads_co.adsorbate_net_charge) > 0.01


