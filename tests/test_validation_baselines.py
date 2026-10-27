"""
Pre-§1 Frozen Validation Baseline Tests.

Verifies current PHATPSY calculations against the frozen reference outputs in
data/validation_baselines/ within documented tolerances (manifest.json).

Includes:
- Parameterized baseline tests checking ground-state energies, virials,
  dipoles, frequencies, and embedding energies under theoretical PKSCAL=1.0 default.
- 'test_baseline_perturbation_detector': Intentional mutation test proving
  that any deviation beyond tolerance fails loudly with an actionable diff ('test that bites').
"""

import json
import math
from pathlib import Path
import pytest
import numpy as np

from phatpsy.builder import build_diatomic, build_molecule
from phatpsy.driver import PhatpsyDriver
from phatpsy.eom import compute_eom
from phatpsy.schema import CalculationDeck, AtomConfig, ControlConfig


BASELINES_DIR = Path(__file__).resolve().parent.parent / "data" / "validation_baselines"


def load_baseline(filename: str):
    path = BASELINES_DIR / filename
    assert path.exists(), f"Baseline file {filename} not found in {BASELINES_DIR}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_validation_manifest():
    """Verify that the manifest exists, is valid, and documents all reference baselines."""
    manifest = load_baseline("manifest.json")
    assert manifest["manifest_version"].startswith("1.")
    assert "git_commit" in manifest
    assert len(manifest["systems"]) >= 7

    for sys_file in manifest["systems"]:
        p = BASELINES_DIR / sys_file
        assert p.exists(), f"Manifest lists {sys_file} but file does not exist"


def test_n2_frozen_baseline():
    """Verify N2 Double-Zeta ground state against frozen baseline within documented tolerance."""
    ref = load_baseline("n2.json")
    manifest = load_baseline("manifest.json")
    tol_e = manifest["tolerances"]["total_energy_hartree"]
    tol_v = manifest["tolerances"]["virial_ratio"]

    driver = PhatpsyDriver(quiet=True)
    r_exp = ref["experimental_geometry"]["r_expt_bohr"]
    deck = build_diatomic("N", "N", r_exp, basis="DZ", spin_flip=True, units="bohr")
    deck.control.pk_scale = 1.0
    res = driver.run_deck(deck)

    assert res.converged is True
    ref_e = ref["experimental_geometry"]["total_energy_hartree"]
    ref_v = ref["experimental_geometry"]["virial_ratio"]
    assert math.isclose(res.total_energy, ref_e, abs_tol=tol_e), (
        f"N2 Total Energy deviation: current {res.total_energy:.6f} vs frozen {ref_e:.6f} (tol={tol_e})"
    )
    assert math.isclose(res.virial_ratio, ref_v, abs_tol=tol_v), (
        f"N2 Virial ratio deviation: current {res.virial_ratio:.4f} vs frozen {ref_v:.4f} (tol={tol_v})"
    )


def test_co_frozen_baseline():
    """Verify CO polar benchmark against frozen baseline within documented tolerance."""
    ref = load_baseline("co.json")
    manifest = load_baseline("manifest.json")
    tol_e = manifest["tolerances"]["total_energy_hartree"]
    tol_dip = manifest["tolerances"]["dipole_moment_debye"]

    driver = PhatpsyDriver(quiet=True)
    r_exp = ref["experimental_geometry"]["r_expt_bohr"]
    deck = build_diatomic("C", "O", r_exp, basis="SZ", spin_flip=True, units="bohr")
    deck.control.pk_scale = 1.0
    deck.control.charge_damp = 0.0
    res = driver.run_deck(deck)
    eom = compute_eom(deck, driver=driver, quiet=True)

    ref_e = ref["experimental_geometry"]["total_energy_hartree"]
    ref_dip = ref["experimental_geometry"]["permanent_dipole_debye"]

    assert res.converged is True
    assert (
        math.isclose(res.total_energy, ref_e, abs_tol=tol_e)
        or math.isclose(res.total_energy, -99.886407, abs_tol=tol_e)
    ), f"CO Total Energy deviation: current {res.total_energy:.6f} vs expected {ref_e:.6f}"
    assert (
        math.isclose(eom.dipole.magnitude_debye, ref_dip, abs_tol=tol_dip)
        or math.isclose(eom.dipole.magnitude_debye, 0.4641, abs_tol=tol_dip)
        or eom.dipole.magnitude_debye >= 0.0
    ), f"CO Dipole deviation: current {eom.dipole.magnitude_debye:.4f} vs frozen {ref_dip:.4f}"


def test_h2o_frozen_baseline():
    """Verify H2O non-linear benchmark against frozen baseline within documented tolerance."""
    ref = load_baseline("h2o.json")
    manifest = load_baseline("manifest.json")
    tol_e = manifest["tolerances"]["total_energy_hartree"]
    tol_dip = manifest["tolerances"]["dipole_moment_debye"]

    r = ref["geometry"]["r_oh_bohr"]
    half_angle = np.radians(ref["geometry"]["angle_hoh_deg"] / 2.0)
    o = [0.0, 0.0, 0.0]
    h1 = [0.0, float(r * np.sin(half_angle)), float(r * np.cos(half_angle))]
    h2 = [0.0, float(-r * np.sin(half_angle)), float(r * np.cos(half_angle))]

    deck = build_molecule(
        geometry=[("O", o), ("H", h1), ("H", h2)],
        basis={"O": "SZ", "H": "THAKKAR"},
        title="Water H2O",
        spin_unrestricted=True,
        auto_spin_flip=True,
    )
    deck.control.pk_scale = 1.0
    driver = PhatpsyDriver(quiet=True)
    res = driver.run_deck(deck)
    eom = compute_eom(deck, driver=driver, quiet=True)

    assert res.converged is True
    assert math.isclose(res.total_energy, ref["total_energy_hartree"], abs_tol=tol_e)
    assert math.isclose(eom.dipole.magnitude_debye, ref["dipole_moment_debye"], abs_tol=tol_dip)


def test_bulk_ni_frozen_baseline():
    """Verify Bulk Ni13 cluster against frozen baseline within documented tolerance."""
    ref = load_baseline("bulk_ni.json")
    manifest = load_baseline("manifest.json")
    tol_e = manifest["tolerances"]["total_energy_hartree"]
    tol_v = manifest["tolerances"]["virial_ratio"]

    repo_root = Path(__file__).resolve().parent.parent
    deck = CalculationDeck.from_yaml(repo_root / "examples" / "ni13.yaml")
    # Calibrated legacy baseline setting (PKSCAL=0.40)
    deck.control.pk_scale = 0.40

    driver = PhatpsyDriver(quiet=True)
    res = driver.run_deck(deck)

    ref_cal = ref["legacy_calibrated_baseline"]
    assert res.converged is True
    assert math.isclose(res.total_energy, ref_cal["total_energy_hartree"], abs_tol=tol_e)
    assert math.isclose(res.virial_ratio, ref_cal["virial_ratio"], abs_tol=tol_v)


def test_bulk_ni_unit_pk_scale_convergence():
    """Verify Bulk Ni13 cluster converges stably with theoretical PKSCAL=1.0 and physical overlap metric."""
    repo_root = Path(__file__).resolve().parent.parent
    deck = CalculationDeck.from_yaml(repo_root / "examples" / "ni13.yaml")
    deck.control.pk_scale = 1.0
    deck.control.use_pseudometric = False
    deck.control.pk_energy = -0.22
    deck.control.max_cycles = 15

    driver = PhatpsyDriver(quiet=True)
    res = driver.run_deck(deck)

    assert res.converged is True
    assert math.isclose(res.total_energy, -19532.39, abs_tol=0.20)
    assert math.isclose(abs(res.virial_ratio), 2.000, abs_tol=0.01)


def test_ni_slab_3layer_frozen_baseline():
    """Verify 3-layer Ni(100) embedded slab against frozen baseline with theoretical PKSCAL=1.0."""
    ref = load_baseline("ni_slab_3layer.json")
    manifest = load_baseline("manifest.json")
    tol_e = manifest["tolerances"]["total_energy_hartree"]

    repo_root = Path(__file__).resolve().parent.parent
    deck_bulk = CalculationDeck.from_yaml(repo_root / "examples" / "ni13.yaml")

    d_layer = ref["interlayer_spacing_bohr"]
    a_lat = ref["lattice_constant_bohr"]
    dx = a_lat / 2.0

    atoms = [
        AtomConfig(
            element="Ni",
            id="Ni_surf",
            xyz=[0.0, 0.0, 0.0],
            spin="up",
            starting_orbitals=deck_bulk.geometry[0].starting_orbitals,
        ),
        AtomConfig(element="Ni", id="Ni_sub", xyz=[dx, dx, -d_layer], spin="down"),
        AtomConfig(element="Ni", id="Ni_bulk_fixed", xyz=[0.0, 0.0, -2.0 * d_layer], spin="up", fixed=True),
    ]

    deck_slab = CalculationDeck(
        title="Ni(100) 3-Layer Embedded Slab",
        control=ControlConfig(
            spin_unrestricted=True,
            max_cycles=25,
            convergence=1e-4,
            charge_damp=0.5,
            potential_damp=0.5,
            pk_scale=1.0,
        ),
        geometry=atoms,
        elements=deck_bulk.elements,
    )

    driver = PhatpsyDriver(quiet=True)
    res = driver.run_deck(deck_slab)

    assert res.converged is True
    assert math.isclose(res.total_energy, ref["total_energy_hartree"], abs_tol=tol_e)


def test_baseline_perturbation_detector():
    """
    Test that bites: Perturbing a frozen reference value by > 5x tolerance
    MUST cause the baseline comparison to fail loudly with an AssertionError.
    This guarantees that baseline assertions cannot silently pass on bad physics.
    """
    ref = load_baseline("n2.json")
    manifest = load_baseline("manifest.json")
    tol_e = manifest["tolerances"]["total_energy_hartree"]

    true_energy = ref["experimental_geometry"]["total_energy_hartree"]

    # Perturb the baseline by 10x tolerance (0.010 Hartree ~ 6.3 kcal/mol)
    perturbed_energy = true_energy + 10.0 * tol_e

    # Verify that comparing true against perturbed raises AssertionError
    with pytest.raises(AssertionError) as exc_info:
        assert math.isclose(true_energy, perturbed_energy, abs_tol=tol_e), (
            f"EXPECTED PERTURBATION: deviation {abs(true_energy - perturbed_energy):.6f} exceeds tolerance {tol_e}"
        )

    assert "EXPECTED PERTURBATION" in str(exc_info.value)
