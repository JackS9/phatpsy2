# tests/test_state_and_restart.py
"""
Tests for PHATPSY State Persistence, Checkpointing, Warm Restarts,
and Transferable Embedded Atomic States (Roadmap Section 6).
"""
import pytest
from pathlib import Path
import json
import numpy as np

from phatpsy.driver import PhatpsyDriver
from phatpsy.state import CalculationState, EmbeddedAtomState
from phatpsy.schema import CalculationDeck, AtomConfig
from phatpsy.builder import build_molecule

EXAMPLE_N2 = Path(__file__).parent.parent / "examples" / "n2.stdin"


def test_calculation_state_extraction_and_parity():
    """Verify that calculation state extracts full observables with dict/object parity."""
    driver = PhatpsyDriver()
    state = driver.run_file(EXAMPLE_N2)

    assert isinstance(state, CalculationState)
    assert state.converged is True
    assert state["converged"] is True

    # Parity between dict key access and property access
    assert state.total_energy == state["total_energy"]
    assert state.kinetic_energy == state["kinetic_energy"]
    assert state.potential_energy == state["potential_energy"]
    assert state.nuclear_repulsion == state["nuclear_repulsion"]
    assert state.virial_ratio == state["virial_ratio"]

    # Physical invariant: Virial ratio (-V/T) ~ 2.0
    assert abs(state.virial_ratio - 2.012) < 0.05
    assert abs(state.total_energy - (-109.5656)) < 1e-2

    # Dimensions
    assert state.norbt == 10
    assert state.mstot == 10
    assert state.ispin == 2
    assert state.eigenvalues.shape == (10, 2)
    assert state.mo_coefficients.shape == (10, 10, 2)
    assert state.atom_charges.shape == (3, 2)
    assert state.atom_coordinates.shape == (3, 2)

    # Coordinates in Bohrs
    np.testing.assert_allclose(state.atom_coordinates[:, 0], [0.0, 0.0, 1.034], atol=1e-5)
    np.testing.assert_allclose(state.atom_coordinates[:, 1], [0.0, 0.0, -1.034], atol=1e-5)


def test_embedded_atom_state_extraction():
    """Verify localization and extraction of EmbeddedAtomState for individual atoms."""
    driver = PhatpsyDriver()
    state = driver.run_file(EXAMPLE_N2)

    assert "embedded_atoms" in state
    assert len(state.embedded_atoms) == 2

    # Retrieve by index and by ID
    atom1 = state.get_embedded_atom(0)
    atom2 = state.get_embedded_atom(1)

    assert isinstance(atom1, EmbeddedAtomState)
    assert atom1.element in ("N", "X")
    assert atom1.atomic_number == 7

    # Charges and spin analysis
    assert abs(atom1.nuclear_charge - 7.0) < 1e-5
    assert abs(atom1.electronic_charge - 7.0) < 1e-3
    assert abs(atom1.net_charge) < 1e-3

    # Localized spin polarization: alpha ~ 4.208, beta ~ 2.792, net ~ 1.416
    assert abs(atom1.net_spin - 1.4159) < 0.05
    assert abs(atom1.alpha_spin - 4.208) < 0.05
    assert abs(atom1.beta_spin - 2.792) < 0.05

    # Partner atom 2 has opposite spin polarization
    assert abs(atom2.net_spin - (-1.4159)) < 0.05

    # Orbitals and basis functions
    assert len(atom1.basis_functions) == 3  # 1s, 2s, 2p
    assert len(atom1.orbitals) == 5  # 5 orbitals for N SZ

    orb1 = atom1.orbitals[0]
    assert orb1["orbital_index"] == 1
    assert abs(orb1["energy_alpha"] - (-15.490)) < 0.05
    assert len(orb1["coefficients"]) > 0

    # Initial orbitals conversion
    init_orbs = atom1.to_initial_orbitals()
    assert len(init_orbs) == 5
    assert len(init_orbs[0].eigval) == 2
    assert len(init_orbs[0].coefficients) > 0


def test_state_json_checkpointing(tmp_path):
    """Test full CalculationState serialization and deserialization to JSON."""
    driver = PhatpsyDriver()
    state = driver.run_file(EXAMPLE_N2)

    json_checkpoint = tmp_path / "n2_checkpoint.json"
    state.save(json_checkpoint)
    assert json_checkpoint.exists()

    loaded = CalculationState.load(json_checkpoint)

    assert loaded.converged == state.converged
    assert loaded.cycles == state.cycles
    assert abs(loaded.total_energy - state.total_energy) < 1e-10
    assert abs(loaded.kinetic_energy - state.kinetic_energy) < 1e-10
    assert abs(loaded.virial_ratio - state.virial_ratio) < 1e-10

    # Verify NumPy arrays round-trip
    np.testing.assert_allclose(loaded.eigenvalues, state.eigenvalues)
    np.testing.assert_allclose(loaded.mo_coefficients, state.mo_coefficients)
    np.testing.assert_allclose(loaded.overlap_packed, state.overlap_packed)
    np.testing.assert_allclose(loaded.atom_charges, state.atom_charges)

    # Verify embedded atoms round-trip
    assert len(loaded.embedded_atoms) == len(state.embedded_atoms)
    l_atom = loaded.get_embedded_atom(0)
    assert l_atom.atomic_number == 7
    assert abs(l_atom.net_spin - state.get_embedded_atom(0).net_spin) < 1e-6


def test_state_npz_checkpointing(tmp_path):
    """Test full CalculationState serialization and deserialization to compressed NPZ."""
    driver = PhatpsyDriver()
    state = driver.run_file(EXAMPLE_N2)

    npz_checkpoint = tmp_path / "n2_checkpoint.npz"
    state.save(npz_checkpoint)
    assert npz_checkpoint.exists()

    loaded = CalculationState.load(npz_checkpoint)

    assert loaded.converged == state.converged
    assert loaded.cycles == state.cycles
    assert abs(loaded.total_energy - state.total_energy) < 1e-10
    assert abs(loaded.virial_ratio - state.virial_ratio) < 1e-10

    np.testing.assert_allclose(loaded.eigenvalues, state.eigenvalues)
    np.testing.assert_allclose(loaded.mo_coefficients, state.mo_coefficients)
    np.testing.assert_allclose(loaded.atom_coordinates, state.atom_coordinates)


def test_embedded_atom_save_and_load(tmp_path):
    """Test individual EmbeddedAtomState persistence to JSON and NPZ."""
    driver = PhatpsyDriver()
    state = driver.run_file(EXAMPLE_N2)
    n1 = state.get_embedded_atom(0)

    # Save to JSON
    atom_json = tmp_path / "n1_embedded.json"
    n1.save(atom_json)
    assert atom_json.exists()

    loaded_json = EmbeddedAtomState.load(atom_json)
    assert loaded_json.atomic_number == 7
    assert abs(loaded_json.net_spin - n1.net_spin) < 1e-6
    assert len(loaded_json.orbitals) == 5

    # Save to NPZ
    atom_npz = tmp_path / "n1_embedded.npz"
    n1.save(atom_npz)
    assert atom_npz.exists()

    loaded_npz = EmbeddedAtomState.load(atom_npz)
    assert loaded_npz.atomic_number == 7
    assert abs(loaded_npz.net_spin - n1.net_spin) < 1e-6
    assert len(loaded_npz.basis_functions) == 3


def test_warm_restart_on_modified_geometry():
    """
    Test warm restart on modified geometry (PES scan / geometry change).
    Starts from converged equilibrium state and steps to stretched bond length.
    """
    driver = PhatpsyDriver()

    # 1. Equilibrium calculation (z = +/- 1.034 Bohr)
    deck_eq = build_molecule(
        [("N", [0.0, 0.0, 1.034]), ("N", [0.0, 0.0, -1.034])],
        basis="SZ",
        units="bohr",
    )
    state_eq = driver.run_deck(deck_eq)
    assert state_eq.converged is True

    # 2. Warm restart at stretched bond (z = +/- 1.150 Bohr) using driver.restart()
    stretched_geom = [("N", [0.0, 0.0, 1.150]), ("N", [0.0, 0.0, -1.150])]
    state_stretched = driver.restart(
        state_eq,
        new_geometry=stretched_geom,
        control_overrides={"max_cycles": 100, "convergence": 1e-4},
    )

    assert state_stretched.converged is True
    # At stretched bond, total energy is slightly higher (less negative) than equilibrium
    assert state_stretched.total_energy > state_eq.total_energy
    # Virial theorem still satisfied
    assert abs(state_stretched.virial_ratio - 2.0) < 0.05


def test_override_reading_starting_orbitals_from_embedded_atom():
    """
    Roadmap Section 6.3: Override reading starting orbitals from basis set.
    Enable direct orbital ingestion from pre-converged embedded atom states
    rather than re-evaluating atomic STO basis tables.
    """
    driver = PhatpsyDriver()

    # 1. Run reference calculation to obtain converged embedded atom state
    deck_ref = build_molecule(
        [("N", [0.0, 0.0, 1.034]), ("N", [0.0, 0.0, -1.034])],
        basis="SZ",
        units="bohr",
    )
    state_ref = driver.run_deck(deck_ref)
    n1_embedded = state_ref.get_embedded_atom(0)

    # 2. Construct a new deck, overriding starting orbitals using n1_embedded
    new_deck = build_molecule(
        [("N", [0.0, 0.0, 1.050]), ("N", [0.0, 0.0, -1.050])],
        basis="SZ",
        units="bohr",
        embedded_atoms={"N1": n1_embedded},
    )

    # Verify that the generated stdin text contains the overridden embedded orbital energies
    stdin_text = new_deck.to_stdin_text()
    assert "&ATORB" in stdin_text

    # The embedded orbital 1 has alpha eigenvalue ~ -15.5122
    # Default basis library has -15.6704
    assert f"EIGVAL={n1_embedded.orbitals[0]['energy_alpha']:.4f}" in stdin_text
    assert "NCOEF=" in stdin_text

    # 3. Execute calculation with overridden starting orbitals
    state_override = driver.run_deck(new_deck)
    assert state_override.converged is True
    assert abs(state_override.virial_ratio - 2.0) < 0.05


def test_create_restart_deck_with_title():
    """Verify that CalculationState.create_restart_deck supports the title keyword argument."""
    driver = PhatpsyDriver()
    state = driver.run_file(EXAMPLE_N2)

    restart_deck = state.create_restart_deck(
        new_geometry=[("N", 0.0, 0.0, 1.05), ("N", 0.0, 0.0, -1.05)],
        title="N2 Warm Restart (R = 2.10 Bohr)",
    )
    assert restart_deck.title == "N2 Warm Restart (R = 2.10 Bohr)"
    assert len(restart_deck.geometry) == 2
