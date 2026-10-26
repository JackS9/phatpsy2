# tests/test_subsystem_energy.py
"""
Tests for Phase 1: Embedded Subsystem Energy Formalism & Granular Cycle Diagnostics.

Verifies:
1. Exact subsystem energy conservation: E_total == E_cluster + E_env + E_embed
2. Physical bulk cohesive reference for frozen atoms (-1502.5650 Ha for Ni)
3. Granular per-cycle diagnostic tracking (kinetic energy, virial ratio, PK penalty, RMS residuals)
4. Integration with CalculationState properties and ONIOM embedded slab models
"""

import copy
import pytest
import numpy as np

from phatpsy.builder import build_diatomic
from phatpsy.driver import PhatpsyDriver
from phatpsy.schema import AtomConfig, CalculationDeck, ControlConfig
from phatpsy.subsystem import (
    decompose_subsystem_energies,
    get_isolated_atomic_energy,
    get_bulk_cohesive_energy,
    SubsystemDecomposition,
    BULK_COHESIVE_ENERGIES,
    ISOLATED_ATOMIC_ENERGIES,
)


@pytest.fixture(scope="module")
def driver():
    return PhatpsyDriver(quiet=True)


def test_reference_energies():
    """Verify standard reference energy lookup for isolated atoms and bulk cohesive states."""
    # Isolated atoms
    e_h = get_isolated_atomic_energy("H")
    assert abs(e_h - (-0.5)) < 1e-12

    e_n = get_isolated_atomic_energy("N")
    assert abs(e_n - (-54.397951)) < 1e-5

    e_ni_iso = get_isolated_atomic_energy("Ni")
    assert abs(e_ni_iso - (-1506.7517)) < 1e-3

    # Bulk references
    e_ni_bulk = get_bulk_cohesive_energy("Ni")
    assert abs(e_ni_bulk - (-1502.5650005333994)) < 1e-10


def test_subsystem_energy_conservation_pure_cluster(driver):
    """
    Verify exact subsystem energy decomposition on a pure active cluster (N2).
    E_total = E_cluster + E_env + E_embed, with E_env = 0.
    """
    deck = build_diatomic("N", "N", 2.074, basis="SZ", spin_flip=True, units="bohr")
    deck.control.max_cycles = 15
    deck.control.convergence = 1e-5

    res = driver.run_deck(deck)
    assert res.converged
    assert res.total_energy < -100.0

    # Subsystem decomposition via method and properties
    decomp = res.decompose_subsystems()
    assert isinstance(decomp, SubsystemDecomposition)
    assert len(decomp.active_atoms) == 2
    assert len(decomp.frozen_atoms) == 0
    assert len(decomp.virtual_atoms) == 0

    # Exact conservation: E_total == E_cluster + E_env + E_embed
    assert abs(res.total_energy - (decomp.energy_cluster + decomp.energy_environment + decomp.energy_embedding)) < 1e-12
    assert decomp.energy_environment == 0.0
    assert abs(decomp.energy_excess - res.total_energy) < 1e-12

    # Verify CalculationState property accessors
    assert abs(res.energy_cluster - decomp.energy_cluster) < 1e-12
    assert abs(res.energy_environment - 0.0) < 1e-12
    assert abs(res.energy_embedding - decomp.energy_embedding) < 1e-12
    assert abs(res.energy_excess - res.total_energy) < 1e-12

    # Isolated N atom is -54.397951 Ha; two atoms = -108.795902 Ha
    assert abs(res.energy_cluster - (2.0 * -54.397951)) < 1e-4

    # Binding/embedding energy should be negative (stable diatomic)
    assert res.energy_embedding < 0.0


def test_subsystem_energy_conservation_embedded_slab(driver):
    """
    Verify ONIOM / DFT-in-DFT exact energy partitioning on an embedded 3-layer slab
    with active surface/subsurface atoms and frozen bulk environment atoms.
    """
    # 3-layer embedded slab:
    # L1: Ni_surf (active)
    # L2: Ni_sub (active)
    # L3: Ni_bulk_fixed (frozen)
    d_layer = 3.326
    a_lat = 6.652
    dx = a_lat / 2.0

    slab_geometry = [
        AtomConfig(element="Ni", id="Ni_surf", xyz=[0.0, 0.0, 0.0], spin="up"),
        AtomConfig(element="Ni", id="Ni_sub", xyz=[dx, dx, -d_layer], spin="down"),
        AtomConfig(element="Ni", id="Ni_bulk_fixed", xyz=[0.0, 0.0, -2.0 * d_layer], spin="up", fixed=True),
    ]

    from phatpsy.builder import build_molecule
    deck_slab = build_molecule(
        geometry=slab_geometry,
        basis="SZ",
        title="3-Layer Ni(100) Embedded Slab",
        control=ControlConfig(
            spin_unrestricted=True,
            max_cycles=15,
            convergence=1e-4,
            charge_damp=0.5,
            potential_damp=0.5,
            pk_scale=1.0,
        ),
    )

    res = driver.run_deck(deck_slab)
    assert res.converged

    decomp = res.subsystem_decomposition
    assert decomp.active_atoms == ["Ni_surf", "Ni_sub"]
    assert decomp.frozen_atoms == ["Ni_bulk_fixed"]
    assert len(decomp.virtual_atoms) == 0

    # Bulk reference for single fixed Ni atom
    e_bulk_ref = -1502.5650005333994
    assert abs(decomp.energy_environment - e_bulk_ref) < 1e-10

    # Exact conservation: E_total == E_cluster + E_env + E_embed
    assert abs(res.total_energy - (decomp.energy_cluster + decomp.energy_environment + decomp.energy_embedding)) < 1e-12

    # Excess energy of active cluster embedded in bulk: E_excess = E_total - E_env
    assert abs(decomp.energy_excess - (res.total_energy - e_bulk_ref)) < 1e-12
    assert abs(res.energy_excess - decomp.energy_excess) < 1e-12


def test_ghost_virtual_atoms_in_subsystem():
    """
    Verify virtual ghost atoms are correctly isolated and do not add to cluster or env reference energies.
    """
    geom = [
        AtomConfig(element="Ni", id="Ni1", xyz=[0.0, 0.0, 0.0], fixed=False),
        AtomConfig(element="Ni", id="Ni_bulk", xyz=[0.0, 0.0, -3.0], fixed=True),
        AtomConfig(element="Ni", id="Ni_ghost", xyz=[0.0, 0.0, 3.0], virtual=True),
    ]
    deck = CalculationDeck(title="Virtual Ghost Test", geometry=geom)

    decomp = decompose_subsystem_energies(deck, total_energy=-3000.0)
    assert decomp.active_atoms == ["Ni1"]
    assert decomp.frozen_atoms == ["Ni_bulk"]
    assert decomp.virtual_atoms == ["Ni_ghost"]

    assert abs(decomp.energy_environment - (-1502.5650005333994)) < 1e-10
    assert abs(decomp.energy_cluster - (-1506.751700)) < 1e-5
    assert abs(decomp.total_energy - (-3000.0)) < 1e-12
    assert abs(decomp.total_energy - (decomp.energy_cluster + decomp.energy_environment + decomp.energy_embedding)) < 1e-12


def test_cycle_history_and_diagnostics(driver):
    """
    Verify fine-grained per-cycle logging and residual diagnostics:
    - Kinetic energy <T> and virial ratio -V/T
    - Model potential fitting RMS deviation (Coulomb and exchange residuals)
    - Projection energy penalty dE_PK
    - Charge transfer delta q
    """
    deck = build_diatomic("C", "O", 2.132, basis="SZ", spin_flip=True, units="bohr")
    deck.control.max_cycles = 10
    deck.control.convergence = 1e-5

    res = driver.run_deck(deck)
    assert res.converged

    # Verify cycle_history is captured
    history = res.cycle_history
    assert isinstance(history, list)
    assert len(history) >= 2  # Completed multiple cycles

    for cycle_info in history:
        assert "cycle" in cycle_info
        assert "total_energy" in cycle_info
        assert "kinetic_energy" in cycle_info
        assert "potential_energy" in cycle_info
        assert "virial_ratio" in cycle_info
        assert "pk_penalty" in cycle_info
        assert "rms_coulomb" in cycle_info
        assert "rms_exchange" in cycle_info
        assert "charge_transfer" in cycle_info

        # Physical sanity checks
        assert cycle_info["kinetic_energy"] > 0.0
        assert isinstance(cycle_info["pk_penalty"], float)
        assert isinstance(cycle_info["rms_coulomb"], float)
        assert cycle_info["rms_coulomb"] >= 0.0

    # Final cycle matches state and satisfies physical virial theorem (-V/T ~ 2.0)
    last_cycle = history[-1]
    assert abs(last_cycle["total_energy"] - res.total_energy) < 1e-6
    assert abs(last_cycle["virial_ratio"] - res.virial_ratio) < 1e-6
    assert 1.8 < last_cycle["virial_ratio"] < 2.2
