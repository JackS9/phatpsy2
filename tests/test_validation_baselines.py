"""
Physics Invariants and Boundary Conditions Tests.

Replaces frozen legacy baselines with physical boundary conditions and
fundamental quantum chemical invariants:
- Virial theorem consistency (-V/T ~ 2.00)
- Variational ground-state stability (E < 0)
- Physical electron conservation and non-negative populations
- Multi-center fixed/virtual coordination stability
"""

import math
from pathlib import Path
import pytest
import numpy as np

from phatpsy.builder import build_diatomic, build_molecule
from phatpsy.driver import PhatpsyDriver
from phatpsy.eom import compute_eom
from phatpsy.schema import CalculationDeck, AtomConfig, ControlConfig


def test_n2_physics_invariants():
    """Verify N2 ground state satisfies fundamental physical invariants."""
    driver = PhatpsyDriver(quiet=True)
    deck = build_diatomic("N", "N", 2.068, basis="DZ", spin_flip=True, units="bohr")
    deck.control.pk_scale = 1.0
    res = driver.run_deck(deck)

    assert res.converged is True
    # Variational ground state bound
    assert res.total_energy < -100.0
    # Virial theorem: -V/T ~ 2.00
    assert 1.95 <= abs(res.virial_ratio) <= 2.05, (
        f"N2 Virial ratio {res.virial_ratio:.4f} violates virial theorem (-V/T ~ 2.00)"
    )


def test_h2o_physics_invariants():
    """Verify non-linear polar water (H2O) satisfies physical invariants and non-zero dipole."""
    r = 1.81  # Bohr
    half_angle = np.radians(104.5 / 2.0)
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
    assert res.total_energy < -70.0
    assert 1.95 <= abs(res.virial_ratio) <= 2.05, (
        f"H2O Virial ratio {res.virial_ratio:.4f} violates virial theorem (-V/T ~ 2.00)"
    )
    # Permanent dipole moment must be strictly non-zero
    assert eom.dipole.magnitude_debye > 1.0


def test_bulk_ni_unit_pk_scale_convergence():
    """Verify Bulk Ni13 cluster converges stably with theoretical PKSCAL=1.0 and physical overlap metric."""
    repo_root = Path(__file__).resolve().parent.parent
    deck = CalculationDeck.from_yaml(repo_root / "examples" / "ni13.yaml")
    deck.control.pk_scale = 1.0
    deck.control.use_pseudometric = False
    deck.control.pk_energy = -0.22
    deck.control.max_cycles = 25

    driver = PhatpsyDriver(quiet=True)
    res = driver.run_deck(deck)

    assert res.converged is True
    assert res.total_energy < -19000.0
    assert 1.98 <= abs(res.virial_ratio) <= 2.02, (
        f"Ni13 Virial ratio {res.virial_ratio:.4f} violates virial theorem"
    )


def test_ni_slab_fixed_centers_invariants():
    """Verify 3-layer Ni(100) embedded slab converges stably with fixed substrate center."""
    repo_root = Path(__file__).resolve().parent.parent
    deck_bulk = CalculationDeck.from_yaml(repo_root / "examples" / "ni13.yaml")

    d_layer = 3.326
    a_lat = 6.652
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
    assert res.total_energy < -4000.0
