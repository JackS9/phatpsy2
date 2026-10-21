# phatpsy/__init__.py
"""
PHATPSY: A Projected Hamiltonian Approach to Polyatomic Systems.
Modernized Python API and driver.
"""

from phatpsy.driver import PhatpsyDriver
from phatpsy.diis import DIISAccelerator
from phatpsy.ewmo import solve_ewmo_propagator, unpack_symmetric
from phatpsy.io import (
    parse_stdin_file,
    parse_stdin_to_deck,
    generate_diatomic_deck,
    save_results_to_json,
)
from phatpsy.schema import (
    CalculationDeck,
    ControlConfig,
    AtomConfig,
    ElementConfig,
    SymmetrySpec,
)
from phatpsy.builder import (
    build_molecule,
    build_diatomic,
    parse_geometry,
    BOHR_PER_ANGSTROM,
)
from phatpsy.basis import get_element_basis, list_available_elements
from phatpsy.spin import assign_alternating_spin
from phatpsy.visualizer import visualize_deck, show_deck

__all__ = [
    "PhatpsyDriver",
    "DIISAccelerator",
    "solve_ewmo_propagator",
    "unpack_symmetric",
    "parse_stdin_file",
    "parse_stdin_to_deck",
    "generate_diatomic_deck",
    "save_results_to_json",
    "CalculationDeck",
    "ControlConfig",
    "AtomConfig",
    "ElementConfig",
    "SymmetrySpec",
    "get_element_basis",
    "list_available_elements",
    "build_molecule",
    "build_diatomic",
    "parse_geometry",
    "BOHR_PER_ANGSTROM",
    "assign_alternating_spin",
    "visualize_deck",
    "show_deck",
]
