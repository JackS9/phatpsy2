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
from phatpsy.state import CalculationState, EmbeddedAtomState
from phatpsy.coords import (
    BondCoordinate,
    AngleCoordinate,
    DihedralCoordinate,
    InternalCoordinates,
    ZMatrix,
    ZMatrixEntry,
)
from phatpsy.pes import (
    scan_1d,
    scan_2d,
    PESScan1DResult,
    PESScan2DResult,
)
from phatpsy.optimize import (
    optimize_geometry,
    OptimizationResult,
    OptimizationStep,
)
from phatpsy.supercell import (
    Lattice,
    PeriodicAtom,
    PeriodicCell,
    build_slab,
    attach_adsorbate,
)

__all__ = [
    "PhatpsyDriver",
    "CalculationState",
    "EmbeddedAtomState",
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
    "BondCoordinate",
    "AngleCoordinate",
    "DihedralCoordinate",
    "InternalCoordinates",
    "ZMatrix",
    "ZMatrixEntry",
    "scan_1d",
    "scan_2d",
    "PESScan1DResult",
    "PESScan2DResult",
    "optimize_geometry",
    "OptimizationResult",
    "OptimizationStep",
    "Lattice",
    "PeriodicAtom",
    "PeriodicCell",
    "build_slab",
    "attach_adsorbate",
]

