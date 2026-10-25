# phatpsy/__init__.py
"""
PHATPSY: A Projected Hamiltonian Approach to Polyatomic Systems.
Modernized Python API and driver.
"""

from phatpsy.driver import PhatpsyDriver, silence_stdout
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
from phatpsy.visualizer import (
    visualize_deck,
    show_deck,
    visualize_grid,
    show_grid,
    visualize_pathway,
    show_pathway,
    visualize_trajectory,
    show_trajectory,
)
from phatpsy.volumetric import (
    VolumetricGrid,
    compute_grid,
    compute_esp_grid,
    compute_deformation_density,
)

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
from phatpsy.neb import (
    neb_search,
    interpolate_images,
    calculate_tangents,
    calculate_neb_forces,
    NEBResult,
    NEBImage,
)
from phatpsy.md import (
    run_md,
    MDResult,
    MDStep,
    init_velocities,
    get_atomic_mass_au,
)
from phatpsy.vibrations import (
    compute_vibrations,
    compute_hessian,
    compute_thermochemistry,
    VibrationResult,
    NormalMode,
    Thermochemistry,
)
from phatpsy.irc import (
    compute_irc,
    IRCResult,
    IRCPoint,
)
from phatpsy.dimer import (
    MullerBrownSurface,
    DimerStep,
    DimerResult,
    dimer_optimize_vector,
    dimer_search,
    refine_neb_saddle,
)
from phatpsy.eom import (
    compute_eom,
    EOMResult,
    ElectronicTransition,
    DipoleMoment,
    PolarizabilityTensor,
    MagneticSusceptibility,
)
from phatpsy.visualizer import (
    visualize_vibrations,
    show_vibrations,
    visualize_irc,
    show_irc,
    visualize_dimer,
    show_dimer,
    visualize_eom,
    show_eom,
    generate_eom_html,
    generate_eom_visualization_data,
)

__all__ = [
    "PhatpsyDriver",
    "silence_stdout",
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
    "visualize_grid",
    "show_grid",
    "visualize_pathway",
    "show_pathway",
    "VolumetricGrid",
    "compute_grid",
    "compute_esp_grid",
    "compute_deformation_density",

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
    "neb_search",
    "interpolate_images",
    "calculate_tangents",
    "calculate_neb_forces",
    "NEBResult",
    "NEBImage",
    "run_md",
    "MDResult",
    "MDStep",
    "init_velocities",
    "get_atomic_mass_au",
    "visualize_trajectory",
    "show_trajectory",
    "compute_vibrations",
    "compute_hessian",
    "compute_thermochemistry",
    "VibrationResult",
    "NormalMode",
    "Thermochemistry",
    "visualize_vibrations",
    "show_vibrations",
    "compute_irc",
    "IRCResult",
    "IRCPoint",
    "visualize_irc",
    "show_irc",
    "MullerBrownSurface",
    "DimerStep",
    "DimerResult",
    "dimer_optimize_vector",
    "dimer_search",
    "refine_neb_saddle",
    "visualize_dimer",
    "show_dimer",
    "compute_eom",
    "EOMResult",
    "ElectronicTransition",
    "DipoleMoment",
    "PolarizabilityTensor",
    "MagneticSusceptibility",
    "visualize_eom",
    "show_eom",
    "generate_eom_html",
    "generate_eom_visualization_data",
]


