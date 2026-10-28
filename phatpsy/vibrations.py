# phatpsy/vibrations.py
"""
Harmonic vibrational frequency analysis, normal mode displacements, and
Rigid-Rotor Harmonic Oscillator (RRHO) ideal-gas thermochemistry for PHATPSY 2.0.

Key capabilities:
- Numerical Cartesian Hessian matrix evaluation (central and forward finite differences).
- Mass-weighted Hessian and Eckart translational & rotational projection.
- Harmonic vibrational frequencies (cm^-1), reduced masses (amu), force constants (mDyne/Å).
- Normal mode displacement vectors with automatic saddle-point / transition state classification.
- Full statistical thermochemistry:
    - Zero-Point Vibrational Energy (ZPVE)
    - Thermal internal energy U(T), enthalpy H(T), entropy S(T) (Sackur-Tetrode + RRHO),
      and Gibbs free energy G(T) = H(T) - T * S(T)
    - Constant-volume heat capacity C_v(T)
- Interactive 3D Three.js WebGL visualization and mode animation.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
import copy
import json
import math
import numpy as np

from phatpsy.schema import CalculationDeck, AtomConfig
from phatpsy.driver import PhatpsyDriver
from phatpsy.state import CalculationState
from phatpsy.md import ATOMIC_MASSES_AMU, HARTREE_TO_KCAL, BOHR_TO_ANGSTROM


# ==============================================================================
# Fundamental Physical & Conversion Constants (CODATA / IUPAC)
# ==============================================================================

# Atomic units to SI & CGS
HARTREE_TO_JOULE: float = 4.3597447222071e-18
BOHR_TO_METER: float = 0.529177210903e-10
BOHR_TO_ANGSTROM_VAL: float = BOHR_TO_ANGSTROM
AMU_TO_KG: float = 1.66053906660e-27
SPEED_OF_LIGHT_CM_S: float = 2.99792458e10
PLANCK_CONSTANT_J_S: float = 6.62607015e-34
HBAR_J_S: float = PLANCK_CONSTANT_J_S / (2.0 * math.pi)
BOLTZMANN_CONSTANT_J_K: float = 1.380649e-23
AVOGADRO_CONSTANT: float = 6.02214076e23
GAS_CONSTANT_R_J_MOL_K: float = 8.31446261815324
GAS_CONSTANT_R_CAL_MOL_K: float = 1.98720425864083
STANDARD_PRESSURE_PA: float = 101325.0  # 1 atm in Pascals

# Frequency conversion: sqrt(Hartree / (Bohr^2 * amu)) -> cm^-1
# 1 / (2 * pi * c) * sqrt(E_h / (a_0^2 * amu)) ≈ 5140.4871437 cm^-1
FREQ_CONV_FACTOR: float = (
    1.0 / (2.0 * math.pi * SPEED_OF_LIGHT_CM_S)
) * math.sqrt(HARTREE_TO_JOULE / (BOHR_TO_METER**2 * AMU_TO_KG))

# Force constant conversion: Hartree / Bohr^2 -> mDyne / Å
# 1 Hartree / Bohr^2 ≈ 1556.8931 N/m = 15.568931 mDyne / Å
FORCE_CONSTANT_CONV: float = (HARTREE_TO_JOULE / (BOHR_TO_METER**2)) * 1e-2


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class NormalMode:
    """
    Representation of a single molecular normal vibrational mode.
    """
    index: int
    frequency_cm: float
    is_imaginary: bool
    reduced_mass_amu: float
    force_constant_mdyne_a: float
    displacements: List[List[float]]  # (N_atoms, 3) normalized Cartesian displacements
    is_tr_mode: bool = False
    symmetry_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert normal mode to dictionary."""
        return {
            "index": int(self.index),
            "frequency_cm": round(float(self.frequency_cm), 2),
            "is_imaginary": bool(self.is_imaginary),
            "reduced_mass_amu": round(float(self.reduced_mass_amu), 4),
            "force_constant_mdyne_a": round(float(self.force_constant_mdyne_a), 4),
            "displacements": [[round(float(c), 6) for c in atom_disp] for atom_disp in self.displacements],
            "is_tr_mode": bool(self.is_tr_mode),
            "symmetry_label": self.symmetry_label,
        }


@dataclass
class Thermochemistry:
    """
    Standard ideal gas / Rigid-Rotor Harmonic Oscillator (RRHO) thermochemical analysis.
    """
    temperature_k: float
    pressure_atm: float
    electronic_energy: float              # Hartree
    zero_point_energy_kcal: float         # kcal/mol
    zero_point_energy_au: float           # Hartree
    thermal_energy_kcal: float            # kcal/mol (U_therm = E_trans + E_rot + E_vib)
    thermal_enthalpy_kcal: float          # kcal/mol (H_therm = U_therm + RT)
    entropy_cal_mol_k: Dict[str, float]   # {'translation', 'rotation', 'vibration', 'electronic', 'total'}
    heat_capacity_cal_mol_k: Dict[str, float]  # {'cv_trans', 'cv_rot', 'cv_vib', 'cv_total'}
    total_energy_u0: float                # Hartree (E_elec + ZPVE)
    total_energy_u_t: float               # Hartree (E_elec + U_therm)
    total_enthalpy_h_t: float             # Hartree (E_elec + H_therm)
    total_gibbs_free_energy_g_t: float    # Hartree (H(T) - T * S_total)
    thermal_energy_components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert thermochemistry to dictionary."""
        return {
            "temperature_k": round(float(self.temperature_k), 2),
            "pressure_atm": round(float(self.pressure_atm), 3),
            "electronic_energy": round(float(self.electronic_energy), 8),
            "zero_point_energy_kcal": round(float(self.zero_point_energy_kcal), 4),
            "zero_point_energy_au": round(float(self.zero_point_energy_au), 8),
            "thermal_energy_kcal": round(float(self.thermal_energy_kcal), 4),
            "thermal_enthalpy_kcal": round(float(self.thermal_enthalpy_kcal), 4),
            "thermal_energy_components": {k: round(float(v), 4) for k, v in self.thermal_energy_components.items()},
            "entropy_cal_mol_k": {k: round(float(v), 3) for k, v in self.entropy_cal_mol_k.items()},
            "heat_capacity_cal_mol_k": {k: round(float(v), 3) for k, v in self.heat_capacity_cal_mol_k.items()},
            "total_energy_u0": round(float(self.total_energy_u0), 8),
            "total_energy_u_t": round(float(self.total_energy_u_t), 8),
            "total_enthalpy_h_t": round(float(self.total_enthalpy_h_t), 8),
            "total_gibbs_free_energy_g_t": round(float(self.total_gibbs_free_energy_g_t), 8),
        }

    def summary(self) -> str:
        """Formatted thermochemical summary report."""
        lines = [
            "=" * 72,
            " PHATPSY 2.0 THERMOCHEMICAL ANALYSIS (RRHO / IDEAL GAS)",
            "=" * 72,
            f" Temperature:                        {self.temperature_k:12.2f} K",
            f" Pressure:                           {self.pressure_atm:12.3f} atm",
            f" Electronic Energy:                  {self.electronic_energy:16.8f} Hartree",
            f" Zero-Point Vibrational Energy:      {self.zero_point_energy_kcal:12.4f} kcal/mol",
            f"                                    ({self.zero_point_energy_au:16.8f} Hartree)",
            "-" * 72,
            " Thermal Energy Corrections (at T):",
            f"   Translational Energy (E_trans):   {self.thermal_energy_components.get('trans', 0.0):12.4f} kcal/mol",
            f"   Rotational Energy (E_rot):        {self.thermal_energy_components.get('rot', 0.0):12.4f} kcal/mol",
            f"   Vibrational Energy (E_vib):       {self.thermal_energy_components.get('vib', 0.0):12.4f} kcal/mol",
            f"   Total Thermal Energy (U_therm):   {self.thermal_energy_kcal:12.4f} kcal/mol",
            f"   Total Thermal Enthalpy (H_therm): {self.thermal_enthalpy_kcal:12.4f} kcal/mol",
            "-" * 72,
            " Entropies (S at T):",
            f"   Translational Entropy:            {self.entropy_cal_mol_k.get('translation', 0.0):12.3f} cal/(mol·K)",
            f"   Rotational Entropy:               {self.entropy_cal_mol_k.get('rotation', 0.0):12.3f} cal/(mol·K)",
            f"   Vibrational Entropy:              {self.entropy_cal_mol_k.get('vibration', 0.0):12.3f} cal/(mol·K)",
            f"   Electronic Entropy:               {self.entropy_cal_mol_k.get('electronic', 0.0):12.3f} cal/(mol·K)",
            f"   TOTAL ENTROPY:                    {self.entropy_cal_mol_k.get('total', 0.0):12.3f} cal/(mol·K)",
            "-" * 72,
            " Heat Capacities (Cv at T):",
            f"   Translational Cv:                 {self.heat_capacity_cal_mol_k.get('cv_trans', 0.0):12.3f} cal/(mol·K)",
            f"   Rotational Cv:                    {self.heat_capacity_cal_mol_k.get('cv_rot', 0.0):12.3f} cal/(mol·K)",
            f"   Vibrational Cv:                   {self.heat_capacity_cal_mol_k.get('cv_vib', 0.0):12.3f} cal/(mol·K)",
            f"   TOTAL HEAT CAPACITY (Cv):         {self.heat_capacity_cal_mol_k.get('cv_total', 0.0):12.3f} cal/(mol·K)",
            "=" * 72,
            " TOTAL THERMODYNAMIC ENERGIES:",
            f"   Internal Energy at 0 K   (U_0):   {self.total_energy_u0:16.8f} Hartree",
            f"   Internal Energy at T     (U_T):   {self.total_energy_u_t:16.8f} Hartree",
            f"   Enthalpy at T            (H_T):   {self.total_enthalpy_h_t:16.8f} Hartree",
            f"   Gibbs Free Energy at T   (G_T):   {self.total_gibbs_free_energy_g_t:16.8f} Hartree",
            "=" * 72,
        ]
        return "\n".join(lines)



@dataclass
class VibrationResult:
    """
    Complete result of a harmonic vibrational frequency calculation.
    """
    modes: List[NormalMode]
    hessian_cartesian: List[List[float]]
    hessian_mass_weighted: List[List[float]]
    atoms: List[Dict[str, Any]]
    thermochemistry: Optional[Thermochemistry] = None

    @property
    def num_atoms(self) -> int:
        return len(self.atoms)

    @property
    def num_modes(self) -> int:
        return len(self.modes)

    @property
    def vibrational_modes(self) -> List[NormalMode]:
        """Returns non-TR genuine vibrational modes."""
        return [m for m in self.modes if not m.is_tr_mode]

    @property
    def num_vibrational_modes(self) -> int:
        return len(self.vibrational_modes)

    @property
    def imaginary_modes(self) -> List[NormalMode]:
        """Returns modes with imaginary frequencies."""
        return [m for m in self.vibrational_modes if m.is_imaginary]

    @property
    def num_imaginary(self) -> int:
        return len(self.imaginary_modes)

    @property
    def is_minimum(self) -> bool:
        """True if stationary point is a local minimum (0 imaginary frequencies)."""
        return self.num_imaginary == 0

    @property
    def is_transition_state(self) -> bool:
        """True if stationary point is a first-order saddle point / transition state (1 imaginary frequency)."""
        return self.num_imaginary == 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert vibration result to a serializable dictionary."""
        return {
            "num_atoms": self.num_atoms,
            "num_modes": self.num_modes,
            "num_vibrational_modes": self.num_vibrational_modes,
            "num_imaginary": self.num_imaginary,
            "is_minimum": self.is_minimum,
            "is_transition_state": self.is_transition_state,
            "atoms": self.atoms,
            "modes": [m.to_dict() for m in self.modes],
            "hessian_cartesian": [[round(float(c), 6) for c in row] for row in self.hessian_cartesian],
            "hessian_mass_weighted": [[round(float(c), 6) for c in row] for row in self.hessian_mass_weighted],
            "thermochemistry": self.thermochemistry.to_dict() if self.thermochemistry else None,
        }

    def save_json(self, filepath: Union[str, Path]) -> None:
        """Saves vibration analysis result to a JSON file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self) -> str:
        """Prints a structured text summary table of normal modes and frequencies."""
        lines = [
            "=" * 72,
            " PHATPSY 2.0 HARMONIC VIBRATIONAL FREQUENCY ANALYSIS",
            "=" * 72,
            f" Number of Atoms:             {self.num_atoms}",
            f" Vibrational Normal Modes:    {self.num_vibrational_modes}",
            f" Imaginary Frequencies:       {self.num_imaginary}",
            f" Stationary Point Type:       {'Transition State (Saddle Point)' if self.is_transition_state else ('Local Minimum' if self.is_minimum else f'Higher-order Saddle ({self.num_imaginary} imaginary)')}",
            "-" * 72,
            f"{'Mode':>5}  {'Frequency (cm^-1)':>18}  {'Red. Mass (amu)':>16}  {'Force Const (mDyne/Å)':>22}",
            "-" * 72,
        ]

        for m in self.vibrational_modes:
            freq_str = f"{m.frequency_cm:10.2f} i" if m.is_imaginary else f"{m.frequency_cm:12.2f}"
            lines.append(
                f"{m.index:5d}  {freq_str:>18}  {m.reduced_mass_amu:16.4f}  {m.force_constant_mdyne_a:22.4f}"
            )
        lines.append("=" * 72)

        if self.thermochemistry:
            lines.append("")
            lines.append(self.thermochemistry.summary())

        return "\n".join(lines)

    def show(self, mode_index: Optional[int] = None, amplitude: float = 0.5, height: int = 520) -> Any:
        """Displays interactive 3D WebGL vibration visualizer widget in Jupyter/IPython."""
        from phatpsy.visualizer import show_vibrations
        return show_vibrations(self, mode_index=mode_index, amplitude=amplitude, height=height)

    def visualize(self, output_path: Optional[Union[str, Path]] = None, mode_index: Optional[int] = None, amplitude: float = 0.5) -> Any:
        """Generates standalone 3D WebGL HTML visualization for normal modes."""
        from phatpsy.visualizer import visualize_vibrations
        return visualize_vibrations(self, output_path=output_path, mode_index=mode_index, amplitude=amplitude)


# ==============================================================================
# Translational & Rotational Projector (Eckart / Sayvetz)
# ==============================================================================

def project_translations_rotations(
    H_mw: np.ndarray,
    coords: np.ndarray,
    masses: np.ndarray,
    is_linear: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Projects out the 5 (linear) or 6 (non-linear) center-of-mass translational
    and infinitesimal rotational degrees of freedom from the mass-weighted Hessian.

    Parameters
    ----------
    H_mw : np.ndarray
        (3N, 3N) mass-weighted Cartesian Hessian matrix.
    coords : np.ndarray
        (N, 3) Cartesian atomic coordinates in Bohr.
    masses : np.ndarray
        (N,) atomic masses in amu.
    is_linear : Optional[bool]
        Whether the molecule is strictly linear. If None, auto-detected from inertia tensor.

    Returns
    -------
    H_proj : np.ndarray
        (3N, 3N) projected mass-weighted Hessian matrix with TR modes set to zero.
    P : np.ndarray
        (3N, 3N) orthogonal projection matrix (I - D_TR * D_TR^T).
    n_tr : int
        Number of projected translational/rotational degrees of freedom (5 or 6).
    """
    n_atoms = len(coords)
    dim = 3 * n_atoms

    # Shift coordinates to center of mass
    total_mass = float(np.sum(masses))
    com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    r = coords - com

    # Inertia tensor to detect molecular linearity
    I_tensor = np.zeros((3, 3), dtype=np.float64)
    for i in range(n_atoms):
        x, y, z = r[i]
        m = masses[i]
        I_tensor[0, 0] += m * (y * y + z * z)
        I_tensor[1, 1] += m * (x * x + z * z)
        I_tensor[2, 2] += m * (x * x + y * y)
        I_tensor[0, 1] -= m * x * y
        I_tensor[0, 2] -= m * x * z
        I_tensor[1, 2] -= m * y * z
    I_tensor[1, 0] = I_tensor[0, 1]
    I_tensor[2, 0] = I_tensor[0, 2]
    I_tensor[2, 1] = I_tensor[1, 2]

    eig_I, principal_axes = np.linalg.eigh(I_tensor)

    if is_linear is None:
        if n_atoms <= 2:
            is_linear = True
        else:
            # Linear if smallest principal moment is negligible relative to largest
            is_linear = (eig_I[0] < 1e-4 * eig_I[2]) or (eig_I[2] < 1e-8)

    # Construct mass-weighted translational basis vectors
    sqrt_m = np.sqrt(masses)
    T = np.zeros((3, dim), dtype=np.float64)
    for i in range(n_atoms):
        T[0, i * 3] = sqrt_m[i]      # Tx
        T[1, i * 3 + 1] = sqrt_m[i]  # Ty
        T[2, i * 3 + 2] = sqrt_m[i]  # Tz

    # Construct mass-weighted rotational basis vectors along principal axes
    rot_vectors = []
    for axis_idx in range(3):
        # If linear, skip rotation along the linear molecular axis (smallest moment of inertia)
        if is_linear and axis_idx == 0:
            continue
        p_axis = principal_axes[:, axis_idx]
        R_vec = np.zeros(dim, dtype=np.float64)
        for i in range(n_atoms):
            # displacement = p_axis x r_i
            disp = np.cross(p_axis, r[i])
            R_vec[i * 3: (i + 1) * 3] = disp * sqrt_m[i]
        norm = np.linalg.norm(R_vec)
        if norm > 1e-8:
            rot_vectors.append(R_vec / norm)

    n_tr = 3 + len(rot_vectors)
    TR = np.vstack([T] + rot_vectors)  # shape (n_tr, 3N)

    # Normalize rows
    for k in range(len(TR)):
        nrm = np.linalg.norm(TR[k])
        if nrm > 1e-12:
            TR[k] /= nrm

    # Orthonormalize via QR decomposition
    Q_qr, _ = np.linalg.qr(TR.T, mode="complete")
    D_tr = Q_qr[:, :n_tr]

    # Form projection matrix P = I - D_tr * D_tr^T
    P = np.eye(dim, dtype=np.float64) - D_tr @ D_tr.T

    # Projected mass-weighted Hessian: H_proj = P * H_mw * P
    H_proj = P @ H_mw @ P
    H_proj = 0.5 * (H_proj + H_proj.T)

    return H_proj, P, n_tr


# ==============================================================================
# Cartesian Hessian Matrix Computation
# ==============================================================================

def compute_hessian(
    deck: CalculationDeck,
    driver: Optional[PhatpsyDriver] = None,
    step_size: float = 0.005,
    method: str = "central",
    use_analytical_gradients: bool = True,
    quiet: bool = True,
) -> Tuple[np.ndarray, CalculationDeck, List[int]]:
    """
    Computes the full Cartesian Hessian matrix (d^2 E / dX_i dX_j) in Hartree / Bohr^2.
    When use_analytical_gradients=True (default), computes central differences of analytical
    gradients with basis-derivative (Pulay) corrections:
        H_{:, j} = (g(X + h*e_j) - g(X - h*e_j)) / (2h)
    suppressing numerical grid/SCF noise by orders of magnitude compared to energy second differences.
    """
    if driver is None:
        driver = PhatpsyDriver(quiet=quiet)

    calc_deck = copy.deepcopy(deck)
    calc_deck.resolve_coordinates()

    # Clear symmetry locks and equivalencies on active unfrozen atoms
    # so all 3N coordinates can vibrate independently
    for a in calc_deck.geometry:
        if not a.fixed and not a.virtual:
            a.symmetry = None
            a.symmetry_op = None
            a.symmetry_axis = None
            a.equivalent_to = None

    atoms = calc_deck.geometry
    n_atoms = len(atoms)
    dim = 3 * n_atoms

    # Identify unfrozen, non-virtual degrees of freedom
    active_indices = []
    for i, a in enumerate(atoms):
        if not a.fixed and not a.virtual:
            active_indices.extend([i * 3, i * 3 + 1, i * 3 + 2])

    H = np.zeros((dim, dim), dtype=np.float64)
    h = float(step_size)

    if use_analytical_gradients:
        for j in active_indices:
            atom_idx = j // 3
            coord_idx = j % 3

            d_plus = copy.deepcopy(calc_deck)
            d_plus.geometry[atom_idx].xyz[coord_idx] += h
            g_plus = driver.gradient(d_plus, quiet=quiet).flatten()

            d_minus = copy.deepcopy(calc_deck)
            d_minus.geometry[atom_idx].xyz[coord_idx] -= h
            g_minus = driver.gradient(d_minus, quiet=quiet).flatten()

            H[:, j] = (g_plus - g_minus) / (2.0 * h)

        H = 0.5 * (H + H.T)
        return H, calc_deck, active_indices

    # Reference energy E0 (fallback energy second differences)
    res0 = driver.run_deck(calc_deck, quiet=quiet)
    E0 = float(res0.total_energy)

    coords0 = np.array([a.xyz for a in atoms], dtype=np.float64).flatten()

    def eval_displaced_energy(d_coords: np.ndarray) -> float:
        d_deck = copy.deepcopy(calc_deck)
        for idx, a in enumerate(d_deck.geometry):
            a.xyz = [float(d_coords[idx * 3]), float(d_coords[idx * 3 + 1]), float(d_coords[idx * 3 + 2])]
        r = driver.run_deck(d_deck, quiet=quiet)
        return float(r.total_energy)

    # 1. Diagonal elements: d^2 E / dX_i^2 = (E(X_i + h) - 2*E0 + E(X_i - h)) / h^2
    E_plus: Dict[int, float] = {}
    E_minus: Dict[int, float] = {}

    for i in active_indices:
        cp = coords0.copy()
        cp[i] += h
        ep = eval_displaced_energy(cp)
        E_plus[i] = ep

        cm = coords0.copy()
        cm[i] -= h
        em = eval_displaced_energy(cm)
        E_minus[i] = em

        H[i, i] = (ep - 2.0 * E0 + em) / (h * h)

    # 2. Off-diagonal elements: d^2 E / dX_i dX_j
    method_lower = method.strip().lower()
    for idx_i, i in enumerate(active_indices):
        for j in active_indices[idx_i + 1:]:
            if method_lower == "forward":
                # Forward cross difference: (E(++ ) - E(+0) - E(0+) + E0) / h^2
                cpp = coords0.copy()
                cpp[i] += h
                cpp[j] += h
                epp = eval_displaced_energy(cpp)
                val = (epp - E_plus[i] - E_plus[j] + E0) / (h * h)
            else:
                # Central cross difference: (E(++) - E(+-) - E(-+) + E(--)) / (4*h^2)
                cpp = coords0.copy()
                cpp[i] += h
                cpp[j] += h
                epp = eval_displaced_energy(cpp)

                cpm = coords0.copy()
                cpm[i] += h
                cpm[j] -= h
                epm = eval_displaced_energy(cpm)

                cmp = coords0.copy()
                cmp[i] -= h
                cmp[j] += h
                emp = eval_displaced_energy(cmp)

                cmm = coords0.copy()
                cmm[i] -= h
                cmm[j] -= h
                emm = eval_displaced_energy(cmm)

                val = (epp - epm - emp + emm) / (4.0 * h * h)

            H[i, j] = val
            H[j, i] = val

    # Symmetrize
    H = 0.5 * (H + H.T)
    return H, calc_deck, active_indices


# ==============================================================================
# Statistical Thermochemistry Engine (RRHO)
# ==============================================================================

def compute_thermochemistry(
    frequencies: List[float],
    coords: np.ndarray,
    masses: np.ndarray,
    electronic_energy: float,
    spin_multiplicity: int = 1,
    temperature_k: float = 298.15,
    pressure_atm: float = 1.0,
    symmetry_number: Optional[int] = None,
    is_linear: Optional[bool] = None,
) -> Thermochemistry:
    """
    Computes standard ideal-gas thermochemical properties at temperature T and pressure P
    using the Rigid-Rotor Harmonic Oscillator (RRHO) approximation.

    Parameters
    ----------
    frequencies : List[float]
        Harmonic vibrational frequencies in cm^-1.
    coords : np.ndarray
        (N, 3) Cartesian coordinates in Bohr.
    masses : np.ndarray
        (N,) atomic masses in amu.
    electronic_energy : float
        Electronic ground-state energy in Hartree.
    spin_multiplicity : int
        Electronic spin multiplicity (2S + 1, default: 1).
    temperature_k : float
        Temperature in Kelvin (default: 298.15).
    pressure_atm : float
        Pressure in standard atmospheres (default: 1.0).
    symmetry_number : Optional[int]
        Rotational symmetry number sigma (default: 1, or 2 for homonuclear diatomics).
    is_linear : Optional[bool]
        Whether molecule is linear. Auto-detected if None.

    Returns
    -------
    Thermochemistry
        Complete thermochemical analysis object.
    """
    T = float(temperature_k)
    P_pa = float(pressure_atm) * STANDARD_PRESSURE_PA
    R_cal = GAS_CONSTANT_R_CAL_MOL_K
    R_j = GAS_CONSTANT_R_J_MOL_K
    k_B = BOLTZMANN_CONSTANT_J_K
    h = PLANCK_CONSTANT_J_S
    c_cm = SPEED_OF_LIGHT_CM_S

    n_atoms = len(coords)

    # 1. Zero-Point Vibrational Energy (ZPVE)
    # Sum over real (positive) vibrational frequencies
    real_freqs = [f for f in frequencies if f > 0.0]
    # ZPVE = 1/2 * sum(h * c * nu)
    # 1 cm^-1 = (h * c * N_A / 4184) kcal/mol ≈ 0.002859143 kcal/mol
    cm_to_kcal_mol = (h * c_cm * AVOGADRO_CONSTANT) / 4184.0
    zpve_kcal = 0.5 * sum(real_freqs) * cm_to_kcal_mol
    zpve_au = zpve_kcal / HARTREE_TO_KCAL

    # 2. Vibrational Contributions
    E_vib_cal = 0.0
    S_vib_cal = 0.0
    Cv_vib_cal = 0.0

    for nu in real_freqs:
        # Characteristic vibrational temperature: theta_v = (h * c * nu) / k_B
        theta_v = (h * c_cm * nu) / k_B
        x = theta_v / T
        if x < 1e-4:
            # High-temperature limit
            E_vib_cal += R_cal * T
            S_vib_cal += R_cal * (1.0 - math.log(x))
            Cv_vib_cal += R_cal
        elif x > 80.0:
            # Low-temperature limit (ground vibrational state only)
            E_vib_cal += 0.5 * R_cal * theta_v
        else:
            exp_x = math.exp(x)
            E_vib_cal += R_cal * theta_v * (0.5 + 1.0 / (exp_x - 1.0))
            S_vib_cal += R_cal * (x / (exp_x - 1.0) - math.log(1.0 - math.exp(-x)))
            Cv_vib_cal += R_cal * (x * x * exp_x) / ((exp_x - 1.0) ** 2)

    E_vib_kcal = E_vib_cal / 1000.0

    # 3. Translational Contributions (Sackur-Tetrode)
    total_mass_amu = float(np.sum(masses))
    total_mass_kg = total_mass_amu * AMU_TO_KG

    V_m3 = (k_B * T) / P_pa
    thermal_de_broglie = math.sqrt((h * h) / (2.0 * math.pi * total_mass_kg * k_B * T))
    q_trans = V_m3 / (thermal_de_broglie ** 3)

    S_trans_cal = R_cal * (math.log(q_trans) + 2.5)
    E_trans_kcal = (1.5 * R_cal * T) / 1000.0
    Cv_trans_cal = 1.5 * R_cal

    # 4. Rotational Contributions
    com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass_amu
    r_bohr = coords - com
    r_m = r_bohr * BOHR_TO_METER

    I_tensor = np.zeros((3, 3), dtype=np.float64)
    for i in range(n_atoms):
        x, y, z = r_m[i]
        m = masses[i] * AMU_TO_KG
        I_tensor[0, 0] += m * (y * y + z * z)
        I_tensor[1, 1] += m * (x * x + z * z)
        I_tensor[2, 2] += m * (x * x + y * y)
        I_tensor[0, 1] -= m * x * y
        I_tensor[0, 2] -= m * x * z
        I_tensor[1, 2] -= m * y * z
    I_tensor[1, 0] = I_tensor[0, 1]
    I_tensor[2, 0] = I_tensor[0, 2]
    I_tensor[2, 1] = I_tensor[1, 2]

    eig_I, _ = np.linalg.eigh(I_tensor)

    if is_linear is None:
        if n_atoms == 1:
            is_linear = False
        elif n_atoms == 2:
            is_linear = True
        else:
            is_linear = (eig_I[0] < 1e-4 * eig_I[2]) or (eig_I[2] < 1e-40)

    # Symmetry number sigma
    if symmetry_number is None:
        if n_atoms == 2 and abs(masses[0] - masses[1]) < 1e-3:
            sigma = 2  # Homonuclear diatomic (e.g. N2)
        else:
            sigma = 1
    else:
        sigma = int(symmetry_number)

    hbar = HBAR_J_S
    if n_atoms == 1:
        # Monatomic gas: no rotational degrees of freedom
        E_rot_kcal = 0.0
        S_rot_cal = 0.0
        Cv_rot_cal = 0.0
    elif is_linear:
        # Linear rotor: 2 rotational degrees of freedom
        I_eff = eig_I[2] if eig_I[2] > 1e-40 else eig_I[1]
        theta_rot = (hbar * hbar) / (2.0 * I_eff * k_B)
        q_rot = T / (float(sigma) * theta_rot)
        S_rot_cal = R_cal * (math.log(max(1e-12, q_rot)) + 1.0)
        E_rot_kcal = (R_cal * T) / 1000.0
        Cv_rot_cal = R_cal
    else:
        # Non-linear rotor: 3 rotational degrees of freedom
        theta_1 = (hbar * hbar) / (2.0 * max(1e-45, eig_I[0]) * k_B)
        theta_2 = (hbar * hbar) / (2.0 * max(1e-45, eig_I[1]) * k_B)
        theta_3 = (hbar * hbar) / (2.0 * max(1e-45, eig_I[2]) * k_B)
        q_rot = (math.sqrt(math.pi) / float(sigma)) * math.sqrt(
            (T * T * T) / (theta_1 * theta_2 * theta_3)
        )
        S_rot_cal = R_cal * (1.5 + math.log(max(1e-12, q_rot)))
        E_rot_kcal = (1.5 * R_cal * T) / 1000.0
        Cv_rot_cal = 1.5 * R_cal

    # 5. Electronic Contributions
    g0 = max(1, spin_multiplicity)
    S_elec_cal = R_cal * math.log(g0)
    E_elec_kcal = 0.0
    Cv_elec_cal = 0.0

    # 6. Totals
    # Thermal energy U_therm = E_trans + E_rot + E_vib
    U_therm_kcal = E_trans_kcal + E_rot_kcal + E_vib_kcal
    # Thermal enthalpy H_therm = U_therm + R*T
    H_therm_kcal = U_therm_kcal + (R_cal * T) / 1000.0

    S_total_cal = S_trans_cal + S_rot_cal + S_vib_cal + S_elec_cal
    Cv_total_cal = Cv_trans_cal + Cv_rot_cal + Cv_vib_cal + Cv_elec_cal

    # Thermodynamic functions in Hartree
    # U0 = E_elec + ZPVE
    U0 = electronic_energy + zpve_au
    # U(T) = E_elec + U_therm
    U_t = electronic_energy + (U_therm_kcal / HARTREE_TO_KCAL)
    # H(T) = E_elec + H_therm
    H_t = electronic_energy + (H_therm_kcal / HARTREE_TO_KCAL)
    # G(T) = H(T) - T * S
    TS_kcal = (T * S_total_cal) / 1000.0
    G_t = H_t - (TS_kcal / HARTREE_TO_KCAL)

    return Thermochemistry(
        temperature_k=T,
        pressure_atm=float(pressure_atm),
        electronic_energy=electronic_energy,
        zero_point_energy_kcal=zpve_kcal,
        zero_point_energy_au=zpve_au,
        thermal_energy_kcal=U_therm_kcal,
        thermal_enthalpy_kcal=H_therm_kcal,
        entropy_cal_mol_k={
            "translation": S_trans_cal,
            "rotation": S_rot_cal,
            "vibration": S_vib_cal,
            "electronic": S_elec_cal,
            "total": S_total_cal,
        },
        heat_capacity_cal_mol_k={
            "cv_trans": Cv_trans_cal,
            "cv_rot": Cv_rot_cal,
            "cv_vib": Cv_vib_cal,
            "cv_total": Cv_total_cal,
        },
        total_energy_u0=U0,
        total_energy_u_t=U_t,
        total_enthalpy_h_t=H_t,
        total_gibbs_free_energy_g_t=G_t,
        thermal_energy_components={
            "trans": E_trans_kcal,
            "rot": E_rot_kcal,
            "vib": E_vib_kcal,
            "zpve": zpve_kcal,
            "total": U_therm_kcal,
        },
    )




# ==============================================================================
# Master Driver: compute_vibrations()
# ==============================================================================

def compute_vibrations(
    deck: CalculationDeck,
    driver: Optional[PhatpsyDriver] = None,
    step_size: float = 0.005,
    method: str = "central",
    project_tr: bool = True,
    use_analytical_gradients: bool = True,
    temperature_k: float = 298.15,
    pressure_atm: float = 1.0,
    symmetry_number: Optional[int] = None,
    spin_multiplicity: int = 1,
    quiet: bool = True,
) -> VibrationResult:
    """
    Performs full harmonic vibrational frequency analysis, normal mode displacement
    evaluation, and RRHO ideal-gas thermochemistry for a molecular calculation deck.

    Parameters
    ----------
    deck : CalculationDeck
        The molecular calculation deck.
    driver : Optional[PhatpsyDriver]
        PhatpsyDriver instance. Created if None.
    step_size : float
        Displacement step for Cartesian finite-difference Hessian (default: 0.005 Bohr).
    method : str
        Finite-difference stencil: "central" (default, O(h^2)) or "forward" (fast).
    project_tr : bool
        Whether to project out center-of-mass translation and rotation (default: True).
        Automatically disabled if any atoms are frozen (fixed=True).
    use_analytical_gradients : bool
        Whether to compute Hessian via central differences of analytical Pulay gradients
        (default: True). Suppresses unphysical finite-difference basis shift noise.
    temperature_k : float
        Thermochemistry reference temperature (default: 298.15 K).
    pressure_atm : float
        Thermochemistry reference pressure (default: 1.0 atm).
    symmetry_number : Optional[int]
        Rotational symmetry number (sigma). Auto-detected if None.
    spin_multiplicity : int
        Electronic spin multiplicity (2S + 1, default: 1).
    quiet : bool
        Whether to silence verbose Fortran SCF stdout (default: True).

    Returns
    -------
    VibrationResult
        Comprehensive vibrational analysis, normal mode vectors, and thermochemistry.
    """
    if driver is None:
        driver = PhatpsyDriver(quiet=quiet)

    # 1. Compute Cartesian Hessian matrix
    H_cart, calc_deck, active_indices = compute_hessian(
        deck=deck,
        driver=driver,
        step_size=step_size,
        method=method,
        use_analytical_gradients=use_analytical_gradients,
        quiet=quiet,
    )

    atoms = calc_deck.geometry
    n_atoms = len(atoms)
    dim = 3 * n_atoms

    # Extract atomic coordinates and masses
    coords = np.array([a.xyz for a in atoms], dtype=np.float64)
    masses = np.array([ATOMIC_MASSES_AMU.get(a.element.strip().upper(), 12.0) for a in atoms], dtype=np.float64)

    # Check if any atom is fixed (substrate / surface slab)
    has_fixed_atoms = any(a.fixed for a in atoms)

    # 2. Form Mass-Weighted Hessian: H_mw_{i, j} = H_{i, j} / sqrt(M_A * M_B)
    coord_masses = np.repeat(masses, 3)
    inv_sqrt_m = 1.0 / np.sqrt(coord_masses)
    H_mw = H_cart * np.outer(inv_sqrt_m, inv_sqrt_m)

    # 3. Project Translations & Rotations if requested and no fixed atoms
    n_tr = 0
    if project_tr and not has_fixed_atoms and n_atoms > 1:
        H_diag, P_proj, n_tr = project_translations_rotations(H_mw, coords, masses)
    else:
        H_diag = H_mw.copy()

    # 4. Diagonalize mass-weighted Hessian
    evals, evecs = np.linalg.eigh(H_diag)

    # 5. Build NormalMode objects
    # evals are in Hartree / (Bohr^2 * amu)
    modes: List[NormalMode] = []
    vib_freqs_for_thermo: List[float] = []

    # Identify TR modes by eigenvalue proximity to zero if projected (smallest absolute eigenvalues)
    if project_tr and not has_fixed_atoms and n_tr > 0:
        abs_evals = np.abs(evals)
        tr_indices = set(np.argsort(abs_evals)[:n_tr])
    else:
        tr_indices = set()

    for k in range(dim):
        lam = float(evals[k])
        is_tr = (k in tr_indices)

        if is_tr:
            freq = 0.0
            is_imag = False
        elif lam >= 0.0:
            freq = FREQ_CONV_FACTOR * math.sqrt(lam)
            is_imag = False
        else:
            freq = -FREQ_CONV_FACTOR * math.sqrt(abs(lam))
            is_imag = True

        # Mass-weighted eigenvector to Cartesian displacement
        # d_{A, alpha} = L_{A, alpha} / sqrt(M_A)
        L_mw = evecs[:, k]
        d_cart = L_mw * inv_sqrt_m
        norm_d = float(np.linalg.norm(d_cart))
        if norm_d > 1e-12:
            q_cart = d_cart / norm_d
        else:
            q_cart = d_cart

        # Reduced mass: mu_k = 1 / sum(q_{A, alpha}^2 / M_A)
        inv_mu = float(np.sum((q_cart ** 2) / coord_masses))
        mu_k = (1.0 / inv_mu) if inv_mu > 1e-12 else 1.0

        # Effective force constant in mDyne / Å: k = lambda * mu * conv
        k_mdyne = abs(lam) * mu_k * FORCE_CONSTANT_CONV

        disp_atoms = [
            [float(q_cart[i * 3]), float(q_cart[i * 3 + 1]), float(q_cart[i * 3 + 2])]
            for i in range(n_atoms)
        ]

        mode_obj = NormalMode(
            index=k + 1,
            frequency_cm=freq,
            is_imaginary=is_imag,
            reduced_mass_amu=mu_k,
            force_constant_mdyne_a=k_mdyne,
            displacements=disp_atoms,
            is_tr_mode=is_tr,
        )
        modes.append(mode_obj)

        if not is_tr and not is_imag and freq > 1e-3:
            vib_freqs_for_thermo.append(freq)

    # Reference electronic energy from unperturbed calculation
    res_ref = driver.run_deck(calc_deck, quiet=quiet)
    E_elec = float(res_ref.total_energy)

    # 6. Statistical Thermochemistry
    thermo = compute_thermochemistry(
        frequencies=vib_freqs_for_thermo,
        coords=coords,
        masses=masses,
        electronic_energy=E_elec,
        spin_multiplicity=spin_multiplicity,
        temperature_k=temperature_k,
        pressure_atm=pressure_atm,
        symmetry_number=symmetry_number,
    )

    atom_meta = [
        {
            "element": a.element,
            "id": a.id or f"{a.element}{idx + 1}",
            "xyz": [round(float(c), 6) for c in a.xyz],
            "mass_amu": round(float(masses[idx]), 4),
            "fixed": bool(a.fixed),
            "virtual": bool(a.virtual),
        }
        for idx, a in enumerate(atoms)
    ]

    return VibrationResult(
        modes=modes,
        hessian_cartesian=H_cart.tolist(),
        hessian_mass_weighted=H_mw.tolist(),
        atoms=atom_meta,
        thermochemistry=thermo,
    )
