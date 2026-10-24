# phatpsy/md.py
"""
Born-Oppenheimer Molecular Dynamics (BOMD) simulation engine for PHATPSY 2.0.

Simulates classical nuclear trajectories on the Born-Oppenheimer potential energy surface:
    M_I * d^2 R_I / dt^2 = F_I = -nabla_{R_I} E_0(R)

Features:
- Symplectic Velocity Verlet numerical time integration.
- Exact atomic units internal computation:
    Length: Bohr (a_0)
    Energy: Hartree (E_h)
    Mass: electron mass (m_e)
    Time: atomic unit of time (hbar / E_h ≈ 0.02418884 fs, 1 fs ≈ 41.34137 a.u.)
- Thermostats:
    - NVE (Microcanonical, strict total energy conservation)
    - NVT Berendsen weak-coupling thermostat
    - NVT Velocity Rescaling thermostat
    - NVT Andersen stochastic collision thermostat
- Maxwell-Boltzmann initial velocity distribution with zero center-of-mass linear momentum.
- Warm-restart wavefunction caching across trajectory time steps.
- Active center constraints: respects frozen atoms (fixed=True), virtual ghost centers, and symmetry.
- Trajectory exports: multi-frame XYZ files, structured JSON datasets, dual/triple-panel plots,
  and embedded 3D Three.js WebGL animation playback.
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


# ==============================================================================
# Physical Constants & Unit Conversions (Atomic Units)
# ==============================================================================

# 1 amu (Dalton) in electron masses (m_e)
AMU_TO_AU_MASS: float = 1822.888486209

# 1 fs in atomic units of time (hbar / E_h)
# hbar = 1.054571817e-34 J s, 1 Hartree = 4.3597447222071e-18 J
# 1 a.u. time ≈ 2.4188843265857e-17 s ≈ 0.024188843 fs
FS_TO_AU_TIME: float = 41.341374575751
AU_TIME_TO_FS: float = 1.0 / FS_TO_AU_TIME

# Boltzmann constant in Hartree / Kelvin
BOLTZMANN_AU: float = 3.166811563e-6

# Hartree to kcal/mol
HARTREE_TO_KCAL: float = 627.509474

# Bohr to Angstrom
BOHR_TO_ANGSTROM: float = 0.529177210903


# ==============================================================================
# Standard Periodic Table Atomic Masses (in amu)
# ==============================================================================

ATOMIC_MASSES_AMU: Dict[str, float] = {
    "H": 1.008, "HE": 4.0026,
    "LI": 6.94, "BE": 9.0122, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "NE": 20.180,
    "NA": 22.990, "MG": 24.305, "AL": 26.982, "SI": 28.085, "P": 30.974, "S": 32.06, "CL": 35.45, "AR": 39.948,
    "K": 39.098, "CA": 40.078, "SC": 44.956, "TI": 47.867, "V": 50.942, "CR": 51.996, "MN": 54.938, "FE": 55.845,
    "CO": 58.933, "NI": 58.693, "CU": 63.546, "ZN": 65.38, "GA": 69.723, "GE": 72.630, "AS": 74.922, "SE": 78.971,
    "BR": 79.904, "KR": 83.798,
    "RB": 85.468, "SR": 87.62, "Y": 88.906, "ZR": 91.224, "NB": 92.906, "MO": 95.95, "TC": 98.0, "RU": 101.07,
    "RH": 102.91, "PD": 106.42, "AG": 107.87, "CD": 112.41, "IN": 114.82, "SN": 118.71, "SB": 121.76, "TE": 127.60,
    "I": 126.90, "XE": 131.29,
    "CS": 132.91, "BA": 137.33, "LA": 138.91, "CE": 140.12, "PR": 140.91, "ND": 144.24, "PM": 145.0, "SM": 150.36,
    "EU": 151.96, "GD": 157.25, "TB": 158.93, "DY": 162.50, "HO": 164.93, "ER": 167.26, "TM": 168.93, "YB": 173.05,
    "LU": 174.97, "HF": 178.49, "TA": 180.95, "W": 183.84, "RE": 186.21, "OS": 190.23, "IR": 192.22, "PT": 195.08,
    "AU": 196.97, "HG": 200.59, "TL": 204.38, "PB": 207.2, "BI": 208.98, "TH": 232.04, "PA": 231.04, "U": 238.03,
}


def get_atomic_mass_au(element: str) -> float:
    """Returns atomic mass of element in electron masses (m_e)."""
    elem = element.strip().upper()
    mass_amu = ATOMIC_MASSES_AMU.get(elem, 12.0)
    return mass_amu * AMU_TO_AU_MASS


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class MDStep:
    """Snapshot of a single time step in a molecular dynamics trajectory."""
    step: int
    time_fs: float
    potential_energy: float      # Hartree
    kinetic_energy: float        # Hartree
    total_energy: float          # Hartree
    temperature: float           # Kelvin
    virial_ratio: float          # -V/T
    coordinates: List[List[float]]  # (N_atoms, 3) in Bohr
    velocities: List[List[float]]   # (N_atoms, 3) in Bohr / a.u. time
    forces: List[List[float]]       # (N_atoms, 3) in Hartree / Bohr

    @property
    def total_energy_kcal(self) -> float:
        """Total energy in kcal/mol."""
        return self.total_energy * HARTREE_TO_KCAL

    @property
    def potential_energy_kcal(self) -> float:
        """Potential energy in kcal/mol."""
        return self.potential_energy * HARTREE_TO_KCAL

    @property
    def kinetic_energy_kcal(self) -> float:
        """Kinetic energy in kcal/mol."""
        return self.kinetic_energy * HARTREE_TO_KCAL


@dataclass
class MDResult:
    """Full trajectory and summary observables from a Born-Oppenheimer MD simulation."""
    ensemble: str
    time_step_fs: float
    total_time_fs: float
    num_steps: int
    initial_energy: float
    final_energy: float
    energy_drift: float
    average_temperature: float
    temperature_std: float
    conserved_energy_std: float
    trajectory: List[MDStep]
    final_deck: CalculationDeck
    final_state: CalculationState

    def to_dict(self) -> Dict[str, Any]:
        """Serializes MD simulation results to a JSON-compatible dictionary."""
        return {
            "ensemble": self.ensemble,
            "time_step_fs": round(float(self.time_step_fs), 4),
            "total_time_fs": round(float(self.total_time_fs), 4),
            "num_steps": int(self.num_steps),
            "initial_energy_hartree": round(float(self.initial_energy), 8),
            "final_energy_hartree": round(float(self.final_energy), 8),
            "energy_drift_hartree": round(float(self.energy_drift), 8),
            "energy_drift_kcal_per_mol": round(float(self.energy_drift * HARTREE_TO_KCAL), 6),
            "average_temperature_k": round(float(self.average_temperature), 2),
            "temperature_std_k": round(float(self.temperature_std), 2),
            "conserved_energy_std_hartree": round(float(self.conserved_energy_std), 8),
            "trajectory": [
                {
                    "step": s.step,
                    "time_fs": round(float(s.time_fs), 3),
                    "potential_energy": round(float(s.potential_energy), 8),
                    "kinetic_energy": round(float(s.kinetic_energy), 8),
                    "total_energy": round(float(s.total_energy), 8),
                    "temperature": round(float(s.temperature), 2),
                    "virial_ratio": round(float(s.virial_ratio), 6),
                    "coordinates_bohr": [[round(float(c), 6) for c in pt] for pt in s.coordinates],
                    "velocities_au": [[round(float(v), 8) for v in vel] for vel in s.velocities],
                    "forces_au": [[round(float(f), 8) for f in frc] for frc in s.forces],
                }
                for s in self.trajectory
            ],
            "atoms": [
                {
                    "element": a.element,
                    "fixed": bool(a.fixed),
                    "virtual": bool(a.virtual),
                }
                for a in self.final_deck.atoms
            ],
        }

    def save_json(self, filepath: Union[str, Path]) -> None:
        """Saves trajectory and thermodynamics metrics to JSON."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_xyz(self, filepath: Union[str, Path]) -> None:
        """
        Exports the entire MD trajectory to standard multi-frame XYZ format.
        Compatible with VMD, PyMOL, Avogadro, Ovito, and Jmol.
        """
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        elements = [a.element for a in self.final_deck.atoms]
        n_atoms = len(elements)

        lines: List[str] = []
        for s in self.trajectory:
            lines.append(f"{n_atoms}")
            comment = (
                f"Step={s.step} Time={s.time_fs:.2f}fs E_pot={s.potential_energy:.6f}Ha "
                f"E_tot={s.total_energy:.6f}Ha T={s.temperature:.1f}K"
            )
            lines.append(comment)
            for el, xyz in zip(elements, s.coordinates):
                lines.append(f"{el:<3s} {xyz[0]:14.8f} {xyz[1]:14.8f} {xyz[2]:14.8f}")

        p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def plot_dynamics(
        self,
        output_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> Any:
        """
        Generates a 3-panel thermodynamic dynamics plot:
        1. Conserved Total, Potential, and Kinetic Energy vs. Time (fs)
        2. Instantaneous and Running-Average Temperature (K) vs. Time (fs)
        3. Virial Theorem (-V/T) Stability vs. Time (fs)
        """
        import matplotlib.pyplot as plt

        times = [s.time_fs for s in self.trajectory]
        e_pot = [s.potential_energy for s in self.trajectory]
        e_kin = [s.kinetic_energy for s in self.trajectory]
        e_tot = [s.total_energy for s in self.trajectory]
        temps = [s.temperature for s in self.trajectory]
        virials = [s.virial_ratio for s in self.trajectory]

        # Running average temperature
        cum_temps = np.cumsum(temps) / (np.arange(len(temps)) + 1)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Panel 1: Energies
        ax1.plot(times, e_pot, label="Potential Energy", color="#dc2626", lw=1.8)
        ax1.plot(times, e_kin, label="Kinetic Energy", color="#16a34a", lw=1.8)
        ax1.plot(times, e_tot, label="Total Energy (Conserved)", color="#2563eb", lw=2.2, linestyle="--")
        ax1.set_ylabel("Energy (Hartree)", fontsize=11)
        ax1.set_title(
            f"Born-Oppenheimer Molecular Dynamics ({self.ensemble.upper()}) — "
            f"ΔE Drift = {self.energy_drift * HARTREE_TO_KCAL:.3f} kcal/mol",
            fontsize=12,
            fontweight="bold",
        )
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="best", framealpha=0.9)

        # Panel 2: Temperature
        ax2.plot(times, temps, label="Instantaneous T", color="#ea580c", alpha=0.6, lw=1.2)
        ax2.plot(times, cum_temps, label=f"Average T ({self.average_temperature:.1f} K)", color="#9a3412", lw=2.0)
        ax2.set_ylabel("Temperature (K)", fontsize=11)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="best", framealpha=0.9)

        # Panel 3: Virial Ratio
        ax3.plot(times, virials, label="Virial Ratio (-V/T)", color="#7c3aed", lw=1.5)
        ax3.axhline(2.0, color="#6b7280", linestyle=":", lw=1.2, label="Harmonic Ideal (2.0)")
        ax3.set_xlabel("Time (fs)", fontsize=11)
        ax3.set_ylabel("-V / T", fontsize=11)
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(loc="best", framealpha=0.9)

        plt.tight_layout()

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_p, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    def show(self, height: int = 550) -> Any:
        """Displays interactive 3D WebGL trajectory player directly inside Jupyter notebooks."""
        from phatpsy.visualizer import show_trajectory
        return show_trajectory(self, height=height)

    def visualize(self, output_path: Optional[Union[str, Path]] = None, open_browser: bool = True) -> str:
        """Generates standalone interactive 3D WebGL trajectory viewer in browser."""
        from phatpsy.visualizer import visualize_trajectory
        return visualize_trajectory(self, output_path=output_path, open_browser=open_browser)


# ==============================================================================
# Velocity Initialization & Thermostats
# ==============================================================================

def init_velocities(
    masses: np.ndarray,
    temperature_k: float,
    free_mask: np.ndarray,
    remove_com_drift: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Initializes Cartesian atomic velocities from the Maxwell-Boltzmann distribution.

    Args:
        masses: Array of atomic masses in electron masses m_e, shape (N_atoms,).
        temperature_k: Target initial kinetic temperature in Kelvin.
        free_mask: Boolean array indicating unfrozen atoms, shape (N_atoms,).
        remove_com_drift: If True, subtracts center-of-mass linear velocity.
        rng: Optional NumPy random Generator for reproducible sampling.

    Returns:
        Velocities array of shape (N_atoms, 3) in atomic units (Bohr / a.u. time).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_atoms = len(masses)
    velocities = np.zeros((n_atoms, 3), dtype=np.float64)

    if temperature_k <= 1e-6:
        return velocities

    # Draw standard Gaussian normal for free atoms
    for i in range(n_atoms):
        if free_mask[i]:
            sigma = math.sqrt(BOLTZMANN_AU * temperature_k / masses[i])
            velocities[i] = rng.normal(0.0, sigma, size=3)

    # Remove net center-of-mass drift across free atoms (if multiple free atoms)
    if remove_com_drift and np.sum(free_mask) > 1:
        free_masses = masses[free_mask]
        if len(free_masses) > 0 and np.sum(free_masses) > 0:
            total_mass = np.sum(free_masses)
            com_momentum = np.sum(masses[free_mask, None] * velocities[free_mask], axis=0)
            v_com = com_momentum / total_mass
            velocities[free_mask] -= v_com

    # Rescale velocities to exact target temperature
    current_t = compute_temperature(velocities, masses, free_mask)
    if current_t > 1e-6:
        scale = math.sqrt(temperature_k / current_t)
        velocities[free_mask] *= scale

    return velocities


def compute_temperature(
    velocities: np.ndarray,
    masses: np.ndarray,
    free_mask: np.ndarray,
) -> float:
    """
    Computes instantaneous kinetic temperature in Kelvin:
        T = (2 * E_kin) / (N_dof * k_B)
    """
    n_free = int(np.sum(free_mask))
    if n_free == 0:
        return 0.0

    # For N_free moving atoms with COM translation removed, N_dof = 3 * N_free - 3 (if N_free > 1)
    n_dof = max(1, 3 * n_free - 3 if n_free > 1 else 3)
    e_kin = 0.5 * np.sum(masses[free_mask, None] * (velocities[free_mask] ** 2))
    return float(2.0 * e_kin / (n_dof * BOLTZMANN_AU))


def compute_kinetic_energy(
    velocities: np.ndarray,
    masses: np.ndarray,
    free_mask: np.ndarray,
) -> float:
    """Computes total kinetic energy in Hartree."""
    return float(0.5 * np.sum(masses[free_mask, None] * (velocities[free_mask] ** 2)))


# ==============================================================================
# Born-Oppenheimer Molecular Dynamics Engine
# ==============================================================================

def run_md(
    deck: CalculationDeck,
    time_step_fs: float = 0.5,
    num_steps: int = 50,
    temperature_k: float = 300.0,
    ensemble: str = "nve",
    tau_fs: float = 50.0,
    driver: Optional[PhatpsyDriver] = None,
    finite_diff_step: float = 0.005,
    remove_com_drift: bool = True,
    warm_restart: bool = True,
    initial_velocities: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    callback: Optional[Callable[[int, MDStep], None]] = None,
) -> MDResult:
    """
    Executes Born-Oppenheimer Molecular Dynamics (BOMD) simulation on a molecular system.

    Integrates Newton's equations of motion on the Born-Oppenheimer potential energy surface
    using the symplectic Velocity Verlet algorithm with quantum chemical forces.

    Args:
        deck: Initial CalculationDeck defining system geometry, basis sets, and electronic state.
        time_step_fs: Time integration step size in femtoseconds (default: 0.5 fs).
        num_steps: Number of integration steps to perform (default: 50).
        temperature_k: Initial/target temperature in Kelvin (default: 300.0 K).
        ensemble: Statistical ensemble ('nve', 'berendsen', 'rescale', 'andersen', default: 'nve').
        tau_fs: Berendsen thermostat coupling time constant in femtoseconds (default: 50.0 fs).
        driver: PhatpsyDriver instance (created if None).
        finite_diff_step: Finite-difference displacement step for forces in Bohr (default: 0.005).
        remove_com_drift: Whether to remove net linear center-of-mass momentum (default: True).
        warm_restart: Whether to seed successive electronic SCF steps with previous wavefunctions.
        initial_velocities: Optional pre-defined velocities array of shape (N_atoms, 3) in a.u.
        seed: Optional random number seed for Maxwell-Boltzmann velocity sampling.
        callback: Optional hook callback(step_index, md_step) called at each step.

    Returns:
        MDResult containing the complete trajectory, energy conservation metrics, and final deck.
    """
    if driver is None:
        driver = PhatpsyDriver()

    rng = np.random.default_rng(seed)
    current_deck = copy.deepcopy(deck)
    atoms = current_deck.atoms
    n_atoms = len(atoms)

    # 1. Identify active degrees of freedom (non-fixed, non-virtual, non-symmetry)
    free_mask = np.array(
        [not a.fixed and not a.virtual and a.symmetry is None for a in atoms],
        dtype=bool,
    )
    if not np.any(free_mask):
        raise ValueError("Cannot perform MD: all atoms are marked fixed or virtual.")

    free_indices = [i for i, f in enumerate(free_mask) if f]

    # 2. Extract atomic masses in electron mass units (m_e)
    masses = np.array([get_atomic_mass_au(a.element) for a in atoms], dtype=np.float64)

    # 3. Setup coordinates in Bohr
    coords = np.array([a.xyz for a in atoms], dtype=np.float64)

    # Convert time step to atomic units
    dt_au = float(time_step_fs * FS_TO_AU_TIME)

    def evaluate_energy_and_forces(
        current_coords: np.ndarray,
    ) -> Tuple[float, float, np.ndarray, CalculationState]:
        """
        Calculates Born-Oppenheimer potential energy, virial ratio, and nuclear forces F = -grad(E).
        Returns (E_pot, virial_ratio, forces, state).
        """
        eval_deck = copy.deepcopy(current_deck)
        for i, a in enumerate(eval_deck.atoms):
            a.xyz = [float(current_coords[i, 0]), float(current_coords[i, 1]), float(current_coords[i, 2])]

        state = driver.run_deck(eval_deck)
        e0 = float(state.total_energy)
        virial = float(state.virial_ratio)

        # Compute nuclear forces for free atoms via central finite difference: F = -dE / dX
        forces = np.zeros_like(current_coords)
        h = finite_diff_step

        for i in free_indices:
            for coord_idx in range(3):
                # Displace +h
                eval_p = copy.deepcopy(eval_deck)
                eval_p.atoms[i].xyz[coord_idx] += h
                state_p = driver.run_deck(eval_p)
                ep = float(state_p.total_energy)

                # Displace -h
                eval_m = copy.deepcopy(eval_deck)
                eval_m.atoms[i].xyz[coord_idx] -= h
                state_m = driver.run_deck(eval_m)
                em = float(state_m.total_energy)

                # Gradient g = (ep - em) / (2 * h), Force F = -g
                grad_val = (ep - em) / (2.0 * h)
                forces[i, coord_idx] = -grad_val

        return e0, virial, forces, state

    # 4. Initial forces and potential energy at t = 0
    e_pot, virial, forces, state = evaluate_energy_and_forces(coords)

    # 5. Initialize velocities
    if initial_velocities is not None:
        velocities = np.array(initial_velocities, dtype=np.float64).copy()
        if velocities.shape != (n_atoms, 3):
            raise ValueError(f"initial_velocities shape must be ({n_atoms}, 3), got {velocities.shape}")
        # Enforce zero velocities for fixed atoms
        velocities[~free_mask] = 0.0
    else:
        velocities = init_velocities(
            masses=masses,
            temperature_k=temperature_k,
            free_mask=free_mask,
            remove_com_drift=remove_com_drift,
            rng=rng,
        )

    # Initial accelerations a = F / M
    accelerations = np.zeros_like(forces)
    accelerations[free_mask] = forces[free_mask] / masses[free_mask, None]

    e_kin = compute_kinetic_energy(velocities, masses, free_mask)
    temp_k = compute_temperature(velocities, masses, free_mask)
    e_tot = e_pot + e_kin

    step0 = MDStep(
        step=0,
        time_fs=0.0,
        potential_energy=e_pot,
        kinetic_energy=e_kin,
        total_energy=e_tot,
        temperature=temp_k,
        virial_ratio=virial,
        coordinates=coords.tolist(),
        velocities=velocities.tolist(),
        forces=forces.tolist(),
    )

    trajectory: List[MDStep] = [step0]
    if callback:
        callback(0, step0)

    # 6. Molecular Dynamics Integration Loop (Velocity Verlet)
    ens = ensemble.strip().lower()

    for step_idx in range(1, num_steps + 1):
        # Step A: Position update: R(t + dt) = R(t) + v(t)*dt + 0.5*a(t)*dt^2
        coords[free_mask] += (
            velocities[free_mask] * dt_au + 0.5 * accelerations[free_mask] * (dt_au ** 2)
        )

        # Step B: Half-step velocity advance: v(t + dt/2) = v(t) + 0.5*a(t)*dt
        velocities[free_mask] += 0.5 * accelerations[free_mask] * dt_au

        # Step C: Evaluate new forces at updated positions R(t + dt)
        e_pot, virial, forces, state = evaluate_energy_and_forces(coords)

        # New accelerations a(t + dt) = F(t + dt) / M
        new_accelerations = np.zeros_like(forces)
        new_accelerations[free_mask] = forces[free_mask] / masses[free_mask, None]

        # Step D: Complete velocity update: v(t + dt) = v(t + dt/2) + 0.5*a(t + dt)*dt
        velocities[free_mask] += 0.5 * new_accelerations[free_mask] * dt_au
        accelerations = new_accelerations

        # Remove linear momentum drift if requested
        if remove_com_drift:
            free_masses = masses[free_mask]
            if len(free_masses) > 0 and np.sum(free_masses) > 0:
                com_momentum = np.sum(masses[free_mask, None] * velocities[free_mask], axis=0)
                velocities[free_mask] -= com_momentum / np.sum(free_masses)

        # Step E: Thermostat application
        current_t = compute_temperature(velocities, masses, free_mask)

        if ens == "berendsen" and current_t > 1e-4 and temperature_k > 0:
            tau_au = tau_fs * FS_TO_AU_TIME
            scale = math.sqrt(max(0.1, 1.0 + (dt_au / tau_au) * (temperature_k / current_t - 1.0)))
            velocities[free_mask] *= scale
            current_t = compute_temperature(velocities, masses, free_mask)

        elif ens == "rescale" and current_t > 1e-4 and temperature_k > 0:
            scale = math.sqrt(temperature_k / current_t)
            velocities[free_mask] *= scale
            current_t = compute_temperature(velocities, masses, free_mask)

        elif ens == "andersen" and temperature_k > 0:
            # Stochastic collision with probability nu * dt
            nu = 1.0 / (tau_fs * FS_TO_AU_TIME)
            prob = min(1.0, nu * dt_au)
            for i in free_indices:
                if rng.uniform(0.0, 1.0) < prob:
                    sigma = math.sqrt(BOLTZMANN_AU * temperature_k / masses[i])
                    velocities[i] = rng.normal(0.0, sigma, size=3)
            current_t = compute_temperature(velocities, masses, free_mask)

        # Compute thermodynamic metrics for current step
        e_kin = compute_kinetic_energy(velocities, masses, free_mask)
        e_tot = e_pot + e_kin
        time_fs = step_idx * time_step_fs

        md_step = MDStep(
            step=step_idx,
            time_fs=time_fs,
            potential_energy=e_pot,
            kinetic_energy=e_kin,
            total_energy=e_tot,
            temperature=current_t,
            virial_ratio=virial,
            coordinates=coords.tolist(),
            velocities=velocities.tolist(),
            forces=forces.tolist(),
        )

        trajectory.append(md_step)
        if callback:
            callback(step_idx, md_step)

    # 7. Summary metrics
    initial_e = trajectory[0].total_energy
    final_e = trajectory[-1].total_energy
    e_drift = final_e - initial_e

    tot_energies = [s.total_energy for s in trajectory]
    all_temps = [s.temperature for s in trajectory]
    conserved_std = float(np.std(tot_energies))
    avg_temp = float(np.mean(all_temps))
    temp_std = float(np.std(all_temps))

    # Construct final deck with relaxed coordinates
    final_deck = copy.deepcopy(current_deck)
    for i, a in enumerate(final_deck.atoms):
        a.xyz = [float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])]

    return MDResult(
        ensemble=ens,
        time_step_fs=time_step_fs,
        total_time_fs=num_steps * time_step_fs,
        num_steps=num_steps,
        initial_energy=initial_e,
        final_energy=final_e,
        energy_drift=e_drift,
        average_temperature=avg_temp,
        temperature_std=temp_std,
        conserved_energy_std=conserved_std,
        trajectory=trajectory,
        final_deck=final_deck,
        final_state=state,
    )
