# phatpsy/irc.py
"""
Intrinsic Reaction Coordinate (IRC) Reaction Pathway Tracing for PHATPSY 2.0.

Key capabilities:
- Mass-weighted steepest-descent path integration down the minimum energy valley:
    x_i = sqrt(M_i) * r_i
    dx(s) / ds = - g_x / ||g_x||
- Saddle point initiation from transition state (TS) along normal mode displacement (imaginary frequency mode Q_imag).
- Tracing both forward (+s, product) and reverse (-s, reactant) branches.
- Support for frozen substrate atoms (fixed=True) and ghost centers (is_virtual=True).
- Adaptive step-size with energy-decrease line search and convergence criteria (||g|| <= tol or Delta E <= tol).
- Structured IRCResult with barrier heights (Delta E_f^ddagger, Delta E_r^ddagger) and reaction energy Delta E_rxn.
- Multi-frame XYZ export, structured JSON checkpointing, 2D reaction coordinate profile plotting (plot_pathway).
- Interactive 3D WebGL Three.js reaction path playback player.
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
from phatpsy.vibrations import NormalMode, compute_vibrations


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class IRCPoint:
    """A single discrete point along the Intrinsic Reaction Coordinate."""
    step: int
    direction: str          # "reactant", "ts", "product"
    s: float                # Reaction coordinate arc length in amu^(1/2) * Bohr
    energy: float           # Total molecular electronic energy in Hartree
    delta_energy_kcal: float  # Energy relative to TS: (E - E_TS) in kcal/mol
    grad_norm: float        # Cartesian gradient norm ||g|| in Hartree / Bohr
    geometry: List[Dict[str, Any]]  # List of atom dictionaries with element, xyz, fixed, virtual
    virial_ratio: float = -2.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "direction": self.direction,
            "s": round(float(self.s), 4),
            "energy": round(float(self.energy), 8),
            "delta_energy_kcal": round(float(self.delta_energy_kcal), 4),
            "grad_norm": round(float(self.grad_norm), 6),
            "virial_ratio": round(float(self.virial_ratio), 6),
            "geometry": [
                {
                    "element": a["element"],
                    "xyz": [round(float(c), 6) for c in a["xyz"]],
                    "fixed": bool(a.get("fixed", False)),
                    "virtual": bool(a.get("virtual", False)),
                }
                for a in self.geometry
            ],
        }


@dataclass
class IRCResult:
    """Comprehensive Intrinsic Reaction Coordinate (IRC) trajectory and reaction analytics."""
    points: List[IRCPoint]          # Full reaction path from reactant minimum -> TS -> product minimum
    forward_points: List[IRCPoint]  # Forward branch: TS -> product
    reverse_points: List[IRCPoint]  # Reverse branch: TS -> reactant
    ts_point: IRCPoint              # Saddle point reference (s = 0.0)
    ts_energy: float
    barrier_forward_kcal: float     # Delta E_f^ddagger = E_TS - E_reactant (kcal/mol)
    barrier_reverse_kcal: float     # Delta E_r^ddagger = E_TS - E_product (kcal/mol)
    reaction_energy_kcal: float     # Delta E_rxn = E_product - E_reactant (kcal/mol)
    converged_forward: bool
    converged_reverse: bool
    initial_mode_frequency_cm: Optional[float] = None

    @property
    def num_points(self) -> int:
        return len(self.points)

    @property
    def reactant_point(self) -> IRCPoint:
        return self.points[0]

    @property
    def product_point(self) -> IRCPoint:
        return self.points[-1]

    def to_dict(self) -> Dict[str, Any]:
        """Converts result to a structured JSON-serializable dictionary."""
        return {
            "num_points": self.num_points,
            "ts_energy": round(float(self.ts_energy), 8),
            "barrier_forward_kcal": round(float(self.barrier_forward_kcal), 4),
            "barrier_reverse_kcal": round(float(self.barrier_reverse_kcal), 4),
            "reaction_energy_kcal": round(float(self.reaction_energy_kcal), 4),
            "converged_forward": self.converged_forward,
            "converged_reverse": self.converged_reverse,
            "initial_mode_frequency_cm": round(float(self.initial_mode_frequency_cm), 2) if self.initial_mode_frequency_cm is not None else None,
            "points": [pt.to_dict() for pt in self.points],
        }

    def save_json(self, filepath: Union[str, Path]) -> None:
        """Saves IRC results and full pathway trajectory to JSON."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_xyz(self, filepath: Union[str, Path]) -> None:
        """Exports multi-frame XYZ trajectory of the reaction path."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for pt in self.points:
                f.write(f"{len(pt.geometry)}\n")
                f.write(
                    f"IRC step={pt.step} s={pt.s:+.4f} E={pt.energy:.8f} dE={pt.delta_energy_kcal:+.2f} kcal/mol dir={pt.direction}\n"
                )
                for a in pt.geometry:
                    el = a["element"]
                    x, y, z = [c * BOHR_TO_ANGSTROM for c in a["xyz"]]
                    f.write(f"{el:<3} {x:12.6f} {y:12.6f} {z:12.6f}\n")

    def plot_pathway(
        self,
        output_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> Any:
        """Plots the IRC energy profile vs reaction coordinate s."""
        import matplotlib.pyplot as plt

        s_vals = [pt.s for pt in self.points]
        e_kcal = [pt.delta_energy_kcal for pt in self.points]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(s_vals, e_kcal, "o-", color="#2563eb", lw=2.5, markersize=5, label="IRC Minimum Energy Path")
        ax.axvline(0.0, color="#dc2626", linestyle="--", alpha=0.7, label=f"TS (E = {self.ts_energy:.6f} Ha)")
        ax.axhline(0.0, color="#64748b", linestyle=":", alpha=0.5)

        ax.set_title("Intrinsic Reaction Coordinate (IRC) Profile", fontsize=13, fontweight="bold")
        ax.set_xlabel("Reaction Coordinate $s$ ($\\mathrm{amu}^{1/2} \\cdot \\mathrm{Bohr}$)", fontsize=11)
        ax.set_ylabel("Relative Energy $\\Delta E$ (kcal/mol)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", frameon=True)

        # Annotate barriers
        annot = (
            f"Forward Barrier (dE_f): {self.barrier_forward_kcal:.2f} kcal/mol\n"
            f"Reverse Barrier (dE_r): {self.barrier_reverse_kcal:.2f} kcal/mol\n"
            f"Reaction Energy (dE_rxn): {self.reaction_energy_kcal:+.2f} kcal/mol"
        )
        ax.text(
            0.03, 0.05, annot, transform=ax.transAxes,
            fontsize=9.5, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.9),
        )

        plt.tight_layout()
        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    def summary(self) -> str:
        """Returns structured ASCII report of the reaction pathway."""
        lines = [
            "=" * 76,
            " PHATPSY 2.0 INTRINSIC REACTION COORDINATE (IRC) ANALYSIS",
            "=" * 76,
            f" Transition State Energy:           {self.ts_energy:16.8f} Hartree",
            f" Forward Activation Barrier (dE_f): {self.barrier_forward_kcal:16.2f} kcal/mol",
            f" Reverse Activation Barrier (dE_r): {self.barrier_reverse_kcal:16.2f} kcal/mol",
            f" Reaction Energy (dE_rxn):          {self.reaction_energy_kcal:16.2f} kcal/mol",
            f" Total IRC Pathway Points:          {len(self.points)}",
            f" Forward Branch Converged:          {self.converged_forward}",
            f" Reverse Branch Converged:          {self.converged_reverse}",
        ]
        if self.initial_mode_frequency_cm is not None:
            lines.append(f" Imaginary Mode Frequency:          {self.initial_mode_frequency_cm:16.2f} i cm^-1")
        lines.append("-" * 76)
        lines.append(f"{'Step':>5}  {'Dir':>8}  {'s (amu^1/2·Bohr)':>18}  {'Energy (Hartree)':>18}  {'dE (kcal/mol)':>14}  {'||grad||':>10}")
        lines.append("-" * 76)

        for pt in self.points:
            lines.append(
                f"{pt.step:5d}  {pt.direction:>8}  {pt.s:18.4f}  {pt.energy:18.8f}  {pt.delta_energy_kcal:14.2f}  {pt.grad_norm:10.5f}"
            )
        lines.append("=" * 76)
        return "\n".join(lines)

    def show(self, height: int = 520) -> Any:
        """Displays interactive 3D WebGL IRC visualizer widget in Jupyter/IPython."""
        from phatpsy.visualizer import show_irc
        return show_irc(self, height=height)

    def visualize(self, output_path: Optional[Union[str, Path]] = None, open_browser: bool = False) -> Any:
        """Generates standalone 3D WebGL HTML visualizer for IRC trajectory playback."""
        from phatpsy.visualizer import visualize_irc
        return visualize_irc(self, output_path=output_path, open_browser=open_browser)


# ==============================================================================
# IRC Engine: compute_irc()
# ==============================================================================

def compute_irc(
    deck: CalculationDeck,
    driver: Optional[PhatpsyDriver] = None,
    step_size: float = 0.1,
    max_steps: int = 40,
    direction: str = "both",
    grad_tol: float = 1e-3,
    energy_tol: float = 1e-6,
    initial_displacement: Optional[float] = None,
    normal_mode: Optional[Union[NormalMode, np.ndarray, List[List[float]]]] = None,
    finite_diff_step: float = 0.005,
    warm_restart: bool = True,
    quiet: bool = True,
) -> IRCResult:
    """
    Traces the Intrinsic Reaction Coordinate (IRC) from a transition state along mass-weighted steepest descent.

    Parameters
    ----------
    deck : CalculationDeck
        Transition state calculation deck.
    driver : Optional[PhatpsyDriver]
        PhatpsyDriver instance. Created if None.
    step_size : float
        Arc length integration step size Delta s in amu^(1/2) * Bohr (default: 0.1).
    max_steps : int
        Maximum integration steps per branch (default: 40).
    direction : str
        Integration direction: "both", "forward", or "reverse" (default: "both").
    grad_tol : float
        Cartesian gradient norm convergence threshold in Hartree / Bohr (default: 1e-3).
    energy_tol : float
        Energy difference threshold for termination (default: 1e-6).
    initial_displacement : Optional[float]
        Initial displacement along transition vector in amu^(1/2) * Bohr (defaults to step_size).
    normal_mode : Optional[Union[NormalMode, np.ndarray, List[List[float]]]]
        Normal mode displacement vector for the transition state. If None, automatically
        computed via harmonic vibrational frequency analysis.
    finite_diff_step : float
        Cartesian displacement in Bohr for numerical gradient evaluations (default: 0.005).
    warm_restart : bool
        Whether to warm-start wavefunctions across successive steps (default: True).
    quiet : bool
        Whether to suppress verbose Fortran SCF stdout (default: True).

    Returns
    -------
    IRCResult
        Complete reaction path, forward/reverse points, energy barriers, and trajectory.
    """
    if driver is None:
        driver = PhatpsyDriver(quiet=quiet)

    calc_deck = copy.deepcopy(deck)
    calc_deck.resolve_coordinates()

    # Clear symmetry locks on active unfrozen atoms so coordinates can move freely
    for a in calc_deck.geometry:
        if not a.fixed and not a.virtual:
            a.symmetry = None
            a.symmetry_op = None
            a.symmetry_axis = None
            a.equivalent_to = None

    atoms = calc_deck.geometry
    n_atoms = len(atoms)
    dim = 3 * n_atoms

    # Atomic masses in amu
    masses = np.array([ATOMIC_MASSES_AMU.get(a.element.strip().upper(), 12.0) for a in atoms], dtype=np.float64)
    coord_masses = np.repeat(masses, 3)
    sqrt_m = np.sqrt(coord_masses)
    inv_sqrt_m = 1.0 / sqrt_m

    # Active unfrozen coordinate indices
    active_indices = []
    for i, a in enumerate(atoms):
        if not a.fixed and not a.virtual:
            active_indices.extend([i * 3, i * 3 + 1, i * 3 + 2])
    active_mask = np.zeros(dim, dtype=bool)
    active_mask[active_indices] = True

    coords0 = np.array([a.xyz for a in atoms], dtype=np.float64).flatten()

    last_state: Optional[CalculationState] = None

    def evaluate_energy_and_gradient(c_flat: np.ndarray) -> Tuple[float, np.ndarray, float]:
        """Evaluates electronic energy and numerical Cartesian gradient."""
        nonlocal last_state
        test_atoms = copy.deepcopy(atoms)
        for i in range(n_atoms):
            test_atoms[i].xyz = [float(c_flat[i * 3]), float(c_flat[i * 3 + 1]), float(c_flat[i * 3 + 2])]

        test_deck = copy.deepcopy(calc_deck)
        test_deck.geometry = test_atoms

        if warm_restart and last_state is not None and last_state.converged:
            new_geom = [(a.element, a.xyz[0], a.xyz[1], a.xyz[2]) for a in test_atoms]
            test_deck = last_state.create_restart_deck(new_geometry=new_geom)

        state = driver.run_deck(test_deck, quiet=quiet)
        if state.converged:
            last_state = state
        E = float(state.total_energy)
        virial = float(state.virial_ratio)

        # Numerical Cartesian gradient
        grad = np.zeros(dim, dtype=np.float64)
        h = float(finite_diff_step)
        for idx in active_indices:
            cp = c_flat.copy()
            cp[idx] += h
            atoms_p = copy.deepcopy(atoms)
            for i in range(n_atoms):
                atoms_p[i].xyz = [float(cp[i * 3]), float(cp[i * 3 + 1]), float(cp[i * 3 + 2])]
            dp = copy.deepcopy(calc_deck)
            dp.geometry = atoms_p
            if warm_restart and last_state is not None and last_state.converged:
                dp = last_state.create_restart_deck([(a.element, a.xyz[0], a.xyz[1], a.xyz[2]) for a in atoms_p])
            sp = driver.run_deck(dp, quiet=quiet)
            ep = float(sp.total_energy)

            cm = c_flat.copy()
            cm[idx] -= h
            atoms_m = copy.deepcopy(atoms)
            for i in range(n_atoms):
                atoms_m[i].xyz = [float(cm[i * 3]), float(cm[i * 3 + 1]), float(cm[i * 3 + 2])]
            dm = copy.deepcopy(calc_deck)
            dm.geometry = atoms_m
            if warm_restart and last_state is not None and last_state.converged:
                dm = last_state.create_restart_deck([(a.element, a.xyz[0], a.xyz[1], a.xyz[2]) for a in atoms_m])
            sm = driver.run_deck(dm, quiet=quiet)
            em = float(sm.total_energy)

            grad[idx] = (ep - em) / (2.0 * h)

        return E, grad, virial

    # 1. Reference Transition State Evaluation
    E_ts, grad_ts, virial_ts = evaluate_energy_and_gradient(coords0)
    grad_norm_ts = float(np.linalg.norm(grad_ts))

    def make_geom_dict(c_flat: np.ndarray) -> List[Dict[str, Any]]:
        return [
            {
                "element": atoms[i].element,
                "xyz": [float(c_flat[i * 3]), float(c_flat[i * 3 + 1]), float(c_flat[i * 3 + 2])],
                "fixed": bool(atoms[i].fixed),
                "virtual": bool(atoms[i].virtual),
            }
            for i in range(n_atoms)
        ]

    ts_pt = IRCPoint(
        step=0,
        direction="ts",
        s=0.0,
        energy=E_ts,
        delta_energy_kcal=0.0,
        grad_norm=grad_norm_ts,
        geometry=make_geom_dict(coords0),
        virial_ratio=virial_ts,
    )

    # 2. Determine Transition Vector (Mode of Imaginary Frequency)
    imag_freq_cm: Optional[float] = None
    if normal_mode is not None:
        if isinstance(normal_mode, NormalMode):
            disp = np.array(normal_mode.displacements, dtype=np.float64).flatten()
            imag_freq_cm = normal_mode.frequency_cm
        else:
            disp = np.array(normal_mode, dtype=np.float64).flatten()
    else:
        # Automatically compute vibrational frequencies to find imaginary mode
        vib_res = compute_vibrations(
            deck=calc_deck,
            driver=driver,
            step_size=finite_diff_step,
            method="forward",
            project_tr=True,
            quiet=quiet,
        )
        if vib_res.num_imaginary > 0:
            # Pick the lowest (most imaginary) normal mode
            target_mode = vib_res.imaginary_modes[0]
            disp = np.array(target_mode.displacements, dtype=np.float64).flatten()
            imag_freq_cm = target_mode.frequency_cm
        elif vib_res.vibrational_modes:
            # Fallback to lowest vibrational mode
            target_mode = vib_res.vibrational_modes[0]
            disp = np.array(target_mode.displacements, dtype=np.float64).flatten()
            imag_freq_cm = target_mode.frequency_cm
        elif vib_res.modes:
            # Fallback to last mode
            target_mode = vib_res.modes[-1]
            disp = np.array(target_mode.displacements, dtype=np.float64).flatten()
            imag_freq_cm = target_mode.frequency_cm
        else:
            disp = np.zeros(dim, dtype=np.float64)
            disp[active_indices] = 1.0 / math.sqrt(len(active_indices))

    # Zero out displacements on inactive (frozen/virtual) atoms
    disp[~active_mask] = 0.0

    # Convert Cartesian displacement disp to mass-weighted transition vector L_mw:
    # L_mw = disp * sqrt(M)
    L_mw = disp * sqrt_m
    norm_L = np.linalg.norm(L_mw)
    if norm_L > 1e-12:
        L_mw /= norm_L
    else:
        # Fallback random active perturbation
        L_mw[active_indices] = 1.0 / math.sqrt(len(active_indices))

    # Delta s_0 initial step size
    ds0 = float(initial_displacement if initial_displacement is not None else step_size)

    # 3. Path Integration Function (Mass-Weighted Steepest Descent)
    def trace_branch(branch_dir: int, branch_label: str) -> Tuple[List[IRCPoint], bool]:
        """
        Traces one branch (+1 for forward, -1 for reverse) down the energy valley.
        """
        branch_points: List[IRCPoint] = []
        converged = False

        # Initial step from TS in mass-weighted coordinates:
        # dx_0 = sign * ds0 * L_mw
        # In Cartesian: dr_0 = dx_0 / sqrt(M)
        dr0 = (branch_dir * ds0) * (L_mw * inv_sqrt_m)
        c_curr = coords0 + dr0
        s_curr = branch_dir * ds0

        E_curr, grad_curr, virial_curr = evaluate_energy_and_gradient(c_curr)
        g_norm = float(np.linalg.norm(grad_curr))

        pt0 = IRCPoint(
            step=1,
            direction=branch_label,
            s=s_curr,
            energy=E_curr,
            delta_energy_kcal=(E_curr - E_ts) * HARTREE_TO_KCAL,
            grad_norm=g_norm,
            geometry=make_geom_dict(c_curr),
            virial_ratio=virial_curr,
        )
        branch_points.append(pt0)

        current_ds = float(step_size)

        for step_idx in range(2, max_steps + 1):
            if g_norm <= grad_tol:
                converged = True
                break

            # Mass-weighted gradient: g_x = grad_r / sqrt(M)
            gx = grad_curr * inv_sqrt_m
            gx[~active_mask] = 0.0
            norm_gx = float(np.linalg.norm(gx))
            if norm_gx < 1e-12:
                converged = True
                break

            # Normalized descent direction in mass-weighted space: u = -gx / ||gx||
            u = -gx / norm_gx

            # Line search for energy reduction
            step_accepted = False
            trial_ds = current_ds
            for _ in range(5):
                # Displacement in mass-weighted: dx = trial_ds * u
                # In Cartesian: dr = dx / sqrt(M) = trial_ds * (u / sqrt(M))
                dr_trial = trial_ds * (u * inv_sqrt_m)
                c_trial = c_curr + dr_trial

                E_trial, grad_trial, virial_trial = evaluate_energy_and_gradient(c_trial)

                if E_trial <= E_curr + 1e-7:
                    # Step accepted!
                    c_curr = c_trial
                    E_prev = E_curr
                    E_curr = E_trial
                    grad_curr = grad_trial
                    virial_curr = virial_trial
                    g_norm = float(np.linalg.norm(grad_curr))
                    s_curr += branch_dir * trial_ds
                    step_accepted = True

                    # Adapt step size slightly upward if decrease was smooth
                    current_ds = min(trial_ds * 1.1, float(step_size) * 1.5)
                    break
                else:
                    # Backtrack
                    trial_ds *= 0.5
                    if trial_ds < 1e-4:
                        break

            if not step_accepted:
                # Reached local energy minimum at bottom of valley
                converged = True
                break

            pt = IRCPoint(
                step=step_idx,
                direction=branch_label,
                s=s_curr,
                energy=E_curr,
                delta_energy_kcal=(E_curr - E_ts) * HARTREE_TO_KCAL,
                grad_norm=g_norm,
                geometry=make_geom_dict(c_curr),
                virial_ratio=virial_curr,
            )
            branch_points.append(pt)

            if abs(E_curr - E_prev) < energy_tol:
                converged = True
                break

        if g_norm <= grad_tol:
            converged = True

        return branch_points, converged

    # 4. Execute Branches
    forward_pts: List[IRCPoint] = []
    reverse_pts: List[IRCPoint] = []
    conv_fwd = False
    conv_rev = False

    dir_lower = direction.strip().lower()
    if dir_lower in ("both", "forward"):
        forward_pts, conv_fwd = trace_branch(branch_dir=+1, branch_label="product")

    if dir_lower in ("both", "reverse"):
        reverse_pts, conv_rev = trace_branch(branch_dir=-1, branch_label="reactant")

    # 5. Assemble Full Continuous Path: Reactants -> TS -> Products
    # Reverse points are sorted from most negative s to 0
    ordered_rev = list(reversed(reverse_pts))
    full_path: List[IRCPoint] = []
    step_counter = 1

    for pt in ordered_rev:
        pt.step = step_counter
        full_path.append(pt)
        step_counter += 1

    ts_pt.step = step_counter
    full_path.append(ts_pt)
    step_counter += 1

    for pt in forward_pts:
        pt.step = step_counter
        full_path.append(pt)
        step_counter += 1

    # 6. Barrier Heights & Reaction Energies
    # Reactant energy is the lowest point along reactant branch (or TS if empty)
    E_reactant = min([pt.energy for pt in reverse_pts]) if reverse_pts else E_ts
    # Product energy is the lowest point along product branch (or TS if empty)
    E_product = min([pt.energy for pt in forward_pts]) if forward_pts else E_ts

    barrier_fwd = (E_ts - E_reactant) * HARTREE_TO_KCAL
    barrier_rev = (E_ts - E_product) * HARTREE_TO_KCAL
    delta_rxn = (E_product - E_reactant) * HARTREE_TO_KCAL

    return IRCResult(
        points=full_path,
        forward_points=forward_pts,
        reverse_points=reverse_pts,
        ts_point=ts_pt,
        ts_energy=E_ts,
        barrier_forward_kcal=barrier_fwd,
        barrier_reverse_kcal=barrier_rev,
        reaction_energy_kcal=delta_rxn,
        converged_forward=conv_fwd,
        converged_reverse=conv_rev,
        initial_mode_frequency_cm=imag_freq_cm,
    )
