# phatpsy/dimer.py
"""
Dimer Method for Transition State (Saddle Point) Refinement in PHATPSY 2.0.

Gradient-only saddle-point searching without constructing or storing Hessians:
- Henkelman-Jónsson (1999) & Heyden-Bell (2005) dimer rotation and translation.
- Rotates the dimer axis N to find the lowest-curvature mode using single-trial analytic rotation.
- Inverts the force component along N while relaxing all orthogonal modes:
    F_eff = F_0 - 2 * (F_0 . N) * N   (for curvature C < 0)
    F_eff = - (F_0 . N) * N           (for curvature C > 0, uphill climb)
- Strictly gradient-only: 3 to 4 gradient evaluations per cycle.
- Standalone search from reactant / perturbed geometry or refinement handoff from CI-NEB.
- Respects frozen substrate atoms (fixed=True) and ghost centers (is_virtual=True).
- Includes the analytical Müller-Brown testbed for benchmark validation.
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
# Analytical Benchmark: Müller-Brown Potential Surface
# ==============================================================================

class MullerBrownSurface:
    """
    Standard 2D Müller-Brown potential energy surface and analytic gradients.
    Used for unit testing and benchmarking saddle-point search algorithms.
    """
    A = np.array([-200.0, -100.0, -170.0, 15.0], dtype=np.float64)
    a = np.array([-1.0, -1.0, -6.5, 0.7], dtype=np.float64)
    b = np.array([0.0, 0.0, 11.0, 0.6], dtype=np.float64)
    c = np.array([-10.0, -10.0, -6.5, 0.7], dtype=np.float64)
    x0 = np.array([1.0, 0.0, -0.5, -1.0], dtype=np.float64)
    y0 = np.array([0.0, 0.5, 1.5, 1.0], dtype=np.float64)

    # Known analytical stationary points
    SADDLE_1 = np.array([-0.82200156, 0.62431279], dtype=np.float64)  # E ≈ -40.665
    SADDLE_2 = np.array([0.21248658, 0.29298832], dtype=np.float64)   # E ≈ -72.249
    MIN_1 = np.array([-0.558, 1.442], dtype=np.float64)               # E ≈ -146.7
    MIN_2 = np.array([0.623, 0.028], dtype=np.float64)                # E ≈ -108.2
    MIN_3 = np.array([-0.050, 0.467], dtype=np.float64)               # E ≈ -80.5

    @classmethod
    def energy(cls, r: np.ndarray) -> float:
        """Evaluates Müller-Brown energy at r = [x, y]."""
        x, y = float(r[0]), float(r[1])
        dx = x - cls.x0
        dy = y - cls.y0
        arg = cls.a * dx**2 + cls.b * dx * dy + cls.c * dy**2
        return float(np.sum(cls.A * np.exp(arg)))

    @classmethod
    def gradient(cls, r: np.ndarray) -> np.ndarray:
        """Evaluates analytical gradient dV/dr at r = [x, y]."""
        x, y = float(r[0]), float(r[1])
        dx = x - cls.x0
        dy = y - cls.y0
        arg = cls.a * dx**2 + cls.b * dx * dy + cls.c * dy**2
        exp_term = cls.A * np.exp(arg)
        gx = np.sum(exp_term * (2.0 * cls.a * dx + cls.b * dy))
        gy = np.sum(exp_term * (cls.b * dx + 2.0 * cls.c * dy))
        return np.array([gx, gy], dtype=np.float64)

    @classmethod
    def force(cls, r: np.ndarray) -> np.ndarray:
        """Evaluates force F = -dV/dr at r = [x, y]."""
        return -cls.gradient(r)


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class DimerStep:
    """A single cycle record in the Dimer optimization trajectory."""
    step: int
    energy: float
    curvature: float
    force_eff_norm: float
    rot_torque_norm: float
    coordinates: List[List[float]]
    orientation: List[List[float]]


@dataclass
class DimerResult:
    """Comprehensive result of a Dimer transition state refinement calculation."""
    converged: bool
    saddle_energy: float
    initial_energy: float
    num_cycles: int
    num_gradient_evals: int
    final_curvature: float
    final_force_norm: float
    trajectory: List[DimerStep]
    saddle_deck: CalculationDeck
    saddle_state: Optional[CalculationState] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts result to a structured JSON-serializable dictionary."""
        atoms = self.saddle_deck.geometry
        return {
            "converged": self.converged,
            "saddle_energy": round(float(self.saddle_energy), 8),
            "initial_energy": round(float(self.initial_energy), 8),
            "energy_change_kcal": round((self.saddle_energy - self.initial_energy) * HARTREE_TO_KCAL, 4),
            "num_cycles": self.num_cycles,
            "num_gradient_evals": self.num_gradient_evals,
            "final_curvature": round(float(self.final_curvature), 6),
            "final_force_norm": round(float(self.final_force_norm), 6),
            "saddle_geometry": [
                {
                    "element": a.element,
                    "xyz": [round(float(c), 6) for c in a.xyz],
                    "fixed": bool(a.fixed),
                    "virtual": bool(a.virtual),
                }
                for a in atoms
            ],
            "trajectory": [
                {
                    "step": s.step,
                    "energy": round(float(s.energy), 8),
                    "curvature": round(float(s.curvature), 6),
                    "force_eff_norm": round(float(s.force_eff_norm), 6),
                    "rot_torque_norm": round(float(s.rot_torque_norm), 6),
                    "coordinates": [[round(float(c), 6) for c in pt] for pt in s.coordinates],
                }
                for s in self.trajectory
            ],
        }

    def save_json(self, filepath: Union[str, Path]) -> None:
        """Saves Dimer refinement results to a JSON file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_xyz(self, filepath: Union[str, Path]) -> None:
        """Exports multi-frame XYZ trajectory of the saddle-point refinement."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        atoms = self.saddle_deck.geometry
        with open(p, "w", encoding="utf-8") as f:
            for s in self.trajectory:
                f.write(f"{len(atoms)}\n")
                f.write(
                    f"Dimer cycle={s.step} E={s.energy:.8f} Ha C={s.curvature:+.4f} ||F_eff||={s.force_eff_norm:.5f}\n"
                )
                for i, a in enumerate(atoms):
                    el = a.element
                    xyz_ang = [c * BOHR_TO_ANGSTROM for c in s.coordinates[i]]
                    f.write(f"{el:<3} {xyz_ang[0]:12.6f} {xyz_ang[1]:12.6f} {xyz_ang[2]:12.6f}\n")

    def visualize(
        self,
        output_path: Optional[Union[str, Path]] = None,
        open_browser: bool = False,
        title: Optional[str] = None,
    ) -> str:
        """Renders standalone HTML with interactive 3D WebGL Dimer refinement player."""
        from phatpsy.visualizer import visualize_dimer
        return visualize_dimer(self, output_path=output_path, open_browser=open_browser, title=title)

    def show(self, height: int = 550, title: Optional[str] = None) -> Any:
        """Displays interactive 3D Dimer visualizer inside Jupyter notebooks."""
        from phatpsy.visualizer import show_dimer
        return show_dimer(self, height=height, title=title)

    def summary(self) -> str:
        """Returns structured text summary report of the dimer calculation."""
        lines = [
            "=" * 72,
            " PHATPSY 2.0 DIMER TRANSITION-STATE REFINEMENT REPORT",
            "=" * 72,
            f" Convergence Status:            {'CONVERGED' if self.converged else 'FAILED / MAX CYCLES'}",
            f" Saddle Point Energy:           {self.saddle_energy:18.8f} Hartree",
            f" Initial Energy:                {self.initial_energy:18.8f} Hartree",
            f" Energy Change:                 {(self.saddle_energy - self.initial_energy) * HARTREE_TO_KCAL:18.4f} kcal/mol",
            f" Final Curvature (C):           {self.final_curvature:18.6f} Hartree/Bohr^2",
            f" Effective Force Norm:          {self.final_force_norm:18.6f} Hartree/Bohr",
            f" Total Dimer Cycles:            {self.num_cycles:8d}",
            f" Total Gradient Evaluations:    {self.num_gradient_evals:8d}",
            "-" * 72,
            f"{'Cycle':>6}  {'Energy (Hartree)':>18}  {'Curvature (C)':>16}  {'||F_eff||':>12}  {'||F_rot||':>12}",
            "-" * 72,
        ]
        for s in self.trajectory:
            lines.append(
                f"{s.step:6d}  {s.energy:18.8f}  {s.curvature:16.4f}  {s.force_eff_norm:12.5f}  {s.rot_torque_norm:12.5f}"
            )
        lines.append("=" * 72)
        return "\n".join(lines)


# ==============================================================================
# Vector Dimer Engine (Pure Gradient Optimizer)
# ==============================================================================

def dimer_optimize_vector(
    force_func: Callable[[np.ndarray], np.ndarray],
    energy_func: Callable[[np.ndarray], float],
    x0: np.ndarray,
    n0: Optional[np.ndarray] = None,
    dimer_dist: float = 0.005,
    max_cycles: int = 60,
    force_tol: float = 1e-3,
    rot_tol: float = 1e-3,
    max_rot_per_cycle: int = 3,
    trial_rot_angle: float = 0.01,
    dt: float = 0.02,
    max_step: float = 0.05,
    active_mask: Optional[np.ndarray] = None,
) -> Tuple[bool, np.ndarray, np.ndarray, float, float, int, int, List[Dict[str, Any]]]:
    """
    Pure gradient-only Dimer optimization on an arbitrary potential energy surface.
    Does not evaluate or store Hessians.

    Parameters
    ----------
    force_func : Callable[[np.ndarray], np.ndarray]
        Returns force vector F(x) = -grad V(x).
    energy_func : Callable[[np.ndarray], float]
        Returns scalar potential energy V(x).
    x0 : np.ndarray
        Initial coordinate vector.
    n0 : Optional[np.ndarray]
        Initial dimer orientation unit vector. Generated randomly if None.
    dimer_dist : float
        Dimer half-length Delta R (default: 0.005).
    max_cycles : int
        Maximum dimer translation cycles (default: 60).
    force_tol : float
        Convergence threshold for ||F_eff|| (default: 1e-3).
    rot_tol : float
        Convergence threshold for rotational torque ||F_R^perp|| (default: 1e-3).
    max_rot_per_cycle : int
        Maximum rotation iterations per translation cycle (default: 3).
    trial_rot_angle : float
        Trial rotation angle Delta theta in radians (default: 0.01).
    dt : float
        QuickMin translation time step (default: 0.02).
    max_step : float
        Maximum translation displacement clamp (default: 0.05).
    active_mask : Optional[np.ndarray]
        Boolean mask of active coordinates (unfrozen).

    Returns
    -------
    Tuple:
        (converged, x_saddle, n_final, final_energy, final_curvature, num_cycles, num_grad_evals, traj_records)
    """
    dim = len(x0)
    R = x0.copy().astype(np.float64)
    dR = float(dimer_dist)

    if active_mask is None:
        mask = np.ones(dim, dtype=bool)
    else:
        mask = np.asarray(active_mask, dtype=bool)

    if n0 is not None:
        N = n0.copy().astype(np.float64)
        N[~mask] = 0.0
    else:
        # Initial guess along active degrees of freedom
        N = np.zeros(dim, dtype=np.float64)
        active_indices = np.where(mask)[0]
        if len(active_indices) > 0:
            N[active_indices] = 1.0 / math.sqrt(len(active_indices))
        else:
            N[0] = 1.0

    norm_N = np.linalg.norm(N)
    if norm_N > 1e-12:
        N /= norm_N

    grad_evals = 0

    def eval_force(x_pos: np.ndarray) -> np.ndarray:
        nonlocal grad_evals
        grad_evals += 1
        f = force_func(x_pos)
        f[~mask] = 0.0
        return f

    v = np.zeros_like(R)
    converged = False
    traj_records: List[Dict[str, Any]] = []

    last_C = 0.0
    last_F_eff_norm = 0.0

    for cycle in range(1, max_cycles + 1):
        # 1. Evaluate force at center R
        F0 = eval_force(R)
        E0 = energy_func(R)

        # 2. Forward and backward force evaluations along current orientation N
        F1 = eval_force(R + dR * N)
        F2 = eval_force(R - dR * N)

        # Rotational force on dimer FR = 0.5 * (F1 - F2)
        FR = 0.5 * (F1 - F2)
        FR[~mask] = 0.0
        FR_perp = FR - np.dot(FR, N) * N
        FR_perp[~mask] = 0.0
        norm_FR_perp = float(np.linalg.norm(FR_perp))

        # Curvature along unrotated N
        C0 = float(np.dot(F2 - F1, N) / (2.0 * dR))

        # 3. Heyden-Bell single-trial rotation step (4th gradient evaluation)
        if norm_FR_perp > rot_tol and max_rot_per_cycle > 0:
            Theta = FR_perp / norm_FR_perp
            dtheta = float(trial_rot_angle)

            # Trial rotation
            N_trial = N * math.cos(dtheta) + Theta * math.sin(dtheta)
            N_trial[~mask] = 0.0
            norm_Nt = np.linalg.norm(N_trial)
            if norm_Nt > 1e-12:
                N_trial /= norm_Nt

            # 4th gradient evaluation: F1 at trial orientation
            # (F2_trial is estimated via Heyden-Bell relation: F2_trial ≈ 2*F0 - F1_trial)
            F1_trial = eval_force(R + dR * N_trial)
            FR_trial = F1_trial - F0
            FR_trial[~mask] = 0.0

            Theta_trial = -N * math.sin(dtheta) + Theta * math.cos(dtheta)
            Theta_trial[~mask] = 0.0

            g0 = -float(np.dot(FR, Theta))
            g_trial = -float(np.dot(FR_trial, Theta_trial))

            B = g0
            sin_2dt = math.sin(2.0 * dtheta)
            if abs(sin_2dt) > 1e-8:
                A = (B * math.cos(2.0 * dtheta) - g_trial) / sin_2dt
            else:
                A = 0.0

            phi = 0.5 * math.atan2(B, A)
            d2 = -2.0 * A * math.cos(2.0 * phi) - 2.0 * B * math.sin(2.0 * phi)
            if d2 < 0.0:
                phi += 0.5 * math.pi

            N_new = N * math.cos(phi) + Theta * math.sin(phi)
            N_new[~mask] = 0.0
            norm_N = np.linalg.norm(N_new)
            if norm_N > 1e-12:
                N = N_new / norm_N

            # Curvature along rotated N (analytical Fourier form)
            delta_C = (A * (math.cos(2.0 * phi) - 1.0) + B * math.sin(2.0 * phi)) / (dR * dR)
            C = C0 + delta_C
        else:
            C = C0

        # 4. Effective Translation Force
        F0_dot_N = float(np.dot(F0, N))
        if C < 0.0:
            # Below bifurcation point: invert force along lowest curvature mode
            F_eff = F0 - 2.0 * F0_dot_N * N
        else:
            # Positive curvature: uphill climb along N to escape minimum
            F_eff = -F0_dot_N * N
        F_eff[~mask] = 0.0
        norm_F_eff = float(np.linalg.norm(F_eff))

        last_C = C
        last_F_eff_norm = norm_F_eff

        traj_records.append({
            "step": cycle,
            "energy": E0,
            "curvature": C,
            "force_eff_norm": norm_F_eff,
            "rot_torque_norm": norm_FR_perp,
            "coordinates": R.copy(),
            "orientation": N.copy(),
        })

        # 5. Check Convergence (Stationary point reached with negative curvature)
        if norm_F_eff <= force_tol and C < 0.0:
            converged = True
            break

        # 6. QuickMin Translation Step
        v_dot_F = float(np.dot(v, F_eff))
        if v_dot_F > 0.0:
            v = (v_dot_F / (norm_F_eff**2)) * F_eff
        else:
            v = np.zeros_like(R)
        v += dt * F_eff

        step_vec = dt * v
        norm_step = float(np.linalg.norm(step_vec))
        if norm_step > max_step:
            step_vec = (max_step / norm_step) * step_vec

        R += step_vec
        R[~mask] = x0[~mask]

    E_final = energy_func(R)
    return converged, R, N, E_final, last_C, len(traj_records), grad_evals, traj_records


# ==============================================================================
# Molecular Dimer Refinement: dimer_search()
# ==============================================================================

def dimer_search(
    deck: CalculationDeck,
    driver: Optional[PhatpsyDriver] = None,
    initial_direction: Optional[Union[np.ndarray, List[List[float]]]] = None,
    dimer_dist: float = 0.005,
    finite_diff_step: float = 0.001,
    max_cycles: int = 40,
    force_tol: float = 1e-3,
    rot_tol: float = 1e-3,
    max_rot_per_cycle: int = 3,
    dt: float = 0.02,
    max_step: float = 0.04,
    warm_restart: bool = True,
    quiet: bool = True,
) -> DimerResult:
    """
    Refines a molecular transition state structure using the gradient-only Dimer method.
    No Cartesian Hessian is constructed or stored during optimization.

    Parameters
    ----------
    deck : CalculationDeck
        Input geometry near saddle point (or initial guess).
    driver : Optional[PhatpsyDriver]
        PhatpsyDriver instance. Created if None.
    initial_direction : Optional[Union[np.ndarray, List[List[float]]]]
        Initial dimer orientation vector N_0. If None, derived from active coordinates.
    dimer_dist : float
        Dimer separation Delta R in Bohr (default: 0.005).
    finite_diff_step : float
        Coordinate displacement step h for numerical gradient evaluation (default: 0.001 Bohr).
    max_cycles : int
        Maximum translation cycles (default: 40).
    force_tol : float
        Effective force convergence threshold in Hartree / Bohr (default: 1e-3).
    rot_tol : float
        Rotational torque convergence threshold (default: 1e-3).
    max_rot_per_cycle : int
        Maximum rotation iterations per translation step (default: 3).
    dt : float
        QuickMin translation time step (default: 0.02).
    max_step : float
        Maximum translation displacement clamp per step in Bohr (default: 0.04).
    warm_restart : bool
        Whether to warm-start SCF wavefunctions across steps (default: True).
    quiet : bool
        Whether to silence verbose Fortran SCF stdout (default: True).

    Returns
    -------
    DimerResult
        Saddle point deck, energies, curvature, trajectory, and evaluation counts.
    """
    if driver is None:
        driver = PhatpsyDriver(quiet=quiet)

    calc_deck = copy.deepcopy(deck)
    calc_deck.resolve_coordinates()

    # Clear symmetry locks on active unfrozen atoms
    for a in calc_deck.geometry:
        if not a.fixed and not a.virtual:
            a.symmetry = None
            a.symmetry_op = None
            a.symmetry_axis = None
            a.equivalent_to = None

    atoms = calc_deck.geometry
    n_atoms = len(atoms)
    dim = 3 * n_atoms

    # Active mask (unfrozen, non-virtual)
    active_mask = np.zeros(dim, dtype=bool)
    active_indices = []
    for i, a in enumerate(atoms):
        if not a.fixed and not a.virtual:
            active_mask[i * 3: i * 3 + 3] = True
            active_indices.extend([i * 3, i * 3 + 1, i * 3 + 2])

    coords0 = np.array([a.xyz for a in atoms], dtype=np.float64).flatten()

    last_state: Optional[CalculationState] = None

    def evaluate_energy(c_flat: np.ndarray) -> float:
        nonlocal last_state
        test_atoms = copy.deepcopy(atoms)
        for i in range(n_atoms):
            test_atoms[i].xyz = [float(c_flat[i * 3]), float(c_flat[i * 3 + 1]), float(c_flat[i * 3 + 2])]
        d = copy.deepcopy(calc_deck)
        d.geometry = test_atoms
        if warm_restart and last_state is not None and last_state.converged:
            d = last_state.create_restart_deck([(a.element, a.xyz[0], a.xyz[1], a.xyz[2]) for a in test_atoms])
        st = driver.run_deck(d, quiet=quiet)
        if st.converged:
            last_state = st
        return float(st.total_energy)

    def evaluate_force(c_flat: np.ndarray) -> np.ndarray:
        """Evaluates numerical Cartesian force F = -dE/dr."""
        nonlocal last_state
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

        return -grad

    E_init = evaluate_energy(coords0)

    # Initial orientation vector
    if initial_direction is not None:
        n0_vec = np.array(initial_direction, dtype=np.float64).flatten()
    else:
        n0_vec = None

    conv, r_saddle, n_saddle, e_saddle, c_saddle, n_cyc, n_evals, traj_recs = dimer_optimize_vector(
        force_func=evaluate_force,
        energy_func=evaluate_energy,
        x0=coords0,
        n0=n0_vec,
        dimer_dist=dimer_dist,
        max_cycles=max_cycles,
        force_tol=force_tol,
        rot_tol=rot_tol,
        max_rot_per_cycle=max_rot_per_cycle,
        dt=dt,
        max_step=max_step,
        active_mask=active_mask,
    )

    # Build saddle CalculationDeck
    saddle_atoms = copy.deepcopy(atoms)
    for i in range(n_atoms):
        saddle_atoms[i].xyz = [float(r_saddle[i * 3]), float(r_saddle[i * 3 + 1]), float(r_saddle[i * 3 + 2])]
    saddle_deck = copy.deepcopy(calc_deck)
    saddle_deck.geometry = saddle_atoms

    # Convert trajectory records into DimerStep objects
    dimer_steps: List[DimerStep] = []
    for rec in traj_recs:
        c_step = rec["coordinates"]
        n_step = rec["orientation"]
        coords_list = [
            [float(c_step[i * 3]), float(c_step[i * 3 + 1]), float(c_step[i * 3 + 2])]
            for i in range(n_atoms)
        ]
        n_list = [
            [float(n_step[i * 3]), float(n_step[i * 3 + 1]), float(n_step[i * 3 + 2])]
            for i in range(n_atoms)
        ]
        dimer_steps.append(
            DimerStep(
                step=rec["step"],
                energy=rec["energy"],
                curvature=rec["curvature"],
                force_eff_norm=rec["force_eff_norm"],
                rot_torque_norm=rec["rot_torque_norm"],
                coordinates=coords_list,
                orientation=n_list,
            )
        )

    return DimerResult(
        converged=conv,
        saddle_energy=e_saddle,
        initial_energy=E_init,
        num_cycles=n_cyc,
        num_gradient_evals=n_evals,
        final_curvature=c_saddle,
        final_force_norm=traj_recs[-1]["force_eff_norm"] if traj_recs else 0.0,
        trajectory=dimer_steps,
        saddle_deck=saddle_deck,
        saddle_state=last_state,
    )


# ==============================================================================
# Handoff from CI-NEB: refine_neb_saddle()
# ==============================================================================

def refine_neb_saddle(
    neb_result: Any,
    driver: Optional[PhatpsyDriver] = None,
    dimer_dist: float = 0.005,
    max_cycles: int = 30,
    force_tol: float = 1e-3,
    rot_tol: float = 1e-3,
    quiet: bool = True,
) -> DimerResult:
    """
    Refines the approximate saddle point located by Climbing-Image NEB to exact transition-state precision.

    Parameters
    ----------
    neb_result : NEBResult
        Converged or intermediate result from neb_search().
    driver : Optional[PhatpsyDriver]
        PhatpsyDriver instance.
    dimer_dist : float
        Dimer separation in Bohr (default: 0.005).
    max_cycles : int
        Maximum dimer cycles (default: 30).
    force_tol : float
        Convergence threshold in Hartree/Bohr (default: 1e-3).
    rot_tol : float
        Rotational torque tolerance (default: 1e-3).
    quiet : bool
        Whether to silence SCF stdout (default: True).

    Returns
    -------
    DimerResult
        Refined saddle point geometry, curvature, and energy.
    """
    if isinstance(neb_result, (str, Path)):
        with open(neb_result, "r", encoding="utf-8") as f:
            neb_result = json.load(f)

    if isinstance(neb_result, dict):
        ts_idx = neb_result.get("transition_state_index", 0)
        raw_images = neb_result.get("images", [])
        images = []
        for img in raw_images:
            atom_list = [
                AtomConfig(
                    element=a["element"],
                    xyz=a["xyz"],
                    fixed=a.get("fixed", False),
                    virtual=a.get("virtual", False),
                )
                for a in img.get("atoms", [])
            ]
            images.append(CalculationDeck(geometry=atom_list))
    else:
        ts_idx = getattr(neb_result, "transition_state_index", int(np.argmax(neb_result.energies)))
        images = neb_result.images

    ts_deck = images[ts_idx]

    # Initial dimer axis N_0 derived from NEB tangent along the pathway
    if len(images) > 1:
        if ts_idx == 0:
            deck_next = images[1]
            deck_prev = ts_deck
        elif ts_idx == len(images) - 1:
            deck_next = ts_deck
            deck_prev = images[-2]
        else:
            deck_next = images[ts_idx + 1]
            deck_prev = images[ts_idx - 1]

        r_next = np.array([a.xyz for a in deck_next.geometry], dtype=np.float64).flatten()
        r_prev = np.array([a.xyz for a in deck_prev.geometry], dtype=np.float64).flatten()
        tangent = r_next - r_prev
        norm_t = np.linalg.norm(tangent)
        if norm_t > 1e-12:
            n0 = tangent / norm_t
        else:
            n0 = None
    else:
        n0 = None

    return dimer_search(
        deck=ts_deck,
        driver=driver,
        initial_direction=n0,
        dimer_dist=dimer_dist,
        max_cycles=max_cycles,
        force_tol=force_tol,
        rot_tol=rot_tol,
        quiet=quiet,
    )
