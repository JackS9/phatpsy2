# phatpsy/optimize.py
"""
Ground-state geometry optimization and structure relaxation for PHATPSY 2.0.
Minimizes total molecular energy using BFGS / L-BFGS-B or finite-difference gradients,
respecting fixed/virtual coordination centers and warm-restarting calculation states.
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


@dataclass
class OptimizationStep:
    """A single step in a geometry optimization trajectory."""
    step: int
    energy: float
    virial_ratio: float
    grad_norm: float
    coordinates: List[List[float]]


@dataclass
class OptimizationResult:
    """Result of a geometry optimization calculation."""
    converged: bool
    initial_energy: float
    final_energy: float
    energy_change: float
    num_steps: int
    trajectory: List[OptimizationStep]
    optimized_deck: CalculationDeck
    final_state: CalculationState

    def to_dict(self) -> Dict[str, Any]:
        """Convert optimization result to a JSON-serializable dictionary."""
        return {
            "converged": bool(self.converged),
            "initial_energy": round(float(self.initial_energy), 8),
            "final_energy": round(float(self.final_energy), 8),
            "energy_change": round(float(self.energy_change), 8),
            "num_steps": int(self.num_steps),
            "trajectory": [
                {
                    "step": s.step,
                    "energy": round(float(s.energy), 8),
                    "virial_ratio": round(float(s.virial_ratio), 6),
                    "grad_norm": round(float(s.grad_norm), 6),
                    "coordinates": [[round(float(c), 6) for c in pt] for pt in s.coordinates],
                }
                for s in self.trajectory
            ],
            "optimized_geometry": [
                {
                    "element": a.element,
                    "xyz": [round(float(c), 6) for c in a.xyz],
                    "fixed": bool(a.fixed),
                    "virtual": bool(a.virtual),
                }
                for a in self.optimized_deck.atoms
            ],
        }

    def save_json(self, filepath: Union[str, Path]) -> None:
        """Saves optimization result and trajectory to JSON."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def plot_convergence(self, output_path: Optional[Union[str, Path]] = None, show: bool = False) -> Any:
        """Plots energy and gradient norm convergence history."""
        import matplotlib.pyplot as plt

        steps = [s.step for s in self.trajectory]
        enes = [s.energy for s in self.trajectory]
        grads = [s.grad_norm for s in self.trajectory]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        ax1.plot(steps, enes, "o-", color="#2563eb", lw=2, markersize=5)
        ax1.set_title("Energy Minimization Profile", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Optimization Step", fontsize=11)
        ax1.set_ylabel("Total Energy (Hartree)", fontsize=11)
        ax1.grid(True, linestyle=":", alpha=0.6)

        ax2.semilogy(steps, grads, "s-", color="#dc2626", lw=2, markersize=5)
        ax2.set_title("Gradient Norm Convergence", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Optimization Step", fontsize=11)
        ax2.set_ylabel("||grad E|| (Hartree / Bohr)", fontsize=11)
        ax2.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_p, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig


def optimize_geometry(
    deck: CalculationDeck,
    driver: Optional[PhatpsyDriver] = None,
    max_steps: int = 30,
    grad_tol: float = 1e-3,
    energy_tol: float = 1e-5,
    finite_diff_step: float = 0.005,
    step_limit: float = 0.15,
    warm_restart: bool = True,
) -> OptimizationResult:
    """
    Optimizes molecular geometry to locate ground-state energy minima using BFGS quasi-Newton relaxation.

    Args:
        deck: Initial CalculationDeck.
        driver: PhatpsyDriver instance (created if None).
        max_steps: Maximum optimization cycles (default: 30).
        grad_tol: Maximum Cartesian gradient component threshold (Hartree/Bohr).
        energy_tol: Energy difference convergence threshold (Hartree).
        finite_diff_step: Coordinate displacement step for numerical gradient (Bohr).
        step_limit: Maximum displacement permitted per step (trust-region clamp).
        warm_restart: Whether to seed successive steps with converged wavefunctions.

    Returns:
        OptimizationResult with trajectory and optimized deck.
    """
    if driver is None:
        driver = PhatpsyDriver()

    current_deck = copy.deepcopy(deck)
    atoms = current_deck.atoms
    n_atoms = len(atoms)

    # Identify free (unfrozen) coordinate degrees of freedom
    # Each non-fixed, non-virtual atom contributes 3 degrees of freedom
    free_indices = [i for i, a in enumerate(atoms) if not a.fixed and not a.virtual and a.symmetry is None]
    if not free_indices:
        # All atoms fixed; return initial state
        state = driver.run_deck(current_deck)
        step0 = OptimizationStep(0, state.total_energy, state.virial_ratio, 0.0, [a.xyz for a in atoms])
        return OptimizationResult(True, state.total_energy, state.total_energy, 0.0, 0, [step0], current_deck, state)

    def pack_free_coords(atom_list: List[AtomConfig]) -> np.ndarray:
        coords = []
        for i in free_indices:
            coords.extend(atom_list[i].xyz)
        return np.array(coords, dtype=np.float64)

    def unpack_free_coords(free_x: np.ndarray, base_atoms: List[AtomConfig]) -> List[AtomConfig]:
        new_atoms = copy.deepcopy(base_atoms)
        idx = 0
        for i in free_indices:
            new_atoms[i].xyz = [float(free_x[idx]), float(free_x[idx + 1]), float(free_x[idx + 2])]
            idx += 3
        return new_atoms

    last_state: Optional[CalculationState] = None

    def evaluate_energy_and_state(x: np.ndarray) -> Tuple[float, CalculationState]:
        nonlocal last_state
        test_atoms = unpack_free_coords(x, atoms)
        test_deck = copy.deepcopy(current_deck)
        test_deck.atoms = test_atoms

        if warm_restart and last_state is not None and last_state.converged:
            new_geom = [(a.element, a.xyz[0], a.xyz[1], a.xyz[2]) for a in test_atoms]
            test_deck = last_state.create_restart_deck(new_geometry=new_geom)

        state = driver.run_deck(test_deck)
        if state.converged:
            last_state = state
        return float(state.total_energy), state

    def evaluate_gradient(x: np.ndarray, f0: float) -> np.ndarray:
        grad = np.zeros_like(x)
        h = finite_diff_step
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += h
            e_plus, _ = evaluate_energy_and_state(x_plus)

            x_minus = x.copy()
            x_minus[i] -= h
            e_minus, _ = evaluate_energy_and_state(x_minus)

            grad[i] = (e_plus - e_minus) / (2.0 * h)
        return grad

    # Initial state evaluation
    x = pack_free_coords(atoms)
    e_current, state_current = evaluate_energy_and_state(x)
    initial_energy = e_current
    last_state = state_current

    # Initial gradient
    grad = evaluate_gradient(x, e_current)
    grad_norm = float(np.linalg.norm(grad))

    trajectory: List[OptimizationStep] = [
        OptimizationStep(0, e_current, state_current.virial_ratio, grad_norm, [a.xyz for a in atoms])
    ]

    # Initialize inverse Hessian estimate as identity
    dim = len(x)
    H = np.eye(dim, dtype=np.float64)

    converged = False

    for step_num in range(1, max_steps + 1):
        if grad_norm < grad_tol:
            converged = True
            break

        # BFGS search direction: p = -H * g
        p = -H.dot(grad)
        p_norm = float(np.linalg.norm(p))

        # Trust-region step limit clamp
        if p_norm > step_limit:
            p = p * (step_limit / p_norm)

        # Backtracking line search with Armijo condition
        alpha = 1.0
        c1 = 1e-4
        x_new = x + alpha * p
        e_new, state_new = evaluate_energy_and_state(x_new)

        while e_new > e_current + c1 * alpha * np.dot(grad, p) and alpha > 0.05:
            alpha *= 0.5
            x_new = x + alpha * p
            e_new, state_new = evaluate_energy_and_state(x_new)

        # Evaluate new gradient
        grad_new = evaluate_gradient(x_new, e_new)
        grad_norm_new = float(np.linalg.norm(grad_new))

        # Check energy convergence
        d_e = abs(e_new - e_current)

        # Update trajectory
        updated_atoms = unpack_free_coords(x_new, atoms)
        trajectory.append(
            OptimizationStep(step_num, e_new, state_new.virial_ratio, grad_norm_new, [a.xyz for a in updated_atoms])
        )

        if grad_norm_new < grad_tol or (d_e < energy_tol and grad_norm_new < 3.0 * grad_tol):
            x = x_new
            e_current = e_new
            state_current = state_new
            grad_norm = grad_norm_new
            converged = True
            break

        # BFGS Hessian update
        s = x_new - x
        y = grad_new - grad
        ys = float(np.dot(y, s))

        if ys > 1e-10:
            rho = 1.0 / ys
            I_mat = np.eye(dim)
            H = (I_mat - rho * np.outer(s, y)).dot(H).dot(I_mat - rho * np.outer(y, s)) + rho * np.outer(s, s)

        x = x_new
        grad = grad_new
        grad_norm = grad_norm_new
        e_current = e_new
        state_current = state_new

    # Final optimized deck
    final_atoms = unpack_free_coords(x, atoms)
    optimized_deck = copy.deepcopy(current_deck)
    optimized_deck.atoms = final_atoms

    return OptimizationResult(
        converged=converged,
        initial_energy=initial_energy,
        final_energy=e_current,
        energy_change=e_current - initial_energy,
        num_steps=len(trajectory) - 1,
        trajectory=trajectory,
        optimized_deck=optimized_deck,
        final_state=state_current,
    )
