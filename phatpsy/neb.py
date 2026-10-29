# phatpsy/neb.py
"""
Nudged Elastic Band (NEB) and Climbing-Image NEB (CI-NEB) for transition state
searching and minimum energy reaction pathway (MEP) optimization in PHATPSY 2.0.

Implements:
- Linear and internal coordinate pathway interpolation between reactant and product.
- Henkelman-Jónsson improved upwind tangent estimate.
- Decoupled parallel spring forces and perpendicular potential forces.
- Climbing-Image NEB (CI-NEB) for rigorous saddle-point / transition state convergence.
- QuickMin damped dynamics velocity-projection optimizer.
- Pathway energy barrier plotting and multi-frame XYZ / JSON export.
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
class NEBImage:
    """An intermediate geometry image along the elastic band."""
    index: int
    deck: CalculationDeck
    energy: float
    coordinates: List[List[float]]
    forces: Optional[np.ndarray] = None
    true_force: Optional[np.ndarray] = None
    spring_force: Optional[np.ndarray] = None
    tangent: Optional[np.ndarray] = None
    state: Optional[CalculationState] = None
    is_climbing: bool = False


@dataclass
class NEBResult:
    """Result of a Nudged Elastic Band reaction pathway calculation."""
    converged: bool
    num_iterations: int
    images: List[CalculationDeck]
    energies: List[float]
    path_distances: List[float]
    barrier_forward: float
    barrier_reverse: float
    reaction_energy: float
    transition_state_index: int
    transition_state_deck: CalculationDeck
    transition_state_energy: float
    climbing_image_enabled: bool
    trajectory_energies: List[List[float]] = field(default_factory=list)

    @property
    def barrier_forward_kcal(self) -> float:
        """Forward activation barrier in kcal/mol (1 Hartree = 627.509474 kcal/mol)."""
        return self.barrier_forward * 627.509474

    @property
    def barrier_reverse_kcal(self) -> float:
        """Reverse activation barrier in kcal/mol."""
        return self.barrier_reverse * 627.509474

    @property
    def reaction_energy_kcal(self) -> float:
        """Reaction energy in kcal/mol."""
        return self.reaction_energy * 627.509474

    def to_dict(self) -> Dict[str, Any]:
        """Converts NEB result to a JSON-serializable dictionary."""
        return {
            "converged": bool(self.converged),
            "num_iterations": int(self.num_iterations),
            "climbing_image_enabled": bool(self.climbing_image_enabled),
            "transition_state_index": int(self.transition_state_index),
            "transition_state_energy": round(float(self.transition_state_energy), 8),
            "barrier_forward_hartree": round(float(self.barrier_forward), 8),
            "barrier_forward_kcal": round(float(self.barrier_forward_kcal), 4),
            "barrier_reverse_hartree": round(float(self.barrier_reverse), 8),
            "barrier_reverse_kcal": round(float(self.barrier_reverse_kcal), 4),
            "reaction_energy_hartree": round(float(self.reaction_energy), 8),
            "reaction_energy_kcal": round(float(self.reaction_energy_kcal), 4),
            "path_distances_bohr": [round(float(d), 6) for d in self.path_distances],
            "energies_hartree": [round(float(e), 8) for e in self.energies],
            "relative_energies_kcal": [
                round(float((e - self.energies[0]) * 627.509474), 4) for e in self.energies
            ],
            "images": [
                {
                    "index": i,
                    "energy": round(float(self.energies[i]), 8),
                    "is_transition_state": (i == self.transition_state_index),
                    "atoms": [
                        {
                            "element": a.element,
                            "xyz": [round(float(c), 6) for c in a.xyz],
                            "fixed": bool(a.fixed),
                            "virtual": bool(a.virtual),
                        }
                        for a in deck.atoms
                    ],
                }
                for i, deck in enumerate(self.images)
            ],
        }

    def save_json(self, filepath: Union[str, Path]) -> None:
        """Saves NEB results and pathway data to a JSON file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_xyz(self, filepath: Union[str, Path]) -> None:
        """
        Saves the entire reaction pathway as a multi-frame standard XYZ file.
        Each frame includes the image index and energy in the comment line.
        """
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        BOHR_TO_ANG = 0.529177210903
        lines = []
        for i, deck in enumerate(self.images):
            atoms = deck.atoms
            lines.append(str(len(atoms)))
            e_rel_kcal = (self.energies[i] - self.energies[0]) * 627.509474
            tag = " [Transition State]" if i == self.transition_state_index else ""
            lines.append(
                f"Image {i:2d} | Energy = {self.energies[i]:.8f} Hartree | DeltaE = {e_rel_kcal:+.2f} kcal/mol{tag}"
            )
            for a in atoms:
                # Convert Bohrs to Angstroms for standard XYZ format
                x_ang = a.xyz[0] * BOHR_TO_ANG
                y_ang = a.xyz[1] * BOHR_TO_ANG
                z_ang = a.xyz[2] * BOHR_TO_ANG
                lines.append(f"{a.element:<4} {x_ang:14.6f} {y_ang:14.6f} {z_ang:14.6f}")

        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def plot_pathway(
        self,
        output_path: Optional[Union[str, Path]] = None,
        units: str = "kcal",
        show: bool = False,
    ) -> Any:
        """
        Generates a publication-quality potential energy reaction profile plot.

        Args:
            output_path: Path to save the image (.png, .pdf, .svg).
            units: Energy unit for y-axis ('kcal' for kcal/mol or 'hartree').
            show: If True, calls plt.show().
        """
        import matplotlib.pyplot as plt

        distances = self.path_distances
        e0 = self.energies[0]
        if units.lower() == "hartree":
            rel_energies = [e - e0 for e in self.energies]
            ylabel = "Relative Energy (Hartree)"
            barrier_str = f"Ea = {self.barrier_forward:.4f} Eh"
        else:
            rel_energies = [(e - e0) * 627.509474 for e in self.energies]
            ylabel = "Relative Energy (kcal/mol)"
            barrier_str = f"Ea = {self.barrier_forward_kcal:.2f} kcal/mol"

        fig, ax = plt.subplots(figsize=(8, 4.8))

        # Smooth cubic spline interpolation for reaction curve if >= 4 images
        if len(distances) >= 4:
            try:
                from scipy.interpolate import CubicSpline
                cs = CubicSpline(distances, rel_energies, bc_type="natural")
                xs = np.linspace(distances[0], distances[-1], 200)
                ys = cs(xs)
                ax.plot(xs, ys, "-", color="#3b82f6", lw=2.2, label="Reaction Path (MEP)", zorder=2)
            except Exception:
                ax.plot(distances, rel_energies, "-", color="#3b82f6", lw=2, zorder=2)
        else:
            ax.plot(distances, rel_energies, "-", color="#3b82f6", lw=2, zorder=2)

        # Plot intermediate images
        for i, (d, e) in enumerate(zip(distances, rel_energies)):
            if i == 0:
                ax.scatter(d, e, color="#10b981", s=90, zorder=3, label="Reactant (R)")
            elif i == len(distances) - 1:
                ax.scatter(d, e, color="#6366f1", s=90, zorder=3, label="Product (P)")
            elif i == self.transition_state_index:
                ax.scatter(d, e, color="#ef4444", s=130, marker="*", zorder=4, label=f"TS ({barrier_str})")
            else:
                ax.scatter(d, e, color="#3b82f6", s=50, zorder=3)

        # Label TS peak
        ts_d = distances[self.transition_state_index]
        ts_e = rel_energies[self.transition_state_index]
        ax.annotate(
            barrier_str,
            xy=(ts_d, ts_e),
            xytext=(ts_d, ts_e + (max(rel_energies) - min(rel_energies)) * 0.08),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#ef4444",
            arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.2),
        )

        ax.set_title("Minimum Energy Pathway (Nudged Elastic Band)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Reaction Coordinate / Cumulative Distance (Bohr)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", framealpha=0.9)
        plt.tight_layout()

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_p, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def show(self, height: int = 550) -> Any:
        """Displays interactive 3D reaction pathway slider inside Jupyter notebooks."""
        from phatpsy.visualizer import show_pathway
        return show_pathway(self, height=height)

    def visualize(self, output_path: Optional[Union[str, Path]] = None, open_browser: bool = True) -> str:
        """Generates standalone interactive 3D WebGL pathway viewer in browser."""
        from phatpsy.visualizer import visualize_pathway
        return visualize_pathway(self, output_path=output_path, open_browser=open_browser)


def interpolate_images(
    initial_deck: CalculationDeck,
    final_deck: CalculationDeck,
    num_images: int = 7,
) -> List[CalculationDeck]:
    """
    Interpolates molecular geometries linearly between initial (reactant) and final (product) states.

    Args:
        initial_deck: Reactant CalculationDeck.
        final_deck: Product CalculationDeck.
        num_images: Total number of images including endpoints (minimum 3).

    Returns:
        List of CalculationDeck instances from initial to final state.
    """
    if num_images < 3:
        raise ValueError(f"num_images must be at least 3, got {num_images}")

    init_atoms = initial_deck.atoms
    final_atoms = final_deck.atoms

    if len(init_atoms) != len(final_atoms):
        raise ValueError(
            f"Atom count mismatch: initial deck has {len(init_atoms)} atoms, "
            f"final deck has {len(final_atoms)} atoms"
        )

    for i, (a1, a2) in enumerate(zip(init_atoms, final_atoms)):
        if a1.element.upper() != a2.element.upper():
            raise ValueError(
                f"Element mismatch at atom index {i}: '{a1.element}' vs '{a2.element}'"
            )

    images: List[CalculationDeck] = []
    alphas = np.linspace(0.0, 1.0, num_images)

    for step_idx, alpha in enumerate(alphas):
        deck_i = copy.deepcopy(initial_deck)
        new_atoms: List[AtomConfig] = []
        for a1, a2 in zip(init_atoms, final_atoms):
            atom_copy = copy.deepcopy(a1)
            # Linear interpolation of Cartesian coordinates
            atom_copy.xyz = [
                float((1.0 - alpha) * a1.xyz[0] + alpha * a2.xyz[0]),
                float((1.0 - alpha) * a1.xyz[1] + alpha * a2.xyz[1]),
                float((1.0 - alpha) * a1.xyz[2] + alpha * a2.xyz[2]),
            ]
            new_atoms.append(atom_copy)
        deck_i.atoms = new_atoms
        deck_i.title = f"{initial_deck.title or 'NEB Image'} [Image {step_idx}/{num_images - 1}]"
        images.append(deck_i)

    return images


def calculate_tangents(
    coords: List[np.ndarray],
    energies: List[float],
) -> List[np.ndarray]:
    """
    Computes normalized upwind tangents along the elastic band using the Henkelman-Jónsson method.

    Args:
        coords: List of Cartesian coordinate vectors (1D arrays) for all images.
        energies: List of total energies for all images.

    Returns:
        List of unit tangent vectors for intermediate images.
    """
    n_images = len(coords)
    tangents: List[np.ndarray] = [np.zeros_like(coords[0]) for _ in range(n_images)]

    for i in range(1, n_images - 1):
        tau_plus = coords[i + 1] - coords[i]
        tau_minus = coords[i] - coords[i - 1]

        e_plus = energies[i + 1]
        e_i = energies[i]
        e_minus = energies[i - 1]

        if e_plus > e_i > e_minus:
            tau = tau_plus
        elif e_plus < e_i < e_minus:
            tau = tau_minus
        else:
            # Image is an energy extremum: weighted combination based on energy slope
            d_e_max = max(abs(e_plus - e_i), abs(e_minus - e_i))
            d_e_min = min(abs(e_plus - e_i), abs(e_minus - e_i))

            if e_plus > e_minus:
                tau = tau_plus * d_e_max + tau_minus * d_e_min
            else:
                tau = tau_plus * d_e_min + tau_minus * d_e_max

        norm = float(np.linalg.norm(tau))
        if norm > 1e-12:
            tangents[i] = tau / norm
        else:
            tangents[i] = np.zeros_like(tau)

    return tangents


def calculate_neb_forces(
    coords: List[np.ndarray],
    energies: List[float],
    gradients: List[np.ndarray],
    tangents: List[np.ndarray],
    k_spring: float = 0.1,
    climbing_index: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Computes total NEB forces on intermediate images:
    - Perpendicular true force: F_perp = -g + (g . tau) * tau
    - Parallel spring force: F_parallel^s = k * (|X_{i+1} - X_i| - |X_i - X_{i-1}|) * tau
    - Climbing Image force: F_ci = -g + 2 * (g . tau) * tau

    Returns:
        (total_forces, true_forces, spring_forces)
    """
    n_images = len(coords)
    total_forces: List[np.ndarray] = [np.zeros_like(coords[0]) for _ in range(n_images)]
    true_forces: List[np.ndarray] = [np.zeros_like(coords[0]) for _ in range(n_images)]
    spring_forces: List[np.ndarray] = [np.zeros_like(coords[0]) for _ in range(n_images)]

    for i in range(1, n_images - 1):
        g = gradients[i]
        f_true = -g
        true_forces[i] = f_true
        tau = tangents[i]

        f_parallel_comp = float(np.dot(f_true, tau))
        f_perp = f_true - f_parallel_comp * tau

        if climbing_index is not None and i == climbing_index:
            # Climbing image: invert force along reaction path tangent, spring force removed
            f_ci = f_true - 2.0 * f_parallel_comp * tau
            total_forces[i] = f_ci
            spring_forces[i] = np.zeros_like(f_true)
        else:
            # Standard NEB image with spring force
            d_plus = float(np.linalg.norm(coords[i + 1] - coords[i]))
            d_minus = float(np.linalg.norm(coords[i] - coords[i - 1]))
            f_spring = k_spring * (d_plus - d_minus) * tau
            spring_forces[i] = f_spring
            total_forces[i] = f_perp + f_spring

    return total_forces, true_forces, spring_forces


def neb_search(
    initial_deck: CalculationDeck,
    final_deck: CalculationDeck,
    num_images: int = 7,
    driver: Optional[PhatpsyDriver] = None,
    k_spring: float = 0.1,
    max_iterations: int = 40,
    force_tol: float = 0.003,
    finite_diff_step: float = 0.005,
    step_limit: float = 0.15,
    climb: bool = True,
    warm_restart: bool = True,
    callback: Optional[Callable[[int, List[float], float], None]] = None,
    quiet: bool = True,
) -> NEBResult:
    """
    Performs Nudged Elastic Band (NEB) or Climbing-Image NEB (CI-NEB) transition state search.

    Args:
        initial_deck: Reactant state CalculationDeck.
        final_deck: Product state CalculationDeck.
        num_images: Total number of pathway images including endpoints (default: 7).
        driver: PhatpsyDriver instance (created if None).
        k_spring: Inter-image elastic spring constant in Hartree/Bohr^2 (default: 0.1).
        max_iterations: Maximum optimization cycles (default: 40).
        force_tol: Convergence threshold on maximum force norm across intermediate images (default: 0.003 Hartree/Bohr).
        finite_diff_step: Coordinate displacement step for numerical gradients (Bohr, default: 0.005).
        step_limit: Maximum Cartesian coordinate displacement per step (default: 0.15 Bohr).
        climb: If True, enables Climbing-Image NEB to drive highest-energy image to saddle point.
        warm_restart: Whether to seed successive SCF evaluations with previous wavefunctions.
        callback: Optional callback func(iteration, energies, max_force).
        quiet: Whether to suppress verbose Fortran output (default: True).

    Returns:
        NEBResult with full pathway, transition state, and barrier metrics.
    """
    if driver is None:
        driver = PhatpsyDriver(quiet=quiet)

    # 1. Generate initial interpolated images
    images = interpolate_images(initial_deck, final_deck, num_images=num_images)
    n_images = len(images)
    ref_atoms = images[0].geometry if hasattr(images[0], "geometry") else images[0].atoms

    # Identify unfrozen degrees of freedom (non-fixed, non-virtual)
    free_indices = [
        i for i, a in enumerate(ref_atoms) if not a.fixed and not a.virtual and a.symmetry is None
    ]
    if not free_indices:
        raise ValueError("Cannot perform NEB: all atoms are marked fixed or virtual.")

    def pack_coords(atom_list: List[AtomConfig]) -> np.ndarray:
        coords = []
        for i in free_indices:
            coords.extend(atom_list[i].xyz)
        return np.array(coords, dtype=np.float64)

    def unpack_coords(x: np.ndarray, base_atoms: List[AtomConfig]) -> List[AtomConfig]:
        new_atoms = copy.deepcopy(base_atoms)
        idx = 0
        for i in free_indices:
            new_atoms[i].xyz = [float(x[idx]), float(x[idx + 1]), float(x[idx + 2])]
            idx += 3
        return new_atoms

    # Track wavefunctions for warm restarts
    image_states: List[Optional[CalculationState]] = [None] * n_images

    def evaluate_image(img_idx: int, x: np.ndarray) -> Tuple[float, CalculationState]:
        img_atoms = images[img_idx].geometry if hasattr(images[img_idx], "geometry") else images[img_idx].atoms
        current_atoms = unpack_coords(x, img_atoms)
        deck_eval = copy.deepcopy(images[img_idx])
        if hasattr(deck_eval, "geometry"):
            deck_eval.geometry = current_atoms
        else:
            deck_eval.atoms = current_atoms

        if warm_restart and image_states[img_idx] is not None and image_states[img_idx].converged:
            new_geom = [(a.element, a.xyz[0], a.xyz[1], a.xyz[2]) for a in current_atoms]
            deck_eval = image_states[img_idx].create_restart_deck(new_geometry=new_geom)

        state = driver.run_deck(deck_eval, quiet=quiet)
        if state.converged:
            image_states[img_idx] = state
        return float(state.total_energy), state

    def evaluate_gradient(img_idx: int, x: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(x)
        h = finite_diff_step
        for dim in range(len(x)):
            xp = x.copy()
            xp[dim] += h
            ep, _ = evaluate_image(img_idx, xp)

            xm = x.copy()
            xm[dim] -= h
            em, _ = evaluate_image(img_idx, xm)

            raw_g = (ep - em) / (2.0 * h)
            grad[dim] = float(np.clip(raw_g, -20.0, 20.0))
        return grad

    # 2. Evaluate initial energies of all images
    x_coords: List[np.ndarray] = [pack_coords(img.atoms) for img in images]
    energies: List[float] = [0.0] * n_images

    # Endpoints are fixed
    e_start, s_start = evaluate_image(0, x_coords[0])
    energies[0] = e_start
    image_states[0] = s_start

    e_end, s_end = evaluate_image(n_images - 1, x_coords[-1])
    energies[-1] = e_end
    image_states[-1] = s_end

    for i in range(1, n_images - 1):
        ei, si = evaluate_image(i, x_coords[i])
        energies[i] = ei
        image_states[i] = si

    # Velocities for QuickMin optimizer
    velocities: List[np.ndarray] = [np.zeros_like(x_coords[0]) for _ in range(n_images)]
    dt = 0.8  # QuickMin integration time step

    converged = False
    trajectory_energies: List[List[float]] = [list(energies)]

    for iteration in range(1, max_iterations + 1):
        # 3. Compute gradients for all intermediate images
        gradients: List[np.ndarray] = [np.zeros_like(x_coords[0]) for _ in range(n_images)]
        for i in range(1, n_images - 1):
            gradients[i] = evaluate_gradient(i, x_coords[i])

        # 4. Compute Henkelman-Jónsson tangents
        tangents = calculate_tangents(x_coords, energies)

        # 5. Identify climbing image if enabled
        climbing_idx = None
        if climb:
            # Highest energy intermediate image
            intermediate_energies = [(energies[i], i) for i in range(1, n_images - 1)]
            climbing_idx = max(intermediate_energies, key=lambda pair: pair[0])[1]

        # 6. Calculate NEB forces
        forces, true_f, spring_f = calculate_neb_forces(
            x_coords,
            energies,
            gradients,
            tangents,
            k_spring=k_spring,
            climbing_index=climbing_idx,
        )

        # Compute maximum force norm across intermediate images
        force_norms = [float(np.linalg.norm(forces[i])) for i in range(1, n_images - 1)]
        max_force = max(force_norms) if force_norms else 0.0

        if callback:
            callback(iteration, list(energies), max_force)

        if max_force < force_tol:
            converged = True
            break

        # 7. QuickMin velocity-projection update for intermediate images
        for i in range(1, n_images - 1):
            f_i = forces[i]
            v_i = velocities[i]

            f_norm = float(np.linalg.norm(f_i))
            if f_norm > 1e-12:
                f_unit = f_i / f_norm
                v_proj = float(np.dot(v_i, f_unit))
                if v_proj > 0.0:
                    velocities[i] = v_proj * f_unit
                else:
                    velocities[i] = np.zeros_like(v_i)
            else:
                velocities[i] = np.zeros_like(v_i)

            # Coordinate displacement: dx = v * dt + 0.5 * f * dt^2
            dx = velocities[i] * dt + 0.5 * f_i * (dt ** 2)
            dx_norm = float(np.linalg.norm(dx))
            if dx_norm > step_limit:
                dx = dx * (step_limit / dx_norm)

            x_coords[i] += dx
            velocities[i] += f_i * dt

            # Update atom positions in image deck
            images[i].atoms = unpack_coords(x_coords[i], images[i].atoms)

            # Re-evaluate energy at updated position
            ei, si = evaluate_image(i, x_coords[i])
            energies[i] = ei
            image_states[i] = si

        trajectory_energies.append(list(energies))

    # 8. Compute final cumulative path distances
    path_distances = [0.0]
    for i in range(1, n_images):
        diff = np.linalg.norm(x_coords[i] - x_coords[i - 1])
        path_distances.append(float(path_distances[-1] + diff))

    # Identify Transition State (highest energy along pathway, prioritizing intermediate images)
    if n_images > 2:
        intermediate_energies = energies[1:-1]
        valid_indices = [idx for idx, e in enumerate(intermediate_energies, start=1) if not math.isnan(e)]
        if valid_indices:
            ts_idx = max(valid_indices, key=lambda idx: energies[idx])
        else:
            ts_idx = n_images // 2
    else:
        ts_idx = int(np.nanargmax(energies)) if not all(math.isnan(e) for e in energies) else 0

    ts_deck = images[ts_idx]
    ts_energy = energies[ts_idx]

    e_reactant = energies[0]
    e_product = energies[-1]

    barrier_fwd = max(0.0, ts_energy - e_reactant) if (not math.isnan(ts_energy) and not math.isnan(e_reactant)) else 0.0
    barrier_rev = max(0.0, ts_energy - e_product) if (not math.isnan(ts_energy) and not math.isnan(e_product)) else 0.0
    rxn_e = (e_product - e_reactant) if (not math.isnan(e_product) and not math.isnan(e_reactant)) else 0.0

    return NEBResult(
        converged=converged,
        num_iterations=iteration,
        images=images,
        energies=energies,
        path_distances=path_distances,
        barrier_forward=barrier_fwd,
        barrier_reverse=barrier_rev,
        reaction_energy=rxn_e,
        transition_state_index=ts_idx,
        transition_state_deck=ts_deck,
        transition_state_energy=ts_energy,
        climbing_image_enabled=bool(climb),
        trajectory_energies=trajectory_energies,
    )
