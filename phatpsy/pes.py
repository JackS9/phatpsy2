# phatpsy/pes.py
"""
1D and 2D Potential Energy Surface (PES) mapping for PHATPSY 2.0.
Automates rigid and relaxed coordinate scans, tracking total electronic energies,
virial theorem ratios (-V/T), and frontier orbital eigenvalues.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union, Sequence
import copy
import json
import numpy as np

from phatpsy.schema import CalculationDeck, AtomConfig
from phatpsy.driver import PhatpsyDriver
from phatpsy.state import CalculationState
from phatpsy.coords import (
    BondCoordinate,
    AngleCoordinate,
    DihedralCoordinate,
    normalize_vector,
)


@dataclass
class PESScan1DResult:
    """Results from a 1D Potential Energy Surface (PES) coordinate scan."""
    coordinate_name: str
    values: List[float]
    energies: List[float]
    virial_ratios: List[float]
    converged: List[bool]
    states: List[CalculationState] = field(default_factory=list)
    min_index: int = 0
    min_value: float = 0.0
    min_energy: float = 0.0
    equilibrium_value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a serializable dictionary."""
        return {
            "coordinate_name": self.coordinate_name,
            "values": [round(float(v), 6) for v in self.values],
            "energies": [round(float(e), 8) for e in self.energies],
            "virial_ratios": [round(float(vr), 6) for vr in self.virial_ratios],
            "converged": list(self.converged),
            "min_index": int(self.min_index),
            "min_value": round(float(self.min_value), 6),
            "min_energy": round(float(self.min_energy), 8),
            "equilibrium_value": round(float(self.equilibrium_value), 6),
        }

    def save_json(self, filepath: Union[str, Path]) -> None:
        """Saves scan result to a JSON file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def plot(self, output_path: Optional[Union[str, Path]] = None, show: bool = False) -> Any:
        """Plots the 1D PES curve alongside virial ratio and orbital correlation."""
        import matplotlib.pyplot as plt

        vals = np.array(self.values)
        enes = np.array(self.energies)
        virs = np.array(self.virial_ratios)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # 1. Total Energy PES curve
        ax1.plot(vals, enes, "o-", color="#2563eb", lw=2.2, markersize=6, label="E(q)")
        ax1.axvline(self.min_value, color="#dc2626", linestyle="--", alpha=0.75, label=f"Min: {self.min_value:.3f}")
        if abs(self.equilibrium_value - self.min_value) > 1e-4:
            ax1.axvline(self.equilibrium_value, color="#ea580c", linestyle=":", alpha=0.85, label=f"Equil Fit: {self.equilibrium_value:.3f}")

        ax1.set_title(f"1D Potential Energy Surface: {self.coordinate_name}", fontsize=12, fontweight="bold", pad=10)
        ax1.set_xlabel(f"Coordinate {self.coordinate_name}", fontsize=11)
        ax1.set_ylabel("Total Energy (Hartree)", fontsize=11)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(frameon=True, facecolor="#f8fafc")

        # 2. Virial theorem ratio (-V/T)
        ax2.plot(vals, virs, "s-", color="#059669", lw=2.2, markersize=6, label="-V / T")
        ax2.axhline(2.0, color="#475569", linestyle="--", label="Exact Invariant (2.0)")
        ax2.set_title("Virial Theorem Ratio Check (-V/T)", fontsize=12, fontweight="bold", pad=10)
        ax2.set_xlabel(f"Coordinate {self.coordinate_name}", fontsize=11)
        ax2.set_ylabel("-V / T Ratio", fontsize=11)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(frameon=True, facecolor="#f8fafc")

        plt.tight_layout()

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_p, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig


@dataclass
class PESScan2DResult:
    """Results from a 2D Potential Energy Surface (PES) coordinate scan."""
    coord1_name: str
    values1: List[float]
    coord2_name: str
    values2: List[float]
    energies: List[List[float]]
    virial_ratios: List[List[float]]
    min_indices: Tuple[int, int] = (0, 0)
    min_values: Tuple[float, float] = (0.0, 0.0)
    min_energy: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a serializable dictionary."""
        return {
            "coord1_name": self.coord1_name,
            "values1": [round(float(v), 6) for v in self.values1],
            "coord2_name": self.coord2_name,
            "values2": [round(float(v), 6) for v in self.values2],
            "energies": [[round(float(e), 8) for e in row] for row in self.energies],
            "virial_ratios": [[round(float(vr), 6) for vr in row] for row in self.virial_ratios],
            "min_indices": [int(i) for i in self.min_indices],
            "min_values": [round(float(v), 6) for v in self.min_values],
            "min_energy": round(float(self.min_energy), 8),
        }

    def save_json(self, filepath: Union[str, Path]) -> None:
        """Saves scan result to a JSON file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def plot(self, output_path: Optional[Union[str, Path]] = None, show: bool = False) -> Any:
        """Plots the 2D PES contour surface."""
        import matplotlib.pyplot as plt

        X, Y = np.meshgrid(self.values1, self.values2)
        Z = np.array(self.energies).T  # Transpose to align with meshgrid (Y rows, X cols)

        fig, ax = plt.subplots(figsize=(8, 6.5))
        cp = ax.contourf(X, Y, Z, levels=25, cmap="viridis")
        fig.colorbar(cp, ax=ax, label="Total Energy (Hartree)")
        cs = ax.contour(X, Y, Z, levels=15, colors="white", linewidths=0.7, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.3f")

        # Mark minimum
        m_x, m_y = self.min_values
        ax.plot(m_x, m_y, "r*", markersize=14, label=f"Min: ({m_x:.3f}, {m_y:.3f})")

        ax.set_title(f"2D Potential Energy Surface: {self.coord1_name} vs. {self.coord2_name}", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel(self.coord1_name, fontsize=11)
        ax.set_ylabel(self.coord2_name, fontsize=11)
        ax.legend(frameon=True, facecolor="#f8fafc", loc="upper right")

        plt.tight_layout()

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_p, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig


def _apply_bond_length(atoms: List[AtomConfig], atom1: int, atom2: int, target_length: float) -> List[AtomConfig]:
    """Modifies the distance between atom1 and atom2 to target_length while preserving direction and center of mass."""
    new_atoms = copy.deepcopy(atoms)
    p1 = np.array(new_atoms[atom1].xyz, dtype=np.float64)
    p2 = np.array(new_atoms[atom2].xyz, dtype=np.float64)

    # If one of the atoms is symmetry-generated from the other / origin:
    if new_atoms[atom2].symmetry is not None:
        u1, _ = normalize_vector(p1)
        if np.linalg.norm(u1) < 1e-12:
            u1 = np.array([0.0, 0.0, 1.0])
        new_atoms[atom1].xyz = [float(c) for c in (0.5 * target_length * u1)]
        return new_atoms
    elif new_atoms[atom1].symmetry is not None:
        u2, _ = normalize_vector(p2)
        if np.linalg.norm(u2) < 1e-12:
            u2 = np.array([0.0, 0.0, 1.0])
        new_atoms[atom2].xyz = [float(c) for c in (0.5 * target_length * u2)]
        return new_atoms

    center = 0.5 * (p1 + p2)
    u12, current_length = normalize_vector(p2 - p1)
    if current_length < 1e-12:
        u12 = np.array([0.0, 0.0, 1.0])

    new_p1 = center - 0.5 * target_length * u12
    new_p2 = center + 0.5 * target_length * u12

    new_atoms[atom1].xyz = [float(c) for c in new_p1]
    new_atoms[atom2].xyz = [float(c) for c in new_p2]
    return new_atoms


def _apply_angle(atoms: List[AtomConfig], atom1: int, atom2: int, atom3: int, target_angle_rad: float) -> List[AtomConfig]:
    """Modifies the valence angle atom1-atom2-atom3 (vertex at atom2) to target_angle_rad."""
    new_atoms = copy.deepcopy(atoms)
    p1 = np.array(new_atoms[atom1].xyz, dtype=np.float64)
    p2 = np.array(new_atoms[atom2].xyz, dtype=np.float64)  # Vertex
    p3 = np.array(new_atoms[atom3].xyz, dtype=np.float64)

    v21, d21 = normalize_vector(p1 - p2)
    v23, d23 = normalize_vector(p3 - p2)

    # Axis perpendicular to angle plane
    rot_axis, norm_axis = normalize_vector(np.cross(v21, v23))
    if norm_axis < 1e-10:
        # Collinear: pick arbitrary orthogonal axis
        rot_axis = np.array([1.0, 0.0, 0.0]) if abs(v21[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        rot_axis, _ = normalize_vector(np.cross(v21, rot_axis))

    # Current angle
    current_angle = math.acos(float(np.clip(np.dot(v21, v23), -1.0, 1.0)))
    delta_half = 0.5 * (target_angle_rad - current_angle)

    # Rodrigues' rotation formula
    def rotate_vec(v: np.ndarray, axis: np.ndarray, theta: float) -> np.ndarray:
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        return v * cos_t + np.cross(axis, v) * sin_t + axis * np.dot(axis, v) * (1.0 - cos_t)

    # Rotate v21 by -delta_half and v23 by +delta_half
    new_v21 = rotate_vec(v21, rot_axis, -delta_half)
    new_v23 = rotate_vec(v23, rot_axis, delta_half)

    new_atoms[atom1].xyz = [float(c) for c in (p2 + d21 * new_v21)]
    new_atoms[atom3].xyz = [float(c) for c in (p2 + d23 * new_v23)]
    return new_atoms


def parse_coordinate_spec(coord_spec: Any) -> Tuple[str, Any]:
    """Parses user coordinate specification into a standardized type and descriptor."""
    if isinstance(coord_spec, BondCoordinate):
        return ("bond", (coord_spec.atom1, coord_spec.atom2))
    elif isinstance(coord_spec, AngleCoordinate):
        return ("angle", (coord_spec.atom1, coord_spec.atom2, coord_spec.atom3))
    elif isinstance(coord_spec, str):
        parts = coord_spec.strip().split("-")
        if len(parts) == 2:
            return ("bond", (int(parts[0]), int(parts[1])))
        elif len(parts) == 3:
            return ("angle", (int(parts[0]), int(parts[1]), int(parts[2])))
    elif isinstance(coord_spec, (tuple, list)):
        kind = str(coord_spec[0]).lower()
        if kind in ("bond", "b"):
            return ("bond", (int(coord_spec[1]), int(coord_spec[2])))
        elif kind in ("angle", "a"):
            return ("angle", (int(coord_spec[1]), int(coord_spec[2]), int(coord_spec[3])))
    raise ValueError(f"Unrecognized coordinate specification: {coord_spec}")


def modify_deck_geometry(deck: CalculationDeck, coord_type: str, indices: Tuple[int, ...], value: float) -> CalculationDeck:
    """Returns a new CalculationDeck with the specified internal coordinate modified."""
    new_deck = copy.deepcopy(deck)
    atoms = new_deck.atoms

    if coord_type == "bond":
        a1, a2 = indices
        new_atoms = _apply_bond_length(atoms, a1, a2, target_length=value)
    elif coord_type == "angle":
        a1, a2, a3 = indices
        rad = math.radians(value) if value > 2.0 * math.pi else value  # Auto-detect degrees
        new_atoms = _apply_angle(atoms, a1, a2, a3, target_angle_rad=rad)
    else:
        raise ValueError(f"Unsupported coordinate type: {coord_type}")

    new_deck.atoms = new_atoms
    return new_deck


def scan_1d(
    deck: CalculationDeck,
    coordinate: Any,
    values: Sequence[float],
    driver: Optional[PhatpsyDriver] = None,
    warm_restart: bool = True,
    coord_name: Optional[str] = None,
) -> PESScan1DResult:
    """
    Performs an automated 1D Potential Energy Surface (PES) coordinate scan.

    Args:
        deck: Base CalculationDeck defining molecule and basis options.
        coordinate: Coordinate specifier (e.g. ('bond', 0, 1), '0-1', or BondCoordinate).
        values: Sequence of coordinate values to evaluate.
        driver: PhatpsyDriver instance (creates a new one if None).
        warm_restart: Whether to seed subsequent geometries with converged states.
        coord_name: Optional label for the coordinate.

    Returns:
        PESScan1DResult containing energies, virial ratios, minimum point, and states.
    """
    if driver is None:
        driver = PhatpsyDriver()

    c_type, c_indices = parse_coordinate_spec(coordinate)
    if coord_name is None:
        if c_type == "bond":
            coord_name = f"Bond R({deck.atoms[c_indices[0]].element}{c_indices[0]+1}-{deck.atoms[c_indices[1]].element}{c_indices[1]+1})"
        elif c_type == "angle":
            coord_name = f"Angle A({deck.atoms[c_indices[0]].element}{c_indices[0]+1}-{deck.atoms[c_indices[1]].element}{c_indices[1]+1}-{deck.atoms[c_indices[2]].element}{c_indices[2]+1})"
        else:
            coord_name = f"{c_type}_{c_indices}"

    energies: List[float] = []
    virials: List[float] = []
    converged: List[bool] = []
    states: List[CalculationState] = []

    last_state: Optional[CalculationState] = None

    for val in values:
        current_deck = modify_deck_geometry(deck, c_type, c_indices, val)

        # Warm restart if previous state is available
        if warm_restart and last_state is not None and last_state.converged:
            new_geom = [(a.element, a.xyz[0], a.xyz[1], a.xyz[2]) for a in current_deck.atoms]
            current_deck = last_state.create_restart_deck(new_geometry=new_geom)

        state = driver.run_deck(current_deck)
        energies.append(float(state.total_energy))
        virials.append(float(state.virial_ratio))
        converged.append(bool(state.converged))
        states.append(state)

        if state.converged:
            last_state = state

    energies_arr = np.array(energies)
    min_idx = int(np.argmin(energies_arr))
    min_val = float(values[min_idx])
    min_ene = float(energies_arr[min_idx])

    # Parabolic fit near minimum to estimate equilibrium value
    equil_val = min_val
    if 0 < min_idx < len(values) - 1:
        # 3-point quadratic fit
        x0, x1, x2 = values[min_idx - 1], values[min_idx], values[min_idx + 1]
        y0, y1, y2 = energies[min_idx - 1], energies[min_idx], energies[min_idx + 1]
        denom = 2.0 * ((x1 - x0) * (y2 - y1) - (x2 - x1) * (y1 - y0))
        if abs(denom) > 1e-12:
            x_fit = 0.5 * ((x1**2 - x0**2) * (y2 - y1) - (x2**2 - x1**2) * (y1 - y0)) / denom
            if min(x0, x2) <= x_fit <= max(x0, x2):
                equil_val = float(x_fit)

    return PESScan1DResult(
        coordinate_name=coord_name,
        values=[float(v) for v in values],
        energies=energies,
        virial_ratios=virials,
        converged=converged,
        states=states,
        min_index=min_idx,
        min_value=min_val,
        min_energy=min_ene,
        equilibrium_value=equil_val,
    )


def scan_2d(
    deck: CalculationDeck,
    coord1: Any,
    values1: Sequence[float],
    coord2: Any,
    values2: Sequence[float],
    driver: Optional[PhatpsyDriver] = None,
    warm_restart: bool = True,
    coord1_name: Optional[str] = None,
    coord2_name: Optional[str] = None,
) -> PESScan2DResult:
    """
    Performs an automated 2D grid Potential Energy Surface (PES) coordinate scan.
    """
    if driver is None:
        driver = PhatpsyDriver()

    c1_type, c1_indices = parse_coordinate_spec(coord1)
    c2_type, c2_indices = parse_coordinate_spec(coord2)

    name1 = coord1_name or f"{c1_type}_{c1_indices}"
    name2 = coord2_name or f"{c2_type}_{c2_indices}"

    energies_grid: List[List[float]] = []
    virials_grid: List[List[float]] = []

    last_state: Optional[CalculationState] = None

    for v1 in values1:
        deck_v1 = modify_deck_geometry(deck, c1_type, c1_indices, v1)
        row_e: List[float] = []
        row_v: List[float] = []

        for v2 in values2:
            deck_v1_v2 = modify_deck_geometry(deck_v1, c2_type, c2_indices, v2)

            if warm_restart and last_state is not None and last_state.converged:
                new_geom = [(a.element, a.xyz[0], a.xyz[1], a.xyz[2]) for a in deck_v1_v2.atoms]
                deck_v1_v2 = last_state.create_restart_deck(new_geometry=new_geom)

            state = driver.run_deck(deck_v1_v2)
            row_e.append(float(state.total_energy))
            row_v.append(float(state.virial_ratio))

            if state.converged:
                last_state = state

        energies_grid.append(row_e)
        virials_grid.append(row_v)

    enes_arr = np.array(energies_grid)
    min_i, min_j = np.unravel_index(np.argmin(enes_arr), enes_arr.shape)
    min_vals = (float(values1[min_i]), float(values2[min_j]))
    min_ene = float(enes_arr[min_i, min_j])

    return PESScan2DResult(
        coord1_name=name1,
        values1=[float(v) for v in values1],
        coord2_name=name2,
        values2=[float(v) for v in values2],
        energies=energies_grid,
        virial_ratios=virials_grid,
        min_indices=(int(min_i), int(min_j)),
        min_values=min_vals,
        min_energy=min_ene,
    )
