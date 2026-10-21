# phatpsy/spin.py
"""
Automated alternating-spin (Néel / Localized Valence-Bond) solver for PHATPSY.
Constructs molecular connectivity graphs using covalent radii and computes
optimal alternating spin assignments ('up' / 'down') to maximize antiferromagnetically
paired adjacent centers while minimizing net molecular spin.
"""
from typing import List, Dict, Tuple, Optional, Set, Union, Any
import math
import numpy as np
from phatpsy.schema import AtomConfig

# Standard covalent radii in Angstroms (Cordero et al. / Pyykko)
COVALENT_RADII_AA: Dict[str, float] = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39, "Fe": 1.32,
    "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20,
    "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, "Tc": 1.47, "Ru": 1.46,
    "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38,
    "I": 1.39, "Xe": 1.40,
    "Cs": 2.44, "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01, "Pm": 1.99, "Sm": 1.98,
    "Eu": 1.98, "Gd": 1.96, "Tb": 1.94, "Dy": 1.92, "Ho": 1.92, "Er": 1.89, "Tm": 1.90, "Yb": 1.87,
    "Lu": 1.87, "Hf": 1.75, "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36,
    "Au": 1.36, "Hg": 1.32, "Tl": 1.45, "Pb": 1.46, "Bi": 1.48, "Po": 1.40, "At": 1.50, "Rn": 1.50,
}
BOHR_PER_ANGSTROM: float = 1.8897261245650618


def get_covalent_radius_bohr(element: str) -> float:
    """Returns covalent radius in Bohr."""
    elem = element.strip().capitalize()
    r_aa = COVALENT_RADII_AA.get(elem, 1.40)
    return r_aa * BOHR_PER_ANGSTROM


def build_connectivity_graph(
    atoms: List[AtomConfig],
    tolerance: float = 1.25,
) -> List[List[int]]:
    """
    Builds adjacency list representing molecular bonds between atoms.

    Args:
        atoms: List of AtomConfig objects (coordinates in Bohr).
        tolerance: Distance scaling factor (default: 1.25).

    Returns:
        Adjacency list where adj[i] is a list of neighbor indices for atom i.
    """
    n = len(atoms)
    adj: List[List[int]] = [[] for _ in range(n)]
    radii = [get_covalent_radius_bohr(a.element) for a in atoms]

    for i in range(n):
        xi, yi, zi = atoms[i].xyz
        ri = radii[i]
        for j in range(i + 1, n):
            xj, yj, zj = atoms[j].xyz
            rj = radii[j]
            dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
            max_bond = (ri + rj) * tolerance
            if 0.1 < dist <= max_bond:
                adj[i].append(j)
                adj[j].append(i)

    return adj


def solve_alternating_spin(
    adj: List[List[int]],
    start_spin: str = "up",
    locked_spins: Optional[Dict[int, str]] = None,
) -> List[str]:
    """
    Solves for the optimal alternating spin assignment ('up' / 'down') across the graph.

    For bipartite networks, performs exact 2-coloring.
    For frustrated networks (e.g. triangular clusters like Ag13), minimizes the
    antiferromagnetic Ising energy sum_{(i,j)} s_i s_j + lambda (sum s_i)^2 via
    multi-start greedy local search.

    Args:
        adj: Adjacency list of the graph.
        start_spin: Spin for the initial seed atom ('up' or 'down').
        locked_spins: Optional dictionary of {atom_index: 'up'/'down'} for constrained atoms.

    Returns:
        List of 'up' and 'down' strings for each atom index.
    """
    n = len(adj)
    if n == 0:
        return []

    locked = locked_spins or {}
    spins = [0] * n  # +1 for up, -1 for down, 0 for unassigned
    for idx, s in locked.items():
        if s.lower() == "up":
            spins[idx] = 1
        elif s.lower() == "down":
            spins[idx] = -1

    init_val = 1 if start_spin.lower() in ("up", "alpha", "+1") else -1

    # Find connected components and attempt BFS 2-coloring
    visited = set()
    is_frustrated = False

    for root in range(n):
        if root in visited:
            continue

        comp_nodes = []
        queue = [root]
        visited.add(root)
        if spins[root] == 0:
            spins[root] = init_val

        comp_frustrated = False
        while queue:
            curr = queue.pop(0)
            comp_nodes.append(curr)
            curr_spin = spins[curr]

            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    if spins[neighbor] == 0:
                        spins[neighbor] = -curr_spin
                    elif spins[neighbor] == curr_spin:
                        comp_frustrated = True
                    queue.append(neighbor)
                else:
                    if spins[neighbor] == curr_spin:
                        comp_frustrated = True

        if comp_frustrated:
            is_frustrated = True
            # Optimize frustrated component using antiferromagnetic Ising relaxation
            spins = _optimize_frustrated_spins(comp_nodes, adj, spins, locked)

    result = ["up" if s >= 0 else "down" for s in spins]
    return result


def _optimize_frustrated_spins(
    nodes: List[int],
    adj: List[List[int]],
    initial_spins: List[int],
    locked: Dict[int, str],
    num_restarts: int = 20,
) -> List[int]:
    """
    Greedy multi-start local search to minimize antiferromagnetic frustration
    E = sum_{(i,j)} s_i s_j + 0.5 * (sum s_i)^2.
    """
    best_spins = list(initial_spins)
    best_energy = _calc_spin_energy(nodes, adj, best_spins)

    rng = np.random.default_rng(42)

    for restart in range(num_restarts):
        curr_spins = list(initial_spins)
        if restart > 0:
            for node in nodes:
                if node not in locked:
                    curr_spins[node] = 1 if rng.random() > 0.5 else -1

        improved = True
        step = 0
        while improved and step < 50:
            step += 1
            improved = False
            for node in nodes:
                if node in locked:
                    continue

                curr_e = _calc_spin_energy(nodes, adj, curr_spins)
                curr_spins[node] = -curr_spins[node]
                new_e = _calc_spin_energy(nodes, adj, curr_spins)
                if new_e < curr_e:
                    improved = True
                else:
                    curr_spins[node] = -curr_spins[node]

        final_e = _calc_spin_energy(nodes, adj, curr_spins)
        if final_e < best_energy:
            best_energy = final_e
            best_spins = list(curr_spins)

    return best_spins


def _calc_spin_energy(nodes: List[int], adj: List[List[int]], spins: List[int]) -> float:
    """Calculates Ising bond frustration + net spin penalty."""
    bond_e = 0.0
    net_spin = 0.0
    for i in nodes:
        si = spins[i]
        net_spin += si
        for j in adj[i]:
            if j > i and j in nodes:
                # Parallel spins (+1) penalize; antiparallel (-1) reward
                bond_e += si * spins[j]

    # Strong penalty on net polarization to maintain near-singlet state
    penalty = 1.0 * (net_spin ** 2)
    return bond_e + penalty


def assign_alternating_spin(
    geometry: Union[List[AtomConfig], Any],
    start_spin: str = "up",
    tolerance: float = 1.25,
) -> List[AtomConfig]:
    """
    Assigns alternating spin ('up' / 'down') to a list of atoms based on connectivity.

    Args:
        geometry: List of AtomConfig objects.
        start_spin: Starting spin direction ('up' or 'down').
        tolerance: Adjacency distance scaling factor.

    Returns:
        Updated list of AtomConfig objects with assigned spin attributes.
    """
    if hasattr(geometry, "geometry"):
        atoms = geometry.geometry
    else:
        atoms = geometry

    if not atoms:
        return atoms

    # Check for user-locked spins
    locked: Dict[int, str] = {}
    for idx, a in enumerate(atoms):
        if a.spin in ("up", "down") and not getattr(a, "_auto_assigned", False):
            locked[idx] = a.spin

    adj = build_connectivity_graph(atoms, tolerance=tolerance)
    spins = solve_alternating_spin(adj, start_spin=start_spin, locked_spins=locked)

    for idx, a in enumerate(atoms):
        a.spin = spins[idx]
        a.spin_flip = (spins[idx] == "down")
        setattr(a, "_auto_assigned", True)

    return atoms
