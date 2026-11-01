# phatpsy/spin.py
"""
Automated alternating-spin (Néel / Localized Valence-Bond) solver for PHATPSY.
Constructs molecular connectivity graphs using covalent radii and computes
optimal alternating spin assignments ('up' / 'down') to maximize antiferromagnetically
paired adjacent centers while minimizing net molecular spin.
"""
from dataclasses import dataclass, asdict
import warnings
from typing import List, Dict, Tuple, Optional, Set, Union, Any
import math
import numpy as np
from phatpsy.schema import AtomConfig


class NonBipartiteSpinGraphError(ValueError):
    """
    Raised when attempting automated alternating spin assignment or pairing
    on a molecular or symmetry graph containing odd cycles (frustrated rings)
    when strict bipartite graph enforcement is active.
    """
    pass


@dataclass
class FerromagneticLinkReport:
    """
    Detailed physical report of a ferromagnetic coupling (parallel spin link)
    forced by an odd cycle in a non-bipartite molecular spin graph.

    In valence bond theory, odd rings break the perfect-pairing singlet (PP-VB)
    network. The identified bond represents the ferromagnetic anomaly (parallel
    spin alignment) where local singlet pairing is frustrated.
    """
    atom_pair: Tuple[str, str]
    ring_atoms: List[str]
    ring_size: int
    distance_bohr: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "atom_pair": list(self.atom_pair),
            "ring_atoms": list(self.ring_atoms),
            "ring_size": self.ring_size,
            "distance_bohr": round(float(self.distance_bohr), 4),
            "reason": self.reason,
        }



def pp_vb_diatomic_spins() -> List[str]:
    """
    Returns the explicit Perfect-Pairing Valence-Bond (PP-VB) alternating
    spin configuration ['up', 'down'] for homonuclear diatomics.
    """
    return ["up", "down"]


def find_odd_cycle(adj: List[List[int]]) -> Optional[List[int]]:
    """
    Finds an odd cycle (frustrated ring) in an undirected graph if one exists.

    Returns:
        List of node indices forming the odd cycle, or None if the graph is bipartite.
    """
    n = len(adj)
    color: Dict[int, int] = {}
    parent: Dict[int, Optional[int]] = {}

    for start in range(n):
        if start in color:
            continue
        color[start] = 1
        parent[start] = None
        queue = [start]

        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if v not in color:
                    color[v] = -color[u]
                    parent[v] = u
                    queue.append(v)
                elif color[v] == color[u]:
                    # Odd cycle detected! Trace paths back to lowest common ancestor
                    path_u = []
                    curr: Optional[int] = u
                    while curr is not None:
                        path_u.append(curr)
                        curr = parent[curr]

                    path_v = []
                    curr = v
                    while curr is not None:
                        path_v.append(curr)
                        curr = parent[curr]

                    set_v = set(path_v)
                    lca = None
                    for node in path_u:
                        if node in set_v:
                            lca = node
                            break

                    cycle = []
                    for node in path_u:
                        cycle.append(node)
                        if node == lca:
                            break
                    v_part = []
                    for node in path_v:
                        if node == lca:
                            break
                        v_part.append(node)
                    cycle = cycle + list(reversed(v_part))
                    return cycle
    return None


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
    require_bipartite: bool = True,
) -> List[str]:
    """
    Solves for the optimal alternating spin assignment ('up' / 'down') across the graph.

    For bipartite networks, performs exact 2-coloring.
    If require_bipartite is True (default), raises NonBipartiteSpinGraphError if any
    odd cycle is detected.
    If require_bipartite is False, minimizes the antiferromagnetic Ising energy
    sum_{(i,j)} s_i s_j + lambda (sum s_i)^2 via multi-start greedy local search.

    Args:
        adj: Adjacency list of the graph.
        start_spin: Spin for the initial seed atom ('up' or 'down').
        locked_spins: Optional dictionary of {atom_index: 'up'/'down'} for constrained atoms.
        require_bipartite: If True (default), raises NonBipartiteSpinGraphError on frustrated odd cycles.

    Returns:
        List of 'up' and 'down' strings for each atom index.
    """
    n = len(adj)
    if n == 0:
        return []

    if require_bipartite:
        cycle = find_odd_cycle(adj)
        if cycle is not None:
            raise NonBipartiteSpinGraphError(
                f"Cannot assign alternating spin: atom graph is non-bipartite and contains an odd cycle "
                f"(frustrated ring): {cycle}. Non-bipartite graphs cannot be 2-colored; explicit per-atom spins are required."
            )

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


def detect_ferromagnetic_links(
    adj: List[List[int]],
    coords: List[List[float]],
    atom_ids: List[str],
    user_links: Optional[List[Tuple[Union[str, int], Union[str, int]]]] = None,
) -> Tuple[List[List[int]], List[Tuple[int, int]], List[FerromagneticLinkReport]]:
    """
    Identifies all odd cycles (frustrated rings) in the molecular connectivity graph,
    and determines the minimal set of bonds where parallel spins (ferromagnetic links)
    must be placed to yield a bipartite spanning subgraph.

    Default selection rule:
    In each detected odd cycle, the bond with the largest internuclear distance (longest /
    weakest covalent coupling) is selected as the ferromagnetic anomaly, preserving
    antiferromagnetic singlet pairing on the stronger adjacent bonds.
    Ties are broken lexicographically by atom IDs.

    User-specified override:
    If `user_links` contains a bond lying in the odd cycle, that bond is selected.

    Returns:
        (bipartite_adj, chosen_edges, reports)
    """
    id_to_idx = {aid: idx for idx, aid in enumerate(atom_ids)}
    user_edge_set: Set[Tuple[int, int]] = set()
    if user_links:
        for link in user_links:
            if len(link) != 2:
                continue
            u_raw, v_raw = link
            u_idx = id_to_idx.get(str(u_raw)) if str(u_raw) in id_to_idx else None
            if u_idx is None and isinstance(u_raw, int) and 0 <= u_raw < len(atom_ids):
                u_idx = u_raw
            v_idx = id_to_idx.get(str(v_raw)) if str(v_raw) in id_to_idx else None
            if v_idx is None and isinstance(v_raw, int) and 0 <= v_raw < len(atom_ids):
                v_idx = v_raw
            if u_idx is not None and v_idx is not None:
                user_edge_set.add((min(u_idx, v_idx), max(u_idx, v_idx)))

    current_adj = [list(nbrs) for nbrs in adj]
    chosen_edges: List[Tuple[int, int]] = []
    reports: List[FerromagneticLinkReport] = []

    while True:
        cycle = find_odd_cycle(current_adj)
        if cycle is None:
            break

        k = len(cycle)
        cycle_edges = [
            (min(cycle[i], cycle[(i + 1) % k]), max(cycle[i], cycle[(i + 1) % k]))
            for i in range(k)
        ]

        candidate: Optional[Tuple[int, int]] = None
        reason: str = ""
        for edge in cycle_edges:
            if edge in user_edge_set:
                candidate = edge
                reason = "user-specified ferromagnetic link"
                break

        if candidate is None:
            best_edge: Optional[Tuple[int, int]] = None
            best_dist = -1.0
            best_tiebreak: Optional[Tuple[str, str]] = None
            for u, v in cycle_edges:
                xu, yu, zu = coords[u]
                xv, yv, zv = coords[v]
                dist = math.sqrt((xu - xv) ** 2 + (yu - yv) ** 2 + (zu - zv) ** 2)
                tiebreak = (min(atom_ids[u], atom_ids[v]), max(atom_ids[u], atom_ids[v]))
                if dist > best_dist + 1e-6:
                    best_dist = dist
                    best_edge = (u, v)
                    best_tiebreak = tiebreak
                elif abs(dist - best_dist) <= 1e-6:
                    if best_tiebreak is None or tiebreak < best_tiebreak:
                        best_dist = dist
                        best_edge = (u, v)
                        best_tiebreak = tiebreak

            candidate = best_edge
            reason = "longest bond (weakest covalent coupling)"

        assert candidate is not None
        u, v = candidate
        xu, yu, zu = coords[u]
        xv, yv, zv = coords[v]
        dist = math.sqrt((xu - xv) ** 2 + (yu - yv) ** 2 + (zu - zv) ** 2)
        pair_ids = (min(atom_ids[u], atom_ids[v]), max(atom_ids[u], atom_ids[v]))
        ring_ids = [atom_ids[idx] for idx in cycle]

        reports.append(FerromagneticLinkReport(
            atom_pair=pair_ids,
            ring_atoms=ring_ids,
            ring_size=len(cycle),
            distance_bohr=dist,
            reason=reason,
        ))
        chosen_edges.append(candidate)

        current_adj[u].remove(v)
        current_adj[v].remove(u)

    return current_adj, chosen_edges, reports


def assign_alternating_spin(
    geometry: Union[List[AtomConfig], Any],
    start_spin: str = "up",
    tolerance: float = 1.25,
    ferromagnetic_links: Optional[List[Tuple[Union[str, int], Union[str, int]]]] = None,
    require_bipartite: bool = False,
    return_reports: bool = False,
    quiet: bool = False,
) -> Union[List[AtomConfig], Tuple[List[AtomConfig], List[FerromagneticLinkReport]]]:
    """
    Assigns alternating spin ('up' / 'down') to a list of atoms based on connectivity.

    .. deprecated::
        Automated spin guessing via alternating spin is deprecated in favor of explicit
        per-atom spin assignments. Non-bipartite atom graphs (containing odd cycles/rings)
        automatically identify and report ferromagnetic couplings, or raise
        NonBipartiteSpinGraphError if require_bipartite=True.

    Args:
        geometry: List of AtomConfig objects.
        start_spin: Starting spin direction ('up' or 'down').
        tolerance: Adjacency distance scaling factor.
        ferromagnetic_links: Optional list of user-specified bonds [(id1, id2), ...]
                             to select as ferromagnetic couplings in frustrated rings.
        require_bipartite: If True, strictly refuses non-bipartite graphs with NonBipartiteSpinGraphError.
                           If False (default), detects and reports ferromagnetic couplings.
        return_reports: If True, returns (atoms, reports) tuple.
        quiet: If True, suppresses printed reporting messages.

    Returns:
        Updated list of AtomConfig objects (or (atoms, reports) if return_reports=True).
    """
    warnings.warn(
        "assign_alternating_spin / auto_spin_flip is deprecated and will be removed in a future release. "
        "Specify explicit per-atom spins ('up' or 'down') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if hasattr(geometry, "geometry"):
        atoms = geometry.geometry
    else:
        atoms = geometry

    if not atoms:
        return (atoms, []) if return_reports else atoms

    # Check for user-locked spins
    locked: Dict[int, str] = {}
    for idx, a in enumerate(atoms):
        if a.spin in ("up", "down") and not getattr(a, "_auto_assigned", False):
            locked[idx] = a.spin

    adj = build_connectivity_graph(atoms, tolerance=tolerance)
    odd_cycle = find_odd_cycle(adj)

    reports: List[FerromagneticLinkReport] = []
    if odd_cycle is not None:
        cycle_atoms = [atoms[i].id or f"{atoms[i].element}{i+1}" for i in odd_cycle]
        if require_bipartite:
            raise NonBipartiteSpinGraphError(
                f"Cannot assign alternating spin: molecular graph contains an odd cycle (frustrated ring): {cycle_atoms}. "
                f"Non-bipartite graphs cannot be 2-colored; explicit per-atom spins are required."
            )

        coords = [a.xyz for a in atoms]
        atom_ids = [a.id or f"{a.element}{i+1}" for i, a in enumerate(atoms)]
        bipartite_adj, _, reports = detect_ferromagnetic_links(
            adj, coords, atom_ids, user_links=ferromagnetic_links
        )
        spins = solve_alternating_spin(
            bipartite_adj,
            start_spin=start_spin,
            locked_spins=locked,
            require_bipartite=False,
        )

        if not quiet:
            for rep in reports:
                print(
                    f"[SPIN] Non-bipartite (frustrated) ring detected: {rep.ring_atoms} (size {rep.ring_size})\n"
                    f"[SPIN] Ferromagnetic coupling located on bond: {rep.atom_pair} "
                    f"[d = {rep.distance_bohr:.3f} Bohr] ({rep.reason})"
                )
            spin_summary = ", ".join(f"{atom_ids[i]}: {spins[i]}" for i in range(len(atoms)))
            n_up = sum(1 for s in spins if s == "up")
            n_down = sum(1 for s in spins if s == "down")
            print(f"[SPIN] Full spin assignment: {spin_summary} (net_spin = {n_up - n_down})")
    else:
        spins = solve_alternating_spin(
            adj,
            start_spin=start_spin,
            locked_spins=locked,
            require_bipartite=True,
        )

    for idx, a in enumerate(atoms):
        a.spin = spins[idx]
        a.spin_flip = (spins[idx] == "down")
        setattr(a, "_auto_assigned", True)

    if return_reports:
        return atoms, reports
    return atoms


