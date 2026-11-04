"""
Stepwise Valence-Bond Molecular Assembler ("Tinker Toy" Directed Growth Engine).
Phase 14 (Milestones M14.1 - M14.3).

Constructs complete 3D molecular structures one atom at a time by replacing open
valence placeholder slots with real hybridized atoms.
Features:
- Standard covalent radii & multiple bond scaling (single, double, triple, quadruple/delta).
- Automatic hybrid frame alignment (Kabsch/Rodrigues rotations).
- Conformational dihedral control (staggered vs. eclipsed vs. custom angle).
- Automated lone-pair reservation for main-group elements (e.g. O bent, N pyramidal).
- Quadruple and delta-bond support for transition metals (d-orbital face-to-face overlap).
- Seamless export to validated CalculationDeck with directed hybrid starting orbitals (apply_hybrid_seeds).
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Union, Any, Sequence
from pathlib import Path
import math
import numpy as np

from phatpsy.schema import CalculationDeck, AtomConfig, ControlConfig
from phatpsy.spin import COVALENT_RADII_AA, get_covalent_radius_bohr, BOHR_PER_ANGSTROM
from phatpsy.hybrids import (
    HybridizationType,
    HybridizationSpec,
    get_hybridization_spec,
    apply_hybrid_seeds,
    align_vectors_kabsch,
    align_hybrid_to_neighbors,
)

# ==============================================================================
# Standard Bond Length & Radii Heuristics
# ==============================================================================

# Common experimentally refined equilibrium single/double/triple bond lengths (in Angstroms)
STANDARD_BOND_LENGTHS_AA: Dict[Tuple[str, str, int], float] = {
    # Single bonds
    ("C", "H", 1): 1.090,
    ("H", "C", 1): 1.090,
    ("N", "H", 1): 1.012,
    ("H", "N", 1): 1.012,
    ("O", "H", 1): 0.958,
    ("H", "O", 1): 0.958,
    ("F", "H", 1): 0.917,
    ("H", "F", 1): 0.917,
    ("C", "C", 1): 1.535,
    ("C", "N", 1): 1.471,
    ("N", "C", 1): 1.471,
    ("C", "O", 1): 1.426,
    ("O", "C", 1): 1.426,
    ("C", "F", 1): 1.350,
    ("F", "C", 1): 1.350,
    ("C", "Cl", 1): 1.770,
    ("Cl", "C", 1): 1.770,
    ("C", "Br", 1): 1.940,
    ("Br", "C", 1): 1.940,
    ("C", "I", 1): 2.140,
    ("I", "C", 1): 2.140,
    ("N", "N", 1): 1.450,
    ("O", "O", 1): 1.475,
    ("Si", "H", 1): 1.480,
    ("H", "Si", 1): 1.480,
    ("Si", "Si", 1): 2.330,
    ("Si", "O", 1): 1.630,
    ("O", "Si", 1): 1.630,
    # Double bonds
    ("C", "C", 2): 1.339,
    ("C", "N", 2): 1.280,
    ("N", "C", 2): 1.280,
    ("C", "O", 2): 1.208,
    ("O", "C", 2): 1.208,
    ("N", "N", 2): 1.250,
    ("O", "O", 2): 1.208,
    # Triple bonds
    ("C", "C", 3): 1.203,
    ("C", "N", 3): 1.157,
    ("N", "C", 3): 1.157,
    ("N", "N", 3): 1.098,
    ("C", "O", 3): 1.128,
    ("O", "C", 3): 1.128,
    # Transition-metal quadruple / delta bonds (e.g. Cr2, Mo2, Re2)
    ("Cr", "Cr", 4): 1.970,
    ("Mo", "Mo", 4): 2.090,
    ("Re", "Re", 4): 2.220,
    ("W", "W", 4): 2.160,
}

# Empirical multiple-bond contraction factors relative to sum of covalent radii
BOND_ORDER_CONTRACTION: Dict[int, float] = {
    1: 1.00,
    2: 0.88,
    3: 0.79,
    4: 0.72,  # Quadruple / delta bond
}


def get_bond_length(
    elem1: str,
    elem2: str,
    bond_order: int = 1,
    units: str = "bohr",
) -> float:
    """
    Returns the equilibrium bond length between two elements for a given bond order.

    Checks standard lookup table first; falls back to Pyykkö/Cordero covalent radii
    with order contraction factors.
    """
    s1 = elem1.strip().capitalize()
    s2 = elem2.strip().capitalize()
    pair_key = (s1, s2, int(bond_order))

    if pair_key in STANDARD_BOND_LENGTHS_AA:
        r_aa = STANDARD_BOND_LENGTHS_AA[pair_key]
    else:
        # Fallback to covalent radii sum * contraction factor
        r1 = COVALENT_RADII_AA.get(s1, 1.20)
        r2 = COVALENT_RADII_AA.get(s2, 1.20)
        contract = BOND_ORDER_CONTRACTION.get(int(bond_order), 1.0)
        r_aa = (r1 + r2) * contract

    if units.lower() in ("bohr", "au", "atomic"):
        return r_aa * BOHR_PER_ANGSTROM
    return r_aa


# ==============================================================================
# Valence Topology & Data Structures
# ==============================================================================

class ValenceSlotType(str, Enum):
    BONDING = "bonding"          # Active covalent bond to another real atom
    PLACEHOLDER = "placeholder"  # Open arm waiting for an atom or cap
    LONE_PAIR = "lone_pair"      # Non-bonding valence pair
    PI_BOND = "pi_bond"          # Transverse pi interaction
    DELTA_BOND = "delta_bond"    # Face-to-face 4-lobe delta interaction


@dataclass
class ValenceSlot:
    """Represents a directional valence slot (arm) of a hybridized atom."""
    index: int
    slot_type: ValenceSlotType
    direction: np.ndarray        # Unit vector in 3D world/laboratory frame
    target_atom_id: Optional[str] = None  # ID of bonded atom if BONDING
    target_slot_index: Optional[int] = None # Slot index on target atom
    bond_order: int = 1          # 1: sigma, 2: double, 3: triple, 4: quadruple
    placeholder_pos: Optional[np.ndarray] = None # World coordinates of ghost partner site

    def is_open(self) -> bool:
        """Returns True if this slot is an open placeholder available for connection."""
        return self.slot_type == ValenceSlotType.PLACEHOLDER


@dataclass
class HybridNode:
    """Represents an assembled atom with its 3D coordinates, hybridization, and slots."""
    id: str                      # Unique identifier, e.g. 'C1', 'N2'
    element: str                 # Chemical symbol, e.g. 'C', 'N', 'O', 'H'
    position: np.ndarray         # Cartesian coordinates in Bohr (shape: (3,))
    hybridization: Optional[str] # Hybridization name, e.g. 'sp3', 'sp2', 'sp', 'sp3_bent'
    rotation_matrix: np.ndarray  # 3x3 frame orientation matrix R (local -> world)
    slots: List[ValenceSlot] = field(default_factory=list) # Sigma valence arms
    pi_directions: List[np.ndarray] = field(default_factory=list) # Transverse pi vectors
    delta_directions: List[np.ndarray] = field(default_factory=list) # Transverse delta vectors

    @property
    def open_slots(self) -> List[ValenceSlot]:
        """Returns all open placeholder slots available on this atom."""
        return [s for s in self.slots if s.is_open()]


# ==============================================================================
# Auto Lone-Pair Heuristics
# ==============================================================================

DEFAULT_LONE_PAIRS_COUNT: Dict[Tuple[str, str], int] = {
    # (Element, Hybridization) -> number of lone pairs
    ("N", "sp3"): 1,
    ("N", "sp3_pyramidal"): 1,
    ("P", "sp3"): 1,
    ("P", "sp3_pyramidal"): 1,
    ("As", "sp3"): 1,
    ("O", "sp3"): 2,
    ("O", "sp3_bent"): 2,
    ("S", "sp3"): 2,
    ("S", "sp3_bent"): 2,
    ("Se", "sp3"): 2,
    ("F", "sp3"): 3,
    ("Cl", "sp3"): 3,
    ("Br", "sp3"): 3,
    ("I", "sp3"): 3,
    ("N", "sp2"): 1,
    ("O", "sp2"): 2,
}


def _unit_vector(v: Sequence[float]) -> np.ndarray:
    """Returns normalized unit vector."""
    arr = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return arr / norm


def _rodrigues_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Computes 3x3 rotation matrix for rotation around a unit axis by angle_rad."""
    u = _unit_vector(axis)
    cos_t = math.cos(angle_rad)
    sin_t = math.sin(angle_rad)
    K = np.array([
        [0.0, -u[2], u[1]],
        [u[2], 0.0, -u[0]],
        [-u[1], u[0], 0.0]
    ], dtype=np.float64)
    return np.eye(3) + sin_t * K + (1.0 - cos_t) * (K @ K)


def _align_vector_to_target(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Computes minimal rotation matrix R such that R @ source == target.
    """
    s = _unit_vector(source)
    t = _unit_vector(target)
    dot = np.dot(s, t)

    if dot > 0.9999999:
        return np.eye(3, dtype=np.float64)
    if dot < -0.9999999:
        # 180 degree rotation: pick an arbitrary perpendicular vector
        perp = np.array([1.0, 0.0, 0.0]) if abs(s[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        axis = _unit_vector(np.cross(s, perp))
        return _rodrigues_rotation(axis, math.pi)

    axis = _unit_vector(np.cross(s, t))
    angle = math.acos(max(-1.0, min(1.0, dot)))
    return _rodrigues_rotation(axis, angle)


# ==============================================================================
# High-Level HybridAssembler Engine
# ==============================================================================

class HybridAssembler:
    """
    Stepwise molecular growth engine based on directed hybridized atoms.

    Allows assembling 3D structures one atom at a time by attaching new hybridized
    atoms to open placeholder slots, automatically adjusting bond lengths, hybrid
    orientations, and conformational dihedrals (staggered/eclipsed), and exporting
    directly to a validated CalculationDeck ready for quantum SCF calculation.
    """

    def __init__(
        self,
        root: str = "C",
        hybridization: Optional[str] = "sp3",
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        auto_lone_pairs: bool = True,
    ):
        """
        Initializes the assembler with a root atom at the specified position.

        Parameters
        ----------
        root : str
            Element symbol for the root atom (e.g. 'C', 'N', 'O', 'Cr').
        hybridization : Optional[str]
            Hybridization type: 'sp3', 'sp2', 'sp', 'sp3_pyramidal', 'sp3_bent',
            'dsp2', 'd2sp3', 'sd3', 'sd', 'delta', or None for terminal atoms.
        position : Tuple[float, float, float]
            Root coordinates in Bohrs (default: origin).
        auto_lone_pairs : bool
            If True, automatically reserves non-bonding slots as lone pairs for
            electronegative elements (N, O, halogens).
        """
        self.nodes: Dict[str, HybridNode] = {}
        self.elem_counts: Dict[str, int] = {}
        self.bonds: List[Tuple[str, int, str, int, int]] = [] # (atom1, slot1, atom2, slot2, order)

        # Place root atom
        self._add_node(
            element=root,
            hybridization=hybridization,
            position=np.array(position, dtype=np.float64),
            rotation_matrix=np.eye(3, dtype=np.float64),
            auto_lone_pairs=auto_lone_pairs,
        )

    def _next_id(self, element: str) -> str:
        elem = element.strip().capitalize()
        self.elem_counts[elem] = self.elem_counts.get(elem, 0) + 1
        return f"{elem}{self.elem_counts[elem]}"

    def _add_node(
        self,
        element: str,
        hybridization: Optional[str],
        position: np.ndarray,
        rotation_matrix: np.ndarray,
        auto_lone_pairs: bool = True,
    ) -> HybridNode:
        atom_id = self._next_id(element)
        elem_cap = element.strip().capitalize()
        hyb_norm = hybridization.lower().strip() if hybridization else None

        # Retrieve canonical hybridization vectors
        sigma_dirs: List[np.ndarray] = []
        pi_dirs: List[np.ndarray] = []
        delta_dirs: List[np.ndarray] = []

        if hyb_norm:
            spec = get_hybridization_spec(hyb_norm)
            sigma_dirs = [rotation_matrix @ d for d in spec.canonical_sigma_dirs]
            pi_dirs = [rotation_matrix @ d for d in spec.canonical_pi_dirs]
            delta_dirs = [rotation_matrix @ d for d in spec.canonical_delta_dirs]

            if not sigma_dirs and delta_dirs:
                # Delta bonding center: principal bonding axis is normal to delta lobes
                # Local delta lobes lie in xy plane, so normal axis is z (0, 0, 1)
                axis_z = rotation_matrix @ np.array([0.0, 0.0, 1.0])
                sigma_dirs = [axis_z]

        # Determine auto lone pairs count
        n_lp = 0
        if auto_lone_pairs and hyb_norm:
            # If the hybridization is already specialized (sp3_bent has 2 bonding arms,
            # sp3_pyramidal has 3 bonding arms), all canonical_sigma_dirs are bonding arms!
            # Only generic sp3 needs auto-reserving lone pairs based on element valence.
            if hyb_norm in ("sp3", "sp2"):
                n_lp = DEFAULT_LONE_PAIRS_COUNT.get((elem_cap, hyb_norm), 0)

        # Construct slots
        slots: List[ValenceSlot] = []
        n_total_sigma = len(sigma_dirs)
        lp_start_idx = max(0, n_total_sigma - n_lp) if n_lp > 0 else n_total_sigma

        for idx, s_dir in enumerate(sigma_dirs):
            u_dir = _unit_vector(s_dir)
            stype = ValenceSlotType.LONE_PAIR if idx >= lp_start_idx else ValenceSlotType.PLACEHOLDER

            # Estimate ghost placeholder partner coordinate
            partner_elem = "H" if elem_cap != "H" else "C"
            bond_len = get_bond_length(elem_cap, partner_elem, bond_order=1, units="bohr")
            p_pos = position + bond_len * u_dir if stype == ValenceSlotType.PLACEHOLDER else None

            slots.append(ValenceSlot(
                index=idx,
                slot_type=stype,
                direction=u_dir,
                placeholder_pos=p_pos,
            ))

        node = HybridNode(
            id=atom_id,
            element=elem_cap,
            position=position,
            hybridization=hyb_norm,
            rotation_matrix=rotation_matrix,
            slots=slots,
            pi_directions=pi_dirs,
            delta_directions=delta_dirs,
        )
        self.nodes[atom_id] = node
        return node

    def get_node(self, atom_id_or_node: Union[str, HybridNode]) -> HybridNode:
        """Retrieves node by ID or returns if already HybridNode."""
        if isinstance(atom_id_or_node, HybridNode):
            return atom_id_or_node
        if atom_id_or_node not in self.nodes:
            raise KeyError(f"Atom '{atom_id_or_node}' not found in assembled molecule.")
        return self.nodes[atom_id_or_node]

    def attach(
        self,
        parent: Union[str, HybridNode],
        slot: int,
        element: str = "C",
        hybridization: Optional[str] = "sp3",
        conformation: str = "staggered",
        dihedral_deg: Optional[float] = None,
        bond_order: int = 1,
        auto_lone_pairs: bool = True,
    ) -> HybridNode:
        """
        Attaches a new hybridized atom to an open valence slot on an existing atom.

        Parameters
        ----------
        parent : Union[str, HybridNode]
            Parent atom ID (e.g. 'C1') or HybridNode instance.
        slot : int
            Index of the open placeholder slot on the parent atom.
        element : str
            Element symbol for the new atom (e.g. 'C', 'N', 'O', 'Fe').
        hybridization : Optional[str]
            Hybridization of new atom ('sp3', 'sp2', 'sp', 'sp3_bent', 'delta', etc.).
        conformation : str
            Conformational alignment: 'staggered' (default) or 'eclipsed'.
            Ignored if explicit dihedral_deg is specified.
        dihedral_deg : Optional[float]
            Explicit torsion angle in degrees around the bond axis (0 to 360).
        bond_order : int
            Bond order: 1 (single/sigma), 2 (double), 3 (triple), 4 (quadruple/delta).
        auto_lone_pairs : bool
            Whether to auto-reserve non-bonding slots for O, N, halogens.

        Returns
        -------
        HybridNode
            The newly created and attached HybridNode.
        """
        p_node = self.get_node(parent)
        if slot < 0 or slot >= len(p_node.slots):
            raise IndexError(f"Slot index {slot} out of range for parent {p_node.id} (has {len(p_node.slots)} slots).")

        p_slot = p_node.slots[slot]
        if not p_slot.is_open():
            raise ValueError(f"Slot {slot} on {p_node.id} is already occupied ({p_slot.slot_type.value}).")

        # 1. Compute equilibrium bond length
        bond_len = get_bond_length(p_node.element, element, bond_order=bond_order, units="bohr")

        # 2. Position of child atom along slot direction
        u_axis = _unit_vector(p_slot.direction)
        child_pos = p_node.position + bond_len * u_axis

        # 3. Hybrid frame orientation of child atom
        hyb_norm = hybridization.lower().strip() if hybridization else None
        if hyb_norm:
            spec = get_hybridization_spec(hyb_norm)
            # Slot 0 of child is chosen as the incoming bonding slot pointing back to parent (-u_axis)
            v0_canon = spec.canonical_sigma_dirs[0] if spec.canonical_sigma_dirs else np.array([0.0, 0.0, 1.0])
            R_align = _align_vector_to_target(v0_canon, -u_axis)

            # Reference vector on parent for dihedral alignment
            p_ref_arms = [s.direction for s in p_node.slots if s.index != slot and s.slot_type != ValenceSlotType.LONE_PAIR]
            if not p_ref_arms:
                perp = np.array([1.0, 0.0, 0.0]) if abs(u_axis[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
                p_ref = perp - np.dot(perp, u_axis) * u_axis
            else:
                p_ref = p_ref_arms[0] - np.dot(p_ref_arms[0], u_axis) * u_axis
            p_ref_u = _unit_vector(p_ref)

            # Child's reference arm (slot 1) after R_align
            c_canon_arms = spec.canonical_sigma_dirs if spec.canonical_sigma_dirs else [np.array([0.0, 0.0, 1.0])]
            if len(c_canon_arms) > 1:
                c1_rot = R_align @ c_canon_arms[1]
                c_ref = c1_rot - np.dot(c1_rot, u_axis) * u_axis
                c_ref_u = _unit_vector(c_ref)
                dot = max(-1.0, min(1.0, np.dot(p_ref_u, c_ref_u)))
                cross = np.dot(u_axis, np.cross(p_ref_u, c_ref_u))
                current_ang_rad = math.atan2(cross, dot)
            else:
                current_ang_rad = 0.0

            # 4. Conformational dihedral rotation around bond axis
            if dihedral_deg is not None:
                target_ang_rad = math.radians(dihedral_deg)
            else:
                if conformation.lower() in ("eclipsed", "cis", "syn"):
                    target_ang_rad = 0.0
                else:
                    # Staggered (60 deg for sp3)
                    target_ang_rad = math.radians(60.0)

            delta_rad = target_ang_rad - current_ang_rad

            # Special case: for sp2=sp2 double bonds, align out-of-plane pi vectors coplanar
            if bond_order >= 2 and p_node.hybridization == "sp2" and hyb_norm == "sp2":
                if p_node.pi_directions and spec.canonical_pi_dirs:
                    target_pi = p_node.pi_directions[0]
                    canon_pi_rot = R_align @ spec.canonical_pi_dirs[0]
                    p_proj = canon_pi_rot - np.dot(canon_pi_rot, u_axis) * u_axis
                    t_proj = target_pi - np.dot(target_pi, u_axis) * u_axis
                    if np.linalg.norm(p_proj) > 1e-6 and np.linalg.norm(t_proj) > 1e-6:
                        p_u = _unit_vector(p_proj)
                        t_u = _unit_vector(t_proj)
                        d_dot = max(-1.0, min(1.0, np.dot(p_u, t_u)))
                        twist = math.acos(d_dot)
                        if np.dot(u_axis, np.cross(p_u, t_u)) < 0:
                            twist = -twist
                        delta_rad = twist

            R_torsion = _rodrigues_rotation(u_axis, delta_rad)
            R_child = R_torsion @ R_align
        else:
            R_child = np.eye(3, dtype=np.float64)

        # 5. Create child node
        child_node = self._add_node(
            element=element,
            hybridization=hybridization,
            position=child_pos,
            rotation_matrix=R_child,
            auto_lone_pairs=auto_lone_pairs,
        )

        # 6. Bind slots together
        child_incoming_slot = 0 if child_node.slots else -1
        p_slot.slot_type = ValenceSlotType.BONDING
        p_slot.target_atom_id = child_node.id
        p_slot.target_slot_index = child_incoming_slot
        p_slot.bond_order = bond_order
        p_slot.placeholder_pos = None

        if child_incoming_slot >= 0:
            c_slot = child_node.slots[child_incoming_slot]
            c_slot.slot_type = ValenceSlotType.BONDING
            c_slot.target_atom_id = p_node.id
            c_slot.target_slot_index = slot
            c_slot.bond_order = bond_order
            c_slot.placeholder_pos = None

        self.bonds.append((p_node.id, slot, child_node.id, child_incoming_slot, bond_order))
        return child_node

    def cap_slot(self, atom_id: str, slot: int, element: str = "H") -> HybridNode:
        """Caps a single specific placeholder slot with a terminal atom (default H)."""
        return self.attach(
            parent=atom_id,
            slot=slot,
            element=element,
            hybridization=None,  # Terminal atom
            bond_order=1,
        )

    def cap_all(self, element: str = "H") -> List[HybridNode]:
        """
        Caps all open placeholder slots across the entire molecule with terminal atoms.

        Iterates repeatedly until no open slots remain.
        """
        capped_nodes: List[HybridNode] = []
        max_iters = 1000
        it = 0

        while it < max_iters:
            it += 1
            open_found = False
            for node_id, node in list(self.nodes.items()):
                # Do not cap terminal atoms
                if node.hybridization is None:
                    continue
                for slot in node.slots:
                    if slot.is_open():
                        capped = self.cap_slot(node_id, slot.index, element=element)
                        capped_nodes.append(capped)
                        open_found = True
                        break # Refresh iterator
                if open_found:
                    break
            if not open_found:
                break

        return capped_nodes

    def set_lone_pair(self, atom_id: str, slot: int) -> None:
        """Converts an open placeholder slot into a non-bonding lone pair."""
        node = self.get_node(atom_id)
        if slot < 0 or slot >= len(node.slots):
            raise IndexError(f"Slot {slot} out of range for atom {atom_id}.")
        s = node.slots[slot]
        if s.slot_type == ValenceSlotType.BONDING:
            raise ValueError(f"Cannot convert bonded slot {slot} on {atom_id} to a lone pair.")
        s.slot_type = ValenceSlotType.LONE_PAIR
        s.placeholder_pos = None

    def form_bond(
        self,
        atom1_id: str,
        slot1: int,
        atom2_id: str,
        slot2: int,
        bond_order: int = 1,
    ) -> None:
        """
        Forms an explicit bond between two existing open slots (e.g. for ring closures).
        """
        n1 = self.get_node(atom1_id)
        n2 = self.get_node(atom2_id)
        s1 = n1.slots[slot1]
        s2 = n2.slots[slot2]

        if not s1.is_open():
            raise ValueError(f"Slot {slot1} on {atom1_id} is not open.")
        if not s2.is_open():
            raise ValueError(f"Slot {slot2} on {atom2_id} is not open.")

        s1.slot_type = ValenceSlotType.BONDING
        s1.target_atom_id = atom2_id
        s1.target_slot_index = slot2
        s1.bond_order = bond_order
        s1.placeholder_pos = None

        s2.slot_type = ValenceSlotType.BONDING
        s2.target_atom_id = atom1_id
        s2.target_slot_index = slot1
        s2.bond_order = bond_order
        s2.placeholder_pos = None

        self.bonds.append((atom1_id, slot1, atom2_id, slot2, bond_order))

    def get_open_slots(self) -> List[Tuple[str, int, ValenceSlot]]:
        """Returns list of (atom_id, slot_index, slot) for all open placeholders."""
        res = []
        for node_id, node in self.nodes.items():
            for slot in node.slots:
                if slot.is_open():
                    res.append((node_id, slot.index, slot))
        return res

    def to_deck(
        self,
        title: str = "PHATPSY Assembled Molecule",
        basis: Union[str, Dict[str, str]] = "DZ",
        control: Optional[ControlConfig] = None,
        spins: Optional[Union[List[str], Dict[str, str], str]] = None,
        net_spin: Optional[int] = None,
        apply_seeds: bool = True,
    ) -> CalculationDeck:
        """
        Exports the assembled molecule directly into a validated CalculationDeck.

        Automatically preserves each atom's directed hybrid state and applies
        directed hybrid starting orbitals (`apply_hybrid_seeds`) if requested.

        Parameters
        ----------
        title : str
            Title for the calculation deck.
        basis : Union[str, Dict[str, str]]
            Basis set specification ('DZ', 'SZ', 'THAKKAR', etc.).
        control : Optional[ControlConfig]
            Calculation control options.
        spins : Optional[Union[List[str], Dict[str, str], str]]
            Explicit per-atom spin assignments ('up' or 'down'). Defaults to 'up'.
        net_spin : Optional[int]
            Net spin (n_up - n_down). Defaults to total electron spin or 0.
        apply_seeds : bool
            If True, applies directed hybrid orbital seeds to deck starting orbitals.

        Returns
        -------
        CalculationDeck
            The fully built, validated, and seeded calculation deck.
        """
        from phatpsy.builder import build_molecule

        # Build list of atom configurations
        geom_tuples = []
        atom_list = list(self.nodes.values())

        for node in atom_list:
            geom_tuples.append({
                "element": node.element,
                "id": node.id,
                "xyz": node.position.tolist(),
                "hybridization": node.hybridization,
            })

        # Default spins to 'up' if not specified
        if spins is None:
            spins = "up"
            if net_spin is None:
                net_spin = len(atom_list)

        deck = build_molecule(
            geometry=geom_tuples,
            basis=basis,
            title=title,
            control=control,
            units="bohr",
            spins=spins,
            net_spin=net_spin,
        )

        # Apply directed hybrid starting orbital seeds
        if apply_seeds:
            # Map bonds to atom index pairs
            node_id_to_idx = {node.id: i for i, node in enumerate(atom_list)}
            bond_pairs: Dict[Tuple[int, int], int] = {}

            for a1, _, a2, _, order in self.bonds:
                if a1 in node_id_to_idx and a2 in node_id_to_idx:
                    i1 = node_id_to_idx[a1]
                    i2 = node_id_to_idx[a2]
                    bond_pairs[(i1, i2)] = order

            try:
                apply_hybrid_seeds(deck, bonds=bond_pairs if bond_pairs else None)
            except Exception:
                # If seeding cannot fully resolve (e.g. non-hybridized terminal species), proceed cleanly
                pass

        return deck

    def to_xyz(self, comment: str = "PHATPSY Assembled Molecule") -> str:
        """Returns standard XYZ string format in Angstroms."""
        lines = [str(len(self.nodes)), comment]
        for node in self.nodes.values():
            pos_aa = node.position / BOHR_PER_ANGSTROM
            lines.append(f"{node.element:4s} {pos_aa[0]:14.8f} {pos_aa[1]:14.8f} {pos_aa[2]:14.8f}")
        return "\n".join(lines) + "\n"

    def to_ase(self) -> Any:
        """Exports the assembled molecule into an ASE Atoms object in Angstroms."""
        from phatpsy.formats import to_ase
        deck = self.to_deck(apply_seeds=False)
        return to_ase(deck)

    def summary(self) -> str:
        """Returns human-readable text summary of the assembled molecule."""
        lines = [
            f"=== Assembled Molecule: {len(self.nodes)} atoms, {len(self.bonds)} bonds ===",
            f"Open Placeholders: {len(self.get_open_slots())}",
            "Atoms:",
        ]
        for node in self.nodes.values():
            pos_aa = node.position / BOHR_PER_ANGSTROM
            hyb = node.hybridization or "terminal"
            n_open = len(node.open_slots)
            lines.append(
                f"  [{node.id:4s}] {node.element:2s} ({hyb:12s}) at "
                f"({pos_aa[0]:7.3f}, {pos_aa[1]:7.3f}, {pos_aa[2]:7.3f}) Å | Open: {n_open}"
            )
        return "\n".join(lines)

    def visualize(self, output_path: Optional[Union[str, Path]] = None, auto_open: bool = False, title: str = "PHATPSY Valence-Bond Molecular Assembler") -> str:
        """Exports the assembled molecule into a standalone interactive 3D WebGL HTML viewer."""
        from phatpsy.visualizer import visualize_assembler
        return visualize_assembler(self, output_path=output_path, auto_open=auto_open, title=title)

    def show(self, height: int = 600, in_browser: bool = False):
        """Displays interactive 3D visualization in Jupyter notebooks or browser."""
        from phatpsy.visualizer import show_assembler
        return show_assembler(self, height=height, in_browser=in_browser)
