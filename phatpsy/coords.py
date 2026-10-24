# phatpsy/coords.py
"""
Internal coordinates, Wilson B-matrix, and Z-matrix transformations for PHATPSY 2.0.
Supports bond lengths, valence angles, dihedral torsions, and automatic
coordinate generation from Cartesian geometries.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union, Sequence
import math
import numpy as np

from phatpsy.schema import AtomConfig
from phatpsy.spin import build_connectivity_graph, BOHR_PER_ANGSTROM


def normalize_vector(v: np.ndarray) -> Tuple[np.ndarray, float]:
    """Normalizes vector and returns (unit_vector, length)."""
    norm = float(np.linalg.norm(v))
    if norm < 1e-14:
        return np.zeros_like(v), 0.0
    return v / norm, norm


@dataclass
class BondCoordinate:
    """Internal coordinate representing bond distance between atom1 and atom2."""
    atom1: int
    atom2: int
    name: Optional[str] = None

    def value(self, coords: np.ndarray) -> float:
        """Returns bond length in Bohr."""
        r1 = coords[self.atom1]
        r2 = coords[self.atom2]
        return float(np.linalg.norm(r1 - r2))

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        """
        Wilson B-matrix vector for this bond length: dR / dX.
        Returns array of shape (N_atoms, 3).
        """
        n_atoms = len(coords)
        grad = np.zeros((n_atoms, 3), dtype=np.float64)
        r1 = coords[self.atom1]
        r2 = coords[self.atom2]
        u12, length = normalize_vector(r1 - r2)
        grad[self.atom1] = u12
        grad[self.atom2] = -u12
        return grad


@dataclass
class AngleCoordinate:
    """Internal coordinate representing valence angle between atom1-atom2-atom3 (vertex at atom2)."""
    atom1: int
    atom2: int  # Vertex
    atom3: int
    name: Optional[str] = None

    def value(self, coords: np.ndarray, in_degrees: bool = False) -> float:
        """Returns angle in radians (or degrees)."""
        r1 = coords[self.atom1]
        r2 = coords[self.atom2]
        r3 = coords[self.atom3]

        v21, d21 = normalize_vector(r1 - r2)
        v23, d23 = normalize_vector(r3 - r2)

        dot = float(np.clip(np.dot(v21, v23), -1.0, 1.0))
        theta = float(np.arccos(dot))
        return math.degrees(theta) if in_degrees else theta

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        """
        Wilson B-matrix vector for valence angle: d(theta) / dX (in 1/Bohr).
        Returns array of shape (N_atoms, 3).
        """
        n_atoms = len(coords)
        grad = np.zeros((n_atoms, 3), dtype=np.float64)
        r1 = coords[self.atom1]
        r2 = coords[self.atom2]
        r3 = coords[self.atom3]

        u21, d21 = normalize_vector(r1 - r2)
        u23, d23 = normalize_vector(r3 - r2)

        dot = float(np.clip(np.dot(u21, u23), -1.0, 1.0))
        sin_theta = math.sqrt(max(1.0 - dot * dot, 1e-14))

        # s_1 = (cos(theta)*u21 - u23) / (d21 * sin(theta))
        s1 = (dot * u21 - u23) / (d21 * sin_theta)
        # s_3 = (cos(theta)*u23 - u21) / (d23 * sin(theta))
        s3 = (dot * u23 - u21) / (d23 * sin_theta)
        # s_2 = -s1 - s3
        s2 = -s1 - s3

        grad[self.atom1] = s1
        grad[self.atom2] = s2
        grad[self.atom3] = s3
        return grad


@dataclass
class DihedralCoordinate:
    """Internal coordinate representing dihedral torsion angle atom1-atom2-atom3-atom4."""
    atom1: int
    atom2: int
    atom3: int
    atom4: int
    name: Optional[str] = None

    def value(self, coords: np.ndarray, in_degrees: bool = False) -> float:
        """Returns dihedral angle in radians (or degrees) in [-pi, pi]."""
        p1 = coords[self.atom1]
        p2 = coords[self.atom2]
        p3 = coords[self.atom3]
        p4 = coords[self.atom4]

        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3

        # Normal vectors to the two planes
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        n1_u, _ = normalize_vector(n1)
        n2_u, _ = normalize_vector(n2)
        b2_u, _ = normalize_vector(b2)

        m1 = np.cross(n1_u, b2_u)
        x = float(np.dot(n1_u, n2_u))
        y = float(np.dot(m1, n2_u))

        phi = float(np.arctan2(y, x))
        return math.degrees(phi) if in_degrees else phi

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        """Wilson B-matrix vector for dihedral angle: d(phi) / dX."""
        n_atoms = len(coords)
        grad = np.zeros((n_atoms, 3), dtype=np.float64)

        p1 = coords[self.atom1]
        p2 = coords[self.atom2]
        p3 = coords[self.atom3]
        p4 = coords[self.atom4]

        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3

        d2 = float(np.linalg.norm(b2))
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        len_n1_sq = max(float(np.dot(n1, n1)), 1e-14)
        len_n2_sq = max(float(np.dot(n2, n2)), 1e-14)

        s1 = (d2 / len_n1_sq) * n1
        s4 = (-d2 / len_n2_sq) * n2

        torque = np.cross(b1, s1) - np.cross(b2 + b3, s4)
        s3 = -np.cross(b2, torque) / max(d2 * d2, 1e-14)
        s2 = -s1 - s3 - s4

        grad[self.atom1] = s1
        grad[self.atom2] = s2
        grad[self.atom3] = s3
        grad[self.atom4] = s4
        return grad


@dataclass
class InternalCoordinates:
    """Collection of internal coordinates (bonds, angles, dihedrals) defining a molecular system."""
    bonds: List[BondCoordinate] = field(default_factory=list)
    angles: List[AngleCoordinate] = field(default_factory=list)
    dihedrals: List[DihedralCoordinate] = field(default_factory=list)

    @property
    def num_coordinates(self) -> int:
        return len(self.bonds) + len(self.angles) + len(self.dihedrals)

    def calculate(self, coords: np.ndarray, angles_in_degrees: bool = False) -> np.ndarray:
        """Calculates values of all internal coordinates."""
        vals = []
        for b in self.bonds:
            vals.append(b.value(coords))
        for a in self.angles:
            vals.append(a.value(coords, in_degrees=angles_in_degrees))
        for d in self.dihedrals:
            vals.append(d.value(coords, in_degrees=angles_in_degrees))
        return np.array(vals, dtype=np.float64)

    def b_matrix(self, coords: np.ndarray) -> np.ndarray:
        """
        Wilson B-matrix relating internal coordinate changes to Cartesian displacements:
        B_{ki} = dq_k / dx_i
        Shape: (N_internal, 3 * N_atoms).
        """
        n_atoms = len(coords)
        n_coords = self.num_coordinates
        b_mat = np.zeros((n_coords, 3 * n_atoms), dtype=np.float64)

        idx = 0
        for b in self.bonds:
            b_mat[idx] = b.gradient(coords).reshape(-1)
            idx += 1
        for a in self.angles:
            b_mat[idx] = a.gradient(coords).reshape(-1)
            idx += 1
        for d in self.dihedrals:
            b_mat[idx] = d.gradient(coords).reshape(-1)
            idx += 1

        return b_mat

    @classmethod
    def from_atoms(cls, atoms: List[AtomConfig], tolerance: float = 1.30) -> "InternalCoordinates":
        """
        Automatically detects all chemical bonds, valence angles, and dihedral torsions
        from atomic coordinates using covalent radii connectivity.
        """
        n_atoms = len(atoms)
        if n_atoms < 2:
            return cls()

        adj = build_connectivity_graph(atoms, tolerance=tolerance)
        bonds: List[BondCoordinate] = []
        angles: List[AngleCoordinate] = []
        dihedrals: List[DihedralCoordinate] = []

        # 1. Bonds
        for i in range(n_atoms):
            for j in adj[i]:
                if i < j:
                    bonds.append(BondCoordinate(atom1=i, atom2=j, name=f"R({atoms[i].element}{i+1}-{atoms[j].element}{j+1})"))

        # 2. Angles: i-j-k where j is vertex
        for j in range(n_atoms):
            nbrs = adj[j]
            for idx_a, i in enumerate(nbrs):
                for k in nbrs[idx_a + 1:]:
                    angles.append(AngleCoordinate(atom1=i, atom2=j, atom3=k, name=f"A({atoms[i].element}{i+1}-{atoms[j].element}{j+1}-{atoms[k].element}{k+1})"))

        # 3. Dihedrals: i-j-k-l
        for b in bonds:
            j, k = b.atom1, b.atom2
            for i in adj[j]:
                if i == k:
                    continue
                for l in adj[k]:
                    if l == j or l == i:
                        continue
                    dihedrals.append(DihedralCoordinate(
                        atom1=i, atom2=j, atom3=k, atom4=l,
                        name=f"D({atoms[i].element}{i+1}-{atoms[j].element}{j+1}-{atoms[k].element}{k+1}-{atoms[l].element}{l+1})"
                    ))

        return cls(bonds=bonds, angles=angles, dihedrals=dihedrals)


@dataclass
class ZMatrixEntry:
    """Entry for a single atom in a Z-matrix."""
    element: str
    ref_bond: Optional[int] = None       # Index of reference atom for bond length
    bond_length: Optional[float] = None   # Bond distance (Bohr)
    ref_angle: Optional[int] = None      # Index of reference atom for angle
    angle_val: Optional[float] = None    # Angle (degrees)
    ref_dihedral: Optional[int] = None   # Index of reference atom for dihedral
    dihedral_val: Optional[float] = None # Dihedral (degrees)


@dataclass
class ZMatrix:
    """Z-matrix (internal coordinates representation of molecular geometry)."""
    entries: List[ZMatrixEntry] = field(default_factory=list)

    @classmethod
    def from_atoms(cls, atoms: List[AtomConfig]) -> "ZMatrix":
        """Builds a canonical Z-matrix from a list of Cartesian AtomConfigs."""
        coords = np.array([a.xyz for a in atoms], dtype=np.float64)
        entries: List[ZMatrixEntry] = []

        for i, atom in enumerate(atoms):
            elem = atom.element
            if i == 0:
                entries.append(ZMatrixEntry(element=elem))
            elif i == 1:
                r12 = float(np.linalg.norm(coords[1] - coords[0]))
                entries.append(ZMatrixEntry(element=elem, ref_bond=0, bond_length=r12))
            elif i == 2:
                r23 = float(np.linalg.norm(coords[2] - coords[1]))
                ang = AngleCoordinate(0, 1, 2).value(coords, in_degrees=True)
                entries.append(ZMatrixEntry(element=elem, ref_bond=1, bond_length=r23, ref_angle=0, angle_val=ang))
            else:
                # Atom 4+: find best reference chain (e.g. previous three atoms i-1, i-2, i-3)
                ref_b = i - 1
                ref_a = i - 2
                ref_d = i - 3
                r = float(np.linalg.norm(coords[i] - coords[ref_b]))
                ang = AngleCoordinate(ref_a, ref_b, i).value(coords, in_degrees=True)
                dih = DihedralCoordinate(ref_d, ref_a, ref_b, i).value(coords, in_degrees=True)
                entries.append(ZMatrixEntry(
                    element=elem,
                    ref_bond=ref_b, bond_length=r,
                    ref_angle=ref_a, angle_val=ang,
                    ref_dihedral=ref_d, dihedral_val=dih
                ))

        return cls(entries=entries)

    def to_cartesian(self) -> np.ndarray:
        """Converts Z-matrix internal entries back to Cartesian coordinates (in Bohr)."""
        n = len(self.entries)
        coords = np.zeros((n, 3), dtype=np.float64)
        if n == 0:
            return coords

        # Atom 0 at origin
        coords[0] = [0.0, 0.0, 0.0]
        if n == 1:
            return coords

        # Atom 1 along Z-axis
        e1 = self.entries[1]
        coords[1] = [0.0, 0.0, e1.bond_length or 0.0]
        if n == 2:
            return coords

        # Atom 2 in XZ-plane
        e2 = self.entries[2]
        r = e2.bond_length or 0.0
        theta = math.radians(e2.angle_val or 0.0)
        p1 = coords[e2.ref_angle or 0]
        p2 = coords[e2.ref_bond or 1]
        u_bond, _ = normalize_vector(p1 - p2)  # Direction towards ref_angle
        coords[2] = p2 + np.array([r * math.sin(theta), 0.0, -r * math.cos(theta)])
        if n == 3:
            return coords

        # Atom 4+ using standard NeRF (Natural Extension Reference Frame) / dihedral placement
        for i in range(3, n):
            entry = self.entries[i]
            r = entry.bond_length or 1.0
            theta = math.radians(entry.angle_val or 109.5)
            phi = math.radians(entry.dihedral_val or 0.0)

            p_b = coords[entry.ref_bond]
            p_a = coords[entry.ref_angle]
            p_d = coords[entry.ref_dihedral]

            v_ab, _ = normalize_vector(p_b - p_a)
            v_da, _ = normalize_vector(p_a - p_d)

            n_plane, _ = normalize_vector(np.cross(v_da, v_ab))
            if np.linalg.norm(n_plane) < 1e-10:
                # Collinear fallback: pick arbitrary orthogonal vector
                n_plane = np.array([1.0, 0.0, 0.0]) if abs(v_ab[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                n_plane, _ = normalize_vector(np.cross(n_plane, v_ab))

            m_axis = np.cross(n_plane, v_ab)

            # Local coordinate of atom i relative to p_b
            local_vec = (
                -r * math.cos(theta) * v_ab +
                r * math.sin(theta) * math.cos(phi) * m_axis +
                r * math.sin(theta) * math.sin(phi) * n_plane
            )
            coords[i] = p_b + local_vec

        return coords
