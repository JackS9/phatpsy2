# phatpsy/supercell.py
"""
Supercell and Periodic Surface Slab Generator for PHATPSY 2.0.
Supports 3D crystal lattice definitions, translational supercell replication,
oriented surface slab generation along Miller indices (h, k, l), vacuum buffer insertion,
substrate layer freezing (fixed=True / QFIXED=T), virtual coordination perimeter completion
(virtual=True / QVIRTL=T), and high-symmetry adsorbate placement (top, bridge, hollow).
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union, Sequence
import math
import copy
import numpy as np

from phatpsy.schema import CalculationDeck, AtomConfig, ControlConfig
from phatpsy.spin import BOHR_PER_ANGSTROM, get_covalent_radius_bohr

# Standard experimental lattice constants (in Angstroms) for common elements
STANDARD_LATTICE_CONSTANTS: Dict[str, Dict[str, Any]] = {
    "NI": {"structure": "fcc", "a": 3.524},
    "AG": {"structure": "fcc", "a": 4.085},
    "CU": {"structure": "fcc", "a": 3.615},
    "AU": {"structure": "fcc", "a": 4.078},
    "PT": {"structure": "fcc", "a": 3.924},
    "PD": {"structure": "fcc", "a": 3.891},
    "AL": {"structure": "fcc", "a": 4.050},
    "PB": {"structure": "fcc", "a": 4.950},
    "FE": {"structure": "bcc", "a": 2.866},
    "W":  {"structure": "bcc", "a": 3.165},
    "MO": {"structure": "bcc", "a": 3.147},
    "CR": {"structure": "bcc", "a": 2.884},
    "TA": {"structure": "bcc", "a": 3.301},
    "NA": {"structure": "bcc", "a": 4.290},
    "K":  {"structure": "bcc", "a": 5.328},
    "C":  {"structure": "diamond", "a": 3.567},
    "SI": {"structure": "diamond", "a": 5.431},
    "GE": {"structure": "diamond", "a": 5.658},
}


@dataclass
class Lattice:
    """
    3D Crystal Lattice defined by three basis vectors a, b, c in Bohr.
    matrix is of shape (3, 3) where rows are [a, b, c].
    """
    matrix: np.ndarray  # Shape (3, 3) in Bohr

    def __post_init__(self):
        self.matrix = np.array(self.matrix, dtype=np.float64)
        if self.matrix.shape != (3, 3):
            raise ValueError(f"Lattice matrix must be (3, 3), got {self.matrix.shape}")

    @property
    def a_vec(self) -> np.ndarray:
        return self.matrix[0]

    @property
    def b_vec(self) -> np.ndarray:
        return self.matrix[1]

    @property
    def c_vec(self) -> np.ndarray:
        return self.matrix[2]

    @property
    def a(self) -> float:
        """Length of lattice vector a in Bohr."""
        return float(np.linalg.norm(self.a_vec))

    @property
    def b(self) -> float:
        """Length of lattice vector b in Bohr."""
        return float(np.linalg.norm(self.b_vec))

    @property
    def c(self) -> float:
        """Length of lattice vector c in Bohr."""
        return float(np.linalg.norm(self.c_vec))

    @property
    def alpha(self) -> float:
        """Angle between b and c in degrees."""
        dot = np.dot(self.b_vec, self.c_vec) / (self.b * self.c)
        return float(math.degrees(math.acos(np.clip(dot, -1.0, 1.0))))

    @property
    def beta(self) -> float:
        """Angle between a and c in degrees."""
        dot = np.dot(self.a_vec, self.c_vec) / (self.a * self.c)
        return float(math.degrees(math.acos(np.clip(dot, -1.0, 1.0))))

    @property
    def gamma(self) -> float:
        """Angle between a and b in degrees."""
        dot = np.dot(self.a_vec, self.b_vec) / (self.a * self.b)
        return float(math.degrees(math.acos(np.clip(dot, -1.0, 1.0))))

    @property
    def volume(self) -> float:
        """Cell volume in Bohr^3."""
        return float(abs(np.dot(self.a_vec, np.cross(self.b_vec, self.c_vec))))

    def fractional_to_cartesian(self, frac_coords: np.ndarray) -> np.ndarray:
        """Convert fractional coordinates (u, v, w) into Cartesian coordinates (x, y, z) in Bohr."""
        frac = np.asarray(frac_coords, dtype=np.float64)
        return np.dot(frac, self.matrix)

    def cartesian_to_fractional(self, cart_coords: np.ndarray) -> np.ndarray:
        """Convert Cartesian coordinates (x, y, z) in Bohr into fractional coordinates (u, v, w)."""
        cart = np.asarray(cart_coords, dtype=np.float64)
        inv_mat = np.linalg.inv(self.matrix)
        return np.dot(cart, inv_mat)

    @property
    def reciprocal_lattice(self) -> "Lattice":
        """Computes reciprocal lattice (2*pi * (b x c) / V, etc.)."""
        v = self.volume
        b1 = 2.0 * math.pi * np.cross(self.b_vec, self.c_vec) / v
        b2 = 2.0 * math.pi * np.cross(self.c_vec, self.a_vec) / v
        b3 = 2.0 * math.pi * np.cross(self.a_vec, self.b_vec) / v
        return Lattice(matrix=np.array([b1, b2, b3]))

    @classmethod
    def from_parameters(
        cls,
        a: float,
        b: float,
        c: float,
        alpha: float = 90.0,
        beta: float = 90.0,
        gamma: float = 90.0,
        units: str = "angstrom",
    ) -> "Lattice":
        """
        Builds a Lattice from parameters (a, b, c, alpha, beta, gamma).
        Angles are in degrees.
        """
        scale = BOHR_PER_ANGSTROM if units.lower() in ("angstrom", "angstroms", "aa") else 1.0
        a_b = a * scale
        b_b = b * scale
        c_b = c * scale

        a_rad = math.radians(alpha)
        b_rad = math.radians(beta)
        g_rad = math.radians(gamma)

        # a along x-axis
        v_a = np.array([a_b, 0.0, 0.0])
        # b in xy-plane
        v_b = np.array([b_b * math.cos(g_rad), b_b * math.sin(g_rad), 0.0])
        # c pointing into z
        cx = c_b * math.cos(b_rad)
        cy = c_b * (math.cos(a_rad) - math.cos(b_rad) * math.cos(g_rad)) / math.sin(g_rad)
        cz_sq = c_b * c_b - cx * cx - cy * cy
        cz = math.sqrt(max(cz_sq, 0.0))
        v_c = np.array([cx, cy, cz])

        return cls(matrix=np.array([v_a, v_b, v_c]))

    @classmethod
    def cubic(cls, a: float, units: str = "angstrom") -> "Lattice":
        return cls.from_parameters(a, a, a, 90.0, 90.0, 90.0, units=units)

    @classmethod
    def fcc(cls, a: float, units: str = "angstrom", primitive: bool = False) -> "Lattice":
        """FCC lattice (conventional cubic or primitive rhombohedral)."""
        scale = BOHR_PER_ANGSTROM if units.lower() in ("angstrom", "angstroms", "aa") else 1.0
        a_b = a * scale
        if not primitive:
            return cls.cubic(a, units=units)
        # Primitive FCC vectors
        v_a = 0.5 * a_b * np.array([0.0, 1.0, 1.0])
        v_b = 0.5 * a_b * np.array([1.0, 0.0, 1.0])
        v_c = 0.5 * a_b * np.array([1.0, 1.0, 0.0])
        return cls(matrix=np.array([v_a, v_b, v_c]))

    @classmethod
    def bcc(cls, a: float, units: str = "angstrom", primitive: bool = False) -> "Lattice":
        """BCC lattice (conventional cubic or primitive)."""
        scale = BOHR_PER_ANGSTROM if units.lower() in ("angstrom", "angstroms", "aa") else 1.0
        a_b = a * scale
        if not primitive:
            return cls.cubic(a, units=units)
        # Primitive BCC vectors
        v_a = 0.5 * a_b * np.array([-1.0, 1.0, 1.0])
        v_b = 0.5 * a_b * np.array([1.0, -1.0, 1.0])
        v_c = 0.5 * a_b * np.array([1.0, 1.0, -1.0])
        return cls(matrix=np.array([v_a, v_b, v_c]))

    @classmethod
    def tetragonal(cls, a: float, c: float, units: str = "angstrom") -> "Lattice":
        return cls.from_parameters(a, a, c, 90.0, 90.0, 90.0, units=units)

    @classmethod
    def orthorhombic(cls, a: float, b: float, c: float, units: str = "angstrom") -> "Lattice":
        return cls.from_parameters(a, b, c, 90.0, 90.0, 90.0, units=units)

    @classmethod
    def hexagonal(cls, a: float, c: float, units: str = "angstrom") -> "Lattice":
        return cls.from_parameters(a, a, c, 90.0, 90.0, 120.0, units=units)


@dataclass
class PeriodicAtom:
    """An atom within a periodic unit cell or supercell."""
    element: str
    frac_coords: np.ndarray  # (u, v, w) in [0, 1)
    cart_coords: np.ndarray  # (x, y, z) in Bohr
    fixed: bool = False      # QFIXED=T in PHATPSY
    virtual: bool = False    # QVIRTL=T in PHATPSY
    spin: Optional[str] = None  # 'up', 'down', or None
    layer: Optional[int] = None # Atomic layer index in slab
    tag: Optional[str] = None   # Custom label (e.g. 'substrate', 'adsorbate', 'buffer')
    grid_indices: Tuple[int, int, int] = (0, 0, 0) # Supercell translation indices (ix, iy, iz)

    def to_atom_config(self) -> AtomConfig:
        return AtomConfig(
            element=self.element,
            xyz=[float(c) for c in self.cart_coords],
            fixed=self.fixed,
            virtual=self.virtual,
            spin=self.spin,
        )


@dataclass
class PeriodicCell:
    """A periodic unit cell or supercell with crystal lattice and basis atoms."""
    lattice: Lattice
    atoms: List[PeriodicAtom] = field(default_factory=list)
    title: str = "Periodic System"

    @property
    def formula(self) -> str:
        """Returns empirical chemical formula string (e.g. 'Ni4', 'Fe2', 'C8')."""
        from collections import Counter
        counts = Counter(a.element for a in self.atoms if not a.virtual)
        parts = []
        for elem in sorted(counts.keys()):
            count = counts[elem]
            parts.append(f"{elem}{count if count > 1 else ''}")
        return "".join(parts) if parts else "Empty"

    @classmethod
    def bulk(
        cls,
        element: str,
        crystal_structure: Optional[str] = None,
        a: Optional[float] = None,
        c: Optional[float] = None,
        units: str = "angstrom",
    ) -> "PeriodicCell":
        """
        Creates a conventional bulk unit cell for a standard chemical element.
        """
        elem = element.strip().upper()
        info = STANDARD_LATTICE_CONSTANTS.get(elem, {"structure": "fcc", "a": 4.0})

        struct = (crystal_structure or info.get("structure", "fcc")).lower()
        a_val = a if a is not None else info.get("a", 4.0)

        if struct == "fcc":
            # Conventional FCC (4 atoms per unit cell)
            lat = Lattice.cubic(a_val, units=units)
            basis_frac = [
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.5, 0.5]),
                np.array([0.5, 0.0, 0.5]),
                np.array([0.5, 0.5, 0.0]),
            ]
        elif struct == "bcc":
            # Conventional BCC (2 atoms per unit cell)
            lat = Lattice.cubic(a_val, units=units)
            basis_frac = [
                np.array([0.0, 0.0, 0.0]),
                np.array([0.5, 0.5, 0.5]),
            ]
        elif struct == "sc":
            # Simple cubic (1 atom per unit cell)
            lat = Lattice.cubic(a_val, units=units)
            basis_frac = [np.array([0.0, 0.0, 0.0])]
        elif struct == "diamond":
            # Diamond cubic (8 atoms per unit cell)
            lat = Lattice.cubic(a_val, units=units)
            fcc_pts = [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ]
            basis_frac = [np.array(p) for p in fcc_pts] + [np.array(p) + 0.25 for p in fcc_pts]
        else:
            raise ValueError(f"Unsupported bulk crystal structure: '{struct}'")

        atoms = []
        for frac in basis_frac:
            cart = lat.fractional_to_cartesian(frac)
            atoms.append(PeriodicAtom(
                element=elem.capitalize(),
                frac_coords=frac,
                cart_coords=cart,
                layer=0,
                tag="bulk",
            ))

        return cls(lattice=lat, atoms=atoms, title=f"Bulk {elem} ({struct.upper()})")

    def build_supercell(
        self,
        repeat: Tuple[int, int, int] = (1, 1, 1),
        auto_spin: bool = False,
        center: bool = False,
    ) -> "PeriodicCell":
        """
        Generates an nx x ny x nz translational supercell.
        """
        nx, ny, nz = repeat
        if nx < 1 or ny < 1 or nz < 1:
            raise ValueError(f"Repeat factors must be >= 1, got {repeat}")

        # Scaled lattice vectors
        new_mat = np.array([
            nx * self.lattice.a_vec,
            ny * self.lattice.b_vec,
            nz * self.lattice.c_vec,
        ])
        super_lattice = Lattice(matrix=new_mat)

        new_atoms: List[PeriodicAtom] = []
        atom_count = 0

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    shift_frac = np.array([ix / nx, iy / ny, iz / nz], dtype=np.float64)
                    shift_cart = ix * self.lattice.a_vec + iy * self.lattice.b_vec + iz * self.lattice.c_vec

                    for base_a in self.atoms:
                        atom_frac = (base_a.frac_coords + np.array([ix, iy, iz])) / np.array([nx, ny, nz])
                        atom_cart = base_a.cart_coords + shift_cart

                        # Bipartite alternating spin logic
                        atom_spin = base_a.spin
                        if auto_spin:
                            is_even = (ix + iy + iz) % 2 == 0
                            atom_spin = "up" if is_even else "down"

                        new_atoms.append(PeriodicAtom(
                            element=base_a.element,
                            frac_coords=atom_frac,
                            cart_coords=atom_cart,
                            fixed=base_a.fixed,
                            virtual=base_a.virtual,
                            spin=atom_spin,
                            layer=base_a.layer,
                            tag=base_a.tag,
                            grid_indices=(ix, iy, iz),
                        ))
                        atom_count += 1

        if center:
            # Shift centroid to origin in x and y
            all_cart = np.array([a.cart_coords for a in new_atoms])
            mean_xy = np.mean(all_cart[:, :2], axis=0)
            for a in new_atoms:
                a.cart_coords[:2] -= mean_xy
                a.frac_coords = super_lattice.cartesian_to_fractional(a.cart_coords)

        return PeriodicCell(
            lattice=super_lattice,
            atoms=new_atoms,
            title=f"{self.title} ({nx}x{ny}x{nz} supercell)",
        )

    def to_deck(self, basis: str = "SZ", title: Optional[str] = None) -> CalculationDeck:
        """Converts the periodic cell atoms to a PHATPSY CalculationDeck."""
        atom_configs = [a.to_atom_config() for a in self.atoms]
        c_title = title or self.title
        deck = CalculationDeck(
            title=c_title,
            control=ControlConfig(),
            geometry=atom_configs,
        )
        deck.resolve_elements()
        return deck

    def to_xyz(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """Exports cell geometry to standard XYZ format in Angstroms."""
        lines = [
            f"{len(self.atoms)}",
            f"{self.title} | Lattice=\"{' '.join(f'{x:.6f}' for x in (self.lattice.matrix / BOHR_PER_ANGSTROM).reshape(-1))}\"",
        ]
        for a in self.atoms:
            x, y, z = a.cart_coords / BOHR_PER_ANGSTROM
            lines.append(f"{a.element:<3s} {x:12.6f} {y:12.6f} {z:12.6f}")
        text = "\n".join(lines) + "\n"
        if filepath:
            p = Path(filepath)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        return text

    def to_poscar(self, filepath: Optional[Union[str, Path]] = None, comment: str = "") -> str:
        """Exports cell geometry to VASP POSCAR format."""
        c_mat_aa = self.lattice.matrix / BOHR_PER_ANGSTROM
        elem_order: List[str] = []
        elem_counts: Dict[str, int] = {}
        for a in self.atoms:
            if a.element not in elem_counts:
                elem_order.append(a.element)
                elem_counts[a.element] = 0
            elem_counts[a.element] += 1

        lines = [
            comment or self.title,
            "1.0",
            f"  {c_mat_aa[0,0]:16.8f} {c_mat_aa[0,1]:16.8f} {c_mat_aa[0,2]:16.8f}",
            f"  {c_mat_aa[1,0]:16.8f} {c_mat_aa[1,1]:16.8f} {c_mat_aa[1,2]:16.8f}",
            f"  {c_mat_aa[2,0]:16.8f} {c_mat_aa[2,1]:16.8f} {c_mat_aa[2,2]:16.8f}",
            "  " + "  ".join(elem_order),
            "  " + "  ".join(str(elem_counts[e]) for e in elem_order),
            "Direct",
        ]
        for elem in elem_order:
            for a in self.atoms:
                if a.element == elem:
                    u, v, w = a.frac_coords
                    lines.append(f"  {u:16.8f} {v:16.8f} {w:16.8f}")

        text = "\n".join(lines) + "\n"
        if filepath:
            p = Path(filepath)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        return text

    def to_cif(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """Exports cell geometry to standard Crystallographic Information File (CIF)."""
        lat = self.lattice
        lines = [
            f"data_{self.title.replace(' ', '_')}",
            "_audit_creation_method  'PHATPSY 2.0 Supercell Generator'",
            f"_cell_length_a    {lat.a / BOHR_PER_ANGSTROM:.6f}",
            f"_cell_length_b    {lat.b / BOHR_PER_ANGSTROM:.6f}",
            f"_cell_length_c    {lat.c / BOHR_PER_ANGSTROM:.6f}",
            f"_cell_angle_alpha {lat.alpha:.4f}",
            f"_cell_angle_beta  {lat.beta:.4f}",
            f"_cell_angle_gamma {lat.gamma:.4f}",
            "_symmetry_space_group_name_H-M    'P 1'",
            "_symmetry_Int_Tables_number       1",
            "",
            "loop_",
            "  _atom_site_label",
            "  _atom_site_type_symbol",
            "  _atom_site_fract_x",
            "  _atom_site_fract_y",
            "  _atom_site_fract_z",
        ]
        for idx, a in enumerate(self.atoms, 1):
            lbl = f"{a.element}{idx}"
            u, v, w = a.frac_coords
            lines.append(f"  {lbl:<6s} {a.element:<3s} {u:12.6f} {v:12.6f} {w:12.6f}")

        text = "\n".join(lines) + "\n"
        if filepath:
            p = Path(filepath)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        return text


def build_slab(
    element: str = "Ni",
    crystal_structure: str = "fcc",
    a: Optional[float] = None,
    miller_index: Tuple[int, int, int] = (1, 0, 0),
    layers: int = 3,
    vacuum: float = 12.0,  # Vacuum gap in Angstroms
    repeat: Tuple[int, int] = (2, 2),  # In-plane lateral supercell
    fixed_layers: int = 1, # Number of bottom substrate layers frozen
    add_virtual_perimeter: bool = False,
    units: str = "angstrom",
) -> PeriodicCell:
    """
    Constructs an oriented periodic surface slab with specified Miller indices,
    layers, vacuum padding, and substrate layer freezing.

    Args:
        element: Chemical element symbol (e.g. 'Ni', 'Ag', 'Cu', 'Fe').
        crystal_structure: Bulk structure ('fcc', 'bcc').
        a: Lattice parameter (defaults to standard experimental constant).
        miller_index: Surface facet Miller indices: (1, 0, 0), (1, 1, 1), or (1, 1, 0).
        layers: Total number of atomic layers.
        vacuum: Vacuum space above the slab in Angstroms.
        repeat: (nx, ny) in-plane replication factors.
        fixed_layers: Number of bottom substrate layers tagged with fixed=True (QFIXED=T).
        add_virtual_perimeter: Whether to append virtual coordination buffer atoms (QVIRTL=T).
        units: 'angstrom' or 'bohr'.

    Returns:
        PeriodicCell containing the slab and surface environment.
    """
    if isinstance(crystal_structure, (tuple, list)):
        miller_index = tuple(crystal_structure)
        crystal_structure = None

    elem = element.strip().upper()
    info = STANDARD_LATTICE_CONSTANTS.get(elem, {"structure": "fcc", "a": 3.524})
    struct = (crystal_structure or info.get("structure", "fcc")).lower()
    a_val = a if a is not None else info.get("a", 3.524)

    scale = BOHR_PER_ANGSTROM if units.lower() in ("angstrom", "angstroms", "aa") else 1.0
    a_bohr = a_val * scale
    vac_bohr = vacuum * scale

    h, k, l = miller_index

    # 1. Generate basis for the requested facet
    if (h, k, l) == (1, 0, 0):
        if struct == "fcc":
            # FCC(100): square surface net with in-plane spacing a/sqrt(2)
            d_xy = a_bohr / math.sqrt(2.0)
            d_z = 0.5 * a_bohr  # Interlayer spacing d_100 = a / 2
            v_a = np.array([d_xy, 0.0, 0.0])
            v_b = np.array([0.0, d_xy, 0.0])

            # Layer offsets in fractional coordinates: alternating (0, 0) and (0.5, 0.5)
            layer_offsets = [np.array([0.0, 0.0]) if i % 2 == 0 else np.array([0.5, 0.5]) for i in range(layers)]
        elif struct == "bcc":
            # BCC(100): square surface net with in-plane spacing a
            d_xy = a_bohr
            d_z = 0.5 * a_bohr
            v_a = np.array([d_xy, 0.0, 0.0])
            v_b = np.array([0.0, d_xy, 0.0])
            layer_offsets = [np.array([0.0, 0.0]) if i % 2 == 0 else np.array([0.5, 0.5]) for i in range(layers)]
        else:
            raise ValueError(f"Unsupported structure {struct} for (100) slab")
    elif (h, k, l) == (1, 1, 1):
        if struct == "fcc":
            # FCC(111): hexagonal close-packed surface net
            d_nn = a_bohr / math.sqrt(2.0)  # Nearest-neighbor spacing
            d_z = a_bohr / math.sqrt(3.0)   # Interlayer spacing d_111 = a / sqrt(3)
            v_a = np.array([d_nn, 0.0, 0.0])
            v_b = np.array([0.5 * d_nn, 0.5 * math.sqrt(3.0) * d_nn, 0.0])

            # ABC stacking
            layer_offsets = []
            for i in range(layers):
                stack = i % 3
                if stack == 0:
                    layer_offsets.append(np.array([0.0, 0.0]))
                elif stack == 1:
                    layer_offsets.append(np.array([1.0 / 3.0, 1.0 / 3.0]))
                else:
                    layer_offsets.append(np.array([2.0 / 3.0, 2.0 / 3.0]))
        else:
            raise ValueError(f"Unsupported structure {struct} for (111) slab")
    elif (h, k, l) == (1, 1, 0):
        if struct == "fcc":
            # FCC(110): rectangular surface net
            dx = a_bohr / math.sqrt(2.0)
            dy = a_bohr
            d_z = a_bohr / (2.0 * math.sqrt(2.0))
            v_a = np.array([dx, 0.0, 0.0])
            v_b = np.array([0.0, dy, 0.0])
            layer_offsets = [np.array([0.0, 0.0]) if i % 2 == 0 else np.array([0.5, 0.5]) for i in range(layers)]
        else:
            raise ValueError(f"Unsupported structure {struct} for (110) slab")
    else:
        raise ValueError(f"Unsupported Miller index ({h}, {k}, {l}); supported: (1,0,0), (1,1,1), (1,1,0)")

    # Slab thickness and total z-dimension
    slab_thickness = (layers - 1) * d_z
    total_z = slab_thickness + vac_bohr
    v_c = np.array([0.0, 0.0, total_z])

    unit_slab_lattice = Lattice(matrix=np.array([v_a, v_b, v_c]))

    # Unit in-plane cell atoms
    unit_atoms: List[PeriodicAtom] = []
    for layer_idx, offset in enumerate(layer_offsets):
        z_pos = layer_idx * d_z
        frac = np.array([offset[0], offset[1], z_pos / total_z])
        cart = np.array([
            offset[0] * v_a[0] + offset[1] * v_b[0],
            offset[0] * v_a[1] + offset[1] * v_b[1],
            z_pos
        ])
        is_fixed = layer_idx < fixed_layers
        unit_atoms.append(PeriodicAtom(
            element=elem.capitalize(),
            frac_coords=frac,
            cart_coords=cart,
            fixed=is_fixed,
            layer=layer_idx + 1,
            tag=f"layer_{layer_idx+1}",
        ))

    unit_slab = PeriodicCell(
        lattice=unit_slab_lattice,
        atoms=unit_atoms,
        title=f"{elem.capitalize()}({h}{k}{l}) {layers}-layer slab",
    )

    # 2. Replicate in-plane
    nx, ny = repeat
    slab_supercell = unit_slab.build_supercell((nx, ny, 1), auto_spin=False, center=True)

    # 3. Apply Localized Valence-Bond (LVB) checkerboard alternating spin on top surface layer
    for a_atom in slab_supercell.atoms:
        if a_atom.layer == layers:  # Surface layer
            # In-plane grid parity determines local spin polarization
            gx, gy = a_atom.grid_indices[0], a_atom.grid_indices[1]
            a_atom.spin = "up" if (gx + gy) % 2 == 0 else "down"

    # 4. Optional: add virtual coordination perimeter atoms
    if add_virtual_perimeter:
        virt_atoms = []
        r_cov = get_covalent_radius_bohr(elem)
        # Add perimeter ghost shell around lateral boundaries
        for edge_a in slab_supercell.atoms:
            if not edge_a.fixed:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    v_cart = edge_a.cart_coords.copy()
                    v_cart[0] += dx * (2.2 * r_cov)
                    v_cart[1] += dy * (2.2 * r_cov)
                    v_frac = slab_supercell.lattice.cartesian_to_fractional(v_cart)
                    virt_atoms.append(PeriodicAtom(
                        element=elem.capitalize(),
                        frac_coords=v_frac,
                        cart_coords=v_cart,
                        virtual=True,
                        fixed=True,
                        tag="ghost_perimeter",
                    ))
        slab_supercell.atoms.extend(virt_atoms)

    return slab_supercell


def attach_adsorbate(
    slab: PeriodicCell,
    adsorbate: Union[str, CalculationDeck, List[AtomConfig]],
    site_type: str = "top",
    site_index: int = 0,
    height: float = 2.0,  # Height above surface in Angstroms
    tilt_deg: float = 0.0,
    units: str = "angstrom",
) -> PeriodicCell:
    """
    Attaches an adsorbate molecule onto the top surface layer of a periodic slab.

    Args:
        slab: PeriodicCell representing the substrate surface.
        adsorbate: Chemical formula (e.g. 'CO', 'NO', 'H2'), XYZ text/filepath, or CalculationDeck.
        site_type: 'top' (on-top of surface atom), 'bridge' (between two atoms), or 'hollow' (center of surface polygon).
        site_index: Index of target surface site.
        height: Vertical clearance z above top atomic plane in Angstroms or Bohr.
        tilt_deg: Optional molecular tilt angle in degrees.
        units: 'angstrom' or 'bohr'.

    Returns:
        New PeriodicCell containing combined substrate + adsorbate.
    """
    new_slab = copy.deepcopy(slab)
    scale = BOHR_PER_ANGSTROM if units.lower() in ("angstrom", "angstroms", "aa") else 1.0
    h_bohr = height * scale

    # Find surface layer atoms (highest z coordinates)
    real_atoms = [a for a in new_slab.atoms if not a.virtual]
    if not real_atoms:
        raise ValueError("Slab has no real physical atoms.")

    max_z = max(a.cart_coords[2] for a in real_atoms)
    surface_atoms = [a for a in real_atoms if abs(a.cart_coords[2] - max_z) < 0.2]

    # Determine site XY position
    if site_type.lower() == "top":
        target_atom = surface_atoms[site_index % len(surface_atoms)]
        site_xy = target_atom.cart_coords[:2]
    elif site_type.lower() == "bridge":
        if len(surface_atoms) < 2:
            site_xy = surface_atoms[0].cart_coords[:2]
        else:
            p1 = surface_atoms[site_index % len(surface_atoms)].cart_coords[:2]
            p2 = surface_atoms[(site_index + 1) % len(surface_atoms)].cart_coords[:2]
            site_xy = 0.5 * (p1 + p2)
    elif site_type.lower() in ("hollow", "fcc_hollow", "fourfold"):
        # Centroid of top surface atoms
        site_xy = np.mean([a.cart_coords[:2] for a in surface_atoms[:4]], axis=0)
    else:
        raise ValueError(f"Unknown adsorption site type: '{site_type}'. Use 'top', 'bridge', or 'hollow'.")

    anchor_z = max_z + h_bohr

    # Parse adsorbate atoms
    if isinstance(adsorbate, str):
        adj_str = adsorbate.strip().upper()
        if adj_str == "CO":
            # Classic C-O perpendicular (C bonded to surface, O on top)
            # Experimental bond length R(C-O) = 1.128 A = 2.132 Bohr
            r_co = 1.128 * scale
            ads_atoms = [
                AtomConfig(element="C", xyz=[0.0, 0.0, 0.0]),
                AtomConfig(element="O", xyz=[0.0, 0.0, r_co]),
            ]
        elif adj_str == "NO":
            r_no = 1.154 * scale
            ads_atoms = [
                AtomConfig(element="N", xyz=[0.0, 0.0, 0.0]),
                AtomConfig(element="O", xyz=[0.0, 0.0, r_no]),
            ]
        elif adj_str in ("H2", "H"):
            ads_atoms = [AtomConfig(element="H", xyz=[0.0, 0.0, 0.0])]
        elif adj_str == "O":
            ads_atoms = [AtomConfig(element="O", xyz=[0.0, 0.0, 0.0])]
        else:
            # Try parsing as XYZ text/file
            from phatpsy.builder import parse_geometry
            ads_atoms = parse_geometry(adsorbate, units=units)
    elif hasattr(adsorbate, "atoms"):
        ads_atoms = copy.deepcopy(adsorbate.atoms)
    elif isinstance(adsorbate, list):
        ads_atoms = copy.deepcopy(adsorbate)
    else:
        raise ValueError(f"Unsupported adsorbate input: {adsorbate}")

    # Position adsorbate relative to anchor
    # Anchor atom (index 0) placed at (site_x, site_y, anchor_z)
    anchor_orig = np.array(ads_atoms[0].xyz, dtype=np.float64)

    # Optional tilt around y-axis
    rad_tilt = math.radians(tilt_deg)
    cos_t, sin_t = math.cos(rad_tilt), math.sin(rad_tilt)
    rot_mat = np.array([
        [cos_t, 0.0, sin_t],
        [0.0, 1.0, 0.0],
        [-sin_t, 0.0, cos_t]
    ])

    for a in ads_atoms:
        local_rel = np.array(a.xyz, dtype=np.float64) - anchor_orig
        tilted = np.dot(rot_mat, local_rel)
        new_pos = np.array([site_xy[0] + tilted[0], site_xy[1] + tilted[1], anchor_z + tilted[2]])
        frac = new_slab.lattice.cartesian_to_fractional(new_pos)

        new_slab.atoms.append(PeriodicAtom(
            element=a.element,
            frac_coords=frac,
            cart_coords=new_pos,
            fixed=False,
            virtual=False,
            tag="adsorbate",
        ))

    new_slab.title = f"{new_slab.title} + {adsorbate if isinstance(adsorbate, str) else 'adsorbate'} ({site_type})"
    return new_slab
