# phatpsy/schema.py
"""
Modern YAML and JSON input schemas and serializers for PHATPSY calculations.
Provides two-way conversion between high-level Python/YAML/JSON representations
and legacy Fortran stdin decks.
"""
from dataclasses import dataclass, field, asdict, replace
from typing import List, Dict, Tuple, Optional, Any, Union, Sequence
from pathlib import Path
import yaml
import json
import re
import math
from phatpsy.basis import get_element_basis, BasisFunction, ElementData

BOHR_PER_ANGSTROM: float = 1.8897261246257702


def normalize_symmetry_axis(axis: Union[int, str, None]) -> Optional[int]:
    """
    Normalizes a symmetry axis identifier into the legacy Fortran mnemonic:
    1 = Z axis (or XY plane reflection, normal Z)
    10 = Y axis (or XZ plane reflection, normal Y)
    100 = X axis (or YZ plane reflection, normal X)
    """
    if axis is None:
        return None
    if isinstance(axis, int):
        return axis
    s = str(axis).strip().lower()
    if s in ("1", "z", "axis_z", "001", "xy"):
        return 1
    if s in ("10", "y", "axis_y", "010", "xz"):
        return 10
    if s in ("100", "x", "axis_x", "100", "yz"):
        return 100
    try:
        return int(s)
    except ValueError:
        return 1


def normalize_symmetry_op(op: Union[int, str, None]) -> Optional[int]:
    """
    Normalizes a symmetry operation identifier into the legacy Fortran convention:
    -1 = Pure reflection (mirror plane)
    -2 = Inversion center
    1 or 0 = Identity
    N (>0) = N-fold proper rotation (C_n)
    -N (< -2) = N-fold improper rotation / roto-reflection (S_n)
    """
    if op is None:
        return None
    if isinstance(op, int):
        return op
    s = str(op).strip().lower()
    if s in ("reflection", "mirror", "refl", "sigma", "plane"):
        return -1
    if s in ("inversion", "inv", "i", "center"):
        return -2
    if s in ("identity", "e", "none"):
        return 1
    # Match C2, C3, C4, etc.
    m = re.match(r"^c?(\d+)$", s)
    if m:
        return int(m.group(1))
    # Match S3, S4, etc.
    m = re.match(r"^s(\d+)$", s)
    if m:
        return -int(m.group(1))
    try:
        return int(s)
    except ValueError:
        return -1


def apply_symmetry_operation(
    xyz: List[float],
    isym: Union[int, str] = -1,
    ixyz: Union[int, str] = 1,
) -> List[float]:
    """
    Transforms Cartesian coordinates [x, y, z] by a symmetry operation.
    Corresponds to legacy Fortran SYMXYZ logic:
    - isym = 1 or 0: Identity
    - isym = -1: Pure reflection across the plane perpendicular to ixyz (1=Z, 10=Y, 100=X)
    - isym = -2: Inversion through the origin
    - isym = N (>0): N-fold rotation around axis ixyz (1=Z, 10=Y, 100=X)
    - isym = -N (< -2): N-fold roto-reflection
    """
    isym_norm = normalize_symmetry_op(isym) or 1
    ixyz_norm = normalize_symmetry_axis(ixyz) or 1

    if isym_norm in (0, 1):
        return [float(c) for c in xyz]

    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])

    if isym_norm == -2:
        return [-x, -y, -z]

    if isym_norm == -1:
        # Reflection through plane with normal along axis:
        # ixyz == 100 (X normal, YZ plane): x -> -x
        # ixyz == 10 (Y normal, XZ plane): y -> -y
        # ixyz == 1 (Z normal, XY plane): z -> -z
        if ixyz_norm == 100:
            return [-x, y, z]
        elif ixyz_norm == 10:
            return [x, -y, z]
        else:  # ixyz == 1 or default Z
            return [x, y, -z]

    # N-fold rotation or roto-reflection
    nfold = abs(isym_norm)
    angle = 2.0 * math.pi / float(nfold)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    if ixyz_norm == 100:
        # Rotation around X axis
        xr = x
        yr = cos_a * y - sin_a * z
        zr = sin_a * y + cos_a * z
    elif ixyz_norm == 10:
        # Rotation around Y axis
        xr = cos_a * x + sin_a * z
        yr = y
        zr = -sin_a * x + cos_a * z
    else:  # ixyz == 1 or default Z
        # Rotation around Z axis
        xr = cos_a * x - sin_a * y
        yr = sin_a * x + cos_a * y
        zr = z

    if isym_norm < -2:
        # Roto-reflection: rotate then reflect through plane perpendicular to axis
        if ixyz_norm == 100:
            xr = -xr
        elif ixyz_norm == 10:
            yr = -yr
        else:
            zr = -zr

    return [0.0 if abs(val) < 1e-12 else val for val in [xr, yr, zr]]


RETIRED_YAML_CONTROL_PARAMS: set = {
    "time_limit",
    "charge_damp",
    "charge_accel",
    "potential_damp",
    "potential_accel",
    "pk_scale",
    "use_pseudometric",
    "pk_energy",
    "use_constrained_fit",
    "use_diis",
    "use_dynamic_exponents",
}


@dataclass
class ControlConfig:
    spin_unrestricted: bool = True
    max_cycles: int = 200
    convergence: float = 1.0e-6
    time_limit: float = 360.0
    charge_damp: float = 0.5
    charge_accel: float = 1.0
    potential_damp: float = 0.5
    potential_accel: float = 1.0
    pk_scale: float = 1.0
    use_pseudometric: bool = True
    pk_energy: float = 0.0
    use_constrained_fit: bool = False
    use_grid_fit: bool = True  # Fit each atom's Coulomb potential on a radial grid as a convex mixture of screening exponentials (see model_potentials).
    atomic_smearing: float = 0.0  # Fermi-Dirac width in Hartree for atomic orbital occupations; 0 disables smearing.
    use_diis: bool = False
    use_global_fermi: bool = False
    fermi_temperature: Optional[float] = None
    use_dynamic_exponents: bool = False
    exchange_kernel: str = "hartree-fock"
    exchange_alpha: float = 2.0 / 3.0
    correlation_kernel: Optional[str] = None
    use_dispersion: bool = False
    dispersion_model: str = "d3bj"
    default_basis: Optional[str] = None
    separate_atoms_first_cycle: bool = True
    debug: bool = False
    trace: bool = False
    extra_namelist: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.atomic_smearing < 0.0:
            raise ValueError(
                f"atomic_smearing must be >= 0 (Fermi-Dirac width in Hartree; 0 disables), "
                f"got {self.atomic_smearing!r}"
            )

    def to_dict(self, include_retired: bool = False) -> Dict[str, Any]:
        d = asdict(self)
        if not self.extra_namelist:
            del d["extra_namelist"]
        if self.default_basis is None:
            del d["default_basis"]
        if not include_retired:
            for k in RETIRED_YAML_CONTROL_PARAMS:
                d.pop(k, None)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ControlConfig":
        valid_keys = {
            "spin_unrestricted", "max_cycles", "convergence", "time_limit",
            "charge_damp", "charge_accel", "potential_damp", "potential_accel",
            "pk_scale", "use_pseudometric", "pk_energy", "use_constrained_fit",
            "use_grid_fit", "atomic_smearing",
            "use_diis", "use_global_fermi", "fermi_temperature",
            "use_dynamic_exponents", "exchange_kernel", "exchange_alpha",
            "correlation_kernel", "use_dispersion", "dispersion_model",
            "default_basis", "separate_atoms_first_cycle", "debug", "trace"
        }
        kwargs = {k: v for k, v in data.items() if k in valid_keys}
        extra = {k: v for k, v in data.items() if k not in valid_keys}
        return cls(**kwargs, extra_namelist=extra)


@dataclass
class SymmetrySpec:
    ixyz: int
    isym: int = -1
    qflip: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Periodic table atomic numbers (Z = 1..118), cross-checked against phatpsy.formats.PERIODIC_TABLE
ATOMIC_NUMBERS: Dict[str, int] = {
    "H": 1, "He": 2,
    "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48,
    "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
    "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
    "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100, "Md": 101, "No": 102, "Lr": 103,
    "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112,
    "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118,
}

NOBLE_CORES: Tuple[int, ...] = (0, 2, 10, 18, 36, 54, 86)
NOBLE_NEXT: Tuple[int, ...] = (2, 10, 18, 36, 54, 86, 118)

VALID_ORBITAL_TYPES: Tuple[str, ...] = ("sigma", "pi", "delta", "lone_pair")
VALID_ORIENTATIONS: Tuple[str, ...] = ("in_plane", "out_of_plane")


def core(Z: int) -> int:
    """Returns the noble-gas core electron count for atomic number Z."""
    z_int = int(Z)
    if z_int < 1 or z_int > 118:
        raise ValueError(f"Atomic number Z must be in 1..118, got {Z!r}")
    return max(c for c in NOBLE_CORES if c <= z_int)


def core_next(Z: int) -> int:
    """Returns the next noble-gas shell closure electron count for atomic number Z."""
    z_int = int(Z)
    if z_int < 1 or z_int > 118:
        raise ValueError(f"Atomic number Z must be in 1..118, got {Z!r}")
    return min(c for c in NOBLE_NEXT if c >= z_int)


def total_valency_available(Z: int) -> int:
    """Returns total valence electron capacity (core_next(Z) - core(Z)) in {0, 2, 8, 18, 32}."""
    return core_next(Z) - core(Z)


def n_valence_orbitals(Z: int) -> int:
    """Returns the number of valence orbitals (total_valency_available(Z) // 2)."""
    return total_valency_available(Z) // 2


def occupied_valency(Z: int, charge: float = 0.0) -> int:
    """Returns the number of valence electrons (Z - core(Z) - charge)."""
    if isinstance(charge, bool) or abs(float(charge) - round(float(charge))) > 1e-9:
        raise ValueError(f"Formal charge must be integer-valued for occupied_valency, got {charge!r}")
    occ = int(Z) - core(Z) - int(round(float(charge)))
    if occ < 0 or occ > total_valency_available(Z):
        raise ValueError(
            f"Invalid occupied_valency={occ} for Z={Z}, charge={charge} "
            f"(allowed range 0..{total_valency_available(Z)})"
        )
    return occ


@dataclass
class ValenceOrbital:
    """
    Orbital-level valence electron and orientation record on an AtomConfig.
    """
    spin: str = "none"
    # 'up' | 'down' | 'none' — same vocabulary as AtomConfig.spin
    orbital_type: str = "sigma"
    # role / symmetry: sigma | pi | delta | lone_pair
    occupation: int = 0
    # electrons in this orbital: 0 (empty), 1 (singly occupied), 2 (doubly occupied)
    connect_to: Optional[str] = None
    # atom label this orbital is directed at / interacts with
    plane: Optional[List[str]] = None
    # None / unknown, or 1-3 atom labels
    orientation: str = "in_plane"
    # in_plane (default) | out_of_plane

    def __post_init__(self):
        # Rule 1: Normalize and validate spin
        if isinstance(self.spin, bool) or self.spin is None:
            raise ValueError(f"ValenceOrbital.spin must be 'up', 'down', or 'none' (or +1/-1/0), got {self.spin!r}")
        if isinstance(self.spin, int):
            if self.spin == 1:
                self.spin = "up"
            elif self.spin == -1:
                self.spin = "down"
            elif self.spin == 0:
                self.spin = "none"
            else:
                raise ValueError(f"ValenceOrbital.spin integer must be +1, -1, or 0, got {self.spin!r}")
        elif isinstance(self.spin, str):
            s = self.spin.strip().lower()
            if s in ("up", "alpha", "+1", "+"):
                self.spin = "up"
            elif s in ("down", "beta", "-1", "-"):
                self.spin = "down"
            elif s in ("none", "0"):
                self.spin = "none"
            else:
                raise ValueError(f"ValenceOrbital.spin must be 'up', 'down', or 'none', got {self.spin!r}")
        else:
            raise ValueError(f"ValenceOrbital.spin must be 'up', 'down', or 'none', got {self.spin!r}")

        # Rule 2: Validate orbital_type
        if not isinstance(self.orbital_type, str):
            raise ValueError(f"ValenceOrbital.orbital_type must be a string in {VALID_ORBITAL_TYPES}, got {self.orbital_type!r}")
        ot = self.orbital_type.strip().lower()
        if ot not in VALID_ORBITAL_TYPES:
            raise ValueError(
                f"ValenceOrbital.orbital_type must be one of {VALID_ORBITAL_TYPES} "
                f"(emptiness is an occupation, not a role), got {self.orbital_type!r}"
            )
        self.orbital_type = ot

        # Rule 3: Validate occupation and consistency with spin
        if isinstance(self.occupation, bool) or not isinstance(self.occupation, int):
            if isinstance(self.occupation, float) and self.occupation.is_integer():
                self.occupation = int(self.occupation)
            else:
                raise ValueError(f"ValenceOrbital.occupation must be an integer in {{0, 1, 2}}, got {self.occupation!r}")
        if self.occupation not in (0, 1, 2):
            raise ValueError(f"ValenceOrbital.occupation must be in {{0, 1, 2}}, got {self.occupation!r}")

        if self.occupation == 1 and self.spin not in ("up", "down"):
            raise ValueError(
                f"Singly-occupied ValenceOrbital (occupation=1) must have spin 'up' or 'down', got spin={self.spin!r}"
            )
        if self.occupation in (0, 2) and self.spin != "none":
            raise ValueError(
                f"ValenceOrbital with occupation={self.occupation} must have spin='none', got spin={self.spin!r}"
            )

        # Rule 4: lone_pair => occupation == 2 (and spin == 'none')
        if self.orbital_type == "lone_pair" and self.occupation != 2:
            raise ValueError(
                f"ValenceOrbital with orbital_type='lone_pair' must have occupation=2 (and spin='none'), "
                f"got occupation={self.occupation}, spin={self.spin!r}"
            )

        # Normalize connect_to
        if self.connect_to is not None:
            ct = str(self.connect_to).strip()
            if not ct:
                raise ValueError("ValenceOrbital.connect_to cannot be an empty string")
            self.connect_to = ct

        # Validate plane and orientation (Section 3 & Rule 6)
        if not isinstance(self.orientation, str):
            raise ValueError(f"ValenceOrbital.orientation must be in {VALID_ORIENTATIONS}, got {self.orientation!r}")
        ori = self.orientation.strip().lower()
        if ori not in VALID_ORIENTATIONS:
            raise ValueError(f"ValenceOrbital.orientation must be in {VALID_ORIENTATIONS}, got {self.orientation!r}")
        self.orientation = ori

        if self.plane is not None:
            if not isinstance(self.plane, (list, tuple)) or not (1 <= len(self.plane) <= 3):
                raise ValueError(
                    f"ValenceOrbital.plane must be None or a list of 1 to 3 atom labels, got {self.plane!r}"
                )
            norm_plane: List[str] = []
            for p_lbl in self.plane:
                if not isinstance(p_lbl, str) or not p_lbl.strip():
                    raise ValueError(f"Invalid atom label {p_lbl!r} in ValenceOrbital.plane={self.plane!r}")
                norm_plane.append(p_lbl.strip())
            if len(set(norm_plane)) != len(norm_plane):
                raise ValueError(f"ValenceOrbital.plane cannot contain duplicate atom labels: {self.plane!r}")
            self.plane = norm_plane

        if self.plane is None and self.orientation != "in_plane":
            raise ValueError(
                "ValenceOrbital.orientation='out_of_plane' requires a plane of >= 2 atom labels, got plane=None"
            )
        if self.orientation == "out_of_plane" and (self.plane is None or len(self.plane) < 2):
            raise ValueError(
                f"ValenceOrbital.orientation='out_of_plane' requires a plane of >= 2 atom labels, got plane={self.plane!r}"
            )

    @property
    def occno_pair(self) -> Tuple[float, float]:
        """Returns the (n_alpha, n_beta) occupation pair per Encoding Rule 3."""
        if self.occupation == 1:
            return (1.0, 0.0) if self.spin == "up" else (0.0, 1.0)
        elif self.occupation == 2:
            return (1.0, 1.0)
        return (0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "spin": self.spin,
            "orbital_type": self.orbital_type,
            "occupation": self.occupation,
        }
        if self.connect_to is not None:
            d["connect_to"] = self.connect_to
        if self.plane is not None:
            d["plane"] = list(self.plane)
        if self.orientation != "in_plane" or self.plane is not None:
            d["orientation"] = self.orientation
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValenceOrbital":
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict or ValenceOrbital, got {type(data).__name__}")
        valid_keys = {"spin", "orbital_type", "occupation", "connect_to", "plane", "orientation"}
        unknown = set(data.keys()) - valid_keys
        if unknown:
            raise ValueError(f"Unknown keys in ValenceOrbital specification: {sorted(unknown)}")
        return cls(
            spin=data.get("spin", "none"),
            orbital_type=data.get("orbital_type", "sigma"),
            occupation=data.get("occupation", 0),
            connect_to=data.get("connect_to"),
            plane=data.get("plane"),
            orientation=data.get("orientation", "in_plane"),
        )


def orbital_counts(valence_config: Sequence[ValenceOrbital]) -> Dict[str, int]:
    """
    Returns a dictionary of orbital role counts {'sigma': ..., 'pi': ..., 'delta': ..., 'lone_pair': ...}
    over explicitly listed entries in valence_config (padded empties are not included).
    """
    counts = {t: 0 for t in VALID_ORBITAL_TYPES}
    for entry in valence_config:
        orb = ValenceOrbital.from_dict(entry) if isinstance(entry, dict) else entry
        counts[orb.orbital_type] = counts.get(orb.orbital_type, 0) + 1
    return counts


def occupancy_counts(
    valence_config: Sequence[ValenceOrbital],
    n_orbitals: Optional[int] = None,
) -> Tuple[int, int, int]:
    """
    Returns (n_empty, n_singly, n_doubly) from the occupation field (0, 1, 2).
    If n_orbitals is provided and len(valence_config) < n_orbitals, pads missing entries as empty (occupation=0).
    """
    n_empty = 0
    n_singly = 0
    n_doubly = 0
    for entry in valence_config:
        orb = ValenceOrbital.from_dict(entry) if isinstance(entry, dict) else entry
        if orb.occupation == 0:
            n_empty += 1
        elif orb.occupation == 1:
            n_singly += 1
        elif orb.occupation == 2:
            n_doubly += 1
    if n_orbitals is not None and len(valence_config) < n_orbitals:
        n_empty += (n_orbitals - len(valence_config))
    return (n_empty, n_singly, n_doubly)


def valence_net_spin(valence_config: Sequence[ValenceOrbital]) -> int:
    """Returns (#up) - (#down) over the entries in valence_config."""
    n_up = 0
    n_down = 0
    for entry in valence_config:
        orb = ValenceOrbital.from_dict(entry) if isinstance(entry, dict) else entry
        if orb.spin == "up":
            n_up += 1
        elif orb.spin == "down":
            n_down += 1
    return n_up - n_down


net_orbital_spin = valence_net_spin


@dataclass
class AtomConfig:
    """
    Modernized representation of an atom in a PHATPSY calculation.

    Key Concepts & Invariants:
    - id: Unique atom identifier (e.g. 'Ni1', 'O_top').
    - equivalent_to: Intrinsic reference to another atom ID whose localized atomic state
      and orbitals this atom shares.
    - symmetry_op / symmetry_axis: Secondary properties of equivalency defining the geometric
      transformation relative to the reference atom (e.g. op='reflection' or -1, axis='z' or 100).
    - spin: Intrinsic local spin state ('up' / alpha, 'down' / beta, or None).
    - virtual: True if atom is a ghost/coordination center to complete the local potential
      during atomic calculations, but excluded from extended (molecular) EWMO Hamiltonians.
    - fixed: True if atom is frozen/fixed in the substrate.
    """
    element: str
    id: Optional[str] = None
    xyz: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    equivalent_to: Optional[str] = None
    symmetry_op: Optional[Union[int, str]] = None
    symmetry_axis: Optional[Union[int, str]] = None
    symmetry: Optional[SymmetrySpec] = None  # Backward compatibility
    charge: float = 0.0
    fixed: bool = False
    virtual: bool = False
    spin_flip: bool = False  # Backward compatibility fallback
    spin: Optional[str] = None  # "up", "down", or None
    basis: Optional[str] = None  # Atom-specific basis override (e.g. 'ADF-TZ2P')
    hybridization: Optional[str] = None  # Named hybridization type override (e.g. 'sp3', 'sp2', 'dsp2', 'd2sp3')
    valence_config: Optional[List[ValenceOrbital]] = None  # Explicit orbital-level valence configuration
    embedded_state: Optional[Any] = None
    starting_orbitals: Optional[List[Any]] = None

    def __post_init__(self):
        # Defensively handle positional AtomConfig("N", [x, y, z]) where xyz was passed as 2nd arg (id)
        if isinstance(self.id, (list, tuple)):
            self.xyz = [float(c) for c in self.id]
            self.id = None

        # 1. Normalize spin and hybridization
        if self.spin is not None:
            s = str(self.spin).strip().lower()
            if s in ("up", "alpha", "+1", "1", "+"):
                self.spin = "up"
            elif s in ("down", "beta", "-1", "-"):
                self.spin = "down"
                self.spin_flip = True
            elif s in ("none", "0", "closed"):
                self.spin = None
        elif self.spin_flip:
            self.spin = "down"

        if self.hybridization is not None:
            self.hybridization = str(self.hybridization).strip().lower()

        # 2. Validate valence_config if present (Checks 1-7)
        if self.valence_config is not None:
            if not isinstance(self.valence_config, (list, tuple)):
                raise ValueError(f"AtomConfig.valence_config must be a list of ValenceOrbital entries, got {type(self.valence_config).__name__}")
            self.valence_config = [
                (ValenceOrbital.from_dict(entry) if isinstance(entry, dict) else entry)
                for entry in self.valence_config
            ]
            for entry in self.valence_config:
                if not isinstance(entry, ValenceOrbital):
                    raise TypeError(f"Expected ValenceOrbital or dict in valence_config, got {type(entry).__name__}")
            self.validate_valence_config()

        # 3. Reconcile / validate against starting_orbitals if present
        if self.starting_orbitals:
            net_spin = self.net_orbital_spin
            if net_spin is not None and abs(net_spin) > 1e-6:
                expected_spin = "up" if net_spin > 0 else "down"
                if self.spin is not None and self.spin != expected_spin:
                    atom_name = self.id or self.element
                    tot_a = sum(orb.occno[0] for orb in self.starting_orbitals if len(orb.occno) > 0)
                    tot_b = sum(orb.occno[1] for orb in self.starting_orbitals if len(orb.occno) > 1)
                    raise ValueError(
                        f"Conflicting spin specification for atom '{atom_name}': "
                        f"explicit spin='{self.spin}', but starting_orbitals define net {expected_spin} spin "
                        f"(alpha={tot_a:.2f}, beta={tot_b:.2f}, delta={net_spin:.2f})."
                    )
                elif self.spin is None:
                    self.spin = expected_spin
                    self.spin_flip = (expected_spin == "down")

        # 4. Synchronize legacy SymmetrySpec with modern symmetry_op / symmetry_axis
        if self.symmetry is not None:
            if self.symmetry_op is None:
                self.symmetry_op = self.symmetry.isym
            if self.symmetry_axis is None:
                self.symmetry_axis = self.symmetry.ixyz
            if self.spin is None and self.symmetry.qflip:
                self.spin = "down"
                self.spin_flip = True
        elif self.symmetry_op is not None or self.symmetry_axis is not None:
            op = normalize_symmetry_op(self.symmetry_op) or -1
            ax = normalize_symmetry_axis(self.symmetry_axis) or 1
            do_flip = (self.spin == "down") if self.spin is not None else self.spin_flip
            self.symmetry = SymmetrySpec(ixyz=ax, isym=op, qflip=do_flip)
            self.symmetry_op = op
            self.symmetry_axis = ax

    def validate_valence_config(self) -> None:
        """
        Validates atom-local valence_config invariants (Section 4 encoding rules and Section 5 Checks 1-7).
        """
        if self.valence_config is None:
            return

        atom_name = self.id or self.element
        sym = self.element.strip().capitalize()
        if sym not in ATOMIC_NUMBERS:
            raise ValueError(f"Unknown chemical element '{self.element}' for atom '{atom_name}'")
        Z = ATOMIC_NUMBERS[sym]

        # Rule 8: Formal charge must be integer-valued when valence_config is present
        if isinstance(self.charge, bool) or abs(float(self.charge) - round(float(self.charge))) > 1e-9:
            raise ValueError(
                f"Atom '{atom_name}': charge must be integer-valued (formal charge) when valence_config is present, "
                f"got {self.charge!r}"
            )

        # Rule 5: Self-reference check (when id is known)
        if self.id:
            for orb in self.valence_config:
                if orb.connect_to == self.id:
                    raise ValueError(
                        f"Atom '{atom_name}': ValenceOrbital.connect_to cannot reference the atom itself ('{self.id}')"
                    )
                if orb.plane and self.id in orb.plane:
                    raise ValueError(
                        f"Atom '{atom_name}': ValenceOrbital.plane cannot contain the atom itself ('{self.id}')"
                    )

        n_orb = n_valence_orbitals(Z)
        occ_val = occupied_valency(Z, self.charge)

        # Check 1 (Length): len(valence_config) <= n_valence_orbitals
        if len(self.valence_config) > n_orb:
            raise ValueError(
                f"Atom '{atom_name}' ({sym}, Z={Z}): valence_config has {len(self.valence_config)} entries, "
                f"exceeding n_valence_orbitals={n_orb}"
            )

        # Check 2 (Orbital count after padding missing entries as empty)
        n_empty, n_singly, n_doubly = occupancy_counts(self.valence_config, n_orbitals=n_orb)
        if n_empty + n_singly + n_doubly != n_orb:
            raise ValueError(
                f"Atom '{atom_name}' ({sym}, Z={Z}): orbital count mismatch "
                f"({n_empty} + {n_singly} + {n_doubly} != {n_orb})"
            )

        # Check 3 (Electron count): n_singly + 2 * n_doubly == occupied_valency
        counted_electrons = n_singly + 2 * n_doubly
        if counted_electrons != occ_val:
            raise ValueError(
                f"Atom '{atom_name}' ({sym}, Z={Z}, charge={self.charge}): valence_config counts "
                f"{counted_electrons} valence electrons (n_singly={n_singly}, n_doubly={n_doubly}), "
                f"but occupied_valency is {occ_val}"
            )

        # Check 4 (Spin sign): net_orbital_spin must agree in sign with atom's spin label when nonzero
        net_s = valence_net_spin(self.valence_config)
        if net_s > 0:
            if self.spin is not None and self.spin != "up":
                raise ValueError(
                    f"Atom '{atom_name}': spin='{self.spin}' contradicts positive valence_config net orbital spin (+{net_s})"
                )
            self.spin = "up"
            self.spin_flip = False
        elif net_s < 0:
            if self.spin is not None and self.spin != "down":
                raise ValueError(
                    f"Atom '{atom_name}': spin='{self.spin}' contradicts negative valence_config net orbital spin ({net_s})"
                )
            self.spin = "down"
            self.spin_flip = True

        # Check 5 (Hybridization-label consistency over explicitly listed entries only)
        if self.hybridization is not None:
            from phatpsy.hybrids import get_hybridization_spec
            try:
                spec = get_hybridization_spec(self.hybridization)
            except KeyError as exc:
                raise ValueError(f"Atom '{atom_name}': {exc}") from exc

            counts = orbital_counts(self.valence_config)
            n_hybrids = spec.n_sigma + spec.n_lone_pairs
            actual_hybrids = counts["sigma"] + counts["lone_pair"]
            pi_deficit = spec.n_pi - counts["pi"]
            used_dative_as_pi = (
                pi_deficit if (0 <= pi_deficit <= counts["lone_pair"] and actual_hybrids - pi_deficit == n_hybrids) else 0
            )
            if (actual_hybrids - used_dative_as_pi) != n_hybrids:
                raise ValueError(
                    f"Atom '{atom_name}' (hybridization='{self.hybridization}'): "
                    f"n_sigma ({counts['sigma']}) + n_lone_pair ({counts['lone_pair']}) = {actual_hybrids}, "
                    f"expected {n_hybrids} hybrids for '{spec.name}'"
                )
            if (counts["pi"] + used_dative_as_pi) != spec.n_pi:
                raise ValueError(
                    f"Atom '{atom_name}' (hybridization='{self.hybridization}'): "
                    f"n_pi={counts['pi']}, expected {spec.n_pi} for '{spec.name}'"
                )
            allows_d = ("d" in self.hybridization.lower()) or ("d" in spec.name.lower()) or spec.has_d_orbitals
            if counts["delta"] > 0 and not allows_d:
                raise ValueError(
                    f"Atom '{atom_name}' (hybridization='{self.hybridization}'): "
                    f"n_delta={counts['delta']} is only allowed when hybridization label contains 'd'"
                )

        # Check 7 (starting_orbitals agreement when both are present)
        if self.starting_orbitals:
            self._validate_starting_orbitals_against_valence_config(Z, n_orb)

    def _validate_starting_orbitals_against_valence_config(self, Z: int, n_orb: int) -> None:
        """
        Verifies that starting_orbitals alpha/beta occupations match valence_config (Check 7).
        """
        atom_name = self.id or self.element
        from phatpsy.basis import InitialOrbital

        orbs = [
            (InitialOrbital.from_dict(o) if isinstance(o, dict) else o)
            for o in self.starting_orbitals
        ]
        padded_vc = list(self.valence_config) + [
            ValenceOrbital(spin="none", orbital_type="sigma", occupation=0)
            for _ in range(n_orb - len(self.valence_config))
        ]
        vc_pairs = [o.occno_pair for o in padded_vc]

        def _classify_pair(a: float, b: float) -> str:
            if abs(a - 1.0) < 1e-6 and abs(b - 0.0) < 1e-6:
                return "up"
            if abs(a - 0.0) < 1e-6 and abs(b - 1.0) < 1e-6:
                return "down"
            if abs(a - 1.0) < 1e-6 and abs(b - 1.0) < 1e-6:
                return "doubly"
            if abs(a - 0.0) < 1e-6 and abs(b - 0.0) < 1e-6:
                return "empty"
            return f"fractional({a:.4f},{b:.4f})"

        n_core = core(Z) // 2
        if len(orbs) <= n_orb:
            # Valence-only starting_orbitals list
            val_orbs = orbs
            # Check direct positional match if same length as explicit valence_config or padded
            if len(val_orbs) == len(self.valence_config):
                for idx, (so, vo) in enumerate(zip(val_orbs, self.valence_config)):
                    sa = float(so.occno[0]) if len(so.occno) > 0 else 0.0
                    sb = float(so.occno[1]) if len(so.occno) > 1 else 0.0
                    va, vb = vo.occno_pair
                    if abs(sa - va) > 1e-6 or abs(sb - vb) > 1e-6:
                        raise ValueError(
                            f"Atom '{atom_name}': starting_orbitals[{idx}] occno=({sa}, {sb}) conflicts with "
                            f"valence_config[{idx}] occno=({va}, {vb})"
                        )
                return
        else:
            # Full-atom starting_orbitals (includes n_core doubly-occupied core orbitals)
            doubly_indices = [
                idx for idx, o in enumerate(orbs)
                if len(o.occno) >= 2 and abs(float(o.occno[0]) - 1.0) < 1e-6 and abs(float(o.occno[1]) - 1.0) < 1e-6
            ]
            if len(doubly_indices) < n_core:
                raise ValueError(
                    f"Atom '{atom_name}': starting_orbitals has only {len(doubly_indices)} doubly-occupied orbitals, "
                    f"fewer than the {n_core} required core orbitals for Z={Z}"
                )
            # Sort doubly occupied candidates by lowest eigenvalue if present, else index
            doubly_indices.sort(
                key=lambda idx: (float(orbs[idx].eigval[0]) if orbs[idx].eigval else 0.0, idx)
            )
            core_set = set(doubly_indices[:n_core])
            val_orbs = [o for idx, o in enumerate(orbs) if idx not in core_set]

        sum_so_a = sum(float(o.occno[0]) if len(o.occno) > 0 else 0.0 for o in val_orbs)
        sum_so_b = sum(float(o.occno[1]) if len(o.occno) > 1 else 0.0 for o in val_orbs)
        sum_vc_a = sum(va for va, _ in vc_pairs)
        sum_vc_b = sum(vb for _, vb in vc_pairs)
        if abs(sum_so_a - sum_vc_a) > 1e-5 or abs(sum_so_b - sum_vc_b) > 1e-5:
            raise ValueError(
                f"Atom '{atom_name}': starting_orbitals valence spin totals (alpha={sum_so_a:.4f}, beta={sum_so_b:.4f}) "
                f"conflict with valence_config totals (alpha={sum_vc_a:.4f}, beta={sum_vc_b:.4f})"
            )

        has_dative_entry = any(
            (o.orbital_type == "lone_pair" and o.connect_to is not None)
            or (o.occupation == 0 and o.connect_to is not None)
            for o in self.valence_config
        )
        if not has_dative_entry:
            so_counts = {"up": 0, "down": 0, "doubly": 0, "empty": 0}
            for o in val_orbs:
                sa = float(o.occno[0]) if len(o.occno) > 0 else 0.0
                sb = float(o.occno[1]) if len(o.occno) > 1 else 0.0
                cat = _classify_pair(sa, sb)
                if cat not in so_counts:
                    raise ValueError(
                        f"Atom '{atom_name}': starting_orbitals contains non-integer occupation ({sa}, {sb}) "
                        f"incompatible with valence_config"
                    )
                so_counts[cat] += 1

            vc_counts = {"up": 0, "down": 0, "doubly": 0, "empty": 0}
            for va, vb in vc_pairs:
                vc_counts[_classify_pair(va, vb)] += 1

            if (
                so_counts["up"] != vc_counts["up"]
                or so_counts["down"] != vc_counts["down"]
                or so_counts["doubly"] != vc_counts["doubly"]
            ):
                raise ValueError(
                    f"Atom '{atom_name}': starting_orbitals valence occupations {so_counts} conflict with "
                    f"valence_config occupations {vc_counts}"
                )

    @property
    def is_spin_down(self) -> bool:
        return self.spin == "down" or self.spin_flip

    @property
    def is_spin_up(self) -> bool:
        return self.spin == "up"

    @property
    def net_orbital_spin(self) -> Optional[float]:
        """Returns sum(alpha_occ) - sum(beta_occ) across valence_config or starting orbitals, if defined."""
        if self.valence_config is not None:
            return float(valence_net_spin(self.valence_config))
        if not self.starting_orbitals:
            return None
        tot_alpha = sum(orb.occno[0] for orb in self.starting_orbitals if len(orb.occno) > 0)
        tot_beta = sum(orb.occno[1] for orb in self.starting_orbitals if len(orb.occno) > 1)
        return tot_alpha - tot_beta

    @property
    def is_virtual(self) -> bool:
        return self.virtual

    @property
    def is_fixed(self) -> bool:
        return self.fixed

    @property
    def is_equivalent(self) -> bool:
        return self.equivalent_to is not None or self.symmetry is not None

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "element": self.element,
            "xyz": [round(float(c), 6) for c in self.xyz],
        }
        if self.id:
            res["id"] = self.id
        if self.equivalent_to:
            res["equivalent_to"] = self.equivalent_to
        if self.symmetry_op is not None:
            res["symmetry_op"] = self.symmetry_op
        if self.symmetry_axis is not None:
            res["symmetry_axis"] = self.symmetry_axis
        elif self.symmetry is not None and not self.equivalent_to:
            res["symmetry"] = self.symmetry.to_dict()
        if self.charge != 0.0:
            res["charge"] = self.charge
        if self.fixed:
            res["fixed"] = True
        if self.virtual:
            res["virtual"] = True
        if self.spin:
            res["spin"] = self.spin
        elif self.spin_flip:
            res["spin"] = "down"
        if self.basis:
            res["basis"] = self.basis
        if self.hybridization:
            res["hybridization"] = self.hybridization
        if self.valence_config is not None:
            res["valence_config"] = [
                (o.to_dict() if hasattr(o, "to_dict") else o)
                for o in self.valence_config
            ]
        if self.embedded_state:
            if hasattr(self.embedded_state, "to_dict"):
                res["embedded_state"] = self.embedded_state.to_dict()
            elif isinstance(self.embedded_state, dict):
                res["embedded_state"] = self.embedded_state
        if self.starting_orbitals:
            res["starting_orbitals"] = [
                (o.to_dict() if hasattr(o, "to_dict") else o)
                for o in self.starting_orbitals
            ]
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AtomConfig":
        sym = None
        if "symmetry" in data and data["symmetry"]:
            sym = SymmetrySpec(**data["symmetry"])
        xyz = data.get("xyz", [0.0, 0.0, 0.0])
        spin_val = data.get("spin")
        spin_flip_val = bool(data.get("spin_flip", False))
        if spin_val is None and spin_flip_val:
            spin_val = "down"

        emb_data = data.get("embedded_state")
        if emb_data and isinstance(emb_data, dict):
            from phatpsy.state import EmbeddedAtomState
            emb = EmbeddedAtomState.from_dict(emb_data)
        else:
            emb = emb_data

        val_cfg = None
        if "valence_config" in data and data["valence_config"] is not None:
            if not isinstance(data["valence_config"], (list, tuple)):
                raise ValueError(f"valence_config must be a list, got {type(data['valence_config']).__name__}")
            val_cfg = [
                (ValenceOrbital.from_dict(o) if isinstance(o, dict) else o)
                for o in data["valence_config"]
            ]

        starting_orbs = None
        if "starting_orbitals" in data and data["starting_orbitals"]:
            from phatpsy.basis import InitialOrbital
            starting_orbs = [
                (InitialOrbital.from_dict(o) if isinstance(o, dict) else o)
                for o in data["starting_orbitals"]
            ]
            if spin_val is None and val_cfg is None:
                tot_alpha = sum(orb.occno[0] for orb in starting_orbs if len(orb.occno) > 0)
                tot_beta = sum(orb.occno[1] for orb in starting_orbs if len(orb.occno) > 1)
                if tot_beta > tot_alpha:
                    spin_val = "down"
                    spin_flip_val = True
                elif tot_alpha > tot_beta:
                    spin_val = "up"

        if spin_val is None and spin_flip_val:
            spin_val = "down"

        return cls(
            element=data["element"].strip().capitalize(),
            id=data.get("id"),
            xyz=[float(x) for x in xyz],
            equivalent_to=data.get("equivalent_to"),
            symmetry_op=data.get("symmetry_op"),
            symmetry_axis=data.get("symmetry_axis"),
            symmetry=sym,
            charge=float(data.get("charge", 0.0)),
            fixed=bool(data.get("fixed", False)),
            virtual=bool(data.get("virtual", False)),
            spin_flip=spin_flip_val,
            spin=spin_val,
            basis=data.get("basis"),
            hybridization=data.get("hybridization"),
            valence_config=val_cfg,
            embedded_state=emb,
            starting_orbitals=starting_orbs,
        )


@dataclass
class ElementConfig:
    symbol: str
    atomic_number: Optional[int] = None
    basis: Optional[str] = None  # "SZ", "DZ", etc.
    description: Optional[str] = None
    functions: List[BasisFunction] = field(default_factory=list)
    potential_exponents: List[float] = field(default_factory=list)  # Model-potential screening exponents (the 1/r term is implicit); empty keeps the built-in set.
    potential_exponent_scale: Optional[float] = None  # Multiplier applied to potential_exponents; null keeps the engine's Z-based scale (1.12956 * Z**(1/3)), 1.0 means absolute exponents.

    # Fit coefficients need at least two exponentials; the engine arrays hold at most 20 model-potential terms.
    MIN_POTENTIAL_EXPONENTS = 2
    MAX_POTENTIAL_EXPONENTS = 19

    def __post_init__(self):
        if self.potential_exponents:
            exps = [float(e) for e in self.potential_exponents]
            if not all(math.isfinite(e) and e > 0.0 for e in exps):
                raise ValueError(
                    f"{self.symbol}: potential_exponents must be finite and > 0, got {self.potential_exponents!r}"
                )
            if not (self.MIN_POTENTIAL_EXPONENTS <= len(exps) <= self.MAX_POTENTIAL_EXPONENTS):
                raise ValueError(
                    f"{self.symbol}: potential_exponents needs between {self.MIN_POTENTIAL_EXPONENTS} and "
                    f"{self.MAX_POTENTIAL_EXPONENTS} values, got {len(exps)}"
                )
            ordered = sorted(exps)
            if any(b - a <= 1.0e-6 * b for a, b in zip(ordered, ordered[1:])):
                raise ValueError(
                    f"{self.symbol}: potential_exponents must be distinct (duplicates make the fit singular), "
                    f"got {self.potential_exponents!r}"
                )
            self.potential_exponents = exps
        if self.potential_exponent_scale is not None:
            if not (math.isfinite(self.potential_exponent_scale) and self.potential_exponent_scale > 0.0):
                raise ValueError(
                    f"{self.symbol}: potential_exponent_scale must be a finite number > 0, "
                    f"got {self.potential_exponent_scale!r}"
                )
            if not self.potential_exponents:
                raise ValueError(
                    f"{self.symbol}: potential_exponent_scale only applies together with potential_exponents"
                )

    def to_dict(self) -> Dict[str, Any]:
        res = {"symbol": self.symbol}
        if self.atomic_number is not None:
            res["atomic_number"] = self.atomic_number
        if self.basis:
            res["basis"] = self.basis
        if self.description:
            res["description"] = self.description
        if self.functions:
            res["functions"] = [f.to_dict() for f in self.functions]
        if self.potential_exponents:
            res["potential_exponents"] = list(self.potential_exponents)
        if self.potential_exponent_scale is not None:
            res["potential_exponent_scale"] = self.potential_exponent_scale
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ElementConfig":
        funcs = []
        if "functions" in data and data["functions"]:
            funcs = [BasisFunction.from_dict(f) for f in data["functions"]]
        return cls(
            symbol=data["symbol"].strip().capitalize(),
            atomic_number=data.get("atomic_number"),
            basis=data.get("basis"),
            description=data.get("description"),
            functions=funcs,
            potential_exponents=list(data.get("potential_exponents") or []),
            potential_exponent_scale=data.get("potential_exponent_scale"),
        )


def _spin_swapped_orbital(orb: Any) -> Any:
    """Returns a copy of an initial-orbital record with alpha and beta swapped.

    Swaps ``occno``, ``eigval`` and (for 4-column entries) the alpha/beta
    expansion coefficients.  Three-column coefficient entries are spin-symmetric
    and are left unchanged.
    """
    kwargs: Dict[str, Any] = {}
    if orb.occno:
        kwargs["occno"] = type(orb.occno)((orb.occno[1], orb.occno[0])) if not isinstance(orb.occno, list) else [orb.occno[1], orb.occno[0]]
    if orb.eigval:
        kwargs["eigval"] = type(orb.eigval)((orb.eigval[1], orb.eigval[0])) if not isinstance(orb.eigval, list) else [orb.eigval[1], orb.eigval[0]]
    if orb.coefficients:
        kwargs["coefficients"] = [
            (c[0], c[1], c[3], c[2]) + tuple(c[4:]) if len(c) >= 4 else c
            for c in orb.coefficients
        ]
    return replace(orb, **kwargs)


def resolve_atom_coordinates(atoms: List[AtomConfig], force_symmetry: bool = False) -> List[AtomConfig]:
    """
    Resolves unset [0, 0, 0] or duplicate coordinates on equivalent atoms by
    applying their symmetry operations sequentially (matching Fortran SYMXYZ: LAST = IATOM - 1).
    If force_symmetry is True, always recomputes symmetry-equivalent atom coordinates from their reference atom.
    """
    id_map = {a.id: a for a in atoms if a.id}
    last_atom_by_elem: Dict[str, AtomConfig] = {}

    for a in atoms:
        has_sym = (a.equivalent_to is not None or a.symmetry is not None or a.symmetry_op is not None)
        if has_sym:
            op = a.symmetry_op if a.symmetry_op is not None else (a.symmetry.isym if a.symmetry else 1)
            ax = a.symmetry_axis if a.symmetry_axis is not None else (a.symmetry.ixyz if a.symmetry else 1)
            op_val = normalize_symmetry_op(op) or 1
            ax_val = normalize_symmetry_axis(ax) or 1

            if op_val not in (0, 1):
                prev_atom = last_atom_by_elem.get(a.element)
                ref = id_map.get(a.equivalent_to) if a.equivalent_to else None
                source = prev_atom if prev_atom else ref

                if source and (force_symmetry or a.xyz == [0.0, 0.0, 0.0] or a.xyz == source.xyz):
                    a.xyz = apply_symmetry_operation(source.xyz, isym=op_val, ixyz=ax_val)
        last_atom_by_elem[a.element] = a
    return atoms


@dataclass
class CalculationDeck:
    title: str = "PHATPSY Calculation"
    control: ControlConfig = field(default_factory=ControlConfig)
    geometry: List[AtomConfig] = field(default_factory=list)
    elements: Dict[str, ElementConfig] = field(default_factory=dict)
    basis: Optional[str] = None
    net_spin: int = 0
    ferromagnetic_links: List[Tuple[str, str]] = field(default_factory=list)
    ferromagnetic_reports: List[Any] = field(default_factory=list)
    cell: Optional[List[List[float]]] = None
    pbc: Optional[List[bool]] = None

    def __post_init__(self):
        if self.geometry:
            self.validate()

    @property
    def atoms(self) -> List[AtomConfig]:
        """Alias for self.geometry."""
        return self.geometry

    @atoms.setter
    def atoms(self, val: List[AtomConfig]) -> None:
        self.geometry = val

    @property
    def spins(self) -> List[str]:
        """Returns the explicit per-atom spin assignments ('up' or 'down')."""
        return [a.spin for a in self.geometry if a.spin is not None]

    @property
    def physical_net_spin(self) -> Optional[int]:
        """
        Returns the molecule's physical net spin 2S = sum_A net_orbital_spin(A) across all
        non-virtual atoms when every non-virtual atom has an orbital spin source
        (valence_config or starting_orbitals); otherwise returns None.
        """
        real_atoms = [a for a in self.geometry if not a.virtual]
        if not real_atoms:
            return None
        total = 0.0
        for a in real_atoms:
            s_val = a.net_orbital_spin
            if s_val is None:
                return None
            total += float(s_val)
        return int(round(total))

    def validate(self) -> None:
        """
        Validates calculation deck invariants:
        1. Explicit per-atom spins, no None: Every atom in a built deck carries spin 'up' or 'down'.
        2. Net spin conservation: (n_up - n_down) == self.net_spin.
        3. Pairings sanity check: Genuine symmetry-equivalent centers for spin-flip pairing,
           and bipartite flip-pairing graph without odd cycles.
        4. Ferromagnetic links consistency: links reference existing atoms and have parallel spins.
        5. Valence configuration cross-atom checks (Checks 8-11).
        """
        if not self.geometry:
            return

        self.ensure_atom_ids()
        id_map = {a.id: a for a in self.geometry if a.id}

        # Re-run atom-local valence_config validation now that atom IDs are guaranteed,
        # and perform cross-atom valence_config checks (Checks 8-10).
        self._validate_valence_configs(id_map)

        if not self.control.spin_unrestricted:
            # Restricted (closed-shell) decks have a single spin channel: per-atom
            # up/down labels carry no meaning (the engine ignores QFLIP when QOPEN=F),
            # so they are optional and not validated.
            return

        n_up = 0
        n_down = 0
        for a in self.geometry:
            if a.virtual and a.spin is None:
                # Virtual (ghost) centers carry 0 electrons and have no spin channel
                continue
            if a.spin not in ("up", "down"):
                raise ValueError(
                    f"Atom '{a.id or a.element}' has unassigned or invalid spin ({a.spin!r}). "
                    f"Every atom in a built calculation deck must carry an explicit spin ('up' or 'down'). "
                    f"None anywhere in the spin path is forbidden."
                )
            if a.spin == "up":
                n_up += 1
            else:
                n_down += 1

        calc_net = n_up - n_down
        if calc_net != self.net_spin:
            raise ValueError(
                f"Net spin mismatch: calculated n_up ({n_up}) - n_down ({n_down}) = {calc_net}, "
                f"but deck net_spin is {self.net_spin}. "
                f"Explicit per-atom spins must satisfy (n_up - n_down) == net_spin."
            )

        # Sanity-check symmetry pairings and bipartiteness
        pair_adj: Dict[str, List[str]] = {a.id: [] for a in self.geometry if a.id}
        for a in self.geometry:
            if a.equivalent_to:
                ref = id_map.get(a.equivalent_to)
                if not ref:
                    raise ValueError(
                        f"Atom '{a.id}' has equivalent_to='{a.equivalent_to}', "
                        f"but no reference atom with that ID was found."
                    )
                if ref.element.strip().capitalize() != a.element.strip().capitalize():
                    raise ValueError(
                        f"Invalid spin-pairing: Atom '{a.id}' ({a.element}) references '{ref.id}' ({ref.element}) by symmetry, "
                        f"but they are different elements. Spin-flip symmetry pairing requires genuine symmetry equivalents."
                    )
                if a.spin != ref.spin:
                    pair_adj[a.id].append(ref.id)
                    pair_adj[ref.id].append(a.id)

        atom_ids = list(pair_adj.keys())
        id_to_idx = {aid: idx for idx, aid in enumerate(atom_ids)}
        adj_indices = [
            [id_to_idx[nbr] for nbr in pair_adj[aid]]
            for aid in atom_ids
        ]
        from phatpsy.spin import find_odd_cycle, NonBipartiteSpinGraphError
        odd_cycle = find_odd_cycle(adj_indices)
        if odd_cycle is not None:
            cycle_atoms = [atom_ids[idx] for idx in odd_cycle]
            raise NonBipartiteSpinGraphError(
                f"Flip-pairing graph contains an odd cycle (frustrated symmetry ring): {cycle_atoms}. "
                f"Flip-pairing graphs must be bipartite."
            )

        # Validate ferromagnetic_links if declared
        if self.ferromagnetic_links:
            for link in self.ferromagnetic_links:
                if len(link) != 2:
                    raise ValueError(f"Invalid ferromagnetic link {link}: expected 2-tuple of atom IDs.")
                id1, id2 = str(link[0]), str(link[1])
                a1 = id_map.get(id1)
                a2 = id_map.get(id2)
                if not a1:
                    raise ValueError(f"Ferromagnetic link references atom '{id1}', but no atom with that ID was found.")
                if not a2:
                    raise ValueError(f"Ferromagnetic link references atom '{id2}', but no atom with that ID was found.")
                if a1.valence_config is not None and a2.valence_config is not None:
                    # Amendment 2: validate declared ferromagnetic link at the orbital level
                    if not self._has_same_spin_facing_orbitals(a1, a2):
                        raise ValueError(
                            f"Ferromagnetic link ({id1}, {id2}) does not have same-spin facing singly-occupied orbitals "
                            f"in valence_config."
                        )
                else:
                    if a1.spin != a2.spin:
                        raise ValueError(
                            f"Ferromagnetic link ({id1}, {id2}) has antiparallel spins ({a1.spin} vs {a2.spin}). "
                            f"Ferromagnetic links represent parallel spin couplings forced by non-bipartite frustration."
                        )

    @staticmethod
    def _has_same_spin_facing_orbitals(a1: AtomConfig, a2: AtomConfig) -> bool:
        """
        Returns True if atoms a1 and a2 have facing singly-occupied orbitals (mutual connect_to)
        that carry the same spin ('up' == 'up' or 'down' == 'down').
        """
        if a1.valence_config is None or a2.valence_config is None:
            return False
        facing_1 = [o for o in a1.valence_config if o.occupation == 1 and o.connect_to == a2.id]
        facing_2 = [o for o in a2.valence_config if o.occupation == 1 and o.connect_to == a1.id]
        if not facing_1 or not facing_2:
            return False
        # Group by orbital_type first (sigma with sigma, pi with pi, delta with delta)
        by_type_1: Dict[str, List[ValenceOrbital]] = {}
        for o in facing_1:
            by_type_1.setdefault(o.orbital_type, []).append(o)
        by_type_2: Dict[str, List[ValenceOrbital]] = {}
        for o in facing_2:
            by_type_2.setdefault(o.orbital_type, []).append(o)

        common_types = set(by_type_1.keys()) & set(by_type_2.keys())
        if common_types:
            for t in common_types:
                list1 = by_type_1[t]
                list2 = by_type_2[t]
                # Try to pair antiparallel spins first; if any forced pair has identical spin, it is ferromagnetic
                rem2 = [o.spin for o in list2]
                for o1 in list1:
                    opp = "down" if o1.spin == "up" else "up"
                    if opp in rem2:
                        rem2.remove(opp)
                    elif o1.spin in rem2:
                        return True
            return False

        # Fallback if facing orbitals have different roles
        return any(o1.spin == o2.spin for o1 in facing_1 for o2 in facing_2)

    def _validate_valence_configs(self, id_map: Dict[str, AtomConfig]) -> None:
        """
        Runs cross-atom valence_config sanity checks (Checks 8, 9, and 10).
        """
        import warnings

        for a in self.geometry:
            if a.valence_config is not None:
                a.validate_valence_config()

        # Check 8: Every connect_to and plane label must name an existing atom id (and not self)
        for a in self.geometry:
            if a.valence_config is None:
                continue
            for orb in a.valence_config:
                if orb.connect_to is not None:
                    if orb.connect_to == a.id:
                        raise ValueError(
                            f"Atom '{a.id}': ValenceOrbital.connect_to cannot name the atom itself ('{a.id}')"
                        )
                    if orb.connect_to not in id_map:
                        raise ValueError(
                            f"Atom '{a.id}': ValenceOrbital.connect_to='{orb.connect_to}' does not match any "
                            f"atom id in the molecule ({list(id_map.keys())})"
                        )
                if orb.plane is not None:
                    for p_lbl in orb.plane:
                        if p_lbl == a.id:
                            raise ValueError(
                                f"Atom '{a.id}': ValenceOrbital.plane cannot contain the atom itself ('{a.id}')"
                            )
                        if p_lbl not in id_map:
                            raise ValueError(
                                f"Atom '{a.id}': ValenceOrbital.plane label '{p_lbl}' does not match any "
                                f"atom id in the molecule ({list(id_map.keys())})"
                            )

        # Check 9: Dative pairing
        # A lone_pair entry with connect_to = X must be matched by an entry on X with occupation == 0 and connect_to = this atom.
        for a in self.geometry:
            if a.valence_config is None:
                continue
            for orb in a.valence_config:
                if orb.orbital_type == "lone_pair" and orb.connect_to is not None:
                    target = id_map[orb.connect_to]
                    if target.valence_config is None:
                        warnings.warn(
                            f"Dative donor lone_pair on '{a.id}' points to '{target.id}', which has no valence_config; "
                            f"cannot verify matching vacant acceptor orbital.",
                            UserWarning,
                            stacklevel=2,
                        )
                    else:
                        has_acceptor = any(
                            t_orb.occupation == 0 and t_orb.connect_to == a.id
                            for t_orb in target.valence_config
                        )
                        if not has_acceptor:
                            raise ValueError(
                                f"Unmatched dative bond: '{a.id}' has a lone_pair with connect_to='{target.id}', "
                                f"but '{target.id}' has no vacant acceptor orbital (occupation=0, connect_to='{a.id}')"
                            )
                elif orb.occupation == 0 and orb.connect_to is not None:
                    donor = id_map[orb.connect_to]
                    if donor.valence_config is None:
                        warnings.warn(
                            f"Vacant acceptor orbital on '{a.id}' points to '{donor.id}', which has no valence_config; "
                            f"cannot verify matching donor lone_pair.",
                            UserWarning,
                            stacklevel=2,
                        )
                    else:
                        has_donor = any(
                            d_orb.orbital_type == "lone_pair" and d_orb.occupation == 2 and d_orb.connect_to == a.id
                            for d_orb in donor.valence_config
                        )
                        if not has_donor:
                            raise ValueError(
                                f"Unmatched dative acceptor: '{a.id}' has a vacant orbital (occupation=0) with "
                                f"connect_to='{donor.id}', but '{donor.id}' has no lone_pair with connect_to='{a.id}'"
                            )

        # Check 10: Ferromagnetic declaration
        # Two atoms whose facing singly-occupied orbitals carry the same spin must appear in ferromagnetic_links.
        declared_pairs = {
            frozenset((str(link[0]), str(link[1])))
            for link in (self.ferromagnetic_links or [])
            if len(link) == 2
        }
        n_atoms = len(self.geometry)
        for i in range(n_atoms):
            a1 = self.geometry[i]
            if a1.valence_config is None:
                continue
            for j in range(i + 1, n_atoms):
                a2 = self.geometry[j]
                if a2.valence_config is None:
                    continue
                if self._has_same_spin_facing_orbitals(a1, a2):
                    if frozenset((a1.id, a2.id)) not in declared_pairs:
                        raise ValueError(
                            f"Undeclared ferromagnetic coupling between '{a1.id}' and '{a2.id}': "
                            f"facing singly-occupied orbitals carry the same spin, so ('{a1.id}', '{a2.id}') "
                            f"must be declared in ferromagnetic_links."
                        )



    def get_effective_basis_for_atom(self, atom: AtomConfig) -> str:
        """
        Determines the effective basis set for an atom using cascading hierarchy:
        1. Atom-specific basis (atom.basis)
        2. Reference atom's basis if equivalent_to is declared
        3. Element-specific basis (self.elements[sym].basis)
        4. Control default basis (self.control.default_basis)
        5. Deck default basis (self.basis)
        6. Global fallback ('DZ')
        """
        if atom.basis:
            return atom.basis
        if atom.equivalent_to:
            ref = next((a for a in self.geometry if a.id == atom.equivalent_to), None)
            if ref and ref.basis:
                return ref.basis
        sym = atom.element.strip().capitalize()
        user_elem = self.elements.get(sym)
        if user_elem and user_elem.basis:
            return user_elem.basis
        if self.control and self.control.default_basis:
            return self.control.default_basis
        if self.basis:
            return self.basis
        return "DZP"

    def get_atom_element_key(self, atom: AtomConfig) -> str:
        """
        Returns the element key for this atom. If atoms of the same atomic symbol
        use different basis sets, groups are keyed by f"{sym}_{basis}".
        Otherwise, the plain symbol sym is used.
        """
        sym = atom.element.strip().capitalize()
        eff_basis = self.get_effective_basis_for_atom(atom)
        elem_bases = {
            self.get_effective_basis_for_atom(a)
            for a in self.geometry
            if a.element.strip().capitalize() == sym
        }
        if len(elem_bases) > 1:
            return f"{sym}_{eff_basis}"
        return sym

    def resolve_elements(self) -> Dict[str, ElementData]:
        """
        Resolves each atom's element data, pulling from the built-in library
        or applying custom user basis overrides, supporting cascading and mixed basis sets.
        """
        self.ensure_atom_ids()
        keys_ordered: List[str] = []
        key_to_atoms: Dict[str, List[AtomConfig]] = {}
        for a in self.geometry:
            ekey = self.get_atom_element_key(a)
            if ekey not in keys_ordered:
                keys_ordered.append(ekey)
                key_to_atoms[ekey] = []
            key_to_atoms[ekey].append(a)

        resolved: Dict[str, ElementData] = {}
        for ekey in keys_ordered:
            sample_atom = key_to_atoms[ekey][0]
            sym = sample_atom.element.strip().capitalize()
            eff_basis = self.get_effective_basis_for_atom(sample_atom)

            user_elem = self.elements.get(sym)
            lib_elem = get_element_basis(sym, basis_name=eff_basis)

            # Check if any atom in this element group has an embedded_state
            embedded_atom = next(
                (a for a in key_to_atoms[ekey] if a.embedded_state),
                None
            )

            if user_elem and user_elem.functions and (eff_basis == (user_elem.basis or "CUSTOM") or sample_atom.basis is None):
                # Custom user basis functions override library
                funcs = user_elem.functions
                desc = user_elem.description or f"Custom Basis for {sym}"
                z = user_elem.atomic_number or lib_elem.atomic_number
                base_lib = get_element_basis(sym, "DZ") if len(funcs) < len(lib_elem.functions) else lib_elem
                orbs = [
                    o for o in base_lib.default_orbitals
                    if all(int(c[0]) <= len(funcs) for c in o.coefficients)
                ]
                if embedded_atom and hasattr(embedded_atom.embedded_state, "to_initial_orbitals"):
                    orbs = embedded_atom.embedded_state.to_initial_orbitals()
                resolved[ekey] = ElementData(
                    symbol=sym,
                    atomic_number=z,
                    description=desc,
                    basis_name=user_elem.basis or "CUSTOM",
                    functions=funcs,
                    nvterm=lib_elem.nvterm,
                    default_orbitals=orbs,
                )
            elif embedded_atom and hasattr(embedded_atom.embedded_state, "basis_functions") and embedded_atom.embedded_state.basis_functions:
                funcs = embedded_atom.embedded_state.to_basis_functions()
                orbs = embedded_atom.embedded_state.to_initial_orbitals()
                z = embedded_atom.embedded_state.atomic_number or lib_elem.atomic_number
                desc = f"Pre-converged Embedded Atom Basis for {sym}"
                resolved[ekey] = ElementData(
                    symbol=sym,
                    atomic_number=z,
                    description=desc,
                    basis_name="EMBEDDED",
                    functions=funcs,
                    nvterm=lib_elem.nvterm,
                    default_orbitals=orbs,
                )
            else:
                if embedded_atom and hasattr(embedded_atom.embedded_state, "to_initial_orbitals"):
                    orbs = embedded_atom.embedded_state.to_initial_orbitals()
                    resolved[ekey] = ElementData(
                        symbol=sym,
                        atomic_number=lib_elem.atomic_number,
                        description=lib_elem.description,
                        basis_name=lib_elem.basis_name,
                        functions=lib_elem.functions,
                        nvterm=lib_elem.nvterm,
                        default_orbitals=orbs,
                    )
                else:
                    resolved[ekey] = lib_elem

        # User-selected model-potential exponents (per element). Always applied to a copy so
        # shared library ElementData objects are never mutated.
        for ekey, ed in list(resolved.items()):
            user_elem = self.elements.get(ed.symbol.strip().capitalize())
            if user_elem and user_elem.potential_exponents:
                k = len(user_elem.potential_exponents)
                # Term 1 is the 1/r point-charge term; the fit refits every coefficient from
                # cycle 1, so the starting weights are simply uniform over the exponentials.
                terms = [(0, 0, 0.0, 1.0)] + [(0, 0, float(e), -1.0 / k) for e in user_elem.potential_exponents]
                resolved[ekey] = replace(
                    ed,
                    nvterm=k + 1,
                    model_potential_terms=terms,
                    potential_exponent_scale=user_elem.potential_exponent_scale,
                )

        return resolved

    def ensure_atom_ids(self) -> None:
        """Assigns default unique IDs to any atoms lacking one."""
        existing_ids = {a.id for a in self.geometry if a.id}
        elem_counts: Dict[str, int] = {}
        for a in self.geometry:
            sym = a.element.strip().capitalize()
            elem_counts[sym] = elem_counts.get(sym, 0) + 1
            if not a.id:
                cand_id = f"{sym}{elem_counts[sym]}"
                while cand_id in existing_ids:
                    elem_counts[sym] += 1
                    cand_id = f"{sym}{elem_counts[sym]}"
                a.id = cand_id
                existing_ids.add(cand_id)

    def group_atoms_by_equivalency(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Groups atoms into primary reference atoms and their attached equivalent atoms.
        Enforces that reference atoms precede atoms that reference them.
        """
        self.ensure_atom_ids()
        result_by_elem: Dict[str, List[Dict[str, Any]]] = {}

        # Preserve element group encounter order
        for a in self.geometry:
            ekey = self.get_atom_element_key(a)
            if ekey not in result_by_elem:
                result_by_elem[ekey] = []

        for ekey, primary_groups in result_by_elem.items():
            atoms_of_elem = [a for a in self.geometry if self.get_atom_element_key(a) == ekey]
            groups_by_id: Dict[str, Dict[str, Any]] = {}

            # First pass: identify primary reference atoms
            for a in atoms_of_elem:
                if a.equivalent_to is None and a.symmetry is None:
                    grp = {"primary": a, "equivalent": []}
                    primary_groups.append(grp)
                    groups_by_id[a.id] = grp

            # Fallback if all atoms have symmetry (e.g. legacy deck)
            if not primary_groups and atoms_of_elem:
                legacy_candidates = [a for a in atoms_of_elem if a.equivalent_to is None]
                if legacy_candidates:
                    first = legacy_candidates[0]
                    grp = {"primary": first, "equivalent": []}
                    primary_groups.append(grp)
                    groups_by_id[first.id] = grp
                    remaining = [a for a in atoms_of_elem if a is not first]
                else:
                    raise ValueError(
                        f"All atoms of element '{ekey}' declare 'equivalent_to', but no reference atom with that ID was found in element {ekey}."
                    )
            else:
                remaining = [a for a in atoms_of_elem if a.equivalent_to is not None or a.symmetry is not None]

            # Second pass: attach equivalent atoms to their reference
            last_primary = primary_groups[0]["primary"] if primary_groups else None
            for a in remaining:
                if a is primary_groups[0]["primary"]:
                    continue
                if a.equivalent_to:
                    if a.equivalent_to in groups_by_id:
                        groups_by_id[a.equivalent_to]["equivalent"].append(a)
                    else:
                        raise ValueError(
                            f"Atom '{a.id}' has equivalent_to='{a.equivalent_to}', "
                            f"but no reference atom with that ID was found in element group {ekey}."
                        )
                elif last_primary is not None:
                    # Legacy positional equivalence
                    groups_by_id[last_primary.id]["equivalent"].append(a)
                else:
                    grp = {"primary": a, "equivalent": []}
                    primary_groups.append(grp)
                    groups_by_id[a.id] = grp
                    last_primary = a

        return result_by_elem

    def resolve_coordinates(self) -> "CalculationDeck":
        """
        Resolves unset [0, 0, 0] or duplicate coordinates on equivalent atoms by
        applying their symmetry operations sequentially (matching Fortran SYMXYZ: LAST = IATOM - 1).
        """
        self.ensure_atom_ids()
        resolve_atom_coordinates(self.geometry)
        return self

    def to_stdin_text(self, include_coefficients: bool = True) -> str:
        """
        Generates the standard Fortran .stdin text representation from this deck.
        """
        if (
            len(self.geometry) == 2
            and all(getattr(a, "hybridization", None) == "sp" for a in self.geometry)
            and all(not a.embedded_state for a in self.geometry)
        ):
            resolve_atom_coordinates(self.geometry, force_symmetry=True)
            from phatpsy.hybrids import apply_hybrid_seeds
            apply_hybrid_seeds(self, bonds={(0, 1): 3})

        elem_map = self.resolve_elements()
        unique_elems = list(elem_map.keys())
        nelem = len(unique_elems)

        # Count total atoms and basis totals
        natom = len(self.geometry)
        total_orbt = 0
        mxnsto = 0
        mxmsto = 0
        mxnorb = 0
        max_n = 2
        max_l = 1

        for atom in self.geometry:
            ekey = self.get_atom_element_key(atom)
            ed = elem_map[ekey]
            norb_atom = len(ed.default_orbitals) if ed.default_orbitals else ed.msto
            total_orbt += norb_atom
            if ed.nsto > mxnsto:
                mxnsto = ed.nsto
            if ed.msto > mxmsto:
                mxmsto = ed.msto
            if norb_atom > mxnorb:
                mxnorb = norb_atom
            for f in ed.functions:
                if f.n > max_n:
                    max_n = f.n
                if f.l > max_l:
                    max_l = f.l
                if f.l > max_l:
                    max_l = f.l

        ctrl = self.control
        lines = [
            self.title[:80],
            " &CTLINP",
            f"  QOPEN={'T' if ctrl.spin_unrestricted else 'F'},",
            f"  QSEPAT={'T' if ctrl.separate_atoms_first_cycle else 'F'},",
            f"  NELEM={nelem}, NATOM={natom}, NORBT={total_orbt},",
            f"  MXNSTO={mxnsto}, MXMSTO={mxmsto}, MXNORB={mxnorb},",
            f"  NMAX={max(2, max_n)}, LMAX={max(1, max_l)},",
            f"  CONVRG={ctrl.convergence:.1E}, MXCYCL={ctrl.max_cycles}, TIMEND={ctrl.time_limit:.1f}D0,",
            f"  CDAMP={ctrl.charge_damp:.2f}, CACCEL={ctrl.charge_accel:.2f},",
            f"  VDAMP={ctrl.potential_damp:.2f}, VACCEL={ctrl.potential_accel:.2f},",
            f"  PKSCAL={ctrl.pk_scale:.2f},",
            f"  QDEBUG={'T' if ctrl.debug else 'F'}, QTRACE={'T' if ctrl.trace else 'F'},",
        ]

        lines.append(f"  QPSMET={'T' if ctrl.use_pseudometric else 'F'}, PKE={ctrl.pk_energy:.4f},")

        lines.append(f"  QCSTFT={'T' if ctrl.use_constrained_fit else 'F'},")
        lines.append(f"  QGRDFT={'T' if ctrl.use_grid_fit else 'F'}, SMEARW={ctrl.atomic_smearing:.6f},")
        lines.append(f"  QGFERM={'T' if ctrl.use_global_fermi else 'F'},")

        if ctrl.fermi_temperature is not None:
            lines.append(f"  THETA={ctrl.fermi_temperature:.2f},")

        lines.append(f"  QDYNEX={'T' if ctrl.use_dynamic_exponents else 'F'},")

        if ctrl.exchange_kernel.lower() not in ("hf", "hartree-fock"):
            kernel_map = {
                "slater": 1,
                "lda": 1,
                "dirac": 1,
                "becke88": 2,
                "b88": 2,
                "pbe": 3,
            }
            k_type = kernel_map.get(ctrl.exchange_kernel.lower(), 1)
            lines.append("  QDFTEX=T,")
            lines.append(f"  EXCH_TYPE={k_type},")
            lines.append(f"  EXCH_ALPHA={ctrl.exchange_alpha:.6f},")

            corr_map = {
                "pw92": 1,
                "lda": 1,
                "pbe": 2,
                "gga": 2,
            }
            c_type = 0
            if ctrl.correlation_kernel:
                c_type = corr_map.get(ctrl.correlation_kernel.lower(), 0)
            elif ctrl.exchange_kernel.lower() == "pbe":
                c_type = 2
            if c_type > 0:
                lines.append(f"  CORR_TYPE={c_type},")
        else:
            lines.append("  QDFTEX=F,")

        # Append any extra namelist keys
        for k, v in ctrl.extra_namelist.items():
            if isinstance(v, bool):
                val_str = "T" if v else "F"
            elif isinstance(v, float):
                val_str = f"{v:.4f}"
            else:
                val_str = str(v)
            lines.append(f"  {k.upper()}={val_str},")

        lines.append(" &END")

        # Element and basis blocks
        atom_groups_by_elem = self.group_atoms_by_equivalency()

        for sym in unique_elems:
            ed = elem_map[sym]
            norb_elem = len(ed.default_orbitals) if ed.default_orbitals else ed.msto
            lines.append(" &BASDIM")
            lines.append(f"  MSTO={ed.msto}, NSTO={ed.nsto}, NORB={norb_elem}, NVTERM={ed.nvterm},")
            lines.append(" &END")
            lines.append(" &ELEM")

            primary_groups = atom_groups_by_elem.get(sym, [])
            mdifat = len(primary_groups)
            lines.append(f"  LATOM='{ed.symbol}', NUCZ={ed.atomic_number},")
            lines.append(f"  DESCR='{ed.description[:40]}',")
            custom_terms = bool(ed.model_potential_terms)
            elem_flags = f"  MDIFAT={mdifat}, QMNBAS=T, QTFPOT={'F' if custom_terms else 'T'},"
            if custom_terms and ed.potential_exponent_scale is not None:
                elem_flags += f" RSCALE={ed.potential_exponent_scale:.6f},"
            lines.append(elem_flags)
            lines.append(" &END")

            # Basis functions in exact 2i3, f10.6 format
            for f in ed.functions:
                lines.append(f"{f.n:3d}{f.l:3d}{f.zeta:10.5f}")
            if custom_terms:
                # All NVTERM model-potential terms, 2i3, 2f10.6: lv, mv, exponent, starting coefficient
                for lv, mv, expn, coef in ed.model_potential_terms:
                    lines.append(f"{lv:3d}{mv:3d}{expn:10.6f}{coef:10.6f}")
            else:
                lines.append(f"{0:3d}{0:3d}   0.150")

            # Emit atoms for this element
            for group_idx, grp in enumerate(primary_groups):
                p_atom = grp["primary"]
                eq_atoms = grp["equivalent"]
                neqat = len(eq_atoms)
                x, y, z = p_atom.xyz

                flags = []
                if p_atom.fixed:
                    flags.append("QFIXED=T")
                if p_atom.virtual:
                    flags.append("QVIRTL=T")
                has_custom_orbs = bool(p_atom.embedded_state or p_atom.starting_orbitals or p_atom.valence_config)
                # QFLIP is never combined with custom starting orbitals: the engine skips the
                # &ATORB cards for a flipped atom, and custom orbitals already carry their spin.
                if (group_idx > 0 and p_atom.is_spin_down and not has_custom_orbs
                        and p_atom.spin != primary_groups[0]["primary"].spin):
                    flags.append("QFLIP=T")
                if group_idx > 0 and has_custom_orbs:
                    flags.append("QDFVEC=T")
                flag_str = (", " + ", ".join(flags)) if flags else ""

                if p_atom.symmetry is not None and not p_atom.equivalent_to:
                    s = p_atom.symmetry
                    lines.append(f" &ATOM IXYZ={s.ixyz:03d}, ISYM={s.isym}, QFLIP={'T' if s.qflip else 'F'}{flag_str} &END")
                elif neqat > 0:
                    lines.append(f" &ATOM XYZ={x:.5f},{y:.5f},{z:.5f}, NEQAT={neqat}{flag_str} &END")
                else:
                    lines.append(f" &ATOM XYZ={x:.5f},{y:.5f},{z:.5f}{flag_str} &END")

                # Initial orbital cards for the primary atom (overridden by embedded_state if present)
                if group_idx == 0 or has_custom_orbs:
                    orbitals_to_use = ed.default_orbitals
                    # The library's default orbitals are up-polarized (alpha excess).
                    # A spin-down primary atom that uses them must start with alpha and
                    # beta swapped; group-0 atoms cannot use QFLIP because the engine
                    # skips the &ATORB cards when QFLIP is set.  Atoms in later groups
                    # (group_idx > 0) are flipped relative to group 0 via QFLIP above.
                    swap_default_spin = (
                        group_idx == 0 and p_atom.is_spin_down
                        and not p_atom.embedded_state and not p_atom.starting_orbitals and not p_atom.valence_config
                    )
                    if swap_default_spin:
                        orbitals_to_use = [_spin_swapped_orbital(o) for o in ed.default_orbitals]
                    if p_atom.embedded_state:
                        emb = p_atom.embedded_state
                        if isinstance(emb, dict):
                            from phatpsy.state import EmbeddedAtomState
                            emb = EmbeddedAtomState.from_dict(emb)
                        if hasattr(emb, "to_initial_orbitals"):
                            orbitals_to_use = emb.to_initial_orbitals()
                    elif p_atom.starting_orbitals:
                        orbitals_to_use = p_atom.starting_orbitals
                    elif p_atom.valence_config:
                        from phatpsy.hybrids import build_atom_orbitals_from_valence_config
                        atoms_by_id = {a.id: a for a in self.geometry if getattr(a, "id", None)}
                        orbitals_to_use = build_atom_orbitals_from_valence_config(
                            elem=ed,
                            atom=p_atom,
                            atoms_by_id=atoms_by_id,
                        )

                    for orb in orbitals_to_use:
                        parts = ["&ATORB"]
                        if orb.eigval:
                            parts.append(f"EIGVAL={orb.eigval[0]:.4f},{orb.eigval[1]:.4f}")
                        if include_coefficients and orb.coefficients:
                            parts.append(f"NCOEF={len(orb.coefficients)}")
                        if orb.occno:
                            parts.append(f"OCCNO={orb.occno[0]:.2f},{orb.occno[1]:.2f}")
                        parts.append("&END")
                        lines.append(" " + " ".join(parts))

                        if include_coefficients and orb.coefficients:
                            for c_entry in orb.coefficients:
                                if len(c_entry) == 3:
                                    isto_val, ml_val, c1_val = c_entry
                                    c2_val = c1_val
                                else:
                                    isto_val, ml_val, c1_val, c2_val = c_entry[:4]
                                lines.append(f"{isto_val:3d}{ml_val:3d}{c1_val:10.5f}{c2_val:10.5f}")

                for s_atom in eq_atoms:
                    s = s_atom.symmetry
                    has_explicit_sym = (
                        s_atom.symmetry_op is not None
                        or (s is not None and s.isym is not None and s.isym != 0)
                    )
                    if has_explicit_sym:
                        isym_val = s_atom.symmetry_op if s_atom.symmetry_op is not None else s.isym
                        ixyz_val = s_atom.symmetry_axis if s_atom.symmetry_axis is not None else (s.ixyz if s else 1)
                        isym_val = normalize_symmetry_op(isym_val) or -1
                        ixyz_val = normalize_symmetry_axis(ixyz_val) or 1
                    else:
                        isym_val = 0
                        ixyz_val = 0

                    s_flags = []
                    if s_atom.fixed:
                        s_flags.append("QFIXED=T")
                    if s_atom.virtual:
                        s_flags.append("QVIRTL=T")

                    # QFLIP is determined by comparing intrinsic spin with reference atom
                    if s_atom.spin is not None and p_atom.spin is not None:
                        qflip_val = (s_atom.spin != p_atom.spin)
                    elif s is not None:
                        qflip_val = bool(s.qflip or s_atom.is_spin_down)
                    else:
                        qflip_val = bool(s_atom.is_spin_down)

                    s_flag_str = (", " + ", ".join(s_flags)) if s_flags else ""
                    if isym_val == 0:
                        sx, sy, sz = s_atom.xyz if s_atom.xyz and len(s_atom.xyz) == 3 else (0.0, 0.0, 0.0)
                        lines.append(
                            f" &ATOM XYZ={sx:.5f},{sy:.5f},{sz:.5f}, ISYM=0, QFLIP={'T' if qflip_val else 'F'}{s_flag_str} &END"
                        )
                    else:
                        lines.append(
                            f" &ATOM IXYZ={ixyz_val:03d}, ISYM={isym_val}, QFLIP={'T' if qflip_val else 'F'}{s_flag_str} &END"
                        )

        return "\n".join(lines) + "\n"

    def summary(self) -> str:
        """Returns a human-readable summary of the calculation deck."""
        lines = [
            f"CalculationDeck: {self.title}",
            f"Atoms: {len(self.geometry)}, Basis: {self.basis or 'DZ'}, Net Spin: {self.net_spin}",
            f"Spins: {', '.join(f'{a.id}:{a.spin}' for a in self.geometry)}",
        ]
        if self.ferromagnetic_links:
            lines.append(f"Ferromagnetic Links (non-bipartite frustration): {self.ferromagnetic_links}")
            for rep in self.ferromagnetic_reports:
                if hasattr(rep, "atom_pair"):
                    lines.append(
                        f"  - Ring {rep.ring_atoms} (size {rep.ring_size}): "
                        f"parallel spins on {rep.atom_pair} (d = {rep.distance_bohr:.3f} Bohr, {rep.reason})"
                    )
        return "\n".join(lines)

    def assign_alternating_spin(
        self,
        start_spin: str = "up",
        ferromagnetic_links: Optional[List[Tuple[Union[str, int], Union[str, int]]]] = None,
        require_bipartite: bool = False,
        quiet: bool = True,
    ) -> "CalculationDeck":
        """
        Computes and assigns optimal alternating-spin (Néel / Localized Valence-Bond)
        labels ('up' / 'down') to the geometry based on molecular connectivity.
        Non-bipartite rings identify and report their ferromagnetic coupling anomaly.
        """
        from phatpsy.spin import assign_alternating_spin
        self.geometry, reports = assign_alternating_spin(
            self.geometry,
            start_spin=start_spin,
            ferromagnetic_links=ferromagnetic_links,
            require_bipartite=require_bipartite,
            return_reports=True,
            quiet=quiet,
        )
        self.ferromagnetic_reports = reports
        self.ferromagnetic_links = [r.atom_pair for r in reports]
        n_up = sum(1 for a in self.geometry if a.spin == "up")
        n_down = sum(1 for a in self.geometry if a.spin == "down")
        self.net_spin = n_up - n_down
        return self

    def visualize(self, output_path: Optional[Union[str, Path]] = None, auto_open: bool = False, edit_mode: bool = False) -> str:
        """
        Generates an interactive 3D WebGL visualization & parameter editor of this CalculationDeck.
        """
        from phatpsy.visualizer import visualize_deck
        return visualize_deck(self, output_path=output_path, auto_open=auto_open, edit_mode=edit_mode)

    def show(self, height: int = 600, edit_mode: bool = False):
        """
        Displays the interactive 3D visualization inside a Jupyter notebook.
        """
        from phatpsy.visualizer import show_deck
        return show_deck(self, height=height, edit_mode=edit_mode)

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "title": self.title,
            "control": self.control.to_dict(),
            "geometry": [a.to_dict() for a in self.geometry],
            "net_spin": self.net_spin,
        }
        if self.ferromagnetic_links:
            res["ferromagnetic_links"] = [list(link) for link in self.ferromagnetic_links]
        if self.elements:
            res["elements"] = {k: v.to_dict() for k, v in self.elements.items()}
        if self.basis:
            res["basis"] = self.basis
        if self.cell is not None:
            res["cell"] = self.cell
        if self.pbc is not None:
            res["pbc"] = self.pbc
        return res

    def to_yaml(self, path: Optional[Union[str, Path]] = None) -> str:
        text = yaml.dump(self.to_dict(), sort_keys=False, indent=2)
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def to_json(self, path: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent)
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def to_ase(
        self,
        cell: Optional[Any] = None,
        pbc: Optional[Any] = None,
    ) -> Any:
        """
        Exports the CalculationDeck to an ASE Atoms object.
        Preserves coordinates, elements, charges, magnetic moments, constraints, and cell.
        """
        from phatpsy.formats import to_ase
        return to_ase(self, cell=cell, pbc=pbc)

    def to_mol2(self, title: Optional[str] = None) -> str:
        """Exports the CalculationDeck to Tripos Mol2 format text."""
        from phatpsy.formats import to_mol2
        return to_mol2(self, title=title)

    def to_cml(self, title: Optional[str] = None) -> str:
        """Exports the CalculationDeck to Chemical Markup Language (CML) XML text."""
        from phatpsy.formats import to_cml
        return to_cml(self, title=title)

    def to_openbabel(self) -> Any:
        """Exports the CalculationDeck to an OpenBabel / Pybel Molecule object."""
        from phatpsy.formats import to_openbabel
        return to_openbabel(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalculationDeck":
        title = data.get("title", "PHATPSY Calculation")
        control = ControlConfig.from_dict(data.get("control", {}))
        geom_data = data.get("geometry", [])
        geometry = [AtomConfig.from_dict(a) for a in geom_data]

        units_raw = str(data.get("units", "bohr") or "bohr").strip().lower()
        if units_raw in ("angstrom", "angstroms", "aa", "a"):
            for atom in geometry:
                if atom.xyz is not None and not atom.equivalent_to:
                    atom.xyz = [float(c) * BOHR_PER_ANGSTROM for c in atom.xyz]
        elif units_raw not in ("bohr", "bohrs", "au", "a.u."):
            raise ValueError(
                f"Unsupported units '{data.get('units')}' in CalculationDeck; expected 'bohr' or 'angstrom'."
            )

        elem_data = data.get("elements", {})
        elements = {}
        for k, v in elem_data.items():
            if isinstance(v, dict) and "symbol" not in v:
                v["symbol"] = k
            elements[k] = ElementConfig.from_dict(v)
        basis = data.get("basis")
        net_spin = int(data.get("net_spin", 0))
        raw_f_links = data.get("ferromagnetic_links", [])
        ferromagnetic_links = [tuple(link) for link in raw_f_links]
        cell = data.get("cell")
        pbc = data.get("pbc")
        deck = cls(
            title=title,
            control=control,
            geometry=geometry,
            elements=elements,
            basis=basis,
            net_spin=net_spin,
            ferromagnetic_links=ferromagnetic_links,
            cell=cell,
            pbc=pbc,
        )
        deck.resolve_coordinates()
        if (
            len(deck.geometry) == 2
            and bool(deck.elements)
            and all(getattr(a, "hybridization", None) == "sp" for a in deck.geometry)
            and any(not a.starting_orbitals and not a.embedded_state for a in deck.geometry)
        ):
            from phatpsy.hybrids import apply_hybrid_seeds
            apply_hybrid_seeds(deck, bonds={(0, 1): 3})
        return deck

    @classmethod
    def from_yaml(cls, path_or_str: Union[str, Path]) -> "CalculationDeck":
        s = str(path_or_str)
        if "\n" not in s and (Path(s).is_file() or "/" in s or "\\" in s or s.endswith((".yaml", ".yml"))):
            p = Path(s)
            if not p.is_file() and (Path.cwd().parent / s).is_file():
                p = Path.cwd().parent / s
            elif not p.is_file() and (Path(__file__).resolve().parent.parent / s).is_file():
                p = Path(__file__).resolve().parent.parent / s
            if not p.is_file():
                raise FileNotFoundError(f"Calculation deck YAML file not found: {s}")
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            data = yaml.safe_load(s)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML deck format: expected mapping/dictionary, got {type(data).__name__}")
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path_or_str: Union[str, Path]) -> "CalculationDeck":
        s = str(path_or_str)
        if "\n" not in s and (Path(s).is_file() or "/" in s or "\\" in s or s.endswith(".json")):
            p = Path(s)
            if not p.is_file() and (Path.cwd().parent / s).is_file():
                p = Path.cwd().parent / s
            elif not p.is_file() and (Path(__file__).resolve().parent.parent / s).is_file():
                p = Path(__file__).resolve().parent.parent / s
            if not p.is_file():
                raise FileNotFoundError(f"Calculation deck JSON file not found: {s}")
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(s)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid JSON deck format: expected mapping/dictionary, got {type(data).__name__}")
        return cls.from_dict(data)

    @classmethod
    def from_stdin(cls, path_or_str: Union[str, Path]) -> "CalculationDeck":
        s = str(path_or_str)
        p = Path(s)
        if not p.is_file() and (Path.cwd().parent / s).is_file():
            p = Path.cwd().parent / s
        elif not p.is_file() and (Path(__file__).resolve().parent.parent / s).is_file():
            p = Path(__file__).resolve().parent.parent / s
        from phatpsy.io import parse_stdin_to_deck
        return parse_stdin_to_deck(p)

    @classmethod
    def from_geometry(
        cls,
        geometry: Union[str, Path, List[Any]],
        basis: Union[str, Dict[str, str]] = "DZP",
        title: str = "PHATPSY Calculation",
        control: Optional[ControlConfig] = None,
        units: str = "bohr",
        spin_unrestricted: bool = True,
        spins: Optional[Union[List[str], Dict[str, str], str]] = None,
        net_spin: Optional[int] = None,
        extra_namelist: Optional[Dict[str, Any]] = None,
    ) -> "CalculationDeck":
        """
        Builds a CalculationDeck from molecular geometry using Clementi or Thakkar STO basis sets.
        """
        from phatpsy.builder import build_molecule
        return build_molecule(
            geometry=geometry,
            basis=basis,
            title=title,
            control=control,
            units=units,
            spin_unrestricted=spin_unrestricted,
            spins=spins,
            net_spin=net_spin,
            extra_namelist=extra_namelist,
        )

    @classmethod
    def from_diatomic(
        cls,
        element1: str,
        element2: Optional[str] = None,
        bond_length: float = 2.0744,
        basis: Union[str, Dict[str, str]] = "DZP",
        units: str = "bohr",
        use_symmetry: bool = True,
        spins: Optional[Union[List[str], Tuple[str, str]]] = None,
        net_spin: Optional[int] = None,
        spin_flip: Optional[bool] = None,
        title: Optional[str] = None,
        control: Optional[ControlConfig] = None,
        extra_namelist: Optional[Dict[str, Any]] = None,
    ) -> "CalculationDeck":
        """
        Builds a CalculationDeck for a diatomic molecule.
        """
        from phatpsy.builder import build_diatomic
        return build_diatomic(
            element1=element1,
            element2=element2,
            bond_length=bond_length,
            basis=basis,
            units=units,
            use_symmetry=use_symmetry,
            spins=spins,
            net_spin=net_spin,
            spin_flip=spin_flip,
            title=title,
            control=control,
            extra_namelist=extra_namelist,
        )

    @classmethod
    def from_ase(
        cls,
        atoms: Any,
        basis: Union[str, Dict[str, str]] = "DZP",
        title: str = "PHATPSY Calculation (ASE)",
        control: Optional[ControlConfig] = None,
        units: str = "bohr",
        spin_unrestricted: bool = True,
        spins: Optional[Union[List[str], Dict[str, str], str]] = None,
        net_spin: Optional[int] = None,
        extra_namelist: Optional[Dict[str, Any]] = None,
    ) -> "CalculationDeck":
        """
        Builds a CalculationDeck directly from an ASE Atoms object.
        Preserves lattice cell and PBC if present on the ASE Atoms.
        """
        from phatpsy.builder import build_molecule
        from phatpsy.formats import BOHR_PER_ANGSTROM
        import numpy as np

        deck = build_molecule(
            geometry=atoms,
            basis=basis,
            title=title,
            control=control,
            units=units,
            spin_unrestricted=spin_unrestricted,
            spins=spins,
            net_spin=net_spin,
            extra_namelist=extra_namelist,
        )
        if hasattr(atoms, "cell") and np.any(atoms.cell):
            deck.cell = (np.array(atoms.cell, dtype=np.float64) * BOHR_PER_ANGSTROM).tolist()
        if hasattr(atoms, "pbc"):
            deck.pbc = atoms.pbc.tolist() if hasattr(atoms.pbc, "tolist") else list(atoms.pbc)
        return deck

    @classmethod
    def from_mol2(
        cls,
        content_or_path: Any,
        basis: Union[str, Dict[str, str]] = "DZP",
        title: Optional[str] = None,
        control: Optional[ControlConfig] = None,
        spins: Optional[Union[List[str], Dict[str, str], str]] = None,
        net_spin: Optional[int] = None,
        extra_namelist: Optional[Dict[str, Any]] = None,
    ) -> "CalculationDeck":
        """Builds a CalculationDeck from Tripos Mol2 format."""
        from phatpsy.formats import parse_mol2
        from phatpsy.builder import build_molecule
        atoms = parse_mol2(content_or_path)
        return build_molecule(
            geometry=atoms,
            basis=basis,
            title=title or "Mol2 Molecule",
            control=control,
            spins=spins,
            net_spin=net_spin,
            extra_namelist=extra_namelist,
        )

    @classmethod
    def from_cml(
        cls,
        content_or_path: Any,
        basis: Union[str, Dict[str, str]] = "DZP",
        title: Optional[str] = None,
        control: Optional[ControlConfig] = None,
        spins: Optional[Union[List[str], Dict[str, str], str]] = None,
        net_spin: Optional[int] = None,
        extra_namelist: Optional[Dict[str, Any]] = None,
    ) -> "CalculationDeck":
        """Builds a CalculationDeck from Chemical Markup Language (CML)."""
        from phatpsy.formats import parse_cml
        from phatpsy.builder import build_molecule
        atoms = parse_cml(content_or_path)
        return build_molecule(
            geometry=atoms,
            basis=basis,
            title=title or "CML Molecule",
            control=control,
            spins=spins,
            net_spin=net_spin,
            extra_namelist=extra_namelist,
        )

    @classmethod
    def from_openbabel(
        cls,
        mol: Any,
        basis: Union[str, Dict[str, str]] = "DZP",
        title: Optional[str] = None,
        control: Optional[ControlConfig] = None,
        spins: Optional[Union[List[str], Dict[str, str], str]] = None,
        net_spin: Optional[int] = None,
        extra_namelist: Optional[Dict[str, Any]] = None,
    ) -> "CalculationDeck":
        """Builds a CalculationDeck from an OpenBabel / Pybel Molecule."""
        from phatpsy.formats import from_openbabel
        from phatpsy.builder import build_molecule
        atoms = from_openbabel(mol)
        return build_molecule(
            geometry=atoms,
            basis=basis,
            title=title or "OpenBabel Molecule",
            control=control,
            spins=spins,
            net_spin=net_spin,
            extra_namelist=extra_namelist,
        )

