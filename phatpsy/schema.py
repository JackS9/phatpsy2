# phatpsy/schema.py
"""
Modern YAML and JSON input schemas and serializers for PHATPSY calculations.
Provides two-way conversion between high-level Python/YAML/JSON representations
and legacy Fortran stdin decks.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
import yaml
import json
import re
import math
from phatpsy.basis import get_element_basis, BasisFunction, ElementData


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
    "charge_damp",
    "charge_accel",
    "potential_damp",
    "potential_accel",
    "pk_scale",
    "use_pseudometric",
    "pk_energy",
}


@dataclass
class ControlConfig:
    spin_unrestricted: bool = True
    max_cycles: int = 100
    convergence: float = 1.0e-4
    time_limit: float = 360.0
    charge_damp: float = 0.5
    charge_accel: float = 1.0
    potential_damp: float = 0.5
    potential_accel: float = 1.0
    pk_scale: float = 1.0
    use_pseudometric: bool = True
    pk_energy: float = 0.0
    use_constrained_fit: bool = False
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
            "use_diis", "use_global_fermi", "fermi_temperature",
            "use_dynamic_exponents", "exchange_kernel", "exchange_alpha",
            "correlation_kernel", "use_dispersion", "dispersion_model",
            "default_basis", "separate_atoms_first_cycle", "debug", "trace"
        }
        kwargs = {k: v for k, v in data.items() if k in valid_keys}
        extra = {k: v for k, v in data.items() if k not in valid_keys}
        return cls(**kwargs, extra_namelist=extra)

    @classmethod
    def first_principles(
        cls,
        pk_scale: float = 1.0,
        use_diis: bool = True,
        use_constrained_fit: bool = True,
        use_dynamic_exponents: bool = True,
        use_global_fermi: bool = False,
        use_pseudometric: bool = True,
        charge_damp: float = 0.5,
        potential_damp: float = 0.5,
        default_basis: Optional[str] = None,
        **kwargs,
    ) -> "ControlConfig":
        """
        Creates a first-principles ControlConfig locking PKSCAL=1.0 and retiring
        empirical damping knobs in favor of self-consistent DIIS acceleration,
        constrained LAPACK cusp fitting, and dynamic STO radial exponents.
        """
        return cls(
            pk_scale=pk_scale,
            use_diis=use_diis,
            use_constrained_fit=use_constrained_fit,
            use_dynamic_exponents=use_dynamic_exponents,
            use_global_fermi=use_global_fermi,
            use_pseudometric=use_pseudometric,
            charge_damp=charge_damp,
            potential_damp=potential_damp,
            default_basis=default_basis,
            **kwargs,
        )


@dataclass
class SymmetrySpec:
    ixyz: int
    isym: int = -1
    qflip: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
    embedded_state: Optional[Any] = None
    starting_orbitals: Optional[List[Any]] = None

    def __post_init__(self):
        # 1. Normalize spin
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

        # 2. Reconcile / validate against starting_orbitals if present
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

        # 2. Synchronize legacy SymmetrySpec with modern symmetry_op / symmetry_axis
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

    @property
    def is_spin_down(self) -> bool:
        return self.spin == "down" or self.spin_flip

    @property
    def is_spin_up(self) -> bool:
        return self.spin == "up"

    @property
    def net_orbital_spin(self) -> Optional[float]:
        """Returns sum(alpha_occ) - sum(beta_occ) across starting orbitals, if defined."""
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

        starting_orbs = None
        if "starting_orbitals" in data and data["starting_orbitals"]:
            from phatpsy.basis import InitialOrbital
            starting_orbs = [
                (InitialOrbital.from_dict(o) if isinstance(o, dict) else o)
                for o in data["starting_orbitals"]
            ]
            if spin_val is None:
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
        )


def resolve_atom_coordinates(atoms: List[AtomConfig]) -> List[AtomConfig]:
    """
    Resolves unset [0, 0, 0] or duplicate coordinates on equivalent atoms by
    applying their symmetry operations sequentially (matching Fortran SYMXYZ: LAST = IATOM - 1).
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

                if source and (a.xyz == [0.0, 0.0, 0.0] or a.xyz == source.xyz):
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

    @property
    def atoms(self) -> List[AtomConfig]:
        """Alias for self.geometry."""
        return self.geometry

    @atoms.setter
    def atoms(self, val: List[AtomConfig]) -> None:
        self.geometry = val

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
        return "DZ"

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
                orbs = lib_elem.default_orbitals
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
            lines.append(f"  MDIFAT={mdifat}, QMNBAS=T, QTFPOT=T,")
            lines.append(" &END")

            # Basis functions in exact 2i3, f10.6 format
            for f in ed.functions:
                lines.append(f"{f.n:3d}{f.l:3d}{f.zeta:10.5f}")
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
                if p_atom.is_spin_down:
                    flags.append("QFLIP=T")
                has_custom_orbs = bool(p_atom.embedded_state or p_atom.starting_orbitals)
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
                    if p_atom.embedded_state:
                        emb = p_atom.embedded_state
                        if isinstance(emb, dict):
                            from phatpsy.state import EmbeddedAtomState
                            emb = EmbeddedAtomState.from_dict(emb)
                        if hasattr(emb, "to_initial_orbitals"):
                            orbitals_to_use = emb.to_initial_orbitals()
                    elif p_atom.starting_orbitals:
                        orbitals_to_use = p_atom.starting_orbitals

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
                    isym_val = s_atom.symmetry_op if s_atom.symmetry_op is not None else (s.isym if s else -1)
                    ixyz_val = s_atom.symmetry_axis if s_atom.symmetry_axis is not None else (s.ixyz if s else 1)
                    isym_val = normalize_symmetry_op(isym_val) or -1
                    ixyz_val = normalize_symmetry_axis(ixyz_val) or 1

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
                    lines.append(f" &ATOM IXYZ={ixyz_val:03d}, ISYM={isym_val}, QFLIP={'T' if qflip_val else 'F'}{s_flag_str} &END")

        return "\n".join(lines) + "\n"

    def assign_alternating_spin(self, start_spin: str = "up") -> "CalculationDeck":
        """
        Computes and assigns optimal alternating-spin (Néel / Localized Valence-Bond)
        labels ('up' / 'down') to the geometry based on molecular connectivity.
        """
        from phatpsy.spin import assign_alternating_spin
        self.geometry = assign_alternating_spin(self.geometry, start_spin=start_spin)
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
        }
        if self.elements:
            res["elements"] = {k: v.to_dict() for k, v in self.elements.items()}
        if self.basis:
            res["basis"] = self.basis
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalculationDeck":
        title = data.get("title", "PHATPSY Calculation")
        control = ControlConfig.from_dict(data.get("control", {}))
        geom_data = data.get("geometry", [])
        geometry = [AtomConfig.from_dict(a) for a in geom_data]

        elem_data = data.get("elements", {})
        elements = {}
        for k, v in elem_data.items():
            if isinstance(v, dict) and "symbol" not in v:
                v["symbol"] = k
            elements[k] = ElementConfig.from_dict(v)
        basis = data.get("basis")
        deck = cls(title=title, control=control, geometry=geometry, elements=elements, basis=basis)
        deck.resolve_coordinates()
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
        basis: Union[str, Dict[str, str]] = "DZ",
        title: str = "PHATPSY Calculation",
        control: Optional[ControlConfig] = None,
        units: str = "bohr",
        spin_unrestricted: bool = True,
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
            extra_namelist=extra_namelist,
        )

    @classmethod
    def from_diatomic(
        cls,
        element1: str,
        element2: Optional[str] = None,
        bond_length: float = 2.068,
        basis: Union[str, Dict[str, str]] = "DZ",
        units: str = "bohr",
        use_symmetry: bool = True,
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
            spin_flip=spin_flip,
            title=title,
            control=control,
            extra_namelist=extra_namelist,
        )
