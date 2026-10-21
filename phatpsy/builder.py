# phatpsy/builder.py
"""
High-level molecule builder for PHATPSY calculations.
Constructs CalculationDecks from standard Cartesian coordinates, XYZ files,
or diatomic specifications, automatically binding Clementi & Roetti and Thakkar
STO basis functions, exponents, and starting orbital configurations.
"""
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import re

from phatpsy.schema import (
    CalculationDeck,
    ControlConfig,
    AtomConfig,
    ElementConfig,
    SymmetrySpec,
)
from phatpsy.basis import list_available_elements, get_element_basis

BOHR_PER_ANGSTROM: float = 1.8897261245650618


def parse_geometry(
    geometry: Union[str, Path, List[Any]],
    units: str = "bohr",
) -> List[AtomConfig]:
    """
    Parses molecular geometry from various Python representations or XYZ files.

    Supported inputs:
    - Path or file path to an .xyz file
    - Multi-line XYZ string or list of lines
    - List of tuples: [("C", [0.0, 0.0, 0.0]), ("O", [0.0, 0.0, 2.13])]
    - List of 4-tuples: [("C", 0.0, 0.0, 0.0), ("O", 0.0, 0.0, 2.13)]
    - List of dicts: [{"element": "C", "xyz": [0, 0, 0]}, ...]
    - List of AtomConfig objects

    Args:
        geometry: Input geometry representation.
        units: 'bohr' (default, atomic units) or 'angstrom' / 'aa'.

    Returns:
        List of AtomConfig objects with Cartesian coordinates in Bohr.
    """
    to_bohr = BOHR_PER_ANGSTROM if units.lower() in ("angstrom", "angstroms", "aa", "a") else 1.0

    # 1. Path or filename
    if isinstance(geometry, (str, Path)):
        s_path = Path(str(geometry))
        if s_path.is_file():
            content = s_path.read_text(encoding="utf-8")
            return _parse_xyz_string(content, to_bohr)
        elif isinstance(geometry, str) and ("\n" in geometry or ";" in geometry):
            return _parse_xyz_string(geometry, to_bohr)
        else:
            raise ValueError(f"Geometry string could not be resolved as a file or XYZ block: {geometry}")

    # 2. List of inputs
    if not isinstance(geometry, (list, tuple)):
        raise TypeError(f"Unsupported geometry type: {type(geometry)}")

    atom_configs: List[AtomConfig] = []
    for item in geometry:
        if isinstance(item, AtomConfig):
            # Scale coordinates if requested
            if to_bohr != 1.0:
                xyz_scaled = [c * to_bohr for c in item.xyz]
                atom_configs.append(AtomConfig(
                    element=item.element,
                    xyz=xyz_scaled,
                    symmetry=item.symmetry,
                    charge=item.charge,
                    fixed=item.fixed,
                    virtual=item.virtual,
                    spin_flip=item.spin_flip,
                    spin=item.spin,
                ))
            else:
                atom_configs.append(item)

        elif isinstance(item, dict):
            raw_sym = str(item.get("element", item.get("symbol", ""))).strip()
            sym, is_virt, is_fix, spin_tag = _extract_atom_modifiers(raw_sym)
            xyz = item.get("xyz", [0.0, 0.0, 0.0])
            xyz_b = [float(c) * to_bohr for c in xyz]
            symm = item.get("symmetry")
            symm_obj = SymmetrySpec(**symm) if isinstance(symm, dict) else symm
            spin_val = item.get("spin", spin_tag)
            spin_flip_val = bool(item.get("spin_flip", spin_val == "down"))
            atom_configs.append(AtomConfig(
                element=sym,
                xyz=xyz_b,
                symmetry=symm_obj,
                charge=float(item.get("charge", 0.0)),
                fixed=bool(item.get("fixed", is_fix)),
                virtual=bool(item.get("virtual", is_virt)),
                spin_flip=spin_flip_val,
                spin=spin_val,
            ))

        elif isinstance(item, (list, tuple)):
            if len(item) == 2:
                # ("C", [x, y, z]) or ("C", (x, y, z))
                raw_sym = str(item[0]).strip()
                sym, is_virt, is_fix, spin_tag = _extract_atom_modifiers(raw_sym)
                coords = [float(c) * to_bohr for c in item[1]]
                if len(coords) != 3:
                    raise ValueError(f"Expected 3 coordinates for atom {sym}, got {len(coords)}")
                atom_configs.append(AtomConfig(
                    element=sym,
                    xyz=coords,
                    fixed=is_fix,
                    virtual=is_virt,
                    spin_flip=(spin_tag == "down"),
                    spin=spin_tag,
                ))
            elif len(item) == 3 and isinstance(item[2], (dict, str)):
                # ("C", [x, y, z], {...}) or ("C", [x, y, z], "virtual")
                raw_sym = str(item[0]).strip()
                sym, is_virt, is_fix, spin_tag = _extract_atom_modifiers(raw_sym)
                coords = [float(c) * to_bohr for c in item[1]]
                opts = item[2]
                if isinstance(opts, str):
                    opt_lower = opts.lower()
                    is_virt = is_virt or ("virt" in opt_lower or "ghost" in opt_lower)
                    is_fix = is_fix or ("fix" in opt_lower)
                    if opt_lower in ("down", "beta", "flip", "-"):
                        spin_tag = "down"
                    elif opt_lower in ("up", "alpha", "+"):
                        spin_tag = "up"
                    symm_obj = None
                    charge = 0.0
                else:
                    is_virt = bool(opts.get("virtual", is_virt))
                    is_fix = bool(opts.get("fixed", is_fix))
                    spin_tag = opts.get("spin", spin_tag)
                    if opts.get("spin_flip"):
                        spin_tag = "down"
                    charge = float(opts.get("charge", 0.0))
                    symm = opts.get("symmetry")
                    symm_obj = SymmetrySpec(**symm) if isinstance(symm, dict) else symm
                atom_configs.append(AtomConfig(
                    element=sym,
                    xyz=coords,
                    symmetry=symm_obj,
                    charge=charge,
                    fixed=is_fix,
                    virtual=is_virt,
                    spin_flip=(spin_tag == "down"),
                    spin=spin_tag,
                ))
            elif len(item) == 4:
                # ("C", x, y, z)
                raw_sym = str(item[0]).strip()
                sym, is_virt, is_fix, spin_tag = _extract_atom_modifiers(raw_sym)
                coords = [float(item[1]) * to_bohr, float(item[2]) * to_bohr, float(item[3]) * to_bohr]
                atom_configs.append(AtomConfig(
                    element=sym,
                    xyz=coords,
                    fixed=is_fix,
                    virtual=is_virt,
                    spin_flip=(spin_tag == "down"),
                    spin=spin_tag,
                ))
            else:
                raise ValueError(f"Unrecognized atom tuple format with length {len(item)}: {item}")
        else:
            raise TypeError(f"Unrecognized atom item in geometry list: {item}")

    return atom_configs


def _extract_atom_modifiers(raw_sym: str) -> Tuple[str, bool, bool, Optional[str]]:
    """Helper to detect ghost/virtual, fixed, or spin-flipped prefixes on element symbols."""
    is_virt = False
    is_fix = False
    spin = None
    clean = raw_sym.strip()

    if re.match(r"^(gh\.|x\.|bq\.|ghost[-_])", clean, re.IGNORECASE):
        is_virt = True
        clean = re.sub(r"^(gh\.|x\.|bq\.|ghost[-_])", "", clean, flags=re.IGNORECASE)
    elif re.match(r"^(fixed[\.\-_])", clean, re.IGNORECASE):
        is_fix = True
        clean = re.sub(r"^(fixed[\.\-_])", "", clean, flags=re.IGNORECASE)
    elif re.match(r"^(flip[\.\-_]|down[\.\-_]|beta[\.\-_])", clean, re.IGNORECASE):
        spin = "down"
        clean = re.sub(r"^(flip[\.\-_]|down[\.\-_]|beta[\.\-_])", "", clean, flags=re.IGNORECASE)
    elif re.match(r"^(up[\.\-_]|alpha[\.\-_])", clean, re.IGNORECASE):
        spin = "up"
        clean = re.sub(r"^(up[\.\-_]|alpha[\.\-_])", "", clean, flags=re.IGNORECASE)

    return clean.capitalize(), is_virt, is_fix, spin


def _parse_xyz_string(text: str, to_bohr: float) -> List[AtomConfig]:
    """Helper to parse raw XYZ text format with optional modifiers."""
    lines = [line.strip() for line in text.replace(";", "\n").splitlines() if line.strip()]
    if not lines:
        return []

    start_idx = 0
    if lines[0].isdigit():
        start_idx = 2  # skip count and comment line

    atoms: List[AtomConfig] = []
    for line in lines[start_idx:]:
        tokens = line.split()
        if len(tokens) >= 4:
            sym, is_virt, is_fix, spin_tag = _extract_atom_modifiers(tokens[0])
            if not sym.isalpha():
                continue
            x = float(tokens[1]) * to_bohr
            y = float(tokens[2]) * to_bohr
            z = float(tokens[3]) * to_bohr

            if len(tokens) >= 5:
                tag = tokens[4].strip().lower()
                if "virt" in tag or "ghost" in tag:
                    is_virt = True
                if "fix" in tag:
                    is_fix = True
                if tag in ("down", "beta", "flip", "-"):
                    spin_tag = "down"
                elif tag in ("up", "alpha", "+"):
                    spin_tag = "up"

            atoms.append(AtomConfig(
                element=sym,
                xyz=[x, y, z],
                fixed=is_fix,
                virtual=is_virt,
                spin_flip=(spin_tag == "down"),
                spin=spin_tag,
            ))

    return atoms


def build_molecule(
    geometry: Union[str, Path, List[Any]],
    basis: Union[str, Dict[str, str]] = "DZ",
    title: str = "PHATPSY Calculation",
    control: Optional[ControlConfig] = None,
    units: str = "bohr",
    spin_unrestricted: bool = True,
    auto_spin_flip: bool = False,
    extra_namelist: Optional[Dict[str, Any]] = None,
) -> CalculationDeck:
    """
    Builds a complete, validated CalculationDeck for any molecule.

    Args:
        geometry: Atoms and coordinates (list of tuples/dicts, or XYZ text/file).
        basis: Global basis set name ('DZ', 'SZ', 'THAKKAR') or per-element dict
               (e.g. {'C': 'DZ', 'H': 'THAKKAR'}).
        title: Title for the calculation deck.
        control: Optional ControlConfig instance.
        units: 'bohr' (default) or 'angstrom'.
        spin_unrestricted: True for UHF (QOPEN=T), False for RHF.
        auto_spin_flip: If True, automatically assigns alternating spin ('up'/'down')
                        using molecular graph connectivity for Localized Valence-Bond pairing.
        extra_namelist: Additional &CTLINP parameters.

    Returns:
        A ready-to-run CalculationDeck.
    """
    atoms = parse_geometry(geometry, units=units)
    if not atoms:
        raise ValueError("Cannot build molecule deck from empty geometry.")

    if control is None:
        ctrl_extra = dict(extra_namelist or {})
        control = ControlConfig(
            spin_unrestricted=spin_unrestricted,
            extra_namelist=ctrl_extra,
        )
    elif extra_namelist:
        control.extra_namelist.update(extra_namelist)

    # Determine unique elements and populate ElementConfigs
    unique_symbols = []
    for atom in atoms:
        if atom.element not in unique_symbols:
            unique_symbols.append(atom.element)

    elements_dict: Dict[str, ElementConfig] = {}
    for sym in unique_symbols:
        if isinstance(basis, dict):
            b_name = basis.get(sym, "DZ")
        else:
            b_name = basis

        ed = get_element_basis(sym, basis_name=b_name)
        elements_dict[sym] = ElementConfig(
            symbol=sym,
            atomic_number=ed.atomic_number,
            basis=ed.basis_name,
            description=ed.description,
            functions=ed.functions,
        )

    deck = CalculationDeck(
        title=title,
        control=control,
        geometry=atoms,
        elements=elements_dict,
    )
    if auto_spin_flip:
        deck.assign_alternating_spin()
    return deck


def build_diatomic(
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
) -> CalculationDeck:
    """
    Convenience builder for diatomic molecules.

    In PHATPSY methodology, spin-flipping (QFLIP=T) is heavily utilized even
    for nominally 'closed-shell' molecules to set up an alternating-spin
    Perfect-Pairing Valence-Bond (PP-VB) representation. By swapping alpha and
    beta spin roles across bonded atom centers, electrons pair across the bond
    in localized, atomic-like channels with zero net molecular spin, preventing
    the spurious ionic configurations of standard single-determinant RHF.

    Args:
        element1: Symbol of the first atom (e.g. 'N', 'C', 'Ag').
        element2: Symbol of second atom. Defaults to element1 (homodiatomic).
        bond_length: Interatomic distance (default: 2.068 Bohr for N2).
        basis: Basis set ('DZ', 'SZ', 'THAKKAR').
        units: 'bohr' or 'angstrom'.
        use_symmetry: If True and homodiatomic, use reflection/inversion symmetry card.
        spin_flip: Whether to spin-flip the second atom (QFLIP=T). Defaults to True
                   for homonuclear diatomics to generate the alternating-spin PP-VB state.
        title: Deck title.
        control: ControlConfig options.
        extra_namelist: Additional namelist options.

    Returns:
        CalculationDeck for the diatomic system.
    """
    sym1 = element1.strip().capitalize()
    sym2 = (element2 or element1).strip().capitalize()

    to_bohr = BOHR_PER_ANGSTROM if units.lower() in ("angstrom", "angstroms", "aa", "a") else 1.0
    r_bohr = bond_length * to_bohr

    # For homonuclear diatomics, default to alternating-spin PP-VB representation
    is_homo = (sym1 == sym2)
    do_flip = is_homo if spin_flip is None else bool(spin_flip)

    if title is None:
        title = f"{sym1}{sym2} Diatomic (R={r_bohr:.3f} Bohr)"

    if is_homo and use_symmetry:
        # Homonuclear diatomic with symmetry along Z axis
        z_half = r_bohr / 2.0
        geometry = [
            AtomConfig(element=sym1, xyz=[0.0, 0.0, z_half], spin="up"),
            AtomConfig(
                element=sym1,
                xyz=[0.0, 0.0, 0.0],
                symmetry=SymmetrySpec(ixyz=1, isym=-1, qflip=do_flip),
                spin_flip=do_flip,
                spin="down" if do_flip else "up",
            ),
        ]
    else:
        # Heteronuclear or explicit coordinates
        geometry = [
            AtomConfig(element=sym1, xyz=[0.0, 0.0, 0.0], spin="up" if is_homo else None),
            AtomConfig(element=sym2, xyz=[0.0, 0.0, r_bohr], spin_flip=do_flip, spin="down" if do_flip else None),
        ]

    return build_molecule(
        geometry=geometry,
        basis=basis,
        title=title,
        control=control,
        units="bohr",
        extra_namelist=extra_namelist,
    )
