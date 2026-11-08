# phatpsy/builder.py
"""
High-level molecule builder for PHATPSY calculations.
Constructs CalculationDecks from standard Cartesian coordinates, XYZ files,
or diatomic specifications, automatically binding Clementi & Roetti and Thakkar
STO basis functions, exponents, and starting orbital configurations.
"""
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union, Sequence
import re
import numpy as np

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

    # 0. ASE Atoms object or duck-typed structure
    if hasattr(geometry, "get_positions") and hasattr(geometry, "get_chemical_symbols"):
        from phatpsy.formats import from_ase
        return from_ase(geometry, units=units)

    # 1. Path or filename
    if isinstance(geometry, (str, Path)):
        from phatpsy.formats import (
            detect_format,
            parse_pdb,
            parse_gaussian,
            parse_poscar,
            parse_mol2,
            parse_cml,
            parse_ase,
            HAS_ASE,
        )
        s_str = str(geometry)
        is_path_file = False
        if "\n" not in s_str and "\r" not in s_str:
            try:
                is_path_file = Path(s_str).is_file()
            except (OSError, ValueError):
                is_path_file = False

        fmt = detect_format(geometry)
        if fmt == "pdb":
            return parse_pdb(geometry, units=units)
        elif fmt == "gaussian":
            atoms, _, _, _ = parse_gaussian(geometry, units=units)
            return atoms
        elif fmt == "poscar":
            atoms, _, _ = parse_poscar(geometry, units=units)
            return atoms
        elif fmt == "mol2":
            return parse_mol2(geometry, units=units)
        elif fmt == "cml":
            return parse_cml(geometry, units=units)
        elif fmt in ("cif", "gro", "mol", "sdf", "gen", "car", "arc", "res", "cell", "extxyz", "lammps-data", "ase"):
            return parse_ase(geometry, format=fmt if fmt not in ("ase", "lammps-data") else None, units=units)
        elif is_path_file:
            content = Path(s_str).read_text(encoding="utf-8")
            xyz_atoms = _parse_xyz_string(content, to_bohr)
            if xyz_atoms:
                return xyz_atoms
            if HAS_ASE:
                try:
                    return parse_ase(Path(s_str), units=units)
                except Exception:
                    pass
            raise ValueError(f"Geometry file could not be parsed: {geometry}")
        elif isinstance(geometry, str) and ("\n" in geometry or ";" in geometry):
            # Check if string is Mol2 format
            if fmt == "mol2" or "@<TRIPOS>" in geometry:
                return parse_mol2(geometry, units=units)
            # Check if string is CML format
            if fmt == "cml" or "<molecule" in geometry.lower() or "<cml" in geometry.lower() or "xml-cml.org" in geometry.lower():
                return parse_cml(geometry, units=units)
            # Check if string is CIF format
            if HAS_ASE and (fmt == "cif" or "data_" in geometry[:50]):
                try:
                    return parse_ase(geometry, format="cif", units=units)
                except Exception:
                    pass
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
                    id=item.id,
                    xyz=xyz_scaled,
                    equivalent_to=item.equivalent_to,
                    symmetry_op=item.symmetry_op,
                    symmetry_axis=item.symmetry_axis,
                    symmetry=item.symmetry,
                    charge=item.charge,
                    fixed=item.fixed,
                    virtual=item.virtual,
                    spin_flip=item.spin_flip,
                    spin=item.spin,
                    basis=item.basis,
                    hybridization=item.hybridization,
                    embedded_state=item.embedded_state,
                    starting_orbitals=item.starting_orbitals,
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
                id=item.get("id"),
                xyz=xyz_b,
                equivalent_to=item.get("equivalent_to", item.get("eq")),
                symmetry_op=item.get("symmetry_op", item.get("sym")),
                symmetry_axis=item.get("symmetry_axis", item.get("axis")),
                symmetry=symm_obj,
                charge=float(item.get("charge", 0.0)),
                fixed=bool(item.get("fixed", item.get("is_fixed", is_fix))),
                virtual=bool(item.get("virtual", item.get("is_virtual", is_virt))),
                spin_flip=spin_flip_val,
                spin=spin_val,
                basis=item.get("basis"),
                hybridization=item.get("hybridization"),
                embedded_state=item.get("embedded_state"),
                starting_orbitals=item.get("starting_orbitals"),
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
                basis_val = None
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
                    basis_val = opts.get("basis")
                atom_configs.append(AtomConfig(
                    element=sym,
                    xyz=coords,
                    symmetry=symm_obj,
                    charge=charge,
                    fixed=is_fix,
                    virtual=is_virt,
                    spin_flip=(spin_tag == "down"),
                    spin=spin_tag,
                    basis=basis_val,
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


def _normalize_spin_entry(entry: Any) -> Tuple[Optional[str], Optional[List[str]]]:
    """Normalizes a spin entry (either a single spin string or a sequence of per-orbital spins)."""
    spin_map = {
        "up": "up", "alpha": "up", "+1": "up", "1": "up", "+": "up",
        "down": "down", "beta": "down", "-1": "down", "-": "down",
    }
    if isinstance(entry, (list, tuple)):
        if not entry:
            return None, None
        orb_list: List[str] = []
        for item in entry:
            if item is None:
                return None, None
            s_norm = spin_map.get(str(item).strip().lower())
            if s_norm is None:
                return None, None
            orb_list.append(s_norm)
        n_up = sum(1 for x in orb_list if x == "up")
        n_down = len(orb_list) - n_up
        maj = "up" if n_up >= n_down else "down"
        return maj, orb_list
    if entry is None:
        return None, None
    s_norm = spin_map.get(str(entry).strip().lower())
    return s_norm, None


def build_molecule(
    geometry: Union[str, Path, List[Any]],
    basis: Union[str, Dict[str, str]] = "DZ",
    title: Optional[str] = None,
    control: Optional[ControlConfig] = None,
    units: str = "bohr",
    spin_unrestricted: bool = True,
    spins: Optional[Union[List[Any], Dict[str, Any], str]] = None,
    net_spin: Optional[int] = None,
    ferromagnetic_links: Optional[List[Tuple[Union[str, int], Union[str, int]]]] = None,
    auto_spin_flip: Optional[bool] = None,
    require_bipartite: bool = False,
    embedded_atoms: Optional[Dict[Union[str, int], Any]] = None,
    hybridization: Optional[Union[str, Sequence[Optional[str]], Dict[str, str]]] = None,
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
        spins: Explicit per-atom spin assignments ('up' or 'down'). Can be a list of strings
               matching atoms, a dict of {atom_id: 'up'/'down'}, or a uniform string.
        net_spin: Integer (n_up - n_down), default 0. Asserts (n_up - n_down) == net_spin.
        ferromagnetic_links: Optional list of bond pairs [(id1, id2), ...] selected as
                             ferromagnetic couplings in frustrated (non-bipartite) rings.
        auto_spin_flip: [DEPRECATED] Automated alternating spin guessing. Identifies and
                        reports ferromagnetic couplings, or raises if require_bipartite=True.
        require_bipartite: If True, strictly refuses non-bipartite graphs with NonBipartiteSpinGraphError.
        embedded_atoms: Optional mapping of atom ID, index, or element to EmbeddedAtomState
                        or dict, overriding default basis starting orbitals.
        extra_namelist: Additional &CTLINP parameters.

    Returns:
        A ready-to-run CalculationDeck.
    """
    import warnings
    parsed_title = ""
    poscar_cell = None
    poscar_pbc = None

    if hasattr(geometry, "get_positions") and hasattr(geometry, "get_chemical_symbols"):
        if not title:
            parsed_title = "PHATPSY Calculation (ASE)"
        try:
            magmoms = geometry.get_initial_magnetic_moments()
            n_up = sum(1 for m in magmoms if m > 0)
            n_down = sum(1 for m in magmoms if m < 0)
            if net_spin is None and (n_up > 0 or n_down > 0):
                net_spin = n_up - n_down
            if spins is None and (n_up > 0 or n_down > 0):
                if all(m != 0 for m in magmoms):
                    spins = ["up" if m > 0 else "down" for m in magmoms]
        except Exception:
            pass

    elif isinstance(geometry, (str, Path)):
        from phatpsy.formats import detect_format, parse_gaussian, parse_poscar
        fmt = detect_format(geometry)
        if fmt == "gaussian":
            _, g_charge, g_mult, parsed_title = parse_gaussian(geometry, units=units)
            if net_spin is None and g_mult > 1:
                net_spin = g_mult - 1
        elif fmt == "poscar":
            _, poscar_cell, poscar_title = parse_poscar(geometry, units=units)
            parsed_title = poscar_title
            poscar_pbc = [True, True, True]

    if not title:
        title = parsed_title or "PHATPSY Calculation"

    atoms = parse_geometry(geometry, units=units)
    if not atoms:
        raise ValueError("Cannot build molecule deck from empty geometry.")

    elem_counts: Dict[str, int] = {}
    for a in atoms:
        sym = a.element.strip().capitalize()
        elem_counts[sym] = elem_counts.get(sym, 0) + 1
        if not a.id:
            a.id = f"{sym}{elem_counts[sym]}"

    detected_ferromagnetic_reports: List[Any] = []
    detected_ferromagnetic_links: List[Tuple[str, str]] = []

    # Deprecated auto_spin_flip shim
    if auto_spin_flip is not None:
        warnings.warn(
            "The 'auto_spin_flip' parameter in build_molecule is deprecated and will be removed in a future release. "
            "Specify explicit per-atom spins or pass 'spins' parameter instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if auto_spin_flip and spins is None:
            from phatpsy.spin import assign_alternating_spin
            atoms, reports = assign_alternating_spin(
                atoms,
                ferromagnetic_links=ferromagnetic_links,
                require_bipartite=require_bipartite,
                return_reports=True,
            )
            detected_ferromagnetic_reports = reports
            detected_ferromagnetic_links = [r.atom_pair for r in reports]


    # Assign explicit spins if provided
    if spins is not None:
        if isinstance(spins, (list, tuple)):
            if len(spins) != len(atoms):
                raise ValueError(
                    f"Length of explicit spins list ({len(spins)}) must match number of atoms ({len(atoms)})."
                )
            for idx, s in enumerate(spins):
                if s is None:
                    raise ValueError(f"Explicit spin at index {idx} is None. Spin must be 'up' or 'down'.")
                maj_s, orb_s = _normalize_spin_entry(s)
                if maj_s is None:
                    raise ValueError(
                        f"Atom '{atoms[idx].id or atoms[idx].element}' has unassigned or invalid spin ({s!r}). "
                        f"Every atom in a built deck must carry an explicit spin ('up' or 'down'). "
                        f"None anywhere in the spin path is a hard error."
                    )
                atoms[idx].spin = maj_s
                atoms[idx].spin_flip = (maj_s == "down")
                if orb_s is not None:
                    atoms[idx].orbital_spins = orb_s
        elif isinstance(spins, dict):
            for idx, a in enumerate(atoms):
                s = spins.get(a.id) if a.id else None
                if s is None:
                    s = spins.get(idx, spins.get(str(idx), spins.get(a.element)))
                if s is not None:
                    maj_s, orb_s = _normalize_spin_entry(s)
                    atoms[idx].spin = maj_s if maj_s is not None else s
                    if maj_s == "down":
                        atoms[idx].spin_flip = True
                    if orb_s is not None:
                        atoms[idx].orbital_spins = orb_s
        elif isinstance(spins, str):
            maj_s, _ = _normalize_spin_entry(spins)
            for a in atoms:
                a.spin = maj_s if maj_s is not None else spins
                if maj_s == "down":
                    a.spin_flip = True

    if hybridization is not None:
        if isinstance(hybridization, str):
            for a in atoms:
                a.hybridization = hybridization.strip().lower()
        elif isinstance(hybridization, (list, tuple)):
            if len(hybridization) != len(atoms):
                raise ValueError(
                    f"Length of hybridization list ({len(hybridization)}) must match number of atoms ({len(atoms)})."
                )
            for idx, h in enumerate(hybridization):
                if h is not None:
                    atoms[idx].hybridization = str(h).strip().lower()
        elif isinstance(hybridization, dict):
            for idx, a in enumerate(atoms):
                h = hybridization.get(a.id) if a.id else None
                if h is None:
                    h = hybridization.get(idx, hybridization.get(str(idx), hybridization.get(a.element)))
                if h is not None:
                    a.hybridization = str(h).strip().lower()

    # Restricted (closed-shell) decks have one spin channel; per-atom labels are optional
    # and ignored by the engine.
    restricted = (not control.spin_unrestricted) if control is not None else (not spin_unrestricted)
    if restricted and all(a.spin is None for a in atoms):
        resolved_net_spin = 0 if net_spin is None else net_spin
    else:
        # Enforce Requirement 1: Explicit per-atom spins, no None
        for a in atoms:
            if a.spin not in ("up", "down"):
                raise ValueError(
                    f"Atom '{a.id or a.element}' has unassigned or invalid spin ({a.spin!r}). "
                    f"Every atom in a built deck must carry an explicit spin ('up' or 'down'). "
                    f"None anywhere in the spin path is a hard error."
                )

        # Enforce Requirement 2: Net spin check (n_up - n_down == net_spin)
        n_up = sum(1 for a in atoms if a.spin == "up")
        n_down = sum(1 for a in atoms if a.spin == "down")
        calc_net = n_up - n_down
        has_orb_spins = any(getattr(a, "orbital_spins", None) for a in atoms)
        orb_up = sum(
            (sum(1 for x in a.orbital_spins if x == "up") if getattr(a, "orbital_spins", None) else (1 if a.spin == "up" else 0))
            for a in atoms
        )
        orb_down = sum(
            (sum(1 for x in a.orbital_spins if x == "down") if getattr(a, "orbital_spins", None) else (1 if a.spin == "down" else 0))
            for a in atoms
        )
        calc_orb_net = orb_up - orb_down

        if net_spin is None:
            resolved_net_spin = calc_orb_net if has_orb_spins else calc_net
        else:
            if calc_net != net_spin and not (has_orb_spins and calc_orb_net == net_spin):
                raise ValueError(
                    f"Net spin mismatch in build_molecule: calculated n_up ({n_up}) - n_down ({n_down}) = {calc_net}, "
                    f"but net_spin={net_spin} was requested."
                )
            resolved_net_spin = net_spin

    if embedded_atoms:
        for idx, atom in enumerate(atoms):
            emb = None
            if atom.id and atom.id in embedded_atoms:
                emb = embedded_atoms[atom.id]
            elif idx in embedded_atoms:
                emb = embedded_atoms[idx]
            elif str(idx) in embedded_atoms:
                emb = embedded_atoms[str(idx)]
            elif atom.element in embedded_atoms:
                emb = embedded_atoms[atom.element]
            if emb is not None:
                atom.embedded_state = emb

    if isinstance(basis, dict):
        for idx, atom in enumerate(atoms):
            if atom.id and atom.id in basis:
                atom.basis = basis[atom.id]
            elif idx in basis:
                atom.basis = basis[idx]
            elif str(idx) in basis:
                atom.basis = basis[str(idx)]

    if control is None:
        ctrl_extra = dict(extra_namelist or {})
        control = ControlConfig(
            spin_unrestricted=spin_unrestricted,
            extra_namelist=ctrl_extra,
            default_basis=(basis if isinstance(basis, str) else basis.get("default", basis.get("*")) if isinstance(basis, dict) else None),
        )
    else:
        if extra_namelist:
            control.extra_namelist.update(extra_namelist)
        if control.default_basis is None:
            if isinstance(basis, str):
                control.default_basis = basis
            elif isinstance(basis, dict) and ("default" in basis or "*" in basis):
                control.default_basis = basis.get("default", basis.get("*"))

    # Determine unique elements and populate ElementConfigs
    unique_symbols = []
    for atom in atoms:
        if atom.element not in unique_symbols:
            unique_symbols.append(atom.element)

    elements_dict: Dict[str, ElementConfig] = {}
    for sym in unique_symbols:
        b_name = None
        if isinstance(basis, dict) and sym in basis:
            b_name = basis[sym]
        elif isinstance(basis, str):
            b_name = basis

        if b_name:
            ed = get_element_basis(sym, basis_name=b_name)
            elements_dict[sym] = ElementConfig(
                symbol=sym,
                atomic_number=ed.atomic_number,
                basis=ed.basis_name,
                description=ed.description,
                functions=ed.functions,
            )

    resolved_f_links = detected_ferromagnetic_links or (
        [tuple(str(x) for x in link) for link in ferromagnetic_links] if ferromagnetic_links else []
    )
    deck = CalculationDeck(
        title=title,
        control=control,
        geometry=atoms,
        elements=elements_dict,
        basis=(basis if isinstance(basis, str) else None),
        net_spin=resolved_net_spin,
        ferromagnetic_links=resolved_f_links,
        ferromagnetic_reports=detected_ferromagnetic_reports,
    )

    if hasattr(geometry, "cell") and np.any(geometry.cell):
        deck.cell = (np.array(geometry.cell, dtype=np.float64) * BOHR_PER_ANGSTROM).tolist()
    elif poscar_cell is not None:
        deck.cell = poscar_cell.tolist()

    if hasattr(geometry, "pbc"):
        deck.pbc = geometry.pbc.tolist() if hasattr(geometry.pbc, "tolist") else list(geometry.pbc)
    elif poscar_pbc is not None:
        deck.pbc = poscar_pbc

    if any(getattr(a, "hybridization", None) or getattr(a, "orbital_spins", None) for a in deck.geometry):
        # Populate starting_orbitals for atoms carrying explicit hybridization or per-orbital spins
        if any(not a.starting_orbitals and not a.embedded_state for a in deck.geometry):
            from phatpsy.hybrids import apply_hybrid_seeds
            bonds_inferred = None
            if len(deck.geometry) == 2 and all(
                getattr(a, "hybridization", None) == "sp" or len(getattr(a, "orbital_spins", None) or []) == 3
                for a in deck.geometry
            ):
                for a in deck.geometry:
                    if not a.hybridization:
                        a.hybridization = "sp"
                bonds_inferred = {(0, 1): 3}
            apply_hybrid_seeds(deck, bonds=bonds_inferred)

    return deck


def build_diatomic(
    element1: str,
    element2: Optional[str] = None,
    bond_length: float = 2.068,
    basis: Union[str, Dict[str, str]] = "DZ",
    units: str = "bohr",
    use_symmetry: bool = True,
    spins: Optional[Union[List[Any], Tuple[Any, Any]]] = None,
    net_spin: Optional[int] = None,
    spin_flip: Optional[bool] = None,
    hybridization: Optional[Union[str, Sequence[Optional[str]]]] = None,
    title: Optional[str] = None,
    control: Optional[ControlConfig] = None,
    extra_namelist: Optional[Dict[str, Any]] = None,
) -> CalculationDeck:
    """
    Convenience builder for diatomic molecules with explicit spin assignment.

    In PHATPSY methodology, spin-flipping (QFLIP=T) is utilized for homonuclear
    diatomics (e.g. N2, H2) to generate an alternating-spin Perfect-Pairing
    Valence-Bond (PP-VB) state (['up', 'down'], net_spin=0).
    For heteronuclear diatomics (e.g. CO), atoms default to un-flipped configuration
    (['up', 'up'], net_spin=2) or can be configured with explicit hybridized orbital
    spins (e.g. hybridization='sp', spins=[['up', 'up', 'up'], ['down', 'down', 'down']],
    net_spin=0 for the C(-)O(+) triple bond iso-electronic to N2).

    Args:
        element1: Symbol of the first atom (e.g. 'N', 'C', 'Ag').
        element2: Symbol of second atom. Defaults to element1 (homodiatomic).
        bond_length: Interatomic distance (default: 2.068 Bohr for N2).
        basis: Basis set ('DZ', 'SZ', 'THAKKAR').
        units: 'bohr' or 'angstrom'.
        use_symmetry: If True and homodiatomic, use reflection/inversion symmetry card.
        spins: Explicit 2-element spin list (e.g. ['up', 'down'], ['up', 'up'], or
               per-orbital spin lists [['up', 'up', 'up'], ['down', 'down', 'down']]).
        net_spin: Integer (n_up - n_down). Defaults to 0 for alternating spins, 2 for un-flipped.
        spin_flip: [DEPRECATED] Backward compatibility shim. Maps to spins=['up', 'down'] (True)
                   or spins=['up', 'up'] (False). Emits DeprecationWarning.
        hybridization: Optional named hybridization (e.g. 'sp') or 2-element sequence of hybridizations.
        title: Deck title.
        control: ControlConfig options.
        extra_namelist: Additional namelist options.

    Returns:
        CalculationDeck for the diatomic system with validated explicit spins.
    """
    import warnings

    sym1 = element1.strip().capitalize()
    sym2 = (element2 or element1).strip().capitalize()
    is_homo = (sym1 == sym2)

    # Deprecated spin_flip migration shim
    if spin_flip is not None:
        warnings.warn(
            "The 'spin_flip' parameter in build_diatomic is deprecated and will be removed in a future release. "
            "Use explicit 'spins' (e.g. spins=['up', 'down']) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if spins is None:
            spins = ["up", "down"] if spin_flip else ["up", "up"]

    # Explicit default spin lists (Requirement 4 & 6)
    if spins is None:
        if is_homo:
            # Homonuclear PP-VB alternating pattern
            spins = ["up", "down"]
        else:
            # Heteronuclear diatomic gets no flip unless explicitly requested
            spins = ["up", "up"]

    if len(spins) != 2:
        raise ValueError(f"Diatomic spins must contain exactly 2 spin values, got {len(spins)}: {spins}")

    s1, orb_s1 = _normalize_spin_entry(spins[0])
    s2, orb_s2 = _normalize_spin_entry(spins[1])

    if s1 is None or s2 is None:
        raise ValueError(
            f"Explicit spins must be 'up' or 'down', got {spins}. None anywhere in spin path is forbidden."
        )

    n_up = (1 if s1 == "up" else 0) + (1 if s2 == "up" else 0)
    n_down = (1 if s1 == "down" else 0) + (1 if s2 == "down" else 0)
    calc_net = n_up - n_down

    has_orb_spins = (orb_s1 is not None or orb_s2 is not None)
    orb_up = (sum(1 for x in orb_s1 if x == "up") if orb_s1 else (1 if s1 == "up" else 0)) + (
        sum(1 for x in orb_s2 if x == "up") if orb_s2 else (1 if s2 == "up" else 0)
    )
    orb_down = (sum(1 for x in orb_s1 if x == "down") if orb_s1 else (1 if s1 == "down" else 0)) + (
        sum(1 for x in orb_s2 if x == "down") if orb_s2 else (1 if s2 == "down" else 0)
    )
    calc_orb_net = orb_up - orb_down

    if net_spin is None:
        resolved_net_spin = calc_orb_net if has_orb_spins else calc_net
    else:
        if calc_net != net_spin and not (has_orb_spins and calc_orb_net == net_spin):
            raise ValueError(
                f"Net spin mismatch in build_diatomic: calculated n_up ({n_up}) - n_down ({n_down}) = {calc_net} "
                f"from spins={spins}, but net_spin={net_spin} was requested."
            )
        resolved_net_spin = net_spin

    to_bohr = BOHR_PER_ANGSTROM if units.lower() in ("angstrom", "angstroms", "aa", "a") else 1.0
    r_bohr = bond_length * to_bohr

    if title is None:
        title = f"{sym1}{sym2} Diatomic (R={r_bohr:.3f} Bohr)"

    do_flip = (s1 != s2)

    if is_homo and use_symmetry and hybridization is None and not has_orb_spins:
        # Homonuclear diatomic with symmetry along Z axis
        z_half = r_bohr / 2.0
        geometry = [
            AtomConfig(id=f"{sym1}1", element=sym1, xyz=[0.0, 0.0, z_half], spin=s1),
            AtomConfig(
                id=f"{sym1}2",
                element=sym1,
                xyz=[0.0, 0.0, 0.0],
                equivalent_to=f"{sym1}1",
                symmetry_op=-1,
                symmetry_axis=1,
                symmetry=SymmetrySpec(ixyz=1, isym=-1, qflip=do_flip),
                spin_flip=do_flip,
                spin=s2,
            ),
        ]
    else:
        # Heteronuclear or explicit coordinates
        geometry = [
            AtomConfig(
                id=f"{sym1}1",
                element=sym1,
                xyz=[0.0, 0.0, 0.0],
                spin=s1,
                orbital_spins=orb_s1,
            ),
            AtomConfig(
                id=f"{sym2}2",
                element=sym2,
                xyz=[0.0, 0.0, r_bohr],
                spin=s2,
                orbital_spins=orb_s2,
                spin_flip=do_flip if is_homo else False,
            ),
        ]

    return build_molecule(
        geometry=geometry,
        basis=basis,
        title=title,
        control=control,
        units="bohr",
        spins=[orb_s1 if orb_s1 is not None else s1, orb_s2 if orb_s2 is not None else s2],
        net_spin=resolved_net_spin,
        hybridization=hybridization,
        extra_namelist=extra_namelist,
    )

