"""
Native structure parsers, format detection, and ASE bridge for PHATPSY 2.0 (Phase 13, Tiers 1 & 2).

Provides zero-dependency pure-Python parsers for standard molecular and crystal file formats:
- PDB (.pdb): Protein Data Bank format (ATOM / HETATM records).
- Gaussian (.gjf, .com): Gaussian input decks (route, title, charge/multiplicity, coordinates).
- VASP POSCAR / CONTCAR (POSCAR, .vasp, .poscar): Lattice basis and atomic coordinates.
- XYZ (.xyz): Standard chemical Cartesian coordinates with inline modifiers.

And full bidirectional integration with the Atomic Simulation Environment (ASE):
- `from_ase()`: Convert `ase.Atoms` to PHATPSY `AtomConfig` objects in atomic units.
- `to_ase()`: Export `CalculationDeck` or `AtomConfig` list to `ase.Atoms`.
- `parse_ase()` / `read_ase()`: Open 100+ materials/chemical formats supported by ASE.
"""
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any
import math
import re
import numpy as np

from phatpsy.schema import AtomConfig

try:
    import ase
    import ase.io
    from ase import Atoms
    from ase.constraints import FixAtoms
    HAS_ASE: bool = True
except ImportError:
    ase = None  # type: ignore
    Atoms = None  # type: ignore
    FixAtoms = None  # type: ignore
    HAS_ASE: bool = False

BOHR_PER_ANGSTROM: float = 1.8897261245650618

# Periodic table mappings for atomic numbers 1..118
PERIODIC_TABLE = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]

ELEMENT_BY_Z: Dict[int, str] = {i: sym for i, sym in enumerate(PERIODIC_TABLE) if sym}
Z_BY_ELEMENT: Dict[str, int] = {sym.upper(): i for i, sym in enumerate(PERIODIC_TABLE) if sym}


def _read_content(content_or_path: Union[str, Path]) -> str:
    """Helper to return file contents if given an existing path/filename, else string."""
    if isinstance(content_or_path, (str, Path)):
        p = Path(str(content_or_path))
        try:
            if p.is_file():
                return p.read_text(encoding="utf-8")
        except (OSError, ValueError):
            pass
    return str(content_or_path)


def detect_format(content_or_path: Union[str, Path]) -> str:
    """
    Detects the chemical structure or calculation deck format.

    Returns one of: 'pdb', 'gaussian', 'poscar', 'xyz', 'yaml', 'json', 'stdin', or 'unknown'.
    """
    # 1. Check file extension if path-like
    if isinstance(content_or_path, (str, Path)):
        p = Path(str(content_or_path))
        fname = p.name.upper()
        if fname.startswith("POSCAR") or fname.startswith("CONTCAR") or p.suffix.lower() in (".poscar", ".vasp"):
            return "poscar"
        suffix = p.suffix.lower()
        if suffix == ".pdb":
            return "pdb"
        if suffix in (".gjf", ".com"):
            return "gaussian"
        if suffix == ".xyz":
            return "xyz"
        if suffix in (".yaml", ".yml"):
            return "yaml"
        if suffix == ".json":
            return "json"
        if suffix in (".stdin", ".in"):
            return "stdin"
        if suffix in (".cif", ".gro", ".mol", ".sdf", ".gen", ".car", ".arc", ".res", ".cell", ".extxyz"):
            return suffix[1:]
        if fname.startswith("DATA."):
            return "lammps-data"

    # 2. Sniff text content
    text = _read_content(content_or_path).strip()
    if not text:
        return "unknown"

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "unknown"

    # YAML check
    if text.startswith("---") or ("control:" in text and "geometry:" in text) or ("atoms:" in text):
        return "yaml"
    # JSON check
    if text.startswith("{") and text.endswith("}"):
        return "json"
    # Legacy Stdin check
    if any(line.upper().startswith("&CNTRL") or line.upper().startswith("&ATOM") for line in lines[:10]):
        return "stdin"
    # PDB check: lines start with ATOM or HETATM
    if any(line.startswith("ATOM  ") or line.startswith("HETATM") or line.startswith("HEADER") for line in lines[:15]):
        return "pdb"
    # CIF check: lines start with data_ or contain _cell_ or _atom_site_
    if any(line.lower().startswith("data_") or line.startswith("_cell_") or line.startswith("_atom_site_") for line in lines[:25]):
        return "cif"
    # Gaussian input check: route section starting with #
    if any(line.startswith("#") for line in lines[:10]) or any(line.startswith("%") for line in lines[:5]):
        return "gaussian"
    # POSCAR check: line 1 scale, line 2-4 matrix, line 7 Direct/Cartesian
    if len(lines) >= 7:
        try:
            scale = float(lines[1])
            v1 = [float(x) for x in lines[2].split()[:3]]
            v2 = [float(x) for x in lines[3].split()[:3]]
            v3 = [float(x) for x in lines[4].split()[:3]]
            if len(v1) == 3 and len(v2) == 3 and len(v3) == 3:
                # Line 6 or 7 often has Direct or Cartesian
                for check_line in lines[5:9]:
                    c_lower = check_line.lower()
                    if c_lower.startswith(("direct", "cart", "fractional")):
                        return "poscar"
        except (ValueError, IndexError):
            pass

    # Standard XYZ check: line 0 is integer count
    if lines[0].isdigit():
        return "xyz"

    # Multi-line coordinate fallback check: e.g. "C 0.0 0.0 0.0"
    first_tokens = lines[0].split()
    if len(first_tokens) >= 4:
        cand_sym = first_tokens[0].strip().capitalize()
        if cand_sym.upper() in Z_BY_ELEMENT:
            try:
                float(first_tokens[1]), float(first_tokens[2]), float(first_tokens[3])
                return "xyz"
            except ValueError:
                pass

    return "unknown"


def parse_pdb(
    content_or_path: Union[str, Path],
    units: str = "angstrom",
) -> List[AtomConfig]:
    """
    Parses Protein Data Bank (.pdb) format text or file into a list of AtomConfig objects.

    Extracts standard ATOM and HETATM records with Cartesian coordinates,
    chemical elements, and optional residue labels and formal charges.
    Coordinates are converted to Bohrs in the returned AtomConfig objects.

    Parameters
    ----------
    content_or_path : Union[str, Path]
        PDB file path or multi-line PDB string.
    units : str
        Input unit scale, default 'angstrom' (standard for PDB).
    """
    text = _read_content(content_or_path)
    to_bohr = BOHR_PER_ANGSTROM if units.lower() in ("angstrom", "angstroms", "aa", "a") else 1.0

    atoms: List[AtomConfig] = []
    seen_ids: Dict[str, int] = {}

    for line in text.splitlines():
        record_type = line[:6].strip().upper()
        if record_type not in ("ATOM", "HETATM"):
            continue

        # Extract coordinates (columns 31-38, 39-46, 47-54; 1-based PDB standard)
        try:
            x_str = line[30:38].strip()
            y_str = line[38:46].strip()
            z_str = line[46:54].strip()
            x = float(x_str) * to_bohr
            y = float(y_str) * to_bohr
            z = float(z_str) * to_bohr
        except (ValueError, IndexError):
            continue

        # Extract atom name and residue
        atom_name = line[12:16].strip() if len(line) >= 16 else ""
        res_name = line[17:20].strip() if len(line) >= 20 else ""
        serial_str = line[6:11].strip() if len(line) >= 11 else ""

        # Extract element symbol from columns 77-78
        element_str = ""
        if len(line) >= 78:
            element_str = line[76:78].strip()

        # Fallback element inference from atom name if column 77-78 is blank
        if not element_str and atom_name:
            # Strip digits and punctuation: e.g. "CA1" -> "CA", "1HH" -> "H", "Fe2" -> "Fe"
            cleaned = re.sub(r"[0-9\W]", "", atom_name)
            # Try 2-letter symbol then 1-letter symbol
            if len(cleaned) >= 2 and cleaned[:2].upper() in Z_BY_ELEMENT:
                element_str = cleaned[:2].capitalize()
            elif len(cleaned) >= 1 and cleaned[:1].upper() in Z_BY_ELEMENT:
                element_str = cleaned[:1].capitalize()

        if not element_str:
            element_str = "X"
        else:
            element_str = element_str.capitalize()

        # Extract formal charge from columns 79-80 (e.g. "1+", "2+", "1-")
        formal_charge = 0.0
        if len(line) >= 80:
            charge_str = line[78:80].strip()
            if charge_str:
                m = re.match(r"^([0-9]+)?([+-])$", charge_str)
                if m:
                    mag = float(m.group(1)) if m.group(1) else 1.0
                    formal_charge = mag if m.group(2) == "+" else -mag

        # Generate unique atom ID
        base_id = f"{element_str}_{res_name}_{atom_name}" if (res_name and atom_name) else f"{element_str}_{serial_str or len(atoms)+1}"
        count = seen_ids.get(base_id, 0) + 1
        seen_ids[base_id] = count
        atom_id = f"{base_id}_{count}" if count > 1 else base_id

        atoms.append(AtomConfig(
            element=element_str,
            id=atom_id,
            xyz=[x, y, z],
            charge=formal_charge,
        ))

    return atoms


def parse_gaussian(
    content_or_path: Union[str, Path],
    units: Optional[str] = None,
) -> Tuple[List[AtomConfig], int, int, str]:
    """
    Parses a Gaussian calculation input deck (.gjf, .com).

    Extracts route command, title, net molecular charge, spin multiplicity,
    and Cartesian coordinates.

    Parameters
    ----------
    content_or_path : Union[str, Path]
        Gaussian deck file path or string content.
    units : Optional[str]
        Optional unit override. If not specified, checks the route section
        for 'units=bohr' or 'units=au'; otherwise defaults to 'angstrom'.

    Returns
    -------
    Tuple[List[AtomConfig], int, int, str]
        (atoms, net_charge, spin_multiplicity, title)
    """
    text = _read_content(content_or_path)
    lines = text.splitlines()

    # Split into sections separated by blank lines
    sections: List[List[str]] = []
    current_sec: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_sec:
                sections.append(current_sec)
                current_sec = []
        else:
            current_sec.append(stripped)
    if current_sec:
        sections.append(current_sec)

    if not sections:
        return [], 0, 1, ""

    # 1. Filter out Link 0 commands (%chk, %mem, %nproc) from the first section
    sec_idx = 0
    route_lines: List[str] = []
    while sec_idx < len(sections):
        sec = sections[sec_idx]
        filtered = [l for l in sec if not l.startswith("%")]
        if any(l.startswith("#") for l in filtered):
            route_lines = filtered
            sec_idx += 1
            break
        sec_idx += 1

    route_str = " ".join(route_lines).lower()

    # Determine default unit from route or parameter
    if units is None:
        if "units=bohr" in route_str or "units=au" in route_str:
            units = "bohr"
        else:
            units = "angstrom"

    to_bohr = BOHR_PER_ANGSTROM if units.lower() in ("angstrom", "angstroms", "aa", "a") else 1.0

    # 2. Title section
    title = ""
    if sec_idx < len(sections):
        title = " ".join(sections[sec_idx])
        sec_idx += 1

    # 3. Molecule specification (Charge, Multiplicity, followed by coordinates)
    net_charge = 0
    multiplicity = 1
    mol_lines: List[str] = []

    if sec_idx < len(sections):
        mol_block = sections[sec_idx]
        sec_idx += 1

        # The first line of molecule specification is "<charge> <multiplicity>"
        first_line = mol_block[0]
        cm_parts = first_line.split()
        if len(cm_parts) >= 2:
            try:
                net_charge = int(cm_parts[0])
                multiplicity = int(cm_parts[1])
                mol_lines = mol_block[1:]
            except ValueError:
                # If first line wasn't charge/mult, assume it's coordinates
                mol_lines = mol_block
        else:
            mol_lines = mol_block

    atoms: List[AtomConfig] = []
    for line in mol_lines:
        tokens = line.split()
        if len(tokens) < 4:
            continue

        raw_id = tokens[0].strip()
        # Could be atomic number or chemical symbol
        if raw_id.isdigit():
            z_num = int(raw_id)
            elem = ELEMENT_BY_Z.get(z_num, "X")
        else:
            cleaned = re.sub(r"[0-9\W]", "", raw_id)
            elem = cleaned.capitalize() if cleaned else raw_id

        # Gaussian allows an optional freeze-code column:
        # "C 0.0 0.0 0.0" (4 tokens) or "C 0 0.0 0.0 0.0" or "C -1 0.0 0.0 0.0" (5 tokens)
        is_fixed = False
        if len(tokens) >= 5:
            try:
                freeze_code = int(tokens[1])
                # In Gaussian, freeze_code = -1 means frozen atom!
                if freeze_code == -1:
                    is_fixed = True
                coords_str = tokens[2:5]
            except ValueError:
                coords_str = tokens[1:4]
        else:
            coords_str = tokens[1:4]

        try:
            x = float(coords_str[0]) * to_bohr
            y = float(coords_str[1]) * to_bohr
            z = float(coords_str[2]) * to_bohr
        except (ValueError, IndexError):
            continue

        atoms.append(AtomConfig(
            element=elem,
            xyz=[x, y, z],
            fixed=is_fixed,
        ))

    return atoms, net_charge, multiplicity, title


def parse_poscar(
    content_or_path: Union[str, Path],
    units: str = "angstrom",
) -> Tuple[List[AtomConfig], Optional[np.ndarray], str]:
    """
    Parses VASP POSCAR / CONTCAR crystal structure text or file.

    Extracts universal lattice scaling, 3x3 lattice basis vectors, element species,
    counts, coordinate mode (Direct/Fractional vs Cartesian), selective dynamics
    flags (F F F -> fixed=True), and Cartesian coordinates in Bohrs.

    Parameters
    ----------
    content_or_path : Union[str, Path]
        Path to POSCAR/CONTCAR or file contents string.
    units : str
        Unit scale for lattice vectors, default 'angstrom' (standard for VASP).

    Returns
    -------
    Tuple[List[AtomConfig], Optional[np.ndarray], str]
        (atoms, lattice_matrix_bohr, title)
    """
    text = _read_content(content_or_path)
    to_bohr = BOHR_PER_ANGSTROM if units.lower() in ("angstrom", "angstroms", "aa", "a") else 1.0

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 7:
        raise ValueError("POSCAR content has fewer than 7 lines; invalid VASP structure file.")

    title = lines[0]

    # Universal scaling factor
    try:
        scale = float(lines[1])
    except ValueError:
        scale = 1.0

    # 3x3 Lattice basis matrix (rows are a1, a2, a3)
    lattice_mat = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        parts = [float(v) for v in lines[2 + i].split()[:3]]
        lattice_mat[i] = parts

    if scale > 0:
        lattice_mat *= scale
    elif scale < 0:
        # Negative scale means total cell volume is |scale|
        vol_calc = abs(float(np.linalg.det(lattice_mat)))
        if vol_calc > 1e-12:
            length_factor = (abs(scale) / vol_calc) ** (1.0 / 3.0)
            lattice_mat *= length_factor

    # Convert lattice matrix to Bohrs for internal storage
    lattice_mat_bohr = lattice_mat * to_bohr

    # Parse elements and stoichiometry
    line_5_tokens = lines[5].split()
    line_6_tokens = lines[6].split()

    species: List[str] = []
    counts: List[int] = []

    if line_5_tokens[0].isdigit():
        # VASP 4 format: Line 5 is counts; element symbols inferred from title or generic
        counts = [int(v) for v in line_5_tokens]
        # Attempt to split title into element symbols if counts match
        title_tokens = [t.capitalize() for t in title.split() if t.capitalize() in Z_BY_ELEMENT]
        if len(title_tokens) == len(counts):
            species = title_tokens
        else:
            species = [f"X{i+1}" for i in range(len(counts))]
        current_idx = 6
    else:
        # VASP 5 format: Line 5 is species, Line 6 is counts
        species = [s.capitalize() for s in line_5_tokens]
        counts = [int(v) for v in line_6_tokens]
        current_idx = 7

    # Check for Selective Dynamics
    has_selective = False
    if current_idx < len(lines):
        line_check = lines[current_idx].lower()
        if line_check.startswith("s"):
            has_selective = True
            current_idx += 1

    # Coordinate mode: Direct/Fractional vs Cartesian
    if current_idx >= len(lines):
        raise ValueError("POSCAR missing coordinate type specification line.")

    mode_line = lines[current_idx].lower()
    is_direct = mode_line.startswith(("d", "f"))  # Direct or Fractional
    current_idx += 1

    # Flatten species into element label per atom
    atom_elements: List[str] = []
    for sp, cnt in zip(species, counts):
        atom_elements.extend([sp] * cnt)

    atoms: List[AtomConfig] = []
    for i, elem in enumerate(atom_elements):
        if current_idx >= len(lines):
            break
        atom_line = lines[current_idx]
        current_idx += 1

        parts = atom_line.split()
        if len(parts) < 3:
            continue

        raw_coords = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)

        is_fixed = False
        if has_selective and len(parts) >= 6:
            flags = [f.upper() for f in parts[3:6]]
            # If all three Cartesian coordinates are frozen (F F F), mark fixed
            if flags == ["F", "F", "F"]:
                is_fixed = True

        if is_direct:
            # Fractional coordinates: r = u * a1 + v * a2 + w * a3 = coords @ lattice_mat_bohr
            cart_bohr = raw_coords @ lattice_mat_bohr
        else:
            # Cartesian coordinates in input scale (default Angstrom)
            cart_bohr = raw_coords * scale * to_bohr

        atoms.append(AtomConfig(
            element=elem,
            id=f"{elem}{i+1}",
            xyz=cart_bohr.tolist(),
            fixed=is_fixed,
        ))

    return atoms, lattice_mat_bohr, title


# ==============================================================================
# ASE Ecosystem Bridge (Phase 13, Tier 2)
# ==============================================================================

def from_ase(
    atoms: Any,
    units: str = "bohr",
) -> List[AtomConfig]:
    """
    Converts an ASE Atoms object into a list of PHATPSY AtomConfig objects.

    Transfers:
    - Chemical symbols / atomic numbers -> AtomConfig.element
    - Positions (converted from ASE Angstroms to PHATPSY Bohrs, unless units='angstrom')
    - Formal / initial charges -> AtomConfig.charge
    - Initial magnetic moments -> AtomConfig.spin ('up' if > 0, 'down' if < 0) and spin_flip
    - FixAtoms constraints -> AtomConfig.fixed = True
    - Atom tags / IDs -> AtomConfig.id

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms instance.
    units : str
        Target coordinate units for internal AtomConfig storage (default 'bohr').

    Returns
    -------
    List[AtomConfig]
        Equivalent PHATPSY atom configurations in atomic units (Bohr).
    """
    if not HAS_ASE:
        raise ImportError(
            "The ASE bridge requires 'ase'. Please install it via 'pip install ase'."
        )

    if not (hasattr(atoms, "get_positions") and hasattr(atoms, "get_chemical_symbols")):
        raise TypeError(f"Expected ase.Atoms instance, got {type(atoms)}")

    scale = BOHR_PER_ANGSTROM if units.lower() in ("bohr", "au", "atomic") else 1.0

    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions() * scale

    # Formal / initial charges
    try:
        charges = atoms.get_initial_charges()
    except Exception:
        charges = [0.0] * len(atoms)

    # Initial magnetic moments
    try:
        magmoms = atoms.get_initial_magnetic_moments()
    except Exception:
        magmoms = [0.0] * len(atoms)

    # Fixed atoms constraint (FixAtoms)
    fixed_indices = set()
    if hasattr(atoms, "constraints") and atoms.constraints:
        for c in atoms.constraints:
            if isinstance(c, FixAtoms) or type(c).__name__ == "FixAtoms":
                if hasattr(c, "index"):
                    idx = c.index
                    if isinstance(idx, (list, tuple, np.ndarray)):
                        fixed_indices.update(np.asarray(idx).tolist())
                elif hasattr(c, "indices"):
                    idx = c.indices
                    if isinstance(idx, (list, tuple, np.ndarray)):
                        fixed_indices.update(np.asarray(idx).tolist())

    elem_counts: Dict[str, int] = {}
    atom_configs: List[AtomConfig] = []

    for i, (sym, pos) in enumerate(zip(symbols, positions)):
        s_cap = sym.strip().capitalize()
        elem_counts[s_cap] = elem_counts.get(s_cap, 0) + 1
        atom_id = f"{s_cap}{elem_counts[s_cap]}"

        chg = float(charges[i]) if i < len(charges) else 0.0
        mag = float(magmoms[i]) if i < len(magmoms) else 0.0

        spin_val: Optional[str] = None
        spin_flip_val = False
        if mag > 0.0:
            spin_val = "up"
            spin_flip_val = False
        elif mag < 0.0:
            spin_val = "down"
            spin_flip_val = True

        atom_configs.append(AtomConfig(
            element=s_cap,
            id=atom_id,
            xyz=pos.tolist(),
            charge=chg,
            spin=spin_val,
            spin_flip=spin_flip_val,
            fixed=(i in fixed_indices),
        ))

    return atom_configs


def to_ase(
    atoms_or_deck: Union[Any, List[AtomConfig]],
    cell: Optional[Any] = None,
    pbc: Optional[Any] = None,
) -> Any:
    """
    Converts a PHATPSY CalculationDeck or list of AtomConfig objects into an ASE Atoms object.

    Transfers:
    - Atom coordinates (converted from Bohrs to Angstroms)
    - Chemical elements
    - Charges -> atoms.set_initial_charges()
    - Spins -> atoms.set_initial_magnetic_moments() (up = +1.0, down = -1.0)
    - Fixed atoms -> FixAtoms constraint
    - Unit cell & periodic boundary conditions (PBC) if provided or present on deck

    Parameters
    ----------
    atoms_or_deck : Union[CalculationDeck, List[AtomConfig]]
        Input PHATPSY deck or list of atom configurations.
    cell : Optional[Any]
        Optional 3x3 lattice matrix or 6-element lattice parameters in Bohr.
        If None, attempts to read from deck.cell if available.
    pbc : Optional[Any]
        Optional periodic boundary condition flags (e.g. True or [True, True, True]).
        If None, attempts to read from deck.pbc if available.

    Returns
    -------
    ase.Atoms
        The corresponding ASE Atoms object in Angstrom units.
    """
    if not HAS_ASE:
        raise ImportError(
            "The ASE bridge requires 'ase'. Please install it via 'pip install ase'."
        )

    if hasattr(atoms_or_deck, "geometry"):
        atoms = atoms_or_deck.geometry
        deck_cell = getattr(atoms_or_deck, "cell", None)
        deck_pbc = getattr(atoms_or_deck, "pbc", None)
        if cell is None and deck_cell is not None:
            cell = deck_cell
        if pbc is None and deck_pbc is not None:
            pbc = deck_pbc
    elif isinstance(atoms_or_deck, (list, tuple)):
        atoms = atoms_or_deck
    else:
        raise TypeError(f"Expected CalculationDeck or List[AtomConfig], got {type(atoms_or_deck)}")

    symbols = [a.element.strip().capitalize() for a in atoms]
    positions = np.array([a.xyz for a in atoms], dtype=np.float64) / BOHR_PER_ANGSTROM

    ase_cell = None
    if cell is not None:
        ase_cell = np.array(cell, dtype=np.float64) / BOHR_PER_ANGSTROM

    ase_pbc = pbc if pbc is not None else (False, False, False)

    ase_atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=ase_cell,
        pbc=ase_pbc,
    )

    # Initial charges
    charges = [getattr(a, "charge", 0.0) for a in atoms]
    if any(abs(c) > 1e-6 for c in charges):
        ase_atoms.set_initial_charges(charges)

    # Magnetic moments
    magmoms = []
    has_mag = False
    for a in atoms:
        sp = getattr(a, "spin", None)
        if sp == "up":
            magmoms.append(1.0)
            has_mag = True
        elif sp == "down":
            magmoms.append(-1.0)
            has_mag = True
        else:
            magmoms.append(0.0)
    if has_mag:
        ase_atoms.set_initial_magnetic_moments(magmoms)

    # Fixed atoms constraint
    fixed_indices = [i for i, a in enumerate(atoms) if getattr(a, "fixed", False)]
    if fixed_indices:
        ase_atoms.set_constraint(FixAtoms(indices=fixed_indices))

    return ase_atoms


def parse_ase(
    content_or_path: Union[str, Path],
    format: Optional[str] = None,
    units: str = "bohr",
) -> List[AtomConfig]:
    """
    Parses any structure format supported by ASE (CIF, GRO, MOL, SDF, CAR, etc.)
    into a list of PHATPSY AtomConfig objects.

    Parameters
    ----------
    content_or_path : Union[str, Path]
        File path or string containing structure data.
    format : Optional[str]
        Explicit ASE format specifier (e.g. 'cif', 'gro', 'mol-v3000'). If None,
        ASE auto-detects from file extension or content.
    units : str
        Target coordinate units for AtomConfig storage (default 'bohr').

    Returns
    -------
    List[AtomConfig]
        Parsed atoms in atomic units (Bohr).
    """
    atoms, _, _ = read_ase(content_or_path, format=format, units=units)
    return atoms


def read_ase(
    content_or_path: Union[str, Path],
    format: Optional[str] = None,
    units: str = "bohr",
) -> Tuple[List[AtomConfig], Optional[np.ndarray], Optional[List[bool]]]:
    """
    Reads a structure using ASE, returning (atoms, cell_in_bohr, pbc).

    Parameters
    ----------
    content_or_path : Union[str, Path]
        File path or string containing structure data.
    format : Optional[str]
        Explicit ASE format specifier (e.g. 'cif', 'gro').
    units : str
        Target coordinate units for AtomConfig storage (default 'bohr').

    Returns
    -------
    Tuple[List[AtomConfig], Optional[np.ndarray], Optional[List[bool]]]
        Tuple of (atoms, cell_in_bohr, pbc).
    """
    if not HAS_ASE:
        raise ImportError(
            "The ASE bridge requires 'ase'. Please install it via 'pip install ase'."
        )

    ASE_FORMAT_ALIASES = {
        "gro": "gromacs",
        "poscar": "vasp",
        "contcar": "vasp",
    }
    fmt_target = ASE_FORMAT_ALIASES.get((format or "").lower(), format)

    import io
    s_input = str(content_or_path)
    is_file = False
    if "\n" not in s_input and "\r" not in s_input:
        try:
            is_file = Path(s_input).is_file()
        except (OSError, ValueError):
            is_file = False

    if is_file:
        ase_atoms = ase.io.read(s_input, format=fmt_target)
    else:
        stream = io.StringIO(s_input)
        ase_atoms = ase.io.read(stream, format=fmt_target)

    if isinstance(ase_atoms, list):
        ase_atoms = ase_atoms[-1]

    atoms = from_ase(ase_atoms, units=units)
    cell = None
    if hasattr(ase_atoms, "cell") and np.any(ase_atoms.cell):
        cell = np.array(ase_atoms.cell, dtype=np.float64) * BOHR_PER_ANGSTROM
    pbc = ase_atoms.pbc.tolist() if hasattr(ase_atoms, "pbc") else None
    return atoms, cell, pbc


