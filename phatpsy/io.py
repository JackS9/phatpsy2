# phatpsy/io.py
"""
Modern Python I/O utilities for PHATPSY.
Handles reading and writing legacy .stdin decks, converting geometries,
and extracting structured input dataclasses for automation.
"""
from typing import Union, Dict, List, Tuple, Any, Optional
import re
from pathlib import Path
from phatpsy.schema import (
    CalculationDeck,
    ControlConfig,
    AtomConfig,
    ElementConfig,
    SymmetrySpec,
)
from phatpsy.basis import BasisFunction


def parse_stdin_to_deck(file_path: Union[str, Path]) -> CalculationDeck:
    """
    Parses a legacy PHATPSY .stdin file directly into a structured CalculationDeck,
    extracting CTLINP controls, element definitions, basis functions, and atom coordinates.
    """
    p = Path(file_path)
    with open(p, "r", errors="replace") as f:
        content = f.read()

    lines = content.splitlines()
    title = lines[0].strip() if lines else "PHATPSY Calculation"

    # 1. Parse &CTLINP
    ctlinp = {}
    ctlinp_m = re.search(r"&CTLINP(.*?)&END", content, re.DOTALL | re.IGNORECASE)
    if not ctlinp_m:
        ctlinp_m = re.search(r"&CTLINP(.*?)/", content, re.DOTALL | re.IGNORECASE)

    if ctlinp_m:
        for pair in re.finditer(r"([A-Za-z0-9_]+)\s*=\s*([^,\n/]+)", ctlinp_m.group(1)):
            k = pair.group(1).upper()
            v_str = pair.group(2).strip()
            if v_str.upper() in ("T", ".TRUE."):
                v = True
            elif v_str.upper() in ("F", ".FALSE."):
                v = False
            else:
                try:
                    if "." in v_str or "D" in v_str.upper() or "E" in v_str.upper():
                        v = float(v_str.upper().replace("D", "E"))
                    else:
                        v = int(v_str)
                except ValueError:
                    v = v_str
            ctlinp[k] = v

    control = ControlConfig(
        spin_unrestricted=bool(ctlinp.get("QOPEN", True)),
        max_cycles=int(ctlinp.get("MXCYCL", 100)),
        convergence=float(ctlinp.get("CONVRG", 1.0e-4)),
        time_limit=float(ctlinp.get("TIMEND", 360.0)),
        charge_damp=float(ctlinp.get("CDAMP", 0.5)),
        charge_accel=float(ctlinp.get("CACCEL", 1.0)),
        potential_damp=float(ctlinp.get("VDAMP", 0.5)),
        potential_accel=float(ctlinp.get("VACCEL", 1.0)),
        pk_scale=float(ctlinp.get("PKSCAL", 0.0)),
        separate_atoms_first_cycle=bool(ctlinp.get("QSEPAT", True)),
        debug=bool(ctlinp.get("QDEBUG", False)),
        trace=bool(ctlinp.get("QTRACE", False)),
    )

    deck_body = content[ctlinp_m.end():] if ctlinp_m else content
    elem_blocks = re.split(r"&ELEM\b", deck_body, flags=re.IGNORECASE)
    elements = {}
    geometry = []

    for block in elem_blocks[1:]:
        latom_m = re.search(r"LATOM\s*=\s*['\"]([A-Za-z]+)['\"]", block, re.IGNORECASE)
        sym = latom_m.group(1).strip().capitalize() if latom_m else "X"
        nucz_m = re.search(r"NUCZ\s*=\s*(\d+)", block, re.IGNORECASE)
        nucz = int(nucz_m.group(1)) if nucz_m else None
        descr_m = re.search(r"DESCR\s*=\s*['\"]([^'\"]+)['\"]", block, re.IGNORECASE)
        descr = descr_m.group(1).strip() if descr_m else None

        elem_end = re.search(r"(&END|/)", block)
        first_atom = re.search(r"&ATOM", block, re.IGNORECASE)
        basis_funcs = []
        if elem_end and first_atom:
            basis_text = block[elem_end.end():first_atom.start()]
            for line in basis_text.splitlines():
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        n, l, zeta = int(parts[0]), int(parts[1]), float(parts[2])
                        if n > 0 and zeta > 0:
                            basis_funcs.append(BasisFunction(n=n, l=l, zeta=zeta))
                    except ValueError:
                        pass

        elements[sym] = ElementConfig(
            symbol=sym,
            atomic_number=nucz,
            description=descr,
            functions=basis_funcs,
        )

        atom_entries = re.finditer(r"&ATOM\s+(.*?)(&END|/)", block, re.IGNORECASE | re.DOTALL)
        atom_count_for_elem = 0
        current_primary_atom: Optional[AtomConfig] = None

        for atom_m in atom_entries:
            atom_str = atom_m.group(1)
            fixed = bool(re.search(r"QFIXED\s*=\s*[Tt]", atom_str))
            virtual = bool(re.search(r"QVIRTL\s*=\s*[Tt]", atom_str))
            spin_flip = bool(re.search(r"QFLIP\s*=\s*[Tt]", atom_str))
            xyz_m = re.search(r"XYZ\s*=\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)", atom_str, re.IGNORECASE)
            symm_m = re.search(r"IXYZ\s*=\s*(\d+)", atom_str, re.IGNORECASE)

            atom_count_for_elem += 1
            atom_id = f"{sym}{atom_count_for_elem}"

            if xyz_m:
                xyz = [float(xyz_m.group(1)), float(xyz_m.group(2)), float(xyz_m.group(3))]
                p_atom = AtomConfig(
                    element=sym,
                    id=atom_id,
                    xyz=xyz,
                    fixed=fixed,
                    virtual=virtual,
                    spin="down" if spin_flip else "up",
                    spin_flip=spin_flip,
                )
                current_primary_atom = p_atom
                geometry.append(p_atom)
            elif symm_m:
                ixyz = int(symm_m.group(1))
                isym_m = re.search(r"ISYM\s*=\s*([-\d]+)", atom_str, re.IGNORECASE)
                isym = int(isym_m.group(1)) if isym_m else -1
                qflip_m = re.search(r"QFLIP\s*=\s*([TF])", atom_str, re.IGNORECASE)
                qflip = (qflip_m.group(1).upper() == "T") if qflip_m else True

                ref_id = current_primary_atom.id if current_primary_atom else None
                ref_spin = current_primary_atom.spin if current_primary_atom else "up"
                eq_spin = ("down" if ref_spin == "up" else "up") if qflip else ref_spin

                # Calculate physical coordinates from reference atom if available
                from phatpsy.schema import apply_symmetry_operation
                eq_xyz = (
                    apply_symmetry_operation(current_primary_atom.xyz, isym=isym, ixyz=ixyz)
                    if current_primary_atom
                    else [0.0, 0.0, 0.0]
                )

                geometry.append(AtomConfig(
                    element=sym,
                    id=atom_id,
                    xyz=eq_xyz,
                    equivalent_to=ref_id,
                    symmetry_op=isym,
                    symmetry_axis=ixyz,
                    symmetry=SymmetrySpec(ixyz=ixyz, isym=isym, qflip=qflip),
                    fixed=fixed,
                    virtual=virtual,
                    spin_flip=qflip,
                    spin=eq_spin,
                ))

    return CalculationDeck(
        title=title,
        control=control,
        geometry=geometry,
        elements=elements,
    )


def parse_stdin_file(file_path: Path) -> Dict[str, Any]:
    """
    Parses a legacy PHATPSY .stdin file into structured dictionaries.
    Extracts CTLINP options, basis set exponents, and atom coordinates.
    """
    path = Path(file_path)
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()

    if not lines:
        return {}

    title = lines[0].strip()
    full_text = "".join(lines[1:])

    # Extract CTLINP namelist
    ctlinp = {}
    ctlinp_match = re.search(r"&CTLINP(.*?)&END", full_text, re.DOTALL | re.IGNORECASE)
    if not ctlinp_match:
        ctlinp_match = re.search(r"&CTLINP(.*?)/", full_text, re.DOTALL | re.IGNORECASE)

    if ctlinp_match:
        block = ctlinp_match.group(1)
        for pair in re.finditer(r"([A-Za-z0-9_]+)\s*=\s*([^,\n/]+)", block):
            key = pair.group(1).upper()
            val_str = pair.group(2).strip()
            # Parse types
            if val_str.upper() in ("T", ".TRUE."):
                val = True
            elif val_str.upper() in ("F", ".FALSE."):
                val = False
            else:
                try:
                    if "." in val_str or "D" in val_str.upper() or "E" in val_str.upper():
                        val = float(val_str.upper().replace("D", "E"))
                    else:
                        val = int(val_str)
                except ValueError:
                    val = val_str
            ctlinp[key] = val

    # Extract atom positions
    atom_positions = []
    atom_matches = re.finditer(r"&ATOM\s+XYZ\s*=\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)", full_text, re.IGNORECASE)
    for m in atom_matches:
        atom_positions.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))

    return {
        "title": title,
        "ctlinp": ctlinp,
        "atom_positions": atom_positions,
        "raw_text": "".join(lines),
    }


def generate_diatomic_deck(
    elem_symbol: str,
    atomic_number: int,
    bond_length_bohr: float,
    basis_exponents: List[Tuple[int, int, float]],
    title: str = "Diatomic Molecule",
    qopen: bool = True,
    convergence: float = 1e-4,
) -> str:
    """
    Generates a valid PHATPSY input deck for a homonuclear diatomic molecule
    along the z-axis centered at z = +/- (R / 2).
    """
    z_half = bond_length_bohr / 2.0
    nsto = len(basis_exponents)
    # Count m_l values: s=1, p=3, d=5
    msto = sum(2 * l + 1 for _, l, _ in basis_exponents)

    deck = [
        title,
        " &CTLINP",
        f"  QOPEN={'T' if qopen else 'F'},",
        f"  NELEM=1, NATOM=2, NORBT={msto},",
        f"  MXNSTO={nsto}, MXMSTO={msto}, MXNORB={msto},",
        f"  CONVRG={convergence:.1E}, MXCYCL=50, TIMEND=360.0D0,",
        " &END",
        " &BASDIM",
        f"  MSTO={msto}, NSTO={nsto}, NORB={msto}, NVTERM=5,",
        " &END",
        " &ELEM",
        f"  LATOM='{elem_symbol}', NUCZ={atomic_number},",
        f"  DESCR='{elem_symbol} Basis:',",
        "  MDIFAT=1, QTFPOT=T,",
        " &END",
    ]

    for n, l, eta in basis_exponents:
        deck.append(f"  {n:2d} {l:2d}   {eta:10.5f}")
    deck.append("   0  0   0.150     0.0")

    # Primary atom at +z_half
    deck.append(f" &ATOM XYZ=0.0,0.0,{z_half:.5f}, NEQAT=1 &END")
    # Symmetry partner at -z_half
    deck.append(" &ATOM IXYZ=001, ISYM=-1, QFLIP=T &END")

    return "\n".join(deck) + "\n"


def save_results_to_json(results: Dict[str, Any], json_path: Path, indent: int = 2) -> None:
    """
    Serializes PHATPSY driver calculation results to JSON,
    converting NumPy ndarrays to nested lists and floats to standard Python types.
    """
    import json

    def _convert(obj):
        import numpy as np
        if hasattr(obj, "to_dict"):
            return _convert(obj.to_dict())
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(x) for x in obj]
        return obj

    clean_dict = _convert(results)
    out_path = Path(json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clean_dict, f, indent=indent)
