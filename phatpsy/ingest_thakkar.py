# phatpsy/ingest_thakkar.py
"""
Ingestion script for Thakkar & Koga analytical Hartree-Fock Slater-Type Orbital (STO) tables.
Sources:
- Toshikatsu Koga, K. Kanayama, S. Watanabe, Ajit J. Thakkar, Int. J. Quantum Chem. 71, 491-497 (1999) [Z=1-54]
- Toshikatsu Koga, K. Kanayama, T. Watanabe, T. Imai, Ajit J. Thakkar, Theor. Chem. Acc. 104, 411-413 (2000) [Z=55-103]
Machine-readable ASCII data curated by Susi Lehtola & James Furness (JFurness1/AtomicOrbitals).
"""
import concurrent.futures
import json
import logging
import re
import ssl
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

PERIODIC_TABLE = [
    '', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
    'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
    'Es', 'Fm', 'Md', 'No', 'Lr'
]

SYM_TO_Z = {sym.lower(): z for z, sym in enumerate(PERIODIC_TABLE) if sym}

ELEMENTS_K99 = [
    'h', 'he', 'li', 'be', 'b', 'c', 'n', 'o', 'f', 'ne',
    'na', 'mg', 'al', 'si', 'p', 's', 'cl', 'ar',
    'k', 'ca', 'sc', 'ti', 'v', 'cr', 'mn', 'fe', 'co', 'ni', 'cu', 'zn',
    'ga', 'ge', 'as', 'se', 'br', 'kr',
    'rb', 'sr', 'y', 'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', 'cd',
    'in', 'sn', 'sb', 'te', 'i', 'xe'
]

ELEMENTS_K00 = [
    'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy',
    'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w', 're', 'os', 'ir', 'pt',
    'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn',
    'fr', 'ra', 'ac', 'th', 'pa', 'u', 'np', 'pu', 'am', 'cm', 'bk', 'cf',
    'es', 'fm', 'md', 'no', 'lr'
]

RAW_BASE_URL = "https://raw.githubusercontent.com/JFurness1/AtomicOrbitals/master/literature_data"


def parse_thakkar_occupation(line: str) -> Dict[str, Any]:
    """
    Parses the ground-state electron configuration string into shell occupations.
    Handles noble-gas core shorthand ([XE], [RN], K, L, M) and s, p, d, f shells.
    """
    occdata = line.split()[1].replace(",", "").replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ").split()
    s_occ = [[], []]
    p_occ = [[], []]
    d_occ = [[], []]
    f_occ = [[], []]

    def hunds_rule(nel_str: str, am: int):
        nel = int(nel_str)
        up = min(nel, 2 * am + 1)
        down = nel - up
        return up, down

    ishell = 0
    while ishell < len(occdata):
        token = occdata[ishell].upper()
        if token == 'XE':
            s_occ[0].extend([1, 1, 1, 1, 1])
            s_occ[1].extend([1, 1, 1, 1, 1])
            p_occ[0].extend([3, 3, 3, 3])
            p_occ[1].extend([3, 3, 3, 3])
            d_occ[0].extend([5, 5])
            d_occ[1].extend([5, 5])
            ishell += 1
        elif token == 'RN':
            s_occ[0].extend([1, 1, 1, 1, 1, 1])
            s_occ[1].extend([1, 1, 1, 1, 1, 1])
            p_occ[0].extend([3, 3, 3, 3, 3])
            p_occ[1].extend([3, 3, 3, 3, 3])
            d_occ[0].extend([5, 5, 5])
            d_occ[1].extend([5, 5, 5])
            f_occ[0].append(7)
            f_occ[1].append(7)
            ishell += 1
        elif token == 'K':
            s_occ[0].append(1)
            s_occ[1].append(1)
            ishell += 2
        elif token == 'L':
            s_occ[0].append(1)
            s_occ[1].append(1)
            p_occ[0].append(3)
            p_occ[1].append(3)
            ishell += 2
        elif token == 'M':
            s_occ[0].append(1)
            s_occ[1].append(1)
            p_occ[0].append(3)
            p_occ[1].append(3)
            d_occ[0].append(5)
            d_occ[1].append(5)
            ishell += 2
        elif len(token) >= 2 and token[1] == 'S':
            up, down = hunds_rule(occdata[ishell + 1], 0)
            if up + down > 0:
                s_occ[0].append(up)
                s_occ[1].append(down)
            ishell += 2
        elif len(token) >= 2 and token[1] == 'P':
            up, down = hunds_rule(occdata[ishell + 1], 1)
            if up + down > 0:
                p_occ[0].append(up)
                p_occ[1].append(down)
            ishell += 2
        elif len(token) >= 2 and token[1] == 'D':
            up, down = hunds_rule(occdata[ishell + 1], 2)
            if up + down > 0:
                d_occ[0].append(up)
                d_occ[1].append(down)
            ishell += 2
        elif len(token) >= 2 and token[1] == 'F':
            up, down = hunds_rule(occdata[ishell + 1], 3)
            if up + down > 0:
                f_occ[0].append(up)
                f_occ[1].append(down)
            ishell += 2
        else:
            ishell += 1

    return {
        0: (s_occ[0], s_occ[1]),
        1: (p_occ[0], p_occ[1]),
        2: (d_occ[0], d_occ[1]),
        3: (f_occ[0], f_occ[1]),
    }


def parse_thakkar_ascii(el_key: str, text: str) -> Dict[str, Any]:
    """
    Parses a single Thakkar ASCII output file into a structured dictionary.
    """
    lines = [l.rstrip() for l in text.splitlines()]
    if not lines:
        raise ValueError(f"Empty content for element '{el_key}'")

    header = lines[0].strip()
    z = SYM_TO_Z[el_key.lower()]
    symbol = PERIODIC_TABLE[z]

    tot_energy = None
    for line in lines[1:15]:
        m = re.match(r'^\s*E\s*=\s*([-\d\.\+eE]+)', line)
        if m:
            tot_energy = float(m.group(1))
            break

    occs_by_l = parse_thakkar_occupation(header)

    # Locate basis and orbital section
    idx = -1
    for i, line in enumerate(lines):
        if 'ORBITAL ENERGIES' in line:
            idx = i + 1
            break
    if idx == -1:
        raise ValueError(f"Could not find 'ORBITAL ENERGIES' section in {el_key}")

    basis_functions = []
    orbitals = []
    am_map = {'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4}

    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        parts = line.split()
        if parts[0] in am_map:
            am_char = parts[0]
            l = am_map[am_char]
            orb_names = parts[1:]

            idx += 1
            while idx < len(lines) and not lines[idx].strip():
                idx += 1
            if idx >= len(lines):
                break

            energy_line = lines[idx]
            energy_parts = energy_line.split()
            energies = [float(x) for x in energy_parts[1:]]

            idx += 1
            while idx < len(lines) and not lines[idx].strip():
                idx += 1
            if idx < len(lines) and lines[idx].split()[0] == 'CUSP':
                idx += 1

            block_funcs = []
            while idx < len(lines):
                b_line = lines[idx].strip()
                if not b_line:
                    idx += 1
                    break
                b_parts = b_line.split()
                if len(b_parts) < 2 or b_parts[0] in am_map:
                    break
                tag = b_parts[0]
                n = int(tag[:-1])
                zeta = float(b_parts[1])
                coeffs = [float(x) for x in b_parts[2:]]
                func_idx = len(basis_functions) + 1
                basis_functions.append({
                    'index': func_idx,
                    'n': n,
                    'l': l,
                    'zeta': zeta
                })
                block_funcs.append((func_idx, coeffs))
                idx += 1

            up_occs, down_occs = occs_by_l.get(l, ([], []))
            for o_idx, name in enumerate(orb_names):
                e_val = energies[o_idx] if o_idx < len(energies) else 0.0
                u_occ = up_occs[o_idx] if o_idx < len(up_occs) else 0.0
                d_occ = down_occs[o_idx] if o_idx < len(down_occs) else 0.0
                coef_list = []
                for f_idx, c_row in block_funcs:
                    if o_idx < len(c_row):
                        coef_list.append({'function_index': f_idx, 'c': c_row[o_idx]})
                orbitals.append({
                    'name': name,
                    'l': l,
                    'eigval': e_val,
                    'occ_up': float(u_occ),
                    'occ_down': float(d_occ),
                    'occ_total': float(u_occ + d_occ),
                    'coefficients': coef_list
                })
        else:
            idx += 1

    return {
        'symbol': symbol,
        'atomic_number': z,
        'name': header.split()[0],
        'configuration': header,
        'energy': tot_energy,
        'basis_name': 'THAKKAR',
        'description': f'Thakkar-Koga (1999/2000) RHF STO basis for {symbol} (Z={z})',
        'basis_functions': basis_functions,
        'functions': basis_functions,
        'orbitals': orbitals
    }


def fetch_all_thakkar_data(output_json: Optional[Path] = None, max_workers: int = 16) -> Dict[str, Any]:
    """
    Downloads and parses all 103 Thakkar ASCII files (Z=1 to 103).
    Saves to output_json if provided.
    """
    ctx = ssl._create_unverified_context()

    items = [('k99l/neutral', el) for el in ELEMENTS_K99] + [('k00heavy', el) for el in ELEMENTS_K00]

    def _fetch_one(item):
        folder, el = item
        url = f"{RAW_BASE_URL}/{folder}/{el}"
        req = urllib.request.Request(url, headers={'User-Agent': 'phatpsy-ingest/1.0'})
        with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
            return el, r.read().decode('utf-8', errors='replace')

    print(f"Fetching {len(items)} Thakkar ASCII tables from GitHub repository...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        raw_texts = dict(pool.map(_fetch_one, items))

    database = {}
    for z in range(1, 104):
        sym = PERIODIC_TABLE[z]
        el = sym.lower()
        if el not in raw_texts:
            raise KeyError(f"Missing raw text for element '{el}' (Z={z})")
        parsed = parse_thakkar_ascii(el, raw_texts[el])
        database[sym] = parsed

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(database, f, indent=2)
        print(f"Successfully wrote {len(database)} elements to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    return database


if __name__ == "__main__":
    dest = Path(__file__).parent / "data" / "thakkar_sto.json"
    fetch_all_thakkar_data(output_json=dest)
