#!/usr/bin/env python3
"""
Extract ADF polarization-function exponents from the freely downloadable
ZORA STO basis sets (van Lenthe & Baerends, J. Comput. Chem. 24, 1142 (2003)).

Usage:
    1. Download https://www.scm.com/doc/ADF/_downloads/13a32cb06e43fc959219ca9fb75da628/zorabasis.tar.gz
       (free download, no ADF license needed -- offered on SCM's basis-set page)
    2. tar xzf zorabasis.tar.gz   (extract anywhere, e.g. /tmp/zorabasis)
    3. python3 parse_adf_polarization.py /tmp/zorabasis

What it does:
    For each element H..Ar, reads the all-electron ZORA/DZP and ZORA/DZ files,
    parses their BASIS sections, and reports the polarization functions, i.e.
    the (n, l, zeta) entries present in DZP but not in DZ.  ADF defines
    DZP = DZ + one polarization function, so exactly one is expected per
    element (2p for H/He, 3d for Li..Ar).

Output:
    JSON mapping element symbol -> {"n": int, "l": int, "zeta": float},
    plus a ready-to-paste POLARIZATION_PARAMETERS dict for phatpsy/basis.py.

File format reference:
    https://www.scm.com/doc/ADF/Appendices/Basis_set_file_format.html
    BASIS section: lines between a 'BASIS' line and the next 'END' line;
    each record is '<n><L> <zeta>', e.g. '  3D   2.000'.  zeta is in bohr^-1.
    All-electron files are named exactly the element symbol; frozen-core
    variants carry suffixes like '.2p'.

NOTE: written from the documented format; NOT yet run against real files.
Before trusting the output, verify that exactly one polarization function is
found per element and sanity-check H (expect zeta=1.25) and O (expect
zeta=2.00) against Gasevic et al., J. Phys. Chem. A 2022, 126, 3826-3838.
"""

import json
import os
import sys
from collections import Counter

ELEMENTS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"]
L_MAP = {"S": 0, "P": 1, "D": 2, "F": 3}


def parse_basis_section(path):
    """Return list of (n, l, zeta) from the BASIS section of an ADF basis file."""
    funcs = []
    in_basis = False
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not in_basis:
                if line == "BASIS":
                    in_basis = True
                continue
            if line.upper() == "END":
                break
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            nl, zeta_s = parts[0], parts[1]
            try:
                n = int(nl[:-1])
                l = L_MAP[nl[-1].upper()]
                zeta = float(zeta_s)
            except (ValueError, KeyError):
                continue
            funcs.append((n, l, zeta))
    return funcs


def find_dir(root, *parts):
    """Locate a basis-set subdirectory, tolerating extra nesting in the tarball."""
    direct = os.path.join(root, *parts)
    if os.path.isdir(direct):
        return direct
    target = parts[-1]
    for dirpath, dirnames, _ in os.walk(root):
        if os.path.basename(dirpath) == target:
            return dirpath
    return None


def main(root):
    out = {}
    problems = []
    # In ADF ZORA basis sets, standard single polarization resides in TZP (or DZP in non-ZORA)
    pol_dir = (
        find_dir(root, "ZORA", "DZP")
        or find_dir(root, "DZP")
        or find_dir(root, "ZORA", "TZP")
        or find_dir(root, "TZP")
    )
    dz_dir = find_dir(root, "ZORA", "DZ") or find_dir(root, "DZ")
    if pol_dir is None or dz_dir is None:
        sys.exit(f"could not locate polarized (DZP/TZP) or unpolarized (DZ) directories under {root}")
    pol_name = os.path.basename(pol_dir)
    print(f"# Extracting polarization exponents from {pol_name} vs DZ under {root}", file=sys.stderr)

    for el in ELEMENTS:
        pol_file = os.path.join(pol_dir, el)
        dz_file = os.path.join(dz_dir, el)
        if not os.path.isfile(pol_file) or not os.path.isfile(dz_file):
            problems.append(f"{el}: missing file (pol={pol_file!r} dz={dz_file!r})")
            continue
        f_pol = parse_basis_section(pol_file)
        f_dz = parse_basis_section(dz_file)
        if not f_pol or not f_dz:
            problems.append(f"{el}: empty BASIS section parsed")
            continue

        # Target polarization angular momentum:
        # l=1 (2p) for H, He; l=2 (3d) for Li through Ar
        target_l = 1 if el in ("H", "He") else 2
        pols_in_set = [f for f in f_pol if f[1] == target_l]
        pols_in_dz = [f for f in f_dz if f[1] == target_l]

        if len(pols_in_dz) != 0:
            problems.append(f"{el}: expected 0 polarization-type functions in DZ, found {len(pols_in_dz)}")
            continue

        if len(pols_in_set) != 1:
            problems.append(f"{el}: expected 1 polarization function in {pol_name}, found {len(pols_in_set)}: {pols_in_set}")
            continue

        n, l, zeta = pols_in_set[0]
        out[el] = {"n": n, "l": l, "zeta": zeta}

    # Sanity checks from literature (Gasevic et al. JPCA 2022)
    if "H" in out and abs(out["H"]["zeta"] - 1.25) > 1e-4:
        problems.append(f"H sanity check failed: expected zeta=1.25, got {out['H']['zeta']}")
    if "O" in out and abs(out["O"]["zeta"] - 2.00) > 1e-4:
        problems.append(f"O sanity check failed: expected zeta=2.00, got {out['O']['zeta']}")

    print(json.dumps(out, indent=2))
    print()
    print("# POLARIZATION_PARAMETERS replacement for phatpsy/basis.py")
    print("# (ADF standard polarization exponents, ZORA-contracted;"
          " van Lenthe & Baerends, J. Comput. Chem. 24, 1142 (2003))")
    print("POLARIZATION_PARAMETERS = {")
    for el in ELEMENTS:
        if el in out:
            e = out[el]
            print(f'    "{el}": ({e["n"]}, {e["l"]}, {e["zeta"]:.3f}),')
    print("}")
    if problems:
        print("\n# PROBLEMS (do not use output until resolved):", file=sys.stderr)
        for p in problems:
            print("# " + p, file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: parse_adf_polarization.py <extracted-zorabasis-dir>")
    main(sys.argv[1])
