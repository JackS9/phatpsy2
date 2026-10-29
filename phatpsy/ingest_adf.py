# phatpsy/ingest_adf.py
"""
Ingestion script for Amsterdam Density Functional (ADF) Slater-Type Orbital (STO) basis sets.
Parses the uncontracted all-electron ZORA STO basis sets from:
- E. van Lenthe, E. J. Baerends, J. Comput. Chem. 24, 1142-1156 (2003).
- D. Gasević, D. Stückrath, S. Grimme, M. Bursch, J. Phys. Chem. A 126, 3826-3838 (2022).

Tiers supported:
- DZ:   Double-Zeta valence
- TZP:  Triple-Zeta valence + 1 polarization function
- TZ2P: Triple-Zeta valence + 2 polarization functions
- QZ4P: Quadruple-Zeta valence + 4 polarization functions

Outputs:
- phatpsy/data/adf_sto.json
"""
import json
import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

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
    'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og'
]

SYM_TO_Z = {sym.lower(): z for z, sym in enumerate(PERIODIC_TABLE) if sym}

ADF_FILE_ALIASES = {
    'Ds': 'Uun',
    'Rg': 'Uuu',
    'Cn': 'Uub',
    'Nh': 'Uut',
    'Fl': 'Uuq',
    'Mc': 'Uup',
    'Lv': 'Uuh',
    'Ts': 'Uus',
    'Og': 'Uuo',
}

L_MAP = {'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6}
L_CHARS = ['S', 'P', 'D', 'F', 'G', 'H', 'I']

# Electron configurations for elements 104..118 (Aufbau / standard neutral states)
# Format: (n, l, occ_total, estimated_eigval)
SUPERHEAVY_VALENCE = {
    104: [(7, 0, 2.0, -0.22), (6, 2, 2.0, -0.30)],                      # Rf: 7s^2 6d^2
    105: [(7, 0, 2.0, -0.23), (6, 2, 3.0, -0.31)],                      # Db: 7s^2 6d^3
    106: [(7, 0, 2.0, -0.24), (6, 2, 4.0, -0.32)],                      # Sg: 7s^2 6d^4
    107: [(7, 0, 2.0, -0.25), (6, 2, 5.0, -0.33)],                      # Bh: 7s^2 6d^5
    108: [(7, 0, 2.0, -0.26), (6, 2, 6.0, -0.34)],                      # Hs: 7s^2 6d^6
    109: [(7, 0, 2.0, -0.27), (6, 2, 7.0, -0.35)],                      # Mt: 7s^2 6d^7
    110: [(7, 0, 2.0, -0.28), (6, 2, 8.0, -0.36)],                      # Ds: 7s^2 6d^8
    111: [(7, 0, 2.0, -0.29), (6, 2, 9.0, -0.37)],                      # Rg: 7s^2 6d^9
    112: [(7, 0, 2.0, -0.30), (6, 2, 10.0, -0.38)],                     # Cn: 7s^2 6d^10
    113: [(7, 0, 2.0, -0.31), (6, 2, 10.0, -0.39), (7, 1, 1.0, -0.20)],# Nh: 7s^2 6d^10 7p^1
    114: [(7, 0, 2.0, -0.32), (6, 2, 10.0, -0.40), (7, 1, 2.0, -0.22)],# Fl: 7s^2 6d^10 7p^2
    115: [(7, 0, 2.0, -0.33), (6, 2, 10.0, -0.41), (7, 1, 3.0, -0.24)],# Mc: 7s^2 6d^10 7p^3
    116: [(7, 0, 2.0, -0.34), (6, 2, 10.0, -0.42), (7, 1, 4.0, -0.26)],# Lv: 7s^2 6d^10 7p^4
    117: [(7, 0, 2.0, -0.35), (6, 2, 10.0, -0.43), (7, 1, 5.0, -0.28)],# Ts: 7s^2 6d^10 7p^5
    118: [(7, 0, 2.0, -0.36), (6, 2, 10.0, -0.44), (7, 1, 6.0, -0.30)],# Og: 7s^2 6d^10 7p^6
}


def sto_overlap(na: int, za: float, nb: int, zb: float) -> float:
    """Exact analytical overlap integral between two normalized Slater-Type Orbitals."""
    ca = (2.0 * za) ** (na + 0.5) / math.sqrt(math.factorial(2 * na))
    cb = (2.0 * zb) ** (nb + 0.5) / math.sqrt(math.factorial(2 * nb))
    return ca * cb * math.factorial(na + nb) / ((za + zb) ** (na + nb + 1))


def parse_adf_basis_file(file_path: Path) -> List[Tuple[int, int, float]]:
    """
    Parses the BASIS section of an ADF all-electron basis set file.
    Returns list of (n, l, zeta).
    """
    funcs = []
    in_basis = False
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not in_basis:
                if line == "BASIS":
                    in_basis = True
                continue
            if line.upper() == "END":
                break
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                nl, zeta_str = parts[0], parts[1]
                try:
                    n = int(nl[:-1])
                    l = L_MAP[nl[-1].upper()]
                    zeta = float(zeta_str)
                    funcs.append((n, l, zeta))
                except (ValueError, KeyError):
                    continue
    return funcs


def synthesize_adf_element(
    symbol: str,
    atomic_number: int,
    tier: str,
    primitives: List[Tuple[int, int, float]],
    thakkar_db: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Constructs an ElementData dictionary for an element in the ADF basis set:
    - Basis functions mapping (index, n, l, zeta).
    - Occupied atomic orbitals projected from Thakkar Hartree-Fock STO solutions.
    - Orthonormalized virtual polarization orbitals with proper eigenvalue guesses.
    """
    basis_name = f"ADF-{tier}"
    funcs_list = [
        {"index": idx + 1, "n": n, "l": l, "zeta": zeta}
        for idx, (n, l, zeta) in enumerate(primitives)
    ]

    # Retrieve reference Thakkar Hartree-Fock atomic wavefunctions
    is_superheavy = atomic_number > 103
    ref_sym = "Lr" if is_superheavy else symbol
    th_entry = thakkar_db.get(ref_sym)
    if not th_entry:
        raise KeyError(f"Reference Thakkar entry missing for '{ref_sym}'")

    # Determine occupied shells (n, l)
    occ_shells: Dict[Tuple[int, int], Dict[str, Any]] = {}
    if not is_superheavy:
        for orb in th_entry.get("orbitals", []):
            name = orb["name"]
            n = int(name[:-1])
            l = orb["l"]
            occ_shells[(n, l)] = orb
    else:
        # For superheavies: take Lr core up to Rn + 5f^14, then add superheavy valence
        for orb in th_entry.get("orbitals", []):
            name = orb["name"]
            n = int(name[:-1])
            l = orb["l"]
            # Exclude Lr valence (7s, 7p)
            if (n, l) in ((7, 0), (7, 1), (6, 2)):
                continue
            occ_shells[(n, l)] = orb
        # Add valence
        for n, l, occ_tot, eig in SUPERHEAVY_VALENCE[atomic_number]:
            name = f"{n}{L_CHARS[l]}"
            num_deg = 2 * l + 1
            up = min(occ_tot, float(num_deg))
            down = max(0.0, occ_tot - float(num_deg))
            # Use Lr valence shape as initial projection reference
            ref_lr_orb = next((o for o in th_entry.get("orbitals", []) if o["l"] == l), None)
            occ_shells[(n, l)] = {
                "name": name,
                "l": l,
                "eigval": eig,
                "occ_up": up,
                "occ_down": down,
                "occ_total": occ_tot,
                "coefficients": (ref_lr_orb["coefficients"] if ref_lr_orb else []),
            }

    orbitals: List[Dict[str, Any]] = []

    # 1. Project occupied shells onto ADF primitives of matching l
    for (n_shell, l_shell), orb_info in occ_shells.items():
        adf_indices_l = [
            (idx + 1, p_n, p_z)
            for idx, (p_n, p_l, p_z) in enumerate(primitives)
            if p_l == l_shell
        ]
        if not adf_indices_l:
            continue

        th_funcs_l = [
            (f["index"], f["n"], f["zeta"])
            for f in th_entry["basis_functions"]
            if f["l"] == l_shell
        ]

        c_th = np.zeros(len(th_funcs_l))
        f_idx_to_pos = {f[0]: i for i, f in enumerate(th_funcs_l)}
        for item in orb_info.get("coefficients", []):
            if item["function_index"] in f_idx_to_pos:
                c_th[f_idx_to_pos[item["function_index"]]] = item["c"]

        M = len(adf_indices_l)
        S_adf = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                S_adf[i, j] = sto_overlap(
                    adf_indices_l[i][1], adf_indices_l[i][2],
                    adf_indices_l[j][1], adf_indices_l[j][2]
                )

        if np.any(c_th):
            S_cross = np.zeros((M, len(th_funcs_l)))
            for i in range(M):
                for j in range(len(th_funcs_l)):
                    S_cross[i, j] = sto_overlap(
                        adf_indices_l[i][1], adf_indices_l[i][2],
                        th_funcs_l[j][1], th_funcs_l[j][2]
                    )
            rhs = S_cross @ c_th
            c_adf = np.linalg.solve(S_adf + 1e-12 * np.eye(M), rhs)
        else:
            # Fallback: diagonal assignment to closest matching n primitive
            closest_idx = min(range(M), key=lambda i: abs(adf_indices_l[i][1] - n_shell))
            c_adf = np.zeros(M)
            c_adf[closest_idx] = 1.0

        norm = math.sqrt(float(c_adf @ S_adf @ c_adf))
        if norm > 1e-12:
            c_adf = c_adf / norm

        coeff_list = [
            {"function_index": adf_indices_l[i][0], "c": round(float(c_adf[i]), 6)}
            for i in range(M)
            if abs(c_adf[i]) > 1e-6
        ]

        orbitals.append({
            "name": orb_info["name"],
            "l": l_shell,
            "eigval": round(float(orb_info["eigval"]), 6),
            "occ_up": round(float(orb_info["occ_up"]), 6),
            "occ_down": round(float(orb_info["occ_down"]), 6),
            "occ_total": round(float(orb_info["occ_total"]), 6),
            "coefficients": coeff_list,
        })

    # 2. Orthonormalize and synthesize virtual / polarization orbitals
    # Any primitive with (n, l) not in occupied shells is a virtual polarization primitive
    occ_nl_set = set(occ_shells.keys())
    virt_prims_by_l: Dict[int, List[Tuple[int, int, float]]] = {}
    for idx, (p_n, p_l, p_z) in enumerate(primitives):
        if (p_n, p_l) not in occ_nl_set:
            virt_prims_by_l.setdefault(p_l, []).append((idx + 1, p_n, p_z))

    for p_l, v_list in sorted(virt_prims_by_l.items()):
        K = len(v_list)
        S_virt = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                S_virt[i, j] = sto_overlap(
                    v_list[i][1], v_list[i][2],
                    v_list[j][1], v_list[j][2]
                )

        # Symmetric orthogonalization (Lowdin) S^(-1/2) for mutual orthonormality
        eigvals, eigvecs = np.linalg.eigh(S_virt)
        inv_sqrt_d = [1.0 / math.sqrt(max(ev, 1e-12)) for ev in eigvals]
        S_inv_sqrt = eigvecs @ np.diag(inv_sqrt_d) @ eigvecs.T

        base_eig = 0.10 if p_l == 1 else (0.20 if p_l == 2 else 0.30)
        for k_idx in range(K):
            coeffs_k = S_inv_sqrt[:, k_idx]
            coeff_list = [
                {"function_index": v_list[i][0], "c": round(float(coeffs_k[i]), 6)}
                for i in range(K)
                if abs(coeffs_k[i]) > 1e-6
            ]
            orbitals.append({
                "name": f"VIRT_{v_list[k_idx][1]}{L_CHARS[p_l]}_{k_idx+1}",
                "l": p_l,
                "eigval": round(base_eig + 0.05 * k_idx, 4),
                "occ_up": 0.0,
                "occ_down": 0.0,
                "occ_total": 0.0,
                "coefficients": coeff_list,
            })

    return {
        "symbol": symbol,
        "atomic_number": atomic_number,
        "basis_name": basis_name,
        "tier": tier,
        "description": f"ADF ZORA {tier} STO basis for {symbol} (Z={atomic_number})",
        "basis_functions": funcs_list,
        "functions": funcs_list,
        "orbitals": orbitals,
    }


def ingest_all_adf_basis(
    zorabasis_root: Optional[Path] = None,
    output_json: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Ingests all 4 tiers (DZ, TZP, TZ2P, QZ4P) across all 118 periodic table elements.
    Saves to output_json if provided.
    """
    if zorabasis_root is None:
        zorabasis_root = Path(__file__).resolve().parent.parent / "zorabasis"
    if not zorabasis_root.is_dir():
        raise FileNotFoundError(f"ADF basis directory not found at: {zorabasis_root}")

    thakkar_path = Path(__file__).resolve().parent / "data" / "thakkar_sto.json"
    if not thakkar_path.is_file():
        raise FileNotFoundError(f"Reference Thakkar data not found at: {thakkar_path}")

    with open(thakkar_path, "r", encoding="utf-8") as f:
        thakkar_db = json.load(f)

    tiers = ["DZ", "TZP", "TZ2P", "QZ4P"]
    database: Dict[str, Dict[str, Any]] = {tier: {} for tier in tiers}

    print(f"Ingesting ADF basis sets from: {zorabasis_root} ...")
    total_entries = 0
    for tier in tiers:
        tier_dir = zorabasis_root / tier
        if not tier_dir.is_dir():
            raise FileNotFoundError(f"Missing ADF tier directory: {tier_dir}")

        for z in range(1, 119):
            sym = PERIODIC_TABLE[z]
            fn = ADF_FILE_ALIASES.get(sym, sym)
            fp = tier_dir / fn
            if not fp.is_file():
                raise FileNotFoundError(f"Missing basis file for {sym} (Z={z}) in {tier_dir}: {fn}")

            prims = parse_adf_basis_file(fp)
            if not prims:
                raise ValueError(f"No basis functions parsed from {fp}")

            entry = synthesize_adf_element(
                symbol=sym,
                atomic_number=z,
                tier=tier,
                primitives=prims,
                thakkar_db=thakkar_db,
            )
            database[tier][sym] = entry
            total_entries += 1

    print(f"Successfully processed {total_entries} entries (118 elements x 4 tiers).")

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(database, f, indent=2)
        size_kb = out_path.stat().st_size / 1024
        print(f"Wrote ADF STO database to {out_path} ({size_kb:.1f} KB)")

    return database


if __name__ == "__main__":
    dest = Path(__file__).resolve().parent / "data" / "adf_sto.json"
    ingest_all_adf_basis(output_json=dest)
