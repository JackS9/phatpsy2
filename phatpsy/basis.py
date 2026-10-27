# phatpsy/basis.py
"""
Built-in STO basis set and model potential library for PHATPSY.
Provides standard Clementi SZ (Single-Zeta) and DZ (Double-Zeta) parameters
for common atoms (N, C, O, Ag, Ni) so users do not need to manually enter
exponents for every calculation.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any


@dataclass
class BasisFunction:
    n: int
    l: int
    zeta: float

    def to_dict(self) -> Dict[str, Any]:
        return {"n": self.n, "l": self.l, "zeta": self.zeta}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BasisFunction":
        return cls(n=int(d["n"]), l=int(d["l"]), zeta=float(d["zeta"]))


@dataclass
class InitialOrbital:
    eigval: List[float] = field(default_factory=list)
    occno: List[float] = field(default_factory=list)
    coefficients: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.eigval:
            d["eigval"] = [round(float(x), 6) for x in self.eigval]
        if self.occno:
            d["occno"] = [round(float(x), 6) for x in self.occno]
        if self.coefficients:
            d["coefficients"] = [
                [int(c[0]), int(c[1])] + [round(float(x), 6) for x in c[2:]]
                for c in self.coefficients
            ]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InitialOrbital":
        return cls(
            eigval=[float(x) for x in data.get("eigval", [])],
            occno=[float(x) for x in data.get("occno", [])],
            coefficients=data.get("coefficients", []),
        )


@dataclass
class ElementData:
    symbol: str
    atomic_number: int
    description: str
    basis_name: str
    functions: List[BasisFunction]
    nvterm: int = 5
    model_potential_terms: List[Tuple[int, int, float, float]] = field(default_factory=list)
    default_orbitals: List[InitialOrbital] = field(default_factory=list)

    @property
    def nsto(self) -> int:
        """Total number of Slater-Type Orbitals (radial functions)."""
        return len(self.functions)

    @property
    def msto(self) -> int:
        """Total number of basis components including m_l degeneracy (2l + 1)."""
        return sum(2 * f.l + 1 for f in self.functions)


# Standard library of Clementi & Roetti basis sets from PHATPSY literature
STANDARD_BASIS_LIBRARY: Dict[str, Dict[str, ElementData]] = {
    "N": {
        "SZ": ElementData(
            symbol="N",
            atomic_number=7,
            description="Clementi SZ-Basis for Neutral Nitrogen",
            basis_name="SZ",
            nvterm=5,
            functions=[
                BasisFunction(1, 0, 6.6651),
                BasisFunction(2, 0, 1.9237),
                BasisFunction(2, 1, 1.9170),
            ],
            default_orbitals=[
                InitialOrbital(eigval=[-15.6704, -15.5790], occno=[1.0, 1.0]),
                InitialOrbital(eigval=[-1.1606, -0.7270], occno=[1.0, 1.0]),
                InitialOrbital(eigval=[-0.5691, 0.1912], occno=[1.0, 0.0]),
                InitialOrbital(occno=[1.0, 0.0]),
                InitialOrbital(occno=[1.0, 0.0]),
            ],
        ),
        "DZ": ElementData(
            symbol="N",
            atomic_number=7,
            description="Clementi DZ-Basis for Neutral Nitrogen",
            basis_name="DZ",
            nvterm=5,
            functions=[
                BasisFunction(1, 0, 8.49597),
                BasisFunction(1, 0, 5.98644),
                BasisFunction(2, 0, 2.26086),
                BasisFunction(2, 0, 1.42457),
                BasisFunction(2, 1, 3.24933),
                BasisFunction(2, 1, 1.49924),
            ],
        ),
    },
    "C": {
        "SZ": ElementData(
            symbol="C",
            atomic_number=6,
            description="Clementi SZ-Basis for Neutral Carbon",
            basis_name="SZ",
            nvterm=4,
            functions=[
                BasisFunction(1, 0, 5.67),
                BasisFunction(2, 0, 1.61),
                BasisFunction(2, 1, 1.57),
            ],
            default_orbitals=[
                InitialOrbital(eigval=[-11.3, -11.3], occno=[1.0, 1.0]),
                InitialOrbital(eigval=[-0.68, -0.68], occno=[1.0, 1.0]),
                InitialOrbital(eigval=[-0.40, -0.40], occno=[1.0, 0.0]),
                InitialOrbital(occno=[0.0, 0.0]),
                InitialOrbital(occno=[1.0, 0.0]),
            ],
        ),
    },
    "O": {
        "SZ": ElementData(
            symbol="O",
            atomic_number=8,
            description="Clementi SZ-Basis for Neutral Oxygen",
            basis_name="SZ",
            nvterm=4,
            functions=[
                BasisFunction(1, 0, 7.66),
                BasisFunction(2, 0, 2.25),
                BasisFunction(2, 1, 2.23),
            ],
        ),
    },
    "Ag": {
        "DZ": ElementData(
            symbol="Ag",
            atomic_number=47,
            description="Clementi DZ-Basis for Neutral Silver",
            basis_name="DZ",
            nvterm=5,
            functions=[
                BasisFunction(1, 0, 48.223),
                BasisFunction(1, 0, 33.484),
                BasisFunction(2, 0, 21.921),
                BasisFunction(2, 0, 21.015),
                BasisFunction(2, 1, 30.724),
                BasisFunction(2, 1, 20.019),
                BasisFunction(3, 0, 14.814),
                BasisFunction(3, 0, 10.125),
                BasisFunction(3, 1, 11.128),
                BasisFunction(3, 1, 8.952),
                BasisFunction(3, 2, 16.290),
                BasisFunction(3, 2, 9.223),
                BasisFunction(4, 0, 5.861),
                BasisFunction(4, 0, 3.981),
                BasisFunction(4, 1, 5.689),
                BasisFunction(4, 1, 3.576),
                BasisFunction(4, 2, 4.989),
                BasisFunction(4, 2, 2.584),
                BasisFunction(5, 0, 2.288),
                BasisFunction(5, 0, 1.265),
            ],
        ),
    },
    "Ni": {
        "SZ": ElementData(
            symbol="Ni",
            atomic_number=28,
            description="Clementi SZ-Basis for Neutral Nickel",
            basis_name="SZ",
            nvterm=4,
            functions=[
                BasisFunction(1, 0, 27.347),
                BasisFunction(2, 0, 10.1098),
                BasisFunction(2, 1, 12.0478),
                BasisFunction(3, 0, 4.98602),
                BasisFunction(3, 1, 4.69403),
                BasisFunction(3, 2, 4.17688),
                BasisFunction(4, 0, 1.42748),
            ],
        ),
    },
}


# ---------------------------------------------------------------------------
# Clementi & Roetti (1974) STO Library (Z=2 to 54)
# ---------------------------------------------------------------------------
_CLEMENTI_DATA_PATH = Path(__file__).parent / "data" / "clementi_sto.json"
_CLEMENTI_CACHE: Optional[Dict[str, Any]] = None


def _load_clementi_raw_data() -> Dict[str, Any]:
    global _CLEMENTI_CACHE
    if _CLEMENTI_CACHE is None:
        if not _CLEMENTI_DATA_PATH.exists():
            return {}
        with open(_CLEMENTI_DATA_PATH, "r", encoding="utf-8") as f:
            _CLEMENTI_CACHE = json.load(f)
    return _CLEMENTI_CACHE


# ---------------------------------------------------------------------------
# Thakkar & Koga (1999/2000) STO Library (Z=1 to 103)
# ---------------------------------------------------------------------------
_THAKKAR_DATA_PATH = Path(__file__).parent / "data" / "thakkar_sto.json"
_THAKKAR_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _load_thakkar_raw_data() -> Dict[str, Dict[str, Any]]:
    global _THAKKAR_CACHE
    if _THAKKAR_CACHE is None:
        if not _THAKKAR_DATA_PATH.exists():
            return {}
        with open(_THAKKAR_DATA_PATH, "r", encoding="utf-8") as f:
            _THAKKAR_CACHE = json.load(f)
    return _THAKKAR_CACHE


def _generate_degenerate_orbitals(l: int, eigval: float, n_electrons: float,
                                  coefficients: Optional[List[Dict[str, Any]]] = None) -> List[InitialOrbital]:
    """Generates (2l + 1) spatial orbitals distributed via Hund's rule with orbital expansion coefficients."""
    num_deg = 2 * l + 1
    orbs = []
    up_left = min(n_electrons, float(num_deg))
    down_left = max(0.0, n_electrons - num_deg)
    ml_vals = list(range(-l, l + 1))
    for i in range(num_deg):
        up = 1.0 if i < up_left else 0.0
        down = 1.0 if i < down_left else 0.0
        ml = ml_vals[i]
        c_tuples = []
        if coefficients:
            for item in coefficients:
                c_tuples.append((int(item["function_index"]), ml, float(item["c"])))
        orbs.append(InitialOrbital(eigval=[eigval, eigval], occno=[up, down], coefficients=c_tuples))
    return orbs


def get_clementi_basis(symbol: str, basis_name: str = "DZ") -> ElementData:
    """
    Retrieves the Clementi & Roetti (1974) Double-Zeta (DZ) or Single-Zeta (SZ) basis set (Z=2 to 54).
    """
    sym = symbol.strip().capitalize()
    b_type = basis_name.strip().upper()
    if b_type not in ("DZ", "SZ"):
        raise ValueError(f"Invalid Clementi basis type '{basis_name}'. Choose 'DZ' or 'SZ'.")

    clem_db = _load_clementi_raw_data()
    table_data = clem_db.get(b_type, {})
    if sym not in table_data:
        raise KeyError(
            f"Element '{sym}' not found in Clementi {b_type} library (available: Z=2..54)."
        )

    entry = table_data[sym]
    funcs = [BasisFunction(n=f["n"], l=f["l"], zeta=f["zeta"]) for f in entry["basis_functions"]]

    default_orbs = []
    for orb in entry.get("orbitals", []):
        l = orb["l"]
        eig = orb["eigval"]
        occ_tot = orb.get("occ_total", 0.0)
        coefs = orb.get("coefficients", [])
        if occ_tot > 0:
            default_orbs.extend(_generate_degenerate_orbitals(l, eig, occ_tot, coefficients=coefs))

    return ElementData(
        symbol=sym,
        atomic_number=entry["atomic_number"],
        description=entry.get("description", f"Clementi & Roetti (1974) {b_type} basis for {sym}"),
        basis_name=b_type,
        functions=funcs,
        nvterm=5,
        default_orbitals=default_orbs,
    )


def list_clementi_elements() -> List[str]:
    """Returns a list of all elements available in the Clementi & Roetti library (Z=2..54)."""
    clem_db = _load_clementi_raw_data()
    dz_table = clem_db.get("DZ", {})
    return sorted(list(dz_table.keys()), key=lambda s: dz_table[s].get("atomic_number", 0))


def get_thakkar_basis(symbol: str) -> ElementData:
    """
    Retrieves the Thakkar-Koga analytical Hartree-Fock STO basis set for an element (Z=1 to 103).
    """
    sym = symbol.strip().capitalize()
    raw_db = _load_thakkar_raw_data()
    if sym not in raw_db:
        raise KeyError(
            f"Element '{sym}' not found in Thakkar STO library. "
            f"Available elements: {sorted(list(raw_db.keys())) if raw_db else 'none (dataset missing)'}"
        )

    entry = raw_db[sym]
    funcs = [BasisFunction(n=f["n"], l=f["l"], zeta=f["zeta"]) for f in entry.get("basis_functions", entry.get("functions", []))]

    default_orbs = []
    for orb in entry.get("orbitals", []):
        l = orb["l"]
        eig = orb["eigval"]
        occ_tot = orb.get("occ_total", 0.0)
        coefs = orb.get("coefficients", [])
        if occ_tot > 0:
            default_orbs.extend(_generate_degenerate_orbitals(l, eig, occ_tot, coefficients=coefs))

    return ElementData(
        symbol=sym,
        atomic_number=entry["atomic_number"],
        description=entry.get("description", f"Thakkar-Koga STO basis for {sym}"),
        basis_name="THAKKAR",
        functions=funcs,
        nvterm=5,
        default_orbitals=default_orbs,
    )


def list_thakkar_elements() -> List[str]:
    """Returns a list of all elements available in the Thakkar STO library."""
    raw_db = _load_thakkar_raw_data()
    return sorted(list(raw_db.keys()), key=lambda s: raw_db[s].get("atomic_number", 0))


def get_element_basis(symbol: str, basis_name: Optional[str] = None) -> ElementData:
    """
    Look up element basis from the built-in library.
    Supports 'DZ' and 'SZ' (Clementi & Roetti 1974 for Z=2..54), and 'THAKKAR' (Z=1..103).
    """
    sym = symbol.strip().capitalize()
    raw_thakkar = _load_thakkar_raw_data()
    clem_db = _load_clementi_raw_data()

    if basis_name:
        b_name = basis_name.strip().upper()
        if b_name in ("THAKKAR", "THAKKAR-RHF", "KOGA", "K99", "K00"):
            return get_thakkar_basis(sym)

        if b_name in ("DZ", "SZ"):
            if b_name in clem_db and sym in clem_db[b_name]:
                return get_clementi_basis(sym, b_name)
            if sym in STANDARD_BASIS_LIBRARY and b_name in STANDARD_BASIS_LIBRARY[sym]:
                return STANDARD_BASIS_LIBRARY[sym][b_name]
            if sym in raw_thakkar:
                return get_thakkar_basis(sym)

        if sym in STANDARD_BASIS_LIBRARY and b_name in STANDARD_BASIS_LIBRARY[sym]:
            return STANDARD_BASIS_LIBRARY[sym][b_name]

        raise KeyError(
            f"Basis '{b_name}' for element '{sym}' not found. "
            f"Available basis sets: 'DZ', 'SZ' (Z=2..54), 'THAKKAR' (Z=1..103)"
        )

    # Return default:
    if sym in STANDARD_BASIS_LIBRARY:
        elem_dict = STANDARD_BASIS_LIBRARY[sym]
        if "SZ" in elem_dict:
            return elem_dict["SZ"]
        return next(iter(elem_dict.values()))

    # Fallback to Thakkar library (Z=1 to 103)
    if sym in raw_thakkar:
        return get_thakkar_basis(sym)

    # Fallback to Clementi DZ / SZ
    if "DZ" in clem_db and sym in clem_db["DZ"]:
        return get_clementi_basis(sym, "DZ")
    if "SZ" in clem_db and sym in clem_db["SZ"]:
        return get_clementi_basis(sym, "SZ")

    raise KeyError(
        f"Element '{sym}' not found in standard basis, Clementi, or Thakkar library. "
        f"Available elements in standard: {list(STANDARD_BASIS_LIBRARY.keys())}"
    )


def list_available_elements() -> Dict[str, List[str]]:
    """Returns mapping of available element symbols to available basis names."""
    res = {}
    raw_thakkar = _load_thakkar_raw_data()
    clem_db = _load_clementi_raw_data()

    # Include all 103 periodic table elements from Thakkar
    for sym in list_thakkar_elements():
        bases = []
        if sym in STANDARD_BASIS_LIBRARY:
            bases.extend(list(STANDARD_BASIS_LIBRARY[sym].keys()))
        if "DZ" in clem_db and sym in clem_db["DZ"] and "DZ" not in bases:
            bases.append("DZ")
        if "SZ" in clem_db and sym in clem_db["SZ"] and "SZ" not in bases:
            bases.append("SZ")
        bases.append("THAKKAR")
        res[sym] = bases

    # Include any custom standard entries not in Thakkar
    for sym, bases in STANDARD_BASIS_LIBRARY.items():
        if sym not in res:
            res[sym] = list(bases.keys())

    return res

