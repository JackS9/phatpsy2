# phatpsy/ingest_clementi.py
"""
Ingestion script for Clementi & Roetti (1974) Slater-Type Orbital (STO) tables.
Source:
- E. Clementi and C. Roetti, "Roothaan-Hartree-Fock Atomic Wavefunctions:
  Basis Functions and Their Coefficients for Ground and Certain Excited States
  of Neutral and Ionized Atoms, Z <= 54",
  Atomic Data and Nuclear Data Tables, Vol. 14, Nos. 3-4, pp. 177-478 (1974).
Table 42: Double-Zeta (DZ) functions (Z=2 to 54)
Table 43: Single-Zeta (SZ) functions (Z=2 to 54)
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pypdf

logger = logging.getLogger(__name__)

PDF_PATH = Path(__file__).resolve().parent.parent / "1-s2.0-S0092640X74800161-main.pdf"
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "clementi_sto.json"

PERIODIC_TABLE = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe"
]

ELEMENT_INFO = {
    2: {"name": "Helium", "shells": {"s": [1], "p": [], "d": []}, "occs": {"1s": 2.0}},
    3: {"name": "Lithium", "shells": {"s": [1, 2], "p": [], "d": []}, "occs": {"1s": 2.0, "2s": 1.0}},
    4: {"name": "Beryllium", "shells": {"s": [1, 2], "p": [], "d": []}, "occs": {"1s": 2.0, "2s": 2.0}},
    5: {"name": "Boron", "shells": {"s": [1, 2], "p": [2], "d": []}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 1.0}},
    6: {"name": "Carbon", "shells": {"s": [1, 2], "p": [2], "d": []}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 2.0}},
    7: {"name": "Nitrogen", "shells": {"s": [1, 2], "p": [2], "d": []}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 3.0}},
    8: {"name": "Oxygen", "shells": {"s": [1, 2], "p": [2], "d": []}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 4.0}},
    9: {"name": "Fluorine", "shells": {"s": [1, 2], "p": [2], "d": []}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 5.0}},
    10: {"name": "Neon", "shells": {"s": [1, 2], "p": [2], "d": []}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0}},
    11: {"name": "Sodium", "shells": {"s": [1, 2, 3], "p": [2], "d": []}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 1.0}},
    12: {"name": "Magnesium", "shells": {"s": [1, 2, 3], "p": [2], "d": []}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0}},
    13: {"name": "Aluminum", "shells": {"s": [1, 2, 3], "p": [2, 3], "d": []}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 1.0}},
    14: {"name": "Silicon", "shells": {"s": [1, 2, 3], "p": [2, 3], "d": []}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 2.0}},
    15: {"name": "Phosphorus", "shells": {"s": [1, 2, 3], "p": [2, 3], "d": []}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 3.0}},
    16: {"name": "Sulfur", "shells": {"s": [1, 2, 3], "p": [2, 3], "d": []}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 4.0}},
    17: {"name": "Chlorine", "shells": {"s": [1, 2, 3], "p": [2, 3], "d": []}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 5.0}},
    18: {"name": "Argon", "shells": {"s": [1, 2, 3], "p": [2, 3], "d": []}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0}},
    19: {"name": "Potassium", "shells": {"s": [1, 2, 3, 4], "p": [2, 3], "d": []}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 1.0}},
    20: {"name": "Calcium", "shells": {"s": [1, 2, 3, 4], "p": [2, 3], "d": []}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0}},
    21: {"name": "Scandium", "shells": {"s": [1, 2, 3, 4], "p": [2, 3], "d": [3]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 1.0}},
    22: {"name": "Titanium", "shells": {"s": [1, 2, 3, 4], "p": [2, 3], "d": [3]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 2.0}},
    23: {"name": "Vanadium", "shells": {"s": [1, 2, 3, 4], "p": [2, 3], "d": [3]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 3.0}},
    24: {"name": "Chromium", "shells": {"s": [1, 2, 3, 4], "p": [2, 3], "d": [3]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 1.0, "3d": 5.0}},
    25: {"name": "Manganese", "shells": {"s": [1, 2, 3, 4], "p": [2, 3], "d": [3]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 5.0}},
    26: {"name": "Iron", "shells": {"s": [1, 2, 3, 4], "p": [2, 3], "d": [3]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 6.0}},
    27: {"name": "Cobalt", "shells": {"s": [1, 2, 3, 4], "p": [2, 3], "d": [3]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 7.0}},
    28: {"name": "Nickel", "shells": {"s": [1, 2, 3, 4], "p": [2, 3], "d": [3]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 8.0}},
    29: {"name": "Copper", "shells": {"s": [1, 2, 3, 4], "p": [2, 3], "d": [3]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 1.0, "3d": 10.0}},
    30: {"name": "Zinc", "shells": {"s": [1, 2, 3, 4], "p": [2, 3], "d": [3]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0}},
    31: {"name": "Gallium", "shells": {"s": [1, 2, 3, 4], "p": [2, 3, 4], "d": [3]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 1.0}},
    32: {"name": "Germanium", "shells": {"s": [1, 2, 3, 4], "p": [2, 3, 4], "d": [3]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 2.0}},
    33: {"name": "Arsenic", "shells": {"s": [1, 2, 3, 4], "p": [2, 3, 4], "d": [3]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 3.0}},
    34: {"name": "Selenium", "shells": {"s": [1, 2, 3, 4], "p": [2, 3, 4], "d": [3]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 4.0}},
    35: {"name": "Bromine", "shells": {"s": [1, 2, 3, 4], "p": [2, 3, 4], "d": [3]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 5.0}},
    36: {"name": "Krypton", "shells": {"s": [1, 2, 3, 4], "p": [2, 3, 4], "d": [3]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 6.0}},
    37: {"name": "Rubidium", "shells": {"s": [1, 2, 3, 4, 5], "p": [2, 3, 4], "d": [3]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 6.0, "5s": 1.0}},
    38: {"name": "Strontium", "shells": {"s": [1, 2, 3, 4, 5], "p": [2, 3, 4], "d": [3]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 6.0, "5s": 2.0}},
    39: {"name": "Yttrium", "shells": {"s": [1, 2, 3, 4, 5], "p": [2, 3, 4], "d": [3, 4]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 6.0, "5s": 2.0, "4d": 1.0}},
    40: {"name": "Zirconium", "shells": {"s": [1, 2, 3, 4, 5], "p": [2, 3, 4], "d": [3, 4]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 6.0, "5s": 2.0, "4d": 2.0}},
    41: {"name": "Niobium", "shells": {"s": [1, 2, 3, 4, 5], "p": [2, 3, 4], "d": [3, 4]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 6.0, "5s": 1.0, "4d": 4.0}},
    42: {"name": "Molybdenum", "shells": {"s": [1, 2, 3, 4, 5], "p": [2, 3, 4], "d": [3, 4]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 6.0, "5s": 1.0, "4d": 5.0}},
    43: {"name": "Technetium", "shells": {"s": [1, 2, 3, 4, 5], "p": [2, 3, 4], "d": [3, 4]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 6.0, "5s": 2.0, "4d": 5.0}},
    44: {"name": "Ruthenium", "shells": {"s": [1, 2, 3, 4, 5], "p": [2, 3, 4], "d": [3, 4]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 6.0, "5s": 1.0, "4d": 7.0}},
    45: {"name": "Rhodium", "shells": {"s": [1, 2, 3, 4, 5], "p": [2, 3, 4], "d": [3, 4]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 6.0, "5s": 1.0, "4d": 8.0}},
    46: {"name": "Palladium", "shells": {"s": [1, 2, 3, 4, 5], "p": [2, 3, 4], "d": [3, 4]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 6.0, "5s": 0.0, "4d": 10.0}},
    47: {"name": "Silver", "shells": {"s": [1, 2, 3, 4, 5], "p": [2, 3, 4], "d": [3, 4]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 6.0, "5s": 1.0, "4d": 10.0}},
    48: {"name": "Cadmium", "shells": {"s": [1, 2, 3, 4, 5], "p": [2, 3, 4], "d": [3, 4]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 6.0, "5s": 2.0, "4d": 10.0}},
    49: {"name": "Indium", "shells": {"s": [1, 2, 3, 4, 5], "p": [2, 3, 4, 5], "d": [3, 4]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 6.0, "5s": 2.0, "4d": 10.0, "5p": 1.0}},
    50: {"name": "Tin", "shells": {"s": [1, 2, 3, 4, 5], "p": [2, 3, 4, 5], "d": [3, 4]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 6.0, "5s": 2.0, "4d": 10.0, "5p": 2.0}},
    51: {"name": "Antimony", "shells": {"s": [1, 2, 3, 4, 5], "p": [2, 3, 4, 5], "d": [3, 4]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 6.0, "5s": 2.0, "4d": 10.0, "5p": 3.0}},
    52: {"name": "Tellurium", "shells": {"s": [1, 2, 3, 4, 5], "p": [2, 3, 4, 5], "d": [3, 4]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 6.0, "5s": 2.0, "4d": 10.0, "5p": 4.0}},
    53: {"name": "Iodine", "shells": {"s": [1, 2, 3, 4, 5], "p": [2, 3, 4, 5], "d": [3, 4]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 6.0, "5s": 2.0, "4d": 10.0, "5p": 5.0}},
    54: {"name": "Xenon", "shells": {"s": [1, 2, 3, 4, 5], "p": [2, 3, 4, 5], "d": [3, 4]}, "occs": {"1s": 2.0, "2s": 2.0, "2p": 6.0, "3s": 2.0, "3p": 6.0, "4s": 2.0, "3d": 10.0, "4p": 6.0, "5s": 2.0, "4d": 10.0, "5p": 6.0}}
}

ALIASES = {
    "Helium": ["HELIUM", "PELIUP"],
    "Lithium": ["LITHIUM", "LITHIUH"],
    "Beryllium": ["BERYLLIUM", "BEEYLLIUM"],
    "Boron": ["BORON", "BDRON", "50~0~", "BO~ON"],
    "Carbon": ["CARBON", "CAR8ON"],
    "Nitrogen": ["NITROGEN", "MTROGEN", "KIIROGEN"],
    "Oxygen": ["OXYGEN", "CXYGEE"],
    "Fluorine": ["FLUORINE", "FLJORINE"],
    "Neon": ["NEON", "HEON", "hEON"],
    "Sodium": ["SODIUM", "SO)IUH", "SOOIUM", "SODIUH"],
    "Magnesium": ["MAGNESIUM", "MAGNESIUN", "/'IA3NESIUM"],
    "Aluminum": ["ALUMINUM", "ALUMINUH", "ALUMINU~"],
    "Silicon": ["SILICON", "SILICD~", "SILICDN"],
    "Phosphorus": ["PHOSPHORUS", "PHOSPHORJS", "PHOSPHGRUS"],
    "Sulfur": ["SULFUR", "SULFJR"],
    "Chlorine": ["CHLORINE", "CHLORIN~"],
    "Argon": ["ARGON", "~GO~", "AR6ON"],
    "Potassium": ["POTASSIUM", "POTASSIUN", "PDIASSIUK", "PD/ASSIUM"],
    "Calcium": ["CALCIUM", "CALCIUN"],
    "Scandium": ["SCANDIUM", "SCANDI", "SCANDIUq"],
    "Titanium": ["TITANIUM", "TITANIUH", "TIIANIU~"],
    "Vanadium": ["VANADIUM", "VA~A31UM", "VANADIUH"],
    "Chromium": ["CHROMIUM", "CH~O~IU~", "CHROMIJ~", "CHROMIUN"],
    "Manganese": ["MANGANESE", "FANGANESE"],
    "Iron": ["IRON", "IR3N"],
    "Cobalt": ["COBALT", "C0BALT"],
    "Nickel": ["NICKEL", "NICXEL", "NIBKEL"],
    "Copper": ["COPPER", "C3PPER"],
    "Zinc": ["ZINC"],
    "Gallium": ["GALLIUM", "G#LLIU", "GALLIUH", "G#LLIU~"],
    "Germanium": ["GERMANIUM", "GERMANIUH", "EERMANIU~", "GERMANIJM"],
    "Arsenic": ["ARSENIC"],
    "Selenium": ["SELENIUM", "SELENIU~"],
    "Bromine": ["BROMINE", "BROM", "BROMIN~", "EROHIhE"],
    "Krypton": ["KRYPTON", "KRYPTs"],
    "Rubidium": ["RUBIDIUM", "PUBID", "PUBIDIU~", "RU31DIUM"],
    "Strontium": ["STRONTIUM", "STRONTIUN", "STRONTIU~", "ST~ONTIJH"],
    "Yttrium": ["YTTRIUM", "YTTRIUN", "YTTRIUH"],
    "Zirconium": ["ZIRCONIUM", "ZIRCONIUH", "ZIRCONIJ~", "ZlRCDNI3M"],
    "Niobium": ["NIOBIUM", "NIOBIUH"],
    "Molybdenum": ["MOLYBDENUM", "MDLY3DE~UM", "MOLYBDENUH", "MOLYBDENIUH"],
    "Technetium": ["TECHNETIUM", "TECHNETIUH", "TECfflETIJM"],
    "Ruthenium": ["RUTHENIUM", "RUTHENIUH", "RUTHs"],
    "Rhodium": ["RHODIUM", "RHODIUN", "RHODIUH", "R~SDIU"],
    "Palladium": ["PALLADIUM", "PALLADIUH", "PALLADIJq"],
    "Silver": ["SILVER"],
    "Cadmium": ["CADMIUM", "CADMIUH"],
    "Indium": ["INDIUM", "INDIUN", "INDIU~", "l~IJH"],
    "Tin": ["TIN", "TI~", "~IN"],
    "Antimony": ["ANTIMONY", "ANTIMON~", "ANTIMCNY"],
    "Tellurium": ["TELLURIUM", "TELLURIJM", "TELLU~IJq"],
    "Iodine": ["IODINE", "IODIN~"],
    "Xenon": ["XENON", "XENO~"]
}


def clean_num(tok: Optional[str]) -> Optional[float]:
    """Cleans OCR artifacts and parses a float number."""
    if not tok:
        return None
    raw = tok.replace("~", "").replace("§", "").replace("*", "")
    raw = raw.replace("O", "0").replace("o", "0").replace("D", "").replace("d", "")
    raw = raw.replace("C", "0").replace("c", "0").replace("S", "5").replace("s", "5")
    raw = raw.replace("Z", "2").replace("z", "2").replace("B", "8").replace("b", "6")
    raw = raw.replace("I", "1").replace("i", "1").replace("l", "1")
    raw = raw.replace(",", ".")
    m = re.search(r"[-+]?\d*\.?\d+", raw)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


def normalize_toks(raw_toks: List[str]) -> List[str]:
    """Merges split numeric and tag tokens resulting from OCR spaces."""
    toks = []
    i = 0
    while i < len(raw_toks):
        t = raw_toks[i]
        # Merge space in numbers like -0. C0158
        if t in ("-0.", "0.", "-.", "+.") and i + 1 < len(raw_toks):
            toks.append(t + raw_toks[i+1])
            i += 2
            continue
        # Merge separated tags like 3 ) or 3 ~ or 2 8
        if len(t) == 1 and t in "12345IiZzkK~" and i + 1 < len(raw_toks):
            nxt = raw_toks[i+1]
            if len(nxt) == 1 and nxt in "SPDsfpdFf59$~()80":
                toks.append(t + nxt)
                i += 2
                continue
        toks.append(t)
        i += 1
    return toks


def parse_tag(tok: str, looking_for_s: bool = True, has_p_on_line: bool = False, has_4d: bool = False) -> Optional[Tuple[int, str]]:
    """Identifies principal quantum number n and orbital angular momentum (S, P, D)."""
    m = re.match(r"^([1-5Iil6bkK~zZ])([SPDsfp05o~9\$FfDdOo\)80])?$", tok)
    if not m:
        return None
    n_c, l_c = m.group(1), m.group(2)
    if n_c in ("1", "I", "l"):
        n = 1
    elif n_c in ("2", "Z", "z"):
        n = 2
    elif n_c in ("3", "~"):
        n = 3
    elif n_c in ("4", "6", "b", "k", "K"):
        n = 4
    elif n_c == "5":
        n = 5
    elif n_c == "i":
        n = 2 if not looking_for_s else 1
    else:
        n = 1

    if has_p_on_line:
        am = "D"
    elif l_c in ("S", "5", "9", "$", "s", "8"):
        am = "S"
    elif l_c in ("P", "p", "F", "f", ")"):
        am = "P"
    elif l_c in ("D", "d", "0", "O", "o"):
        if n == 4 and not has_4d:
            am = "P"
        else:
            am = "D"
    elif l_c == "~":
        am = "S" if looking_for_s else "P"
    else:
        am = "S" if looking_for_s else "P"

    if am == "P" and n < 2:
        n = 2
    elif am == "D" and n < 3:
        n = 3
    return n, am


def get_page_lines_with_pos(page) -> List[Tuple[float, List[Tuple[float, str]]]]:
    """Groups text tokens on a PDF page into lines, retaining x coordinates."""
    toks = []
    def visitor(text, cm, tm, font_dict, font_size):
        if text.strip():
            toks.append((tm[4], tm[5], text.strip()))
    page.extract_text(visitor_text=visitor)
    lines = []
    curr_y = None
    curr_toks = []
    for x, y, t in sorted(toks, key=lambda it: (-it[1], it[0])):
        if curr_y is None or abs(y - curr_y) > 3.0:
            if curr_toks:
                curr_toks.sort(key=lambda it: it[0])
                lines.append((curr_y, curr_toks))
            curr_y = y
            curr_toks = [(x, t)]
        else:
            curr_toks.append((x, t))
    if curr_toks:
        curr_toks.sort(key=lambda it: it[0])
        lines.append((curr_y, curr_toks))
    return lines



def parse_element_data(lines: List[str], num_s: int, num_p: int, num_d: int,
                       is_dz: bool = True, has_4d: bool = False) -> Dict[str, Any]:
    """Parses orbital energies, basis functions, and expansion coefficients."""
    n_funcs_s = 2 * num_s if is_dz else num_s
    n_funcs_p = 2 * num_p if is_dz else num_p
    n_funcs_d = 2 * num_d if is_dz else num_d

    s_eigvals = []
    p_eigvals = []
    d_eigvals = []

    # 1. Parse orbital energies
    for l in lines:
        if any(term in l.upper() for term in ["BASIS/ORB", "BASIS/3", "BASIS/O", "8ASI5", "EASISI", "8~SIS", "8&SIS"]):
            neg_nums = []
            for tok in l.split():
                v = clean_num(tok)
                if v is not None and v < 0:
                    neg_nums.append(v)
            if len(neg_nums) >= num_s and not s_eigvals:
                s_eigvals = neg_nums[:num_s]
                rem = neg_nums[num_s:]
                if len(rem) >= num_p and num_p > 0 and not p_eigvals:
                    p_eigvals = rem[:num_p]
                    rem = rem[num_p:]
                if len(rem) >= num_d and num_d > 0 and not d_eigvals:
                    d_eigvals = rem[:num_d]
            elif len(neg_nums) >= (num_p + num_d) and not p_eigvals:
                p_eigvals = neg_nums[:num_p]
                if num_d > 0:
                    d_eigvals = neg_nums[num_p:num_p+num_d]

    # 2. Parse basis functions and coefficients
    s_funcs = []
    p_funcs = []
    d_funcs = []

    s_coefs = [[] for _ in range(num_s)]
    p_coefs = [[] for _ in range(num_p)]
    d_coefs = [[] for _ in range(num_d)]

    for l in lines:
        if any(term in l.upper() for term in ["BASIS/", "8ASI", "EASIS", "8~SIS", "8&SIS", "T.E.", "7.E.", "K.E.", "V.T.", "S IS", "P 2P"]):
            continue

        toks = normalize_toks(l.split())
        idx = 0
        has_p_on_line = False

        while idx < len(toks):
            tok = toks[idx]
            looking_for_s = (len(s_funcs) < n_funcs_s)
            tag_res = parse_tag(tok, looking_for_s, has_p_on_line, has_4d)
            zeta = None

            if tag_res:
                n, am = tag_res
                if idx + 1 < len(toks):
                    zeta = clean_num(toks[idx+1])
                    if zeta is not None and zeta > 0:
                        idx += 2
            else:
                m_m = re.match(r"^([1-5Iil6bkK~zZ][SPDsfp05o~9\$FfDdOo\)80])([0-9\.,]+)$", tok)
                if m_m:
                    tag_res = parse_tag(m_m.group(1), looking_for_s, has_p_on_line, has_4d)
                    if tag_res:
                        n, am = tag_res
                        zeta = clean_num(m_m.group(2))
                        idx += 1

            if tag_res and zeta is not None and zeta > 0:
                if am == 'S' and len(s_funcs) < n_funcs_s:
                    s_funcs.append((n, 0, zeta))
                    for orb_i in range(num_s):
                        if idx < len(toks):
                            c = clean_num(toks[idx])
                            s_coefs[orb_i].append(c if c is not None else 0.0)
                            idx += 1
                    continue
                elif am == 'P' and len(p_funcs) < n_funcs_p:
                    p_funcs.append((n, 1, zeta))
                    has_p_on_line = True
                    for orb_i in range(num_p):
                        if idx < len(toks):
                            c = clean_num(toks[idx])
                            p_coefs[orb_i].append(c if c is not None else 0.0)
                            idx += 1
                    continue
                elif am == 'D' and len(d_funcs) < n_funcs_d:
                    d_funcs.append((n, 2, zeta))
                    for orb_i in range(num_d):
                        if idx < len(toks):
                            c = clean_num(toks[idx])
                            d_coefs[orb_i].append(c if c is not None else 0.0)
                            idx += 1
                    continue
            idx += 1

    return {
        "s_funcs": s_funcs, "p_funcs": p_funcs, "d_funcs": d_funcs,
        "s_eigvals": s_eigvals, "p_eigvals": p_eigvals, "d_eigvals": d_eigvals,
        "s_coefs": s_coefs, "p_coefs": p_coefs, "d_coefs": d_coefs
    }


def parse_total_energy(lines: List[str], z: int) -> Optional[float]:
    """Extracts total Hartree-Fock energy with appropriate scale factor for atomic number Z."""
    # The order of magnitude of neutral atom HF energy is strictly determined by Z:
    # Z=2..3 (He, Li): ~2.8 to 7.4  -> 10^1
    # Z=4..9 (Be..F): ~14 to 99    -> 10^2
    # Z=10..31 (Ne..Ga): ~128 to 1923 -> 10^3
    # Z=32..54 (Ge..Xe): ~2075 to 7232 -> 10^4
    if z <= 3:
        target_exp = 1
    elif z <= 9:
        target_exp = 2
    elif z <= 31:
        target_exp = 3
    else:
        target_exp = 4

    for l in lines:
        s = l.replace(" ", "").replace("@", "4")
        for prefix in [r"(?:[T7Ili1]|P)[\.,]*[E8B]", r"[Kk][\.,]*[E8B]"]:
            m = re.search(prefix + r"[\.,]*[=:\-~\*]+\s*[-+]?([0-9OZbI~il\[\]\(\)\.\,]+)", s)
            if m:
                raw = m.group(1).replace("O", "0").replace("o", "0").replace("b", "6").replace("Z", "2")
                raw = raw.replace("[", "1").replace("]", "1").replace("(", "1").replace(")", "1").replace("~", "4")
                raw = raw.replace(",", ".")
                m_num = re.search(r"(\d+\.\d+)", raw)
                if m_num:
                    base = float(m_num.group(1))
                    if base < 1.0:
                        val = - (base * (10 ** target_exp))
                    else:
                        val = - (base * (10 ** max(0, target_exp - 1)))
                    return round(val, 6)
    return None


def build_element_dict(symbol: str, z: int, energy: Optional[float],
                       basis_data: Dict[str, Any], is_dz: bool) -> Dict[str, Any]:
    """Assembles basis functions and orbital-centric representations for an element."""
    info = ELEMENT_INFO[z]
    s_shells = info["shells"]["s"]
    p_shells = info["shells"]["p"]
    d_shells = info["shells"]["d"]
    occs = info["occs"]

    basis_functions = []
    # 1. S functions
    for f in basis_data["s_funcs"]:
        idx = len(basis_functions) + 1
        basis_functions.append({"index": idx, "n": f[0], "l": f[1], "zeta": f[2]})
    s_start_idx = 1
    s_count = len(basis_data["s_funcs"])

    # 2. P functions
    p_start_idx = len(basis_functions) + 1
    for f in basis_data["p_funcs"]:
        idx = len(basis_functions) + 1
        basis_functions.append({"index": idx, "n": f[0], "l": f[1], "zeta": f[2]})
    p_count = len(basis_data["p_funcs"])

    # 3. D functions
    d_start_idx = len(basis_functions) + 1
    for f in basis_data["d_funcs"]:
        idx = len(basis_functions) + 1
        basis_functions.append({"index": idx, "n": f[0], "l": f[1], "zeta": f[2]})
    d_count = len(basis_data["d_funcs"])

    orbitals = []
    # Build S orbitals
    for i, n_val in enumerate(s_shells):
        name = f"{n_val}S"
        eig = basis_data["s_eigvals"][i] if i < len(basis_data["s_eigvals"]) else 0.0
        occ_tot = occs.get(f"{n_val}s", 2.0)
        occ_up = min(occ_tot, 1.0)
        occ_down = max(0.0, occ_tot - 1.0)
        
        coef_list = []
        if i < len(basis_data["s_coefs"]):
            c_row = basis_data["s_coefs"][i]
            for f_i, c_val in enumerate(c_row):
                if f_i < s_count:
                    coef_list.append({"function_index": s_start_idx + f_i, "c": c_val})
        
        orbitals.append({
            "name": name,
            "l": 0,
            "eigval": eig,
            "occ_up": occ_up,
            "occ_down": occ_down,
            "occ_total": occ_tot,
            "coefficients": coef_list
        })

    # Build P orbitals
    for i, n_val in enumerate(p_shells):
        name = f"{n_val}P"
        eig = basis_data["p_eigvals"][i] if i < len(basis_data["p_eigvals"]) else 0.0
        occ_tot = occs.get(f"{n_val}p", 6.0)
        occ_spatial = occ_tot / 3.0
        occ_up = min(occ_spatial, 1.0)
        occ_down = max(0.0, occ_spatial - 1.0)

        coef_list = []
        if i < len(basis_data["p_coefs"]):
            c_row = basis_data["p_coefs"][i]
            for f_i, c_val in enumerate(c_row):
                if f_i < p_count:
                    coef_list.append({"function_index": p_start_idx + f_i, "c": c_val})

        orbitals.append({
            "name": name,
            "l": 1,
            "eigval": eig,
            "occ_up": occ_up,
            "occ_down": occ_down,
            "occ_total": occ_tot,
            "coefficients": coef_list
        })

    # Build D orbitals
    for i, n_val in enumerate(d_shells):
        name = f"{n_val}D"
        eig = basis_data["d_eigvals"][i] if i < len(basis_data["d_eigvals"]) else 0.0
        occ_tot = occs.get(f"{n_val}d", 10.0)
        occ_spatial = occ_tot / 5.0
        occ_up = min(occ_spatial, 1.0)
        occ_down = max(0.0, occ_spatial - 1.0)

        coef_list = []
        if i < len(basis_data["d_coefs"]):
            c_row = basis_data["d_coefs"][i]
            for f_i, c_val in enumerate(c_row):
                if f_i < d_count:
                    coef_list.append({"function_index": d_start_idx + f_i, "c": c_val})

        orbitals.append({
            "name": name,
            "l": 2,
            "eigval": eig,
            "occ_up": occ_up,
            "occ_down": occ_down,
            "occ_total": occ_tot,
            "coefficients": coef_list
        })

    b_type = "DZ" if is_dz else "SZ"
    return {
        "symbol": symbol,
        "atomic_number": z,
        "name": info["name"],
        "basis_name": b_type,
        "description": f"Clementi & Roetti (1974) {b_type} Basis for Neutral {info['name']}",
        "energy": energy,
        "basis_functions": basis_functions,
        "orbitals": orbitals
    }


def extract_table(reader: pypdf.PdfReader, start_p: int, end_p: int, is_dz: bool) -> Dict[str, Any]:
    """Extracts all 53 neutral elements (Z=2..54) from a Clementi table."""
    elements = {}

    # Collect all pages and tokenized lines
    all_raw_lines = []
    for pno in range(start_p, end_p):
        lines_pos = get_page_lines_with_pos(reader.pages[pno])
        for line_idx, (y, toks) in enumerate(lines_pos):
            line_str = " ".join(t for _, t in toks)
            all_raw_lines.append((pno + 1, line_idx, y, toks, line_str))

    # 1. Locate start of each element Z=2..54
    elem_starts = {}
    curr_search = 0
    for z in range(2, 55):
        name = ELEMENT_INFO[z]["name"]
        aliases = ALIASES[name]
        pat = re.compile(r"(?:^|[^\w])(?:" + "|".join(re.escape(a) for a in aliases) + r")(?:[^\w]|$)", re.IGNORECASE)
        for idx in range(curr_search, len(all_raw_lines)):
            pno, l_idx, y, toks, text = all_raw_lines[idx]
            if any(term in text for term in ["TABLE", "ATOMIC DATA", "SEE PAGE"]):
                continue
            if pat.search(text):
                elem_starts[z] = idx
                if z == 3 and any(b in text for b in ["BEEYLLIUM", "BERYLLIUM"]):
                    curr_search = idx
                else:
                    curr_search = idx + 1
                break

    print(f"Located {len(elem_starts)} / 53 elements for {'DZ' if is_dz else 'SZ'}.")

    # 2. Extract block for each element
    for z in range(2, 55):
        sym = PERIODIC_TABLE[z]
        info = ELEMENT_INFO[z]
        num_s = len(info["shells"]["s"])
        num_p = len(info["shells"]["p"])
        num_d = len(info["shells"]["d"])
        has_4d = (4 in info["shells"]["d"])

        if z == 3 or z == 4:
            start_idx = elem_starts[3]
            end_idx = elem_starts.get(5, len(all_raw_lines))
        else:
            start_idx = elem_starts[z]
            end_idx = elem_starts.get(z + 1, len(all_raw_lines))

        # Build lines for this element
        elem_lines = []
        for idx in range(start_idx, end_idx):
            pno, l_idx, y, toks, text = all_raw_lines[idx]
            if z == 3:
                # Lithium: left side of page (x < 300)
                l_toks = [t for x, t in toks if x < 300]
                if l_toks:
                    elem_lines.append(" ".join(l_toks))
            elif z == 4:
                # Beryllium: right side of page (x >= 300)
                r_toks = [t for x, t in toks if x >= 300]
                if r_toks:
                    elem_lines.append(" ".join(r_toks))
            else:
                elem_lines.append(text)

        energy = parse_total_energy(elem_lines[:4], z)
        b_data = parse_element_data(elem_lines, num_s, num_p, num_d, is_dz=is_dz, has_4d=has_4d)
        elements[sym] = build_element_dict(sym, z, energy, b_data, is_dz=is_dz)

    return elements


def ingest_clementi_tables() -> Dict[str, Any]:
    """Ingests both Table 42 (Double-Zeta) and Table 43 (Single-Zeta) from PDF."""
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Clementi PDF not found at {PDF_PATH}")

    reader = pypdf.PdfReader(str(PDF_PATH))
    print(f"Opened Clementi PDF ({len(reader.pages)} pages).")

    print("\n--- Ingesting Table 42 (Double-Zeta, Z=2..54) ---")
    dz_elements = extract_table(reader, start_p=251, end_p=268, is_dz=True)

    print("\n--- Ingesting Table 43 (Single-Zeta, Z=2..54) ---")
    sz_elements = extract_table(reader, start_p=268, end_p=280, is_dz=False)

    data = {
        "metadata": {
            "source": "E. Clementi and C. Roetti, At. Data Nucl. Data Tables 14, 177-478 (1974)",
            "tables": {
                "DZ": "Table 42: Double-Zeta functions (Z=2 to 54)",
                "SZ": "Table 43: Single-Zeta functions (Z=2 to 54)"
            },
            "format": "orbital-centric",
            "num_elements": 53
        },
        "DZ": dz_elements,
        "SZ": sz_elements
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\nSuccessfully wrote Clementi library to {OUTPUT_PATH}")
    print(f"Total DZ elements: {len(dz_elements)} / 53")
    print(f"Total SZ elements: {len(sz_elements)} / 53")
    return data


if __name__ == "__main__":
    ingest_clementi_tables()
