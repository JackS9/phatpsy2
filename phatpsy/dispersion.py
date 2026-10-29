# phatpsy/dispersion.py
"""
Grimme DFT-D3(BJ) Empirical Dispersion Correction for PHATPSY 2.0.

Implements DFT-D3 dispersion with Becke-Johnson (BJ) rational damping:

    E_disp = - sum_{A < B} [ s_6 * C_6^{AB} / (R_{AB}^6 + (R_0^{AB})^6)
                           + s_8 * C_8^{AB} / (R_{AB}^8 + (R_0^{AB})^8) ]

where the cutoff radius is defined via the Becke-Johnson rational damping formula:

    R_0^{AB} = a_1 * sqrt(C_8^{AB} / C_6^{AB}) + a_2

with analytical Cartesian nuclear gradients:

    dE_disp / dR_A = sum_{B != A} [ 6 * s_6 * C_6^{AB} * R^5 / (R^6 + (R_0^{AB})^6)^2
                                  + 8 * s_8 * C_8^{AB} * R^7 / (R^8 + (R_0^{AB})^8)^2 ] * (R_A - R_B) / R

References:
    - S. Grimme, J. Antony, S. Ehrlich, H. Krieg, J. Chem. Phys. 132, 154104 (2010).
    - S. Grimme, S. Ehrlich, L. Goerigk, J. Comput. Chem. 32, 1456-1465 (2011).
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np


# -----------------------------------------------------------------------------
# Atomic C6 coefficients (Hartree * Bohr^6) and vdW cutoff radii R0 (Bohr)
# Source: Grimme et al. J. Comput. Chem. 32, 1456 (2011)
# -----------------------------------------------------------------------------
D3_ATOMIC_C6: Dict[str, float] = {
    "H": 1.4514,
    "He": 0.5878,
    "Li": 57.00,
    "Be": 39.00,
    "B": 29.00,
    "C": 29.80,
    "N": 18.50,
    "O": 15.60,
    "F": 9.52,
    "Ne": 6.38,
    "Na": 155.0,
    "Mg": 110.0,
    "Al": 107.0,
    "Si": 111.0,
    "P": 78.50,
    "S": 55.50,
    "Cl": 50.70,
    "Ar": 35.80,
    "K": 380.0,
    "Ca": 260.0,
    "Sc": 200.0,
    "Ti": 170.0,
    "V": 150.0,
    "Cr": 140.0,
    "Mn": 130.0,
    "Fe": 126.0,
    "Co": 125.0,
    "Ni": 124.0,
    "Cu": 120.0,
    "Zn": 110.0,
    "Br": 80.0,
    "I": 140.0,
}

D3_ATOMIC_R0: Dict[str, float] = {
    "H": 2.457,
    "He": 2.200,
    "Li": 4.500,
    "Be": 4.200,
    "B": 4.100,
    "C": 4.440,
    "N": 4.090,
    "O": 3.860,
    "F": 3.600,
    "Ne": 3.400,
    "Na": 5.200,
    "Mg": 4.900,
    "Al": 4.800,
    "Si": 4.700,
    "P": 4.500,
    "S": 4.300,
    "Cl": 4.180,
    "Ar": 4.000,
    "K": 5.800,
    "Ca": 5.500,
    "Sc": 5.200,
    "Ti": 5.000,
    "V": 4.900,
    "Cr": 4.850,
    "Mn": 4.820,
    "Fe": 4.810,
    "Co": 4.805,
    "Ni": 4.800,
    "Cu": 4.750,
    "Zn": 4.700,
    "Br": 4.500,
    "I": 4.900,
}

# Standard Becke-Johnson damping parameters (s6, s8, a1, a2) in atomic units
# a2 is converted from Angstrom to Bohr (1 Angstrom = 1.88972612 Bohr)
D3_BJ_PARAMETERS: Dict[str, Dict[str, float]] = {
    "pbe": {
        "s6": 1.0000,
        "s8": 0.7875,
        "a1": 0.4289,
        "a2": 4.4407,  # 2.350 Angstrom
    },
    "b88": {
        "s6": 1.0000,
        "s8": 1.0940,
        "a1": 0.4298,
        "a2": 4.2359,
    },
    "hf": {
        "s6": 1.0000,
        "s8": 0.8500,
        "a1": 0.4000,
        "a2": 4.5000,
    },
    "slater": {
        "s6": 1.0000,
        "s8": 0.5000,
        "a1": 0.4500,
        "a2": 4.5000,
    },
}


def _extract_coords_and_symbols(
    geometry: Any,
) -> Tuple[np.ndarray, List[str]]:
    """Helper to convert various geometry specifications into (coords, symbols)."""
    coords_list = []
    if hasattr(geometry, "geometry") and isinstance(geometry.geometry, list):
        geometry = geometry.geometry
    elif hasattr(geometry, "atoms") and isinstance(geometry.atoms, list):
        geometry = geometry.atoms
    symbols_list = []

    if isinstance(geometry, list):
        for item in geometry:
            if hasattr(item, "element") and hasattr(item, "xyz"):
                # AtomConfig object
                symbols_list.append(str(item.element).strip().capitalize())
                coords_list.append(item.xyz)
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                # (sym, [x, y, z])
                symbols_list.append(str(item[0]).strip().capitalize())
                coords_list.append(item[1])
            elif isinstance(item, dict) and "element" in item and "xyz" in item:
                # dict format
                symbols_list.append(str(item["element"]).strip().capitalize())
                coords_list.append(item["xyz"])
            else:
                raise ValueError(f"Unrecognized atom geometry item: {item}")
    else:
        raise ValueError(f"Unsupported geometry type: {type(geometry)}")

    coords = np.asarray(coords_list, dtype=np.float64)
    return coords, symbols_list


def compute_dispersion_energy(
    geometry: Any,
    functional: str = "pbe",
    s6: Optional[float] = None,
    s8: Optional[float] = None,
    a1: Optional[float] = None,
    a2: Optional[float] = None,
) -> float:
    """
    Compute DFT-D3(BJ) dispersion energy in Hartrees (atomic units).

    Parameters:
        geometry: List of AtomConfig, (element, xyz) tuples, or dicts with coordinates in Bohrs.
        functional: Underlying functional ('pbe', 'b88', 'hf', 'slater').
        s6, s8, a1, a2: Optional override parameters for D3(BJ) damping.

    Returns:
        e_disp: Dispersion energy in Hartrees (strictly <= 0.0).
    """
    coords, symbols = _extract_coords_and_symbols(geometry)
    natom = len(symbols)
    if natom < 2:
        return 0.0

    fn_key = functional.lower().strip()
    params = D3_BJ_PARAMETERS.get(fn_key, D3_BJ_PARAMETERS["pbe"])
    p_s6 = s6 if s6 is not None else params["s6"]
    p_s8 = s8 if s8 is not None else params["s8"]
    p_a1 = a1 if a1 is not None else params["a1"]
    p_a2 = a2 if a2 is not None else params["a2"]

    e_disp = 0.0

    for i in range(natom):
        sym_i = symbols[i]
        c6_i = D3_ATOMIC_C6.get(sym_i, 15.0)
        r0_i = D3_ATOMIC_R0.get(sym_i, 4.0)

        for j in range(i + 1, natom):
            sym_j = symbols[j]
            c6_j = D3_ATOMIC_C6.get(sym_j, 15.0)
            r0_j = D3_ATOMIC_R0.get(sym_j, 4.0)

            rij_vec = coords[i] - coords[j]
            r = float(np.linalg.norm(rij_vec))
            if r < 1.0e-10:
                continue

            c6_ij = np.sqrt(c6_i * c6_j)
            r0_ij = p_a1 * np.sqrt(r0_i * r0_j) + p_a2
            c8_ij = 3.0 * c6_ij * (r0_i * r0_j)

            term6 = p_s6 * c6_ij / (r**6 + r0_ij**6)
            term8 = p_s8 * c8_ij / (r**8 + r0_ij**8)

            e_disp -= (term6 + term8)

    return float(e_disp)


def compute_dispersion_gradient(
    geometry: Any,
    functional: str = "pbe",
    s6: Optional[float] = None,
    s8: Optional[float] = None,
    a1: Optional[float] = None,
    a2: Optional[float] = None,
) -> np.ndarray:
    """
    Compute analytical Cartesian nuclear gradient of DFT-D3(BJ) dispersion energy.

    Parameters:
        geometry: List of AtomConfig, (element, xyz) tuples, or dicts with coordinates in Bohrs.
        functional: Underlying functional ('pbe', 'b88', 'hf', 'slater').
        s6, s8, a1, a2: Optional override parameters for D3(BJ) damping.

    Returns:
        grad: Nuclear gradient array of shape (N, 3) in Hartree/Bohr.
    """
    coords, symbols = _extract_coords_and_symbols(geometry)
    natom = len(symbols)
    grad = np.zeros_like(coords)
    if natom < 2:
        return grad

    fn_key = functional.lower().strip()
    params = D3_BJ_PARAMETERS.get(fn_key, D3_BJ_PARAMETERS["pbe"])
    p_s6 = s6 if s6 is not None else params["s6"]
    p_s8 = s8 if s8 is not None else params["s8"]
    p_a1 = a1 if a1 is not None else params["a1"]
    p_a2 = a2 if a2 is not None else params["a2"]

    for i in range(natom):
        sym_i = symbols[i]
        c6_i = D3_ATOMIC_C6.get(sym_i, 15.0)
        r0_i = D3_ATOMIC_R0.get(sym_i, 4.0)

        for j in range(natom):
            if i == j:
                continue
            sym_j = symbols[j]
            c6_j = D3_ATOMIC_C6.get(sym_j, 15.0)
            r0_j = D3_ATOMIC_R0.get(sym_j, 4.0)

            rij_vec = coords[i] - coords[j]
            r = float(np.linalg.norm(rij_vec))
            if r < 1.0e-10:
                continue

            c6_ij = np.sqrt(c6_i * c6_j)
            r0_ij = p_a1 * np.sqrt(r0_i * r0_j) + p_a2
            c8_ij = 3.0 * c6_ij * (r0_i * r0_j)

            # d/dr [ -(term6 + term8) ] = 6 * s6 * C6 * r^5 / (r^6 + r0^6)^2
            #                           + 8 * s8 * C8 * r^7 / (r^8 + r0^8)^2
            de_dr = (
                6.0 * p_s6 * c6_ij * (r**5) / ((r**6 + r0_ij**6) ** 2)
                + 8.0 * p_s8 * c8_ij * (r**7) / ((r**8 + r0_ij**8) ** 2)
            )

            grad[i] += de_dr * (rij_vec / r)

    return grad


class DispersionModel:
    """Class wrapper for DFT-D3(BJ) dispersion calculations."""

    def __init__(
        self,
        functional: str = "pbe",
        s6: Optional[float] = None,
        s8: Optional[float] = None,
        a1: Optional[float] = None,
        a2: Optional[float] = None,
    ):
        self.functional = functional
        self.s6 = s6
        self.s8 = s8
        self.a1 = a1
        self.a2 = a2

    def energy(self, geometry: Any) -> float:
        return compute_dispersion_energy(
            geometry,
            functional=self.functional,
            s6=self.s6,
            s8=self.s8,
            a1=self.a1,
            a2=self.a2,
        )

    def gradient(self, geometry: Any) -> np.ndarray:
        return compute_dispersion_gradient(
            geometry,
            functional=self.functional,
            s6=self.s6,
            s8=self.s8,
            a1=self.a1,
            a2=self.a2,
        )
