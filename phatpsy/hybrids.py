"""Prototype & modern directed-hybrid starting orbitals (valence-bond / perfect-pairing seeds).

The library's default atomic orbitals (Clementi-Roetti, Thakkar, ADF) are ground-state,
spin-polarized atoms. For molecules built from a bond skeleton or with assigned hybridization,
we want each atom to start in its *valence state*: lone pairs and core doubly occupied,
and each bonding orbital singly occupied with the atom's spin, pointing at its bonding neighbours.

Supports both:
1. Main-group atoms with an s+p valence shell (4D space: s, px, py, pz)
2. Transition metals with an s+p+d valence shell (9D space: s, px, py, pz, dz2, dxz, dyz, dx2-y2, dxy),
   enabling d-orbital hybridization (dsp2, d2sp3, dsp3, sd3, sd) and delta bonds.

Engine real-harmonic convention:
- l = 1 (p): ml = 0 -> z, ml = +1 -> x, ml = -1 -> y
- l = 2 (d): ml = 0 -> dz2, ml = +1 -> dxz, ml = -1 -> dyz, ml = +2 -> dx2-y2, ml = -2 -> dxy
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import itertools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from phatpsy.basis import ElementData, InitialOrbital

# Engine real-harmonic convention:
# l=1 (p): ml -> Cartesian axis index (x=0, y=1, z=2)
ML_TO_AXIS = {-1: 1, 0: 2, 1: 0}

# l=2 (d): ml -> 5d component index (dz2=0, dxz=1, dyz=2, dx2-y2=3, dxy=4)
ML_TO_D_COMP = {0: 0, 1: 1, -1: 2, 2: 3, -2: 4}
D_COMP_TO_ML = {v: k for k, v in ML_TO_D_COMP.items()}

# Traceless quadrupole matrices Q for the 5 real d-orbitals:
# Normalized such that 2 * Tr(Q_a @ Q_b) = delta_ab
Q_MATRICES = [
    np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 2.0]]) / (2.0 * np.sqrt(3.0)),  # dz2 (ml=0)
    np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) / 2.0,                     # dxz (ml=1)
    np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]) / 2.0,                     # dyz (ml=-1)
    np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]]) / 2.0,                     # dx2-y2 (ml=2)
    np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]) / 2.0,                     # dxy (ml=-2)
]


class HybridizationType(str, Enum):
    """Predefined standard hybridization types."""
    S = "s"
    SP = "sp"
    SP2 = "sp2"
    SP2_BENT = "sp2_bent"
    SP3 = "sp3"
    SP3_PYRAMIDAL = "sp3_pyramidal"
    SP3_BENT = "sp3_bent"
    PURE_P = "pure_p"
    # Transition metal / d-orbital types
    DSP2 = "dsp2"
    D2SP3 = "d2sp3"
    SP3D2 = "sp3d2"
    DSP3 = "dsp3"
    SD3 = "sd3"
    SD = "sd"
    DELTA = "delta"


@dataclass(frozen=True)
class HybridizationSpec:
    """Idealized specification of a predefined hybridization type."""
    name: str
    description: str
    geometry_name: str
    n_sigma: int
    n_pi: int
    n_delta: int = 0
    n_lone_pairs: int = 0
    ideal_angles: List[float] = field(default_factory=list)
    canonical_sigma_dirs: List[np.ndarray] = field(default_factory=list)
    canonical_pi_dirs: List[np.ndarray] = field(default_factory=list)
    canonical_delta_dirs: List[np.ndarray] = field(default_factory=list)
    has_d_orbitals: bool = False
    s_in_sigma: bool = True


# Standard canonical directions
_TETRA = [
    np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
    np.array([1.0, -1.0, -1.0]) / np.sqrt(3.0),
    np.array([-1.0, 1.0, -1.0]) / np.sqrt(3.0),
    np.array([-1.0, -1.0, 1.0]) / np.sqrt(3.0),
]

_TRIGONAL = [
    np.array([1.0, 0.0, 0.0]),
    np.array([-0.5, np.sqrt(3.0) / 2.0, 0.0]),
    np.array([-0.5, -np.sqrt(3.0) / 2.0, 0.0]),
]

_PYRAMIDAL = [
    np.array([
        np.cos(phi) * np.sin(np.radians(107.8 / 2.0)),
        np.sin(phi) * np.sin(np.radians(107.8 / 2.0)),
        -np.cos(np.radians(107.8 / 2.0)),
    ])
    for phi in [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]
]

_BENT_104 = [
    np.array([np.sin(np.radians(104.5 / 2.0)), 0.0, np.cos(np.radians(104.5 / 2.0))]),
    np.array([-np.sin(np.radians(104.5 / 2.0)), 0.0, np.cos(np.radians(104.5 / 2.0))]),
]

_OCTAHEDRAL = [
    np.array([1.0, 0.0, 0.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, -1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 0.0, -1.0]),
]

_SQUARE_PLANAR = [
    np.array([1.0, 0.0, 0.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, -1.0, 0.0]),
]

_TRIGONAL_BIPYRAMIDAL = [
    # 3 equatorial
    np.array([1.0, 0.0, 0.0]),
    np.array([-0.5, np.sqrt(3.0) / 2.0, 0.0]),
    np.array([-0.5, -np.sqrt(3.0) / 2.0, 0.0]),
    # 2 axial
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 0.0, -1.0]),
]


HYBRIDIZATION_REGISTRY: Dict[str, HybridizationSpec] = {
    "s": HybridizationSpec(
        name="s",
        description="Unhybridized spherical s orbital (1 sigma slot or spherical valence)",
        geometry_name="Spherical",
        n_sigma=1,
        n_pi=0,
        ideal_angles=[],
        canonical_sigma_dirs=[np.array([0.0, 0.0, 1.0])],
    ),
    "sp": HybridizationSpec(
        name="sp",
        description="Linear digonal sp hybrid (2 collinear sigma bonds, 2 orthogonal pi bonds)",
        geometry_name="Linear (Dinfh)",
        n_sigma=2,
        n_pi=2,
        ideal_angles=[180.0],
        canonical_sigma_dirs=[np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])],
        canonical_pi_dirs=[np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])],
    ),
    "sp2": HybridizationSpec(
        name="sp2",
        description="Trigonal planar sp2 hybrid (3 coplanar sigma bonds at 120 deg, 1 out-of-plane pi bond)",
        geometry_name="Trigonal Planar (D3h)",
        n_sigma=3,
        n_pi=1,
        ideal_angles=[120.0],
        canonical_sigma_dirs=_TRIGONAL,
        canonical_pi_dirs=[np.array([0.0, 0.0, 1.0])],
    ),
    "sp2_bent": HybridizationSpec(
        name="sp2_bent",
        description="Trigonal planar bent sp2 hybrid (2 sigma bonds at ~120 deg, 1 in-plane lone pair, 1 out-of-plane pi)",
        geometry_name="Trigonal Planar Bent (C2v)",
        n_sigma=2,
        n_pi=1,
        n_lone_pairs=1,
        ideal_angles=[120.0],
        canonical_sigma_dirs=[
            np.array([np.cos(np.radians(60.0)), np.sin(np.radians(60.0)), 0.0]),
            np.array([np.cos(np.radians(60.0)), -np.sin(np.radians(60.0)), 0.0]),
        ],
        canonical_pi_dirs=[np.array([0.0, 0.0, 1.0])],
    ),
    "sp3": HybridizationSpec(
        name="sp3",
        description="Tetrahedral sp3 hybrid (4 equivalent sigma bonds at 109.47 deg)",
        geometry_name="Tetrahedral (Td)",
        n_sigma=4,
        n_pi=0,
        ideal_angles=[109.47],
        canonical_sigma_dirs=_TETRA,
    ),
    "sp3_pyramidal": HybridizationSpec(
        name="sp3_pyramidal",
        description="Trigonal pyramidal sp3 hybrid (3 sigma bonds at ~107.8 deg, 1 apical lone pair)",
        geometry_name="Trigonal Pyramidal (C3v)",
        n_sigma=3,
        n_pi=0,
        n_lone_pairs=1,
        ideal_angles=[107.8],
        canonical_sigma_dirs=_PYRAMIDAL,
    ),
    "sp3_bent": HybridizationSpec(
        name="sp3_bent",
        description="Tetrahedral bent sp3 hybrid (2 sigma bonds at ~104.5 deg, 2 lone pairs)",
        geometry_name="Bent (C2v)",
        n_sigma=2,
        n_pi=0,
        n_lone_pairs=2,
        ideal_angles=[104.5],
        canonical_sigma_dirs=_BENT_104,
    ),
    "pure_p": HybridizationSpec(
        name="pure_p",
        description="Mutually orthogonal unhybridized p valence orbitals (3 sigma bonds at 90 deg, non-hybridized s2 lone pair)",
        geometry_name="Orthogonal (C3v)",
        n_sigma=3,
        n_pi=0,
        n_lone_pairs=1,
        ideal_angles=[90.0],
        canonical_sigma_dirs=[np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])],
        s_in_sigma=False,
    ),
    # Transition metal / d-orbital types
    "dsp2": HybridizationSpec(
        name="dsp2",
        description="Square planar dsp2 hybrid (4 coplanar sigma bonds at 90 deg, non-bonding axial dz2 and dxy/pi)",
        geometry_name="Square Planar (D4h)",
        n_sigma=4,
        n_pi=0,
        ideal_angles=[90.0, 180.0],
        canonical_sigma_dirs=_SQUARE_PLANAR,
        has_d_orbitals=True,
    ),
    "d2sp3": HybridizationSpec(
        name="d2sp3",
        description="Octahedral d2sp3 hybrid (6 equivalent sigma bonds at 90 deg along +/-x, +/-y, +/-z, non-bonding t2g)",
        geometry_name="Octahedral (Oh)",
        n_sigma=6,
        n_pi=0,
        ideal_angles=[90.0, 180.0],
        canonical_sigma_dirs=_OCTAHEDRAL,
        has_d_orbitals=True,
    ),
    "dsp3": HybridizationSpec(
        name="dsp3",
        description="Trigonal bipyramidal dsp3 hybrid (3 equatorial sigma bonds at 120 deg, 2 axial sigma bonds at 180 deg)",
        geometry_name="Trigonal Bipyramidal (D3h)",
        n_sigma=5,
        n_pi=0,
        ideal_angles=[90.0, 120.0, 180.0],
        canonical_sigma_dirs=_TRIGONAL_BIPYRAMIDAL,
        has_d_orbitals=True,
    ),
    "sd3": HybridizationSpec(
        name="sd3",
        description="Tetrahedral sd3 hybrid (4 tetrahedral sigma bonds from s + dxy, dxz, dyz without p-promotion)",
        geometry_name="Tetrahedral (Td)",
        n_sigma=4,
        n_pi=0,
        ideal_angles=[109.47],
        canonical_sigma_dirs=_TETRA,
        has_d_orbitals=True,
    ),
    "sd": HybridizationSpec(
        name="sd",
        description="Linear sd hybrid (2 collinear sigma bonds along +/-z from s +/- dz2)",
        geometry_name="Linear (Dinfh)",
        n_sigma=2,
        n_pi=0,
        ideal_angles=[180.0],
        canonical_sigma_dirs=[np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])],
        has_d_orbitals=True,
    ),
    "delta": HybridizationSpec(
        name="delta",
        description="Delta-bonding orbital pair (dx2-y2 and dxy face-to-face 4-lobe overlap with 2 nodal planes)",
        geometry_name="Delta Bonding",
        n_sigma=0,
        n_pi=0,
        n_delta=2,
        canonical_delta_dirs=[np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])],
        has_d_orbitals=True,
    ),
}

# Alias mapping
HYBRIDIZATION_REGISTRY["sp3d2"] = HYBRIDIZATION_REGISTRY["d2sp3"]


def get_hybridization_spec(name: Union[str, HybridizationType]) -> HybridizationSpec:
    """Look up a HybridizationSpec by name or alias."""
    key = str(getattr(name, "value", name)).strip().lower()
    alias_map = {
        "sp3d2": "d2sp3",
        "octahedral": "d2sp3",
        "oh": "d2sp3",
        "square_planar": "dsp2",
        "sq_planar": "dsp2",
        "d4h": "dsp2",
        "linear": "sp",
        "digonal": "sp",
        "trigonal": "sp2",
        "trigonal_planar": "sp2",
        "d3h": "sp2",
        "tetrahedral": "sp3",
        "tetra": "sp3",
        "td": "sp3",
        "pyramidal": "sp3_pyramidal",
        "trigonal_pyramidal": "sp3_pyramidal",
        "bent": "sp3_bent",
        "angular": "sp3_bent",
        "water": "sp3_bent",
        "ammonia": "sp3_pyramidal",
        "methane": "sp3",
        "ethylene": "sp2",
        "acetylene": "sp",
        "trigonal_bipyramidal": "dsp3",
        "tbp": "dsp3",
    }
    canonical_key = alias_map.get(key, key)
    if canonical_key not in HYBRIDIZATION_REGISTRY:
        valid = list(HYBRIDIZATION_REGISTRY.keys())
        raise KeyError(f"Unknown hybridization type '{name}'. Valid types include: {valid}")
    return HYBRIDIZATION_REGISTRY[canonical_key]


# -----------------------------------------------------------------------------
# 3D Frame Rotation and Kabsch Alignment
# -----------------------------------------------------------------------------

def d_orbital_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """
    Computes the 5x5 transformation matrix D^(2)(R) for real d-orbitals under rotation R in SO(3).

    Uses the exact quadrupole tensor contraction:
        D_ba = 2 * Tr( Q_b @ (R @ Q_a @ R.T) )
    where Q is the traceless symmetric quadrupole basis corresponding to (dz2, dxz, dyz, dx2-y2, dxy).
    """
    D = np.zeros((5, 5), dtype=np.float64)
    for a in range(5):
        Qa_rot = R @ Q_MATRICES[a] @ R.T
        for b in range(5):
            D[b, a] = 2.0 * np.trace(Q_MATRICES[b] @ Qa_rot)
    return D


def valence_rotation_matrix(R: np.ndarray, has_d: bool = False) -> np.ndarray:
    """
    Returns the full block-diagonal rotation matrix in valence space:
    - 4x4 (s, px, py, pz) if has_d is False
    - 9x9 (s, px, py, pz, dz2, dxz, dyz, dx2-y2, dxy) if has_d is True
    """
    if not has_d:
        R4 = np.eye(4, dtype=np.float64)
        R4[1:4, 1:4] = R
        return R4
    else:
        R9 = np.eye(9, dtype=np.float64)
        R9[1:4, 1:4] = R
        R9[4:9, 4:9] = d_orbital_rotation_matrix(R)
        return R9


def align_vectors_kabsch(
    canonical_dirs: Sequence[np.ndarray],
    target_dirs: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Finds the optimal rotation matrix R in SO(3) that best aligns canonical directions
    with actual neighbor target directions.
    """
    if len(target_dirs) == 0:
        return np.eye(3, dtype=np.float64)

    # Normalize targets
    u_targets = [_unit(t) for t in target_dirs]

    if len(target_dirs) == 1:
        v = _unit(canonical_dirs[0])
        u = u_targets[0]
        dot = float(np.clip(np.dot(v, u), -1.0, 1.0))
        if dot > 1.0 - 1e-12:
            return np.eye(3, dtype=np.float64)
        if dot < -1.0 + 1e-12:
            perp = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
            perp = perp - np.dot(perp, v) * v
            perp /= np.linalg.norm(perp)
            return 2.0 * np.outer(perp, perp) - np.eye(3, dtype=np.float64)
        axis = np.cross(v, u)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(dot)
        K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
        return np.eye(3, dtype=np.float64) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

    n_canon = len(canonical_dirs)
    n_target = len(u_targets)
    best_R = np.eye(3, dtype=np.float64)
    best_score = -1e9

    k_match = min(n_canon, n_target)
    for perm in itertools.permutations(range(n_canon), k_match):
        V_sub = np.array([canonical_dirs[i] for i in perm])
        U_sub = np.array(u_targets[:k_match])
        H = V_sub.T @ U_sub
        W, _, Yt = np.linalg.svd(H)
        d = np.linalg.det(Yt.T @ W.T)
        diag = np.diag([1.0, 1.0, d])
        R_candidate = Yt.T @ diag @ W.T
        score = sum(float(np.dot(u_targets[k], R_candidate @ canonical_dirs[perm[k]])) for k in range(k_match))
        if score > best_score:
            best_score = score
            best_R = R_candidate

    return best_R


def align_hybrid_to_neighbors(
    spec: HybridizationSpec,
    neighbor_dirs: Sequence[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Rotates a HybridizationSpec's canonical vectors to best align with bonded neighbors.
    Returns (rotated_sigma_dirs, rotated_pi_dirs, rotated_delta_dirs, R_matrix).
    """
    ref_dirs = spec.canonical_sigma_dirs if spec.canonical_sigma_dirs else [np.array([0.0, 0.0, 1.0])]
    R = align_vectors_kabsch(ref_dirs, neighbor_dirs)
    rot_sigma = [R @ d for d in spec.canonical_sigma_dirs]
    rot_pi = [R @ d for d in spec.canonical_pi_dirs]
    rot_delta = [R @ d for d in spec.canonical_delta_dirs]
    return rot_sigma, rot_pi, rot_delta, R


# -----------------------------------------------------------------------------
# Valence Space Decomposition
# -----------------------------------------------------------------------------

@dataclass
class ValenceSpace:
    """Indices and radial coefficient vectors of an element's valence s, p, and d orbitals."""
    s_index: int
    p_indices: List[int]
    d_indices: List[int]
    radial_s: Dict[int, float]
    radial_p: Dict[int, float]
    radial_d: Dict[int, float]
    eps_s: float
    eps_p: float
    eps_d: float
    valence_electrons: int


def _dominant(orb: InitialOrbital) -> Tuple[int, int, float]:
    """(isto, ml, coefficient) of the largest-magnitude coefficient of an orbital."""
    c = max(orb.coefficients, key=lambda e: abs(e[2]))
    return int(c[0]), int(c[1]), float(c[2])


def standard_core_electrons(z: int) -> int:
    """Standard Aufbau closed-shell core electron count for atomic number Z."""
    if z <= 2:
        return 0
    elif z <= 10:
        return 2  # [He]
    elif z <= 18:
        return 10  # [Ne]
    elif z <= 30:
        return 18  # [Ar]
    elif z <= 36:
        return 28  # [Ar] 3d10
    elif z <= 48:
        return 36  # [Kr]
    elif z <= 54:
        return 46  # [Kr] 4d10
    elif z <= 80:
        return 68  # [Xe] 4f14
    else:
        return 68


def valence_space(elem: ElementData) -> ValenceSpace:
    """Identifies the valence s, p, and d subshells and their radial profiles."""
    orbs = elem.default_orbitals
    ls = []
    for o in orbs:
        isto, ml, _ = _dominant(o)
        ls.append((elem.functions[isto - 1].l, ml))

    occupied_s = [k for k, o in enumerate(orbs) if ls[k][0] == 0 and sum(o.occno) > 0]
    if not occupied_s:
        raise ValueError(f"{elem.symbol}: no occupied s orbital in the default orbitals")
    k_s = max(occupied_s, key=lambda k: orbs[k].eigval[0])
    radial_s = {int(c[0]): float(c[2]) for c in orbs[k_s].coefficients}
    if radial_s and max(radial_s.values(), key=abs) < 0:
        radial_s = {k: -v for k, v in radial_s.items()}
    eps_s = float(orbs[k_s].eigval[0])

    # p-orbitals (only part of the valence shell for Z > 2)
    p_orbs = [k for k, o in enumerate(orbs) if ls[k][0] == 1] if elem.atomic_number > 2 else []
    p_group = sorted(p_orbs[-3:]) if len(p_orbs) >= 3 else []
    if p_group:
        k0 = next((k for k in p_group if ls[k][1] == 0), p_group[0])
        radial_p = {int(c[0]): float(c[2]) for c in orbs[k0].coefficients}
        if radial_p and max(radial_p.values(), key=abs) < 0:
            radial_p = {k: -v for k, v in radial_p.items()}
        eps_p = float(orbs[k0].eigval[0])
    else:
        radial_p, eps_p = {}, 0.0

    # d-orbitals (transition metals, heavy elements, or d-hybrid specs)
    d_orbs = [k for k, o in enumerate(orbs) if ls[k][0] == 2] if elem.atomic_number > 2 else []
    d_group = sorted(d_orbs[-5:]) if len(d_orbs) >= 5 else []
    if d_group:
        kd0 = next((k for k in d_group if ls[k][1] == 0), d_group[0])
        radial_d = {int(c[0]): float(c[2]) for c in orbs[kd0].coefficients}
        if radial_d and max(radial_d.values(), key=abs) < 0:
            radial_d = {k: -v for k, v in radial_d.items()}
        eps_d = float(orbs[kd0].eigval[0])
    else:
        radial_d, eps_d = {}, 0.0

    nv = elem.atomic_number - standard_core_electrons(elem.atomic_number)
    return ValenceSpace(
        s_index=k_s,
        p_indices=p_group,
        d_indices=d_group,
        radial_s=radial_s,
        radial_p=radial_p,
        radial_d=radial_d,
        eps_s=eps_s,
        eps_p=eps_p,
        eps_d=eps_d,
        valence_electrons=nv,
    )


def _unit(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero-length direction")
    return v / n


def _canonical_perpendicular_pair(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two unit vectors perpendicular to ``axis``, independent of the axis sign."""
    a = _unit(axis)
    first = next(c for c in a if abs(c) > 1e-9)
    if first < 0:
        a = -a
    helper = np.zeros(3)
    helper[int(np.argmin(np.abs(a)))] = 1.0
    e1 = _unit(np.cross(helper, a))
    e2 = np.cross(a, e1)
    return e1, e2


def _plane_normal(dirs: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    best, best_n = None, 1e-3
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            c = np.cross(dirs[i], dirs[j])
            n = np.linalg.norm(c)
            if n > best_n:
                best, best_n = c / n, n
    return best


def _vec_to_coefficients(u: np.ndarray, vs: ValenceSpace) -> List[Tuple[int, int, float]]:
    """Valence vector (s, px, py, pz, [dz2, dxz, dyz, dx2-y2, dxy]) -> engine coefficients (isto, ml, c)."""
    out: List[Tuple[int, int, float]] = []
    # s orbital
    for isto, c in vs.radial_s.items():
        if abs(u[0] * c) > 1e-12:
            out.append((isto, 0, float(u[0] * c)))
    # p orbitals (indices 1..3)
    if vs.radial_p and len(u) > 1:
        for isto, c in vs.radial_p.items():
            for ml, axis in ML_TO_AXIS.items():
                comp = u[1 + axis] * c
                if abs(comp) > 1e-12:
                    out.append((isto, ml, float(comp)))
    # d orbitals (indices 4..8)
    if vs.radial_d and len(u) > 4:
        for isto, c in vs.radial_d.items():
            for ml, comp_idx in ML_TO_D_COMP.items():
                comp = u[4 + comp_idx] * c
                if abs(comp) > 1e-12:
                    out.append((isto, ml, float(comp)))
    return out


# -----------------------------------------------------------------------------
# Valence Vector Construction (4D and 9D)
# -----------------------------------------------------------------------------

def atom_valence_vectors(
    sigma_dirs: Sequence[np.ndarray],
    pi_dirs: Sequence[np.ndarray],
    n_lone_pairs: int,
    n_extra: int = 0,
    s_in_sigma: bool = True,
    delta_dirs: Sequence[np.ndarray] = (),
    has_d_orbitals: bool = False,
) -> Dict[str, List[np.ndarray]]:
    """
    Constructs orthonormal valence vectors in:
    - 4D (s, px, py, pz) for main-group atoms
    - 9D (s, px, py, pz, dz2, dxz, dyz, dx2-y2, dxy) when has_d_orbitals=True
    """
    dim = 9 if has_d_orbitals else 4
    n_sigma = len(sigma_dirs)
    n_pi = len(pi_dirs)
    n_delta = len(delta_dirs)
    n_sigma_slots = n_sigma + n_lone_pairs + n_extra

    if n_sigma + n_pi + n_delta + n_lone_pairs + n_extra > dim:
        raise ValueError(f"more bonds and lone pairs than available valence orbitals ({dim})")

    rows = []
    if n_sigma:
        w = (1.0 / np.sqrt(max(n_sigma_slots, 1))) if s_in_sigma else 0.0
        for d in sigma_dirs:
            u = _unit(d)
            p_comp = np.sqrt(1.0 - w * w) * u
            v = np.zeros(dim, dtype=np.float64)
            v[0] = w
            v[1:4] = p_comp
            if has_d_orbitals and n_sigma > 4:
                v[4:9] = [float(u @ Q @ u) for Q in Q_MATRICES]
            rows.append(v)
        T = np.array(rows)
        evals, evecs = np.linalg.eigh(T @ T.T)
        inv_sqrt = np.where(evals > 1e-10, np.maximum(evals, 1e-12) ** -0.5, 0.0)
        B = (evecs @ np.diag(inv_sqrt) @ evecs.T) @ T  # symmetric orthonormalization
        basis = [b for b in B]
    else:
        basis = []

    pi_vecs: List[np.ndarray] = []
    for p in pi_dirs:
        v = np.zeros(dim, dtype=np.float64)
        v[1:4] = _unit(p)
        for b in basis + pi_vecs:
            v = v - (b @ v) * b
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-12:
            pi_vecs.append(v / norm_v)

    delta_vecs: List[np.ndarray] = []
    if has_d_orbitals and n_delta:
        # Canonical delta orbitals: dx2-y2 (comp 3 -> index 7) and dxy (comp 4 -> index 8)
        for i_delta in range(min(n_delta, 2)):
            v = np.zeros(dim, dtype=np.float64)
            v[7 + i_delta] = 1.0  # 7: dx2-y2, 8: dxy
            for b in basis + pi_vecs + delta_vecs:
                v = v - (b @ v) * b
            norm_v = np.linalg.norm(v)
            if norm_v > 1e-12:
                delta_vecs.append(v / norm_v)

    used_list = basis + pi_vecs + delta_vecs
    used = np.array(used_list) if used_list else np.zeros((0, dim))
    proj = np.eye(dim) - used.T @ used if len(used) else np.eye(dim)
    # Break degeneracy inside the orthogonal complement by boosting the s-component (index 0)
    # so the unique s-carrying complement hybrid is separated from pure-p/d complement orbitals
    # in a 3D rotation-invariant manner (descending eigenvalue order puts s-hybrid first).
    e0 = np.zeros((1, dim), dtype=np.float64)
    e0[0, 0] = 1.0
    proj_s = proj + 0.5 * (proj @ e0.T @ e0 @ proj)
    evals, evecs = np.linalg.eigh(proj_s)
    order_idx = [k for k in np.argsort(evals)[::-1] if evals[k] > 0.5]
    complement = [evecs[:, k] for k in order_idx]
    # deterministic sign: prefer positive s-component when present, else first non-zero component
    complement = [
        c if (c[0] > 1e-9 or (abs(c[0]) <= 1e-9 and next(x for x in c if abs(x) > 1e-9) > 0)) else -c
        for c in complement
    ]

    need = n_lone_pairs + n_extra
    if len(complement) < need:
        raise ValueError("not enough orthogonal valence orbitals for the requested lone pairs")

    return {
        "lone": complement[:n_lone_pairs],
        "sigma": basis,
        "pi": pi_vecs,
        "delta": delta_vecs,
        "extra": complement[n_lone_pairs:need],
        "empty": complement[need:],
    }


def _construct_d_hybrid_vectors(
    spec: HybridizationSpec,
    rot_matrix: np.ndarray,
) -> Dict[str, List[np.ndarray]]:
    """Constructs explicit canonical d-hybrids (dsp2, d2sp3, sd3, sd, delta) and rotates them."""
    dim = 9
    R9 = valence_rotation_matrix(rot_matrix, has_d=True)

    if spec.name == "dsp2":
        # 4 sigma hybrids in xy plane
        h_px = np.array([0.5, 1.0 / np.sqrt(2.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0])
        h_mx = np.array([0.5, -1.0 / np.sqrt(2.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0])
        h_py = np.array([0.5, 0.0, 1.0 / np.sqrt(2.0), 0.0, 0.0, 0.0, 0.0, -0.5, 0.0])
        h_my = np.array([0.5, 0.0, -1.0 / np.sqrt(2.0), 0.0, 0.0, 0.0, 0.0, -0.5, 0.0])
        sigma = [R9 @ h for h in (h_px, h_mx, h_py, h_my)]
        # Non-bonding: pz, dz2, dxz, dyz, dxy
        nb = [
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # pz
            np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),  # dz2
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),  # dxz
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),  # dyz
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),  # dxy
        ]
        non_bonding = [R9 @ h for h in nb]
        return {"sigma": sigma, "pi": [], "delta": [], "non_bonding": non_bonding, "lone": non_bonding}

    elif spec.name in ("d2sp3", "sp3d2"):
        # 6 sigma hybrids along +/-z, +/-x, +/-y
        h_pz = np.array([1.0 / np.sqrt(6.0), 0.0, 0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(3.0), 0.0, 0.0, 0.0, 0.0])
        h_mz = np.array([1.0 / np.sqrt(6.0), 0.0, 0.0, -1.0 / np.sqrt(2.0), 1.0 / np.sqrt(3.0), 0.0, 0.0, 0.0, 0.0])
        h_px = np.array([1.0 / np.sqrt(6.0), 1.0 / np.sqrt(2.0), 0.0, 0.0, -1.0 / np.sqrt(12.0), 0.0, 0.0, 0.5, 0.0])
        h_mx = np.array([1.0 / np.sqrt(6.0), -1.0 / np.sqrt(2.0), 0.0, 0.0, -1.0 / np.sqrt(12.0), 0.0, 0.0, 0.5, 0.0])
        h_py = np.array([1.0 / np.sqrt(6.0), 0.0, 1.0 / np.sqrt(2.0), 0.0, -1.0 / np.sqrt(12.0), 0.0, 0.0, -0.5, 0.0])
        h_my = np.array([1.0 / np.sqrt(6.0), 0.0, -1.0 / np.sqrt(2.0), 0.0, -1.0 / np.sqrt(12.0), 0.0, 0.0, -0.5, 0.0])
        sigma = [R9 @ h for h in (h_pz, h_mz, h_px, h_mx, h_py, h_my)]
        # Non-bonding t2g: dxz, dyz, dxy
        t2g = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),  # dxz
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),  # dyz
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),  # dxy
        ]
        non_bonding = [R9 @ h for h in t2g]
        return {"sigma": sigma, "pi": [], "delta": [], "non_bonding": non_bonding, "lone": non_bonding}

    elif spec.name == "sd":
        h_pz = np.array([1.0 / np.sqrt(2.0), 0.0, 0.0, 0.0, 1.0 / np.sqrt(2.0), 0.0, 0.0, 0.0, 0.0])
        h_mz = np.array([1.0 / np.sqrt(2.0), 0.0, 0.0, 0.0, -1.0 / np.sqrt(2.0), 0.0, 0.0, 0.0, 0.0])
        sigma = [R9 @ h for h in (h_pz, h_mz)]
        rem = [
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # px
            np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # py
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # pz
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),  # dxz
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),  # dyz
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),  # dx2-y2
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),  # dxy
        ]
        non_bonding = [R9 @ h for h in rem]
        return {"sigma": sigma, "pi": [], "delta": [], "non_bonding": non_bonding, "lone": non_bonding}

    elif spec.name == "delta":
        delta1 = R9 @ np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # dx2-y2
        delta2 = R9 @ np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # dxy
        rem = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # s
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # px
            np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # py
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # pz
            np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),  # dz2
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),  # dxz
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),  # dyz
        ]
        non_bonding = [R9 @ h for h in rem]
        return {"sigma": [], "pi": [], "delta": [delta1, delta2], "non_bonding": non_bonding, "lone": non_bonding}

    else:
        # Fallback to general vector orthonormalization
        rot_sigma = [rot_matrix @ d for d in spec.canonical_sigma_dirs]
        rot_pi = [rot_matrix @ d for d in spec.canonical_pi_dirs]
        rot_delta = [rot_matrix @ d for d in spec.canonical_delta_dirs]
        return atom_valence_vectors(rot_sigma, rot_pi, spec.n_lone_pairs, has_d_orbitals=True, delta_dirs=rot_delta)


# -----------------------------------------------------------------------------
# High-Level Orbital Construction
# -----------------------------------------------------------------------------

def build_atom_orbitals(
    elem: ElementData,
    spin: str,
    sigma_dirs: Sequence[np.ndarray] = (),
    pi_dirs: Sequence[np.ndarray] = (),
    s_in_sigma: bool = True,
    hybridization: Optional[Union[str, HybridizationType]] = None,
    delta_dirs: Sequence[np.ndarray] = (),
    charge: float = 0.0,
    valence_electrons: Optional[int] = None,
) -> List[InitialOrbital]:
    """
    Constructs replacement ``starting_orbitals`` for one atom in its valence state.

    Supports either:
    1. Explicit direction vectors (``sigma_dirs``, ``pi_dirs``, optional ``delta_dirs``)
    2. Named hybridization (``hybridization="sp3"``, ``"dsp2"``, etc.) with automatic
       best-fit frame rotation to any provided ``sigma_dirs`` (neighbor directions).
    """
    if spin not in ("up", "down"):
        raise ValueError("spin must be 'up' or 'down'")

    vs = valence_space(elem)
    lib = [replace(o) for o in elem.default_orbitals]
    bond_occ = [1.0, 0.0] if spin == "up" else [0.0, 1.0]

    # Handle named hybridization if provided
    spec: Optional[HybridizationSpec] = None
    if hybridization is not None:
        spec = get_hybridization_spec(hybridization)

    eff_nv = (
        int(valence_electrons)
        if valence_electrons is not None
        else int(round(vs.valence_electrons - charge))
    )

    # Check s-only valence shell (H, He)
    if not vs.p_indices and not vs.d_indices:
        n_bonds = len(sigma_dirs) if spec is None else spec.n_sigma
        if n_bonds > 1:
            raise ValueError(f"{elem.symbol}: s-only valence shell supports at most 1 bond")
        k = vs.s_index
        lib[k] = replace(lib[k], occno=list(bond_occ))
        return lib

    is_tm_d = bool(vs.d_indices) and (
        elem.atomic_number >= 21
        or any(sum(elem.default_orbitals[k].occno) > 0 for k in vs.d_indices)
    )
    has_d = (
        (spec.has_d_orbitals or bool(delta_dirs))
        if spec is not None
        else (is_tm_d or bool(delta_dirs))
    )

    is_terminal_triple_group14 = False
    is_terminal_triple_group16 = False

    if spec is not None:
        rot_sigma, rot_pi, rot_delta, R = align_hybrid_to_neighbors(spec, sigma_dirs)
        if spec.has_d_orbitals:
            vecs = _construct_d_hybrid_vectors(spec, R)
        else:
            if spec.name == "sp" and len(sigma_dirs) == 1:
                rot_sigma = [rot_sigma[0]]
                n_bond_e = 1 + len(rot_pi) + len(rot_delta)
                if valence_electrons is None and charge == 0.0 and eff_nv in (4, 6):
                    is_terminal_triple_group14 = (eff_nv == 4)
                    is_terminal_triple_group16 = (eff_nv == 6)
                    eff_nv = 5
            else:
                n_bond_e = spec.n_sigma + spec.n_pi + spec.n_delta
                if spec.name == "sp" and valence_electrons is None and charge == 0.0 and eff_nv == 6:
                    is_terminal_triple_group16 = True
                    eff_nv = 5
            leftover = eff_nv - n_bond_e
            if leftover < 0:
                raise ValueError(f"{elem.symbol}: {n_bond_e} bonds exceed {eff_nv} valence electrons")
            n_lp, n_extra = divmod(leftover, 2)
            vecs = atom_valence_vectors(
                rot_sigma,
                rot_pi,
                n_lone_pairs=n_lp,
                n_extra=n_extra,
                s_in_sigma=spec.s_in_sigma,
                has_d_orbitals=has_d,
            )
    else:
        n_bond_e = len(sigma_dirs) + len(pi_dirs) + len(delta_dirs)
        if (
            not has_d
            and len(sigma_dirs) == 1
            and len(pi_dirs) == 2
            and valence_electrons is None
            and charge == 0.0
            and eff_nv in (4, 6)
        ):
            is_terminal_triple_group14 = (eff_nv == 4)
            is_terminal_triple_group16 = (eff_nv == 6)
            eff_nv = 5
        leftover = eff_nv - n_bond_e
        if leftover < 0:
            raise ValueError(f"{elem.symbol}: {n_bond_e} bonds exceed {eff_nv} valence electrons")
        n_lp, n_extra = divmod(leftover, 2)
        vecs = atom_valence_vectors(
            sigma_dirs,
            pi_dirs,
            n_lone_pairs=n_lp,
            n_extra=n_extra,
            s_in_sigma=s_in_sigma,
            delta_dirs=delta_dirs,
            has_d_orbitals=has_d,
        )

    # Order and allocate electrons
    if "non_bonding" in vecs:
        # Explicit non-bonding pool (e.g. from d-hybrids)
        nb_pool = vecs["non_bonding"]
        n_bond_e = len(vecs.get("sigma", [])) + len(vecs.get("pi", [])) + len(vecs.get("delta", []))
        leftover = max(eff_nv - n_bond_e, 0)
        n_lp, n_extra = divmod(leftover, 2)
        ordered: List[Tuple[np.ndarray, List[float]]] = []
        ordered += [(v, [1.0, 1.0]) for v in nb_pool[:n_lp]]
        ordered += [(v, list(bond_occ)) for v in vecs.get("sigma", [])]
        ordered += [(v, list(bond_occ)) for v in vecs.get("pi", [])]
        ordered += [(v, list(bond_occ)) for v in vecs.get("delta", [])]
        if n_extra:
            ordered += [(v, list(bond_occ)) for v in nb_pool[n_lp : n_lp + n_extra]]
        ordered += [(v, [0.0, 0.0]) for v in nb_pool[n_lp + n_extra :]]
    else:
        # Standard pools from atom_valence_vectors
        ordered = []
        lp_occ = [1.0, 1.0]
        if is_terminal_triple_group14:
            lp_occ = [0.0, 1.0] if bond_occ[0] >= bond_occ[1] else [1.0, 0.0]
        ordered += [(v, list(lp_occ)) for v in vecs.get("lone", [])]
        ordered += [(v, list(bond_occ)) for v in vecs.get("sigma", [])]
        for v in vecs.get("pi", []):
            pi_occ = list(bond_occ)
            if is_terminal_triple_group16:
                if pi_occ[0] < pi_occ[1]:
                    pi_occ = [0.5, pi_occ[1]]
                else:
                    pi_occ = [pi_occ[0], 0.5]
            ordered.append((v, pi_occ))
        ordered += [(v, list(bond_occ)) for v in vecs.get("delta", [])]
        ordered += [(v, list(bond_occ)) for v in vecs.get("extra", [])]
        ordered += [(v, [0.0, 0.0]) for v in vecs.get("empty", [])]

    if has_d and vs.d_indices:
        slots = sorted([vs.s_index, *vs.p_indices, *vs.d_indices])
    else:
        slots = sorted([vs.s_index, *vs.p_indices])

    # Assign to lib
    for slot, (v, occ) in zip(slots[:len(ordered)], ordered):
        eps_val = vs.eps_s * (v[0] ** 2)
        if len(v) > 1 and vs.radial_p:
            eps_val += vs.eps_p * sum(v[1:4] ** 2)
        if len(v) > 4 and vs.radial_d:
            eps_val += vs.eps_d * sum(v[4:9] ** 2)
        lib[slot] = InitialOrbital(
            eigval=[float(eps_val), float(eps_val)],
            occno=occ,
            coefficients=_vec_to_coefficients(v, vs),
        )

    return lib


def _valence_orbital_occno(orb) -> List[float]:
    """Converts a ValenceOrbital record into [occ_alpha, occ_beta]."""
    if orb.occupation == 0:
        return [0.0, 0.0]
    if orb.occupation == 2:
        return [1.0, 1.0]
    if orb.spin == "up":
        return [1.0, 0.0]
    if orb.spin == "down":
        return [0.0, 1.0]
    raise ValueError(f"Singly occupied ValenceOrbital must have spin 'up' or 'down', got {orb.spin!r}")


def build_atom_orbitals_from_valence_config(
    elem: ElementData,
    atom,
    atoms_by_id: Optional[Dict[str, Any]] = None,
    s_in_sigma: bool = True,
) -> List[InitialOrbital]:
    """
    Constructs replacement ``starting_orbitals`` for an ``AtomConfig`` directly from its
    ``valence_config`` (a list of ``ValenceOrbital`` entries), honoring per-orbital
    spins (including mixed-spin atoms), occupations, ``connect_to`` bond directions,
    dative ``lone_pair`` directions, and ``plane``/``orientation`` specifiers.
    """
    from phatpsy.schema import ValenceOrbital, n_valence_orbitals

    if not getattr(atom, "valence_config", None):
        raise ValueError(f"Atom '{getattr(atom, 'id', None) or elem.symbol}' has no valence_config.")

    if hasattr(atom, "validate_valence_config"):
        saved_so = getattr(atom, "starting_orbitals", None)
        atom.starting_orbitals = None
        try:
            atom.validate_valence_config()
        finally:
            atom.starting_orbitals = saved_so

    vs = valence_space(elem)
    lib = [replace(o) for o in elem.default_orbitals]
    n_orb = n_valence_orbitals(elem.atomic_number)

    entries: List[ValenceOrbital] = [
        o if isinstance(o, ValenceOrbital) else ValenceOrbital.from_dict(o)
        for o in atom.valence_config
    ]

    # S-only valence shell (H, He)
    if not vs.p_indices and not vs.d_indices:
        if n_orb == 0:
            return lib
        k = vs.s_index
        occ = _valence_orbital_occno(entries[0]) if entries else [0.0, 0.0]
        lib[k] = replace(lib[k], occno=occ)
        return lib

    # Pad partial valence_config up to n_orb with empty orbitals matching hybridization
    hyb_label = str(atom.hybridization).strip().lower() if getattr(atom, "hybridization", None) else None
    n_pi_target = 2 if hyb_label == "sp" else (1 if hyb_label in ("sp2", "sp2_bent") else 0)
    cur_pi = sum(1 for e in entries if e.orbital_type == "pi")
    while len(entries) < n_orb:
        if cur_pi < n_pi_target:
            entries.append(ValenceOrbital(spin="none", orbital_type="pi", occupation=0))
            cur_pi += 1
        else:
            entries.append(ValenceOrbital(spin="none", orbital_type="sigma", occupation=0))

    xyz_a = np.asarray(getattr(atom, "xyz", [0.0, 0.0, 0.0]), dtype=float)

    def _resolve_dir(target_id: Optional[str]) -> Optional[np.ndarray]:
        if not target_id or not atoms_by_id or target_id not in atoms_by_id:
            return None
        other = atoms_by_id[target_id]
        vec = np.asarray(getattr(other, "xyz", [0.0, 0.0, 0.0]), dtype=float) - xyz_a
        if float(np.linalg.norm(vec)) <= 1e-8:
            return None
        return _unit(vec)

    def _resolve_plane_normal(plane_ids: Optional[Sequence[str]]) -> Optional[np.ndarray]:
        if not plane_ids or not atoms_by_id:
            return None
        p_vecs: List[np.ndarray] = []
        for pid in plane_ids:
            u = _resolve_dir(pid)
            if u is not None:
                p_vecs.append(u)
        if len(p_vecs) >= 2:
            return _plane_normal(p_vecs)
        return None

    is_tm_d = bool(vs.d_indices) and (
        elem.atomic_number >= 21
        or n_orb == 9
        or any(e.orbital_type == "delta" for e in entries)
        or any(sum(elem.default_orbitals[k].occno) > 0 for k in vs.d_indices)
    )
    spec: Optional[HybridizationSpec] = get_hybridization_spec(hyb_label) if hyb_label else None
    has_d = (spec.has_d_orbitals if spec is not None else False) or is_tm_d
    dim = 9 if has_d else 4

    # 1. Distinguish dative pi lone_pair donors from dative sigma lone_pair donors
    sigma_targets = {orb.connect_to for orb in entries if orb.orbital_type == "sigma" and orb.connect_to}
    n_explicit_pi = sum(1 for orb in entries if orb.orbital_type == "pi")
    pi_deficit = max(0, (spec.n_pi - n_explicit_pi)) if spec is not None else 0
    dative_pi_indices: set[int] = set()
    my_id = getattr(atom, "id", None)
    for idx_e, orb in enumerate(entries):
        if orb.orbital_type == "lone_pair" and orb.connect_to:
            is_dative_pi = orb.connect_to in sigma_targets
            if not is_dative_pi and atoms_by_id and orb.connect_to in atoms_by_id:
                partner = atoms_by_id[orb.connect_to]
                p_vc = getattr(partner, "valence_config", None) or []
                if any(
                    getattr(po, "occupation", None) == 0
                    and getattr(po, "orbital_type", None) == "pi"
                    and (not my_id or getattr(po, "connect_to", None) == my_id)
                    for po in p_vc
                ):
                    is_dative_pi = True
            if not is_dative_pi and pi_deficit > len(dative_pi_indices):
                is_dative_pi = True
            if is_dative_pi:
                dative_pi_indices.add(idx_e)

    # Identify directed sigma-hybrid orbitals (sigma with connect_to, plus dative sigma lone_pair with connect_to)
    directed_sigma_indices: List[int] = []
    directed_sigma_dirs: List[np.ndarray] = []
    for idx_e, orb in enumerate(entries):
        if (orb.orbital_type == "sigma" or (orb.orbital_type == "lone_pair" and idx_e not in dative_pi_indices)) and orb.connect_to:
            u = _resolve_dir(orb.connect_to)
            if u is not None:
                directed_sigma_indices.append(idx_e)
                directed_sigma_dirs.append(u)

    # Transition-metal undirected atomic valence_config path:
    # When a transition-metal atom (e.g. Ni 3d8 4s2 4p0) has no explicit hybridization
    # and no directed connect_to / plane vectors, preserve the GCHF atomic s, d, and p
    # eigenvectors from elem.default_orbitals while applying the requested occupations/spins.
    if (
        is_tm_d
        and spec is None
        and not directed_sigma_dirs
        and not any(orb.connect_to or orb.plane for orb in entries)
    ):
        slots_energy_order = [vs.s_index] + list(vs.d_indices) + list(vs.p_indices)
        doubly = [orb for orb in entries if orb.occupation == 2]
        singly = [orb for orb in entries if orb.occupation == 1]
        empty = [orb for orb in entries if orb.occupation == 0]
        ordered_entries = doubly + singly + empty
        for slot, orb in zip(slots_energy_order[: len(ordered_entries)], ordered_entries):
            lib[slot] = replace(lib[slot], occno=list(_valence_orbital_occno(orb)))
        return lib

    # Fallback when no connect_to coordinates could be resolved (e.g. isolated atom or single-atom test)
    if not directed_sigma_dirs:
        sigma_entry_indices = [
            idx_e for idx_e, orb in enumerate(entries)
            if orb.orbital_type == "sigma" or (orb.orbital_type == "lone_pair" and orb.connect_to and idx_e not in dative_pi_indices)
        ]
        if spec is not None and spec.canonical_sigma_dirs:
            n_take = min(len(sigma_entry_indices), len(spec.canonical_sigma_dirs))
            directed_sigma_indices = sigma_entry_indices[:n_take]
            directed_sigma_dirs = [np.array(d, dtype=float) for d in spec.canonical_sigma_dirs[:n_take]]
        elif sigma_entry_indices:
            n_pi_total = sum(1 for orb in entries if orb.orbital_type == "pi")
            fallback_hyb = "sp" if n_pi_total >= 2 else ("sp2" if n_pi_total == 1 else "sp3")
            fb_spec = get_hybridization_spec(fallback_hyb)
            n_take = min(len(sigma_entry_indices), len(fb_spec.canonical_sigma_dirs))
            directed_sigma_indices = sigma_entry_indices[:n_take]
            directed_sigma_dirs = [np.array(d, dtype=float) for d in fb_spec.canonical_sigma_dirs[:n_take]]

    # Transition-metal explicit d-hybridization path (dsp2, d2sp3, sd, delta)
    if has_d and spec is not None and spec.has_d_orbitals:
        _, _, _, R = align_hybrid_to_neighbors(spec, directed_sigma_dirs)
        d_vecs = _construct_d_hybrid_vectors(spec, R)
        sigma_pool = list(d_vecs.get("sigma", []))
        pi_pool = list(d_vecs.get("pi", []))
        delta_pool = list(d_vecs.get("delta", []))
        nb_pool = list(d_vecs.get("non_bonding", d_vecs.get("lone", [])))
        ordered: List[Tuple[np.ndarray, List[float]]] = []
        for orb in entries:
            occ = _valence_orbital_occno(orb)
            if orb.orbital_type == "sigma" and sigma_pool:
                v = sigma_pool.pop(0)
            elif orb.orbital_type == "pi" and pi_pool:
                v = pi_pool.pop(0)
            elif orb.orbital_type == "delta" and delta_pool:
                v = delta_pool.pop(0)
            elif nb_pool:
                v = nb_pool.pop(0)
            elif sigma_pool:
                v = sigma_pool.pop(0)
            else:
                raise ValueError(f"{elem.symbol}: insufficient d-hybrid vectors for valence_config")
            ordered.append((v, occ))
    else:
        # 2. Construct Löwdin-orthonormalized directed sigma hybrids
        use_s = spec.s_in_sigma if spec is not None else s_in_sigma
        if directed_sigma_dirs:
            rows = []
            for d in directed_sigma_dirs:
                u = _unit(d)
                row = np.zeros(dim, dtype=np.float64)
                row[0] = 1.0 if use_s else 0.0
                row[1:4] = np.sqrt(max(len(directed_sigma_dirs), 1)) * u if use_s else u
                if has_d and len(directed_sigma_dirs) > 4:
                    row[4:9] = np.sqrt(max(len(directed_sigma_dirs), 1)) * np.array(
                        [float(u @ Q @ u) for Q in Q_MATRICES], dtype=np.float64
                    )
                rows.append(row)
            T = np.array(rows, dtype=np.float64)
            evals, evecs = np.linalg.eigh(T @ T.T)
            inv_sqrt = np.where(evals > 1e-10, np.maximum(evals, 1e-12) ** -0.5, 0.0)
            B = (evecs @ np.diag(inv_sqrt) @ evecs.T) @ T
            sigma_basis = [b for b in B]
        else:
            sigma_basis = []

        # 3. Resolve 3D directions for pi orbitals and orthonormalize against sigma_basis
        pi_indices = [
            idx_e for idx_e, orb in enumerate(entries)
            if orb.orbital_type == "pi" or idx_e in dative_pi_indices
        ]
        pi_vecs: List[np.ndarray] = []
        raw_pi_dirs: List[np.ndarray] = []
        default_mol_normal = _plane_normal(directed_sigma_dirs) if len(directed_sigma_dirs) >= 2 else None

        for p_pos, idx_e in enumerate(pi_indices):
            orb = entries[idx_e]
            p_dir: Optional[np.ndarray] = None
            n_plane = _resolve_plane_normal(orb.plane)
            u_conn = _resolve_dir(orb.connect_to)

            if n_plane is not None:
                if orb.orientation == "in_plane" and u_conn is not None:
                    in_p = np.cross(n_plane, u_conn)
                    if float(np.linalg.norm(in_p)) > 1e-8:
                        p_dir = _unit(in_p)
                if p_dir is None:
                    p_dir = n_plane
            elif u_conn is not None:
                c1, c2 = _canonical_perpendicular_pair(u_conn)
                if default_mol_normal is not None and abs(float(np.dot(default_mol_normal, u_conn))) < 0.95:
                    c1 = _unit(default_mol_normal - float(np.dot(default_mol_normal, u_conn)) * u_conn)
                    c2 = _unit(np.cross(u_conn, c1))
                # Pick first perpendicular vector not parallel to an already chosen raw_pi_dir
                p_dir = c1
                for prev in raw_pi_dirs:
                    if abs(float(np.dot(prev, c1))) > 0.9:
                        p_dir = c2
                        break
            elif default_mol_normal is not None:
                p_dir = default_mol_normal
            elif directed_sigma_dirs:
                c1, c2 = _canonical_perpendicular_pair(directed_sigma_dirs[0])
                p_dir = c1 if p_pos == 0 else c2
            else:
                canon_pi = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])]
                p_dir = canon_pi[min(p_pos, 2)]

            raw_pi_dirs.append(p_dir)
            v = np.zeros(dim, dtype=np.float64)
            v[1:4] = _unit(p_dir)
            for b in sigma_basis + pi_vecs:
                v = v - (b @ v) * b
            norm_v = float(np.linalg.norm(v))
            if norm_v <= 1e-8 and directed_sigma_dirs:
                # Fallback to second canonical perpendicular if first was in span
                for cand in _canonical_perpendicular_pair(directed_sigma_dirs[0]):
                    v_try = np.zeros(dim, dtype=np.float64)
                    v_try[1:4] = cand
                    for b in sigma_basis + pi_vecs:
                        v_try = v_try - (b @ v_try) * b
                    if float(np.linalg.norm(v_try)) > 1e-8:
                        v = v_try
                        norm_v = float(np.linalg.norm(v))
                        break
            if norm_v > 1e-8:
                pi_vecs.append(v / norm_v)

        # 4. Delta orbitals (transition metals)
        delta_indices = [idx_e for idx_e, orb in enumerate(entries) if orb.orbital_type == "delta"]
        delta_vecs: List[np.ndarray] = []
        if has_d and delta_indices:
            for i_delta in range(min(len(delta_indices), 2)):
                v = np.zeros(dim, dtype=np.float64)
                v[7 + i_delta] = 1.0
                for b in sigma_basis + pi_vecs + delta_vecs:
                    v = v - (b @ v) * b
                norm_v = float(np.linalg.norm(v))
                if norm_v > 1e-8:
                    delta_vecs.append(v / norm_v)

        # 5. Orthogonal complement for undirected lone pairs and remaining undirected orbitals
        used_list = sigma_basis + pi_vecs + delta_vecs
        used = np.array(used_list) if used_list else np.zeros((0, dim))
        proj = np.eye(dim) - used.T @ used if len(used) else np.eye(dim)
        e0 = np.zeros((1, dim), dtype=np.float64)
        e0[0, 0] = 1.0
        proj_s = proj + 0.5 * (proj @ e0.T @ e0 @ proj)
        evals, evecs = np.linalg.eigh(proj_s)
        order_idx = [k for k in np.argsort(evals)[::-1] if evals[k] > 0.5]
        complement = [evecs[:, k] for k in order_idx]
        complement = [
            c if (c[0] > 1e-9 or (abs(c[0]) <= 1e-9 and next(x for x in c if abs(x) > 1e-9) > 0)) else -c
            for c in complement
        ]

        # Map each entry to its constructed vector
        assigned_vecs: Dict[int, np.ndarray] = {}
        for idx_e, vec in zip(directed_sigma_indices, sigma_basis):
            assigned_vecs[idx_e] = vec
        for idx_e, vec in zip(pi_indices, pi_vecs):
            assigned_vecs[idx_e] = vec
        for idx_e, vec in zip(delta_indices, delta_vecs):
            assigned_vecs[idx_e] = vec

        rem_indices = [idx_e for idx_e in range(len(entries)) if idx_e not in assigned_vecs]
        avail_comp = list(complement)

        # First match any out_of_plane oriented lone pairs/orbitals to the normal-aligned complement vector
        for idx_e in list(rem_indices):
            orb = entries[idx_e]
            if orb.orientation == "out_of_plane" and avail_comp:
                n_plane = _resolve_plane_normal(orb.plane)
                if n_plane is None:
                    n_plane = default_mol_normal
                if n_plane is not None:
                    best_k = max(
                        range(len(avail_comp)),
                        key=lambda k: abs(float(np.dot(avail_comp[k][1:4], n_plane))),
                    )
                    v_pick = avail_comp.pop(best_k)
                    if float(np.dot(v_pick[1:4], n_plane)) < 0:
                        v_pick = -v_pick
                    assigned_vecs[idx_e] = v_pick
                    rem_indices.remove(idx_e)

        # Assign remaining undirected entries (in_plane lone_pair gets s-rich complement[0] first)
        for idx_e in rem_indices:
            if not avail_comp:
                raise ValueError(f"{elem.symbol}: insufficient orthogonal complement orbitals for valence_config")
            assigned_vecs[idx_e] = avail_comp.pop(0)

        ordered = [(assigned_vecs[idx_e], _valence_orbital_occno(entries[idx_e])) for idx_e in range(len(entries))]

        # Terminal sp dative-pi triple bond (e.g. CO, CN-, isocyanides):
        # Match build_atom_orbitals(hybridization="sp") so that:
        # 1. Group 14 acceptor (4 valence e-, empty dative pi_x acceptor orbital) places its
        #    outward sigma lone-pair spin opposite its sigma_in/pi_y bonding spin ([0,1] vs [1,0]),
        #    and shares half an electron per pi orbital from the Group 16 donor ([1,0] on both
        #    pi_x and pi_y, conserving 4.0 total valence electrons on C).
        # 2. Group 16 donor (6 valence e-, doubly occupied dative pi_x donor in dative_pi_indices)
        #    shares 0.5 electron from its dative pi_x lone pair across pi_x and pi_y ([0.5, 1.0]
        #    or [1.0, 0.5]), conserving 6.0 total valence electrons on O and breaking the closed
        #    4-electron minority-spin shell that otherwise suppresses sp sigma hybridization.
        if (
            not has_d
            and hyb_label == "sp"
            and len(directed_sigma_indices) == 1
            and len(pi_indices) == 2
        ):
            total_val_e = sum(sum(occ) for _, occ in ordered)
            sigma_idx = directed_sigma_indices[0]
            sigma_occ = ordered[sigma_idx][1]
            # Group 14 acceptor: 4 valence electrons, 1 empty pi orbital with connect_to
            has_empty_dative_pi = any(
                entries[p_i].occupation == 0 and entries[p_i].connect_to
                for p_i in pi_indices
            )
            if abs(total_val_e - 4.0) < 1e-6 and has_empty_dative_pi and rem_indices:
                bond_occ = [1.0, 0.0] if sigma_occ[0] >= sigma_occ[1] else [0.0, 1.0]
                lp_occ = [0.0, 1.0] if bond_occ[0] >= bond_occ[1] else [1.0, 0.0]
                ordered[rem_indices[0]] = (ordered[rem_indices[0]][0], lp_occ)
                ordered[sigma_idx] = (ordered[sigma_idx][0], list(bond_occ))
                for p_i in pi_indices:
                    ordered[p_i] = (ordered[p_i][0], list(bond_occ))
            # Group 16 donor: 6 valence electrons, 1 doubly-occupied dative pi donor in dative_pi_indices
            elif abs(total_val_e - 6.0) < 1e-6 and len(dative_pi_indices) == 1 and rem_indices:
                bond_occ = [1.0, 0.0] if sigma_occ[0] >= sigma_occ[1] else [0.0, 1.0]
                pi_occ = [0.5, 1.0] if bond_occ[0] < bond_occ[1] else [1.0, 0.5]
                ordered[rem_indices[0]] = (ordered[rem_indices[0]][0], [1.0, 1.0])
                ordered[sigma_idx] = (ordered[sigma_idx][0], list(bond_occ))
                for p_i in pi_indices:
                    ordered[p_i] = (ordered[p_i][0], list(pi_occ))

    if has_d and vs.d_indices:
        slots = sorted([vs.s_index, *vs.p_indices, *vs.d_indices])
    else:
        slots = sorted([vs.s_index, *vs.p_indices])

    for slot, (v, occ) in zip(slots[: len(ordered)], ordered):
        eps_val = vs.eps_s * (v[0] ** 2)
        if len(v) > 1 and vs.radial_p:
            eps_val += vs.eps_p * sum(v[1:4] ** 2)
        if len(v) > 4 and vs.radial_d:
            eps_val += vs.eps_d * sum(v[4:9] ** 2)
        lib[slot] = InitialOrbital(
            eigval=[float(eps_val), float(eps_val)],
            occno=list(occ),
            coefficients=_vec_to_coefficients(v, vs),
        )

    return lib


def apply_hybrid_seeds(
    deck,
    bonds: Optional[Dict[Tuple[int, int], int]] = None,
    s_in_sigma: bool = True,
    auto_propose: bool = True,
) -> None:
    """
    Set ``starting_orbitals`` on every primary atom of ``deck`` using:
    1. Explicit ``AtomConfig.valence_config`` when present on an atom,
    2. Explicit ``AtomConfig.hybridization`` when present on an atom,
    3. Explicit ``bonds`` graph when passed to ``apply_hybrid_seeds``, or
    4. Automated ``propose_valence_config(deck)`` when ``bonds is None`` and ``auto_propose=True``.
    """
    from phatpsy.spin import build_connectivity_graph

    elems = deck.resolve_elements()
    xyz = [np.asarray(a.xyz, dtype=float) for a in deck.geometry]
    n_atoms = len(xyz)

    # Ensure every atom has a stable ID for connect_to resolution
    elem_counts: Dict[str, int] = {}
    existing_ids = {a.id for a in deck.geometry if getattr(a, "id", None)}
    for a in deck.geometry:
        sym = a.element.strip().capitalize()
        elem_counts[sym] = elem_counts.get(sym, 0) + 1
        if not getattr(a, "id", None):
            cand = f"{sym}{elem_counts[sym]}"
            while cand in existing_ids:
                elem_counts[sym] += 1
                cand = f"{sym}{elem_counts[sym]}"
            a.id = cand
            existing_ids.add(cand)

    atoms_by_id = {a.id: a for a in deck.geometry if getattr(a, "id", None)}

    neighbours: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(n_atoms)}
    if bonds:
        for (i, j), order in bonds.items():
            neighbours[i].append((j, int(order)))
            neighbours[j].append((i, int(order)))
    else:
        adj = build_connectivity_graph(deck.geometry)
        for i_at in range(n_atoms):
            nbr_list = list(adj[i_at]) if i_at < len(adj) else []
            nbr_list.sort(key=lambda j_at: float(np.linalg.norm(xyz[j_at] - xyz[i_at])))
            hyb_i = getattr(deck.geometry[i_at], "hybridization", None)
            if hyb_i:
                spec_i = get_hybridization_spec(hyb_i)
                nbr_list = nbr_list[: max(spec_i.n_sigma, len(spec_i.canonical_sigma_dirs))]
            for j_at in nbr_list:
                order_ij = (
                    3
                    if (
                        getattr(deck.geometry[i_at], "hybridization", None) == "sp"
                        and getattr(deck.geometry[j_at], "hybridization", None) == "sp"
                    )
                    else 1
                )
                neighbours[i_at].append((j_at, order_ij))
        if any(getattr(a, "hybridization", None) == "sp" for a in deck.geometry):
            if n_atoms == 2 and not neighbours[0]:
                neighbours[0].append((1, 3))
                neighbours[1].append((0, 3))
            else:
                sp_indices = [idx for idx, a in enumerate(deck.geometry) if getattr(a, "hybridization", None) == "sp"]
                for i_sp in sp_indices:
                    if not neighbours[i_sp]:
                        other_sp = [j_sp for j_sp in sp_indices if j_sp != i_sp]
                        if other_sp:
                            j_nearest = min(other_sp, key=lambda j_sp: float(np.linalg.norm(xyz[j_sp] - xyz[i_sp])))
                            neighbours[i_sp].append((j_nearest, 3))

    def sigma_dirs_of(i: int) -> List[np.ndarray]:
        return [_unit(xyz[j] - xyz[i]) for j, _ in neighbours[i] if float(np.linalg.norm(xyz[j] - xyz[i])) > 1e-8]

    # Lazily compute automated valence_config proposals if needed for unconfigured atoms
    proposed_vc: Optional[Dict[str, List[Any]]] = None

    for i, atom in enumerate(deck.geometry):
        if getattr(atom, "virtual", False) or getattr(atom, "equivalent_to", None):
            continue
        sym = atom.element.strip().capitalize()
        sigma = sigma_dirs_of(i)
        atom_charge = float(getattr(atom, "charge", 0.0) or 0.0)

        # 1. Explicit per-orbital valence_config on atom takes highest precedence
        if getattr(atom, "valence_config", None):
            atom.starting_orbitals = build_atom_orbitals_from_valence_config(
                elem=elems[sym],
                atom=atom,
                atoms_by_id=atoms_by_id,
                s_in_sigma=s_in_sigma,
            )
            continue

        # 2. Explicit named hybridization on atom
        if getattr(atom, "hybridization", None):
            atom.starting_orbitals = build_atom_orbitals(
                elem=elems[sym],
                spin=atom.spin or "up",
                sigma_dirs=sigma,
                hybridization=atom.hybridization,
                charge=atom_charge,
            )
            continue

        # 3. Explicit bonds dictionary passed to apply_hybrid_seeds
        if bonds is not None:
            if not neighbours[i]:
                continue
            n_pi = sum(order - 1 for _, order in neighbours[i])
            pi_dirs: List[np.ndarray] = []
            if n_pi:
                normal = _plane_normal(sigma)
                if normal is None:  # borrow the partner's plane
                    for j, order in neighbours[i]:
                        if order > 1:
                            normal = _plane_normal(sigma_dirs_of(j))
                            if normal is not None:
                                break
                if n_pi == 1:
                    if normal is None and neighbours[i]:
                        j = next((j for j, order in neighbours[i] if order > 1), neighbours[i][0][0])
                        normal = _canonical_perpendicular_pair(xyz[j] - xyz[i])[0]
                    elif normal is None:
                        normal = np.array([0.0, 0.0, 1.0])
                    pi_dirs = [normal]
                elif n_pi == 2:
                    if len(sigma) == 1:
                        axis = sigma[0]
                    elif len(sigma) == 2 and abs(float(np.dot(sigma[0], sigma[1])) + 1.0) < 1e-3:
                        axis = sigma[0]
                    else:
                        raise NotImplementedError("two pi bonds on a non-linear centre")
                    pi_dirs = list(_canonical_perpendicular_pair(axis))
                else:
                    raise NotImplementedError("more than two pi bonds on one atom")

            atom.starting_orbitals = build_atom_orbitals(
                elem=elems[sym],
                spin=atom.spin or "up",
                sigma_dirs=sigma,
                pi_dirs=pi_dirs,
                s_in_sigma=s_in_sigma,
                charge=atom_charge,
            )
            continue

        # 4. Automated proposal via propose_valence_config when neither valence_config, hybridization, nor bonds is given
        if auto_propose:
            if proposed_vc is None:
                from phatpsy.builder import propose_valence_config
                proposed_vc = propose_valence_config(deck)
            vc_for_atom = proposed_vc.get(atom.id)
            if vc_for_atom:
                temp_atom = replace(atom, valence_config=vc_for_atom)
                atom.starting_orbitals = build_atom_orbitals_from_valence_config(
                    elem=elems[sym],
                    atom=temp_atom,
                    atoms_by_id=atoms_by_id,
                    s_in_sigma=s_in_sigma,
                )

