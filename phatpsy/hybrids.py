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
    eps_s = float(orbs[k_s].eigval[0])

    # p-orbitals
    p_orbs = [k for k, o in enumerate(orbs) if ls[k][0] == 1]
    p_group = sorted(p_orbs[-3:]) if len(p_orbs) >= 3 else []
    if p_group:
        k0 = next((k for k in p_group if ls[k][1] == 0), p_group[0])
        radial_p = {int(c[0]): float(c[2]) for c in orbs[k0].coefficients}
        eps_p = float(orbs[k0].eigval[0])
    else:
        radial_p, eps_p = {}, 0.0

    # d-orbitals (transition metals, heavy elements)
    d_orbs = [k for k, o in enumerate(orbs) if ls[k][0] == 2]
    d_group = sorted(d_orbs[-5:]) if len(d_orbs) >= 5 else []
    if d_group:
        kd0 = next((k for k in d_group if ls[k][1] == 0), d_group[0])
        radial_d = {int(c[0]): float(c[2]) for c in orbs[kd0].coefficients}
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
            p_comp = np.sqrt(1.0 - w * w) * _unit(d)
            v = np.zeros(dim, dtype=np.float64)
            v[0] = w
            v[1:4] = p_comp
            rows.append(v)
        T = np.array(rows)
        evals, evecs = np.linalg.eigh(T @ T.T)
        B = (evecs @ np.diag(evals ** -0.5) @ evecs.T) @ T  # symmetric orthonormalization
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
    evals, evecs = np.linalg.eigh(proj)
    complement = [evecs[:, k] for k in range(dim) if evals[k] > 0.5]
    # deterministic sign
    complement = [c if next(x for x in c if abs(x) > 1e-9) > 0 else -c for c in complement]

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

    # Check s-only valence shell (H, He)
    if not vs.p_indices and not vs.d_indices:
        n_bonds = len(sigma_dirs) if spec is None else spec.n_sigma
        if n_bonds > 1:
            raise ValueError(f"{elem.symbol}: s-only valence shell supports at most 1 bond")
        k = vs.s_index
        lib[k] = replace(lib[k], occno=list(bond_occ))
        return lib

    has_d = (spec is not None and spec.has_d_orbitals) or bool(vs.d_indices) or bool(delta_dirs)

    if spec is not None:
        rot_sigma, rot_pi, rot_delta, R = align_hybrid_to_neighbors(spec, sigma_dirs)
        if spec.has_d_orbitals:
            vecs = _construct_d_hybrid_vectors(spec, R)
        else:
            n_bond_e = spec.n_sigma + spec.n_pi + spec.n_delta
            leftover = vs.valence_electrons - n_bond_e
            if leftover < 0:
                raise ValueError(f"{elem.symbol}: {n_bond_e} bonds exceed {vs.valence_electrons} valence electrons")
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
        leftover = vs.valence_electrons - n_bond_e
        if leftover < 0:
            raise ValueError(f"{elem.symbol}: {n_bond_e} bonds exceed {vs.valence_electrons} valence electrons")
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
        leftover = max(vs.valence_electrons - n_bond_e, 0)
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
        ordered: List[Tuple[np.ndarray, List[float]]] = []
        ordered += [(v, [1.0, 1.0]) for v in vecs.get("lone", [])]
        ordered += [(v, list(bond_occ)) for v in vecs.get("sigma", [])]
        ordered += [(v, list(bond_occ)) for v in vecs.get("pi", [])]
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


def apply_hybrid_seeds(
    deck,
    bonds: Optional[Dict[Tuple[int, int], int]] = None,
    s_in_sigma: bool = True,
) -> None:
    """
    Set ``starting_orbitals`` on every atom of ``deck`` using either explicit bonds
    or assigned named hybridizations on ``AtomConfig.hybridization``.
    """
    elems = deck.resolve_elements()
    xyz = [np.asarray(a.xyz, dtype=float) for a in deck.geometry]
    neighbours: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(len(xyz))}
    if bonds:
        for (i, j), order in bonds.items():
            neighbours[i].append((j, int(order)))
            neighbours[j].append((i, int(order)))

    def sigma_dirs_of(i: int) -> List[np.ndarray]:
        return [_unit(xyz[j] - xyz[i]) for j, _ in neighbours[i]]

    for i, atom in enumerate(deck.geometry):
        sym = atom.element.strip().capitalize()
        sigma = sigma_dirs_of(i)

        # Check explicit named hybridization on atom
        if getattr(atom, "hybridization", None):
            atom.starting_orbitals = build_atom_orbitals(
                elem=elems[sym],
                spin=atom.spin or "up",
                sigma_dirs=sigma,
                hybridization=atom.hybridization,
            )
            continue

        # Automatic derivation from bond graph
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
        )
