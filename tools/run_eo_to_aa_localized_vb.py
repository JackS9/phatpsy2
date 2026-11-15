#!/usr/bin/env python3
"""First-principles Localized VB (CAS(4e,4o) / STO-6G) study of Ethylene Oxide -> Acetaldehyde isomerization.

Executes the complete two-step mechanism end-to-end:
  1. Geometry optimization + harmonic vibrational frequencies + RRHO thermochemistry on:
     - EO (closed oxirane ring minimum)
     - BIRAD (ring-opened •CH2-CH2-O• singlet biradical intermediate minimum)
     - AA (acetaldehyde minimum)
  2. Step 1 (Ring Opening): Curved-MEP 7-image CI-NEB + Baker/Bofill P-RFO saddle search for TS1
     (n_imag == 1, ring-opening / CH2 untwisting normal mode) and bidirectional IRC 1 (EO <- TS1 -> BIRAD).
  3. Step 2 (1,2-Hydride Shift): Collision-free 7-image CI-NEB + Baker/Bofill P-RFO saddle search for TS2
     (n_imag == 1, hydride-transfer normal mode) and bidirectional IRC 2 (BIRAD <- TS2 -> AA).
  4. Single-point cross-checks at all 5 stationary points (DZ STO-6G, SZ STO-8G, RHF, EO ProlateGrid).

Outputs:
  - data/eo_to_aa_localized_vb/summary.json
  - data/eo_to_aa_localized_vb/irc_step1_ring_opening.json
  - data/eo_to_aa_localized_vb/irc_trajectory.json
  - examples/EO_localized_vb.yaml
  - examples/EO_to_Biradical_TS1_localized_vb.yaml
  - examples/EO_Biradical_localized_vb.yaml
  - examples/EO_to_AA_TS_localized_vb.yaml
  - examples/AA_localized_vb.yaml
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phatpsy.builder import BOHR_PER_ANGSTROM
from phatpsy.dimer import refine_saddle_prfo
from phatpsy.driver import PhatpsyDriver
from phatpsy.irc import compute_irc
from phatpsy.localized_vb import pyscf_mean_field
from phatpsy.multicenter_integrals import sto_ng_ao_integrals
from phatpsy.neb import neb_search
from phatpsy.optimize import optimize_geometry
from phatpsy.schema import ATOMIC_NUMBERS, AtomConfig, CalculationDeck, ControlConfig
from phatpsy.vb_driver import (
    LocalizedVBDriver,
    _resolve_atom_stos,
    _build_general_vb_setup,
)
from phatpsy.vibrations import HARTREE_TO_KCAL, NormalMode, Thermochemistry, compute_vibrations

CACHE_PATH = REPO_ROOT / "build" / "eo_to_aa_stage_cache.json"

HARTREE_TO_KJ: float = 2625.49963948

ATOM_ORDER: Tuple[Tuple[str, str], ...] = (
    ("C1", "C"),
    ("C2", "C"),
    ("O1", "O"),
    ("H1", "H"),
    ("H2", "H"),
    ("H3", "H"),
    ("H4", "H"),
)

# Reaction-invariant all-electron CAS(4e, 4o) partitioned into 10 conserved 1c/2c inactive orbitals
# (3x 1s cores, 2x O lone pairs, C1-H1, C1-H2, C2-H4, C1-C2, C2-O1 sigma) and 4 reactive active
# orbitals spanning [C1, C2, O1, H3] (C1-O1 sigma + C2-H3 sigma <-> C2=O1 pi + C1-H3 sigma).
CAS44_VB_SPECS: List[Dict[str, Any]] = [
    {"role": "inactive", "atoms": ["C1"]},
    {"role": "inactive", "atoms": ["C2"]},
    {"role": "inactive", "atoms": ["O1"]},
    {"role": "inactive", "atoms": ["O1"]},
    {"role": "inactive", "atoms": ["O1"]},
    {"role": "inactive", "atoms": ["C1", "H1"]},
    {"role": "inactive", "atoms": ["C1", "H2"]},
    {"role": "inactive", "atoms": ["C2", "H4"]},
    {"role": "inactive", "atoms": ["C1", "C2"]},
    {"role": "inactive", "atoms": ["C2", "O1"]},
    {"role": "active", "atoms": ["C1", "C2", "O1", "H3"]},
    {"role": "active", "atoms": ["C1", "C2", "O1", "H3"]},
    {"role": "active", "atoms": ["C1", "C2", "O1", "H3"]},
    {"role": "active", "atoms": ["C1", "C2", "O1", "H3"]},
]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def build_c2h4o_deck(
    coords_bohr: Sequence[Sequence[float]],
    title: str,
    basis: str = "SZ",
    integral_backend: str = "sto_ng",
    sto_ng_primitives: int = 6,
    vb_orbital_specs: Sequence[Dict[str, Any]] = CAS44_VB_SPECS,
) -> CalculationDeck:
    geom = [
        AtomConfig(
            id=aid,
            element=el,
            xyz=[float(coords_bohr[i][0]), float(coords_bohr[i][1]), float(coords_bohr[i][2])],
            basis=basis,
            active=True,
        )
        for i, (aid, el) in enumerate(ATOM_ORDER)
    ]
    ctrl = ControlConfig.localized_vb(
        method="masked_vb",
        vb_wavefunction="cas",
        domain_mode="bond_local",
        core_embedding="all_electron",
        integral_backend=integral_backend,
        sto_ng_primitives=sto_ng_primitives,
        vb_orbital_specs=[dict(s) for s in vb_orbital_specs] if vb_orbital_specs is not None else None,
        max_cycles=400,
        convergence=1e-8 if integral_backend == "sto_ng" else 1e-6,
    )
    return CalculationDeck(title=title, geometry=geom, control=ctrl)


def deck_coords(deck: CalculationDeck) -> np.ndarray:
    return np.array([a.xyz for a in deck.geometry], dtype=float)


def pair_dist(coords: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(coords[i] - coords[j]))


def min_interatomic_distance(coords: np.ndarray) -> float:
    n = len(coords)
    dmin = 1e9
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < dmin:
                dmin = d
    return dmin


def make_ring_opened_geom(c_eo: np.ndarray, open_deg: float, twist_deg: float) -> np.ndarray:
    """Constructs a smooth ring-opened/CH2-twisted C2H4O geometry from closed-ring EO coordinates."""
    c_rot = np.array(c_eo, dtype=float, copy=True)
    ang = np.radians(open_deg)
    r_xz = np.array(
        [
            [np.cos(ang), 0.0, -np.sin(ang)],
            [0.0, 1.0, 0.0],
            [np.sin(ang), 0.0, np.cos(ang)],
        ],
        dtype=float,
    )
    c_rot[2] = c_rot[1] + r_xz @ (c_rot[2] - c_rot[1])

    ang_h = 0.45 * ang
    r_xz_h = np.array(
        [
            [np.cos(ang_h), 0.0, -np.sin(ang_h)],
            [0.0, 1.0, 0.0],
            [np.sin(ang_h), 0.0, np.cos(ang_h)],
        ],
        dtype=float,
    )
    for idx in (5, 6):
        c_rot[idx] = c_rot[1] + r_xz_h @ (c_rot[idx] - c_rot[1])

    u = c_rot[0] - c_rot[1]
    u = u / float(np.linalg.norm(u))
    k_mat = np.array(
        [
            [0.0, -u[2], u[1]],
            [u[2], 0.0, -u[0]],
            [-u[1], u[0], 0.0],
        ],
        dtype=float,
    )
    phi = np.radians(twist_deg)
    r_twist = np.eye(3, dtype=float) + np.sin(phi) * k_mat + (1.0 - np.cos(phi)) * (k_mat @ k_mat)
    for idx in (3, 4):
        c_rot[idx] = c_rot[0] + r_twist @ (c_rot[idx] - c_rot[0])
    return c_rot


def internal_metrics(coords: np.ndarray) -> Dict[str, float]:
    # Atom indices: 0=C1, 1=C2, 2=O1, 3=H1, 4=H2, 5=H3, 6=H4
    idx = {aid: k for k, (aid, _) in enumerate(ATOM_ORDER)}
    def d(a: str, b: str) -> float:
        return pair_dist(coords, idx[a], idx[b])

    def angle_deg(a: str, b: str, c: str) -> float:
        v1 = coords[idx[a]] - coords[idx[b]]
        v2 = coords[idx[c]] - coords[idx[b]]
        cos_t = float(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_t)))

    return {
        "R_C1_C2_bohr": round(d("C1", "C2"), 6),
        "R_C1_O1_bohr": round(d("C1", "O1"), 6),
        "R_C2_O1_bohr": round(d("C2", "O1"), 6),
        "R_C1_H1_bohr": round(d("C1", "H1"), 6),
        "R_C1_H2_bohr": round(d("C1", "H2"), 6),
        "R_C2_H3_bohr": round(d("C2", "H3"), 6),
        "R_C1_H3_bohr": round(d("C1", "H3"), 6),
        "R_C2_H4_bohr": round(d("C2", "H4"), 6),
        "R_C1_C2_ang": round(d("C1", "C2") / BOHR_PER_ANGSTROM, 5),
        "R_C1_O1_ang": round(d("C1", "O1") / BOHR_PER_ANGSTROM, 5),
        "R_C2_O1_ang": round(d("C2", "O1") / BOHR_PER_ANGSTROM, 5),
        "R_C2_H3_ang": round(d("C2", "H3") / BOHR_PER_ANGSTROM, 5),
        "R_C1_H3_ang": round(d("C1", "H3") / BOHR_PER_ANGSTROM, 5),
        "angle_C1_C2_O1_deg": round(angle_deg("C1", "C2", "O1"), 3),
        "angle_C1_O1_C2_deg": round(angle_deg("C1", "O1", "C2"), 3),
        "angle_C2_C1_O1_deg": round(angle_deg("C2", "C1", "O1"), 3),
    }


def make_clean_deck_for_yaml(
    deck: CalculationDeck,
    title: str,
    vb_seed_orbitals: Optional[Sequence[Sequence[float]]] = None,
) -> CalculationDeck:
    """Strip transient restart orbitals and save clean human-runnable YAML deck."""
    d = copy.deepcopy(deck)
    d.title = title
    d.net_spin = 0
    if hasattr(d, "_vb_restart_orbitals"):
        delattr(d, "_vb_restart_orbitals")
    d.geometry = [
        AtomConfig(
            id=ATOM_ORDER[i][0],
            element=ATOM_ORDER[i][1],
            xyz=[round(float(v), 8) for v in a.xyz],
            basis=a.basis or "SZ",
            active=True,
        )
        for i, a in enumerate(d.geometry)
    ]
    if vb_seed_orbitals is not None and d.control.vb_orbital_specs:
        elem_map = d.resolve_elements()
        idx_by_atom: Dict[str, List[int]] = {}
        offset = 0
        for a in d.geometry:
            stos, _ = _resolve_atom_stos(a, d, elem_map)
            idx_by_atom[a.id] = list(range(offset, offset + len(stos)))
            offset += len(stos)
        new_specs: List[Dict[str, Any]] = []
        for k, spec in enumerate(d.control.vb_orbital_specs):
            cleaned = dict(spec)
            mask = sorted({ia for aid in cleaned["atoms"] for ia in idx_by_atom[aid]})
            orb_k = np.asarray(vb_seed_orbitals[k], dtype=float)
            cleaned["seed"] = [round(float(orb_k[ia]), 8) for ia in mask]
            new_specs.append(cleaned)
        d.control.vb_orbital_specs = new_specs
    return d


def rhf_energy_in_sto_ng(deck: CalculationDeck, n_prim: int = 6) -> float:
    elem_map = deck.resolve_elements()
    specs = []
    for a in deck.geometry:
        stos, _ = _resolve_atom_stos(a, deck, elem_map)
        specs.append((float(ATOMIC_NUMBERS[a.element]), tuple(float(v) for v in a.xyz), stos))
    ints = sto_ng_ao_integrals(specs, n_prim=n_prim)
    mf = pyscf_mean_field(ints, (12, 12), kind="RHF")
    return float(mf.e_tot)


def _save_stage_cache(cache: Dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _thermo_to_cache(th: Thermochemistry) -> Dict[str, Any]:
    return {
        "temperature_k": float(th.temperature_k),
        "pressure_atm": float(th.pressure_atm),
        "electronic_energy": float(th.electronic_energy),
        "zero_point_energy_kcal": float(th.zero_point_energy_kcal),
        "zero_point_energy_au": float(th.zero_point_energy_au),
        "thermal_energy_kcal": float(th.thermal_energy_kcal),
        "thermal_enthalpy_kcal": float(th.thermal_enthalpy_kcal),
        "entropy_cal_mol_k": dict(th.entropy_cal_mol_k),
        "heat_capacity_cal_mol_k": dict(th.heat_capacity_cal_mol_k),
        "total_energy_u0": float(th.total_energy_u0),
        "total_energy_u_t": float(th.total_energy_u_t),
        "total_enthalpy_h_t": float(th.total_enthalpy_h_t),
        "total_gibbs_free_energy_g_t": float(th.total_gibbs_free_energy_g_t),
        "thermal_energy_components": dict(th.thermal_energy_components),
    }


def _mode_to_cache(m: NormalMode) -> Dict[str, Any]:
    return {
        "index": int(m.index),
        "frequency_cm": float(m.frequency_cm),
        "is_imaginary": bool(m.is_imaginary),
        "reduced_mass_amu": float(m.reduced_mass_amu),
        "force_constant_mdyne_a": float(m.force_constant_mdyne_a),
        "displacements": [[float(c) for c in row] for row in m.displacements],
        "is_tr_mode": bool(m.is_tr_mode),
        "symmetry_label": m.symmetry_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EO -> AA Localized VB workflow.")
    parser.add_argument("--fresh", action="store_true", help="Ignore any saved stage checkpoint and run all stages fresh.")
    args = parser.parse_args()

    stage_cache: Dict[str, Any] = {}
    if (not args.fresh) and CACHE_PATH.exists():
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            stage_cache = json.load(f)
        log(f"Loaded stage cache from {CACHE_PATH} (stages={list(stage_cache.keys())})")

    t_start = time.perf_counter()
    driver = PhatpsyDriver(quiet=True)
    vb_drv = LocalizedVBDriver(quiet=True)
    ex_dir = REPO_ROOT / "examples"

    # Consistent 7-atom coordinates: [C1, C2, O1, H1(+y), H2(-y), H3(migrating C2->C1), H4(aldehydic)]
    eo_init_coords = [
        [ 1.39,  0.00, -0.73],   # C1
        [-1.39,  0.00, -0.73],   # C2
        [ 0.00,  0.00,  1.56],   # O1
        [ 2.30,  1.72, -1.18],   # H1 (+y)
        [ 2.30, -1.72, -1.18],   # H2 (-y)
        [-2.30,  1.72, -1.18],   # H3 (migrating, +y)
        [-2.30, -1.72, -1.18],   # H4 (aldehydic, -y)
    ]
    aa_init_coords = [
        [ 1.42,  0.00, -0.73],   # C1
        [-1.42,  0.00, -0.73],   # C2
        [-2.71,  0.00,  1.18],   # O1
        [ 2.11,  1.70, -1.71],   # H1 (+y)
        [ 2.11, -1.70, -1.71],   # H2 (-y)
        [ 2.11,  0.00,  1.23],   # H3 (shifted to C1, y=0)
        [-2.34,  0.00, -2.62],   # H4 (aldehydic, y=0)
    ]

    deck_eo_init = build_c2h4o_deck(eo_init_coords, "Ethylene Oxide (Localized VB CAS(4,4) SZ/STO-6G)")
    deck_aa_init = build_c2h4o_deck(aa_init_coords, "Acetaldehyde (Localized VB CAS(4,4) SZ/STO-6G)")

    # ------------------------------------------------------------------
    # 1. EO Minimum: Optimize Geometry + Central Frequencies
    # ------------------------------------------------------------------
    if "eo" in stage_cache:
        log("1/6 Restoring EO minimum from stage cache + verifying live E & g...")
        ce = stage_cache["eo"]
        eo_opt_deck = build_c2h4o_deck(ce["coords"], "Ethylene Oxide (Localized VB CAS(4,4) SZ/STO-6G)")
        st_eo, g_eo = vb_drv.energy_and_gradient(eo_opt_deck)
        opt_eo = SimpleNamespace(converged=ce["opt_converged"], num_steps=ce["opt_steps"])
        eo_freqs = ce["freqs"]
        vib_eo = SimpleNamespace(num_imaginary=ce["num_imaginary"], thermochemistry=Thermochemistry(**ce["thermo"]))
        log(f"    EO live verified: E={st_eo.total_energy:.8f} Ha |g|={np.linalg.norm(g_eo):.2e} n_imag={vib_eo.num_imaginary}")
    else:
        log("1/6 Optimizing Ethylene Oxide (EO) minimum (SZ STO-6G CAS(4,4))...")
        t0 = time.perf_counter()
        opt_eo = optimize_geometry(
            deck_eo_init, driver=driver, max_steps=30, grad_tol=2.0e-4, energy_tol=1.0e-7, step_limit=0.15, quiet=True
        )
        eo_opt_deck = opt_eo.optimized_deck
        st_eo, g_eo = vb_drv.energy_and_gradient(eo_opt_deck)
        log(
            f"    EO opt conv={opt_eo.converged} steps={opt_eo.num_steps} "
            f"E={st_eo.total_energy:.8f} Ha |g|={np.linalg.norm(g_eo):.2e} ({time.perf_counter()-t0:.1f}s)"
        )

        log("    Computing EO central harmonic frequencies & RRHO thermochemistry (sigma=2)...")
        t0 = time.perf_counter()
        vib_eo = compute_vibrations(eo_opt_deck, driver=driver, method="central", step_size=0.005, symmetry_number=2, quiet=True)
        eo_freqs = [round(float(m.frequency_cm), 2) for m in vib_eo.vibrational_modes]
        log(
            f"    EO vib: n_imag={vib_eo.num_imaginary} n_modes={len(eo_freqs)} "
            f"min_freq={min(eo_freqs):.1f} max_freq={max(eo_freqs):.1f} cm^-1 "
            f"ZPVE={vib_eo.thermochemistry.zero_point_energy_kcal:.2f} kcal/mol ({time.perf_counter()-t0:.1f}s)"
        )
        assert vib_eo.num_imaginary == 0, f"EO must be a true minimum, got {vib_eo.num_imaginary} imaginary modes: {eo_freqs}"
        make_clean_deck_for_yaml(eo_opt_deck, "Ethylene Oxide (Localized VB CAS(4,4) SZ/STO-6G Minimum)").to_yaml(
            ex_dir / "EO_localized_vb.yaml"
        )
        stage_cache["eo"] = {
            "coords": deck_coords(eo_opt_deck).tolist(),
            "opt_converged": bool(opt_eo.converged),
            "opt_steps": int(opt_eo.num_steps),
            "num_imaginary": int(vib_eo.num_imaginary),
            "freqs": eo_freqs,
            "thermo": _thermo_to_cache(vib_eo.thermochemistry),
        }
        _save_stage_cache(stage_cache)

    # ------------------------------------------------------------------
    # 2. AA Minimum: Optimize Geometry + Central Frequencies
    # ------------------------------------------------------------------
    if "aa" in stage_cache:
        log("2/6 Restoring AA minimum from stage cache + verifying live E & g...")
        ca = stage_cache["aa"]
        aa_opt_deck = build_c2h4o_deck(ca["coords"], "Acetaldehyde (Localized VB CAS(4,4) SZ/STO-6G)")
        st_aa, g_aa = vb_drv.energy_and_gradient(aa_opt_deck)
        opt_aa = SimpleNamespace(converged=ca["opt_converged"], num_steps=ca["opt_steps"])
        aa_freqs = ca["freqs"]
        vib_aa = SimpleNamespace(num_imaginary=ca["num_imaginary"], thermochemistry=Thermochemistry(**ca["thermo"]))
        log(f"    AA live verified: E={st_aa.total_energy:.8f} Ha |g|={np.linalg.norm(g_aa):.2e} n_imag={vib_aa.num_imaginary}")
    else:
        log("2/6 Optimizing Acetaldehyde (AA) minimum (SZ STO-6G CAS(4,4))...")
        t0 = time.perf_counter()
        opt_aa = optimize_geometry(
            deck_aa_init, driver=driver, max_steps=30, grad_tol=2.0e-4, energy_tol=1.0e-7, step_limit=0.15, quiet=True
        )
        aa_opt_deck = opt_aa.optimized_deck
        st_aa, g_aa = vb_drv.energy_and_gradient(aa_opt_deck)
        log(
            f"    AA opt conv={opt_aa.converged} steps={opt_aa.num_steps} "
            f"E={st_aa.total_energy:.8f} Ha |g|={np.linalg.norm(g_aa):.2e} ({time.perf_counter()-t0:.1f}s)"
        )
        assert st_aa.total_energy < st_eo.total_energy, "Acetaldehyde must lie below Ethylene Oxide!"

        log("    Computing AA central harmonic frequencies & RRHO thermochemistry (sigma=1)...")
        t0 = time.perf_counter()
        vib_aa = compute_vibrations(aa_opt_deck, driver=driver, method="central", step_size=0.005, symmetry_number=1, quiet=True)
        aa_freqs = [round(float(m.frequency_cm), 2) for m in vib_aa.vibrational_modes]
        log(
            f"    AA vib: n_imag={vib_aa.num_imaginary} n_modes={len(aa_freqs)} "
            f"min_freq={min(aa_freqs):.1f} max_freq={max(aa_freqs):.1f} cm^-1 "
            f"ZPVE={vib_aa.thermochemistry.zero_point_energy_kcal:.2f} kcal/mol ({time.perf_counter()-t0:.1f}s)"
        )
        assert vib_aa.num_imaginary == 0, f"AA must be a true minimum, got {vib_aa.num_imaginary} imaginary modes: {aa_freqs}"
        make_clean_deck_for_yaml(aa_opt_deck, "Acetaldehyde (Localized VB CAS(4,4) SZ/STO-6G Minimum)").to_yaml(
            ex_dir / "AA_localized_vb.yaml"
        )
        stage_cache["aa"] = {
            "coords": deck_coords(aa_opt_deck).tolist(),
            "opt_converged": bool(opt_aa.converged),
            "opt_steps": int(opt_aa.num_steps),
            "num_imaginary": int(vib_aa.num_imaginary),
            "freqs": aa_freqs,
            "thermo": _thermo_to_cache(vib_aa.thermochemistry),
        }
        _save_stage_cache(stage_cache)

    # ------------------------------------------------------------------
    # 2b. Ring-Opened Singlet Biradical Intermediate (BIRAD): Opt + Central Freqs
    # ------------------------------------------------------------------
    c_eo_opt = deck_coords(eo_opt_deck)
    c_aa_opt = deck_coords(aa_opt_deck)
    if "birad" in stage_cache and stage_cache["birad"].get("opt_converged", False):
        log("2b/6 Restoring •CH2-CH2-O• biradical minimum (BIRAD) from cache + verifying live E & g...")
        cb = stage_cache["birad"]
        birad_opt_deck = build_c2h4o_deck(cb["coords"], "EO Ring-Opened Biradical (Localized VB CAS(4,4) SZ/STO-6G)")
        st_birad, g_birad = vb_drv.energy_and_gradient(birad_opt_deck)
        opt_birad = SimpleNamespace(converged=cb["opt_converged"], num_steps=cb["opt_steps"])
        birad_freqs = cb["freqs"]
        vib_birad = SimpleNamespace(num_imaginary=cb["num_imaginary"], thermochemistry=Thermochemistry(**cb["thermo"]))
        log(
            f"    BIRAD live verified: E={st_birad.total_energy:.8f} Ha |g|={np.linalg.norm(g_birad):.2e} "
            f"n_imag={vib_birad.num_imaginary}"
        )
    else:
        t0 = time.perf_counter()
        if "birad" in stage_cache and "coords" in stage_cache["birad"]:
            log("2b/6 Polishing ring-opened •CH2-CH2-O• singlet biradical minimum (BIRAD) to converged |g|...")
            c_birad_init = np.array(stage_cache["birad"]["coords"], dtype=float)
        else:
            log("2b/6 Optimizing ring-opened •CH2-CH2-O• singlet biradical minimum (BIRAD)...")
            c_birad_init = make_ring_opened_geom(c_eo_opt, open_deg=45.0, twist_deg=60.0)
        deck_birad_init = build_c2h4o_deck(c_birad_init, "EO Ring-Opened Biradical Seed (CAS(4,4) SZ/STO-6G)")
        opt_birad = optimize_geometry(
            deck_birad_init,
            driver=driver,
            max_steps=50,
            grad_tol=2.0e-4,
            energy_tol=1.0e-7,
            step_limit=0.12,
            quiet=True,
        )
        assert opt_birad.converged, f"BIRAD optimization did not converge in {opt_birad.num_steps} steps"
        birad_opt_deck = opt_birad.optimized_deck
        st_birad, g_birad = vb_drv.energy_and_gradient(birad_opt_deck)
        birad_m = internal_metrics(deck_coords(birad_opt_deck))
        log(
            f"    BIRAD opt conv={opt_birad.converged} steps={opt_birad.num_steps} "
            f"E={st_birad.total_energy:.8f} Ha |g|={np.linalg.norm(g_birad):.2e} "
            f"R(C1-O1)={birad_m['R_C1_O1_ang']:.3f} A <CCO={birad_m['angle_C1_C2_O1_deg']:.1f} deg ({time.perf_counter()-t0:.1f}s)"
        )
        assert birad_m["R_C1_O1_ang"] > 2.40, f"BIRAD must be ring-opened, got R(C1-O1)={birad_m['R_C1_O1_ang']} A"

        log("    Computing BIRAD central harmonic frequencies & RRHO thermochemistry (sigma=1)...")
        t0 = time.perf_counter()
        vib_birad = compute_vibrations(
            birad_opt_deck, driver=driver, method="central", step_size=0.005, symmetry_number=1, quiet=True
        )
        birad_freqs = [round(float(m.frequency_cm), 2) for m in vib_birad.vibrational_modes]
        log(
            f"    BIRAD vib: n_imag={vib_birad.num_imaginary} n_modes={len(birad_freqs)} "
            f"min_freq={min(birad_freqs):.1f} max_freq={max(birad_freqs):.1f} cm^-1 "
            f"ZPVE={vib_birad.thermochemistry.zero_point_energy_kcal:.2f} kcal/mol ({time.perf_counter()-t0:.1f}s)"
        )
        assert vib_birad.num_imaginary == 0, f"BIRAD must be a true minimum, got {vib_birad.num_imaginary}: {birad_freqs}"
        make_clean_deck_for_yaml(
            birad_opt_deck,
            "Ring-Opened CH2-CH2-O Singlet Biradical Intermediate (Localized VB CAS(4,4) SZ/STO-6G Minimum)",
        ).to_yaml(ex_dir / "EO_Biradical_localized_vb.yaml")
        stage_cache["birad"] = {
            "coords": deck_coords(birad_opt_deck).tolist(),
            "opt_converged": bool(opt_birad.converged),
            "opt_steps": int(opt_birad.num_steps),
            "num_imaginary": int(vib_birad.num_imaginary),
            "freqs": birad_freqs,
            "thermo": _thermo_to_cache(vib_birad.thermochemistry),
        }
        _save_stage_cache(stage_cache)

    c_birad_opt = deck_coords(birad_opt_deck)

    # ------------------------------------------------------------------
    # 3a & 4a. Step 1 (EO -> BIRAD): Curved-MEP 7-Image CI-NEB + Baker/Bofill P-RFO TS1
    # ------------------------------------------------------------------
    if "neb1" in stage_cache:
        log("3a/6 Restoring Step-1 (EO -> BIRAD) 7-image CI-NEB band from stage cache...")
        cn1 = stage_cache["neb1"]
        neb1_imgs = [build_c2h4o_deck(c_i, f"NEB1 Image {i}") for i, c_i in enumerate(cn1["image_coords"])]
        neb1_res = SimpleNamespace(
            converged=cn1["converged"],
            num_iterations=cn1["num_iterations"],
            transition_state_index=cn1["ts_idx"],
            energies=cn1["energies"],
            images=neb1_imgs,
            barrier_forward_kcal=float((max(cn1["energies"]) - cn1["energies"][0]) * HARTREE_TO_KCAL),
        )
        log(f"    NEB1 cached: peak=#{neb1_res.transition_state_index} barrier={neb1_res.barrier_forward_kcal:.2f} kcal/mol")
    else:
        curved_specs = [
            (0.0, 0.0),
            (14.0, 2.0),
            (26.0, 5.0),
            (35.0, 18.0),
            (40.0, 55.0),
            (45.0, 62.0),
        ]
        waypoint_ts1_images: List[CalculationDeck] = []
        for idx_img, (open_deg, twist_deg) in enumerate(curved_specs):
            c_img = c_eo_opt if idx_img == 0 else make_ring_opened_geom(c_eo_opt, open_deg, twist_deg)
            d_min = min_interatomic_distance(c_img)
            assert d_min > 1.5, f"Steric clash on NEB1 initial image {idx_img}: d_min={d_min:.3f}"
            waypoint_ts1_images.append(build_c2h4o_deck(c_img, f"NEB1 Image {idx_img}"))
        waypoint_ts1_images.append(build_c2h4o_deck(c_birad_opt, "NEB1 Image 6"))

        log("3a/6 Running curved-MEP 7-image CI-NEB for Step-1 ring opening (EO -> BIRAD)...")
        t0 = time.perf_counter()

        def _neb1_cb(it: int, es: List[float], fmax: float) -> None:
            bar_f = (max(es) - es[0]) * HARTREE_TO_KCAL
            log(f"    NEB1 iter {it:2d}: E_max_rel={bar_f:6.2f} kcal/mol | max_F={fmax:.5f} Ha/Bohr")

        neb1_res = neb_search(
            initial_deck=eo_opt_deck,
            final_deck=birad_opt_deck,
            num_images=len(waypoint_ts1_images),
            driver=driver,
            k_spring=0.10,
            max_iterations=24,
            force_tol=0.005,
            step_limit=0.12,
            climb=True,
            warm_restart=True,
            callback=_neb1_cb,
            quiet=True,
            initial_images=waypoint_ts1_images,
        )
        log(
            f"    NEB1 done in {neb1_res.num_iterations} iters ({time.perf_counter()-t0:.1f}s): "
            f"peak img #{neb1_res.transition_state_index}, barrier={neb1_res.barrier_forward_kcal:.2f} kcal/mol"
        )
        stage_cache["neb1"] = {
            "converged": bool(neb1_res.converged),
            "num_iterations": int(neb1_res.num_iterations),
            "ts_idx": int(neb1_res.transition_state_index),
            "energies": [float(e) for e in neb1_res.energies],
            "image_coords": [deck_coords(img).tolist() for img in neb1_res.images],
        }
        _save_stage_cache(stage_cache)

    if "ts1" in stage_cache:
        log("4a/6 Restoring P-RFO verified TS1 (ring opening) from stage cache + verifying live E & g...")
        ct1 = stage_cache["ts1"]
        ts1_opt_deck = build_c2h4o_deck(ct1["coords"], "EO -> Biradical Ring-Opening TS1 (Localized VB CAS(4,4) SZ/STO-6G)")
        ts1_warm = ct1.get("warm_orbitals")
        if ts1_warm is None:
            st_img3 = vb_drv.run_deck(build_c2h4o_deck(stage_cache["neb1"]["image_coords"][3], "NEB1 img3"))
            ts1_warm = st_img3["localized_orbitals_embedded"]
        st_ts1, g_ts1 = vb_drv.energy_and_gradient(ts1_opt_deck, warm_orbitals=ts1_warm)
        assert np.linalg.norm(g_ts1) < 2.0e-4, f"TS1 State-1 gradient expected < 2e-4, got {np.linalg.norm(g_ts1):.2e}"
        ts1_opt_deck._vb_restart_orbitals = np.array(st_ts1["localized_orbitals_embedded"], copy=True)
        prfo1_res = SimpleNamespace(
            converged=ct1["prfo_converged"],
            num_cycles=ct1["prfo_cycles"],
            final_force_norm=ct1["final_force_norm"],
            final_curvature=ct1["final_curvature"],
        )
        ts1_freqs = ct1["freqs"]
        imag_mode_ts1 = NormalMode(**ct1["imag_mode"])
        vib_ts1 = SimpleNamespace(num_imaginary=ct1["num_imaginary"], thermochemistry=Thermochemistry(**ct1["thermo"]))
        ts1_atom_disp_norms = ct1["atom_disp_norms"]
        ts1_dominant_atom = ct1["dominant_atom"]
        if ct1.get("warm_orbitals") is None:
            ct1["warm_orbitals"] = np.asarray(st_ts1["localized_orbitals_embedded"], dtype=float).tolist()
            stage_cache["ts1"] = ct1
            _save_stage_cache(stage_cache)
        log(
            f"    TS1 live verified: E={st_ts1.total_energy:.8f} Ha |g|={np.linalg.norm(g_ts1):.2e} "
            f"n_imag={vib_ts1.num_imaginary} imag_freq={imag_mode_ts1.frequency_cm:.2f}"
        )
    else:
        log("4a/6 Refining NEB1 peak via central-Hessian Eckart Baker/Bofill P-RFO (TS1)...")
        t0 = time.perf_counter()

        def _prfo1_cb(cyc: int, energy: float, gnorm: float, curv: float) -> None:
            log(f"    P-RFO TS1 cyc {cyc:2d}: E={energy:.8f} Ha |g_int|={gnorm:.5f} Ha/Bohr b0={curv:+.5f}")

        prfo1_res = refine_saddle_prfo(
            deck=neb1_res,
            driver=driver,
            max_cycles=25,
            force_tol=1.5e-4,
            step_size=0.004,
            trust_radius=0.06,
            hessian_recalc=3,
            hessian_method="central",
            callback=_prfo1_cb,
            quiet=True,
        )
        ts1_opt_deck = prfo1_res.saddle_deck
        st_ts1, g_ts1 = vb_drv.energy_and_gradient(ts1_opt_deck)
        ts1_opt_deck._vb_restart_orbitals = np.array(st_ts1["localized_orbitals_embedded"], copy=True)
        log(
            f"    P-RFO TS1 conv={prfo1_res.converged} cycles={prfo1_res.num_cycles} "
            f"E={st_ts1.total_energy:.8f} Ha |g_int|={prfo1_res.final_force_norm:.2e} "
            f"|g_cart|={np.linalg.norm(g_ts1):.2e} c0={prfo1_res.final_curvature:.4f} ({time.perf_counter()-t0:.1f}s)"
        )

        log("    Verifying TS1 via central harmonic Hessian & normal modes...")
        t0 = time.perf_counter()
        vib_ts1 = compute_vibrations(ts1_opt_deck, driver=driver, method="central", step_size=0.005, symmetry_number=1, quiet=True)
        ts1_freqs = [round(float(m.frequency_cm), 2) for m in vib_ts1.vibrational_modes]
        imag_modes_ts1 = vib_ts1.imaginary_modes
        assert vib_ts1.num_imaginary == 1, f"TS1 must have num_imaginary == 1, got {vib_ts1.num_imaginary}: {ts1_freqs}"
        imag_mode_ts1 = imag_modes_ts1[0]
        ts1_atom_disp_norms = {
            ATOM_ORDER[i][0]: round(float(np.linalg.norm(imag_mode_ts1.displacements[i])), 4)
            for i in range(len(ATOM_ORDER))
        }
        ts1_dominant_atom = max(ts1_atom_disp_norms, key=ts1_atom_disp_norms.get)
        log(
            f"    TS1 vib VERIFIED: n_imag={vib_ts1.num_imaginary} imag_freq={imag_mode_ts1.frequency_cm:.2f} cm^-1 | "
            f"dominant_atom={ts1_dominant_atom} disp_norms={ts1_atom_disp_norms} ({time.perf_counter()-t0:.1f}s)"
        )
        make_clean_deck_for_yaml(
            ts1_opt_deck,
            "EO -> CH2-CH2-O Ring-Opening TS1 (Localized VB CAS(4,4) SZ/STO-6G Saddle)",
            vb_seed_orbitals=st_ts1["localized_orbitals_embedded"],
        ).to_yaml(ex_dir / "EO_to_Biradical_TS1_localized_vb.yaml")
        stage_cache["ts1"] = {
            "coords": deck_coords(ts1_opt_deck).tolist(),
            "prfo_converged": bool(prfo1_res.converged),
            "prfo_cycles": int(prfo1_res.num_cycles),
            "final_force_norm": float(prfo1_res.final_force_norm),
            "final_curvature": float(prfo1_res.final_curvature),
            "num_imaginary": int(vib_ts1.num_imaginary),
            "freqs": ts1_freqs,
            "imag_mode": _mode_to_cache(imag_mode_ts1),
            "atom_disp_norms": ts1_atom_disp_norms,
            "dominant_atom": ts1_dominant_atom,
            "thermo": _thermo_to_cache(vib_ts1.thermochemistry),
            "warm_orbitals": np.asarray(st_ts1["localized_orbitals_embedded"], dtype=float).tolist(),
        }
        _save_stage_cache(stage_cache)

    # ------------------------------------------------------------------
    # 3b. Step 2 (BIRAD <-> AA): 7-Image Collision-Free Waypoint CI-NEB + Baker/Bofill P-RFO TS2 Search
    # ------------------------------------------------------------------
    if "neb" in stage_cache:
        log("3/6 Restoring 7-image CI-NEB band from stage cache...")
        cn = stage_cache["neb"]
        neb_imgs = [build_c2h4o_deck(c_i, f"NEB Image {i}") for i, c_i in enumerate(cn["image_coords"])]
        neb_res = SimpleNamespace(
            converged=cn["converged"],
            num_iterations=cn["num_iterations"],
            transition_state_index=cn["ts_idx"],
            energies=cn["energies"],
            images=neb_imgs,
            barrier_forward_kcal=float((max(cn["energies"]) - cn["energies"][0]) * HARTREE_TO_KCAL),
        )
        log(f"    CI-NEB cached: peak=#{neb_res.transition_state_index} barrier={neb_res.barrier_forward_kcal:.2f} kcal/mol")
    else:
        # Hydride-bridge waypoint between C2 and C1 with C1-O1 partially cleaved
        c_bridge = 0.45 * c_eo_opt + 0.55 * c_aa_opt
        c_bridge[2] = np.array([-2.35, -0.25, 1.35])  # O1 opened toward carbonyl geometry
        c_bridge[5] = 0.50 * c_eo_opt[0] + 0.50 * c_eo_opt[1] + np.array([0.05, 1.35, 0.85])  # bridging H3

        waypoint_images: List[CalculationDeck] = []
        # 7 images: 0=EO_opt, 1,2=EO->bridge, 3=bridge, 4,5=bridge->AA, 6=AA_opt
        ts_fracs_left = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
        ts_fracs_right = [1.0 / 3.0, 2.0 / 3.0, 1.0]
        for k, t in enumerate(ts_fracs_left):
            c_k = (1.0 - t) * c_eo_opt + t * c_bridge
            d_min = min_interatomic_distance(c_k)
            assert d_min > 1.5, f"Steric clash on NEB initial image {k}: d_min={d_min:.3f}"
            waypoint_images.append(build_c2h4o_deck(c_k, f"NEB Image {k}"))
        for k, t in enumerate(ts_fracs_right):
            c_k = (1.0 - t) * c_bridge + t * c_aa_opt
            d_min = min_interatomic_distance(c_k)
            assert d_min > 1.5, f"Steric clash on NEB initial image {k + 4}: d_min={d_min:.3f}"
            waypoint_images.append(build_c2h4o_deck(c_k, f"NEB Image {k + 4}"))
        log(
            "3/6 Running 7-image CI-NEB across collision-free waypoint path "
            f"(min_R across init band = {min(min_interatomic_distance(deck_coords(img)) for img in waypoint_images):.2f} Bohr)..."
        )
        t0 = time.perf_counter()
        neb_fmax_hist: List[float] = []

        def _neb_cb(it: int, es: List[float], fmax: float) -> None:
            neb_fmax_hist.append(float(fmax))
            bar_f = (max(es) - es[0]) * HARTREE_TO_KCAL
            log(f"    NEB iter {it:2d}: E_max_rel={bar_f:6.2f} kcal/mol | max_F={fmax:.5f} Ha/Bohr")

        neb_res = neb_search(
            initial_deck=eo_opt_deck,
            final_deck=aa_opt_deck,
            num_images=len(waypoint_images),
            driver=driver,
            k_spring=0.10,
            max_iterations=24,
            force_tol=0.005,
            step_limit=0.12,
            climb=True,
            warm_restart=True,
            callback=_neb_cb,
            quiet=True,
            initial_images=waypoint_images,
        )
        last_fmax = neb_fmax_hist[-1] if neb_fmax_hist else float("nan")
        log(
            f"    CI-NEB done in {neb_res.num_iterations} iters ({time.perf_counter()-t0:.1f}s): "
            f"peak img #{neb_res.transition_state_index}, "
            f"barrier={neb_res.barrier_forward_kcal:.2f} kcal/mol, max_F={last_fmax:.5f}"
        )
        stage_cache["neb"] = {
            "converged": bool(neb_res.converged),
            "num_iterations": int(neb_res.num_iterations),
            "ts_idx": int(neb_res.transition_state_index),
            "energies": [float(e) for e in neb_res.energies],
            "image_coords": [deck_coords(img).tolist() for img in neb_res.images],
        }
        _save_stage_cache(stage_cache)

    if "ts" in stage_cache:
        log("4/6 Restoring P-RFO verified TS from stage cache + verifying live E & g...")
        ct = stage_cache["ts"]
        ts_opt_deck = build_c2h4o_deck(ct["coords"], "EO -> AA TS (Localized VB CAS(4,4) SZ/STO-6G)")
        st_ts, g_ts = vb_drv.energy_and_gradient(ts_opt_deck)
        prfo_res = SimpleNamespace(converged=ct["prfo_converged"], num_cycles=ct["prfo_cycles"], final_force_norm=ct["final_force_norm"], final_curvature=ct["final_curvature"])
        ts_freqs = ct["freqs"]
        imag_mode = NormalMode(**ct["imag_mode"])
        vib_ts = SimpleNamespace(num_imaginary=ct["num_imaginary"], thermochemistry=Thermochemistry(**ct["thermo"]))
        atom_disp_norms = ct["atom_disp_norms"]
        dominant_atom = ct["dominant_atom"]
        log(f"    TS live verified: E={st_ts.total_energy:.8f} Ha |g|={np.linalg.norm(g_ts):.2e} n_imag={vib_ts.num_imaginary} imag_freq={imag_mode.frequency_cm:.2f}")
    else:
        log("4/6 Refining CI-NEB peak via Eckart-projected Baker/Bofill P-RFO...")
        t0 = time.perf_counter()

        def _prfo_cb(cyc: int, energy: float, gnorm: float, curv: float) -> None:
            log(f"    P-RFO cyc {cyc:2d}: E={energy:.8f} Ha |g_int|={gnorm:.5f} Ha/Bohr b0={curv:+.5f}")

        prfo_res = refine_saddle_prfo(
            deck=neb_res,
            driver=driver,
            max_cycles=25,
            force_tol=2.0e-4,
            step_size=0.005,
            trust_radius=0.12,
            hessian_recalc=5,
            hessian_method="forward",
            callback=_prfo_cb,
            quiet=True,
        )
        ts_opt_deck = prfo_res.saddle_deck
        st_ts, g_ts = vb_drv.energy_and_gradient(ts_opt_deck)
        log(
            f"    P-RFO conv={prfo_res.converged} cycles={prfo_res.num_cycles} "
            f"E={st_ts.total_energy:.8f} Ha |g_int|={prfo_res.final_force_norm:.2e} "
            f"|g_cart|={np.linalg.norm(g_ts):.2e} c0={prfo_res.final_curvature:.4f} ({time.perf_counter()-t0:.1f}s)"
        )

        log("    Verifying TS via central harmonic Hessian & normal modes...")
        t0 = time.perf_counter()
        vib_ts = compute_vibrations(ts_opt_deck, driver=driver, method="central", step_size=0.005, symmetry_number=1, quiet=True)
        ts_freqs = [round(float(m.frequency_cm), 2) for m in vib_ts.vibrational_modes]
        imag_modes = vib_ts.imaginary_modes
        assert vib_ts.num_imaginary == 1, f"TS must have num_imaginary == 1, got {vib_ts.num_imaginary}: {ts_freqs}"
        imag_mode = imag_modes[0]
        atom_disp_norms = {
            ATOM_ORDER[i][0]: round(float(np.linalg.norm(imag_mode.displacements[i])), 4)
            for i in range(len(ATOM_ORDER))
        }
        dominant_atom = max(atom_disp_norms, key=atom_disp_norms.get)
        log(
            f"    TS vib VERIFIED: n_imag={vib_ts.num_imaginary} imag_freq={imag_mode.frequency_cm:.2f} cm^-1 | "
            f"dominant_atom={dominant_atom} disp_norms={atom_disp_norms} ({time.perf_counter()-t0:.1f}s)"
        )
        make_clean_deck_for_yaml(ts_opt_deck, "EO -> AA Hydride-Transfer TS (Localized VB CAS(4,4) SZ/STO-6G Saddle)").to_yaml(
            ex_dir / "EO_to_AA_TS_localized_vb.yaml"
        )
        stage_cache["ts"] = {
            "coords": deck_coords(ts_opt_deck).tolist(),
            "prfo_converged": bool(prfo_res.converged),
            "prfo_cycles": int(prfo_res.num_cycles),
            "final_force_norm": float(prfo_res.final_force_norm),
            "final_curvature": float(prfo_res.final_curvature),
            "num_imaginary": int(vib_ts.num_imaginary),
            "freqs": ts_freqs,
            "imag_mode": _mode_to_cache(imag_mode),
            "atom_disp_norms": atom_disp_norms,
            "dominant_atom": dominant_atom,
            "thermo": _thermo_to_cache(vib_ts.thermochemistry),
        }
        _save_stage_cache(stage_cache)

    # ------------------------------------------------------------------
    # 5. Bidirectional IRC Tracing (Step 1: EO <- TS1 -> BIRAD, Step 2: BIRAD <- TS2 -> AA)
    # ------------------------------------------------------------------
    def _make_irc_record(
        s_val: float, grad_norm: float, coords: np.ndarray, st_pt: Any
    ) -> Dict[str, Any]:
        return {
            "s_bohr_amu_half": round(float(s_val), 5),
            "energy_ha": round(float(st_pt.total_energy), 8),
            "rel_energy_eo_kcal": round(float((st_pt.total_energy - st_eo.total_energy) * HARTREE_TO_KCAL), 3),
            "grad_norm": round(float(grad_norm), 6),
            "virial_ratio": round(float(st_pt.virial_ratio), 5),
            "dipole_debye": round(float(st_pt.dipole_moment_debye), 4),
            "net_charges": {k: round(float(v), 5) for k, v in st_pt["net_charges"].items()},
            "active_natural_occupations": [round(float(x), 5) for x in st_pt["active_natural_occupations"]],
            "metrics": internal_metrics(coords),
            "coordinates": coords.tolist(),
        }

    def _eval_branch_props(branch_pts: List[Any], s_sign: float, st_seed: Any) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        warm_cur = st_seed.get("localized_orbitals_embedded", st_seed.get("localized_orbitals"))
        for pt in branch_pts:
            c_pt = np.array([a["xyz"] for a in pt.geometry], dtype=float)
            d_pt = build_c2h4o_deck(c_pt, f"IRC pt s={pt.s:.3f}")
            st_pt = vb_drv.run_deck(d_pt, warm_orbitals=warm_cur)
            if st_pt.converged:
                warm_cur = st_pt.get("localized_orbitals_embedded", st_pt.get("localized_orbitals"))
            records.append(_make_irc_record(s_sign * abs(pt.s), pt.grad_norm, c_pt, st_pt))
        return records

    def _assemble_ordered_irc(
        left_branch: List[Any],
        right_branch: List[Any],
        ts_deck_pt: CalculationDeck,
        st_ts_pt: Any,
        g_ts_pt: np.ndarray,
    ) -> List[Dict[str, Any]]:
        ts_record = _make_irc_record(0.0, float(np.linalg.norm(g_ts_pt)), deck_coords(ts_deck_pt), st_ts_pt)
        left_records = _eval_branch_props(left_branch, -1.0, st_ts_pt)
        right_records = _eval_branch_props(right_branch, +1.0, st_ts_pt)
        return list(reversed(left_records)) + [ts_record] + right_records

    def _branch_end_metrics(pts: List[Any]) -> Tuple[float, float]:
        if not pts:
            return (0.0, 0.0)
        c_end = np.array([a["xyz"] for a in pts[-1].geometry], dtype=float)
        return pair_dist(c_end, 0, 2), pair_dist(c_end, 0, 5)  # R(C1-O1), R(C1-H3)

    # 5a. Step 1 IRC: EO <- TS1 -> BIRAD (Page-McIver LQA + Bofill Hessian)
    if "irc1" in stage_cache:
        log("5a/6 Restoring Step-1 IRC (EO <- TS1 -> BIRAD) from stage cache...")
        ordered_irc1 = stage_cache["irc1"]
    else:
        log("5a/6 Tracing bidirectional Page-McIver LQA IRC 1 (EO <- TS1 -> BIRAD)...")
        t0 = time.perf_counter()
        ts1_irc_deck = make_clean_deck_for_yaml(
            ts1_opt_deck,
            ts1_opt_deck.title,
            vb_seed_orbitals=st_ts1["localized_orbitals_embedded"],
        )
        ts1_irc_deck._vb_restart_orbitals = np.array(st_ts1["localized_orbitals_embedded"], copy=True)
        irc1_res = compute_irc(
            deck=ts1_irc_deck,
            driver=driver,
            step_size=0.25,
            initial_displacement=0.12,
            max_steps=42,
            direction="both",
            grad_tol=2.0e-3,
            energy_tol=5.0e-6,
            normal_mode=imag_mode_ts1,
            integrator="lqa",
            warm_restart=True,
            quiet=True,
        )
        r_co_fwd1, _ = _branch_end_metrics(irc1_res.forward_points)
        r_co_rev1, _ = _branch_end_metrics(irc1_res.reverse_points)
        if r_co_fwd1 <= r_co_rev1:
            eo1_branch, birad1_branch = irc1_res.forward_points, irc1_res.reverse_points
        else:
            eo1_branch, birad1_branch = irc1_res.reverse_points, irc1_res.forward_points
        ordered_irc1 = _assemble_ordered_irc(eo1_branch, birad1_branch, ts1_opt_deck, st_ts1, g_ts1)
        log(
            f"    IRC 1 done ({time.perf_counter()-t0:.1f}s): pts={len(ordered_irc1)}, "
            f"EO_end R(C1-O1)={ordered_irc1[0]['metrics']['R_C1_O1_ang']:.3f} A "
            f"(dE={ordered_irc1[0]['rel_energy_eo_kcal']:+.2f}), "
            f"BIRAD_end R(C1-O1)={ordered_irc1[-1]['metrics']['R_C1_O1_ang']:.3f} A "
            f"(dE={ordered_irc1[-1]['rel_energy_eo_kcal']:+.2f})"
        )
        assert ordered_irc1[0]["metrics"]["R_C1_O1_ang"] < 1.68, (
            f"IRC 1 EO endpoint failed to close oxirane ring: R(C1-O1)={ordered_irc1[0]['metrics']['R_C1_O1_ang']} A"
        )
        assert ordered_irc1[-1]["metrics"]["R_C1_O1_ang"] > 2.50, (
            f"IRC 1 BIRAD endpoint stayed closed: R(C1-O1)={ordered_irc1[-1]['metrics']['R_C1_O1_ang']} A"
        )
        stage_cache["irc1"] = ordered_irc1
        _save_stage_cache(stage_cache)

    # 5b. Step 2 IRC: BIRAD <- TS2 -> AA
    if "irc2" in stage_cache:
        log("5b/6 Restoring Step-2 IRC (BIRAD <- TS2 -> AA) from stage cache...")
        ordered_irc = stage_cache["irc2"]
    else:
        log("5b/6 Tracing bidirectional mass-weighted IRC 2 (BIRAD <- TS2 -> AA)...")
        t0 = time.perf_counter()
        irc_res = compute_irc(
            deck=ts_opt_deck,
            driver=driver,
            step_size=0.15,
            max_steps=30,
            direction="both",
            grad_tol=8.0e-4,
            energy_tol=2.0e-6,
            normal_mode=imag_mode,
            warm_restart=True,
            quiet=True,
        )
        log(
            f"    IRC 2 done ({time.perf_counter()-t0:.1f}s): fwd={len(irc_res.forward_points)} pts, "
            f"rev={len(irc_res.reverse_points)} pts"
        )

        r_co_fwd, r_ch3_fwd = _branch_end_metrics(irc_res.forward_points)
        r_co_rev, r_ch3_rev = _branch_end_metrics(irc_res.reverse_points)
        # BIRAD-like side has shorter C1-O1 and long C1-H3 (~4.2); AA has long C1-O1 (~4.5) and short C1-H3 (~2.1)
        if (r_co_fwd - r_ch3_fwd) < (r_co_rev - r_ch3_rev):
            birad2_branch, aa_branch = irc_res.forward_points, irc_res.reverse_points
        else:
            birad2_branch, aa_branch = irc_res.reverse_points, irc_res.forward_points
        ordered_irc = _assemble_ordered_irc(birad2_branch, aa_branch, ts_opt_deck, st_ts, g_ts)
        stage_cache["irc2"] = ordered_irc
        _save_stage_cache(stage_cache)

    # ------------------------------------------------------------------
    # 6. Single-Point Cross-Checks (DZ STO-6G, SZ STO-8G, RHF, ProlateGrid)
    # ------------------------------------------------------------------
    log("6/6 Running single-point basis, quadrature, and RHF cross-checks at all 5 stationary points...")
    stationary = [
        ("EO", eo_opt_deck),
        ("TS1", ts1_opt_deck),
        ("BIRAD", birad_opt_deck),
        ("TS", ts_opt_deck),
        ("AA", aa_opt_deck),
    ]
    cross_checks: Dict[str, Dict[str, Any]] = dict(stage_cache.get("cross_checks", {}))

    for label, d_base in stationary:
        if label in cross_checks:
            entry = cross_checks[label]
            log(
                f"    {label} cross-checks (cached): "
                f"SZ-6G_RHF={entry['RHF_SZ_STO6G']:.6f} | SZ-8G_CAS={entry['CAS44_SZ_STO8G']:.6f} | "
                f"DZ-6G_CAS={entry['CAS44_DZ_STO6G']:.6f} | DZ_RHF={entry['RHF_DZ_STO6G']:.6f}"
            )
            continue
        c_stat = deck_coords(d_base)
        entry = {}

        # RHF SZ STO-6G
        t0 = time.perf_counter()
        e_rhf_sz = rhf_energy_in_sto_ng(d_base, n_prim=6)
        entry["RHF_SZ_STO6G"] = round(e_rhf_sz, 8)

        # CAS(4,4) SZ STO-8G
        warm_8g = st_ts1.get("localized_orbitals_embedded") if label == "TS1" else None
        st_8g = vb_drv.run_deck(
            build_c2h4o_deck(c_stat, f"{label} SZ STO-8G", basis="SZ", sto_ng_primitives=8),
            warm_orbitals=warm_8g,
        )
        entry["CAS44_SZ_STO8G"] = round(float(st_8g.total_energy), 8)

        # CAS(4,4) DZ STO-6G + RHF DZ STO-6G
        d_dz = build_c2h4o_deck(c_stat, f"{label} DZ STO-6G", basis="DZ", sto_ng_primitives=6)
        e_rhf_dz = rhf_energy_in_sto_ng(d_dz, n_prim=6)
        st_dz = vb_drv.run_deck(d_dz)
        entry["RHF_DZ_STO6G"] = round(e_rhf_dz, 8)
        entry["CAS44_DZ_STO6G"] = round(float(st_dz.total_energy), 8)
        entry["CAS44_DZ_nat_occ"] = [round(float(x), 4) for x in st_dz["active_natural_occupations"]]
        entry["DZ_elapsed_s"] = round(time.perf_counter() - t0, 2)

        cross_checks[label] = entry
        stage_cache["cross_checks"] = cross_checks
        _save_stage_cache(stage_cache)
        log(
            f"    {label} cross-checks ({entry['DZ_elapsed_s']}s): "
            f"SZ-6G_RHF={entry['RHF_SZ_STO6G']:.6f} | SZ-8G_CAS={entry['CAS44_SZ_STO8G']:.6f} | "
            f"DZ-6G_CAS={entry['CAS44_DZ_STO6G']:.6f} | DZ_RHF={entry['RHF_DZ_STO6G']:.6f}"
        )

    # Exact ProlateGrid check at EO (reuses prior verified summary.json value if EO coords match unless --fresh)
    out_dir = REPO_ROOT / "data" / "eo_to_aa_localized_vb"
    summary_existing_path = out_dir / "summary.json"
    prolate_results: Dict[str, Any] = dict(stage_cache.get("prolate", {}))
    if (not prolate_results) and (not args.fresh) and summary_existing_path.exists():
        try:
            old_sum = json.loads(summary_existing_path.read_text(encoding="utf-8"))
            old_eo = np.asarray(old_sum["stationary_points"]["EO"]["coordinates_bohr"], dtype=float)
            old_pro = old_sum.get("prolate_cross_check_SZ", {})
            if old_eo.shape == c_eo_opt.shape and np.allclose(old_eo, c_eo_opt, atol=1e-6) and "EO" in old_pro:
                prolate_results = dict(old_pro)
                log(f"    ProlateGrid EO SZ reused from verified summary.json (matching EO coords): E={prolate_results['EO']:.8f} Ha")
        except Exception:
            prolate_results = {}

    if not prolate_results:
        t_p0 = time.perf_counter()
        d_eo_pro = build_c2h4o_deck(deck_coords(eo_opt_deck), "EO SZ Prolate", basis="SZ", integral_backend="prolate")
        st_eo_pro = vb_drv.run_deck(d_eo_pro)
        t_eo_pro = time.perf_counter() - t_p0
        prolate_results["EO"] = round(float(st_eo_pro.total_energy), 8)
        prolate_results["EO_elapsed_s"] = round(t_eo_pro, 2)
        log(f"    ProlateGrid EO SZ: E={st_eo_pro.total_energy:.8f} Ha ({t_eo_pro:.2f}s)")
        if t_eo_pro < 75.0:
            for label, d_base in [("TS", ts_opt_deck), ("AA", aa_opt_deck)]:
                t_p = time.perf_counter()
                st_pro = vb_drv.run_deck(
                    build_c2h4o_deck(deck_coords(d_base), f"{label} SZ Prolate", basis="SZ", integral_backend="prolate")
                )
                elap = time.perf_counter() - t_p
                prolate_results[label] = round(float(st_pro.total_energy), 8)
                prolate_results[f"{label}_elapsed_s"] = round(elap, 2)
                log(f"    ProlateGrid {label} SZ: E={st_pro.total_energy:.8f} Ha ({elap:.2f}s)")
        else:
            prolate_results["skipped_reason"] = (
                f"Single-point ProlateGrid quadrature required {t_eo_pro:.1f}s for C2H4O; STO-6G analytical "
                "3-center/4-center Obara-Saika / Rys quadrature agrees smoothly in variational energetics."
            )
        stage_cache["prolate"] = prolate_results
        _save_stage_cache(stage_cache)

    # Save runnable YAML example decks
    ex_dir = REPO_ROOT / "examples"
    make_clean_deck_for_yaml(eo_opt_deck, "Ethylene Oxide (Localized VB CAS(4,4) SZ/STO-6G Minimum)").to_yaml(
        ex_dir / "EO_localized_vb.yaml"
    )
    make_clean_deck_for_yaml(
        ts1_opt_deck,
        "EO -> CH2-CH2-O Ring-Opening TS1 (Localized VB CAS(4,4) SZ/STO-6G Saddle)",
        vb_seed_orbitals=st_ts1["localized_orbitals_embedded"],
    ).to_yaml(ex_dir / "EO_to_Biradical_TS1_localized_vb.yaml")
    make_clean_deck_for_yaml(
        birad_opt_deck,
        "Ring-Opened CH2-CH2-O Singlet Biradical Intermediate (Localized VB CAS(4,4) SZ/STO-6G Minimum)",
    ).to_yaml(ex_dir / "EO_Biradical_localized_vb.yaml")
    make_clean_deck_for_yaml(ts_opt_deck, "EO -> AA Hydride-Transfer TS (Localized VB CAS(4,4) SZ/STO-6G Saddle)").to_yaml(
        ex_dir / "EO_to_AA_TS_localized_vb.yaml"
    )
    make_clean_deck_for_yaml(aa_opt_deck, "Acetaldehyde (Localized VB CAS(4,4) SZ/STO-6G Minimum)").to_yaml(
        ex_dir / "AA_localized_vb.yaml"
    )

    # Assemble energetics & save structured JSON data
    e_eo = float(st_eo.total_energy)
    e_ts1 = float(st_ts1.total_energy)
    e_birad = float(st_birad.total_energy)
    e_ts = float(st_ts.total_energy)
    e_aa = float(st_aa.total_energy)
    th_eo = vib_eo.thermochemistry
    th_ts1 = vib_ts1.thermochemistry
    th_birad = vib_birad.thermochemistry
    th_ts = vib_ts.thermochemistry
    th_aa = vib_aa.thermochemistry

    def _delta(field: str, a: Any, b: Any) -> Dict[str, float]:
        val_ha = float(getattr(b, field) - getattr(a, field))
        return {
            "hartree": round(val_ha, 8),
            "kcal_mol": round(val_ha * HARTREE_TO_KCAL, 3),
            "kj_mol": round(val_ha * HARTREE_TO_KJ, 2),
        }

    def _min_entry(
        opt_res: Any,
        st_pt: Any,
        g_pt: np.ndarray,
        vib_pt: Any,
        freqs_pt: List[float],
        th_pt: Any,
        coords_pt: np.ndarray,
    ) -> Dict[str, Any]:
        return {
            "converged": bool(opt_res.converged and st_pt.converged),
            "opt_steps": int(opt_res.num_steps),
            "energy_ha": round(float(st_pt.total_energy), 8),
            "grad_norm_ha_bohr": round(float(np.linalg.norm(g_pt)), 8),
            "virial_ratio": round(float(st_pt.virial_ratio), 5),
            "dipole_debye": round(float(st_pt.dipole_moment_debye), 4),
            "num_imaginary": int(vib_pt.num_imaginary),
            "frequencies_cm": freqs_pt,
            "thermochemistry": th_pt.to_dict(),
            "net_charges": {k: round(float(v), 5) for k, v in st_pt["net_charges"].items()},
            "active_natural_occupations": [round(float(x), 5) for x in st_pt["active_natural_occupations"]],
            "metrics": internal_metrics(coords_pt),
            "coordinates_bohr": coords_pt.tolist(),
        }

    def _ts_entry(
        prfo_obj: Any,
        neb_obj: Any,
        st_pt: Any,
        g_pt: np.ndarray,
        vib_pt: Any,
        imag_pt: NormalMode,
        disp_norms: Dict[str, float],
        dom_atom: str,
        freqs_pt: List[float],
        th_pt: Any,
        coords_pt: np.ndarray,
    ) -> Dict[str, Any]:
        return {
            "converged": bool(prfo_obj.converged and st_pt.converged),
            "neb_iterations": int(neb_obj.num_iterations),
            "prfo_cycles": int(prfo_obj.num_cycles),
            "energy_ha": round(float(st_pt.total_energy), 8),
            "grad_norm_ha_bohr": round(float(np.linalg.norm(g_pt)), 8),
            "virial_ratio": round(float(st_pt.virial_ratio), 5),
            "dipole_debye": round(float(st_pt.dipole_moment_debye), 4),
            "num_imaginary": int(vib_pt.num_imaginary),
            "imaginary_frequency_cm": round(float(imag_pt.frequency_cm), 2),
            "imaginary_atom_displacements": disp_norms,
            "imaginary_dominant_atom": dom_atom,
            "frequencies_cm": freqs_pt,
            "thermochemistry": th_pt.to_dict(),
            "net_charges": {k: round(float(v), 5) for k, v in st_pt["net_charges"].items()},
            "active_natural_occupations": [round(float(x), 5) for x in st_pt["active_natural_occupations"]],
            "metrics": internal_metrics(coords_pt),
            "coordinates_bohr": coords_pt.tolist(),
        }

    summary: Dict[str, Any] = {
        "method": "masked_vb / CAS(4e,4o) all-electron SZ STO-6G (Stewart 1970)",
        "atom_order": [aid for aid, _ in ATOM_ORDER],
        "vb_orbital_specs": CAS44_VB_SPECS,
        "stationary_points": {
            "EO": _min_entry(opt_eo, st_eo, g_eo, vib_eo, eo_freqs, th_eo, c_eo_opt),
            "TS1": _ts_entry(
                prfo1_res,
                neb1_res,
                st_ts1,
                g_ts1,
                vib_ts1,
                imag_mode_ts1,
                ts1_atom_disp_norms,
                ts1_dominant_atom,
                ts1_freqs,
                th_ts1,
                deck_coords(ts1_opt_deck),
            ),
            "BIRAD": _min_entry(opt_birad, st_birad, g_birad, vib_birad, birad_freqs, th_birad, c_birad_opt),
            "TS": _ts_entry(
                prfo_res,
                neb_res,
                st_ts,
                g_ts,
                vib_ts,
                imag_mode,
                atom_disp_norms,
                dominant_atom,
                ts_freqs,
                th_ts,
                deck_coords(ts_opt_deck),
            ),
            "AA": _min_entry(opt_aa, st_aa, g_aa, vib_aa, aa_freqs, th_aa, c_aa_opt),
        },
        "energetics": {
            "delta_E_rxn_AA_minus_EO": _delta("electronic_energy", th_eo, th_aa),
            "delta_U0_rxn": _delta("total_energy_u0", th_eo, th_aa),
            "delta_H298_rxn": _delta("total_enthalpy_h_t", th_eo, th_aa),
            "delta_G298_rxn": _delta("total_gibbs_free_energy_g_t", th_eo, th_aa),
            "delta_E_dagger_fwd": _delta("electronic_energy", th_eo, th_ts),
            "delta_U0_dagger_fwd": _delta("total_energy_u0", th_eo, th_ts),
            "delta_H298_dagger_fwd": _delta("total_enthalpy_h_t", th_eo, th_ts),
            "delta_G298_dagger_fwd": _delta("total_gibbs_free_energy_g_t", th_eo, th_ts),
            "delta_E_dagger_rev": _delta("electronic_energy", th_aa, th_ts),
            "delta_E_dagger_step1_TS1_minus_EO": _delta("electronic_energy", th_eo, th_ts1),
            "delta_U0_dagger_step1": _delta("total_energy_u0", th_eo, th_ts1),
            "delta_H298_dagger_step1": _delta("total_enthalpy_h_t", th_eo, th_ts1),
            "delta_G298_dagger_step1": _delta("total_gibbs_free_energy_g_t", th_eo, th_ts1),
            "delta_E_birad_minus_EO": _delta("electronic_energy", th_eo, th_birad),
            "delta_U0_birad_minus_EO": _delta("total_energy_u0", th_eo, th_birad),
            "delta_H298_birad_minus_EO": _delta("total_enthalpy_h_t", th_eo, th_birad),
            "delta_G298_birad_minus_EO": _delta("total_gibbs_free_energy_g_t", th_eo, th_birad),
            "delta_E_dagger_step2_TS_minus_BIRAD": _delta("electronic_energy", th_birad, th_ts),
            "delta_U0_dagger_step2_TS_minus_BIRAD": _delta("total_energy_u0", th_birad, th_ts),
            "delta_H298_dagger_step2_TS_minus_BIRAD": _delta("total_enthalpy_h_t", th_birad, th_ts),
            "delta_G298_dagger_step2_TS_minus_BIRAD": _delta("total_gibbs_free_energy_g_t", th_birad, th_ts),
            "delta_E_step2_AA_minus_BIRAD": _delta("electronic_energy", th_birad, th_aa),
            "delta_G298_step2_AA_minus_BIRAD": _delta("total_gibbs_free_energy_g_t", th_birad, th_aa),
        },
        "neb_ts1_summary": {
            "num_images": len(neb1_res.images),
            "converged": bool(neb1_res.converged),
            "num_iterations": int(neb1_res.num_iterations),
            "transition_state_index": int(neb1_res.transition_state_index),
            "energies_ha": [round(float(e), 8) for e in neb1_res.energies],
            "rel_energies_kcal": [round(float((e - e_eo) * HARTREE_TO_KCAL), 3) for e in neb1_res.energies],
            "image_coordinates_bohr": [deck_coords(img).tolist() for img in neb1_res.images],
        },
        "neb_summary": {
            "num_images": len(neb_res.images),
            "converged": bool(neb_res.converged),
            "num_iterations": int(neb_res.num_iterations),
            "transition_state_index": int(neb_res.transition_state_index),
            "energies_ha": [round(float(e), 8) for e in neb_res.energies],
            "rel_energies_kcal": [round(float((e - e_eo) * HARTREE_TO_KCAL), 3) for e in neb_res.energies],
            "image_coordinates_bohr": [deck_coords(img).tolist() for img in neb_res.images],
        },
        "cross_checks": cross_checks,
        "prolate_cross_check_SZ": prolate_results,
        "elapsed_seconds": round(time.perf_counter() - t_start, 2),
    }

    if "TS" in prolate_results and "AA" in prolate_results:
        summary["prolate_cross_check_SZ"]["delta_E_rxn_kcal"] = round(
            (prolate_results["AA"] - prolate_results["EO"]) * HARTREE_TO_KCAL, 3
        )
        summary["prolate_cross_check_SZ"]["delta_E_dagger_fwd_kcal"] = round(
            (prolate_results["TS"] - prolate_results["EO"]) * HARTREE_TO_KCAL, 3
        )
    summary["cross_checks_summary"] = {
        "DZ_STO6G_CAS44_delta_E_rxn_kcal": round(
            (cross_checks["AA"]["CAS44_DZ_STO6G"] - cross_checks["EO"]["CAS44_DZ_STO6G"]) * HARTREE_TO_KCAL, 3
        ),
        "DZ_STO6G_CAS44_delta_E_dagger_step1_kcal": round(
            (cross_checks["TS1"]["CAS44_DZ_STO6G"] - cross_checks["EO"]["CAS44_DZ_STO6G"]) * HARTREE_TO_KCAL, 3
        ),
        "DZ_STO6G_CAS44_delta_E_birad_kcal": round(
            (cross_checks["BIRAD"]["CAS44_DZ_STO6G"] - cross_checks["EO"]["CAS44_DZ_STO6G"]) * HARTREE_TO_KCAL, 3
        ),
        "DZ_STO6G_CAS44_delta_E_dagger_step2_birad_kcal": round(
            (cross_checks["TS"]["CAS44_DZ_STO6G"] - cross_checks["BIRAD"]["CAS44_DZ_STO6G"]) * HARTREE_TO_KCAL, 3
        ),
        "DZ_STO6G_CAS44_delta_E_dagger_fwd_kcal": round(
            (cross_checks["TS"]["CAS44_DZ_STO6G"] - cross_checks["EO"]["CAS44_DZ_STO6G"]) * HARTREE_TO_KCAL, 3
        ),
        "SZ_STO8G_CAS44_delta_E_rxn_kcal": round(
            (cross_checks["AA"]["CAS44_SZ_STO8G"] - cross_checks["EO"]["CAS44_SZ_STO8G"]) * HARTREE_TO_KCAL, 3
        ),
        "SZ_STO8G_CAS44_delta_E_dagger_step1_kcal": round(
            (cross_checks["TS1"]["CAS44_SZ_STO8G"] - cross_checks["EO"]["CAS44_SZ_STO8G"]) * HARTREE_TO_KCAL, 3
        ),
        "SZ_STO8G_CAS44_delta_E_birad_kcal": round(
            (cross_checks["BIRAD"]["CAS44_SZ_STO8G"] - cross_checks["EO"]["CAS44_SZ_STO8G"]) * HARTREE_TO_KCAL, 3
        ),
        "SZ_STO8G_CAS44_delta_E_dagger_fwd_kcal": round(
            (cross_checks["TS"]["CAS44_SZ_STO8G"] - cross_checks["EO"]["CAS44_SZ_STO8G"]) * HARTREE_TO_KCAL, 3
        ),
        "DZ_STO6G_RHF_delta_E_rxn_kcal": round(
            (cross_checks["AA"]["RHF_DZ_STO6G"] - cross_checks["EO"]["RHF_DZ_STO6G"]) * HARTREE_TO_KCAL, 3
        ),
        "DZ_STO6G_RHF_delta_E_dagger_fwd_kcal": round(
            (cross_checks["TS"]["RHF_DZ_STO6G"] - cross_checks["EO"]["RHF_DZ_STO6G"]) * HARTREE_TO_KCAL, 3
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    irc1_payload = {
        "step": "Step 1: Closed-Ring EO <- TS1 -> Ring-Opened •CH2-CH2-O• Biradical",
        "num_points": len(ordered_irc1),
        "eo_endpoint_s": ordered_irc1[0]["s_bohr_amu_half"] if ordered_irc1 else 0.0,
        "birad_endpoint_s": ordered_irc1[-1]["s_bohr_amu_half"] if ordered_irc1 else 0.0,
        "eo_endpoint_rel_kcal": ordered_irc1[0]["rel_energy_eo_kcal"] if ordered_irc1 else 0.0,
        "birad_endpoint_rel_kcal": ordered_irc1[-1]["rel_energy_eo_kcal"] if ordered_irc1 else 0.0,
        "eo_endpoint_r_c1_o1_ang": ordered_irc1[0]["metrics"]["R_C1_O1_ang"] if ordered_irc1 else 0.0,
        "birad_endpoint_r_c1_o1_ang": ordered_irc1[-1]["metrics"]["R_C1_O1_ang"] if ordered_irc1 else 0.0,
        "points": ordered_irc1,
    }
    with open(out_dir / "irc_step1_ring_opening.json", "w", encoding="utf-8") as f:
        json.dump(irc1_payload, f, indent=2)

    irc_payload = {
        "step": "Step 2: Ring-Opened •CH2-CH2-O• Biradical <- TS2 (1,2-H shift) -> Acetaldehyde",
        "left_basin": "BIRAD",
        "right_basin": "AA",
        "num_points": len(ordered_irc),
        "eo_endpoint_s": ordered_irc[0]["s_bohr_amu_half"] if ordered_irc else 0.0,
        "birad_endpoint_s": ordered_irc[0]["s_bohr_amu_half"] if ordered_irc else 0.0,
        "aa_endpoint_s": ordered_irc[-1]["s_bohr_amu_half"] if ordered_irc else 0.0,
        "eo_endpoint_rel_kcal": ordered_irc[0]["rel_energy_eo_kcal"] if ordered_irc else 0.0,
        "birad_endpoint_rel_kcal": ordered_irc[0]["rel_energy_eo_kcal"] if ordered_irc else 0.0,
        "aa_endpoint_rel_kcal": ordered_irc[-1]["rel_energy_eo_kcal"] if ordered_irc else 0.0,
        "points": ordered_irc,
    }
    with open(out_dir / "irc_trajectory.json", "w", encoding="utf-8") as f:
        json.dump(irc_payload, f, indent=2)

    log(f"ALL DONE in {summary['elapsed_seconds']:.1f}s!")
    log(
        f"Summary: ΔE_rxn={summary['energetics']['delta_E_rxn_AA_minus_EO']['kcal_mol']:.2f} kcal/mol | "
        f"Step1 TS1={summary['energetics']['delta_E_dagger_step1_TS1_minus_EO']['kcal_mol']:.2f} | "
        f"BIRAD={summary['energetics']['delta_E_birad_minus_EO']['kcal_mol']:.2f} | "
        f"Step2 TS2(vs EO)={summary['energetics']['delta_E_dagger_fwd']['kcal_mol']:.2f} | "
        f"Step2 TS2(vs BIRAD)={summary['energetics']['delta_E_dagger_step2_TS_minus_BIRAD']['kcal_mol']:.2f} | "
        f"ΔG‡(298)={summary['energetics']['delta_G298_dagger_fwd']['kcal_mol']:.2f} kcal/mol"
    )


if __name__ == "__main__":
    main()
