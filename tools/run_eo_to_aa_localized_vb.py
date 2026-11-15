#!/usr/bin/env python3
"""First-principles Localized VB (CAS(4e,4o) / STO-6G) study of Ethylene Oxide -> Acetaldehyde isomerization.

Executes end-to-end:
  1. Geometry optimization + harmonic vibrational frequencies + RRHO thermochemistry on EO & AA minima.
  2. Collision-free 7-image CI-NEB pathway search followed by Baker/Bofill P-RFO saddle-point refinement
     and vibrational verification (num_imaginary == 1, hydride-transfer normal mode).
  3. Bidirectional IRC trace connecting EO <- TS -> AA with per-step bond lengths, Lowdin charges, and
     CAS(4,4) natural orbital occupations.
  4. Single-point cross-checks at stationary points (DZ STO-6G, SZ STO-8G, RHF, exact ProlateGrid).

Outputs:
  - data/eo_to_aa_localized_vb/summary.json
  - data/eo_to_aa_localized_vb/irc_trajectory.json
  - examples/EO_localized_vb.yaml
  - examples/AA_localized_vb.yaml
  - examples/EO_to_AA_TS_localized_vb.yaml
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
from typing import Any, Dict, List, Sequence, Tuple
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


def make_clean_deck_for_yaml(deck: CalculationDeck, title: str) -> CalculationDeck:
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
    # 3. 7-Image Collision-Free Waypoint CI-NEB + Baker/Bofill P-RFO TS Search
    # ------------------------------------------------------------------
    c_eo_opt = deck_coords(eo_opt_deck)
    c_aa_opt = deck_coords(aa_opt_deck)
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
    # 5. Bidirectional IRC Tracing + Continuous Property Evaluation
    # ------------------------------------------------------------------
    log("5/6 Tracing bidirectional mass-weighted IRC from verified TS...")
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
        f"    IRC done ({time.perf_counter()-t0:.1f}s): fwd={len(irc_res.forward_points)} pts, "
        f"rev={len(irc_res.reverse_points)} pts"
    )

    # Orient branches so reactant EO side is s < 0 and product AA side is s > 0
    def _branch_end_metrics(pts: List[Any]) -> Tuple[float, float]:
        if not pts:
            return (0.0, 0.0)
        c_end = np.array([a["xyz"] for a in pts[-1].geometry], dtype=float)
        return pair_dist(c_end, 0, 2), pair_dist(c_end, 0, 5)  # R(C1-O1), R(C1-H3)

    r_co_fwd, r_ch3_fwd = _branch_end_metrics(irc_res.forward_points)
    r_co_rev, r_ch3_rev = _branch_end_metrics(irc_res.reverse_points)
    # EO has short C1-O1 (~2.7) and long C1-H3 (~4.2); AA has long C1-O1 (~4.5) and short C1-H3 (~2.1)
    if (r_co_fwd - r_ch3_fwd) < (r_co_rev - r_ch3_rev):
        eo_branch = irc_res.forward_points
        aa_branch = irc_res.reverse_points
        sign_eo, sign_aa = -1.0, +1.0
    else:
        eo_branch = irc_res.reverse_points
        aa_branch = irc_res.forward_points
        sign_eo, sign_aa = -1.0, +1.0

    # Walk outward along each branch with sequential warm-start from TS state to extract Lowdin charges & nat occs
    def _eval_branch_props(branch_pts: List[Any], s_sign: float) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        warm_cur = st_ts.get("localized_orbitals_embedded", st_ts.get("localized_orbitals"))
        for pt in branch_pts:
            c_pt = np.array([a["xyz"] for a in pt.geometry], dtype=float)
            d_pt = build_c2h4o_deck(c_pt, f"IRC pt s={pt.s:.3f}")
            st_pt = vb_drv.run_deck(d_pt, warm_orbitals=warm_cur)
            if st_pt.converged:
                warm_cur = st_pt.get("localized_orbitals_embedded", st_pt.get("localized_orbitals"))
            records.append(
                {
                    "s_bohr_amu_half": round(float(s_sign * abs(pt.s)), 5),
                    "energy_ha": round(float(st_pt.total_energy), 8),
                    "rel_energy_eo_kcal": round(float((st_pt.total_energy - st_eo.total_energy) * HARTREE_TO_KCAL), 3),
                    "grad_norm": round(float(pt.grad_norm), 6),
                    "virial_ratio": round(float(st_pt.virial_ratio), 5),
                    "dipole_debye": round(float(st_pt.dipole_moment_debye), 4),
                    "net_charges": {k: round(float(v), 5) for k, v in st_pt["net_charges"].items()},
                    "active_natural_occupations": [round(float(x), 5) for x in st_pt["active_natural_occupations"]],
                    "metrics": internal_metrics(c_pt),
                    "coordinates": c_pt.tolist(),
                }
            )
        return records

    ts_irc_record = {
        "s_bohr_amu_half": 0.0,
        "energy_ha": round(float(st_ts.total_energy), 8),
        "rel_energy_eo_kcal": round(float((st_ts.total_energy - st_eo.total_energy) * HARTREE_TO_KCAL), 3),
        "grad_norm": round(float(np.linalg.norm(g_ts)), 6),
        "virial_ratio": round(float(st_ts.virial_ratio), 5),
        "dipole_debye": round(float(st_ts.dipole_moment_debye), 4),
        "net_charges": {k: round(float(v), 5) for k, v in st_ts["net_charges"].items()},
        "active_natural_occupations": [round(float(x), 5) for x in st_ts["active_natural_occupations"]],
        "metrics": internal_metrics(deck_coords(ts_opt_deck)),
        "coordinates": deck_coords(ts_opt_deck).tolist(),
    }
    eo_records = _eval_branch_props(eo_branch, sign_eo)
    aa_records = _eval_branch_props(aa_branch, sign_aa)
    ordered_irc = list(reversed(eo_records)) + [ts_irc_record] + aa_records

    # ------------------------------------------------------------------
    # 6. Single-Point Cross-Checks (DZ STO-6G, SZ STO-8G, RHF, ProlateGrid)
    # ------------------------------------------------------------------
    log("6/6 Running single-point basis, quadrature, and RHF cross-checks at (EO_opt, TS_opt, AA_opt)...")
    stationary = [("EO", eo_opt_deck), ("TS", ts_opt_deck), ("AA", aa_opt_deck)]
    cross_checks: Dict[str, Dict[str, Any]] = {}

    for label, d_base in stationary:
        c_stat = deck_coords(d_base)
        entry: Dict[str, Any] = {}

        # RHF SZ STO-6G
        t0 = time.perf_counter()
        e_rhf_sz = rhf_energy_in_sto_ng(d_base, n_prim=6)
        entry["RHF_SZ_STO6G"] = round(e_rhf_sz, 8)

        # CAS(4,4) SZ STO-8G
        st_8g = vb_drv.run_deck(build_c2h4o_deck(c_stat, f"{label} SZ STO-8G", basis="SZ", sto_ng_primitives=8))
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
        log(
            f"    {label} cross-checks ({entry['DZ_elapsed_s']}s): "
            f"SZ-6G_RHF={entry['RHF_SZ_STO6G']:.6f} | SZ-8G_CAS={entry['CAS44_SZ_STO8G']:.6f} | "
            f"DZ-6G_CAS={entry['CAS44_DZ_STO6G']:.6f} | DZ_RHF={entry['RHF_DZ_STO6G']:.6f}"
        )

    # Exact ProlateGrid check at (EO_opt, TS_opt, AA_opt), timing EO first to guarantee bounded runtime
    prolate_results: Dict[str, Any] = {}
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

    # Save runnable YAML example decks
    ex_dir = REPO_ROOT / "examples"
    make_clean_deck_for_yaml(eo_opt_deck, "Ethylene Oxide (Localized VB CAS(4,4) SZ/STO-6G Minimum)").to_yaml(
        ex_dir / "EO_localized_vb.yaml"
    )
    make_clean_deck_for_yaml(aa_opt_deck, "Acetaldehyde (Localized VB CAS(4,4) SZ/STO-6G Minimum)").to_yaml(
        ex_dir / "AA_localized_vb.yaml"
    )
    make_clean_deck_for_yaml(ts_opt_deck, "EO -> AA Hydride-Transfer TS (Localized VB CAS(4,4) SZ/STO-6G Saddle)").to_yaml(
        ex_dir / "EO_to_AA_TS_localized_vb.yaml"
    )

    # Assemble energetics & save structured JSON data
    e_eo, e_ts, e_aa = float(st_eo.total_energy), float(st_ts.total_energy), float(st_aa.total_energy)
    th_eo, th_ts, th_aa = vib_eo.thermochemistry, vib_ts.thermochemistry, vib_aa.thermochemistry

    def _delta(field: str, a: Any, b: Any) -> Dict[str, float]:
        val_ha = float(getattr(b, field) - getattr(a, field))
        return {
            "hartree": round(val_ha, 8),
            "kcal_mol": round(val_ha * HARTREE_TO_KCAL, 3),
            "kj_mol": round(val_ha * HARTREE_TO_KJ, 2),
        }

    summary: Dict[str, Any] = {
        "method": "masked_vb / CAS(4e,4o) all-electron SZ STO-6G (Stewart 1970)",
        "atom_order": [aid for aid, _ in ATOM_ORDER],
        "vb_orbital_specs": CAS44_VB_SPECS,
        "stationary_points": {
            "EO": {
                "converged": bool(opt_eo.converged and st_eo.converged),
                "opt_steps": int(opt_eo.num_steps),
                "energy_ha": round(e_eo, 8),
                "grad_norm_ha_bohr": round(float(np.linalg.norm(g_eo)), 8),
                "virial_ratio": round(float(st_eo.virial_ratio), 5),
                "dipole_debye": round(float(st_eo.dipole_moment_debye), 4),
                "num_imaginary": int(vib_eo.num_imaginary),
                "frequencies_cm": eo_freqs,
                "thermochemistry": th_eo.to_dict(),
                "net_charges": {k: round(float(v), 5) for k, v in st_eo["net_charges"].items()},
                "active_natural_occupations": [round(float(x), 5) for x in st_eo["active_natural_occupations"]],
                "metrics": internal_metrics(c_eo_opt),
                "coordinates_bohr": c_eo_opt.tolist(),
            },
            "TS": {
                "converged": bool(prfo_res.converged and st_ts.converged),
                "neb_iterations": int(neb_res.num_iterations),
                "prfo_cycles": int(prfo_res.num_cycles),
                "energy_ha": round(e_ts, 8),
                "grad_norm_ha_bohr": round(float(np.linalg.norm(g_ts)), 8),
                "virial_ratio": round(float(st_ts.virial_ratio), 5),
                "dipole_debye": round(float(st_ts.dipole_moment_debye), 4),
                "num_imaginary": int(vib_ts.num_imaginary),
                "imaginary_frequency_cm": round(float(imag_mode.frequency_cm), 2),
                "imaginary_atom_displacements": atom_disp_norms,
                "imaginary_dominant_atom": dominant_atom,
                "frequencies_cm": ts_freqs,
                "thermochemistry": th_ts.to_dict(),
                "net_charges": {k: round(float(v), 5) for k, v in st_ts["net_charges"].items()},
                "active_natural_occupations": [round(float(x), 5) for x in st_ts["active_natural_occupations"]],
                "metrics": internal_metrics(deck_coords(ts_opt_deck)),
                "coordinates_bohr": deck_coords(ts_opt_deck).tolist(),
            },
            "AA": {
                "converged": bool(opt_aa.converged and st_aa.converged),
                "opt_steps": int(opt_aa.num_steps),
                "energy_ha": round(e_aa, 8),
                "grad_norm_ha_bohr": round(float(np.linalg.norm(g_aa)), 8),
                "virial_ratio": round(float(st_aa.virial_ratio), 5),
                "dipole_debye": round(float(st_aa.dipole_moment_debye), 4),
                "num_imaginary": int(vib_aa.num_imaginary),
                "frequencies_cm": aa_freqs,
                "thermochemistry": th_aa.to_dict(),
                "net_charges": {k: round(float(v), 5) for k, v in st_aa["net_charges"].items()},
                "active_natural_occupations": [round(float(x), 5) for x in st_aa["active_natural_occupations"]],
                "metrics": internal_metrics(c_aa_opt),
                "coordinates_bohr": c_aa_opt.tolist(),
            },
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
        "DZ_STO6G_CAS44_delta_E_dagger_fwd_kcal": round(
            (cross_checks["TS"]["CAS44_DZ_STO6G"] - cross_checks["EO"]["CAS44_DZ_STO6G"]) * HARTREE_TO_KCAL, 3
        ),
        "SZ_STO8G_CAS44_delta_E_rxn_kcal": round(
            (cross_checks["AA"]["CAS44_SZ_STO8G"] - cross_checks["EO"]["CAS44_SZ_STO8G"]) * HARTREE_TO_KCAL, 3
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

    out_dir = REPO_ROOT / "data" / "eo_to_aa_localized_vb"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    irc_payload = {
        "num_points": len(ordered_irc),
        "eo_endpoint_s": ordered_irc[0]["s_bohr_amu_half"] if ordered_irc else 0.0,
        "aa_endpoint_s": ordered_irc[-1]["s_bohr_amu_half"] if ordered_irc else 0.0,
        "eo_endpoint_rel_kcal": ordered_irc[0]["rel_energy_eo_kcal"] if ordered_irc else 0.0,
        "aa_endpoint_rel_kcal": ordered_irc[-1]["rel_energy_eo_kcal"] if ordered_irc else 0.0,
        "points": ordered_irc,
    }
    with open(out_dir / "irc_trajectory.json", "w", encoding="utf-8") as f:
        json.dump(irc_payload, f, indent=2)

    log(f"ALL DONE in {summary['elapsed_seconds']:.1f}s!")
    log(
        f"Summary: ΔE_rxn={summary['energetics']['delta_E_rxn_AA_minus_EO']['kcal_mol']:.2f} kcal/mol | "
        f"ΔH(298)={summary['energetics']['delta_H298_rxn']['kcal_mol']:.2f} | "
        f"ΔE‡_fwd={summary['energetics']['delta_E_dagger_fwd']['kcal_mol']:.2f} kcal/mol | "
        f"ΔG‡(298)={summary['energetics']['delta_G298_dagger_fwd']['kcal_mol']:.2f} kcal/mol"
    )


if __name__ == "__main__":
    main()
