#!/usr/bin/env python3
"""Reproducible Stages 1 & 2 computational runner for NO adsorption on Ni(100).

Bridges Chapter IX of Jack A. Smith's 1978 PhD dissertation (4-fold hollow Ni5
two-orientation hypothesis and HREELS/XPS comparisons) with Phase 18 variational
Localized VB (MaskedVB CAS / exact_backprojected embedding, analytical STO-6G
nuclear gradients, clamped-substrate partial Hessians, harmonic-adsorbate ΔG(T),
spin-resolved Mayer bond orders, and Bagus/Siegbahn CVS-Delta-CASSCF core-hole
1s spectroscopy).

Outputs (under data/no_ni100_dissociation/):
  - stage1_gas_benchmarks.json
  - stage2_adsorption_sites.json
  - stage2_core_level_shifts.json

Stages 3 (NO* -> N* + O* CI-NEB/P-RFO/IRC) and 4 (N2 catalytic network) are
out of scope for this run and deferred to the commissioned follow-on study.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from phatpsy.chemisorption import (
    HARTREE_TO_EV,
    HARTREE_TO_KCAL_MOL,
    compute_delta_casscf_core_1s_be,
    compute_mayer_bond_orders,
    compute_surface_core_level_shifts,
    get_mayer_bond_order,
)
from phatpsy.multicenter_integrals import sto_ng_ao_integrals
from phatpsy.optimize import optimize_geometry
from phatpsy.schema import AtomConfig, CalculationDeck, ControlConfig
from phatpsy.vb_driver import (
    LocalizedVBDriver,
    _atomic_num,
    _resolve_atom_stos,
    build_surface_plaquette_vb_specs,
)
from phatpsy.vibrations import (
    ATOMIC_MASSES_AMU,
    GAS_CONSTANT_R_CAL_MOL_K,
    HARTREE_TO_KCAL,
    compute_thermochemistry,
    compute_vibrations,
)

BOHR_TO_ANGSTROM: float = 0.529177210903
HARTREE_TO_KJ_MOL: float = 2625.499639
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "no_ni100_dissociation"
EXAMPLES_DIR = REPO_ROOT / "examples"


def _atomic_mass(sym: str) -> float:
    return float(ATOMIC_MASSES_AMU.get(sym.strip().upper(), 14.0))


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _run_with_restart(
    drv: LocalizedVBDriver,
    deck: CalculationDeck,
    warm_orbitals: Optional[Sequence[np.ndarray]] = None,
):
    return drv.run_deck(deck, warm_orbitals=warm_orbitals)


# ---------------------------------------------------------------------------
# Gas-phase Stage 1 Builders & Helpers
# ---------------------------------------------------------------------------


def _gas_ctrl(basis: str, convergence: float = 1.0e-7, max_cycles: int = 250) -> ControlConfig:
    return ControlConfig.localized_vb(
        method="masked_vb",
        vb_wavefunction="cas",
        basis=basis,
        domain_mode="bond_local",
        core_embedding="exact_backprojected",
        environment_embedding="none",
        integral_backend="sto_ng",
        sto_ng_primitives=6,
        max_cycles=max_cycles,
        convergence=convergence,
    )


def _build_gas_deck(species: str, basis: str) -> CalculationDeck:
    ctrl = _gas_ctrl(basis)
    if species == "NO":
        geom = [
            AtomConfig(id="N1", element="N", xyz=[0.0, 0.0, 0.0], active=True, fixed=False, basis=basis),
            AtomConfig(id="O1", element="O", xyz=[0.0, 0.0, 2.1747], active=True, fixed=False, basis=basis),
        ]
        spin = 1
    elif species == "N2":
        geom = [
            AtomConfig(id="N1", element="N", xyz=[0.0, 0.0, 0.0], active=True, fixed=False, basis=basis),
            AtomConfig(id="N2", element="N", xyz=[0.0, 0.0, 2.0744], active=True, fixed=False, basis=basis),
        ]
        spin = 0
    elif species == "O2":
        geom = [
            AtomConfig(id="O1", element="O", xyz=[0.0, 0.0, 0.0], active=True, fixed=False, basis=basis),
            AtomConfig(id="O2", element="O", xyz=[0.0, 0.0, 2.2820], active=True, fixed=False, basis=basis),
        ]
        spin = 2
    elif species == "CO":
        geom = [
            AtomConfig(id="C1", element="C", xyz=[0.0, 0.0, 0.0], active=True, fixed=False, basis=basis),
            AtomConfig(id="O1", element="O", xyz=[0.0, 0.0, 2.1322], active=True, fixed=False, basis=basis),
        ]
        spin = 0
    elif species == "N2O":
        geom = [
            AtomConfig(id="N1", element="N", xyz=[0.0, 0.0, 0.0], active=True, fixed=False, basis=basis),
            AtomConfig(id="N2", element="N", xyz=[0.0, 0.0, 2.132], active=True, fixed=False, basis=basis),
            AtomConfig(id="O1", element="O", xyz=[0.0, 0.0, 4.374], active=True, fixed=False, basis=basis),
        ]
        ctrl.domain_mode = "custom"
        ctrl.vb_orbital_specs = [
            {"role": "inactive", "atoms": ["N1"]},
            {"role": "inactive", "atoms": ["O1"]},
        ] + [{"role": "active", "atoms": ["N1", "N2", "O1"]} for _ in range(8)]
        spin = 0
    else:
        raise ValueError(f"Unknown gas species: {species}")
    return CalculationDeck(title=f"Gas {species} ({basis}/STO-6G)", net_spin=spin, control=ctrl, geometry=geom)


def _atom_ref_energy(drv: LocalizedVBDriver, element: str, basis: str) -> float:
    spin_map = {"C": 2, "N": 3, "O": 2}
    dk = CalculationDeck(
        title=f"Isolated {element} atom ({basis})",
        net_spin=spin_map[element],
        control=_gas_ctrl(basis, convergence=1.0e-7, max_cycles=150),
        geometry=[AtomConfig(id=f"{element}1", element=element, xyz=[0.0, 0.0, 0.0], active=True, fixed=False, basis=basis)],
    )
    return float(drv.run_deck(dk)["total_energy"])


def _thermo_at_temps(
    vib_res,
    temps_k: Sequence[float],
    spin_mult: int,
    *,
    is_adsorbate: bool,
    sym_num: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    freqs = [float(m.frequency_cm) for m in vib_res.vibrational_modes if (not m.is_imaginary and m.frequency_cm > 1e-3)]
    coords = np.array([a["xyz"] for a in vib_res.atoms], dtype=float)
    masses = np.array([a["mass_amu"] for a in vib_res.atoms], dtype=float)
    e_elec = float(vib_res.thermochemistry.electronic_energy)
    out: Dict[str, Dict[str, float]] = {}
    for temp in temps_k:
        th = compute_thermochemistry(
            frequencies=freqs,
            coords=coords,
            masses=masses,
            electronic_energy=e_elec,
            spin_multiplicity=spin_mult,
            temperature_k=float(temp),
            symmetry_number=sym_num,
            is_adsorbate=is_adsorbate,
            quasi_rrho_floor_cm=50.0,
        )
        out[f"{temp:g}K"] = th.to_dict()
    return out


def run_stage1_gas_benchmarks(*, force: bool = False) -> Dict[str, Any]:
    """Run Stage 1 gas-phase optimizations, frequencies, thermo, NO PEC, and core-hole spectra."""
    out_path = OUT_DIR / "stage1_gas_benchmarks.json"
    data = _load_json(out_path) if not force else {}
    data.setdefault("metadata", {
        "stage": "Stage 1: Gas-Phase Reference Benchmarks",
        "method": "LocalizedVB MaskedVB CAS / STO-6G / exact_backprojected cores",
        "experimental_report_only": {
            "NO": {"re_bohr": 2.1747, "omega_e_cm": 1904.2, "de_ev": 6.61, "n1s_esca_avg_ev": 410.7, "o1s_esca_avg_ev": 543.4},
            "N2": {"re_bohr": 2.0744, "omega_e_cm": 2358.6, "de_ev": 9.91, "n1s_esca_ev": 409.9},
            "O2": {"re_bohr": 2.2820, "omega_e_cm": 1580.2, "de_ev": 5.21},
            "CO": {"re_bohr": 2.1322, "omega_e_cm": 2169.8, "de_ev": 11.23, "c1s_esca_ev": 296.2, "o1s_esca_ev": 542.6},
            "2NO_to_N2_O2": {"delta_g_298_kj_mol": -173.1},
            "dissertation_1978_NO_Table4": {"re_bohr": 2.17, "omega_e_cm": 1993.0, "de_ev": 6.0},
        },
    })
    data.setdefault("atoms", {})
    data.setdefault("species", {})
    data.setdefault("no_pec", {})
    data.setdefault("reaction_cycle_2NO_to_N2_O2", {})
    data.setdefault("core_1s_gas", {})

    drv = LocalizedVBDriver(quiet=True)

    # 1. Atomic ground-state energies (N, O, C) across SZ, SZP, DZP
    for basis in ("SZ", "SZP", "DZP"):
        b_atoms = data["atoms"].setdefault(basis, {})
        for el in ("C", "N", "O"):
            if el not in b_atoms:
                t0 = time.perf_counter()
                e_el = _atom_ref_energy(drv, el, basis)
                b_atoms[el] = float(e_el)
                print(f"[Stage 1] Atom {el} ({basis}): {e_el:.8f} Ha ({time.perf_counter()-t0:.2f}s)", flush=True)
                _save_json(out_path, data)

    # 2. Molecular stationary points across SZ, SZP, DZP
    atom_consts = {"NO": ["N", "O"], "N2": ["N", "N"], "O2": ["O", "O"], "CO": ["C", "O"], "N2O": ["N", "N", "O"]}
    spin_mults = {"NO": 2, "N2": 1, "O2": 3, "CO": 1, "N2O": 1}
    sym_nums = {"NO": 1, "N2": 2, "O2": 2, "CO": 1, "N2O": 1}
    pairs_map = {"NO": ("N1", "O1"), "N2": ("N1", "N2"), "O2": ("O1", "O2"), "CO": ("C1", "O1")}

    for sp in ("NO", "N2", "O2", "CO", "N2O"):
        sp_entry = data["species"].setdefault(sp, {})
        for basis in ("SZ", "SZP", "DZP"):
            if basis in sp_entry:
                continue
            t0 = time.perf_counter()
            dk0 = _build_gas_deck(sp, basis)
            opt = optimize_geometry(dk0, driver=drv, max_steps=35, grad_tol=5.0e-5, energy_tol=1.0e-7, step_limit=0.15, quiet=True)
            dk_opt = opt.optimized_deck
            res_opt = _run_with_restart(drv, dk_opt)
            vib = compute_vibrations(
                dk_opt,
                driver=drv,
                step_size=0.005,
                spin_multiplicity=spin_mults[sp],
                temperature_k=298.15,
                symmetry_number=sym_nums[sp],
                quiet=True,
            )
            thermo_multi = _thermo_at_temps(
                vib, [298.15, 300.0, 700.0], spin_mults[sp], is_adsorbate=False, sym_num=sym_nums[sp]
            )
            mayer = compute_mayer_bond_orders(res_opt)
            # Extract 1s Koopmans (-eps_1s) directly from koopmans_core_energies_ha
            kmap = res_opt.get("koopmans_core_energies_ha") or {}
            koop_1s = {k[:-3]: round(-float(v) * HARTREE_TO_EV, 4) for k, v in kmap.items() if k.endswith("_1s")}

            e_mol = float(res_opt["total_energy"])
            e_sep = sum(data["atoms"][basis][el] for el in atom_consts[sp])
            de_ha = float(e_sep - e_mol)
            zpve_ha = float(vib.thermochemistry.zero_point_energy_au)
            d0_ha = float(de_ha - zpve_ha)

            coords = [a.xyz for a in dk_opt.geometry]
            bond_lengths_bohr = {}
            if len(coords) == 2:
                r_bohr = float(np.linalg.norm(np.asarray(coords[1]) - np.asarray(coords[0])))
                bond_lengths_bohr["R_eq"] = round(r_bohr, 5)
            elif len(coords) == 3:
                bond_lengths_bohr["R_NN"] = round(float(np.linalg.norm(np.asarray(coords[1]) - np.asarray(coords[0]))), 5)
                bond_lengths_bohr["R_NO"] = round(float(np.linalg.norm(np.asarray(coords[2]) - np.asarray(coords[1]))), 5)

            freqs = sorted([round(float(m.frequency_cm), 2) for m in vib.vibrational_modes])
            rec = {
                "basis": basis,
                "converged": bool(res_opt["converged"] and opt.converged),
                "total_energy_ha": round(e_mol, 8),
                "separated_atoms_ha": round(e_sep, 8),
                "de_ha": round(de_ha, 6),
                "de_ev": round(de_ha * HARTREE_TO_EV, 4),
                "de_kcal_mol": round(de_ha * HARTREE_TO_KCAL_MOL, 3),
                "d0_ev": round(d0_ha * HARTREE_TO_EV, 4),
                "virial_ratio": round(float(res_opt["virial_ratio"]), 6),
                "dipole_debye": round(float(res_opt["dipole_moment_debye"]), 4),
                "bond_lengths_bohr": bond_lengths_bohr,
                "geometry_xyz_bohr": [[round(float(x), 6) for x in c] for c in coords],
                "vibrational_frequencies_cm": freqs,
                "num_imaginary": int(vib.num_imaginary),
                "zpve_kcal_mol": round(float(vib.thermochemistry.zero_point_energy_kcal), 4),
                "thermochemistry": thermo_multi,
                "mayer_bond_orders": mayer["pair_bond_orders"],
                "active_natural_occupations": [round(float(x), 4) for x in res_opt.get("active_natural_occupations", [])],
                "net_charges": {k: round(float(v), 4) for k, v in (res_opt.get("net_charges") or {}).items()},
                "koopmans_1s_be_ev": koop_1s,
                "koopmans_method": "koopmans_orbital_energy",
                "expected_koopmans_offset_ev": 15.0,
            }
            if sp in pairs_map:
                a1, a2 = pairs_map[sp]
                rec["primary_mayer_bo"] = round(get_mayer_bond_order(mayer, a1, a2), 4)
            sp_entry[basis] = rec
            print(
                f"[Stage 1] {sp} ({basis}): R={bond_lengths_bohr}, w={freqs}, De={rec['de_ev']:.3f} eV ({time.perf_counter()-t0:.2f}s)",
                flush=True,
            )
            _save_json(out_path, data)

    # 3. Thermodynamic cycle: 2 NO(g) -> N2(g) + O2(g)
    for basis in ("SZ", "SZP", "DZP"):
        no_b = data["species"]["NO"][basis]
        n2_b = data["species"]["N2"][basis]
        o2_b = data["species"]["O2"][basis]
        de_rxn_ha = float(n2_b["total_energy_ha"]) + float(o2_b["total_energy_ha"]) - 2.0 * float(no_b["total_energy_ha"])
        cycle = {"delta_e_ha": round(de_rxn_ha, 6), "delta_e_kj_mol": round(de_rxn_ha * HARTREE_TO_KJ_MOL, 2)}
        for tk in ("298.15K", "300K", "700K"):
            dh_ha = (
                n2_b["thermochemistry"][tk]["total_enthalpy_h_t"]
                + o2_b["thermochemistry"][tk]["total_enthalpy_h_t"]
                - 2.0 * no_b["thermochemistry"][tk]["total_enthalpy_h_t"]
            )
            dg_ha = (
                n2_b["thermochemistry"][tk]["total_gibbs_free_energy_g_t"]
                + o2_b["thermochemistry"][tk]["total_gibbs_free_energy_g_t"]
                - 2.0 * no_b["thermochemistry"][tk]["total_gibbs_free_energy_g_t"]
            )
            cycle[f"delta_h_{tk}_kj_mol"] = round(dh_ha * HARTREE_TO_KJ_MOL, 2)
            cycle[f"delta_g_{tk}_kj_mol"] = round(dg_ha * HARTREE_TO_KJ_MOL, 2)
        data["reaction_cycle_2NO_to_N2_O2"][basis] = cycle
    _save_json(out_path, data)

    # 4. Full gas NO potential energy curve (SZ & DZP) with warm restart
    r_grid = [1.7, 1.9, 2.05, 2.1747, 2.3, 2.5, 2.8, 3.2, 3.8, 4.5, 6.0, 8.0]
    for basis in ("SZ", "DZP"):
        if basis in data["no_pec"]:
            continue
        t0 = time.perf_counter()
        pts = []
        warm = None
        e_atoms = data["atoms"][basis]["N"] + data["atoms"][basis]["O"]
        for r_val in r_grid:
            dk = _build_gas_deck("NO", basis)
            dk.geometry[1].xyz[2] = float(r_val)
            res = _run_with_restart(drv, dk, warm_orbitals=warm)
            warm = res.get("localized_orbitals_embedded", res.get("localized_orbitals"))
            e_tot = float(res["total_energy"])
            bo = get_mayer_bond_order(compute_mayer_bond_orders(res), "N1", "O1")
            pts.append({
                "r_bohr": float(r_val),
                "energy_ha": round(e_tot, 8),
                "rel_to_atoms_ev": round((e_tot - e_atoms) * HARTREE_TO_EV, 4),
                "mayer_bo": round(bo, 4),
                "active_natural_occupations": [round(float(x), 4) for x in res.get("active_natural_occupations", [])],
            })
        data["no_pec"][basis] = pts
        print(f"[Stage 1] NO PEC ({basis}) computed in {time.perf_counter()-t0:.2f}s", flush=True)
        _save_json(out_path, data)

    # 5. Gas-phase Koopmans + CVS-Delta-CASSCF 1s BEs at converged geometries
    gas_core_targets = [
        ("NO", "SZ", ["N1", "O1"]),
        ("NO", "DZ", ["N1", "O1"]),
        ("N2", "SZ", ["N1"]),
        ("CO", "SZ", ["C1", "O1"]),
    ]
    for sp, basis, tgts in gas_core_targets:
        key = f"{sp}_{basis}"
        if key in data["core_1s_gas"]:
            continue
        t0 = time.perf_counter()
        dk = _build_gas_deck(sp, basis)
        if basis in data["species"][sp]:
            for at, xyz in zip(dk.geometry, data["species"][sp][basis]["geometry_xyz_bohr"]):
                at.xyz = list(xyz)
        res_gs = drv.run_deck(dk)
        entry = {}
        for tid in tgts:
            entry[tid] = compute_delta_casscf_core_1s_be(dk, tid, warm_ground_state=res_gs)
        data["core_1s_gas"][key] = entry
        print(f"[Stage 1] Core 1s BEs ({key}) computed in {time.perf_counter()-t0:.2f}s", flush=True)
        _save_json(out_path, data)

    return data


# ---------------------------------------------------------------------------
# Stage 2 Surface & Cluster Builders
# ---------------------------------------------------------------------------


def _ni4_ni9_geometry(basis: str = "SZ") -> Tuple[List[AtomConfig], List[AtomConfig]]:
    a_nn = 4.71
    h = 0.5 * a_nn
    z2 = -a_nn / math.sqrt(2.0)
    active_ni = [
        AtomConfig(id="Ni1", element="Ni", xyz=[h, h, 0.0], active=True, fixed=True, basis=basis),
        AtomConfig(id="Ni2", element="Ni", xyz=[-h, h, 0.0], active=True, fixed=True, basis=basis),
        AtomConfig(id="Ni3", element="Ni", xyz=[-h, -h, 0.0], active=True, fixed=True, basis=basis),
        AtomConfig(id="Ni4", element="Ni", xyz=[h, -h, 0.0], active=True, fixed=True, basis=basis),
    ]
    spec_coords = [
        ("Ni5", [0.0, 0.0, z2]),
        ("Ni6", [3.0 * h, h, 0.0]),
        ("Ni7", [3.0 * h, -h, 0.0]),
        ("Ni8", [-3.0 * h, h, 0.0]),
        ("Ni9", [-3.0 * h, -h, 0.0]),
        ("Ni10", [h, 3.0 * h, 0.0]),
        ("Ni11", [-h, 3.0 * h, 0.0]),
        ("Ni12", [h, -3.0 * h, 0.0]),
        ("Ni13", [-h, -3.0 * h, 0.0]),
    ]
    spectator_ni = [AtomConfig(id=sid, element="Ni", xyz=c, active=False, fixed=True, basis=basis) for sid, c in spec_coords]
    return active_ni, spectator_ni


def _ni5_1978_geometry(basis: str = "SZ") -> List[AtomConfig]:
    a_nn = 4.71
    h = 0.5 * a_nn
    z2 = -a_nn / math.sqrt(2.0)
    return [
        AtomConfig(id="Ni1", element="Ni", xyz=[h, h, 0.0], active=True, fixed=True, basis=basis),
        AtomConfig(id="Ni2", element="Ni", xyz=[-h, h, 0.0], active=True, fixed=True, basis=basis),
        AtomConfig(id="Ni3", element="Ni", xyz=[-h, -h, 0.0], active=True, fixed=True, basis=basis),
        AtomConfig(id="Ni4", element="Ni", xyz=[h, -h, 0.0], active=True, fixed=True, basis=basis),
        AtomConfig(id="Ni5", element="Ni", xyz=[0.0, 0.0, z2], active=True, fixed=True, basis=basis),
    ]


def _surface_ctrl(basis: str, *, has_spectators: bool, max_cycles: int = 250, convergence: float = 1.0e-6) -> ControlConfig:
    return ControlConfig.localized_vb(
        method="masked_vb",
        vb_wavefunction="cas",
        basis=basis,
        domain_mode="custom",
        core_embedding="exact_backprojected",
        environment_embedding=("exact_backprojected" if has_spectators else "none"),
        integral_backend="sto_ng",
        sto_ng_primitives=6,
        max_cycles=max_cycles,
        convergence=convergence,
    )


def _build_plaquette_matched_gas_no_deck(basis: str = "SZ") -> CalculationDeck:
    """Isolated NO reference using the exact plaquette-matched 5-inactive + 2-active CAS(1e,2o) partition."""
    dk = CalculationDeck(
        title=f"Gas NO Plaquette-Matched Reference ({basis}/STO-6G)",
        net_spin=1,
        control=_surface_ctrl(basis, has_spectators=False, max_cycles=250, convergence=1.0e-7),
        geometry=[
            AtomConfig(id="N1", element="N", xyz=[0.0, 0.0, 0.0], active=True, fixed=False, basis=basis),
            AtomConfig(id="O1", element="O", xyz=[0.0, 0.0, 2.26], active=True, fixed=False, basis=basis),
        ],
    )
    dk.control.vb_orbital_specs = build_surface_plaquette_vb_specs(dk)
    return dk


def _update_yaml_no_coords(yaml_name: str, geom: Sequence[AtomConfig]) -> None:
    p = EXAMPLES_DIR / yaml_name
    if not p.exists():
        return
    dk = CalculationDeck.from_yaml(p)
    pos_map = {a.id: [round(float(v), 6) for v in a.xyz] for a in geom}
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    out_lines = []
    for line in lines:
        updated = False
        for aid in ("N1", "O1"):
            if f"id: {aid}," in line and aid in pos_map:
                import re
                new_xyz = f"xyz: [{pos_map[aid][0]:.6f}, {pos_map[aid][1]:.6f}, {pos_map[aid][2]:.6f}]"
                line = re.sub(r"xyz: \[[^\]]+\]", new_xyz, line)
                updated = True
                break
        out_lines.append(line)
    p.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _extract_ads_geometry_metrics(deck: CalculationDeck, n_id: str = "N1", o_id: str = "O1") -> Dict[str, Any]:
    by_id = {a.id: np.asarray(a.xyz, dtype=float) for a in deck.geometry}
    rN = by_id[n_id]
    rO = by_id[o_id]
    vec = rO - rN
    r_no = float(np.linalg.norm(vec))
    # Tilt angle relative to surface normal +z (0 deg = upright N-down, 180 deg = upright O-down)
    cos_t = max(-1.0, min(1.0, float(vec[2] / max(r_no, 1e-12))))
    tilt_deg = float(math.degrees(math.acos(cos_t)))
    ni_surfs = [np.asarray(a.xyz, dtype=float) for a in deck.geometry if a.element == "Ni" and abs(a.xyz[2]) < 0.5]
    d_ni_n = min(float(np.linalg.norm(rN - rn)) for rn in ni_surfs)
    d_ni_o = min(float(np.linalg.norm(rO - rn)) for rn in ni_surfs)
    out = {
        "r_no_bohr": round(r_no, 5),
        "r_no_angstrom": round(r_no * BOHR_TO_ANGSTROM, 5),
        "z_n_bohr": round(float(rN[2]), 5),
        "z_o_bohr": round(float(rO[2]), 5),
        "d_min_ni_n_bohr": round(d_ni_n, 5),
        "d_min_ni_o_bohr": round(d_ni_o, 5),
        "tilt_angle_deg_from_z": round(tilt_deg, 3),
        "n_xyz_bohr": [round(float(v), 6) for v in rN],
        "o_xyz_bohr": [round(float(v), 6) for v in rO],
    }
    if "O2" in by_id:
        out["o2_xyz_bohr"] = [round(float(v), 6) for v in by_id["O2"]]
    return out


def _surface_electronic_summary(
    res,
    deck: CalculationDeck,
    e_clean_slab: float,
    e_gas_no: float,
    *,
    e_gas_no_ref: Optional[float] = None,
) -> Dict[str, Any]:
    e_tot = float(res["total_energy"])
    e_ref_gas = e_gas_no if e_gas_no_ref is None else e_gas_no_ref
    e_ads_ha = float(e_tot - e_clean_slab - e_ref_gas)
    mayer = compute_mayer_bond_orders(res)
    bo_no = get_mayer_bond_order(mayer, "N1", "O1")
    ni_ids = [aid for aid in mayer["atom_ids"] if aid.startswith("Ni")]
    sum_ni_n = sum(get_mayer_bond_order(mayer, nid, "N1") for nid in ni_ids)
    sum_ni_o = sum(get_mayer_bond_order(mayer, nid, "O1") for nid in ni_ids)
    net_charges = res.get("net_charges") or {}
    q_no = float(net_charges.get("N1", 0.0) + net_charges.get("O1", 0.0))
    kmap = res.get("koopmans_core_energies_ha") or {}
    koop_ev = {k: round(-float(v) * HARTREE_TO_EV, 4) for k, v in kmap.items() if k.endswith("_1s")}
    return {
        "total_energy_ha": round(e_tot, 8),
        "converged": bool(res["converged"]),
        "cycles": int(res["cycles"]),
        "e_ads_ha": round(e_ads_ha, 6),
        "e_ads_ev": round(e_ads_ha * HARTREE_TO_EV, 4),
        "e_ads_kcal_mol": round(e_ads_ha * HARTREE_TO_KCAL_MOL, 3),
        "mayer_bo_no": round(bo_no, 4),
        "mayer_sum_ni_n": round(sum_ni_n, 4),
        "mayer_sum_ni_o": round(sum_ni_o, 4),
        "no_lowdin_net_charge": round(q_no, 4),
        "net_charges": {k: round(float(v), 4) for k, v in net_charges.items()},
        "active_natural_occupations": [round(float(x), 4) for x in res.get("active_natural_occupations", [])],
        "koopmans_1s_be_ev": koop_ev,
    }


def _project_warm_orbitals_cross_basis(
    deck_src: CalculationDeck,
    res_src: Mapping[str, Any],
    deck_dst: CalculationDeck,
) -> List[np.ndarray]:
    """L2-project converged active-site localized orbitals across STO bases via exact STO-KG overlap."""
    from pyscf import gto

    deck_src.ensure_atom_ids()
    deck_dst.ensure_atom_ids()
    n_prim = int(getattr(deck_dst.control, "sto_ng_primitives", 6) or 6)

    act_s = [a for a in deck_src.geometry if not a.virtual and a.is_active]
    em_s = deck_src.resolve_elements()
    b_s = [_resolve_atom_stos(a, deck_src, em_s)[0] for a in act_s]
    _, ctx_s = sto_ng_ao_integrals(
        [(float(_atomic_num(a)), tuple(float(v) for v in a.xyz), b) for a, b in zip(act_s, b_s)],
        n_prim=n_prim,
        return_pyscf_context=True,
        build_eri=False,
    )

    act_d = [a for a in deck_dst.geometry if not a.virtual and a.is_active]
    em_d = deck_dst.resolve_elements()
    b_d = [_resolve_atom_stos(a, deck_dst, em_d)[0] for a in act_d]
    ints_d, ctx_d = sto_ng_ao_integrals(
        [(float(_atomic_num(a)), tuple(float(v) for v in a.xyz), b) for a, b in zip(act_d, b_d)],
        n_prim=n_prim,
        return_pyscf_context=True,
        build_eri=False,
    )

    S_ds = ctx_d.ao_map.T @ gto.intor_cross("int1e_ovlp", ctx_d.mol, ctx_s.mol) @ ctx_s.ao_map
    S_dd = 0.5 * (ints_d.S + ints_d.S.T)
    aid_to_ao_d: Dict[str, List[int]] = {}
    off = 0
    for a, b in zip(act_d, b_d):
        aid_to_ao_d[str(a.id)] = list(range(off, off + len(b)))
        off += len(b)

    orbs_src = np.asarray(res_src["localized_orbitals"], dtype=float)
    warm_d: List[np.ndarray] = []
    for k, spec in enumerate(deck_dst.control.vb_orbital_specs or []):
        m_d = np.array([idx for aid in spec["atoms"] for idx in aid_to_ao_d[aid]], dtype=int)
        rhs = S_ds[m_d, :] @ orbs_src[k]
        c_m = np.linalg.pinv(S_dd[np.ix_(m_d, m_d)], rcond=1e-10) @ rhs
        v_d = np.zeros(ints_d.nao, dtype=float)
        v_d[m_d] = c_m
        v_d /= max(float(np.sqrt(max(float(v_d @ S_dd @ v_d), 1e-16))), 1e-12)
        warm_d.append(v_d)
    return warm_d


def run_stage2_adsorption(stage1_data: Dict[str, Any], *, force: bool = False, skip_dzp_slab: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run Stage 2 embedded Ni4/Ni9, Ni5 (1978), and Ni2 surface adsorption & XPS studies."""
    sites_path = OUT_DIR / "stage2_adsorption_sites.json"
    core_path = OUT_DIR / "stage2_core_level_shifts.json"
    sites = _load_json(sites_path) if not force else {}
    cores = _load_json(core_path) if not force else {}

    sites.setdefault("metadata", {
        "stage": "Stage 2: Molecular NO* Adsorption on Ferromagnetic Embedded Ni(100)",
        "dissertation_year": 1978,
        "lattice_constant_bohr": 4.71,
        "nickel_magnetic_model": "ferromagnetic_3d8_4s2_2_spins_up_per_active_Ni",
        "adsorbate_spin_coupling": "antiferromagnetic_2pi_star_pairing_net_spin_2k_minus_1",
        "modern_uhv_report_only": {
            "hreels_hollow_cm": "1440-1520",
            "hreels_bridge_cm": "1590-1670",
            "hreels_top_cm": "1780-1820",
            "xps_hollow_N1s_ev": "~399.8-400.2",
            "xps_hollow_O1s_ev": "~531.0",
            "brundle_300K_state_N1s_ev": "402-403 (re-assigned to O*-coadsorbed NO*)",
            "dissertation_1978_Tables7_8": {"e_ads_ev": -1.0, "nu_no_cm": 1621.0, "koopmans_N1s_ev": 415.5, "koopmans_O1s_ev": 548.0, "odown_shift_N1s_ev": "+1.9", "odown_shift_O1s_ev": "+0.9"},
        },
    })
    sites.setdefault("clean_references", {})
    sites.setdefault("gas_no_plaquette_matched_references", {})
    sites.setdefault("sites", {})
    sites.setdefault("single_point_refinements", {})

    cores.setdefault("metadata", {
        "koopmans_label": "koopmans_orbital_energy",
        "expected_koopmans_offset_ev": 15.0,
        "cvs_method": "cvs_delta_casscf",
        "shift_reference": "NO_gas_SZ_plaquette_matched",
        "nickel_magnetic_model": "ferromagnetic_3d8_4s2_2_spins_up_per_active_Ni",
    })
    cores.setdefault("systems", {})

    drv = LocalizedVBDriver(quiet=True)

    # Plaquette-matched gas NO reference (5 inactive + 2 active, CAS(1e,2o)) across SZ/SZP/DZP
    # Ensures exact asymptotic E_ads(z -> inf) == 0 against Ni_k + NO ferromagnetic plaquettes.
    no_gas_stage1_sz = stage1_data["species"]["NO"]["SZ"]
    nu_gas_no_stage1_sz = float(max(no_gas_stage1_sz["vibrational_frequencies_cm"]))
    matched_refs = sites["gas_no_plaquette_matched_references"]
    for b_ref in ("SZ", "SZP", "DZP"):
        if b_ref not in matched_refs:
            t0 = time.perf_counter()
            dk_m0 = _build_plaquette_matched_gas_no_deck(b_ref)
            opt_m = optimize_geometry(dk_m0, driver=drv, max_steps=25, grad_tol=5.0e-5, energy_tol=1.0e-7, step_limit=0.15, quiet=True)
            dk_m = opt_m.optimized_deck
            res_m = drv.run_deck(dk_m)
            r_eq_m = float(np.linalg.norm(np.asarray(dk_m.geometry[1].xyz) - np.asarray(dk_m.geometry[0].xyz)))
            bo_m = get_mayer_bond_order(compute_mayer_bond_orders(res_m), "N1", "O1")
            kmap_m = res_m.get("koopmans_core_energies_ha") or {}
            koop_m = {k[:-3]: round(-float(v) * HARTREE_TO_EV, 4) for k, v in kmap_m.items() if k.endswith("_1s")}
            rec_m: Dict[str, Any] = {
                "basis": b_ref,
                "converged": bool(res_m["converged"] and opt_m.converged),
                "total_energy_ha": round(float(res_m["total_energy"]), 8),
                "r_eq_bohr": round(r_eq_m, 5),
                "geometry_xyz_bohr": [[round(float(x), 6) for x in a.xyz] for a in dk_m.geometry],
                "mayer_bo": round(bo_m, 4),
                "koopmans_1s_be_ev": koop_m,
            }
            if b_ref == "SZ":
                vib_m = compute_vibrations(
                    dk_m, driver=drv, step_size=0.005, spin_multiplicity=2, temperature_k=298.15, symmetry_number=1, quiet=True
                )
                th_m = _thermo_at_temps(vib_m, [300.0, 700.0], spin_mult=2, is_adsorbate=False, sym_num=1)
                freqs_m = sorted([round(float(m.frequency_cm), 2) for m in vib_m.vibrational_modes])
                rec_m["nu_no_cm"] = round(float(max(freqs_m)), 2)
                rec_m["zpve_kcal_mol"] = round(float(vib_m.thermochemistry.zero_point_energy_kcal), 4)
                rec_m["thermochemistry"] = th_m
            matched_refs[b_ref] = rec_m
            print(f"[Stage 2] Matched Gas NO ({b_ref}): E={rec_m['total_energy_ha']:.8f} R={r_eq_m:.5f} ({time.perf_counter()-t0:.2f}s)", flush=True)
            _save_json(sites_path, sites)

    no_matched_sz = matched_refs["SZ"]
    e_gas_no_sz = float(no_matched_sz["total_energy_ha"])
    nu_gas_no_sz = float(no_matched_sz["nu_no_cm"])
    dk_gas_no_sz = _build_plaquette_matched_gas_no_deck("SZ")
    for at, xyz in zip(dk_gas_no_sz.geometry, no_matched_sz["geometry_xyz_bohr"]):
        at.xyz = list(xyz)
    res_gas_no_sz = drv.run_deck(dk_gas_no_sz)
    th_gas_300 = float(no_matched_sz["thermochemistry"]["300K"]["total_gibbs_free_energy_g_t"])
    th_gas_700 = float(no_matched_sz["thermochemistry"]["700K"]["total_gibbs_free_energy_g_t"])
    zpve_gas_kcal = float(no_matched_sz["zpve_kcal_mol"])

    # Clean ferromagnetic reference slabs (2 spins up per active Ni atom -> net_spin = 2 * n_active_ni)
    clean_net_spins = {"Ni2": 4, "Ni4_Ni9": 8, "Ni5_1978": 10}

    def _clean_deck(cluster: str, basis: str) -> CalculationDeck:
        if cluster == "Ni4_Ni9":
            a_ni, s_ni = _ni4_ni9_geometry(basis)
            dk = CalculationDeck(
                title=f"Clean Ferromagnetic Ni4/Ni9 ({basis}/STO-6G)",
                net_spin=clean_net_spins[cluster],
                control=_surface_ctrl(basis, has_spectators=True, max_cycles=300, convergence=1.0e-7),
                geometry=a_ni + s_ni,
            )
            dk.control.vb_orbital_specs = build_surface_plaquette_vb_specs(dk)
            return dk
        elif cluster == "Ni5_1978":
            dk = CalculationDeck(
                title=f"Clean Ferromagnetic 1978 Ni5 ({basis}/STO-6G)",
                net_spin=clean_net_spins[cluster],
                control=_surface_ctrl(basis, has_spectators=False, max_cycles=300, convergence=1.0e-7),
                geometry=_ni5_1978_geometry(basis),
            )
            dk.control.vb_orbital_specs = build_surface_plaquette_vb_specs(dk)
            return dk
        elif cluster == "Ni2":
            dk = CalculationDeck(
                title=f"Clean Ferromagnetic Ni2 ({basis}/STO-6G)",
                net_spin=clean_net_spins[cluster],
                control=_surface_ctrl(basis, has_spectators=False, max_cycles=250, convergence=1.0e-7),
                geometry=[
                    AtomConfig(id="Ni1", element="Ni", xyz=[-2.355, 0.0, 0.0], active=True, fixed=True, basis=basis),
                    AtomConfig(id="Ni2", element="Ni", xyz=[2.355, 0.0, 0.0], active=True, fixed=True, basis=basis),
                ],
            )
            dk.control.vb_orbital_specs = build_surface_plaquette_vb_specs(dk)
            return dk
        raise ValueError(cluster)

    for cl in ("Ni2", "Ni5_1978", "Ni4_Ni9"):
        key = f"{cl}_SZ"
        if key not in sites["clean_references"]:
            t0 = time.perf_counter()
            dk_cl = _clean_deck(cl, "SZ")
            res_cl = drv.run_deck(dk_cl)
            sites["clean_references"][key] = {
                "total_energy_ha": round(float(res_cl["total_energy"]), 8),
                "converged": bool(res_cl["converged"]),
                "cycles": int(res_cl["cycles"]),
                "net_spin": clean_net_spins[cl],
            }
            print(f"[Stage 2] Clean {key} (net_spin={clean_net_spins[cl]}): {float(res_cl['total_energy']):.8f} Ha ({time.perf_counter()-t0:.2f}s)", flush=True)
            _save_json(sites_path, sites)

    # Clean O* on ferromagnetic Ni4/Ni9 reference for coadsorbed NO*+O* E_ads (net_spin = 8, identical CAS(8e,8o))
    def _clean_ni4_ni9_with_O_deck(z_o: float = 1.40) -> CalculationDeck:
        a_ni, s_ni = _ni4_ni9_geometry("SZ")
        o_at = AtomConfig(id="O2", element="O", xyz=[0.0, 0.0, z_o], active=True, fixed=False, basis="SZ")
        dk = CalculationDeck(
            title="Clean O* on Ferromagnetic Ni4/Ni9",
            net_spin=8,
            control=_surface_ctrl("SZ", has_spectators=True, max_cycles=300, convergence=1.0e-7),
            geometry=a_ni + [o_at] + s_ni,
        )
        dk.control.vb_orbital_specs = build_surface_plaquette_vb_specs(dk)
        return dk

    # Helper to construct Stage 2 site decks
    def _build_site_deck(site_key: str, basis: str = "SZ") -> Tuple[CalculationDeck, str, Optional[str]]:
        if site_key == "Ni4_Ni9_hollow_Ndown":
            dk = CalculationDeck.from_yaml(EXAMPLES_DIR / "NO_Ni100_hollow_Ndown.yaml")
            yaml_sync = "NO_Ni100_hollow_Ndown.yaml"
            cl = "Ni4_Ni9"
        elif site_key == "Ni4_Ni9_hollow_Ndown_tilted_test":
            dk = CalculationDeck.from_yaml(EXAMPLES_DIR / "NO_Ni100_hollow_Ndown.yaml")
            dk.title = "NO on Ni4/Ni9 4-Fold Hollow Tilted Start Test"
            # Tilt NO by ~25 deg off z-axis with completely unconstrained N & O coordinates
            for a in dk.geometry:
                if a.id == "N1":
                    a.xyz = [0.15, 0.08, 1.20]
                elif a.id == "O1":
                    a.xyz = [1.05, 0.45, 3.35]
            yaml_sync = None
            cl = "Ni4_Ni9"
        elif site_key == "Ni4_Ni9_hollow_Odown":
            dk = CalculationDeck.from_yaml(EXAMPLES_DIR / "NO_Ni100_hollow_Odown.yaml")
            yaml_sync = "NO_Ni100_hollow_Odown.yaml"
            cl = "Ni4_Ni9"
        elif site_key == "Ni4_Ni9_bridge_Ndown":
            dk = CalculationDeck.from_yaml(EXAMPLES_DIR / "NO_Ni100_bridge_Ndown.yaml")
            yaml_sync = "NO_Ni100_bridge_Ndown.yaml"
            cl = "Ni4_Ni9"
        elif site_key == "Ni4_Ni9_top_Ndown":
            dk = CalculationDeck.from_yaml(EXAMPLES_DIR / "NO_Ni100_top_Ndown.yaml")
            yaml_sync = "NO_Ni100_top_Ndown.yaml"
            cl = "Ni4_Ni9"
        elif site_key == "Ni5_1978_hollow_Ndown":
            dk = CalculationDeck.from_yaml(EXAMPLES_DIR / "NO_Ni5_1978_hollow_Ndown.yaml")
            yaml_sync = "NO_Ni5_1978_hollow_Ndown.yaml"
            cl = "Ni5_1978"
        elif site_key == "Ni5_1978_hollow_Odown":
            dk = CalculationDeck.from_yaml(EXAMPLES_DIR / "NO_Ni5_1978_hollow_Odown.yaml")
            yaml_sync = "NO_Ni5_1978_hollow_Odown.yaml"
            cl = "Ni5_1978"
        elif site_key == "Ni2_bridge_Ndown":
            dk = CalculationDeck.from_yaml(EXAMPLES_DIR / "NO_Ni2_bridge_Ndown.yaml")
            yaml_sync = "NO_Ni2_bridge_Ndown.yaml"
            cl = "Ni2"
        elif site_key == "Ni4_Ni9_bridge_Ndown_with_O_coads":
            a_ni, s_ni = _ni4_ni9_geometry(basis)
            no_and_o = [
                AtomConfig(id="N1", element="N", xyz=[0.0, 2.355, 2.35], active=True, fixed=False, basis=basis),
                AtomConfig(id="O1", element="O", xyz=[0.0, 2.355, 4.65], active=True, fixed=False, basis=basis),
                AtomConfig(id="O2", element="O", xyz=[0.0, 0.0, 1.25], active=True, fixed=False, basis=basis),
            ]
            dk = CalculationDeck(
                title="NO*(bridge) + O*(hollow) on Ferromagnetic Ni4/Ni9(100)",
                net_spin=7,
                control=_surface_ctrl(basis, has_spectators=True),
                geometry=a_ni + no_and_o + s_ni,
            )
            dk.control.vb_orbital_specs = build_surface_plaquette_vb_specs(dk)
            return dk, "Ni4_Ni9", None
        else:
            raise ValueError(site_key)

        if basis != "SZ":
            dk.control.default_basis = basis
            for a in dk.geometry:
                a.basis = basis
        return dk, cl, yaml_sync

    # Also record the FIXED 1978 d_Ni-X = 3.70 Bohr Ni5 single points for direct Table 7/8 comparison
    fixed_1978_coords = {
        "Ni5_1978_hollow_Ndown_fixed370": ([0.0, 0.0, 1.6118], [0.0, 0.0, 3.7865]),
        "Ni5_1978_hollow_Odown_fixed370": ([0.0, 0.0, 3.7865], [0.0, 0.0, 1.6118]),
    }
    for fixed_key, base_key in (
        ("Ni5_1978_hollow_Ndown_fixed370", "Ni5_1978_hollow_Ndown"),
        ("Ni5_1978_hollow_Odown_fixed370", "Ni5_1978_hollow_Odown"),
    ):
        if fixed_key not in sites["sites"]:
            t0 = time.perf_counter()
            dk_fx, cl, _ = _build_site_deck(base_key, "SZ")
            n_xyz_fx, o_xyz_fx = fixed_1978_coords[fixed_key]
            for at in dk_fx.geometry:
                if at.id == "N1":
                    at.xyz = list(n_xyz_fx)
                elif at.id == "O1":
                    at.xyz = list(o_xyz_fx)
            res_fx = drv.run_deck(dk_fx)
            e_cl = float(sites["clean_references"][f"{cl}_SZ"]["total_energy_ha"])
            summary = _surface_electronic_summary(res_fx, dk_fx, e_cl, e_gas_no_sz)
            summary["geometry_metrics"] = _extract_ads_geometry_metrics(dk_fx)
            summary["stationary_type"] = "fixed_1978_geometry"
            sites["sites"][fixed_key] = summary
            print(f"[Stage 2] {fixed_key}: E_ads={summary['e_ads_ev']:.3f} eV ({time.perf_counter()-t0:.2f}s)", flush=True)
            _save_json(sites_path, sites)

    # Optimize all Stage 2 sites in SZ + partial Hessian + harmonic adsorbate thermochemistry
    site_order = [
        "Ni2_bridge_Ndown",
        "Ni5_1978_hollow_Ndown",
        "Ni5_1978_hollow_Odown",
        "Ni4_Ni9_hollow_Ndown",
        "Ni4_Ni9_hollow_Ndown_tilted_test",
        "Ni4_Ni9_hollow_Odown",
        "Ni4_Ni9_bridge_Ndown",
        "Ni4_Ni9_top_Ndown",
        "Ni4_Ni9_bridge_Ndown_with_O_coads",
    ]
    for site_key in site_order:
        if site_key in sites["sites"]:
            continue
        t0 = time.perf_counter()
        dk0, cl, yaml_sync = _build_site_deck(site_key, "SZ")
        if site_key in ("Ni2_bridge_Ndown", "Ni4_Ni9_hollow_Ndown_tilted_test"):
            opt = optimize_geometry(
                dk0,
                driver=drv,
                max_steps=25,
                grad_tol=2.0e-4,
                energy_tol=1.0e-6,
                step_limit=0.15,
                quiet=True,
            )
        else:
            opt = optimize_geometry(
                dk0,
                driver=drv,
                max_steps=18,
                grad_tol=2.5e-4,
                energy_tol=1.0e-6,
                step_limit=0.12,
                quiet=True,
            )
        dk_opt = opt.optimized_deck
        if yaml_sync is not None:
            _update_yaml_no_coords(yaml_sync, dk_opt.geometry)

        warm_opt = opt.final_state.get("localized_orbitals_embedded", opt.final_state.get("localized_orbitals"))
        res_opt = drv.run_deck(dk_opt, warm_orbitals=warm_opt)
        if warm_opt is not None:
            dk_opt._vb_restart_orbitals = warm_opt
        e_clean_use = float(sites["clean_references"][f"{cl}_SZ"]["total_energy_ha"])
        clean_spin_mult = int(clean_net_spins[cl]) + 1
        if site_key == "Ni4_Ni9_bridge_Ndown_with_O_coads":
            # Reference clean O* on Ni4/Ni9 slab at the relaxed O2 coordinates for coadsorption E_ads(NO | O*)
            o2_xyz = [a.xyz for a in dk_opt.geometry if a.id == "O2"][0]
            dk_o_clean = _clean_ni4_ni9_with_O_deck(z_o=float(o2_xyz[2]))
            for a_c in dk_o_clean.geometry:
                if a_c.id == "O2":
                    a_c.xyz = list(o2_xyz)
            res_o_clean = drv.run_deck(dk_o_clean)
            e_clean_use = float(res_o_clean["total_energy"])
            clean_spin_mult = 9
            sites["clean_references"]["Ni4_Ni9_with_O_SZ"] = {
                "total_energy_ha": round(e_clean_use, 8),
                "converged": bool(res_o_clean["converged"]),
                "net_spin": 8,
            }

        summary = _surface_electronic_summary(res_opt, dk_opt, e_clean_use, e_gas_no_sz)
        summary["net_spin"] = int(dk_opt.net_spin)
        summary["geometry_metrics"] = _extract_ads_geometry_metrics(dk_opt)
        summary["opt_steps"] = int(opt.num_steps)
        summary["opt_converged"] = bool(opt.converged)
        summary["opt_max_grad"] = round(float(opt.trajectory[-1].grad_norm) if opt.trajectory else 0.0, 6)

        # Clamped-substrate partial Hessian & harmonic-adsorbate thermochemistry (is_adsorbate=True)
        spin_mult_ads = int(dk_opt.net_spin) + 1
        vib = compute_vibrations(
            dk_opt,
            driver=drv,
            step_size=0.01,
            method="forward",
            spin_multiplicity=spin_mult_ads,
            temperature_k=300.0,
            is_adsorbate=True,
            quiet=True,
        )
        all_freqs = sorted([round(float(m.frequency_cm), 2) for m in vib.vibrational_modes])
        real_freqs = [f for f in all_freqs if f > 0.0]
        nu_no = float(max(real_freqs)) if real_freqs else 0.0
        delta_nu = float(nu_no - nu_gas_no_sz)
        delta_nu_stage1 = float(nu_no - nu_gas_no_stage1_sz)
        th_ads = _thermo_at_temps(vib, [300.0, 700.0], spin_mult=spin_mult_ads, is_adsorbate=True)

        # Clean-slab electronic spin entropy -T * R * ln(2 S_clean + 1)
        g_clean_300 = e_clean_use - (300.0 * GAS_CONSTANT_R_CAL_MOL_K / 1000.0) * math.log(max(1, clean_spin_mult)) / HARTREE_TO_KCAL
        g_clean_700 = e_clean_use - (700.0 * GAS_CONSTANT_R_CAL_MOL_K / 1000.0) * math.log(max(1, clean_spin_mult)) / HARTREE_TO_KCAL
        zpve_ads_kcal = float(vib.thermochemistry.zero_point_energy_kcal)
        dg_300_kcal = (th_ads["300K"]["total_gibbs_free_energy_g_t"] - g_clean_300 - th_gas_300) * HARTREE_TO_KCAL_MOL
        dg_700_kcal = (th_ads["700K"]["total_gibbs_free_energy_g_t"] - g_clean_700 - th_gas_700) * HARTREE_TO_KCAL_MOL

        summary["vibrations"] = {
            "num_modes": int(vib.num_modes),
            "num_vibrational_modes": int(vib.num_vibrational_modes),
            "num_imaginary": int(vib.num_imaginary),
            "frequencies_cm": all_freqs,
            "nu_no_cm": round(nu_no, 2),
            "delta_nu_no_cm": round(delta_nu, 2),
            "delta_nu_vs_stage1_cas76_cm": round(delta_nu_stage1, 2),
            "zpve_kcal_mol": round(zpve_ads_kcal, 4),
            "delta_zpve_kcal_mol": round(zpve_ads_kcal - zpve_gas_kcal, 4),
        }
        summary["thermochemistry"] = {
            "300K": th_ads["300K"],
            "700K": th_ads["700K"],
            "clean_spin_multiplicity": int(clean_spin_mult),
            "ads_spin_multiplicity": int(spin_mult_ads),
            "delta_g_ads_300K_kcal_mol": round(dg_300_kcal, 3),
            "delta_g_ads_700K_kcal_mol": round(dg_700_kcal, 3),
        }
        sites["sites"][site_key] = summary
        print(
            f"[Stage 2] {site_key}: E_ads={summary['e_ads_ev']:.3f} eV | R_NO={summary['geometry_metrics']['r_no_bohr']:.4f} | nu_NO={nu_no:.1f} ({delta_nu:+.1f}) cm^-1 | n_imag={vib.num_imaginary} ({time.perf_counter()-t0:.2f}s)",
            flush=True,
        )
        _save_json(sites_path, sites)

    # Spin-quenching comparison at relaxed Ni4/Ni9 hollow N-down (S=7/2 antiparallel vs S=9/2 parallel)
    if "magnetic_quenching_check_hollow_Ndown" not in sites:
        t0 = time.perf_counter()
        dk_par, _, _ = _build_site_deck("Ni4_Ni9_hollow_Ndown", "SZ")
        gm_h = sites["sites"]["Ni4_Ni9_hollow_Ndown"]["geometry_metrics"]
        for at in dk_par.geometry:
            if at.id == "N1":
                at.xyz = list(gm_h["n_xyz_bohr"])
            elif at.id == "O1":
                at.xyz = list(gm_h["o_xyz_bohr"])
        dk_par.net_spin = 9
        dk_par.control.target_spin = 9
        res_par = drv.run_deck(dk_par)
        e_anti = float(sites["sites"]["Ni4_Ni9_hollow_Ndown"]["total_energy_ha"])
        e_par = float(res_par["total_energy"])
        sites["magnetic_quenching_check_hollow_Ndown"] = {
            "e_antiparallel_S7_ha": round(e_anti, 8),
            "e_parallel_S9_ha": round(e_par, 8),
            "parallel_converged": bool(res_par["converged"]),
            "spin_pairing_stabilization_ev": round((e_par - e_anti) * HARTREE_TO_EV, 4),
        }
        print(f"[Stage 2] Magnetic quenching check (S=7/2 vs 9/2): dE={sites['magnetic_quenching_check_hollow_Ndown']['spin_pairing_stabilization_ev']:+.4f} eV ({time.perf_counter()-t0:.2f}s)", flush=True)
        _save_json(sites_path, sites)

    # SZP (and memory-safe DZP / locally-dense DZP_NO_SZP_Ni) single-point refinements
    # Using plaquette-matched gas NO reference at the matching basis level
    szp_targets = [
        ("Ni2_bridge_Ndown", "Ni2"),
        ("Ni4_Ni9_hollow_Ndown", "Ni4_Ni9"),
        ("Ni4_Ni9_hollow_Odown", "Ni4_Ni9"),
        ("Ni4_Ni9_bridge_Ndown", "Ni4_Ni9"),
        ("Ni4_Ni9_top_Ndown", "Ni4_Ni9"),
    ]
    e_gas_szp = float(matched_refs["SZP"]["total_energy_ha"])
    sp_szp = sites["single_point_refinements"].setdefault("SZP", {})
    for sk, cl in szp_targets:
        clean_key = f"{cl}_SZP"
        if clean_key not in sites["clean_references"]:
            t0 = time.perf_counter()
            dk_cl = _clean_deck(cl, "SZP")
            res_cl = drv.run_deck(dk_cl)
            sites["clean_references"][clean_key] = {
                "total_energy_ha": round(float(res_cl["total_energy"]), 8),
                "converged": bool(res_cl["converged"]),
                "net_spin": clean_net_spins[cl],
            }
            print(f"[Stage 2] Clean {clean_key}: {float(res_cl['total_energy']):.8f} Ha ({time.perf_counter()-t0:.2f}s)", flush=True)
            _save_json(sites_path, sites)
        if sk in sp_szp and sp_szp[sk].get("converged") is True:
            continue
        t0 = time.perf_counter()
        dk_sp, _, _ = _build_site_deck(sk, "SZP")
        dk_sp.control.max_cycles = 1500
        gm = sites["sites"][sk]["geometry_metrics"]
        for at in dk_sp.geometry:
            if at.id == "N1":
                at.xyz = list(gm["n_xyz_bohr"])
            elif at.id == "O1":
                at.xyz = list(gm["o_xyz_bohr"])
        res_sp = drv.run_deck(dk_sp)
        e_sp = float(res_sp["total_energy"])
        e_cl_b = float(sites["clean_references"][clean_key]["total_energy_ha"])
        e_ads_sp = float(e_sp - e_cl_b - e_gas_szp)
        sp_szp[sk] = {
            "total_energy_ha": round(e_sp, 8),
            "converged": bool(res_sp["converged"]),
            "e_ads_ha": round(e_ads_sp, 6),
            "e_ads_ev": round(e_ads_sp * HARTREE_TO_EV, 4),
            "e_ads_kcal_mol": round(e_ads_sp * HARTREE_TO_KCAL_MOL, 3),
        }
        print(f"[Stage 2] {sk}//SZP: E_ads={e_ads_sp*HARTREE_TO_EV:.3f} eV conv={res_sp['converged']} ({time.perf_counter()-t0:.2f}s)", flush=True)
        _save_json(sites_path, sites)

    if not skip_dzp_slab:
        e_gas_dzp = float(matched_refs["DZP"]["total_energy_ha"])
        # 2) Full uniform DZP (including Ni 3d shells, Amendment 4 oracle) on Ni2_bridge_Ndown
        if "Ni2_DZP" not in sites["clean_references"] or sites["clean_references"]["Ni2_DZP"].get("converged") is not True:
            t0 = time.perf_counter()
            dk_cl = _clean_deck("Ni2", "DZP")
            res_cl = drv.run_deck(dk_cl)
            sites["clean_references"]["Ni2_DZP"] = {
                "total_energy_ha": round(float(res_cl["total_energy"]), 8),
                "converged": bool(res_cl["converged"]),
                "net_spin": clean_net_spins["Ni2"],
            }
            print(f"[Stage 2] Clean Ni2_DZP: {float(res_cl['total_energy']):.8f} Ha ({time.perf_counter()-t0:.2f}s)", flush=True)
            _save_json(sites_path, sites)
        sp_dzp = sites["single_point_refinements"].setdefault("DZP", {})
        if "Ni2_bridge_Ndown" not in sp_dzp or sp_dzp["Ni2_bridge_Ndown"].get("converged") is not True:
            t0 = time.perf_counter()
            dk_sp, _, _ = _build_site_deck("Ni2_bridge_Ndown", "DZP")
            dk_sp.control.max_cycles = 1500
            gm = sites["sites"]["Ni2_bridge_Ndown"]["geometry_metrics"]
            for at in dk_sp.geometry:
                if at.id == "N1":
                    at.xyz = list(gm["n_xyz_bohr"])
                elif at.id == "O1":
                    at.xyz = list(gm["o_xyz_bohr"])
            res_sp = drv.run_deck(dk_sp)
            e_sp = float(res_sp["total_energy"])
            e_cl_ni2_dzp = float(sites["clean_references"]["Ni2_DZP"]["total_energy_ha"])
            e_ads_sp = float(e_sp - e_cl_ni2_dzp - e_gas_dzp)
            sp_dzp["Ni2_bridge_Ndown"] = {
                "total_energy_ha": round(e_sp, 8),
                "converged": bool(res_sp["converged"]),
                "e_ads_ha": round(e_ads_sp, 6),
                "e_ads_ev": round(e_ads_sp * HARTREE_TO_EV, 4),
                "e_ads_kcal_mol": round(e_ads_sp * HARTREE_TO_KCAL_MOL, 3),
            }
            print(f"[Stage 2] Ni2_bridge_Ndown//DZP: E_ads={e_ads_sp*HARTREE_TO_EV:.3f} eV conv={res_sp['converged']} ({time.perf_counter()-t0:.2f}s)", flush=True)
            _save_json(sites_path, sites)

        # 3) Locally-dense DZP_NO_SZP_Ni (DZP on NO, SZP on active Ni4, SZ on spectator Ni9; ~800 MB)
        ld_clean_key = "Ni4_Ni9_SZP_Ni_SZ_spec"
        if ld_clean_key not in sites["clean_references"] or sites["clean_references"][ld_clean_key].get("converged") is not True:
            t0 = time.perf_counter()
            dk_cl = _clean_deck("Ni4_Ni9", "SZ")
            dk_cl.title = "Clean Ferromagnetic Ni4(SZP)/Ni9(SZ)"
            dk_cl.control.default_basis = "SZP"
            for at in dk_cl.geometry:
                at.basis = "SZP" if at.is_active else "SZ"
            res_cl = drv.run_deck(dk_cl)
            sites["clean_references"][ld_clean_key] = {
                "total_energy_ha": round(float(res_cl["total_energy"]), 8),
                "converged": bool(res_cl["converged"]),
                "net_spin": clean_net_spins["Ni4_Ni9"],
            }
            print(f"[Stage 2] Clean {ld_clean_key}: {float(res_cl['total_energy']):.8f} Ha ({time.perf_counter()-t0:.2f}s)", flush=True)
            _save_json(sites_path, sites)
        e_cl_ld = float(sites["clean_references"][ld_clean_key]["total_energy_ha"])
        sp_ld = sites["single_point_refinements"].setdefault("DZP_NO_SZP_Ni", {})
        for sk in ("Ni4_Ni9_hollow_Ndown", "Ni4_Ni9_hollow_Odown", "Ni4_Ni9_bridge_Ndown", "Ni4_Ni9_top_Ndown"):
            if sk in sp_ld and sp_ld[sk].get("converged") is True:
                continue
            t0 = time.perf_counter()
            gm = sites["sites"][sk]["geometry_metrics"]
            dk_sz_warm, _, _ = _build_site_deck(sk, "SZ")
            for at in dk_sz_warm.geometry:
                if at.id == "N1":
                    at.xyz = list(gm["n_xyz_bohr"])
                elif at.id == "O1":
                    at.xyz = list(gm["o_xyz_bohr"])
            res_sz_warm = drv.run_deck(dk_sz_warm)
            dk_sp, _, _ = _build_site_deck(sk, "SZ")
            dk_sp.control.default_basis = "SZP"
            dk_sp.control.max_cycles = 2500
            for at in dk_sp.geometry:
                if at.id == "N1":
                    at.xyz = list(gm["n_xyz_bohr"])
                    at.basis = "DZP"
                elif at.id == "O1":
                    at.xyz = list(gm["o_xyz_bohr"])
                    at.basis = "DZP"
                elif at.is_active:
                    at.basis = "SZP"
                else:
                    at.basis = "SZ"
            warm_d = _project_warm_orbitals_cross_basis(dk_sz_warm, res_sz_warm, dk_sp)
            res_sp = drv.run_deck(dk_sp, warm_orbitals=warm_d)
            e_sp = float(res_sp["total_energy"])
            e_ads_sp = float(e_sp - e_cl_ld - e_gas_dzp)
            sp_ld[sk] = {
                "total_energy_ha": round(e_sp, 8),
                "converged": bool(res_sp["converged"]),
                "e_ads_ha": round(e_ads_sp, 6),
                "e_ads_ev": round(e_ads_sp * HARTREE_TO_EV, 4),
                "e_ads_kcal_mol": round(e_ads_sp * HARTREE_TO_KCAL_MOL, 3),
            }
            print(f"[Stage 2] {sk}//DZP_NO_SZP_Ni: E_ads={e_ads_sp*HARTREE_TO_EV:.3f} eV conv={res_sp['converged']} ({time.perf_counter()-t0:.2f}s)", flush=True)
            _save_json(sites_path, sites)

    # Mandatory Step 0.4 CVS-Delta-CASSCF & Koopmans Core-Level Binding Energies
    # (Gas NO Stage 1 + plaquette-matched Gas NO + Ni4/Ni9 hollow N-down vs O-down + NO*+O* + 1978 Ni5 + Ni2)
    core_targets = [
        "Ni2_bridge_Ndown",
        "Ni5_1978_hollow_Ndown_fixed370",
        "Ni5_1978_hollow_Odown_fixed370",
        "Ni4_Ni9_hollow_Ndown",
        "Ni4_Ni9_hollow_Odown",
        "Ni4_Ni9_bridge_Ndown_with_O_coads",
    ]
    gas_n1s_s1 = stage1_data["core_1s_gas"]["NO_SZ"]["N1"]
    gas_o1s_s1 = stage1_data["core_1s_gas"]["NO_SZ"]["O1"]
    cores["systems"]["NO_gas_SZ"] = {
        "N1": gas_n1s_s1,
        "O1": gas_o1s_s1,
        "N1s_koopmans_ev": stage1_data["species"]["NO"]["SZ"]["koopmans_1s_be_ev"].get("N1"),
        "O1s_koopmans_ev": stage1_data["species"]["NO"]["SZ"]["koopmans_1s_be_ev"].get("O1"),
    }
    if "NO_gas_SZ_plaquette_matched" not in cores["systems"]:
        n1s_m = compute_delta_casscf_core_1s_be(dk_gas_no_sz, "N1", warm_ground_state=res_gas_no_sz)
        o1s_m = compute_delta_casscf_core_1s_be(dk_gas_no_sz, "O1", warm_ground_state=res_gas_no_sz)
        cores["systems"]["NO_gas_SZ_plaquette_matched"] = {
            "N1": n1s_m,
            "O1": o1s_m,
            "N1s_koopmans_ev": no_matched_sz["koopmans_1s_be_ev"].get("N1"),
            "O1s_koopmans_ev": no_matched_sz["koopmans_1s_be_ev"].get("O1"),
        }
    gas_n1s_ref = cores["systems"]["NO_gas_SZ_plaquette_matched"]["N1"]
    gas_o1s_ref = cores["systems"]["NO_gas_SZ_plaquette_matched"]["O1"]
    _save_json(core_path, cores)

    for sk in core_targets:
        prev_rec = cores["systems"].get(sk, {})
        prev_n1 = prev_rec.get("N1", {})
        prev_o1 = prev_rec.get("O1", {})
        if prev_n1.get("converged") is True and prev_o1.get("converged") is True:
            continue
        t0 = time.perf_counter()
        base_name = sk.replace("_fixed370", "")
        dk_c, _, _ = _build_site_deck(base_name, "SZ")
        gm = sites["sites"][sk]["geometry_metrics"]
        for at in dk_c.geometry:
            if at.id == "N1":
                at.xyz = list(gm["n_xyz_bohr"])
            elif at.id == "O1":
                at.xyz = list(gm["o_xyz_bohr"])
            elif at.id == "O2" and "o2_xyz_bohr" in gm:
                at.xyz = list(gm["o2_xyz_bohr"])
        res_gs = drv.run_deck(dk_c)
        n1s_res = (
            prev_n1
            if prev_n1.get("converged") is True
            else compute_delta_casscf_core_1s_be(dk_c, "N1", warm_ground_state=res_gs, maxiter=500, gtol=1.0e-4)
        )
        o1s_res = (
            prev_o1
            if prev_o1.get("converged") is True
            else compute_delta_casscf_core_1s_be(dk_c, "O1", warm_ground_state=res_gs, maxiter=500, gtol=1.0e-4)
        )
        scls_koop = (
            prev_rec["koopmans_scls"]
            if isinstance(prev_rec.get("koopmans_scls"), dict)
            else compute_surface_core_level_shifts(res_gs, driver=drv, gas_state=res_gas_no_sz).to_dict()
        )
        cores["systems"][sk] = {
            "N1": n1s_res,
            "O1": o1s_res,
            "delta_N1s_vs_gas_cvs_ev": round(n1s_res["be_multiplet_avg_ev"] - gas_n1s_ref["be_multiplet_avg_ev"], 4),
            "delta_O1s_vs_gas_cvs_ev": round(o1s_res["be_multiplet_avg_ev"] - gas_o1s_ref["be_multiplet_avg_ev"], 4),
            "delta_N1s_vs_stage1_cas76_cvs_ev": round(n1s_res["be_multiplet_avg_ev"] - gas_n1s_s1["be_multiplet_avg_ev"], 4),
            "delta_O1s_vs_stage1_cas76_cvs_ev": round(o1s_res["be_multiplet_avg_ev"] - gas_o1s_s1["be_multiplet_avg_ev"], 4),
            "koopmans_scls": scls_koop,
        }
        print(
            f"[Stage 2 Core-1s] {sk}: N1s(CVS)={n1s_res['be_multiplet_avg_ev']:.3f} eV (conv={n1s_res['converged']}, Δ={cores['systems'][sk]['delta_N1s_vs_gas_cvs_ev']:+.3f}) | "
            f"O1s(CVS)={o1s_res['be_multiplet_avg_ev']:.3f} eV (conv={o1s_res['converged']}, Δ={cores['systems'][sk]['delta_O1s_vs_gas_cvs_ev']:+.3f}) ({time.perf_counter()-t0:.2f}s)",
            flush=True,
        )
        _save_json(core_path, cores)

    return sites, cores


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NO on Ni(100) Stages 1-2 computational study.")
    parser.add_argument("--stage", choices=("all", "1", "2"), default="all")
    parser.add_argument("--force", action="store_true", help="Recompute existing cached JSON entries.")
    parser.add_argument("--skip-dzp-slab", action="store_true", help="Skip DZP Ni4/Ni9 single-point refinements.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage in ("all", "1"):
        stage1 = run_stage1_gas_benchmarks(force=args.force)
    else:
        stage1 = _load_json(OUT_DIR / "stage1_gas_benchmarks.json")
    if args.stage in ("all", "2"):
        run_stage2_adsorption(stage1, force=args.force, skip_dzp_slab=args.skip_dzp_slab)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
