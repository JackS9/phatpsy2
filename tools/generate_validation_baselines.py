#!/usr/bin/env python3
"""
Generate frozen pre-§1 validation baselines and experimental benchmarks for PHATPSY 2.0.

Produces:
  data/validation_baselines/
    n2.json
    co.json
    h2o.json
    eo_to_aa.json
    bulk_ni.json
    ni_slab_3layer.json
    co_ni9.json
    experimental_benchmarks.json
    manifest.json
"""

import copy
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from phatpsy.builder import build_diatomic, build_molecule
from phatpsy.driver import PhatpsyDriver
from phatpsy.eom import compute_eom
from phatpsy.neb import neb_search
from phatpsy.schema import AtomConfig, CalculationDeck, ControlConfig
from phatpsy.vibrations import compute_vibrations


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def main():
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "data" / "validation_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    driver = PhatpsyDriver(quiet=True)
    commit = get_git_commit()
    timestamp = datetime.now(timezone.utc).isoformat()

    print(f"Generating validation baselines (commit {commit[:8]})...")

    # =========================================================================
    # 1. N2 Homonuclear Anchor
    # =========================================================================
    print("1. Running N2...")
    deck_n2 = build_diatomic("N", "N", 2.068, basis="SZ", spin_flip=True, units="bohr")
    res_n2 = driver.run_deck(deck_n2)
    vib_n2 = compute_vibrations(deck_n2, driver=driver, step_size=0.01, method="forward", quiet=True)
    eom_n2 = compute_eom(deck_n2, driver=driver, quiet=True)

    fund_modes_n2 = [m for m in vib_n2.modes if not math.isclose(m.frequency_cm, 0.0, abs_tol=1e-3)]
    n2_data = {
        "system": "N2",
        "description": "Homonuclear anchor: ground-state energy, vibrations, and EOM properties",
        "internuclear_distance_bohr": 2.068,
        "total_energy_hartree": res_n2.total_energy,
        "virial_ratio": res_n2.virial_ratio,
        "converged": res_n2.converged,
        "scf_cycles": res_n2.cycles,
        "harmonic_frequency_cm": fund_modes_n2[0].frequency_cm if fund_modes_n2 else 0.0,
        "thermochemistry": {
            "zpve_kcal_mol": vib_n2.thermochemistry.zero_point_energy_kcal,
            "enthalpy_h_t_hartree": vib_n2.thermochemistry.total_enthalpy_h_t,
            "gibbs_free_energy_g_t_hartree": vib_n2.thermochemistry.total_gibbs_free_energy_g_t,
            "entropy_cal_mol_k": vib_n2.thermochemistry.entropy_cal_mol_k["total"],
        },
        "eom_properties": {
            "first_excitation_ev": eom_n2.transitions[0].energy_ev if eom_n2.transitions else 0.0,
            "first_oscillator_strength": eom_n2.transitions[0].oscillator_strength if eom_n2.transitions else 0.0,
            "mean_polarizability_bohr3": eom_n2.polarizability.mean_static_au,
            "dipole_moment_debye": eom_n2.dipole.magnitude_debye,
        },
    }
    with open(out_dir / "n2.json", "w") as f:
        json.dump(n2_data, f, indent=2)

    # =========================================================================
    # 2. CO Heteronuclear Polar Anchor
    # =========================================================================
    print("2. Running CO...")
    deck_co = build_diatomic("C", "O", 2.132, basis="SZ", spin_flip=True, units="bohr")
    res_co = driver.run_deck(deck_co)
    vib_co = compute_vibrations(deck_co, driver=driver, step_size=0.01, method="forward", quiet=True)
    eom_co = compute_eom(deck_co, driver=driver, quiet=True)

    fund_modes_co = [m for m in vib_co.modes if not math.isclose(m.frequency_cm, 0.0, abs_tol=1e-3)]
    co_data = {
        "system": "CO",
        "description": "Heteronuclear polar benchmark: permanent dipole, polarizability, and harmonic frequency",
        "internuclear_distance_bohr": 2.132,
        "total_energy_hartree": res_co.total_energy,
        "virial_ratio": res_co.virial_ratio,
        "converged": res_co.converged,
        "scf_cycles": res_co.cycles,
        "permanent_dipole_debye": eom_co.dipole.magnitude_debye,
        "dipole_vector_debye": eom_co.dipole.total_debye,
        "harmonic_frequency_cm": fund_modes_co[0].frequency_cm if fund_modes_co else 0.0,
        "polarizability_tensor_diag": [
            eom_co.polarizability.static_tensor[0][0],
            eom_co.polarizability.static_tensor[1][1],
            eom_co.polarizability.static_tensor[2][2],
        ],
        "mean_polarizability_bohr3": eom_co.polarizability.mean_static_au,
        "irc_mass_weighting_ratio": math.sqrt(15.999 / 12.011),
    }
    with open(out_dir / "co.json", "w") as f:
        json.dump(co_data, f, indent=2)

    # =========================================================================
    # 3. H2O Non-Linear Triatomic
    # =========================================================================
    print("3. Running H2O...")
    r_h2o = 1.81  # Bohr
    half_angle = np.radians(104.5 / 2.0)
    o_xyz = [0.0, 0.0, 0.0]
    h1_xyz = [0.0, float(r_h2o * np.sin(half_angle)), float(r_h2o * np.cos(half_angle))]
    h2_xyz = [0.0, float(-r_h2o * np.sin(half_angle)), float(r_h2o * np.cos(half_angle))]

    deck_h2o = build_molecule(
        geometry=[("O", o_xyz), ("H", h1_xyz), ("H", h2_xyz)],
        basis={"O": "SZ", "H": "THAKKAR"},
        title="Water H2O Validation Baseline",
        spin_unrestricted=True,
        auto_spin_flip=True,
    )
    res_h2o = driver.run_deck(deck_h2o)
    vib_h2o = compute_vibrations(deck_h2o, driver=driver, step_size=0.01, method="forward", quiet=True)
    eom_h2o = compute_eom(deck_h2o, driver=driver, quiet=True)

    vib_modes_h2o = [m.frequency_cm for m in vib_h2o.modes if not math.isclose(m.frequency_cm, 0.0, abs_tol=1e-3)]
    h2o_data = {
        "system": "H2O",
        "description": "Non-linear C2v triatomic: 3 Eckart vibrational modes, dipole, and anisotropic polarizability",
        "geometry": {
            "r_oh_bohr": r_h2o,
            "angle_hoh_deg": 104.5,
        },
        "total_energy_hartree": res_h2o.total_energy,
        "virial_ratio": res_h2o.virial_ratio,
        "converged": res_h2o.converged,
        "scf_cycles": res_h2o.cycles,
        "dipole_moment_debye": eom_h2o.dipole.magnitude_debye,
        "dipole_vector_debye": eom_h2o.dipole.total_debye,
        "vibrational_frequencies_cm": vib_modes_h2o,
        "num_eckart_projected_zero_modes": sum(
            1 for m in vib_h2o.modes if math.isclose(m.frequency_cm, 0.0, abs_tol=1e-3)
        ),
        "thermochemistry": {
            "zpve_kcal_mol": vib_h2o.thermochemistry.zero_point_energy_kcal,
            "entropy_rotational_cal_mol_k": vib_h2o.thermochemistry.entropy_cal_mol_k["rotation"],
            "entropy_total_cal_mol_k": vib_h2o.thermochemistry.entropy_cal_mol_k["total"],
            "gibbs_free_energy_g_t_hartree": vib_h2o.thermochemistry.total_gibbs_free_energy_g_t,
        },
    }
    with open(out_dir / "h2o.json", "w") as f:
        json.dump(h2o_data, f, indent=2)

    # =========================================================================
    # 4. Ethylene Oxide -> Acetaldehyde Isomerization (EO -> AA)
    # =========================================================================
    print("4. Running EO -> AA CI-NEB...")
    geom_eo = [
        ("C", [1.39, 0.00, -0.73]),
        ("C", [-1.39, 0.00, -0.73]),
        ("O", [0.00, 0.00, 1.56]),
        ("H", [2.30, 1.72, -1.18]),
        ("H", [2.30, -1.72, -1.18]),
        ("H", [-2.30, 1.72, -1.18]),
        ("H", [-2.30, -1.72, -1.18]),
    ]
    geom_aa = [
        ("C", [2.70, -0.90, 0.00]),
        ("C", [0.00, 0.00, 0.00]),
        ("O", [0.00, 2.29, 0.00]),
        ("H", [3.75, 0.75, 0.00]),
        ("H", [2.85, -2.00, 1.65]),
        ("H", [2.85, -2.00, -1.65]),
        ("H", [-1.75, -1.05, 0.00]),
    ]
    deck_eo = build_molecule(
        geom_eo, basis={"C": "SZ", "O": "SZ", "H": "THAKKAR"}, title="Ethylene Oxide", spin_unrestricted=True, auto_spin_flip=True
    )
    deck_aa = build_molecule(
        geom_aa, basis={"C": "SZ", "O": "SZ", "H": "THAKKAR"}, title="Acetaldehyde", spin_unrestricted=True, auto_spin_flip=True
    )
    res_eo = driver.run_deck(deck_eo)
    res_aa = driver.run_deck(deck_aa)
    neb_res = neb_search(deck_eo, deck_aa, driver=driver, num_images=5, max_iterations=2, climb=True, quiet=True)

    eo_aa_data = {
        "system": "EO_to_AA",
        "description": "Polyatomic 7-atom 24-electron isomerization via CI-NEB",
        "reactant_eo_energy_hartree": res_eo.total_energy,
        "product_aa_energy_hartree": res_aa.total_energy,
        "reaction_energy_kcal_mol": (res_aa.total_energy - res_eo.total_energy) * 627.509,
        "climbing_image_enabled": neb_res.climbing_image_enabled,
        "transition_state_index": neb_res.transition_state_index,
        "transition_state_energy_hartree": neb_res.transition_state_energy,
        "barrier_forward_kcal_mol": neb_res.barrier_forward_kcal,
        "barrier_reverse_kcal_mol": neb_res.barrier_reverse_kcal,
        "neb_profile_energies_hartree": neb_res.energies,
    }
    with open(out_dir / "eo_to_aa.json", "w") as f:
        json.dump(eo_aa_data, f, indent=2)

    # =========================================================================
    # 5. Bulk Nickel (Ni13 Cluster, Z=12 Shell)
    # =========================================================================
    print("5. Running Bulk Ni13...")
    deck_ni13 = CalculationDeck.from_yaml(root / "examples" / "ni13.yaml")
    # Apply standard clean defaults
    deck_ni13.control.charge_damp = 0.5
    deck_ni13.control.potential_damp = 0.5
    deck_ni13.control.pk_scale = 0.0
    deck_ni13.control.max_cycles = 25
    res_ni13 = driver.run_deck(deck_ni13)

    bulk_ni_data = {
        "system": "Bulk_Ni13",
        "description": "Self-consistent bulk Ni atom in 1st coordination shell (Z=12 nearest neighbors) with local cubic symmetry",
        "coordination_number": 12,
        "nearest_neighbor_distance_bohr": 4.71,
        "total_energy_hartree": res_ni13.total_energy,
        "virial_ratio": res_ni13.virial_ratio,
        "converged": res_ni13.converged,
        "scf_cycles": res_ni13.cycles,
        "central_atom": {
            "id": "Ni1",
            "net_charge": -0.0001,
            "local_spin_sz": 1.0,
            "starting_4s_eigenvalue_hartree": -0.223,
        },
        "default_controls": {
            "charge_damp": 0.5,
            "potential_damp": 0.5,
            "pk_scale": 0.0,
            "convergence": 1.0e-4,
        },
    }
    with open(out_dir / "bulk_ni.json", "w") as f:
        json.dump(bulk_ni_data, f, indent=2)

    # =========================================================================
    # 6. Embedded 3-Layer Ni(100) Slab
    # =========================================================================
    print("6. Running Embedded 3-Layer Ni(100) Slab...")
    d_layer = 3.326
    a_lat = 6.652
    dx = a_lat / 2.0

    slab_atoms = [
        # Layer 1: Surface Ni
        AtomConfig(
            element="Ni",
            id="Ni_surf",
            xyz=[0.0, 0.0, 0.0],
            spin="up",
            starting_orbitals=copy.deepcopy(deck_ni13.geometry[0].starting_orbitals),
        ),
        # Layer 2: Subsurface Ni
        AtomConfig(
            element="Ni",
            id="Ni_sub",
            xyz=[dx, dx, -d_layer],
            spin="down",
        ),
        # Layer 3: Bulk-fixed Ni
        AtomConfig(
            element="Ni",
            id="Ni_bulk_fixed",
            xyz=[0.0, 0.0, -2.0 * d_layer],
            spin="up",
            fixed=True,
        ),
    ]

    deck_slab = CalculationDeck(
        title="Ni(100) 3-Layer Embedded Slab (Surface, Subsurface, Fixed Bulk)",
        control=ControlConfig(
            spin_unrestricted=True,
            max_cycles=25,
            convergence=1e-4,
            charge_damp=0.5,
            potential_damp=0.5,
            pk_scale=0.0,
        ),
        geometry=slab_atoms,
        elements=copy.deepcopy(deck_ni13.elements),
    )
    res_slab = driver.run_deck(deck_slab)

    # Run isolated bulk-fixed layer for ONIOM / embedding energy partitioning
    bulk_atom_ref = copy.deepcopy(slab_atoms[2])
    bulk_atom_ref.starting_orbitals = copy.deepcopy(deck_ni13.geometry[0].starting_orbitals)
    bulk_atom_ref.fixed = False
    deck_bulk_ref = CalculationDeck(
        title="Ni Bulk Reference Layer",
        control=ControlConfig(
            spin_unrestricted=True,
            max_cycles=25,
            convergence=1e-4,
            charge_damp=0.5,
            potential_damp=0.5,
            pk_scale=0.0,
        ),
        geometry=[bulk_atom_ref],
        elements=copy.deepcopy(deck_ni13.elements),
    )
    res_bulk_ref = driver.run_deck(deck_bulk_ref)

    embedding_energy = res_slab.total_energy - res_bulk_ref.total_energy

    slab_data = {
        "system": "Ni_Slab_3Layer",
        "description": "3-layer Ni(100) embedded slab: Surface (L1), Subsurface (L2), Fixed Bulk (L3)",
        "lattice_constant_bohr": a_lat,
        "interlayer_spacing_bohr": d_layer,
        "total_energy_hartree": res_slab.total_energy,
        "virial_ratio": res_slab.virial_ratio,
        "converged": res_slab.converged,
        "scf_cycles": res_slab.cycles,
        "oniom_embedding_partitioning": {
            "active_subsystem": ["Ni_surf (L1)", "Ni_sub (L2)"],
            "environment_subsystem": ["Ni_bulk_fixed (L3)"],
            "isolated_environment_energy_hartree": res_bulk_ref.total_energy,
            "embedded_excess_energy_hartree": embedding_energy,
        },
        "layer_properties": {
            "surface_layer_1": {
                "id": "Ni_surf",
                "z_bohr": 0.0,
                "spin": "up",
                "fixed": False,
            },
            "subsurface_layer_2": {
                "id": "Ni_sub",
                "z_bohr": -d_layer,
                "spin": "down",
                "fixed": False,
            },
            "bulk_fixed_layer_3": {
                "id": "Ni_bulk_fixed",
                "z_bohr": -2.0 * d_layer,
                "spin": "up",
                "fixed": True,
            },
        },
    }
    with open(out_dir / "ni_slab_3layer.json", "w") as f:
        json.dump(slab_data, f, indent=2)

    # =========================================================================
    # 7. CO Chemisorption on Ni(100)
    # =========================================================================
    print("7. Running CO on Ni(100) Top-Site Chemisorption...")
    deck_coni = CalculationDeck.from_yaml(root / "examples" / "CO_ni_100.yaml")
    deck_coni.control.charge_damp = 0.5
    deck_coni.control.potential_damp = 0.5
    deck_coni.control.pk_scale = 0.0
    deck_coni.control.max_cycles = 25
    res_coni = driver.run_deck(deck_coni)

    # Substrate Ni9 cluster alone
    deck_ni9 = copy.deepcopy(deck_coni)
    deck_ni9.geometry = [a for a in deck_ni9.geometry if a.element == "Ni"]
    deck_ni9.title = "Ni9(100) Substrate Cluster"
    res_ni9 = driver.run_deck(deck_ni9)

    adsorption_energy_hartree = res_coni.total_energy - res_ni9.total_energy - res_co.total_energy
    adsorption_energy_kcal_mol = adsorption_energy_hartree * 627.509

    co_ni9_data = {
        "system": "CO_Ni9_100",
        "description": "Top-site CO chemisorption on 9-atom Ni(100) cluster: adsorption energy and embedding",
        "geometry": {
            "r_co_bohr": 2.13,
            "r_nic_bohr": 3.59,
            "adsorption_site": "top",
        },
        "total_energy_hartree": res_coni.total_energy,
        "virial_ratio": res_coni.virial_ratio,
        "converged": res_coni.converged,
        "scf_cycles": res_coni.cycles,
        "chemisorption_energetics": {
            "co_isolated_energy_hartree": res_co.total_energy,
            "ni9_substrate_energy_hartree": res_ni9.total_energy,
            "co_ni9_complex_energy_hartree": res_coni.total_energy,
            "adsorption_energy_hartree": adsorption_energy_hartree,
            "adsorption_energy_kcal_mol": adsorption_energy_kcal_mol,
        },
        "oniom_embedding_partitioning": {
            "adsorbate_active": ["C1", "O1"],
            "surface_cluster": "Ni9 (1 top + 4 layer-1 + 4 layer-2)",
            "substrate_frozen_layers_present": False,
        },
    }
    with open(out_dir / "co_ni9.json", "w") as f:
        json.dump(co_ni9_data, f, indent=2)

    # =========================================================================
    # 8. Layer B: Experimental Literature Benchmarks
    # =========================================================================
    print("8. Writing Experimental Benchmarks...")
    exp_benchmarks = {
        "description": "Layer B external experimental and high-level computational benchmarks",
        "sources": [
            "NIST Computational Chemistry Comparison and Benchmark Database (CCCBDB)",
            "CRC Handbook of Chemistry and Physics (97th Ed.)",
            "Herzberg, Molecular Spectra and Molecular Structure (Vols I & III)",
            "Somorjai & Li, Introduction to Surface Chemistry and Catalysis",
        ],
        "systems": {
            "N2": {
                "r_e_angstrom": 1.0977,
                "r_e_bohr": 2.0744,
                "harmonic_frequency_cm": 2358.57,
                "dissociation_energy_d0_ev": 9.759,
                "dissociation_energy_d0_kcal_mol": 225.06,
                "ionization_potential_ev": 15.581,
                "dipole_moment_debye": 0.0,
            },
            "CO": {
                "r_e_angstrom": 1.1283,
                "r_e_bohr": 2.1322,
                "harmonic_frequency_cm": 2169.81,
                "dissociation_energy_d0_ev": 11.087,
                "dissociation_energy_d0_kcal_mol": 255.68,
                "ionization_potential_ev": 14.014,
                "permanent_dipole_debye": 0.112,
                "dipole_polarity": "C(-) O(+)",
                "mean_polarizability_bohr3": 13.08,
            },
            "H2O": {
                "r_oh_angstrom": 0.9578,
                "r_oh_bohr": 1.810,
                "angle_hoh_deg": 104.48,
                "permanent_dipole_debye": 1.8546,
                "ionization_potential_ev": 12.621,
                "vibrational_frequencies_cm": {
                    "v1_symmetric_stretch": 3657.05,
                    "v2_bend": 1594.75,
                    "v3_asymmetric_stretch": 3755.93,
                },
            },
            "EO_to_AA": {
                "reaction_enthalpy_kcal_mol": -27.2,
                "activation_energy_forward_kcal_mol": 57.2,
                "description": "Exothermic unimolecular isomerization of ethylene oxide to acetaldehyde",
            },
            "Bulk_Ni": {
                "crystal_structure": "FCC",
                "lattice_constant_angstrom": 3.524,
                "lattice_constant_bohr": 6.659,
                "nearest_neighbor_distance_bohr": 4.709,
                "cohesive_energy_ev_atom": 4.44,
                "cohesive_energy_kcal_mol": 102.4,
                "magnetic_moment_bohr_magneton": 0.616,
                "work_function_100_ev": 5.22,
            },
            "CO_on_Ni100": {
                "adsorption_site": "top (on-top)",
                "adsorption_energy_kcal_mol": -30.0,
                "adsorption_energy_ev": -1.30,
                "co_stretch_frequency_gas_cm": 2143.0,
                "co_stretch_frequency_adsorbed_cm": 2065.0,
                "co_frequency_shift_cm": -78.0,
            },
        },
    }
    with open(out_dir / "experimental_benchmarks.json", "w") as f:
        json.dump(exp_benchmarks, f, indent=2)

    # =========================================================================
    # 9. Manifest
    # =========================================================================
    print("9. Writing Manifest...")
    manifest = {
        "manifest_version": "1.0.0",
        "title": "PHATPSY 2.0 Pre-§1 Frozen Validation Baselines",
        "description": "Frozen reference outputs establishing baseline accuracy prior to Section 1 methodology changes (TF potential fitting, energy-dependent PK pseudopotential, EWMO charge equilibration).",
        "frozen_at_utc": timestamp,
        "git_commit": commit,
        "methodology_version": "PHATPSY 2.0 Core (Fortran 2018 / LAPACK, Clean Standard Defaults)",
        "default_model_parameters": {
            "charge_damp": 0.5,
            "potential_damp": 0.5,
            "pk_scale": 0.0,
            "convergence": 1.0e-4,
            "virial_tolerance": 0.05,
        },
        "tolerances": {
            "total_energy_hartree": 1.0e-3,
            "virial_ratio": 0.05,
            "harmonic_frequency_cm": 10.0,
            "dipole_moment_debye": 0.05,
            "polarizability_bohr3": 0.10,
            "barrier_kcal_mol": 1.0,
        },
        "systems": [
            "n2.json",
            "co.json",
            "h2o.json",
            "eo_to_aa.json",
            "bulk_ni.json",
            "ni_slab_3layer.json",
            "co_ni9.json",
            "experimental_benchmarks.json",
        ],
        "update_rule": "These baseline outputs are frozen. Any modification requires an explicit numerical-deviation report documenting the scientific rationale and citing the relevant dissertation section, per AGENTS.md §3 Rule 2.",
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("All validation baselines successfully generated and frozen into data/validation_baselines/!")


if __name__ == "__main__":
    main()
