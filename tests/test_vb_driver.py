"""End-to-end tests for the Phase 17 LocalizedVBDriver & active-site embedding integration."""
from __future__ import annotations

import copy
import numpy as np
import pytest

from phatpsy.builder import build_diatomic, build_molecule
from phatpsy.driver import PhatpsyDriver
from phatpsy.optimize import optimize_geometry
from phatpsy.pes import scan_1d
from phatpsy.schema import AtomConfig, CalculationDeck, ControlConfig
from phatpsy.state import CalculationState
from phatpsy.vb_driver import run_localized_vb


def test_schema_localized_vb_roundtrip_and_default_backward_compatibility():
    """Default ControlConfig remains phatpsy_scf, while localized_vb() round-trips cleanly."""
    ctrl_default = ControlConfig()
    assert ctrl_default.method == "phatpsy_scf"
    assert ctrl_default.spin_unrestricted is True
    assert "method" not in ctrl_default.to_dict()

    ctrl_vb = ControlConfig.localized_vb(basis="DZ", domain_mode="bond_local", vb_wavefunction="cas")
    assert ctrl_vb.method == "localized_vb"
    assert ctrl_vb.spin_unrestricted is False

    deck = CalculationDeck(
        title="H2 Localized VB YAML test",
        control=ctrl_vb,
        geometry=[
            AtomConfig(id="H1", element="H", xyz=[0.0, 0.0, 0.0], active=True),
            AtomConfig(id="H2", element="H", xyz=[0.0, 0.0, 1.4], active=True),
            AtomConfig(id="He3", element="He", xyz=[0.0, 0.0, 5.0], active=False, fixed=True),
        ],
    )
    deck.validate()

    yaml_text = deck.to_yaml()
    deck_yaml = CalculationDeck.from_yaml(yaml_text)
    assert deck_yaml.control.method == "localized_vb"
    assert deck_yaml.control.domain_mode == "bond_local"
    assert deck_yaml.geometry[0].is_active is True
    assert deck_yaml.geometry[2].is_active is False

    deck_json = CalculationDeck.from_json(deck.to_json())
    assert deck_json.control.environment_embedding == "exact_joint_pk"
    assert deck_json.geometry[2].active is False

    from phatpsy.localized_vb import atom_basis
    from phatpsy.vb_driver import _frozen_env_from_atom

    ni_env = _frozen_env_from_atom(
        AtomConfig(
            id="Ni_s",
            element="Ni",
            xyz=[4.709, 0.0, 0.0],
            basis="DZP",
            active=False,
            embedded_state="data/ni_surf_embedded.json",
        ),
        atom_basis((4.709, 0.0, 0.0), "Ni", "DZP"),
        "DZP",
    )
    assert ni_env.z_nuc == pytest.approx(28.0)
    assert ni_env.C_occ.shape[1] > 0


def test_h2_localized_vb_and_masked_vb_variational_physics_and_state_roundtrip(tmp_path):
    """H2 in ADF-TZP satisfies exact variational bound, virial theorem, binding, zero dipole, and serialization."""
    drv = PhatpsyDriver(quiet=True)

    # 1. Strict 1c Heitler-London (LocalizedVBModel)
    ctrl_hl = ControlConfig.localized_vb(
        basis="ADF-TZP",
        domain_mode="strict_1c",
        vb_wavefunction="hl",
        method="localized_vb",
    )
    deck_hl = build_molecule([("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.43)], basis="ADF-TZP", control=ctrl_hl)
    res_hl = drv.run_deck(deck_hl)
    assert res_hl["converged"] is True
    assert res_hl["total_energy"] > -1.1745  # Exact nonrelativistic H2 limit
    assert res_hl["binding_energy_ev"] > 3.9
    assert abs(res_hl["virial_ratio"] - 2.0) < 0.05
    assert res_hl["dipole_moment_debye"] < 1e-10

    # 2. Bond-local CAS via MaskedVB + regime_comparison + JSON/NPZ persistence round-trip
    ctrl_cas = ControlConfig.localized_vb(basis="ADF-TZP", domain_mode="bond_local", vb_wavefunction="cas")
    deck_cas = build_diatomic(
        "H", "H", bond_length=1.43, basis="ADF-TZP", control=ctrl_cas, spin_unrestricted=False, net_spin=0
    )
    res_cas = run_localized_vb(deck_cas, quiet=True, compare_regimes=True)
    assert res_cas["total_energy"] < res_hl["total_energy"]
    assert res_cas["total_energy"] > -1.1745
    assert res_cas["binding_energy_ev"] > 4.0
    assert abs(res_cas["virial_ratio"] - 2.0) < 0.03
    rc_h2 = res_cas["regime_comparison"]
    assert res_cas["total_energy"] == pytest.approx(rc_h2["bond_local"], abs=1e-9)
    assert rc_h2["delocalized"] <= rc_h2["bond_local"] + 1e-6 <= rc_h2["strict_1c"] + 2e-6

    p_json = tmp_path / "h2_vb.json"
    p_npz = tmp_path / "h2_vb.npz"
    res_cas.save(p_json)
    res_cas.save(p_npz)
    st_json = CalculationState.load(p_json)
    st_npz = CalculationState.load(p_npz)
    assert st_json["total_energy"] == pytest.approx(res_cas["total_energy"], abs=1e-10)
    assert st_npz["total_energy"] == pytest.approx(res_cas["total_energy"], abs=1e-10)
    assert isinstance(st_json["localized_orbitals"], np.ndarray)
    assert isinstance(st_npz["localized_orbitals"], np.ndarray)
    np.testing.assert_allclose(st_json["rdm1_ao"], res_cas["rdm1_ao"], atol=1e-10)


def test_lih_core_embedding_modes_prevent_collapse():
    """LiH (DZ) with core_embedding all_electron, exact_backprojected, and unprojected_eri agree tightly."""
    drv = PhatpsyDriver(quiet=True)
    geom = [("Li", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 3.015)]

    energies = {}
    for cmode in ("all_electron", "exact_backprojected", "unprojected_eri"):
        ctrl = ControlConfig.localized_vb(basis="DZ", domain_mode="bond_local", core_embedding=cmode)
        deck = build_molecule(geom, basis="DZ", control=ctrl)
        res = drv.run_deck(deck)
        assert res["converged"] is True
        energies[cmode] = float(res["total_energy"])
        assert abs(res["virial_ratio"] - 2.0) < 0.05
        assert res["binding_energy_ev"] > 1.2

    assert abs(energies["exact_backprojected"] - energies["unprojected_eri"]) < 1e-4
    assert abs(energies["exact_backprojected"] - energies["all_electron"]) < 5e-3


def test_h2o_multicenter_dipole_symmetry_and_translational_invariance():
    """H2O (SZ) under bond_local MaskedVB preserves C2v dipole symmetry, translational invariance, and locality."""
    drv = PhatpsyDriver(quiet=True)
    ctrl = ControlConfig.localized_vb(basis="SZ", domain_mode="bond_local")
    theta = np.deg2rad(104.5 / 2.0)
    r_oh = 1.809
    geom = [
        ("O", 0.0, 0.0, 0.0),
        ("H", 0.0, r_oh * np.sin(theta), r_oh * np.cos(theta)),
        ("H", 0.0, -r_oh * np.sin(theta), r_oh * np.cos(theta)),
    ]
    deck = build_molecule(geom, basis="SZ", control=ctrl)
    res = run_localized_vb(deck, quiet=True, compare_regimes=True)
    assert res["converged"] is True
    assert res["binding_energy_ev"] > 4.0
    assert abs(res["virial_ratio"] - 2.0) < 0.05
    rc_h2o = res["regime_comparison"]
    assert res["total_energy"] == pytest.approx(rc_h2o["bond_local"], abs=1e-9)
    assert rc_h2o["delocalized"] <= rc_h2o["bond_local"] + 1e-6 <= rc_h2o["strict_1c"] + 2e-6
    assert rc_h2o["delocalized"] < rc_h2o["strict_1c"]

    mu_vec = np.asarray(res["dipole_vector_au"])
    assert abs(mu_vec[0]) < 1e-5
    assert abs(mu_vec[1]) < 1e-5
    assert res["dipole_moment_debye"] > 1.0

    # Check own-atom character locality (> 0.90 on O orbitals)
    o_oac = res["own_atom_character"].get("O1", [])
    assert len(o_oac) > 0 and max(o_oac) > 0.90

    # Global translation invariance of total energy and dipole
    geom_shift = [(sym, x + 2.3, y - 1.7, z + 4.1) for sym, x, y, z in geom]
    deck_shift = build_molecule(geom_shift, basis="SZ", control=ctrl)
    res_shift = drv.run_deck(deck_shift)
    assert abs(res_shift["total_energy"] - res["total_energy"]) < 1e-5
    assert abs(res_shift["dipole_moment_debye"] - res["dipole_moment_debye"]) < 1e-5


def test_h2_plus_frozen_spectator_he_embedding():
    """Frozen spectator He with exact_joint_pk vs none demonstrates Pauli repulsion at short approach."""
    drv = PhatpsyDriver(quiet=True)
    geom_specs = [
        AtomConfig(id="H1", element="H", xyz=[0.0, 0.0, -0.7], active=True),
        AtomConfig(id="H2", element="H", xyz=[0.0, 0.0, 0.7], active=True),
        AtomConfig(id="He3", element="He", xyz=[0.0, 0.0, 2.7], active=False, fixed=True),
    ]

    deck_pk = CalculationDeck(
        title="H2 + He exact_joint_pk",
        control=ControlConfig.localized_vb(basis="DZ", domain_mode="bond_local", environment_embedding="exact_joint_pk"),
        geometry=copy.deepcopy(geom_specs),
    )
    deck_none = CalculationDeck(
        title="H2 + He none",
        control=ControlConfig.localized_vb(basis="DZ", domain_mode="bond_local", environment_embedding="none"),
        geometry=copy.deepcopy(geom_specs),
    )
    res_pk = drv.run_deck(deck_pk)
    res_none = drv.run_deck(deck_none)

    # Spectator interaction shifts active+He total relative to bare active H2
    assert res_pk["total_energy"] != pytest.approx(res_none["total_energy"], abs=1e-2)
    assert res_pk["embedding_diagnostics"]["n_spectator_atoms"] == 1
    assert res_pk["embedding_diagnostics"]["lmin_S_eff"] > 0.0

    # Comparing short vs far He separation: Pauli repulsion makes close He higher energy than far He
    deck_far = copy.deepcopy(deck_pk)
    deck_far.geometry[2].xyz = [0.0, 0.0, 6.0]
    res_far = drv.run_deck(deck_far)
    assert res_pk["total_energy"] > res_far["total_energy"]

    # Also test model_potential spectator mode runs cleanly
    deck_mp = CalculationDeck(
        title="H2 + He model_potential",
        control=ControlConfig.localized_vb(basis="DZ", domain_mode="bond_local", environment_embedding="model_potential"),
        geometry=copy.deepcopy(geom_specs),
    )
    res_mp = drv.run_deck(deck_mp)
    assert res_mp["converged"] is True
    assert res_mp["embedding_diagnostics"]["environment_embedding"] == "model_potential"


def test_h2_pes_scan_and_geometry_optimization_via_driver():
    """scan_1d and optimize_geometry work transparently on a Localized VB H2 deck."""
    drv = PhatpsyDriver(quiet=True)
    ctrl = ControlConfig.localized_vb(basis="ADF-DZ", domain_mode="bond_local", vb_wavefunction="cas")
    deck = build_molecule([("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.50)], basis="ADF-DZ", control=ctrl)

    pes = scan_1d(deck, coordinate="0-1", values=[1.2, 1.43, 1.8], driver=drv, warm_restart=True, quiet=True)
    assert len(pes.energies) == 3
    assert pes.energies[1] < pes.energies[0]
    assert pes.energies[1] < pes.energies[2]

    deck.geometry[0].active = True
    deck.geometry[0].fixed = True
    opt = optimize_geometry(
        deck, driver=drv, max_steps=6, grad_tol=1e-2, energy_tol=1e-5, quiet=True
    )
    r_opt = float(np.linalg.norm(
        np.array(opt.optimized_deck.geometry[1].xyz) - np.array(opt.optimized_deck.geometry[0].xyz)
    ))
    assert 1.35 <= r_opt <= 1.48


def test_localized_vb_diatomic_benchmark_evaluator():
    """Verify non-circular diatomic Localized VB benchmark evaluator computes physical results."""
    from phatpsy.benchmarks import evaluate_diatomic_localized_vb_benchmark

    h2_bm = evaluate_diatomic_localized_vb_benchmark("H2", basis="ADF-DZ")
    assert 1.35 <= h2_bm["r_e_bohr"] <= 1.55
    assert h2_bm["binding_energy_ev"] > 3.5
    assert 1.9 <= h2_bm["virial_ratio"] <= 2.1
    assert h2_bm["harmonic_frequency_cm"] > 3500.0


@pytest.mark.slow
def test_co_ni100_blyholder_benchmark_evaluator():
    """Verify non-circular CO/Ni(100) Blyholder benchmark evaluator computes physical results."""
    from phatpsy.benchmarks import evaluate_co_ni100_blyholder_benchmark

    coni_bm = evaluate_co_ni100_blyholder_benchmark()
    assert coni_bm["lmin_S_eff_exact"] > 0.0
    assert coni_bm["pi_share"] > 0.5
    assert coni_bm["delta_e_pi_ev"] < coni_bm["delta_e_sigma_ev"]

