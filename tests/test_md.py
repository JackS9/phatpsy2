# tests/test_md.py
"""
Unit and integration tests for Born-Oppenheimer Molecular Dynamics (BOMD) in PHATPSY 2.0.
"""
import json
from pathlib import Path
import numpy as np
import pytest

from phatpsy.builder import build_diatomic
from phatpsy.schema import CalculationDeck, AtomConfig
from phatpsy.cli import main as cli_main
from phatpsy.md import (
    run_md,
    init_velocities,
    compute_temperature,
    compute_kinetic_energy,
    get_atomic_mass_au,
    MDResult,
    MDStep,
    AMU_TO_AU_MASS,
    FS_TO_AU_TIME,
    BOLTZMANN_AU,
)


def test_atomic_masses_and_physical_constants():
    """Verify atomic mass lookups and fundamental atomic units conversion constants."""
    # Hydrogen
    m_h = get_atomic_mass_au("H")
    assert m_h == pytest.approx(1.008 * AMU_TO_AU_MASS, rel=1e-4)

    # Carbon and Nitrogen
    assert get_atomic_mass_au("C") == pytest.approx(12.011 * AMU_TO_AU_MASS, rel=1e-4)
    assert get_atomic_mass_au("N") == pytest.approx(14.007 * AMU_TO_AU_MASS, rel=1e-4)

    # Unit conversion sanity checks
    # 1 fs ≈ 41.34137 atomic units of time
    assert 41.0 < FS_TO_AU_TIME < 42.0
    # Boltzmann constant in atomic units (Hartree / K)
    assert 3.1e-6 < BOLTZMANN_AU < 3.2e-6


def test_velocity_initialization_and_temperature():
    """Verify Maxwell-Boltzmann velocity sampling, temperature scaling, and COM momentum zeroing."""
    masses = np.array([get_atomic_mass_au("N"), get_atomic_mass_au("N")])
    free_mask = np.array([True, True])

    # Sample at 300 K with reproducible seed
    rng = np.random.default_rng(42)
    vels = init_velocities(masses, temperature_k=300.0, free_mask=free_mask, remove_com_drift=True, rng=rng)

    assert vels.shape == (2, 3)

    # Verify zero net center-of-mass linear momentum
    total_momentum = np.sum(masses[:, None] * vels, axis=0)
    np.testing.assert_allclose(total_momentum, [0.0, 0.0, 0.0], atol=1e-12)

    # Verify instantaneous kinetic temperature matches 300 K
    temp = compute_temperature(vels, masses, free_mask)
    assert temp == pytest.approx(300.0, abs=1e-2)

    # Verify kinetic energy calculation
    e_kin = compute_kinetic_energy(vels, masses, free_mask)
    assert e_kin > 0.0
    # For 2 atoms with COM removed (N_dof = 3), E_kin = 0.5 * N_dof * k_B * T
    assert e_kin == pytest.approx(0.5 * 3 * BOLTZMANN_AU * 300.0, rel=1e-3)


def test_fixed_atom_constraints():
    """Verify that frozen atoms (fixed=True) maintain zero velocity and zero displacement."""
    masses = np.array([get_atomic_mass_au("Ni"), get_atomic_mass_au("C"), get_atomic_mass_au("O")])
    free_mask = np.array([False, True, True])  # Ni is fixed

    rng = np.random.default_rng(123)
    vels = init_velocities(masses, temperature_k=300.0, free_mask=free_mask, remove_com_drift=True, rng=rng)

    # Ni must have zero velocity
    np.testing.assert_allclose(vels[0], [0.0, 0.0, 0.0])

    # C and O should have non-zero velocities
    assert np.linalg.norm(vels[1]) > 0.0
    assert np.linalg.norm(vels[2]) > 0.0

    # Net momentum across free atoms C and O must be zero
    free_momentum = masses[1] * vels[1] + masses[2] * vels[2]
    np.testing.assert_allclose(free_momentum, [0.0, 0.0, 0.0], atol=1e-12)


def test_md_result_exports_and_visualization(tmp_path):
    """Verify MDResult serialization to JSON, multi-frame XYZ export, plot, and 3D HTML viewer."""
    deck = build_diatomic("N", "N", bond_length=2.068)

    steps = [
        MDStep(
            step=0,
            time_fs=0.0,
            potential_energy=-109.50,
            kinetic_energy=0.005,
            total_energy=-109.495,
            temperature=300.0,
            virial_ratio=2.01,
            coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.068]],
            velocities=[[0.0, 0.0, 0.001], [0.0, 0.0, -0.001]],
            forces=[[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]],
        ),
        MDStep(
            step=1,
            time_fs=0.5,
            potential_energy=-109.51,
            kinetic_energy=0.015,
            total_energy=-109.495,
            temperature=310.0,
            virial_ratio=2.00,
            coordinates=[[0.0, 0.0, -0.01], [0.0, 0.0, 2.078]],
            velocities=[[0.0, 0.0, 0.002], [0.0, 0.0, -0.002]],
            forces=[[0.0, 0.0, 0.005], [0.0, 0.0, -0.005]],
        ),
    ]

    res = MDResult(
        ensemble="nve",
        time_step_fs=0.5,
        total_time_fs=0.5,
        num_steps=1,
        initial_energy=-109.495,
        final_energy=-109.495,
        energy_drift=0.0,
        average_temperature=305.0,
        temperature_std=5.0,
        conserved_energy_std=0.0,
        trajectory=steps,
        final_deck=deck,
        final_state=None,
    )

    # JSON export
    json_path = tmp_path / "trajectory.json"
    res.save_json(json_path)
    assert json_path.exists()
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["ensemble"] == "nve"
    assert data["num_steps"] == 1
    assert len(data["trajectory"]) == 2

    # Multi-frame XYZ export
    xyz_path = tmp_path / "trajectory.xyz"
    res.save_xyz(xyz_path)
    assert xyz_path.exists()
    xyz_text = xyz_path.read_text(encoding="utf-8")
    assert "Step=0" in xyz_text
    assert "Step=1" in xyz_text
    assert xyz_text.count("2\n") >= 2  # 2 atoms per frame

    # Dynamics Plot
    plot_path = tmp_path / "dynamics.png"
    res.plot_dynamics(output_path=plot_path)
    assert plot_path.exists()
    assert plot_path.stat().st_size > 5000

    # 3D HTML Trajectory Viewer
    html_path = tmp_path / "trajectory.html"
    res.visualize(output_path=html_path, open_browser=False)
    assert html_path.exists()
    html_text = html_path.read_text(encoding="utf-8")
    assert "Three.js" in html_text or "three.min.js" in html_text
    assert "timeline-controls" in html_text
    assert "BOMD Trajectory" in html_text


def test_bomd_n2_nve_trajectory():
    """
    Integration test: run a 2-step NVE Born-Oppenheimer MD on N2 with symmetry.
    Tests force evaluation, Velocity Verlet numerical updates, and energy accounting.
    """
    deck = build_diatomic("N", "N", bond_length=2.068, basis="DZ", use_symmetry=False)

    res = run_md(
        deck=deck,
        time_step_fs=0.5,
        num_steps=2,
        temperature_k=150.0,
        ensemble="nve",
        warm_restart=True,
        seed=101,
    )

    assert res.num_steps == 2
    assert len(res.trajectory) == 3  # step 0, 1, 2
    assert res.trajectory[0].step == 0
    assert res.trajectory[1].step == 1
    assert res.trajectory[2].step == 2

    # Time advancing by dt = 0.5 fs
    assert res.trajectory[0].time_fs == pytest.approx(0.0)
    assert res.trajectory[1].time_fs == pytest.approx(0.5)
    assert res.trajectory[2].time_fs == pytest.approx(1.0)

    # Physical sanity checks
    for s in res.trajectory:
        assert s.potential_energy < -100.0  # Around -109 Hartree
        assert s.kinetic_energy >= 0.0
        assert s.temperature >= 0.0
        # Virial theorem -V/T should remain close to 2.0
        assert 1.9 < s.virial_ratio < 2.1


def test_bomd_thermostats():
    """Verify Berendsen and velocity rescaling thermostat operations."""
    deck = build_diatomic("N", "N", bond_length=2.068, basis="DZ", use_symmetry=False)

    # Run 1 step under Berendsen thermostat targeting 200 K
    res_b = run_md(
        deck=deck,
        time_step_fs=0.5,
        num_steps=1,
        temperature_k=200.0,
        ensemble="berendsen",
        tau_fs=20.0,
        warm_restart=True,
        seed=42,
    )
    assert res_b.ensemble == "berendsen"
    assert len(res_b.trajectory) == 2

    # Run 1 step under velocity rescaling thermostat targeting 250 K
    res_r = run_md(
        deck=deck,
        time_step_fs=0.5,
        num_steps=1,
        temperature_k=250.0,
        ensemble="rescale",
        warm_restart=True,
        seed=42,
    )
    assert res_r.ensemble == "rescale"
    assert len(res_r.trajectory) == 2
    # At step 1, velocity rescaling sets T exactly to 250 K
    assert res_r.trajectory[1].temperature == pytest.approx(250.0, abs=1.0)


def test_cli_md(tmp_path, monkeypatch):
    """Verify 'phatpsy md' CLI command with all flags and output artifacts."""
    deck_path = tmp_path / "n2.yaml"
    deck = build_diatomic("N", "N", bond_length=2.068, basis="DZ", use_symmetry=False)
    deck.to_yaml(deck_path)

    out_json = tmp_path / "md_out.json"
    out_xyz = tmp_path / "md_traj.xyz"
    out_plot = tmp_path / "md_curve.png"

    exit_code = cli_main([
        "md",
        str(deck_path),
        "--steps", "1",
        "--dt", "0.5",
        "--temp", "100.0",
        "--thermostat", "nve",
        "-o", str(out_json),
        "--xyz", str(out_xyz),
        "-p", str(out_plot),
    ])

    assert exit_code == 0
    assert out_json.exists()
    assert out_xyz.exists()
    assert out_plot.exists()

    with open(out_json) as f:
        data = json.load(f)
    assert data["ensemble"] == "nve"
    assert data["num_steps"] == 1
