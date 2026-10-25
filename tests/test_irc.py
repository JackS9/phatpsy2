# tests/test_irc.py
"""
Unit and integration tests for Intrinsic Reaction Coordinate (IRC) pathway tracing,
mass-weighted steepest descent, barrier calculations, and 3D WebGL visualizations.
"""
import copy
import json
import math
import tempfile
from pathlib import Path
import numpy as np
import pytest

from phatpsy.driver import PhatpsyDriver
from phatpsy.schema import CalculationDeck
from phatpsy.md import ATOMIC_MASSES_AMU, HARTREE_TO_KCAL, BOHR_TO_ANGSTROM
from phatpsy.vibrations import NormalMode
from phatpsy.irc import IRCPoint, IRCResult, compute_irc
from phatpsy.visualizer import generate_irc_visualization_data, generate_irc_html, visualize_irc


@pytest.fixture(scope="module")
def driver():
    """Shared silent PhatpsyDriver instance."""
    return PhatpsyDriver(quiet=True)


def test_irc_point_and_result_dataclasses():
    """Verify IRCPoint and IRCResult creation, dictionary export, and properties."""
    p_rev = IRCPoint(
        step=1,
        direction="reactant",
        s=-0.2,
        energy=-109.5200,
        delta_energy_kcal=-12.55,
        grad_norm=0.0004,
        geometry=[{"element": "N", "xyz": [0.0, 0.0, 1.15], "fixed": False, "virtual": False}],
    )
    p_ts = IRCPoint(
        step=2,
        direction="ts",
        s=0.0,
        energy=-109.5000,
        delta_energy_kcal=0.0,
        grad_norm=0.0000,
        geometry=[{"element": "N", "xyz": [0.0, 0.0, 1.20], "fixed": False, "virtual": False}],
    )
    p_fwd = IRCPoint(
        step=3,
        direction="product",
        s=+0.2,
        energy=-109.5300,
        delta_energy_kcal=-18.82,
        grad_norm=0.0003,
        geometry=[{"element": "N", "xyz": [0.0, 0.0, 1.25], "fixed": False, "virtual": False}],
    )

    res = IRCResult(
        points=[p_rev, p_ts, p_fwd],
        forward_points=[p_fwd],
        reverse_points=[p_rev],
        ts_point=p_ts,
        ts_energy=-109.5000,
        barrier_forward_kcal=12.55,
        barrier_reverse_kcal=18.82,
        reaction_energy_kcal=-6.27,
        converged_forward=True,
        converged_reverse=True,
        initial_mode_frequency_cm=850.5,
    )

    assert res.num_points == 3
    assert res.reactant_point.s == -0.2
    assert res.product_point.s == +0.2
    assert math.isclose(res.barrier_forward_kcal - res.barrier_reverse_kcal, res.reaction_energy_kcal, abs_tol=1e-2)

    d = res.to_dict()
    assert d["num_points"] == 3
    assert d["ts_energy"] == -109.5
    assert len(d["points"]) == 3
    assert d["points"][1]["direction"] == "ts"


def test_irc_mass_weighting_relation():
    """Verify mass-weighted transformation identity: sum(M_i * dr_i^2) == ds^2."""
    mass_N = ATOMIC_MASSES_AMU["N"]
    masses = np.array([mass_N, mass_N], dtype=np.float64)
    coord_masses = np.repeat(masses, 3)

    # Unit vector in mass-weighted coordinates along z
    u_mw = np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0], dtype=np.float64)
    u_mw /= np.linalg.norm(u_mw)
    assert math.isclose(np.linalg.norm(u_mw), 1.0, abs_tol=1e-12)

    ds = 0.15  # amu^(1/2) * Bohr
    dx_mw = ds * u_mw

    # Cartesian displacement
    dr = dx_mw / np.sqrt(coord_masses)

    # Invariant: sum(M_i * dr_i^2) == ds^2
    computed_ds2 = np.sum(coord_masses * (dr ** 2))
    assert math.isclose(computed_ds2, ds ** 2, rel_tol=1e-10)


def test_compute_irc_with_explicit_normal_mode(driver):
    """Test compute_irc using an explicit transition normal mode on stretched N2."""
    deck = CalculationDeck.from_yaml("examples/n2.yaml")
    # Position N2 at a stretched non-equilibrium geometry
    deck.geometry[0].xyz = [0.0, 0.0, 0.850]
    deck.geometry[1].xyz = [0.0, 0.0, -0.850]

    # Explicit normal mode: stretch along z
    disp = [
        [0.0, 0.0, +1.0 / math.sqrt(2.0)],
        [0.0, 0.0, -1.0 / math.sqrt(2.0)],
    ]
    test_mode = NormalMode(
        index=1,
        frequency_cm=-600.0,
        is_imaginary=True,
        reduced_mass_amu=14.007,
        force_constant_mdyne_a=5.0,
        displacements=disp,
    )

    irc_res = compute_irc(
        deck=deck,
        driver=driver,
        step_size=0.08,
        max_steps=4,
        direction="both",
        grad_tol=1e-3,
        normal_mode=test_mode,
        quiet=True,
    )

    assert irc_res.num_points >= 3
    assert irc_res.ts_point is not None
    assert math.isclose(irc_res.ts_point.s, 0.0, abs_tol=1e-8)
    assert irc_res.reactant_point.s < 0.0
    assert irc_res.product_point.s > 0.0

    # Ensure s is monotonically increasing from reactant to product
    s_list = [pt.s for pt in irc_res.points]
    for i in range(len(s_list) - 1):
        assert s_list[i] < s_list[i + 1]

    # Summary table output
    text_summary = irc_res.summary()
    assert "PHATPSY 2.0 INTRINSIC REACTION COORDINATE" in text_summary
    assert "Transition State Energy" in text_summary


def test_compute_irc_auto_transition_vector(driver):
    """Test compute_irc with automatic transition vector identification."""
    deck = CalculationDeck.from_yaml("examples/n2.yaml")
    deck.geometry[0].xyz = [0.0, 0.0, 0.720]
    deck.geometry[1].xyz = [0.0, 0.0, -0.720]

    irc_res = compute_irc(
        deck=deck,
        driver=driver,
        step_size=0.06,
        max_steps=3,
        direction="forward",
        quiet=True,
    )

    assert irc_res.num_points >= 2
    assert len(irc_res.forward_points) >= 1
    assert len(irc_res.reverse_points) == 0


def test_irc_export_xyz_and_json(driver):
    """Verify JSON and multi-frame XYZ trajectory serialization."""
    deck = CalculationDeck.from_yaml("examples/n2.yaml")
    deck.geometry[0].xyz = [0.0, 0.0, 0.750]
    deck.geometry[1].xyz = [0.0, 0.0, -0.750]

    disp = [[0.0, 0.0, 0.7071], [0.0, 0.0, -0.7071]]
    irc_res = compute_irc(
        deck=deck,
        driver=driver,
        step_size=0.05,
        max_steps=2,
        direction="both",
        normal_mode=disp,
        quiet=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "irc_test.json"
        xyz_path = Path(tmpdir) / "irc_test.xyz"

        irc_res.save_json(json_path)
        irc_res.save_xyz(xyz_path)

        assert json_path.exists()
        assert xyz_path.exists()

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["num_points"] == irc_res.num_points
        assert "points" in data
        assert "barrier_forward_kcal" in data

        xyz_text = xyz_path.read_text(encoding="utf-8")
        assert "IRC step=" in xyz_text
        assert "N  " in xyz_text or "N   " in xyz_text


def test_irc_3d_webgl_visualizer(driver):
    """Verify generation of interactive Three.js 3D WebGL HTML visualizer for IRC."""
    deck = CalculationDeck.from_yaml("examples/n2.yaml")
    deck.geometry[0].xyz = [0.0, 0.0, 0.750]
    deck.geometry[1].xyz = [0.0, 0.0, -0.750]

    disp = [[0.0, 0.0, 0.7071], [0.0, 0.0, -0.7071]]
    irc_res = compute_irc(
        deck=deck,
        driver=driver,
        step_size=0.05,
        max_steps=2,
        direction="both",
        normal_mode=disp,
        quiet=True,
    )

    data = generate_irc_visualization_data(irc_res)
    assert data["num_frames"] == irc_res.num_points
    assert "frames" in data
    assert "barrier_fwd_kcal" in data

    html_code = generate_irc_html(irc_res, title="Test IRC Visualization")
    assert "<!DOCTYPE html>" in html_code
    assert "three.min.js" in html_code
    assert "OrbitControls.js" in html_code
    assert "frame-slider" in html_code
    assert "state-badge" in html_code

    with tempfile.TemporaryDirectory() as tmpdir:
        out_html = Path(tmpdir) / "irc_view.html"
        visualize_irc(irc_res, output_path=out_html, open_browser=False)
        assert out_html.exists()
        assert out_html.stat().st_size > 1000


def test_cli_irc(driver):
    """Test the CLI 'phatpsy irc' subcommand."""
    from phatpsy.cli import main

    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = Path(tmpdir) / "cli_irc.json"
        out_xyz = Path(tmpdir) / "cli_irc.xyz"

        ret = main([
            "irc",
            "examples/n2.yaml",
            "-s", "0.05",
            "-n", "2",
            "--direction", "both",
            "-o", str(out_json),
            "--xyz", str(out_xyz),
        ])

        assert ret == 0
        assert out_json.exists()
        assert out_xyz.exists()

        with open(out_json) as f:
            data = json.load(f)
        assert data["num_points"] >= 3
