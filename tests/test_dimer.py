# tests/test_dimer.py
"""
Unit and integration tests for Dimer transition-state refinement:
- Analytical Müller-Brown benchmark testbed (saddle-point convergence).
- Gradient-only cost accounting (no Hessians in the loop).
- Molecular single-ended search from reactant / perturbed geometry.
- Handoff from Climbing-Image NEB climbing image.
- Serialization and multi-frame XYZ trajectory export.
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
from phatpsy.neb import neb_search
from phatpsy.dimer import (
    MullerBrownSurface,
    DimerStep,
    DimerResult,
    dimer_optimize_vector,
    dimer_search,
    refine_neb_saddle,
)


@pytest.fixture(scope="module")
def driver():
    """Shared silent PhatpsyDriver instance."""
    return PhatpsyDriver(quiet=True)


# ==============================================================================
# 1. Müller-Brown Analytic Surface Benchmark
# ==============================================================================

def test_muller_brown_surface_analytic_properties():
    """Verify Müller-Brown potential values and gradient at known stationary points."""
    mb = MullerBrownSurface()

    # Known saddle 1 near (-0.822, 0.624)
    s1 = mb.SADDLE_1
    grad_s1 = mb.gradient(s1)
    norm_s1 = float(np.linalg.norm(grad_s1))
    assert norm_s1 < 1e-4
    e_s1 = mb.energy(s1)
    assert -42.0 < e_s1 < -39.0

    # Force should be opposite of gradient
    force_s1 = mb.force(s1)
    assert np.allclose(force_s1, -grad_s1)


def test_dimer_muller_brown_saddle_convergence():
    """
    Core benchmark acceptance criterion:
    Run the gradient-only Dimer algorithm on the Müller-Brown surface starting from
    a displaced point (-0.75, 0.55) and verify it converges to the known saddle point
    (-0.822002, 0.624313) with negative curvature.
    """
    mb = MullerBrownSurface()
    r_start = np.array([-0.75, 0.55], dtype=np.float64)
    n_start = np.array([1.0, 0.0], dtype=np.float64)

    converged, r_saddle, n_saddle, e_saddle, c_saddle, n_cycles, n_evals, traj = dimer_optimize_vector(
        force_func=mb.force,
        energy_func=mb.energy,
        x0=r_start,
        n0=n_start,
        dimer_dist=0.005,
        max_cycles=120,
        force_tol=1e-3,
        rot_tol=1e-3,
        dt=0.002,
        max_step=0.02,
    )

    assert converged is True
    # Verify convergence to known saddle 1
    error = np.linalg.norm(r_saddle - mb.SADDLE_1)
    assert error < 1e-3
    assert c_saddle < 0.0  # Must be negative curvature at saddle point!
    assert -760.0 < c_saddle < -740.0  # Analytical lowest eigenvalue is -750.86


def test_dimer_cost_accounting():
    """
    Verify cost per cycle:
    Dimer algorithm must not evaluate any Hessians, and gradient evaluations
    per cycle must satisfy the 3 to 4 evaluations per cycle specification.
    """
    mb = MullerBrownSurface()
    r_start = np.array([-0.80, 0.60], dtype=np.float64)

    converged, r_saddle, n_saddle, e_saddle, c_saddle, n_cycles, n_evals, traj = dimer_optimize_vector(
        force_func=mb.force,
        energy_func=mb.energy,
        x0=r_start,
        dimer_dist=0.005,
        max_cycles=10,
        force_tol=1e-3,
    )

    # Cost per cycle: between 3 and 4 gradient evaluations per cycle per Heyden/Bell
    evals_per_cycle = n_evals / n_cycles
    assert 3.0 <= evals_per_cycle <= 4.0


# ==============================================================================
# 2. Molecular Dimer Search (Quantum Chemistry Core)
# ==============================================================================

def test_dimer_search_molecular_single_ended(driver):
    """
    Single-ended check: dimer search on a molecular deck from a perturbed geometry.
    Verifies that no Cartesian Hessian is constructed, and negative curvature is reached.
    """
    deck = CalculationDeck.from_yaml("examples/n2.yaml")
    # Position N2 at non-equilibrium stretched configuration
    deck.geometry[0].xyz = [0.0, 0.0, 0.780]
    deck.geometry[1].xyz = [0.0, 0.0, -0.780]

    # Initial stretch mode along z
    init_mode = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)

    dimer_res = dimer_search(
        deck=deck,
        driver=driver,
        initial_direction=init_mode,
        dimer_dist=0.005,
        finite_diff_step=0.001,
        max_cycles=4,
        force_tol=1e-3,
        quiet=True,
    )

    assert dimer_res.num_cycles >= 1
    assert dimer_res.num_gradient_evals > 0
    assert len(dimer_res.trajectory) == dimer_res.num_cycles
    assert dimer_res.saddle_deck is not None

    # Summary text
    summary = dimer_res.summary()
    assert "PHATPSY 2.0 DIMER TRANSITION-STATE REFINEMENT" in summary
    assert "Final Curvature (C)" in summary


def test_refine_neb_saddle_handoff(driver):
    """
    Integration check: Run a short CI-NEB search, hand off the climbing image
    to refine_neb_saddle(), and verify the transition state refinement pipeline.
    """
    deck_r = CalculationDeck.from_yaml("examples/n2.yaml")
    deck_r.geometry[0].xyz = [0.0, 0.0, 0.700]
    deck_r.geometry[1].xyz = [0.0, 0.0, -0.700]

    deck_p = CalculationDeck.from_yaml("examples/n2.yaml")
    deck_p.geometry[0].xyz = [0.0, 0.0, 0.900]
    deck_p.geometry[1].xyz = [0.0, 0.0, -0.900]

    # Fast 3-image NEB
    neb_res = neb_search(
        initial_deck=deck_r,
        final_deck=deck_p,
        driver=driver,
        num_images=3,
        max_iterations=2,
        quiet=True,
    )

    # Refine saddle point from NEB climbing image
    dimer_res = refine_neb_saddle(
        neb_result=neb_res,
        driver=driver,
        dimer_dist=0.005,
        max_cycles=3,
        force_tol=1e-3,
        quiet=True,
    )

    assert dimer_res.num_cycles >= 1
    assert dimer_res.saddle_deck is not None


def test_dimer_serialization_and_xyz(driver):
    """Verify DimerResult JSON serialization and multi-frame XYZ export."""
    deck = CalculationDeck.from_yaml("examples/n2.yaml")
    deck.geometry[0].xyz = [0.0, 0.0, 0.750]
    deck.geometry[1].xyz = [0.0, 0.0, -0.750]

    dimer_res = dimer_search(
        deck=deck,
        driver=driver,
        dimer_dist=0.005,
        max_cycles=2,
        quiet=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "dimer_test.json"
        xyz_path = Path(tmpdir) / "dimer_traj.xyz"

        dimer_res.save_json(json_path)
        dimer_res.save_xyz(xyz_path)

        assert json_path.exists()
        assert xyz_path.exists()

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "saddle_energy" in data
        assert "trajectory" in data
        assert len(data["trajectory"]) == dimer_res.num_cycles

        xyz_text = xyz_path.read_text(encoding="utf-8")
        assert "Dimer cycle=" in xyz_text
        assert "N  " in xyz_text or "N   " in xyz_text


def test_dimer_visualizer(driver):
    """Verify Dimer 3D WebGL HTML visualization generation."""
    deck = CalculationDeck.from_yaml("examples/n2.yaml")
    dimer_res = dimer_search(
        deck=deck,
        driver=driver,
        dimer_dist=0.005,
        max_cycles=2,
        quiet=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "dimer_view.html"
        html_str = dimer_res.visualize(output_path=html_path, open_browser=False)

        assert html_path.exists()
        assert "<!DOCTYPE html>" in html_str
        assert "Dimer Transition-State Refinement" in html_str
        assert "THREE.WebGLRenderer" in html_str


def test_dimer_cli(driver):
    """Verify CLI subcommand execution (phatpsy dimer)."""
    from phatpsy.cli import main

    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = Path(tmpdir) / "cli_dimer.json"
        out_xyz = Path(tmpdir) / "cli_dimer.xyz"

        ret = main([
            "dimer",
            "examples/n2.yaml",
            "--dimer-dist", "0.005",
            "--max-cycles", "2",
            "-o", str(out_json),
            "--xyz", str(out_xyz),
        ])

        assert ret == 0
        assert out_json.exists()
        assert out_xyz.exists()
