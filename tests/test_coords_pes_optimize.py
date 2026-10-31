# tests/test_coords_pes_optimize.py
"""
Unit and integration tests for Internal Coordinates, Wilson B-matrix, Z-Matrix transformations,
1D and 2D Potential Energy Surface (PES) scans, and BFGS geometry optimization (Roadmap Section 4).
"""
import pytest
from pathlib import Path
import json
import math
import numpy as np

from phatpsy.schema import AtomConfig, CalculationDeck, ControlConfig
from phatpsy.io import parse_stdin_to_deck
from phatpsy.coords import (
    BondCoordinate,
    AngleCoordinate,
    DihedralCoordinate,
    InternalCoordinates,
    ZMatrix,
    ZMatrixEntry,
    normalize_vector,
)
from phatpsy.pes import (
    scan_1d,
    scan_2d,
    PESScan1DResult,
    PESScan2DResult,
    modify_deck_geometry,
)
from phatpsy.optimize import (
    optimize_geometry,
    OptimizationResult,
    OptimizationStep,
)
from phatpsy.cli import main as cli_main

EXAMPLE_N2 = Path(__file__).parent.parent / "examples" / "n2.stdin"


# ============================================================================
# 1. Internal Coordinates & Wilson B-Matrix Analytic vs Finite-Diff Gradients
# ============================================================================

def test_bond_coordinate_value_and_gradient():
    """Verify bond distance evaluation and Wilson B-matrix gradient."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
    ])
    bond = BondCoordinate(atom1=0, atom2=1, name="R(1-2)")
    assert bond.value(coords) == pytest.approx(2.0, abs=1e-8)

    grad = bond.gradient(coords)
    assert grad.shape == (2, 3)
    np.testing.assert_allclose(grad[0], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(grad[1], [0.0, 0.0, 1.0])

    # Numerical finite-difference check on arbitrary coords
    rng_coords = np.array([
        [0.5, 1.2, -0.3],
        [-0.8, 0.4, 1.7],
    ])
    analytic_b = bond.gradient(rng_coords)
    h = 1e-6
    for atom_idx in (0, 1):
        for dim in range(3):
            cp = rng_coords.copy()
            cm = rng_coords.copy()
            cp[atom_idx, dim] += h
            cm[atom_idx, dim] -= h
            num_grad = (bond.value(cp) - bond.value(cm)) / (2.0 * h)
            assert analytic_b[atom_idx, dim] == pytest.approx(num_grad, abs=1e-5)


def test_angle_coordinate_value_and_gradient():
    """Verify valence angle evaluation and Wilson B-matrix gradient."""
    # Right angle: (1, 0, 0) - (0, 0, 0) - (0, 1, 0)
    coords = np.array([
        [1.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ])
    ang = AngleCoordinate(atom1=0, atom2=1, atom3=2)
    assert ang.value(coords) == pytest.approx(math.pi / 2.0, abs=1e-8)
    assert ang.value(coords, in_degrees=True) == pytest.approx(90.0, abs=1e-6)

    # Numerical gradient check
    analytic_b = ang.gradient(coords)
    assert analytic_b.shape == (3, 3)
    h = 1e-6
    for atom_idx in range(3):
        for dim in range(3):
            cp = coords.copy()
            cm = coords.copy()
            cp[atom_idx, dim] += h
            cm[atom_idx, dim] -= h
            num_grad = (ang.value(cp) - ang.value(cm)) / (2.0 * h)
            assert analytic_b[atom_idx, dim] == pytest.approx(num_grad, abs=1e-5)


def test_dihedral_coordinate_value_and_gradient():
    """Verify dihedral angle calculation and Wilson B-matrix gradient."""
    # Trans conformation (180 deg) in butane-like 4-point chain
    coords = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [1.5, -1.0, 0.0],
    ])
    dih = DihedralCoordinate(atom1=0, atom2=1, atom3=2, atom4=3)
    val_deg = dih.value(coords, in_degrees=True)
    assert abs(abs(val_deg) - 180.0) < 1e-5

    # Skewed dihedral (non-planar) for gradient check
    coords_skew = np.array([
        [0.2, 1.1, 0.4],
        [0.0, 0.1, -0.2],
        [1.4, -0.2, 0.1],
        [1.6, -1.2, 0.9],
    ])
    analytic_b = dih.gradient(coords_skew)
    assert analytic_b.shape == (4, 3)

    h = 1e-6
    for atom_idx in range(4):
        for dim in range(3):
            cp = coords_skew.copy()
            cm = coords_skew.copy()
            cp[atom_idx, dim] += h
            cm[atom_idx, dim] -= h
            num_grad = (dih.value(cp) - dih.value(cm)) / (2.0 * h)
            assert analytic_b[atom_idx, dim] == pytest.approx(num_grad, abs=1e-5)


def test_internal_coordinates_from_atoms_and_b_matrix():
    """Verify automatic InternalCoordinates generation from atoms and Wilson B-matrix assembly."""
    atoms = [
        AtomConfig(element="O", xyz=[0.0, 0.0, 0.0]),
        AtomConfig(element="H", xyz=[0.0, 1.43, 1.10]),
        AtomConfig(element="H", xyz=[0.0, -1.43, 1.10]),
    ]
    ic = InternalCoordinates.from_atoms(atoms)
    assert len(ic.bonds) == 2
    assert len(ic.angles) == 1
    assert len(ic.dihedrals) == 0
    assert ic.num_coordinates == 3

    coords = np.array([a.xyz for a in atoms])
    vals = ic.calculate(coords, angles_in_degrees=True)
    assert len(vals) == 3
    # Two symmetric O-H bonds ~ 1.804 Bohr
    assert vals[0] == pytest.approx(1.8041, abs=1e-3)
    assert vals[1] == pytest.approx(1.8041, abs=1e-3)
    # H-O-H angle ~ 104.86 degrees
    assert vals[2] == pytest.approx(104.86, abs=1e-2)

    b_mat = ic.b_matrix(coords)
    assert b_mat.shape == (3, 9)


# ============================================================================
# 2. Z-Matrix Construction & Cartesian Roundtrip
# ============================================================================

def test_zmatrix_roundtrip_cartesian():
    """Verify Z-matrix generation and exact Cartesian NeRF reconstruction."""
    atoms = [
        AtomConfig(element="C", xyz=[0.0, 0.0, 0.0]),
        AtomConfig(element="C", xyz=[1.5, 0.0, 0.0]),
        AtomConfig(element="H", xyz=[2.0, 1.0, 0.0]),
        AtomConfig(element="H", xyz=[-0.5, -1.0, 0.8]),
    ]
    zm = ZMatrix.from_atoms(atoms)
    assert len(zm.entries) == 4
    assert zm.entries[0].element == "C"
    assert zm.entries[1].bond_length == pytest.approx(1.5, abs=1e-6)

    recon_coords = zm.to_cartesian()
    assert recon_coords.shape == (4, 3)

    # Invariant: all pairwise interatomic distances must match exactly between orig and recon
    orig_coords = np.array([a.xyz for a in atoms])
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            d_orig = float(np.linalg.norm(orig_coords[i] - orig_coords[j]))
            d_recon = float(np.linalg.norm(recon_coords[i] - recon_coords[j]))
            assert d_orig == pytest.approx(d_recon, abs=1e-5), f"Distance mismatch for pair {i}-{j}"


def test_zmatrix_triatomic_ref_index_zero():
    """
    Regression test for ZMatrix n=3 branch where ref_bond or ref_angle is index 0.
    Ensures 0 is not falsely evaluated as falsy (which previously caused 0 or 1 -> 1).
    """
    import math

    # Case 1: Atom 2 bonded to Atom 0 (ref_bond=0, ref_angle=1), e.g. H-O-H bend
    zm_central_zero = ZMatrix(entries=[
        ZMatrixEntry(element="O"),
        ZMatrixEntry(element="H", ref_bond=0, bond_length=1.8),
        ZMatrixEntry(element="H", ref_bond=0, bond_length=1.8, ref_angle=1, angle_val=104.5),
    ])
    coords1 = zm_central_zero.to_cartesian()
    assert coords1.shape == (3, 3)

    d01 = float(np.linalg.norm(coords1[1] - coords1[0]))
    d02 = float(np.linalg.norm(coords1[2] - coords1[0]))
    assert d01 == pytest.approx(1.8, abs=1e-5)
    assert d02 == pytest.approx(1.8, abs=1e-5)

    v01 = (coords1[1] - coords1[0]) / d01
    v02 = (coords1[2] - coords1[0]) / d02
    angle1 = math.degrees(math.acos(np.clip(np.dot(v01, v02), -1.0, 1.0)))
    assert angle1 == pytest.approx(104.5, abs=1e-4)

    # Case 2: Atom 2 bonded to Atom 1 with angle to Atom 0 (ref_bond=1, ref_angle=0), e.g. linear chain
    zm_chain = ZMatrix(entries=[
        ZMatrixEntry(element="C"),
        ZMatrixEntry(element="C", ref_bond=0, bond_length=2.5),
        ZMatrixEntry(element="O", ref_bond=1, bond_length=2.2, ref_angle=0, angle_val=120.0),
    ])
    coords2 = zm_chain.to_cartesian()
    assert coords2.shape == (3, 3)

    d10 = float(np.linalg.norm(coords2[0] - coords2[1]))
    d12 = float(np.linalg.norm(coords2[2] - coords2[1]))
    assert d10 == pytest.approx(2.5, abs=1e-5)
    assert d12 == pytest.approx(2.2, abs=1e-5)

    v10 = (coords2[0] - coords2[1]) / d10
    v12 = (coords2[2] - coords2[1]) / d12
    angle2 = math.degrees(math.acos(np.clip(np.dot(v10, v12), -1.0, 1.0)))
    assert angle2 == pytest.approx(120.0, abs=1e-4)


# ============================================================================
# 3. 1D Potential Energy Surface Scan
# ============================================================================

def test_scan_1d_n2_rigid_stretch(tmp_path):
    """Verify automated 1D PES scan on N2 molecule with virial ratio and min tracking."""
    deck = parse_stdin_to_deck(EXAMPLE_N2)

    # 3 points around potential minimum (1.40 Bohr)
    points = [1.30, 1.40, 1.50]
    res = scan_1d(deck, ("bond", 0, 1), points, warm_restart=True)

    assert isinstance(res, PESScan1DResult)
    assert len(res.values) == 3
    assert len(res.energies) == 3
    assert all(res.converged)

    # Physical invariant: Virial ratio (-V/T) ~ 2.0 for all scanned points
    for vr in res.virial_ratios:
        assert abs(vr - 2.02) < 0.05

    # Minimum should be at 1.40 Bohr
    assert res.min_value == pytest.approx(1.40, abs=1e-3)
    assert res.min_energy < res.energies[0]
    assert res.min_energy < res.energies[2]

    # JSON export test
    json_path = tmp_path / "n2_pes_1d.json"
    res.save_json(json_path)
    assert json_path.exists()
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["min_value"] == pytest.approx(1.40, abs=1e-3)
    assert len(data["energies"]) == 3

    # Plot test
    plot_path = tmp_path / "n2_pes_1d.png"
    res.plot(output_path=plot_path)
    assert plot_path.exists()
    assert plot_path.stat().st_size > 5000


# ============================================================================
# 4. 2D Potential Energy Surface Scan
# ============================================================================

def test_scan_2d_grid(tmp_path):
    """Verify automated 2D grid PES scan and contour plotting."""
    deck = parse_stdin_to_deck(EXAMPLE_N2)

    # Scan 2x2 grid
    vals1 = [2.00, 2.068]
    vals2 = [2.068, 2.15]

    res = scan_2d(
        deck,
        ("bond", 0, 1), vals1,
        ("bond", 0, 1), vals2,
        coord1_name="R1",
        coord2_name="R2",
        warm_restart=True,
    )

    assert isinstance(res, PESScan2DResult)
    assert len(res.energies) == 2
    assert len(res.energies[0]) == 2
    assert res.coord1_name == "R1"
    assert res.coord2_name == "R2"

    # JSON serialization
    json_path = tmp_path / "pes_2d.json"
    res.save_json(json_path)
    assert json_path.exists()

    # Plot export
    plot_path = tmp_path / "pes_2d.png"
    res.plot(output_path=plot_path)
    assert plot_path.exists()
    assert plot_path.stat().st_size > 5000


# ============================================================================
# 5. Geometry Optimization (BFGS)
# ============================================================================

def test_optimize_geometry_stretched_n2(tmp_path):
    """Verify BFGS ground-state geometry relaxation on stretched N2."""
    deck = parse_stdin_to_deck(EXAMPLE_N2)

    # Create stretched deck at R = 2.18 Bohr (equilibrium is 2.068 Bohr)
    stretched_deck = modify_deck_geometry(deck, "bond", (0, 1), 2.18)

    opt = optimize_geometry(
        stretched_deck,
        max_steps=5,
        grad_tol=1e-2,
        energy_tol=1e-4,
        warm_restart=True,
    )

    assert isinstance(opt, OptimizationResult)
    assert opt.num_steps >= 1
    # Energy must decrease during optimization
    assert opt.energy_change < 0.0
    assert opt.final_energy < opt.initial_energy

    # Trajectory tracking
    assert len(opt.trajectory) >= 2
    for step in opt.trajectory:
        assert isinstance(step, OptimizationStep)
        assert abs(step.virial_ratio - 2.012) < 0.05

    # JSON export
    traj_path = tmp_path / "opt_traj.json"
    opt.save_json(traj_path)
    assert traj_path.exists()
    with open(traj_path, encoding="utf-8") as f:
        data = json.load(f)
    assert "trajectory" in data
    assert len(data["trajectory"]) == len(opt.trajectory)

    # Plot convergence history
    conv_plot = tmp_path / "opt_conv.png"
    opt.plot_convergence(output_path=conv_plot)
    assert conv_plot.exists()
    assert conv_plot.stat().st_size > 5000


def test_optimize_geometry_all_fixed():
    """Verify that optimizer immediately finishes when all atoms are fixed."""
    deck = parse_stdin_to_deck(EXAMPLE_N2)
    for a in deck.atoms:
        a.fixed = True

    opt = optimize_geometry(deck, max_steps=5)
    assert opt.converged is True
    assert opt.num_steps == 0
    assert opt.energy_change == 0.0


# ============================================================================
# 6. CLI Command Integration ('scan' and 'opt')
# ============================================================================

def test_cli_scan_1d(tmp_path):
    """Verify 'phatpsy scan' CLI execution."""
    out_json = tmp_path / "cli_scan.json"
    out_plot = tmp_path / "cli_scan.png"
    argv = [
        "scan",
        str(EXAMPLE_N2),
        "-b", "0,1",
        "-r", "2.0:2.1:0.1",
        "-o", str(out_json),
        "-p", str(out_plot),
    ]
    exit_code = cli_main(argv)
    assert exit_code == 0
    assert out_json.exists()
    assert out_plot.exists()


def test_cli_opt(tmp_path):
    """Verify 'phatpsy opt' CLI execution."""
    out_yaml = tmp_path / "cli_opt.yaml"
    out_traj = tmp_path / "cli_traj.json"
    out_plot = tmp_path / "cli_plot.png"
    argv = [
        "opt",
        str(EXAMPLE_N2),
        "-o", str(out_yaml),
        "-t", str(out_traj),
        "-p", str(out_plot),
        "--max-steps", "3",
    ]
    exit_code = cli_main(argv)
    assert exit_code == 0
    assert out_yaml.exists()
    assert out_traj.exists()
    assert out_plot.exists()
