# tests/test_neb.py
"""
Unit and integration tests for Nudged Elastic Band (NEB) and Climbing-Image NEB (CI-NEB)
transition state searching and reaction pathway optimization (Roadmap Section 4).
"""
import pytest
from pathlib import Path
import json
import math
import numpy as np

from phatpsy.schema import AtomConfig, CalculationDeck, ControlConfig
from phatpsy.builder import build_diatomic, build_molecule
from phatpsy.neb import (
    interpolate_images,
    calculate_tangents,
    calculate_neb_forces,
    neb_search,
    NEBResult,
    NEBImage,
)
from phatpsy.visualizer import generate_pathway_visualization_data, generate_pathway_html
from phatpsy.cli import main as cli_main

EXAMPLE_DIR = Path(__file__).parent.parent / "examples"


def test_interpolate_images_linear():
    """Verify linear coordinate interpolation across intermediate images."""
    deck_start = build_diatomic("N", "N", bond_length=2.0, units="bohr", basis="DZ", use_symmetry=False)
    deck_end = build_diatomic("N", "N", bond_length=3.0, units="bohr", basis="DZ", use_symmetry=False)

    images = interpolate_images(deck_start, deck_end, num_images=5)
    assert len(images) == 5

    # Check endpoints
    assert images[0].atoms[0].xyz[2] == pytest.approx(0.0, abs=1e-8)
    assert images[0].atoms[1].xyz[2] == pytest.approx(2.0, abs=1e-8)

    assert images[-1].atoms[0].xyz[2] == pytest.approx(0.0, abs=1e-8)
    assert images[-1].atoms[1].xyz[2] == pytest.approx(3.0, abs=1e-8)

    # Check midpoint (image 2: alpha=0.5 -> bond_length=2.5)
    assert images[2].atoms[0].xyz[2] == pytest.approx(0.0, abs=1e-8)
    assert images[2].atoms[1].xyz[2] == pytest.approx(2.5, abs=1e-8)

    # Basis and atom identity preserved
    for img in images:
        assert img.atoms[0].element == "N"
        assert img.atoms[1].element == "N"
        assert "N" in img.elements
        assert img.elements["N"].basis == "DZ"


def test_interpolate_images_validation_errors():
    """Verify error checking for mismatched atom counts or elements."""
    deck1 = build_diatomic("N", "N", bond_length=2.0)
    deck2 = build_diatomic("C", "O", bond_length=2.0)

    # Element mismatch
    with pytest.raises(ValueError, match="Element mismatch"):
        interpolate_images(deck1, deck2, num_images=5)

    # Atom count mismatch
    deck3 = CalculationDeck(
        title="Triatomic",
        control=ControlConfig(),
        geometry=[
            AtomConfig("N", [0.0, 0.0, 0.0]),
            AtomConfig("N", [0.0, 0.0, 1.0]),
            AtomConfig("N", [0.0, 0.0, 2.0]),
        ],
    )
    with pytest.raises(ValueError, match="Atom count mismatch"):
        interpolate_images(deck1, deck3, num_images=5)

    # Minimum images
    with pytest.raises(ValueError, match="num_images must be at least 3"):
        interpolate_images(deck1, deck1, num_images=2)


def test_calculate_tangents_improved():
    """Verify Henkelman-Jónsson upwind tangent evaluation."""
    # 3 collinear images along z-axis
    c0 = np.array([0.0, 0.0, 0.0])
    c1 = np.array([0.0, 0.0, 1.0])
    c2 = np.array([0.0, 0.0, 2.0])
    coords = [c0, c1, c2]

    # Case 1: Monotonically increasing energy (uphill: E0 < E1 < E2)
    energies_uphill = [0.0, 1.0, 2.0]
    tangents = calculate_tangents(coords, energies_uphill)
    np.testing.assert_allclose(tangents[1], [0.0, 0.0, 1.0])

    # Case 2: Monotonically decreasing energy (downhill: E0 > E1 > E2)
    energies_downhill = [2.0, 1.0, 0.0]
    tangents = calculate_tangents(coords, energies_downhill)
    np.testing.assert_allclose(tangents[1], [0.0, 0.0, 1.0])

    # Case 3: Energy peak at middle image (E1 > E0 and E1 > E2)
    energies_peak = [0.0, 2.0, 1.0]
    tangents = calculate_tangents(coords, energies_peak)
    # Unit vector along z
    assert np.linalg.norm(tangents[1]) == pytest.approx(1.0, abs=1e-8)
    assert tangents[1][2] > 0.0


def test_calculate_neb_forces_spring_and_climbing():
    """Verify perpendicular force projection, spring forces, and climbing-image force inversion."""
    c0 = np.array([0.0, 0.0, 0.0])
    c1 = np.array([0.0, 0.0, 1.0])
    c2 = np.array([0.0, 0.0, 2.5])  # Unequal spacing to induce non-zero spring force
    coords = [c0, c1, c2]
    energies = [0.0, 1.0, 0.5]

    tangents = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0])]

    # Potential gradient: has both parallel (z) and perpendicular (x) components
    grad1 = np.array([2.0, 0.0, 3.0])  # True force = -grad1 = [-2, 0, -3]
    gradients = [np.zeros(3), grad1, np.zeros(3)]

    k_spring = 0.5
    # For standard image 1:
    # d_plus = 1.5, d_minus = 1.0 -> d_plus - d_minus = 0.5
    # F_spring = 0.5 * 0.5 * [0, 0, 1] = [0, 0, 0.25]
    # F_true = [-2, 0, -3]
    # F_perp = F_true - (F_true . tau) * tau = [-2, 0, -3] - (-3)*[0,0,1] = [-2, 0, 0]
    # Total standard force = F_perp + F_spring = [-2, 0, 0.25]
    tot_f, true_f, spring_f = calculate_neb_forces(
        coords, energies, gradients, tangents, k_spring=k_spring, climbing_index=None
    )
    np.testing.assert_allclose(true_f[1], [-2.0, 0.0, -3.0])
    np.testing.assert_allclose(spring_f[1], [0.0, 0.0, 0.25])
    np.testing.assert_allclose(tot_f[1], [-2.0, 0.0, 0.25])

    # Now treat image 1 as climbing image:
    # Spring force is 0.
    # F_ci = F_true - 2 * (F_true . tau) * tau = [-2, 0, -3] - 2*(-3)*[0,0,1] = [-2, 0, +3]
    tot_ci, true_ci, spring_ci = calculate_neb_forces(
        coords, energies, gradients, tangents, k_spring=k_spring, climbing_index=1
    )
    np.testing.assert_allclose(spring_ci[1], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(tot_ci[1], [-2.0, 0.0, 3.0])


def test_neb_result_export_and_visualization(tmp_path):
    """Verify NEBResult dictionary serialization, multi-frame XYZ export, and 3D HTML visualization."""
    deck_start = build_diatomic("N", "N", bond_length=2.0)
    deck_mid = build_diatomic("N", "N", bond_length=2.5)
    deck_end = build_diatomic("N", "N", bond_length=3.0)

    res = NEBResult(
        converged=True,
        num_iterations=12,
        images=[deck_start, deck_mid, deck_end],
        energies=[-109.89, -109.80, -109.85],
        path_distances=[0.0, 0.5, 1.0],
        barrier_forward=0.09,
        barrier_reverse=0.05,
        reaction_energy=0.04,
        transition_state_index=1,
        transition_state_deck=deck_mid,
        transition_state_energy=-109.80,
        climbing_image_enabled=True,
    )

    # Conversion to kcal/mol properties
    assert res.barrier_forward_kcal == pytest.approx(0.09 * 627.509474, abs=1e-2)
    assert res.barrier_reverse_kcal == pytest.approx(0.05 * 627.509474, abs=1e-2)

    # JSON export
    json_path = tmp_path / "neb_path.json"
    res.save_json(json_path)
    assert json_path.exists()
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["converged"] is True
    assert data["transition_state_index"] == 1
    assert len(data["images"]) == 3

    # Multi-frame XYZ export
    xyz_path = tmp_path / "neb_path.xyz"
    res.save_xyz(xyz_path)
    assert xyz_path.exists()
    content = xyz_path.read_text(encoding="utf-8")
    assert "Transition State" in content
    assert content.count("2\n") >= 3  # 2 atoms per frame, 3 frames

    # Plot pathway
    plot_path = tmp_path / "neb_curve.png"
    res.plot_pathway(output_path=plot_path)
    assert plot_path.exists()
    assert plot_path.stat().st_size > 5000

    # 3D Pathway HTML generation
    html_path = tmp_path / "neb_view.html"
    res.visualize(output_path=html_path, open_browser=False)
    assert html_path.exists()
    html_text = html_path.read_text(encoding="utf-8")
    assert "Three.js" in html_text or "three.min.js" in html_text
    assert "timeline-controls" in html_text
    assert "NEB Reaction Pathway" in html_text


def test_neb_search_n2_stretch():
    """
    Integration test: run CI-NEB on diatomic nitrogen bond extension (2.0 to 2.4 Bohrs) with 3 images.
    """
    deck_r = build_diatomic("N", "N", bond_length=2.068, basis="DZ")
    deck_p = build_diatomic("N", "N", bond_length=2.300, basis="DZ")

    # Fast run with 3 images and 2 iterations to exercise the optimizer loop
    res = neb_search(
        initial_deck=deck_r,
        final_deck=deck_p,
        num_images=3,
        k_spring=0.1,
        max_iterations=2,
        force_tol=0.005,
        climb=True,
    )

    assert len(res.images) == 3
    assert len(res.energies) == 3
    assert res.transition_state_index in (0, 1, 2)
    assert res.barrier_forward >= 0.0
    assert len(res.path_distances) == 3
    assert res.path_distances[0] == 0.0
    assert res.path_distances[-1] > 0.0


def test_cli_neb(tmp_path):
    """Verify 'phatpsy neb' CLI command execution."""
    yaml_r = tmp_path / "reactant.yaml"
    yaml_p = tmp_path / "product.yaml"
    out_json = tmp_path / "neb_out.json"
    out_xyz = tmp_path / "neb_out.xyz"
    out_plot = tmp_path / "neb_out.png"

    deck_r = build_diatomic("N", "N", bond_length=2.068, basis="DZ")
    deck_p = build_diatomic("N", "N", bond_length=2.200, basis="DZ")
    deck_r.to_yaml(yaml_r)
    deck_p.to_yaml(yaml_p)

    argv = [
        "neb",
        str(yaml_r),
        str(yaml_p),
        "-n", "3",
        "--max-iter", "2",
        "-o", str(out_json),
        "--xyz", str(out_xyz),
        "-p", str(out_plot),
    ]

    exit_code = cli_main(argv)
    assert exit_code == 0
    assert out_json.exists()
    assert out_xyz.exists()
    assert out_plot.exists()


def test_neb_molecular_eo_to_aa_isomerization():
    """
    Verify CI-NEB pathway initialization and force evaluation on the non-linear,
    polyatomic isomerization of Ethylene Oxide (c-C2H4O) to Acetaldehyde (CH3CHO).
    """
    from phatpsy.driver import PhatpsyDriver

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

    deck_r = build_molecule(
        geom_eo,
        basis="SZ",
        title="Ethylene Oxide (Reactant)",
        spin_unrestricted=True,
        auto_spin_flip=True,
    )
    deck_p = build_molecule(
        geom_aa,
        basis="SZ",
        title="Acetaldehyde (Product)",
        spin_unrestricted=True,
        auto_spin_flip=True,
    )
    # Minimal Single-Zeta basis requires pk_scale=0.0 to prevent static V_PK over-projection
    # on multi-center C-H bonds, matching eo_to_aa.json legacy_calibrated_pathway
    deck_r.control.pk_scale = 0.0
    deck_p.control.pk_scale = 0.0

    driver = PhatpsyDriver(quiet=True)
    res = neb_search(
        initial_deck=deck_r,
        final_deck=deck_p,
        driver=driver,
        num_images=3,
        max_iterations=1,
        climb=True,
        quiet=True,
    )

    # 3 images along 7-atom reaction coordinate
    assert len(res.images) == 3
    assert res.num_iterations >= 1
    assert res.climbing_image_enabled is True
    assert res.transition_state_index == 1
    assert res.barrier_forward_kcal is not None

    # Verify trajectory structures export
    d = res.to_dict()
    assert len(d["images"]) == 3
    assert d["climbing_image_enabled"] is True

