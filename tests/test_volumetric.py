# tests/test_volumetric.py
"""
Unit and regression tests for 3D volumetric evaluation, Gaussian Cube I/O,
marching tetrahedra polygonizer, and property-colored isosurfaces.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile

from phatpsy.volumetric import (
    VolumetricGrid,
    compute_grid,
    compute_esp_grid,
    compute_deformation_density,
    evaluate_real_sto,
)
from phatpsy.marching_cubes import marching_cubes, extract_isosurface
from phatpsy.visualizer import (
    generate_html,
    visualize_grid,
    map_rwb_colors,
    generate_grid_visualization_data,
)
from phatpsy.builder import build_diatomic
from phatpsy.driver import PhatpsyDriver


def test_volumetric_grid_arithmetic_and_integration():
    """Validates 3D grid properties, volume element dV, arithmetic and numerical integration."""
    data1 = np.ones((10, 10, 10))
    origin = np.array([-1.0, -1.0, -1.0])
    voxel_steps = np.eye(3) * 0.2
    atoms = [{"element": "H", "z": 1.0, "xyz": [0.0, 0.0, 0.0]}]

    grid1 = VolumetricGrid(data=data1, origin=origin, voxel_steps=voxel_steps, atoms=atoms)
    assert grid1.dimensions == (10, 10, 10)
    assert pytest.approx(grid1.voxel_volume, rel=1e-6) == 0.008  # 0.2^3

    # Numerical integral = 1000 * 0.008 = 8.0
    assert pytest.approx(grid1.integrate(), rel=1e-6) == 8.0

    # Grid addition and multiplication
    grid2 = grid1 * 2.5
    assert pytest.approx(grid2.integrate(), rel=1e-6) == 20.0

    grid_sum = grid1 + grid2
    assert pytest.approx(grid_sum.integrate(), rel=1e-6) == 28.0

    grid_diff = grid2 - grid1
    assert pytest.approx(grid_diff.integrate(), rel=1e-6) == 12.0


def test_cube_io_roundtrip():
    """Ensures Gaussian Cube export and parsing reproduce identical dimensions, atoms, and values."""
    shape = (12, 14, 16)
    x = np.linspace(-2, 2, shape[0])
    y = np.linspace(-2, 2, shape[1])
    z = np.linspace(-2, 2, shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    data = np.exp(-(X**2 + Y**2 + Z**2))

    origin = np.array([-2.0, -2.0, -2.0])
    voxel_steps = np.diag([4.0 / 11, 4.0 / 13, 4.0 / 15])
    atoms = [
        {"element": "C", "z": 6, "charge": 6.0, "xyz": [0.0, 0.0, 0.0]},
        {"element": "O", "z": 8, "charge": 8.0, "xyz": [0.0, 0.0, 2.1]},
    ]

    grid_orig = VolumetricGrid(
        data=data,
        origin=origin,
        voxel_steps=voxel_steps,
        atoms=atoms,
        title="Test Cube Roundtrip",
    )

    with tempfile.NamedTemporaryFile("w+", suffix=".cube") as tf:
        cube_str = grid_orig.to_cube(tf.name)
        grid_parsed = VolumetricGrid.from_cube(tf.name)

    assert grid_parsed.dimensions == grid_orig.dimensions
    assert np.allclose(grid_parsed.origin, grid_orig.origin, atol=1e-5)
    assert np.allclose(grid_parsed.voxel_steps, grid_orig.voxel_steps, atol=1e-5)
    assert len(grid_parsed.atoms) == 2
    assert grid_parsed.atoms[0]["z"] == 6
    assert grid_parsed.atoms[1]["z"] == 8
    assert np.allclose(grid_parsed.data, grid_orig.data, atol=1e-4)


def _small_cube_grid():
    ax = np.linspace(-6, 6, 11)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    psi = evaluate_real_sto(X, Y, Z, (0.0, 0.0, 0.0), 1, 0, 0, 1.0, 2.0)
    return VolumetricGrid(
        data=psi,
        origin=np.array([-6.0, -6.0, -6.0]),
        voxel_steps=np.diag([ax[1] - ax[0]] * 3),
        atoms=[{"element": "H", "xyz": (0.0, 0.0, 0.0)}],
        title="test",
    )


def test_cube_roundtrip_from_text():
    """from_cube(grid.to_cube()) must parse raw cube text, not treat it as a path."""
    grid = _small_cube_grid()
    parsed = VolumetricGrid.from_cube(grid.to_cube())
    assert parsed.data.shape == grid.data.shape
    assert np.max(np.abs(parsed.data - grid.data)) < 1e-5


def test_cube_roundtrip_from_stringio_and_path(tmp_path):
    """The StringIO, str-path and Path routes all still load identically."""
    import io

    grid = _small_cube_grid()
    text = grid.to_cube()
    cube_file = tmp_path / "g.cube"
    cube_file.write_text(text, encoding="utf-8")

    for src in (io.StringIO(text), str(cube_file), cube_file):
        parsed = VolumetricGrid.from_cube(src)
        assert parsed.data.shape == grid.data.shape
        assert np.max(np.abs(parsed.data - grid.data)) < 1e-5


def test_real_sto_basis_evaluation():
    """Validates spatial evaluation of real Slater-Type Orbitals (s, p, d)."""
    grid_pts = np.linspace(-3.0, 3.0, 31)
    X, Y, Z = np.meshgrid(grid_pts, grid_pts, grid_pts, indexing="ij")

    # 1s orbital centered at origin, zeta=1.0
    sto_1s = evaluate_real_sto(X, Y, Z, (0.0, 0.0, 0.0), n=1, l=0, m=0, zeta=1.0)
    assert sto_1s.shape == (31, 31, 31)
    # Peak at origin
    assert sto_1s[15, 15, 15] > sto_1s[14, 15, 15]
    # s orbital is spherically symmetric: f(x) = f(-x)
    assert np.isclose(sto_1s[12, 15, 15], sto_1s[18, 15, 15])

    # 2pz orbital: nodal plane at Z=0
    sto_2pz = evaluate_real_sto(X, Y, Z, (0.0, 0.0, 0.0), n=2, l=1, m=0, zeta=1.0)
    assert np.isclose(sto_2pz[15, 15, 15], 0.0, atol=1e-10)
    # Anti-symmetric across Z=0: f(x, y, z) = -f(x, y, -z)
    assert np.isclose(sto_2pz[15, 15, 12], -sto_2pz[15, 15, 18])


def test_marching_cubes_isosurface_extraction():
    """Tests zero-dependency Marching Tetrahedra polygonizer on a known analytical sphere."""
    grid_pts = np.linspace(-2.0, 2.0, 21)
    X, Y, Z = np.meshgrid(grid_pts, grid_pts, grid_pts, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)

    # Sphere of radius 1.0 (R = 1.0)
    verts, faces, normals = marching_cubes(
        R,
        isovalue=1.0,
        spacing=(4.0 / 20, 4.0 / 20, 4.0 / 20),
        origin=(-2.0, -2.0, -2.0),
    )

    assert len(verts) > 0
    assert len(faces) > 0
    assert len(normals) == len(verts)

    # Check that all vertices are on sphere of radius ~1.0
    radii = np.linalg.norm(verts, axis=1)
    assert np.allclose(radii, 1.0, atol=0.1)

    # Check that normals are normalized unit vectors and correlate with radial direction
    norm_lengths = np.linalg.norm(normals, axis=1)
    assert np.allclose(norm_lengths, 1.0, atol=1e-5)
    dots = np.sum(verts * normals, axis=1)
    assert np.mean(np.abs(dots)) > 0.5



def test_trilinear_interpolation():
    """Tests trilinear interpolation of scalar fields at arbitrary coordinates."""
    data = np.arange(27, dtype=np.float64).reshape((3, 3, 3))
    grid = VolumetricGrid(data=data, origin=np.array([0.0, 0.0, 0.0]), voxel_steps=np.eye(3))

    # Center of voxel (0.5, 0.5, 0.5)
    val = grid.interpolate(np.array([[0.5, 0.5, 0.5]]))[0]
    expected = (
        data[0, 0, 0] + data[1, 0, 0] + data[0, 1, 0] + data[1, 1, 0] +
        data[0, 0, 1] + data[1, 0, 1] + data[0, 1, 1] + data[1, 1, 1]
    ) / 8.0
    assert pytest.approx(val, rel=1e-5) == expected


def test_red_white_blue_colormap_mapping():
    """Validates Red-White-Blue RGB colormap for charges and electrostatic potentials."""
    vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    colors, v_max = map_rwb_colors(vals, v_max=2.0)
    assert v_max == 2.0
    assert len(colors) == 5 * 3  # 5 vertices x 3 RGB channels

    # First point (-2.0, max negative): Red (1.0, ~0.15, ~0.15)
    assert colors[0] == 1.0
    assert colors[1] < 0.2
    assert colors[2] < 0.2

    # Third point (0.0, neutral): White (1.0, 1.0, 1.0)
    assert colors[6] == 1.0
    assert colors[7] == 1.0
    assert colors[8] == 1.0

    # Fifth point (+2.0, max positive): Blue (~0.15, ~0.40, 1.0)
    assert colors[12] < 0.2
    assert colors[13] < 0.5
    assert colors[14] == 1.0


def test_n2_volumetric_electron_density_and_orbitals():
    """Runs N2 diatomic calculation and evaluates density, MO, deformation, and ESP."""
    deck = build_diatomic("N", "N", bond_length=2.068, units="bohr", basis="DZ")
    driver = PhatpsyDriver()
    res = driver.run_deck(deck)
    assert res is not None

    # 1. Total Density
    grid_dens = compute_grid(res, grid_type="density", spacing=0.3, padding=2.5)
    assert grid_dens.dimensions[0] > 10
    # Integral of density is positive
    int_rho = grid_dens.integrate()
    assert int_rho > 5.0  # Captures dominant electron distribution on regular grid

    # 2. Molecular Orbital (HOMO)
    grid_homo = compute_grid(res, grid_type="mo", mo_index=7, spacing=0.3, padding=2.5)
    # Orbital has both positive and negative phases
    assert grid_homo.min_value < -1e-3
    assert grid_homo.max_value > 1e-3

    # 3. Deformation Density
    grid_def = compute_deformation_density(res, spacing=0.3, padding=2.5)
    assert grid_def.min_value < 0.0  # Depletion in atomic cores
    assert grid_def.max_value > 0.0  # Accumulation in bond region

    # 4. Electrostatic Potential (ESP)
    grid_esp = compute_esp_grid(res, spacing=0.3, padding=2.5)
    assert grid_esp.dimensions == grid_dens.dimensions
    # Nuclear regions are strongly positive
    assert grid_esp.max_value > 1.0

    # 5. Property-colored isosurface mesh generation & HTML export
    html = generate_html(grid_dens, isovalue=0.03, property_grid=grid_esp)
    assert "Mapped Property" in html
    assert "Red/Nucleophilic" in html
    assert "Blue/Electrophilic" in html
    assert "MOL_DATA" in html


def test_cli_cube_subcommand():
    """Tests CLI 'phatpsy cube' subcommand execution on an example calculation deck."""
    from phatpsy.cli import main

    with tempfile.TemporaryDirectory() as tmpdir:
        cube_path = Path(tmpdir) / "n2_dens.cube"
        html_path = Path(tmpdir) / "n2_view.html"

        # Export Gaussian Cube
        rc1 = main(["cube", "examples/n2sz.yaml", "--density", "-o", str(cube_path), "--spacing", "0.4"])
        assert rc1 == 0
        assert cube_path.exists()
        assert cube_path.stat().st_size > 1000

        # Export HTML with property-colored ESP
        rc2 = main(["cube", "examples/n2sz.yaml", "--esp", "-o", str(html_path), "--spacing", "0.4", "-n"])
        assert rc2 == 0
        assert html_path.exists()
        assert "3D Isosurface Envelope" in html_path.read_text(encoding="utf-8")
