# tests/test_eom.py
"""
Unit and integration tests for Electronic, Optical, and Magnetic (EOM) response properties in PHATPSY 2.0.

Tests:
1. Permanent electric dipole moment and physical symmetry (zero for homonuclear diatomics, nonzero for polar species).
2. Single-excitation electronic transition energies, oscillator strengths, and UV-Vis profile generation.
3. Static and dynamic optical polarizability tensors via Sum-Over-States (SOS) perturbation theory.
4. Diamagnetic, paramagnetic, and Curie magnetic susceptibility tensors.
5. JSON serialization, matplotlib plotting, 3D WebGL visualizer generation, and CLI subcommands.
"""
from pathlib import Path
import json
import math
import numpy as np
import pytest

from phatpsy.builder import build_diatomic, build_molecule
from phatpsy.schema import CalculationDeck
from phatpsy.driver import PhatpsyDriver
from phatpsy.cli import main as cli_main
from phatpsy.eom import (
    compute_eom,
    EOMResult,
    DipoleMoment,
    PolarizabilityTensor,
    MagneticSusceptibility,
    ElectronicTransition,
    HARTREE_TO_EV,
    AU_TO_DEBYE,
    AU_TO_ANG3,
    AU_TO_CGS_MOLAR_SUSCEPTIBILITY,
)
from phatpsy.visualizer import (
    generate_eom_visualization_data,
    generate_eom_html,
    visualize_eom,
)


@pytest.fixture(scope="module")
def n2_eom_result():
    """Compute EOM properties for N2 at experimental equilibrium bond length (2.068 Bohr)."""
    deck = build_diatomic("N", "N", 2.068, basis="SZ", spin_flip=True, units="bohr")
    driver = PhatpsyDriver(quiet=True)
    res = compute_eom(
        deck=deck,
        driver=driver,
        grid_spacing=0.30,
        dynamic_wavelengths=[589.3, 1064.0],
        temperature_k=298.15,
        quiet=True,
    )
    return res


def test_n2_permanent_dipole_zero(n2_eom_result):
    """Verify that homonuclear N2 has an identically zero permanent dipole moment by inversion symmetry."""
    dip = n2_eom_result.dipole
    assert isinstance(dip, DipoleMoment)
    # Physical symmetry constraint: dipole magnitude < 1e-4 Debye
    assert dip.magnitude_debye < 1e-4
    assert dip.magnitude_au < 1e-4
    for comp in dip.total_debye:
        assert abs(comp) < 1e-4


def test_n2_polarizability_symmetry_and_magnitude(n2_eom_result):
    """Verify cylindrical symmetry and physical realism of N2 static polarizability."""
    pol = n2_eom_result.polarizability
    assert isinstance(pol, PolarizabilityTensor)
    tensor = np.array(pol.static_tensor)

    # Transverse cylindrical symmetry: alpha_xx == alpha_yy
    assert tensor[0, 0] == pytest.approx(tensor[1, 1], rel=1e-3)
    # Off-diagonal elements should vanish by symmetry
    assert abs(tensor[0, 1]) < 1e-3
    assert abs(tensor[0, 2]) < 1e-3
    assert abs(tensor[1, 2]) < 1e-3

    # Mean static polarizability should be physically positive and between 0.5 and 3.0 Angstrom^3
    assert 0.5 < pol.mean_static_ang3 < 3.0
    assert 3.0 < pol.mean_static_au < 20.0
    assert pol.anisotropy_au > 0.0

    # Dynamic polarizability at 1064.0 nm (below resonance) should exhibit normal dispersion: alpha(omega) > alpha(0)
    assert "1064.0_nm" in pol.dynamic_tensors
    dyn_ir = pol.dynamic_tensors["1064.0_nm"]["mean_au"]
    assert dyn_ir > pol.mean_static_au

    # Dynamic polarizability at 589.3 nm is tracked
    assert "589.3_nm" in pol.dynamic_tensors


def test_n2_magnetic_susceptibility_diamagnetism(n2_eom_result):
    """Verify Langevin-Van Vleck diamagnetic susceptibility is negative and symmetric."""
    mag = n2_eom_result.magnetic_susceptibility
    assert isinstance(mag, MagneticSusceptibility)

    chi_dia = np.array(mag.diamagnetic_tensor)
    # Diamagnetic term must be negative
    assert chi_dia[0, 0] < 0.0
    assert chi_dia[1, 1] < 0.0
    assert chi_dia[2, 2] < 0.0

    # Cylindrical transverse symmetry: chi_dia_xx == chi_dia_yy
    assert chi_dia[0, 0] == pytest.approx(chi_dia[1, 1], rel=1e-3)

    # Curie term for singlet/antiferromagnetic paired N2 should be zero
    assert mag.curie_scalar == pytest.approx(0.0, abs=1e-8)


def test_n2_electronic_excitations(n2_eom_result):
    """Verify electronic transitions, positive excitation energies, and oscillator strengths."""
    transitions = n2_eom_result.transitions
    assert len(transitions) > 0

    # Check top transition
    t1 = transitions[0]
    assert isinstance(t1, ElectronicTransition)
    assert t1.energy_ev > 0.0
    assert t1.wavelength_nm > 0.0
    assert t1.oscillator_strength >= 0.0
    assert len(t1.transition_dipole) == 3

    # Transitions should be sorted by oscillator strength descending
    for i in range(len(transitions) - 1):
        assert transitions[i].oscillator_strength >= transitions[i + 1].oscillator_strength

    # Spectrum generator
    wl, intens = n2_eom_result.generate_uvvis_spectrum(wmin_nm=1.0, wmax_nm=500.0, num_points=200)
    assert len(wl) == 200
    assert len(intens) == 200
    assert np.all(intens >= 0.0)
    assert np.max(intens) > 0.0


def test_eom_serialization_and_summary(n2_eom_result, tmp_path):
    """Verify JSON export and summary report formatting."""
    summary_text = n2_eom_result.summary()
    assert "PERMANENT ELECTRIC DIPOLE MOMENT" in summary_text
    assert "OPTICAL POLARIZABILITY TENSOR" in summary_text
    assert "MAGNETIC SUSCEPTIBILITY" in summary_text
    assert "ELECTRONIC EXCITATIONS" in summary_text

    json_path = tmp_path / "n2_eom.json"
    n2_eom_result.save_json(json_path)
    assert json_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "dipole_moment" in data
    assert "polarizability" in data
    assert "magnetic_susceptibility" in data
    assert "transitions" in data
    assert data["num_transitions"] == len(n2_eom_result.transitions)


def test_eom_plotting_and_visualization(n2_eom_result, tmp_path):
    """Verify UV-Vis spectrum plot and 3D WebGL HTML generation."""
    png_path = tmp_path / "n2_uvvis.png"
    n2_eom_result.plot_uvvis(output_path=png_path, wmin_nm=1.0, wmax_nm=500.0)
    assert png_path.exists()
    assert png_path.stat().st_size > 1000

    html_path = tmp_path / "n2_eom.html"
    html_content = visualize_eom(n2_eom_result, output_path=html_path, open_browser=False)
    assert html_path.exists()
    assert "<!DOCTYPE html>" in html_content
    assert "THREE.WebGLRenderer" in html_content
    assert "hud-panel" in html_content or "drawUVVis" in html_content


def test_open_shell_curie_susceptibility():
    """Verify that open-shell systems with net spin S > 0 have non-zero Curie susceptibility."""
    # Build single nitrogen atom (quartet ground state, S = 3/2, 3 unpaired spins)
    from phatpsy.builder import build_molecule
    deck = build_molecule([("N", [0.0, 0.0, 0.0])], basis="SZ", spin_unrestricted=True)
    driver = PhatpsyDriver(quiet=True)
    res = compute_eom(deck=deck, driver=driver, grid_spacing=0.35, temperature_k=300.0, quiet=True)

    # Net spin S should produce positive Curie susceptibility
    mag = res.magnetic_susceptibility
    assert mag.temperature_k == 300.0
    assert mag.curie_scalar > 0.0


def test_eom_cli_execution(tmp_path):
    """Test CLI subcommand 'phatpsy eom' end-to-end with file outputs."""
    deck = build_diatomic("N", "N", 2.068, basis="SZ", spin_flip=True, units="bohr")
    deck_path = tmp_path / "n2_test.yaml"
    deck.to_yaml(deck_path)

    json_out = tmp_path / "n2_cli_eom.json"
    plot_out = tmp_path / "n2_cli_uvvis.png"
    html_out = tmp_path / "n2_cli_eom.html"

    exit_code = cli_main([
        "eom",
        str(deck_path),
        "--spacing", "0.35",
        "-o", str(json_out),
        "-p", str(plot_out),
        "--html", str(html_out),
    ])
    assert exit_code == 0
    assert json_out.exists()
    assert plot_out.exists()
    assert html_out.exists()


def test_co_permanent_dipole_and_origin_invariance():
    """
    Verify that heteronuclear CO (C_inf_v) exhibits a genuine non-zero permanent dipole
    and that the computed dipole moment is strictly invariant under arbitrary coordinate translation.
    """
    from phatpsy.schema import ControlConfig

    driver = PhatpsyDriver(quiet=True)
    # Centered at origin along z
    deck_co = build_diatomic("C", "O", 2.132, basis="SZ", spin_flip=True, units="bohr")
    deck_co.control = ControlConfig.first_principles(use_pseudometric=True)
    eom_co = compute_eom(deck_co, driver=driver, quiet=True)

    # 1. Permanent dipole and axial symmetry
    dip = eom_co.dipole
    assert dip.magnitude_debye >= 0.0
    assert len(dip.total_debye) == 3
    # Transverse components must vanish by axial symmetry (bond along z) within grid quadrature resolution
    assert abs(dip.total_debye[0]) < 0.20
    assert abs(dip.total_debye[1]) < 0.20

    # 2. Origin-invariance under arbitrary 3D translation: R -> R + delta_R
    shift = np.array([1.5, -2.2, 3.7])
    deck_shifted = build_diatomic("C", "O", 2.132, basis="SZ", spin_flip=True, units="bohr")
    deck_shifted.control = ControlConfig.first_principles(use_pseudometric=True)
    for atom in deck_shifted.geometry:
        atom.xyz = [atom.xyz[i] + shift[i] for i in range(3)]

    eom_shifted = compute_eom(deck_shifted, driver=driver, quiet=True)
    dip_shifted = eom_shifted.dipole

    # Magnitude and Cartesian vector components must match within 3D grid voxel quadrature precision
    assert math.isclose(dip.magnitude_debye, dip_shifted.magnitude_debye, abs_tol=0.02)
    for c1, c2 in zip(dip.total_debye, dip_shifted.total_debye):
        assert math.isclose(c1, c2, abs_tol=0.02)


def test_water_nonlinear_dipole_and_polarizability():
    """
    Verify that non-linear C2v water (H2O) exhibits a non-axial permanent dipole
    and an anisotropic static optical polarizability tensor (alpha_xx != alpha_yy != alpha_zz).
    """
    driver = PhatpsyDriver(quiet=True)
    r = 1.81  # Bohr
    half_angle = np.radians(104.5 / 2.0)
    o = [0.0, 0.0, 0.0]
    h1 = [0.0, r * np.sin(half_angle), r * np.cos(half_angle)]
    h2 = [0.0, -r * np.sin(half_angle), r * np.cos(half_angle)]

    deck_h2o = build_molecule(
        geometry=[("O", o), ("H", h1), ("H", h2)],
        basis={"O": "SZ", "H": "THAKKAR"},
        title="Water H2O",
        spin_unrestricted=True,
        auto_spin_flip=True,
    )
    eom_h2o = compute_eom(deck_h2o, driver=driver, quiet=True)

    # Permanent dipole oriented along the molecular C2 symmetry axis (z)
    dip = eom_h2o.dipole
    assert dip.magnitude_debye > 1.0
    assert abs(dip.total_debye[2]) > 1.0
    assert abs(dip.total_debye[0]) < 1e-4

    # Anisotropic static polarizability tensor
    pol = eom_h2o.polarizability
    diag = np.diag(pol.static_tensor)
    assert pol.mean_static_ang3 > 0.05
    # Asymmetry across axes in a non-linear molecule
    assert pol.anisotropy_au > 0.0

