# tests/test_vibrations.py
"""
Unit and integration tests for harmonic vibrational frequency analysis, normal modes,
Eckart translational & rotational projection, and RRHO statistical thermochemistry in PHATPSY 2.0.
"""
from pathlib import Path
import json
import math
import copy
import numpy as np
import pytest

from phatpsy.builder import build_diatomic, build_molecule
from phatpsy.schema import CalculationDeck, AtomConfig
from phatpsy.driver import PhatpsyDriver
from phatpsy.cli import main as cli_main
from phatpsy.vibrations import (
    compute_vibrations,
    compute_hessian,
    compute_thermochemistry,
    project_translations_rotations,
    NormalMode,
    Thermochemistry,
    VibrationResult,
    FREQ_CONV_FACTOR,
    FORCE_CONSTANT_CONV,
)
from phatpsy.visualizer import (
    generate_vibrations_visualization_data,
    generate_vibrations_html,
    visualize_vibrations,
)


def test_vibrational_physical_constants():
    """Verify fundamental vibrational conversion factors against IUPAC / CODATA standards."""
    # Frequency conversion factor from sqrt(Hartree / (Bohr^2 * amu)) to cm^-1:
    # 1 / (2 * pi * c) * sqrt(E_h / (a_0^2 * amu)) ≈ 5140.487 cm^-1
    assert FREQ_CONV_FACTOR == pytest.approx(5140.487, rel=1e-5)

    # Force constant conversion factor from Hartree / Bohr^2 to mDyne / Å:
    # 1 Hartree / Bohr^2 ≈ 15.56893 mDyne / Å
    assert FORCE_CONSTANT_CONV == pytest.approx(15.56893, rel=1e-4)


def test_project_translations_rotations_linear():
    """Verify Eckart translational and rotational projection on a linear molecule (N_tr = 5)."""
    # 2 nitrogen atoms along z-axis
    m_n = 14.007
    masses = np.array([m_n, m_n])
    coords = np.array([[0.0, 0.0, 0.700], [0.0, 0.0, -0.700]])

    # Arbitrary symmetric 6x6 test matrix
    rng = np.random.default_rng(123)
    A = rng.normal(size=(6, 6))
    H_test = 0.5 * (A + A.T)

    H_proj, P, n_tr = project_translations_rotations(H_test, coords, masses, is_linear=True)

    # Exactly 5 translation/rotation degrees of freedom for a linear diatomic
    assert n_tr == 5

    # Projector invariants: P^T = P (symmetric) and P^2 = P (idempotent)
    np.testing.assert_allclose(P, P.T, atol=1e-12)
    np.testing.assert_allclose(P @ P, P, atol=1e-12)

    # Rank of P must be 3N - 5 = 1
    rank_P = np.linalg.matrix_rank(P, tol=1e-8)
    assert rank_P == 1

    # Diagonalize H_proj: must have exactly 5 zero eigenvalues
    evals, _ = np.linalg.eigh(H_proj)
    # Sort absolute values
    abs_evals = np.sort(np.abs(evals))
    # First 5 eigenvalues should be zero within machine precision
    np.testing.assert_allclose(abs_evals[:5], np.zeros(5), atol=1e-10)


def test_project_translations_rotations_nonlinear():
    """Verify Eckart projection on a non-linear triatomic molecule (N_tr = 6)."""
    # Water-like geometry (C2v in XY plane)
    masses = np.array([15.999, 1.008, 1.008])  # O, H1, H2
    coords = np.array([
        [0.0, 0.0, 0.12],
        [0.0, 1.43, -0.96],
        [0.0, -1.43, -0.96],
    ])

    dim = 9
    rng = np.random.default_rng(456)
    A = rng.normal(size=(dim, dim))
    H_test = 0.5 * (A + A.T)

    H_proj, P, n_tr = project_translations_rotations(H_test, coords, masses, is_linear=False)

    # Non-linear molecule must have 6 translational and rotational modes
    assert n_tr == 6

    # Projector invariants
    np.testing.assert_allclose(P, P.T, atol=1e-12)
    np.testing.assert_allclose(P @ P, P, atol=1e-12)

    # Rank of P must be 3N - 6 = 3 (3 vibrational modes: symmetric stretch, bend, asymmetric stretch)
    rank_P = np.linalg.matrix_rank(P, tol=1e-8)
    assert rank_P == 3

    evals, _ = np.linalg.eigh(H_proj)
    abs_evals = np.sort(np.abs(evals))
    np.testing.assert_allclose(abs_evals[:6], np.zeros(6), atol=1e-10)


def test_thermochemistry_analytical_n2():
    """Verify RRHO statistical thermochemistry on standard N2 against benchmark values."""
    # N2 experimental bond length R = 1.0977 Å ≈ 2.074 Bohr, frequency = 2358.6 cm^-1
    m_n = 14.007
    masses = np.array([m_n, m_n])
    coords = np.array([[0.0, 0.0, 1.037], [0.0, 0.0, -1.037]])
    nu_exp = [2358.6]
    e_elec = -109.565398

    thermo = compute_thermochemistry(
        frequencies=nu_exp,
        coords=coords,
        masses=masses,
        electronic_energy=e_elec,
        spin_multiplicity=1,
        temperature_k=298.15,
        pressure_atm=1.0,
        symmetry_number=2,  # Homonuclear diatomic
    )

    # Sackur-Tetrode translational entropy at 298.15 K: ~35.92 cal/(mol·K)
    s_trans = thermo.entropy_cal_mol_k["translation"]
    assert s_trans == pytest.approx(35.92, abs=0.1)

    # Rotational entropy at 298.15 K for N2 (sigma = 2): ~9.8 cal/(mol·K)
    s_rot = thermo.entropy_cal_mol_k["rotation"]
    assert s_rot == pytest.approx(9.83, abs=0.15)

    # High-frequency vibrational mode (2358.6 cm^-1) has negligible vibrational entropy at 298 K
    s_vib = thermo.entropy_cal_mol_k["vibration"]
    assert s_vib < 0.05

    # Total entropy matches experimental gas-phase standard entropy (45.77 cal/(mol·K))
    s_total = thermo.entropy_cal_mol_k["total"]
    assert s_total == pytest.approx(45.76, abs=0.2)

    # Zero-Point Vibrational Energy: ZPVE = 1/2 * h * c * nu ≈ 3.37 kcal/mol
    assert thermo.zero_point_energy_kcal == pytest.approx(3.372, abs=0.02)

    # Thermodynamic consistency:
    # H(T) - U(T) = R * T
    r_cal = 1.987204
    rt_kcal = (r_cal * 298.15) / 1000.0
    assert (thermo.thermal_enthalpy_kcal - thermo.thermal_energy_kcal) == pytest.approx(rt_kcal, abs=1e-4)

    # G(T) = H(T) - T * S(T)
    hartree_to_kcal = 627.509474
    delta_g_kcal = (thermo.total_enthalpy_h_t - thermo.total_gibbs_free_energy_g_t) * hartree_to_kcal
    ts_kcal = (298.15 * s_total) / 1000.0
    assert delta_g_kcal == pytest.approx(ts_kcal, abs=1e-3)

    # Heat capacity: Cv_trans = 3/2 R, Cv_rot = R for linear molecule
    cv = thermo.heat_capacity_cal_mol_k
    assert cv["cv_trans"] == pytest.approx(1.5 * r_cal, abs=1e-3)
    assert cv["cv_rot"] == pytest.approx(1.0 * r_cal, abs=1e-3)


def get_n2_test_deck() -> CalculationDeck:
    """Helper to load canonical N2 deck with test-local SZ basis for fast numerical test."""
    deck = CalculationDeck.from_yaml("examples/n2.yaml")
    deck.elements["N"].basis = "SZ"
    deck.control.pk_scale = 0.0
    return deck


def test_compute_vibrations_n2_calculation():
    """Verify full end-to-end numerical Hessian and vibrational calculation on N2."""
    driver = PhatpsyDriver(quiet=True)
    deck0 = get_n2_test_deck()
    deck = copy.deepcopy(deck0)
    # Set coordinates near the bound minimum (R = 1.400 Bohr)
    deck.geometry[0].xyz = [0.0, 0.0, 0.700]
    deck.geometry[1].xyz = [0.0, 0.0, -0.700]

    vib_res = compute_vibrations(
        deck=deck,
        driver=driver,
        step_size=0.005,
        method="forward",
        project_tr=True,
        temperature_k=298.15,
        pressure_atm=1.0,
        quiet=True,
    )

    # 2 atoms, 6 total modes, exactly 1 genuine internal vibrational mode
    assert vib_res.num_atoms == 2
    assert vib_res.num_modes == 6
    assert vib_res.num_vibrational_modes == 1

    # Local minimum: 0 imaginary frequencies
    assert vib_res.num_imaginary == 0
    assert vib_res.is_minimum is True
    assert vib_res.is_transition_state is False

    # Check genuine stretch mode
    stretch_mode = vib_res.vibrational_modes[0]
    assert stretch_mode.frequency_cm > 8000.0
    assert stretch_mode.is_imaginary is False

    # The stretch mode displacements must be purely along the z-axis with opposite signs
    disp_n1 = stretch_mode.displacements[0]
    disp_n2 = stretch_mode.displacements[1]
    # x and y displacements should be near zero
    assert abs(disp_n1[0]) < 1e-4
    assert abs(disp_n1[1]) < 1e-4
    assert abs(disp_n2[0]) < 1e-4
    assert abs(disp_n2[1]) < 1e-4
    # z displacements have opposite signs
    assert disp_n1[2] * disp_n2[2] < 0.0
    assert abs(abs(disp_n1[2]) - abs(disp_n2[2])) < 1e-4

    # Thermochemistry attached
    assert vib_res.thermochemistry is not None
    assert vib_res.thermochemistry.zero_point_energy_kcal > 10.0

    # Summary formatting
    summary_text = vib_res.summary()
    assert "HARMONIC VIBRATIONAL FREQUENCY ANALYSIS" in summary_text
    assert "THERMOCHEMICAL ANALYSIS" in summary_text


def test_vibrations_serialization_and_json(tmp_path):
    """Verify dictionary conversion and JSON file export of vibration results."""
    driver = PhatpsyDriver(quiet=True)
    deck0 = get_n2_test_deck()
    deck = copy.deepcopy(deck0)
    deck.geometry[0].xyz = [0.0, 0.0, 0.700]
    deck.geometry[1].xyz = [0.0, 0.0, -0.700]

    vib_res = compute_vibrations(
        deck=deck,
        driver=driver,
        step_size=0.005,
        method="forward",
        quiet=True,
    )

    d = vib_res.to_dict()
    assert d["num_atoms"] == 2
    assert d["is_minimum"] is True
    assert len(d["modes"]) == 6

    out_json = tmp_path / "n2_vibrations.json"
    vib_res.save_json(out_json)
    assert out_json.is_file()

    with open(out_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["num_vibrational_modes"] == 1
    assert loaded["thermochemistry"]["temperature_k"] == 298.15


def test_vibrations_3d_webgl_visualizer(tmp_path):
    """Verify 3D Three.js WebGL visualization HTML generation for normal modes."""
    driver = PhatpsyDriver(quiet=True)
    deck0 = get_n2_test_deck()
    deck = copy.deepcopy(deck0)
    deck.geometry[0].xyz = [0.0, 0.0, 0.700]
    deck.geometry[1].xyz = [0.0, 0.0, -0.700]

    vib_res = compute_vibrations(
        deck=deck,
        driver=driver,
        step_size=0.005,
        method="forward",
        quiet=True,
    )

    # 1. Visualization data dictionary
    vis_data = generate_vibrations_visualization_data(vib_res)
    assert vis_data["num_atoms"] == 2
    assert len(vis_data["atoms"]) == 2
    assert len(vis_data["modes"]) == 6

    # 2. Standalone HTML generation
    html_code = generate_vibrations_html(vib_res, amplitude=0.5)
    assert "three.min.js" in html_code
    assert "OrbitControls.js" in html_code
    assert "PHATPSY 2.0 Normal Mode Vibrations" in html_code
    assert "mode-select" in html_code
    assert "amp-slider" in html_code
    assert "play-btn" in html_code

    # 3. File saving
    out_html = tmp_path / "vibrations_view.html"
    visualize_vibrations(vib_res, output_path=out_html, open_browser=False)
    assert out_html.is_file()
    assert out_html.stat().st_size > 1000


def test_vibrations_cli(tmp_path):
    """Verify 'phatpsy freq' CLI command execution and JSON export."""
    out_json = tmp_path / "cli_freq_out.json"
    deck_yaml = tmp_path / "n2_test.yaml"

    # Create test deck near minimum
    deck0 = get_n2_test_deck()
    deck = copy.deepcopy(deck0)
    deck.geometry[0].xyz = [0.0, 0.0, 0.700]
    deck.geometry[1].xyz = [0.0, 0.0, -0.700]
    deck.to_yaml(deck_yaml)

    # Run CLI command
    ret = cli_main(["freq", str(deck_yaml), "-m", "forward", "-o", str(out_json)])
    assert ret == 0
    assert out_json.is_file()

    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["num_atoms"] == 2
    assert data["num_vibrational_modes"] == 1


def test_nonlinear_triatomic_vibrational_frequencies():
    """
    Verify full Cartesian Hessian construction, 6-mode Eckart projection (N_tr = 6),
    and RRHO thermochemistry on non-linear triatomic water (H2O).
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
        title="Water H2O Vibrations",
        spin_unrestricted=True,
        auto_spin_flip=True,
    )

    # Forward finite-difference Hessian is fast and validates the 3N-6 = 3 projection
    vib_res = compute_vibrations(
        deck=deck_h2o,
        driver=driver,
        step_size=0.01,
        method="forward",
        quiet=True,
    )

    # 3 atoms -> 9 total degrees of freedom
    assert len(vib_res.modes) == 9

    # Exactly 6 modes must have near-zero frequency from translation/rotation projection
    zero_modes = [m for m in vib_res.modes if math.isclose(m.frequency_cm, 0.0, abs_tol=1e-3)]
    assert len(zero_modes) == 6

    # 3 vibrational modes remaining (symmetric stretch, asymmetric stretch, bend)
    vib_modes = [m for m in vib_res.modes if not math.isclose(m.frequency_cm, 0.0, abs_tol=1e-3)]
    assert len(vib_modes) == 3

    # RRHO Thermochemistry should have non-zero rotational entropy and ZPVE
    thermo = vib_res.thermochemistry
    assert thermo.zero_point_energy_kcal > 0.0
    assert thermo.entropy_cal_mol_k["rotation"] > 0.0
    assert thermo.total_gibbs_free_energy_g_t < 0.0

