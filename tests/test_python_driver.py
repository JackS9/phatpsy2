# tests/test_python_driver.py
import pytest
import numpy as np
from pathlib import Path
from phatpsy import (
    PhatpsyDriver,
    DIISAccelerator,
    parse_stdin_file,
    generate_diatomic_deck,
    solve_ewmo_propagator,
    unpack_symmetric,
    save_results_to_json,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_N2 = REPO_ROOT / "examples" / "n2.stdin"
EXAMPLE_AG13DZ = REPO_ROOT / "examples" / "ag13dz.stdin"


def test_io_parse_n2_stdin():
    data = parse_stdin_file(EXAMPLE_N2)
    assert data["ctlinp"]["NATOM"] == 2
    assert data["ctlinp"]["NELEM"] == 1
    assert data["ctlinp"]["NORBT"] == 10
    assert len(data["atom_positions"]) >= 1


def test_io_generate_diatomic_deck():
    basis = [(1, 0, 6.6651), (2, 0, 1.9237), (2, 1, 1.9170)]
    deck = generate_diatomic_deck("N", 7, 2.068, basis)
    assert "LATOM='N'" in deck
    assert "XYZ=0.0,0.0,1.03400" in deck
    assert "&ATOM IXYZ=001, ISYM=-1, QFLIP=T" in deck


def test_diis_accelerator():
    diis = DIISAccelerator(max_subspace=4)
    # Simulate converging 1D Fock updates with oscillating errors
    fock_init = np.array([1.0, 2.0, 3.0])
    err1 = np.array([0.1, -0.05, 0.02])
    fock_out1 = diis.update(fock_init, err1)
    assert np.allclose(fock_out1, fock_init)

    err2 = np.array([-0.08, 0.04, -0.01])
    fock_out2 = diis.update(fock_init + 0.1, err2)
    assert fock_out2.shape == fock_init.shape
    # Extrapolation should damp the oscillations
    assert not np.allclose(fock_out2, fock_init + 0.1)


def test_unpack_symmetric():
    # Packed lower-triangle of 3x3 matrix: [11, 21, 22, 31, 32, 33]
    packed = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    mat = unpack_symmetric(packed, 3)
    assert mat.shape == (3, 3)
    assert mat[0, 0] == 1.0
    assert mat[1, 0] == 2.0
    assert mat[0, 1] == 2.0
    assert mat[1, 1] == 3.0
    assert mat[2, 0] == 4.0
    assert mat[0, 2] == 4.0
    assert mat[2, 2] == 6.0


def test_driver_observables_extraction():
    core = pytest.importorskip("_phatpsy_core")
    state = core.state
    state.init_block_data_and_defaults()
    state.natom = 2
    state.norbt = 10
    state.mstot = 10
    state.compute_derived_dimensions_and_allocate()

    evals = state.get_evalx(10, 2)
    occs = state.get_occx(10, 2)
    assert evals.shape == (10, 2)
    assert occs.shape == (10, 2)


def test_driver_run_file_and_json_export(tmp_path):
    import json
    pytest.importorskip("_phatpsy_core")

    driver = PhatpsyDriver()
    res = driver.run_file(EXAMPLE_N2)

    assert res is not None
    assert "total_energy" in res
    assert "kinetic_energy" in res
    assert "eigenvalues" in res
    assert "cycles" in res
    assert res["converged"] is True

    # Test JSON serialization and round-trip
    out_json = tmp_path / "n2_result.json"
    save_results_to_json(res, out_json)
    assert out_json.exists()

    with open(out_json, "r") as f:
        loaded = json.load(f)

    assert loaded["converged"] is True
    assert loaded["cycles"] > 0
    assert abs(loaded["total_energy"] - (-109.5656)) < 1e-2
    assert len(loaded["eigenvalues"]) == 10
    assert len(loaded["mo_coefficients"]) == 10


def test_basis_library_lookup():
    from phatpsy.basis import get_element_basis, list_available_elements
    elems = list_available_elements()
    assert "N" in elems
    assert "C" in elems
    assert "Ag" in elems

    n_sz = get_element_basis("N", "SZ")
    assert n_sz.atomic_number == 7
    assert n_sz.nsto == 3
    assert n_sz.msto == 5

    n_dz = get_element_basis("N", "DZ")
    assert n_dz.nsto == 6
    assert n_dz.msto == 10


def test_calculation_deck_yaml_and_json_roundtrip(tmp_path):
    from phatpsy.schema import CalculationDeck, ControlConfig, AtomConfig

    deck = CalculationDeck(
        title="Roundtrip Test",
        control=ControlConfig(spin_unrestricted=True, max_cycles=42),
        geometry=[
            AtomConfig(element="N", xyz=[0.0, 0.0, 1.0]),
            AtomConfig(element="N", xyz=[0.0, 0.0, -1.0]),
        ]
    )

    # Test YAML roundtrip
    yaml_file = tmp_path / "deck.yaml"
    deck.to_yaml(yaml_file)
    assert yaml_file.exists()

    loaded_yaml = CalculationDeck.from_yaml(yaml_file)
    assert loaded_yaml.title == "Roundtrip Test"
    assert loaded_yaml.control.max_cycles == 42
    assert len(loaded_yaml.geometry) == 2
    assert loaded_yaml.geometry[0].element == "N"

    # Test JSON roundtrip
    json_file = tmp_path / "deck.json"
    deck.to_json(json_file)
    assert json_file.exists()

    loaded_json = CalculationDeck.from_json(json_file)
    assert loaded_json.control.max_cycles == 42
    assert loaded_json.geometry[1].xyz == [0.0, 0.0, -1.0]


def test_stdin_to_deck_conversion():
    from phatpsy.io import parse_stdin_to_deck
    deck = parse_stdin_to_deck(EXAMPLE_N2)
    assert len(deck.geometry) == 2
    assert "N" in deck.elements
    assert deck.control.spin_unrestricted is True
    assert deck.control.convergence == pytest.approx(1e-4)


def test_driver_run_from_yaml(tmp_path):
    pytest.importorskip("_phatpsy_core")
    from phatpsy.schema import CalculationDeck
    from phatpsy.driver import PhatpsyDriver

    yaml_text = """
title: "N2 from YAML"
control:
  spin_unrestricted: true
  max_cycles: 50
  convergence: 1.0e-4
geometry:
  - element: N
    xyz: [0.0, 0.0, 1.034]
  - element: N
    xyz: [0.0, 0.0, 0.0]
    symmetry:
      ixyz: 1
      isym: -1
      qflip: true
"""
    yaml_file = tmp_path / "n2.yaml"
    with open(yaml_file, "w") as f:
        f.write(yaml_text)

    driver = PhatpsyDriver()
    res = driver.run_file(yaml_file)

    assert res["converged"] is True
    assert res["cycles"] > 0
    assert np.isclose(res["total_energy"], -109.7089, atol=1e-2)


def test_cli_convert_command(tmp_path):
    from phatpsy.cli import main
    out_yaml = tmp_path / "n2_conv.yaml"
    ret = main(["convert", str(EXAMPLE_N2), "-o", str(out_yaml)])
    assert ret == 0
    assert out_yaml.exists()
    assert "LATOM='N'" not in out_yaml.read_text()
    assert "element: N" in out_yaml.read_text()


def test_thakkar_basis_library_coverage():
    from phatpsy.basis import (
        get_element_basis,
        get_thakkar_basis,
        list_thakkar_elements,
        list_available_elements,
    )

    thakkar_elems = list_thakkar_elements()
    assert len(thakkar_elems) == 103
    assert thakkar_elems[0] == "H"
    assert thakkar_elems[-1] == "Lr"
    assert "Si" in thakkar_elems
    assert "Fe" in thakkar_elems
    assert "Au" in thakkar_elems

    all_elems = list_available_elements()
    assert len(all_elems) >= 103
    assert "THAKKAR" in all_elems["C"]
    assert "SZ" in all_elems["C"]

    # Test explicit Thakkar retrieval
    c_thakkar = get_thakkar_basis("C")
    assert c_thakkar.symbol == "C"
    assert c_thakkar.atomic_number == 6
    assert c_thakkar.nsto == 15
    assert c_thakkar.msto == 29
    assert len(c_thakkar.default_orbitals) == 5

    # Test heavy element (Gold, Z=79)
    au_thakkar = get_thakkar_basis("Au")
    assert au_thakkar.symbol == "Au"
    assert au_thakkar.atomic_number == 79
    assert au_thakkar.nsto > 20
    assert any(f.l == 3 for f in au_thakkar.functions)  # Contains f-orbitals

    # Test fallback: Si is not in Clementi standard, so defaults to Thakkar
    si = get_element_basis("Si")
    assert si.symbol == "Si"
    assert si.basis_name == "THAKKAR"
    assert si.atomic_number == 14

    # Test Clementi element defaults to SZ, but can explicitly request THAKKAR
    c_default = get_element_basis("C")
    assert c_default.basis_name == "SZ"
    c_thak = get_element_basis("C", "THAKKAR")
    assert c_thak.basis_name == "THAKKAR"


def test_calculation_deck_with_thakkar_basis():
    from phatpsy.schema import CalculationDeck, ControlConfig, AtomConfig, ElementConfig

    deck = CalculationDeck(
        title="Silicon Atom with Thakkar Basis",
        control=ControlConfig(spin_unrestricted=True, max_cycles=1),
        geometry=[AtomConfig(element="Si", xyz=[0.0, 0.0, 0.0])],
        elements={"Si": ElementConfig(symbol="Si", basis="THAKKAR")}
    )

    stdin_text = deck.to_stdin_text()
    assert "LATOM='Si'" in stdin_text
    assert "NUCZ=14" in stdin_text
    assert "NMAX=" in stdin_text
    assert "LMAX=" in stdin_text
    assert "&ATOM XYZ=0.00000,0.00000,0.00000" in stdin_text
    assert "&ATORB" in stdin_text


def test_diis_charge_acceleration():
    """Verify Phase 4.2: Built-in DIIS charge acceleration in driver."""
    import math
    from phatpsy.builder import build_diatomic
    from phatpsy.driver import PhatpsyDriver

    driver = PhatpsyDriver(quiet=True)
    deck = build_diatomic("C", "O", 2.132, basis="DZ", spin_flip=True, units="bohr")
    deck.control.max_cycles = 15
    deck.control.use_diis = True

    res = driver.run_deck(deck)
    assert res.converged is True
    assert -115.0 < res.total_energy < -70.0
    assert math.isclose(res.virial_ratio, 2.0, abs_tol=0.1)


def test_global_fermi_equilibration():
    """Verify Phase 4.3: Full calculation with global Fermi level chemical potential equilibration."""
    import math
    from phatpsy.builder import build_diatomic
    from phatpsy.driver import PhatpsyDriver

    driver = PhatpsyDriver(quiet=True)
    deck = build_diatomic("N", "N", 2.068, basis="SZ", spin_flip=False, units="bohr")
    deck.control.max_cycles = 10
    deck.control.use_global_fermi = True
    deck.control.fermi_temperature = 1000.0

    stdin_text = deck.to_stdin_text()
    assert "QGFERM=T" in stdin_text
    assert "THETA=1000.00" in stdin_text

    res = driver.run_deck(deck)
    assert res.converged is True
    assert math.isclose(res.total_energy, -110.77, abs_tol=0.2)
    assert math.isclose(res.virial_ratio, 2.0, abs_tol=0.05)
