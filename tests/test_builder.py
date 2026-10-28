# tests/test_builder.py
import pytest
import numpy as np
from pathlib import Path
import tempfile

from phatpsy import (
    PhatpsyDriver,
    CalculationDeck,
    build_molecule,
    build_diatomic,
    parse_geometry,
    BOHR_PER_ANGSTROM,
    AtomConfig,
)
from phatpsy.cli import main as cli_main


def test_parse_geometry_tuples():
    geom = [("C", [0.0, 0.0, 0.0]), ("O", [0.0, 0.0, 2.13])]
    atoms = parse_geometry(geom, units="bohr")
    assert len(atoms) == 2
    assert atoms[0].element == "C"
    assert atoms[0].xyz == [0.0, 0.0, 0.0]
    assert atoms[1].element == "O"
    assert atoms[1].xyz == [0.0, 0.0, 2.13]


def test_parse_geometry_4tuples_and_angstrom():
    geom = [("C", 0.0, 0.0, 0.0), ("O", 0.0, 0.0, 1.128)]
    atoms = parse_geometry(geom, units="angstrom")
    assert len(atoms) == 2
    assert atoms[0].element == "C"
    assert atoms[1].element == "O"
    expected_z = 1.128 * BOHR_PER_ANGSTROM
    assert np.isclose(atoms[1].xyz[2], expected_z, atol=1e-5)


def test_parse_geometry_xyz_string():
    xyz_str = """3
    Water molecule
    O  0.000000  0.000000  0.117790
    H  0.000000  0.755453 -0.471161
    H  0.000000 -0.755453 -0.471161
    """
    atoms = parse_geometry(xyz_str, units="angstrom")
    assert len(atoms) == 3
    assert atoms[0].element == "O"
    assert atoms[1].element == "H"
    assert atoms[2].element == "H"
    assert np.isclose(atoms[0].xyz[2], 0.117790 * BOHR_PER_ANGSTROM, atol=1e-5)


def test_parse_geometry_xyz_file(tmp_path):
    xyz_path = tmp_path / "sample.xyz"
    xyz_path.write_text("2\nCO\nC 0 0 0\nO 0 0 2.13\n")
    atoms = parse_geometry(xyz_path, units="bohr")
    assert len(atoms) == 2
    assert atoms[0].element == "C"
    assert atoms[1].element == "O"
    assert atoms[1].xyz == [0.0, 0.0, 2.13]


def test_build_diatomic_homonuclear_symmetry():
    deck = build_diatomic("N", bond_length=2.068, basis="DZ", use_symmetry=True)
    assert len(deck.geometry) == 2
    assert deck.geometry[0].element == "N"
    assert deck.geometry[0].xyz == [0.0, 0.0, 1.034]
    assert deck.geometry[1].symmetry is not None
    assert deck.geometry[1].symmetry.isym == -1
    assert deck.geometry[1].symmetry.qflip is True

    assert "N" in deck.elements
    assert deck.elements["N"].basis == "DZ"
    assert len(deck.elements["N"].functions) == 6

    # Test stdin text generation
    stdin_text = deck.to_stdin_text()
    assert "LATOM='N'" in stdin_text
    assert "MDIFAT=1" in stdin_text
    assert "NEQAT=1" in stdin_text
    assert "QFLIP=T" in stdin_text
    assert "NCOEF=" in stdin_text


def test_build_diatomic_heteronuclear():
    deck = build_diatomic("C", "O", bond_length=2.13, basis="SZ")
    assert len(deck.geometry) == 2
    assert deck.geometry[0].element == "C"
    assert deck.geometry[1].element == "O"
    assert "C" in deck.elements
    assert "O" in deck.elements

    stdin_text = deck.to_stdin_text()
    assert "LATOM='C'" in stdin_text
    assert "LATOM='O'" in stdin_text
    assert "XYZ=0.00000,0.00000,0.00000" in stdin_text
    assert "XYZ=0.00000,0.00000,2.13000" in stdin_text


def test_build_molecule_water_mixed_basis():
    xyz_water = """
    O  0.000000  0.000000  0.222591
    H  0.000000  1.427599 -0.890365
    H  0.000000 -1.427599 -0.890365
    """
    deck = build_molecule(xyz_water, basis="DZ", units="bohr")
    assert len(deck.geometry) == 3
    assert "O" in deck.elements
    assert "H" in deck.elements
    assert deck.elements["O"].basis == "DZ"
    # H falls back to Thakkar STO
    assert deck.elements["H"].basis == "THAKKAR"

    stdin_text = deck.to_stdin_text()
    assert "LATOM='O'" in stdin_text
    assert "LATOM='H'" in stdin_text
    # 2 independent hydrogen atoms -> MDIFAT=2
    assert "MDIFAT=2" in stdin_text


def test_calculation_deck_classmethods():
    deck1 = CalculationDeck.from_diatomic("N", bond_length=2.068, basis="SZ")
    assert len(deck1.geometry) == 2

    deck2 = CalculationDeck.from_geometry([("C", [0, 0, 0]), ("O", [0, 0, 2.13])], basis="SZ")
    assert len(deck2.geometry) == 2
    assert "C" in deck2.elements
    assert "O" in deck2.elements


def test_cli_build_command(tmp_path):
    xyz_path = tmp_path / "n2.xyz"
    xyz_path.write_text("2\nN2 molecule\nN 0 0 0\nN 0 0 1.094\n")

    out_yaml = tmp_path / "n2_built.yaml"
    ret = cli_main(["build", str(xyz_path), "-b", "DZ", "-o", str(out_yaml)])
    assert ret == 0
    assert out_yaml.exists()
    content = out_yaml.read_text()
    assert "element: N" in content
    assert "basis: DZ" in content


def test_driver_run_molecule_scf():
    pytest.importorskip("_phatpsy_core")
    driver = PhatpsyDriver()

    # Run quick SCF on N2 with Clementi DZ
    res = driver.run_molecule(
        geometry=[("N", [0.0, 0.0, 1.034]), ("N", [0.0, 0.0, -1.034])],
        basis="DZ",
        title="N2 DZ Builder Test",
        extra_namelist={"MXCYCL": 5, "CONVRG": 1e-3},
    )
    assert res is not None
    assert "total_energy" in res
    assert res["cycles"] > 0
    assert res["norbt"] == 10
    # Expected N2 molecular energy in DZ basis with unit PK scaling is around -110.8 Hartree
    assert abs(res["total_energy"] - (-110.8)) < 2.0


def test_symmetry_operations_and_roundtrip(tmp_path):
    from phatpsy.io import parse_stdin_to_deck
    from phatpsy.schema import SymmetrySpec

    deck = build_diatomic("N", bond_length=2.068, basis="SZ", use_symmetry=True)
    assert len(deck.geometry) == 2
    assert deck.geometry[1].symmetry is not None
    assert deck.geometry[1].symmetry.ixyz == 1
    assert deck.geometry[1].symmetry.isym == -1

    stdin_text = deck.to_stdin_text()
    assert "NEQAT=1" in stdin_text
    assert "IXYZ=001" in stdin_text
    assert "ISYM=-1" in stdin_text

    deck_file = tmp_path / "n2_sym.stdin"
    deck_file.write_text(stdin_text)
    reparsed = parse_stdin_to_deck(deck_file)
    assert len(reparsed.geometry) == 2
    assert reparsed.geometry[1].symmetry is not None
    assert reparsed.geometry[1].symmetry.ixyz == 1
    assert reparsed.geometry[1].symmetry.isym == -1


def test_virtual_ghost_atoms(tmp_path):
    from phatpsy.io import parse_stdin_to_deck

    # 1. Via tuple with options
    deck1 = build_molecule(
        geometry=[("N", [0.0, 0.0, 1.0]), ("N", [0.0, 0.0, -1.0], {"virtual": True})],
        basis="SZ",
    )
    assert deck1.geometry[1].virtual is True
    stdin1 = deck1.to_stdin_text()
    assert "QVIRTL=T" in stdin1

    # 2. Via ghost prefix in XYZ string
    xyz_ghost = """2
    N2 with ghost center
    N    0.0  0.0  1.0
    Gh.N 0.0  0.0 -1.0
    """
    deck2 = build_molecule(geometry=xyz_ghost, basis="SZ")
    assert deck2.geometry[0].virtual is False
    assert deck2.geometry[1].virtual is True
    assert deck2.geometry[1].element == "N"

    # Round trip
    deck_file = tmp_path / "ghost.stdin"
    deck_file.write_text(deck2.to_stdin_text())
    reparsed = parse_stdin_to_deck(deck_file)
    assert reparsed.geometry[1].virtual is True


def test_fixed_atoms(tmp_path):
    from phatpsy.io import parse_stdin_to_deck

    deck = build_molecule(
        geometry=[("C", [0.0, 0.0, 0.0], {"fixed": True}), ("O", [0.0, 0.0, 2.13])],
        basis="SZ",
    )
    assert deck.geometry[0].fixed is True
    assert deck.geometry[1].fixed is False

    stdin_text = deck.to_stdin_text()
    assert "QFIXED=T" in stdin_text

    deck_file = tmp_path / "fixed.stdin"
    deck_file.write_text(stdin_text)
    reparsed = parse_stdin_to_deck(deck_file)
    assert reparsed.geometry[0].fixed is True
    assert reparsed.geometry[1].fixed is False


def test_spin_flipped_atoms(tmp_path):
    from phatpsy.io import parse_stdin_to_deck

    # Antiferromagnetic / broken-symmetry dimer
    deck = build_molecule(
        geometry=[("N", [0.0, 0.0, 1.0]), ("N", [0.0, 0.0, -1.0], {"spin_flip": True})],
        basis="SZ",
    )
    assert deck.geometry[0].spin_flip is False
    assert deck.geometry[1].spin_flip is True

    stdin_text = deck.to_stdin_text()
    assert "QFLIP=T" in stdin_text

    deck_file = tmp_path / "flip.stdin"
    deck_file.write_text(stdin_text)
    reparsed = parse_stdin_to_deck(deck_file)
    assert reparsed.geometry[0].spin_flip is False
    assert reparsed.geometry[1].spin_flip is True


def test_driver_run_symmetry_scf():
    pytest.importorskip("_phatpsy_core")
    driver = PhatpsyDriver()

    # Build diatomic N2 using Fortran reflection symmetry (ISYM=-1)
    deck = build_diatomic("N", bond_length=2.068, basis="DZ", use_symmetry=True, extra_namelist={"MXCYCL": 5})
    res = driver.run_deck(deck)
    assert res is not None
    assert "total_energy" in res
    # DZ basis with unit PK scaling gives around -110.6 Hartree after 5 cycles
    assert abs(res["total_energy"] - (-110.6)) < 2.0


def test_parse_stdin_starting_orbitals_and_yaml_roundtrip(tmp_path):
    from phatpsy import parse_stdin_to_deck
    stdin_path = Path("examples/CO_ni_100.stdin")
    assert stdin_path.exists()

    deck = parse_stdin_to_deck(stdin_path)
    # Check that primary atoms have their starting orbitals parsed
    c1 = next(a for a in deck.geometry if a.id == "C1")
    o1 = next(a for a in deck.geometry if a.id == "O1")
    ni1 = next(a for a in deck.geometry if a.id == "Ni1")

    assert c1.starting_orbitals is not None
    assert len(c1.starting_orbitals) == 5
    assert c1.spin == "up"
    assert c1.spin_flip is False
    assert c1.net_orbital_spin == 2.0

    assert o1.starting_orbitals is not None
    assert len(o1.starting_orbitals) == 5
    assert o1.spin == "down"
    assert o1.spin_flip is True
    assert o1.net_orbital_spin == -2.0

    assert ni1.starting_orbitals is not None
    assert len(ni1.starting_orbitals) == 15
    assert ni1.net_orbital_spin == 0.0

    # Check Ni1 custom starting orbital coefficients
    assert len(ni1.starting_orbitals[0].coefficients) == 15
    assert ni1.starting_orbitals[0].coefficients[0][0] == 1  # isto=1
    assert ni1.starting_orbitals[0].coefficients[0][1] == 0  # ml=0
    assert abs(ni1.starting_orbitals[0].coefficients[0][2] - (-0.998911)) < 1e-4

    # Check YAML serialization
    yaml_text = deck.to_yaml()
    assert "starting_orbitals:" in yaml_text
    assert "eigval:" in yaml_text
    assert "coefficients:" in yaml_text

    # Check YAML round-trip reconstruction
    deck_from_yaml = CalculationDeck.from_yaml(yaml_text)
    ni1_rt = next(a for a in deck_from_yaml.geometry if a.id == "Ni1")
    assert ni1_rt.starting_orbitals is not None
    assert len(ni1_rt.starting_orbitals) == 15
    assert len(ni1_rt.starting_orbitals[0].coefficients) == 15

    o1_rt = next(a for a in deck_from_yaml.geometry if a.id == "O1")
    assert o1_rt.spin == "down"
    assert o1_rt.spin_flip is True
    assert o1_rt.net_orbital_spin == -2.0

    # Check stdin text compilation preserves NCOEF and &ATORB
    stdin_compiled = deck_from_yaml.to_stdin_text()
    assert "&ATORB" in stdin_compiled
    assert "NCOEF=15" in stdin_compiled


def test_conflicting_spin_and_starting_orbitals_raises_error():
    from phatpsy.basis import InitialOrbital
    from phatpsy.schema import AtomConfig, CalculationDeck

    beta_excess_orbs = [
        InitialOrbital(eigval=[-10.0, -10.0], occno=[1.0, 1.0]),
        InitialOrbital(eigval=[-0.5, -0.5], occno=[0.0, 1.0]),  # Net beta / down
    ]

    # Explicit spin='up' conflicts with beta excess -> raises ValueError
    with pytest.raises(ValueError, match="Conflicting spin specification.*explicit spin='up'"):
        AtomConfig(
            element="O",
            id="O1",
            spin="up",
            starting_orbitals=beta_excess_orbs,
        )

    # Explicit spin='down' is consistent -> succeeds
    atom_ok = AtomConfig(
        element="O",
        id="O1",
        spin="down",
        starting_orbitals=beta_excess_orbs,
    )
    assert atom_ok.spin == "down"

    # spin omitted -> automatically deduced as 'down'
    atom_auto = AtomConfig(
        element="O",
        id="O1",
        starting_orbitals=beta_excess_orbs,
    )
    assert atom_auto.spin == "down"

    # Conflicting YAML deck -> raises ValueError on from_yaml
    conflicting_yaml = """
title: "Conflicting Spin Test"
control:
  spin_unrestricted: true
geometry:
  - element: O
    id: O1
    spin: up
    starting_orbitals:
      - occno: [0.0, 1.0]
"""
    with pytest.raises(ValueError, match="Conflicting spin specification"):
        CalculationDeck.from_yaml(conflicting_yaml)


