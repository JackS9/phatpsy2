"""
Unit tests for PHATPSY 2.0 structure formats (Phase 13, Tier 1).

Tests native zero-dependency parsers:
- PDB (.pdb)
- Gaussian (.gjf, .com)
- VASP POSCAR / CONTCAR (POSCAR, .vasp)
- Automatic format detection and builder integration.
"""
from pathlib import Path
import tempfile
import pytest
import numpy as np

from phatpsy.formats import (
    detect_format,
    parse_pdb,
    parse_gaussian,
    parse_poscar,
    BOHR_PER_ANGSTROM,
)
from phatpsy.builder import parse_geometry, build_molecule
from phatpsy.driver import PhatpsyDriver


# ==============================================================================
# PDB Parser Tests
# ==============================================================================

SAMPLE_PDB_WATER = """\
HEADER    WATER MOLECULE
HETATM    1  O   HOH A   1       0.000   0.000   0.117  1.00  0.00           O
HETATM    2  H1  HOH A   1       0.000   0.757  -0.469  1.00  0.00           H
HETATM    3  H2  HOH A   1       0.000  -0.757  -0.469  1.00  0.00           H
END
"""

SAMPLE_PDB_METALLO = """\
ATOM      1  N   ALA A   1       1.458   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       2.000   1.200   0.000  1.00 20.00           C
HETATM    3  ZN  ZN1 A   2       5.000   5.000   5.000  1.00 15.00          ZN2+
"""


def test_parse_pdb_water():
    atoms = parse_pdb(SAMPLE_PDB_WATER, units="angstrom")
    assert len(atoms) == 3
    assert [a.element for a in atoms] == ["O", "H", "H"]

    # Coordinates in Bohr: 0.117 Angstrom * 1.889726...
    assert pytest.approx(atoms[0].xyz[2], rel=1e-3) == 0.117 * BOHR_PER_ANGSTROM
    assert pytest.approx(atoms[1].xyz[1], rel=1e-3) == 0.757 * BOHR_PER_ANGSTROM
    assert pytest.approx(atoms[2].xyz[1], rel=1e-3) == -0.757 * BOHR_PER_ANGSTROM


def test_parse_pdb_metallo_and_charge():
    atoms = parse_pdb(SAMPLE_PDB_METALLO, units="angstrom")
    assert len(atoms) == 3
    assert atoms[0].element == "N"
    assert atoms[1].element == "C"
    assert atoms[2].element == "Zn"
    assert atoms[2].charge == 2.0


def test_parse_pdb_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "water.pdb"
        p.write_text(SAMPLE_PDB_WATER, encoding="utf-8")

        atoms = parse_geometry(p, units="angstrom")
        assert len(atoms) == 3
        assert atoms[0].element == "O"


# ==============================================================================
# Gaussian Input Parser Tests
# ==============================================================================

SAMPLE_GAUSSIAN_CARTESIAN = """\
%chk=water.chk
%nprocshared=4
#P B3LYP/6-31G(d) opt freq

Water geometry optimization

0 1
O          0.000000    0.000000    0.117300
H          0.000000    0.757000   -0.469200
H          0.000000   -0.757000   -0.469200

"""

SAMPLE_GAUSSIAN_FROZEN_AND_RADICAL = """\
# HF/STO-3G

Nickel carbon monoxide with frozen Ni

1 2
28   -1    0.000000    0.000000    0.000000
C     0    0.000000    0.000000    1.800000
O     0    0.000000    0.000000    2.930000

"""


def test_parse_gaussian_cartesian():
    atoms, charge, mult, title = parse_gaussian(SAMPLE_GAUSSIAN_CARTESIAN)
    assert len(atoms) == 3
    assert [a.element for a in atoms] == ["O", "H", "H"]
    assert charge == 0
    assert mult == 1
    assert "Water geometry optimization" in title
    assert pytest.approx(atoms[0].xyz[2], rel=1e-3) == 0.1173 * BOHR_PER_ANGSTROM


def test_parse_gaussian_frozen_and_multiplicity():
    atoms, charge, mult, title = parse_gaussian(SAMPLE_GAUSSIAN_FROZEN_AND_RADICAL)
    assert len(atoms) == 3
    assert atoms[0].element == "Ni"
    assert atoms[0].fixed is True
    assert atoms[1].element == "C"
    assert atoms[1].fixed is False
    assert atoms[2].element == "O"
    assert charge == 1
    assert mult == 2
    assert "frozen Ni" in title


def test_build_molecule_from_gaussian_deck():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "calc.gjf"
        p.write_text(SAMPLE_GAUSSIAN_CARTESIAN, encoding="utf-8")

        deck = build_molecule(p, basis="DZ", spins="up")
        assert len(deck.atoms) == 3
        assert "Water geometry optimization" in deck.title
        assert deck.atoms[0].element == "O"


# ==============================================================================
# VASP POSCAR Parser Tests
# ==============================================================================

SAMPLE_POSCAR_DIRECT = """\
FCC Nickel (100) surface unit cell
   3.52400000000000
     1.0000000000000000    0.0000000000000000    0.0000000000000000
     0.0000000000000000    1.0000000000000000    0.0000000000000000
     0.0000000000000000    0.0000000000000000    3.0000000000000000
   Ni
     2
Selective dynamics
Direct
   0.0000000000000000  0.0000000000000000  0.1000000000000000   F   F   F
   0.5000000000000000  0.5000000000000000  0.2000000000000000   T   T   T
"""

SAMPLE_POSCAR_CARTESIAN = """\
CO molecule in box
1.0
   10.0  0.0  0.0
    0.0 10.0  0.0
    0.0  0.0 10.0
   C  O
   1  1
Cartesian
   5.000000  5.000000  4.436000
   5.000000  5.000000  5.564000
"""


def test_parse_poscar_direct_and_selective():
    atoms, lat_bohr, title = parse_poscar(SAMPLE_POSCAR_DIRECT, units="angstrom")
    assert len(atoms) == 2
    assert "Nickel" in title
    assert atoms[0].element == "Ni"
    assert atoms[0].fixed is True   # F F F -> fixed
    assert atoms[1].fixed is False  # T T T -> unfixed

    # Lattice in Bohr: 3.524 * 1.889726...
    assert pytest.approx(lat_bohr[0, 0], rel=1e-3) == 3.524 * BOHR_PER_ANGSTROM
    # Fractional z=0.10 -> 0.10 * 3.0 * 3.524 Angstrom in Bohr
    assert pytest.approx(atoms[0].xyz[2], rel=1e-3) == 0.10 * 3.0 * 3.524 * BOHR_PER_ANGSTROM


def test_parse_poscar_cartesian():
    atoms, lat_bohr, title = parse_poscar(SAMPLE_POSCAR_CARTESIAN, units="angstrom")
    assert len(atoms) == 2
    assert atoms[0].element == "C"
    assert atoms[1].element == "O"
    bond_length = abs(atoms[1].xyz[2] - atoms[0].xyz[2])
    # 5.564 - 4.436 = 1.128 Angstrom * BOHR_PER_ANGSTROM = 2.1316 Bohr
    assert pytest.approx(bond_length, rel=1e-3) == 1.128 * BOHR_PER_ANGSTROM


def test_parse_poscar_from_builder():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "POSCAR"
        p.write_text(SAMPLE_POSCAR_CARTESIAN, encoding="utf-8")

        deck = build_molecule(p, basis="DZ", spins="up")
        assert len(deck.atoms) == 2
        assert deck.atoms[0].element == "C"
        assert deck.atoms[1].element == "O"


# ==============================================================================
# Format Detection Tests
# ==============================================================================

def test_detect_format_by_extension():
    assert detect_format("mol.pdb") == "pdb"
    assert detect_format("input.gjf") == "gaussian"
    assert detect_format("input.com") == "gaussian"
    assert detect_format("POSCAR") == "poscar"
    assert detect_format("CONTCAR") == "poscar"
    assert detect_format("slab.poscar") == "poscar"
    assert detect_format("crystal.vasp") == "poscar"
    assert detect_format("geo.xyz") == "xyz"
    assert detect_format("deck.yaml") == "yaml"
    assert detect_format("deck.json") == "json"
    assert detect_format("deck.stdin") == "stdin"


def test_detect_format_from_text_content():
    assert detect_format(SAMPLE_PDB_WATER) == "pdb"
    assert detect_format(SAMPLE_GAUSSIAN_CARTESIAN) == "gaussian"
    assert detect_format(SAMPLE_POSCAR_DIRECT) == "poscar"
    assert detect_format("3\nComment\nO 0 0 0\nH 0 1 0\nH 0 0 1") == "xyz"


# ==============================================================================
# End-to-End Calculation Run on Imported Structure
# ==============================================================================

def test_scf_calculation_on_imported_pdb():
    """Runs a real SCF calculation on water geometry parsed directly from PDB."""
    driver = PhatpsyDriver()
    deck = build_molecule(SAMPLE_PDB_WATER, basis="DZ", units="angstrom", spins="up", net_spin=3)
    res = driver.run_deck(deck, quiet=True)
    assert res is not None
    assert -78.0 <= res.total_energy <= -72.0
    assert abs(res.virial_ratio - 2.0) <= 0.08


def test_cli_build_and_convert_supported_formats():
    """Tests CLI 'phatpsy build' on PDB, Gaussian, and POSCAR formats."""
    from phatpsy.cli import main

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. PDB file
        pdb_path = Path(tmpdir) / "water.pdb"
        pdb_path.write_text(SAMPLE_PDB_WATER, encoding="utf-8")
        out_yaml1 = Path(tmpdir) / "deck_pdb.yaml"

        rc = main(["build", str(pdb_path), "-o", str(out_yaml1), "--spins", "up", "--net-spin", "3"])
        assert rc == 0
        assert out_yaml1.exists()
        assert "element: O" in out_yaml1.read_text(encoding="utf-8")

        # 2. Gaussian file
        gjf_path = Path(tmpdir) / "calc.gjf"
        gjf_path.write_text(SAMPLE_GAUSSIAN_CARTESIAN, encoding="utf-8")
        out_yaml2 = Path(tmpdir) / "deck_gjf.yaml"

        rc = main(["build", str(gjf_path), "-o", str(out_yaml2), "--spins", "up"])
        assert rc == 0
        assert out_yaml2.exists()
        assert "Water geometry optimization" in out_yaml2.read_text(encoding="utf-8")

        # 3. POSCAR file
        poscar_path = Path(tmpdir) / "POSCAR"
        poscar_path.write_text(SAMPLE_POSCAR_CARTESIAN, encoding="utf-8")
        out_yaml3 = Path(tmpdir) / "deck_poscar.yaml"

        rc = main(["build", str(poscar_path), "-o", str(out_yaml3), "--spins", "up"])
        assert rc == 0
        assert out_yaml3.exists()
        assert "element: C" in out_yaml3.read_text(encoding="utf-8")


# ==============================================================================
# ASE Ecosystem Bridge Tests (Phase 13, Tier 2)
# ==============================================================================

def test_has_ase_flag():
    from phatpsy.formats import HAS_ASE
    assert HAS_ASE is True


def test_from_ase_basic():
    import ase
    from ase import Atoms
    from phatpsy.formats import from_ase, BOHR_PER_ANGSTROM

    ase_atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]])
    phatpsy_atoms = from_ase(ase_atoms)

    assert len(phatpsy_atoms) == 2
    assert phatpsy_atoms[0].element == "C"
    assert phatpsy_atoms[1].element == "O"
    assert phatpsy_atoms[0].id == "C1"
    assert phatpsy_atoms[1].id == "O1"

    # Positions converted to Bohr
    assert pytest.approx(phatpsy_atoms[0].xyz, abs=1e-6) == [0.0, 0.0, 0.0]
    assert pytest.approx(phatpsy_atoms[1].xyz[2], rel=1e-5) == 1.128 * BOHR_PER_ANGSTROM


def test_from_ase_charges_magmoms_and_constraints():
    import ase
    from ase import Atoms
    from ase.constraints import FixAtoms
    from phatpsy.formats import from_ase

    ase_atoms = Atoms("Fe2O", positions=[[0, 0, 0], [2, 0, 0], [1, 1, 0]])
    ase_atoms.set_initial_charges([1.5, 1.5, -3.0])
    ase_atoms.set_initial_magnetic_moments([3.0, -3.0, 0.0])
    ase_atoms.set_constraint(FixAtoms(indices=[0]))

    phatpsy_atoms = from_ase(ase_atoms)
    assert len(phatpsy_atoms) == 3

    # Charges
    assert phatpsy_atoms[0].charge == 1.5
    assert phatpsy_atoms[1].charge == 1.5
    assert phatpsy_atoms[2].charge == -3.0

    # Magnetic moments -> spins
    assert phatpsy_atoms[0].spin == "up"
    assert phatpsy_atoms[0].spin_flip is False
    assert phatpsy_atoms[1].spin == "down"
    assert phatpsy_atoms[1].spin_flip is True
    assert phatpsy_atoms[2].spin is None

    # Constraints
    assert phatpsy_atoms[0].fixed is True
    assert phatpsy_atoms[1].fixed is False
    assert phatpsy_atoms[2].fixed is False


def test_to_ase_and_roundtrip():
    import ase
    from ase import Atoms
    from ase.constraints import FixAtoms
    from phatpsy.formats import from_ase, to_ase

    orig_atoms = Atoms("NO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.15]])
    orig_atoms.set_initial_charges([0.2, -0.2])
    orig_atoms.set_initial_magnetic_moments([1.0, 0.0])
    orig_atoms.set_constraint(FixAtoms(indices=[0]))

    phatpsy_atoms = from_ase(orig_atoms)
    exported_ase = to_ase(phatpsy_atoms)

    assert isinstance(exported_ase, Atoms)
    assert exported_ase.get_chemical_symbols() == ["N", "O"]
    assert pytest.approx(exported_ase.get_positions(), abs=1e-5) == orig_atoms.get_positions()
    assert pytest.approx(exported_ase.get_initial_charges(), abs=1e-5) == [0.2, -0.2]
    assert pytest.approx(exported_ase.get_initial_magnetic_moments(), abs=1e-5) == [1.0, 0.0]
    assert len(exported_ase.constraints) == 1
    assert exported_ase.constraints[0].index.tolist() == [0]


def test_calculation_deck_from_ase_and_to_ase():
    from ase import Atoms
    from phatpsy.schema import CalculationDeck
    from phatpsy.formats import BOHR_PER_ANGSTROM

    ase_mol = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        cell=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
        pbc=[True, True, True],
    )
    deck = CalculationDeck.from_ase(
        ase_mol,
        basis="DZ",
        title="H2 from ASE",
        spins=["up", "down"],
        net_spin=0,
    )

    assert deck.title == "H2 from ASE"
    assert len(deck.geometry) == 2
    assert deck.geometry[0].element == "H"
    assert deck.geometry[1].element == "H"
    assert deck.cell is not None
    assert pytest.approx(deck.cell[0][0], rel=1e-4) == 5.0 * BOHR_PER_ANGSTROM
    assert deck.pbc == [True, True, True]

    # Convert back to ASE
    ase_exported = deck.to_ase()
    assert isinstance(ase_exported, Atoms)
    assert ase_exported.get_chemical_symbols() == ["H", "H"]
    assert pytest.approx(ase_exported.get_positions(), abs=1e-4) == ase_mol.get_positions()
    assert pytest.approx(ase_exported.get_cell(), abs=1e-4) == ase_mol.get_cell()
    assert ase_exported.pbc.tolist() == [True, True, True]


SAMPLE_CIF_SILICON = """\
data_Si
_cell_length_a 5.4309
_cell_length_b 5.4309
_cell_length_c 5.4309
_cell_angle_alpha 90.00
_cell_angle_beta 90.00
_cell_angle_gamma 90.00
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 0.00000 0.00000 0.00000
Si2 0.25000 0.25000 0.25000
"""

def test_parse_ase_cif_and_builder_integration():
    from phatpsy.formats import parse_ase, read_ase, BOHR_PER_ANGSTROM
    from phatpsy.builder import build_molecule

    # 1. Direct parse_ase
    atoms = parse_ase(SAMPLE_CIF_SILICON, format="cif")
    assert len(atoms) == 2
    assert atoms[0].element == "Si"
    assert atoms[1].element == "Si"
    assert pytest.approx(atoms[1].xyz[0], rel=1e-4) == 0.25 * 5.4309 * BOHR_PER_ANGSTROM

    # 2. read_ase with cell
    atoms, cell, pbc = read_ase(SAMPLE_CIF_SILICON, format="cif")
    assert cell is not None
    assert pytest.approx(cell[0][0], rel=1e-4) == 5.4309 * BOHR_PER_ANGSTROM
    assert pbc == [True, True, True]

    # 3. build_molecule directly from CIF string
    deck = build_molecule(SAMPLE_CIF_SILICON, basis="DZ", spins="up", net_spin=2)
    assert len(deck.geometry) == 2
    assert deck.geometry[0].element == "Si"


def test_parse_geometry_delegates_to_ase_for_gro():
    SAMPLE_GRO_WATER = """\
Water GRO model
3
    1SOL     OW    1   0.000   0.000   0.012
    1SOL    HW1    2   0.000   0.076  -0.047
    1SOL    HW2    3   0.000  -0.076  -0.047
   1.00000   1.00000   1.00000
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        gro_file = Path(tmpdir) / "water.gro"
        gro_file.write_text(SAMPLE_GRO_WATER, encoding="utf-8")

        deck = build_molecule(gro_file, basis="DZ", spins="up", net_spin=3)
        assert len(deck.geometry) == 3
        assert [a.element for a in deck.geometry] == ["O", "H", "H"]


def test_missing_ase_graceful_error(monkeypatch):
    import phatpsy.formats
    monkeypatch.setattr(phatpsy.formats, "HAS_ASE", False)

    with pytest.raises(ImportError, match="The ASE bridge requires 'ase'"):
        phatpsy.formats.from_ase(None)

    with pytest.raises(ImportError, match="The ASE bridge requires 'ase'"):
        phatpsy.formats.to_ase([])

    with pytest.raises(ImportError, match="The ASE bridge requires 'ase'"):
        phatpsy.formats.parse_ase("some_file.cif")


def test_cli_convert_to_cif_and_gro():
    from phatpsy.cli import main

    with tempfile.TemporaryDirectory() as tmpdir:
        in_yaml = Path(tmpdir) / "deck.yaml"
        deck = build_molecule(SAMPLE_PDB_WATER, basis="DZ", spins="up", net_spin=3)
        deck.to_yaml(in_yaml)

        # Convert to CIF
        out_cif = Path(tmpdir) / "water.cif"
        rc = main(["convert", str(in_yaml), "-o", str(out_cif)])
        assert rc == 0
        assert out_cif.exists()
        assert "data_" in out_cif.read_text(encoding="utf-8")

        # Convert to GRO
        out_gro = Path(tmpdir) / "water.gro"
        rc = main(["convert", str(in_yaml), "-o", str(out_gro)])
        assert rc == 0
        assert out_gro.exists()
        assert "OW" in out_gro.read_text(encoding="utf-8") or "O" in out_gro.read_text(encoding="utf-8")
