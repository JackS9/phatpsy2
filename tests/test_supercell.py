# tests/test_supercell.py
"""
Unit and integration tests for phatpsy.supercell module.
Covers 3D crystal lattices, supercells, surface slabs (100, 111, 110),
adsorbate binding sites, and multi-format exporters (Deck, POSCAR, CIF, XYZ).
"""
import pytest
import numpy as np
import math

from phatpsy.supercell import (
    Lattice,
    PeriodicAtom,
    PeriodicCell,
    build_slab,
    attach_adsorbate,
    STANDARD_LATTICE_CONSTANTS,
)
from phatpsy.schema import CalculationDeck
from phatpsy.spin import BOHR_PER_ANGSTROM


class TestLattice:
    def test_cubic_lattice(self):
        a_aa = 4.0
        lat = Lattice.cubic(a_aa, units="angstrom")
        a_bohr = a_aa * BOHR_PER_ANGSTROM
        assert pytest.approx(lat.a, rel=1e-5) == a_bohr
        assert pytest.approx(lat.b, rel=1e-5) == a_bohr
        assert pytest.approx(lat.c, rel=1e-5) == a_bohr
        assert pytest.approx(lat.alpha, abs=1e-4) == 90.0
        assert pytest.approx(lat.beta, abs=1e-4) == 90.0
        assert pytest.approx(lat.gamma, abs=1e-4) == 90.0
        assert pytest.approx(lat.volume, rel=1e-5) == a_bohr ** 3

    def test_hexagonal_lattice(self):
        a_aa, c_aa = 2.5, 4.0
        lat = Lattice.hexagonal(a_aa, c_aa, units="angstrom")
        assert pytest.approx(lat.gamma, abs=1e-4) == 120.0
        assert pytest.approx(lat.alpha, abs=1e-4) == 90.0
        assert pytest.approx(lat.beta, abs=1e-4) == 90.0
        expected_vol = (math.sqrt(3.0) / 2.0) * (lat.a ** 2) * lat.c
        assert pytest.approx(lat.volume, rel=1e-5) == expected_vol

    def test_reciprocal_lattice(self):
        lat = Lattice.orthorhombic(3.0, 4.0, 5.0, units="angstrom")
        recip = lat.reciprocal_lattice
        # a . a* = 2*pi, a . b* = 0
        assert pytest.approx(np.dot(lat.a_vec, recip.matrix[0]), rel=1e-5) == 2.0 * math.pi
        assert pytest.approx(np.dot(lat.a_vec, recip.matrix[1]), abs=1e-8) == 0.0
        assert pytest.approx(np.dot(lat.b_vec, recip.matrix[1]), rel=1e-5) == 2.0 * math.pi

    def test_coordinate_conversion_roundtrip(self):
        lat = Lattice.tetragonal(3.5, 5.0, units="angstrom")
        frac_orig = np.array([0.25, 0.33, 0.75])
        cart = lat.fractional_to_cartesian(frac_orig)
        frac_back = lat.cartesian_to_fractional(cart)
        np.testing.assert_allclose(frac_orig, frac_back, atol=1e-7)


class TestPeriodicCellBulk:
    def test_bulk_fcc_ni(self):
        cell = PeriodicCell.bulk("Ni", crystal_structure="fcc")
        assert len(cell.atoms) == 4
        assert cell.formula == "Ni4"
        assert pytest.approx(cell.lattice.alpha, abs=1e-4) == 90.0

    def test_bulk_bcc_fe(self):
        cell = PeriodicCell.bulk("Fe", crystal_structure="bcc")
        assert len(cell.atoms) == 2
        assert cell.formula == "Fe2"

    def test_bulk_diamond_c(self):
        cell = PeriodicCell.bulk("C", crystal_structure="diamond")
        assert len(cell.atoms) == 8
        assert cell.formula == "C8"

    def test_supercell_replication(self):
        cell = PeriodicCell.bulk("Ni", crystal_structure="fcc")
        sc = cell.build_supercell((2, 2, 2), auto_spin=True)
        # 4 atoms * 8 = 32 atoms
        assert len(sc.atoms) == 32
        assert pytest.approx(sc.lattice.a, rel=1e-5) == 2.0 * cell.lattice.a

        # Check spin balance
        up_spins = sum(1 for a in sc.atoms if a.spin == "up")
        down_spins = sum(1 for a in sc.atoms if a.spin == "down")
        assert up_spins == 16
        assert down_spins == 16


class TestSurfaceSlabs:
    def test_fcc_100_slab(self):
        # 3 layers, 2x2 in-plane supercell -> 4 atoms per layer -> 12 atoms
        slab = build_slab(
            element="Ni",
            crystal_structure="fcc",
            miller_index=(1, 0, 0),
            layers=3,
            vacuum=15.0,
            repeat=(2, 2),
            fixed_layers=1,
            add_virtual_perimeter=False,
        )
        assert len(slab.atoms) == 12

        # Layer 1 (bottom) must be fixed
        fixed_atoms = [a for a in slab.atoms if a.fixed]
        assert len(fixed_atoms) == 4

        # Surface layer (layer 3) must have LVB alternating spin
        surface_atoms = [a for a in slab.atoms if a.layer == 3]
        assert len(surface_atoms) == 4
        up = sum(1 for a in surface_atoms if a.spin == "up")
        down = sum(1 for a in surface_atoms if a.spin == "down")
        assert up == 2
        assert down == 2

    def test_fcc_111_slab(self):
        # 3 layers (ABC stacking), 2x2 in-plane
        slab = build_slab(
            element="Pt",
            crystal_structure="fcc",
            miller_index=(1, 1, 1),
            layers=3,
            vacuum=12.0,
            repeat=(2, 2),
            fixed_layers=1,
        )
        assert len(slab.atoms) == 12
        assert slab.atoms[0].element == "Pt"

    def test_fcc_110_slab(self):
        slab = build_slab(
            element="Cu",
            crystal_structure="fcc",
            miller_index=(1, 1, 0),
            layers=4,
            vacuum=10.0,
            repeat=(1, 2),
            fixed_layers=2,
        )
        assert len(slab.atoms) == 8
        fixed = [a for a in slab.atoms if a.fixed]
        assert len(fixed) == 4

    def test_slab_with_virtual_perimeter(self):
        slab = build_slab(
            element="Ni",
            miller_index=(1, 0, 0),
            layers=2,
            repeat=(1, 1),
            fixed_layers=0,
            add_virtual_perimeter=True,
        )
        virtuals = [a for a in slab.atoms if a.virtual]
        assert len(virtuals) > 0
        assert all(v.fixed for v in virtuals)


class TestAdsorbateAttachment:
    def test_attach_co_top_site(self):
        slab = build_slab(
            element="Ni",
            miller_index=(1, 0, 0),
            layers=2,
            repeat=(2, 2),
            fixed_layers=1,
        )
        n_orig = len(slab.atoms)  # 8 atoms
        with_co = attach_adsorbate(
            slab,
            adsorbate="CO",
            site_type="top",
            height=1.9,
            tilt_deg=0.0,
        )
        assert len(with_co.atoms) == n_orig + 2

        # Check that C and O were appended with correct tags and positions
        c_atom = with_co.atoms[-2]
        o_atom = with_co.atoms[-1]
        assert c_atom.element == "C"
        assert o_atom.element == "O"
        assert not c_atom.fixed
        assert not o_atom.fixed
        assert c_atom.tag == "adsorbate"

        # Check vertical ordering: z(O) > z(C) > max(z_substrate)
        max_sub_z = max(a.cart_coords[2] for a in slab.atoms)
        assert c_atom.cart_coords[2] > max_sub_z
        assert o_atom.cart_coords[2] > c_atom.cart_coords[2]
        # C-O bond distance should be ~ 1.128 A = ~ 2.132 Bohr
        co_dist = np.linalg.norm(o_atom.cart_coords - c_atom.cart_coords)
        assert pytest.approx(co_dist, rel=1e-3) == 1.128 * BOHR_PER_ANGSTROM

    def test_attach_bridge_and_hollow(self):
        slab = build_slab("Cu", miller_index=(1, 0, 0), layers=2, repeat=(2, 2))
        bridge_cell = attach_adsorbate(slab, adsorbate="H2", site_type="bridge", height=1.5)
        assert len(bridge_cell.atoms) == len(slab.atoms) + 1

        hollow_cell = attach_adsorbate(slab, adsorbate="O", site_type="hollow", height=1.2)
        assert len(hollow_cell.atoms) == len(slab.atoms) + 1


class TestExporters:
    def test_export_to_deck(self):
        slab = build_slab("Ni", miller_index=(1, 0, 0), layers=2, repeat=(2, 2), fixed_layers=1)
        deck = slab.to_deck(basis="SZ", title="Ni(100) Test")
        assert isinstance(deck, CalculationDeck)
        assert len(deck.geometry) == len(slab.atoms)
        assert any(a.fixed for a in deck.geometry)
        assert deck.title == "Ni(100) Test"

    def test_export_to_poscar(self):
        slab = build_slab("Ni", miller_index=(1, 0, 0), layers=2, repeat=(2, 2))
        poscar = slab.to_poscar()
        assert "Ni(100)" in poscar
        assert "Direct" in poscar
        lines = poscar.strip().split("\n")
        assert len(lines) >= 8 + len(slab.atoms)

    def test_export_to_cif(self):
        cell = PeriodicCell.bulk("Fe", crystal_structure="bcc")
        cif = cell.to_cif()
        assert "data_" in cif and "Fe" in cif
        assert "_cell_length_a" in cif
        assert "_atom_site_fract_x" in cif

    def test_export_to_xyz(self):
        cell = PeriodicCell.bulk("Ni", crystal_structure="fcc")
        xyz = cell.to_xyz()
        lines = xyz.strip().split("\n")
        assert lines[0].strip() == "4"
        assert "Ni" in lines[2]


class TestCLIIntegration:
    def test_cli_supercell_slab_deck(self):
        from phatpsy.cli import main
        # Run CLI with --bulk to build FCC Cu supercell to YAML stdout
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            ret = main(["supercell", "-e", "Cu", "--bulk", "-r", "1,1,1", "--format", "yaml"])
        assert ret == 0
        output = buf.getvalue()
        assert "element: Cu" in output

    def test_cli_supercell_slab_poscar(self, tmp_path):
        from phatpsy.cli import main
        out_file = tmp_path / "slab.poscar"
        ret = main([
            "supercell",
            "-e", "Ni",
            "-m", "1,0,0",
            "-l", "2",
            "-r", "2,2",
            "-o", str(out_file),
        ])
        assert ret == 0
        assert out_file.exists()
        content = out_file.read_text()
        assert "Direct" in content
