# tests/test_basis_adf.py
"""
Tests for first-class native ADF ZORA STO basis library (DZ, TZP, TZ2P, QZ4P).
Verifies element coverage across all 118 periodic table elements, exact electron
conservation, angular polarization functions, and SCF convergence for N2 and H2O.
"""
import pytest
from phatpsy.basis import (
    get_element_basis,
    get_adf_basis,
    list_adf_elements,
    list_available_elements,
)
from phatpsy.builder import build_diatomic, build_molecule
from phatpsy.driver import PhatpsyDriver


def test_adf_element_coverage_and_tiers():
    """Verify that all 118 elements exist across all 4 ADF tiers."""
    for tier in ("DZ", "TZP", "TZ2P", "QZ4P"):
        elems = list_adf_elements(tier)
        assert len(elems) == 118, f"Expected 118 elements for tier {tier}, found {len(elems)}"
        assert "H" in elems
        assert "C" in elems
        assert "Fe" in elems
        assert "U" in elems
        assert "Og" in elems


def test_adf_electron_conservation():
    """Verify that every element's initial orbitals sum to its exact atomic number Z."""
    for sym in ["H", "He", "Li", "C", "N", "O", "Fe", "Ni", "Cu", "Ag", "Au", "U", "Og"]:
        for tier in ("ADF-DZ", "ADF-TZP", "ADF-TZ2P", "ADF-QZ4P"):
            ed = get_element_basis(sym, tier)
            tot_occ = sum(sum(orb.occno) for orb in ed.default_orbitals)
            assert round(tot_occ) == ed.atomic_number, (
                f"{sym} {tier}: total orbital electrons {tot_occ} != Z ({ed.atomic_number})"
            )


def test_adf_basis_function_hierarchy():
    """Verify increasing basis size from DZ -> TZP -> TZ2P -> QZ4P."""
    for sym in ["H", "C", "N", "O", "Fe"]:
        ed_dz = get_element_basis(sym, "ADF-DZ")
        ed_tzp = get_element_basis(sym, "ADF-TZP")
        ed_tz2p = get_element_basis(sym, "ADF-TZ2P")
        ed_qz4p = get_element_basis(sym, "ADF-QZ4P")

        assert ed_dz.nsto < ed_tzp.nsto < ed_qz4p.nsto
        assert ed_tzp.nsto <= ed_tz2p.nsto
        assert len(ed_dz.default_orbitals) <= len(ed_tzp.default_orbitals) <= len(ed_tz2p.default_orbitals)


def test_adf_alias_resolution():
    """Verify resolution of aliases: TZP, TZ2P, QZ4P, and lowercase/hyphen variations."""
    ed1 = get_element_basis("N", "TZP")
    ed2 = get_element_basis("N", "ADF-TZP")
    assert ed1.basis_name == "ADF-TZP"
    assert ed1.nsto == ed2.nsto
    assert len(ed1.default_orbitals) == len(ed2.default_orbitals)

    ed3 = get_element_basis("C", "TZ2P")
    assert ed3.basis_name == "ADF-TZ2P"

    ed4 = get_element_basis("O", "QZ4P")
    assert ed4.basis_name == "ADF-QZ4P"


def test_adf_available_elements_listing():
    """Verify list_available_elements advertising ADF basis sets for all 118 elements."""
    avail = list_available_elements()
    assert len(avail) >= 118
    assert "ADF-DZ" in avail["C"]
    assert "ADF-TZP" in avail["C"]
    assert "ADF-TZ2P" in avail["C"]
    assert "ADF-QZ4P" in avail["C"]

    # Superheavy element check
    assert "ADF-TZP" in avail["Og"]


def test_adf_n2_scf_convergence():
    """Test SCF convergence of N2 with ADF-TZP and ADF-TZ2P."""
    driver = PhatpsyDriver()

    # N2 with ADF-TZP
    deck_tzp = build_diatomic("N", "N", 2.0744, basis="ADF-TZP", units="bohr")
    res_tzp = driver.run_deck(deck_tzp, quiet=True)
    assert res_tzp.converged
    assert -110.0 <= res_tzp.total_energy <= -104.0
    assert abs(res_tzp.virial_ratio - 2.0) <= 0.08

    # N2 with TZ2P
    deck_tz2p = build_diatomic("N", "N", 2.0744, basis="TZ2P", units="bohr")
    res_tz2p = driver.run_deck(deck_tz2p, quiet=True)
    assert res_tz2p.converged
    assert -110.0 <= res_tz2p.total_energy <= -104.0
    assert abs(res_tz2p.virial_ratio - 2.0) <= 0.08


def test_adf_h2o_scf_convergence():
    """Test SCF convergence of H2O with ADF-TZP."""
    driver = PhatpsyDriver()
    geom = [
        ("O", [0.000000, 0.000000, 0.117300]),
        ("H", [0.000000, 0.757000, -0.469200]),
        ("H", [0.000000, -0.757000, -0.469200]),
    ]
    deck = build_molecule(geom, basis="ADF-TZP", units="angstrom")
    deck.control.use_diis = True
    res = driver.run_deck(deck, quiet=True)
    assert res.converged
    assert -78.0 <= res.total_energy <= -72.0
    assert abs(res.virial_ratio - 2.0) <= 0.05


def test_adf_invalid_element_or_basis_errors():
    """Test proper exceptions on non-existent elements or unknown basis names."""
    with pytest.raises(KeyError, match="not found in ADF"):
        get_adf_basis("FakeElement", "ADF-TZP")

    with pytest.raises(ValueError, match="Invalid ADF basis set"):
        get_adf_basis("C", "ADF-INVALID")
