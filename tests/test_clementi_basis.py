# tests/test_clementi_basis.py
"""
Unit tests for the Clementi & Roetti (1974) Slater-Type Orbital (STO) dataset
and library integration in PHATPSY.
"""
import json
import pytest
import numpy as np
from pathlib import Path
from phatpsy.basis import (
    get_clementi_basis,
    get_element_basis,
    list_clementi_elements,
    get_thakkar_basis,
)
from phatpsy.ingest_clementi import ELEMENT_INFO, PERIODIC_TABLE

DATA_DIR = Path(__file__).resolve().parent.parent / "phatpsy" / "data"
CLEMENTI_JSON = DATA_DIR / "clementi_sto.json"
THAKKAR_JSON = DATA_DIR / "thakkar_sto.json"


def test_clementi_dataset_file_exists():
    assert CLEMENTI_JSON.exists(), f"Missing {CLEMENTI_JSON}"
    with open(CLEMENTI_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "DZ" in data
    assert "SZ" in data
    assert data["metadata"]["format"] == "orbital-centric"
    assert len(data["DZ"]) == 53
    assert len(data["SZ"]) == 53


@pytest.mark.parametrize("table_name,is_dz", [("DZ", True), ("SZ", False)])
def test_all_53_clementi_elements_structure(table_name, is_dz):
    """Verifies that all 53 elements (Z=2..54) possess exact basis function and orbital counts."""
    with open(CLEMENTI_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    table = data[table_name]
    for z in range(2, 55):
        sym = PERIODIC_TABLE[z]
        assert sym in table, f"Element {sym} (Z={z}) missing from {table_name}"
        el = table[sym]
        assert el["atomic_number"] == z
        assert el["basis_name"] == table_name
        assert el["energy"] is not None
        assert el["energy"] < 0.0

        info = ELEMENT_INFO[z]
        ns = len(info["shells"]["s"])
        np_ = len(info["shells"]["p"])
        nd = len(info["shells"]["d"])
        exp_funcs = (2 if is_dz else 1) * (ns + np_ + nd)
        exp_orbs = ns + np_ + nd

        funcs = el["basis_functions"]
        orbs = el["orbitals"]

        assert len(funcs) == exp_funcs, (
            f"{sym} ({table_name}): expected {exp_funcs} basis functions, got {len(funcs)}"
        )
        assert len(orbs) == exp_orbs, (
            f"{sym} ({table_name}): expected {exp_orbs} orbitals, got {len(orbs)}"
        )

        # Verify orbital-centric coefficients
        for orb in orbs:
            assert "coefficients" in orb
            assert len(orb["coefficients"]) > 0, f"{sym} orbital {orb['name']} has empty coefficients"
            for c_entry in orb["coefficients"]:
                assert "function_index" in c_entry
                assert "c" in c_entry
                assert 1 <= c_entry["function_index"] <= len(funcs)


def test_nitrogen_dz_numerical_accuracy():
    """Verifies Nitrogen DZ against published Clementi table & examples/NDZ.stdin."""
    n_dz = get_clementi_basis("N", "DZ")
    assert n_dz.nsto == 6
    assert len(n_dz.default_orbitals) == 5

    # Check basis function exponents
    expected_zetas = [8.49597, 5.98644, 2.26086, 1.42457, 3.24933, 1.49924]
    actual_zetas = [f.zeta for f in n_dz.functions]
    np.testing.assert_allclose(actual_zetas, expected_zetas, atol=1e-4)

    # Orbital 1S
    orb_1s = n_dz.default_orbitals[0]
    assert np.isclose(orb_1s.eigval[0], -15.62541, atol=1e-3)
    c_1s = [c for _, _, c in orb_1s.coefficients]
    np.testing.assert_allclose(c_1s, [0.27067, 0.73875, -0.00293, 0.00253], atol=1e-3)

    # Orbital 2S
    orb_2s = n_dz.default_orbitals[1]
    assert np.isclose(orb_2s.eigval[0], -0.94316, atol=1e-3)

    # Orbital 2P degenerate components (m_l = -1, 0, 1)
    for p_orb in n_dz.default_orbitals[2:5]:
        assert np.isclose(p_orb.eigval[0], -0.56587, atol=1e-3)
        c_2p = [c for _, _, c in p_orb.coefficients]
        np.testing.assert_allclose(c_2p, [0.28810, 0.77826], atol=1e-3)


def test_nitrogen_sz_numerical_accuracy():
    """Verifies Nitrogen SZ against published Clementi table & examples/n2.stdin."""
    n_sz = get_clementi_basis("N", "SZ")
    assert n_sz.nsto == 3
    assert len(n_sz.default_orbitals) == 5

    expected_zetas = [6.66507, 1.92368, 1.91703]
    actual_zetas = [f.zeta for f in n_sz.functions]
    np.testing.assert_allclose(actual_zetas, expected_zetas, atol=1e-3)


def test_silver_dz_numerical_accuracy():
    """Verifies Silver DZ (Z=47) against examples/ag13dz.stdin."""
    ag_dz = get_clementi_basis("Ag", "DZ")
    assert ag_dz.nsto == 20
    assert ag_dz.atomic_number == 47
    assert len(ag_dz.default_orbitals) == 24  # 5 S + 3*3 P + 5*2 D = 24 orbitals

    # Verify first 1s exponent matches ag13dz.stdin
    assert np.isclose(ag_dz.functions[0].zeta, 48.223, atol=1e-3)
    assert np.isclose(ag_dz.default_orbitals[0].eigval[0], -914.01686, atol=1e-1)


def test_thakkar_orbital_centric_schema():
    """Verifies that Thakkar STO library also follows the orbital-centric schema."""
    with open(THAKKAR_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    for sym in ["H", "C", "N", "Ag", "Xe"]:
        el = data[sym]
        assert "basis_functions" in el
        assert "orbitals" in el
        for orb in el["orbitals"]:
            assert "coefficients" in orb
            assert len(orb["coefficients"]) > 0


def test_element_basis_selector_clementi_and_thakkar():
    """Verifies get_element_basis can query Clementi DZ/SZ or Thakkar seamlessly."""
    # Explicit DZ request
    c_dz = get_element_basis("C", "DZ")
    assert c_dz.basis_name == "DZ"
    assert c_dz.nsto == 6

    # Explicit SZ request
    c_sz = get_element_basis("C", "SZ")
    assert c_sz.basis_name == "SZ"
    assert c_sz.nsto == 3

    # Explicit Thakkar request
    c_thak = get_element_basis("C", "THAKKAR")
    assert c_thak.basis_name == "THAKKAR"
    assert c_thak.nsto == 15


def test_clementi_known_passing_elements_orthonormality():
    """Verifies that currently verified Clementi elements pass orthonormality C^T S C = I."""
    from check_clementi_orthonormality import check_entry, TOL
    with open(CLEMENTI_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    passing = [("SZ", "C"), ("SZ", "B"), ("DZ", "C"), ("DZ", "O")]
    for group, sym in passing:
        worst, problems = check_entry(sym, data[group][sym])
        assert worst <= TOL, f"{group} {sym} worst deviation {worst} > {TOL}"
        assert not problems, f"{group} {sym} problems: {problems}"

