"""
Integration tests for Phase 17 Localized VB CLI, HybridAssembler export,
3D Volumetric grids, Gaussian Cube export, and WebGL HTML visualizer.
"""
import json
import re
from pathlib import Path
import pytest

import phatpsy
from phatpsy.assembler import HybridAssembler
from phatpsy.builder import build_diatomic, build_molecule
from phatpsy.cli import main as cli_main
from phatpsy.schema import CalculationDeck, ControlConfig
from phatpsy.vb_driver import run_localized_vb
from phatpsy.volumetric import compute_grid, compute_vb_orbital_grid
from phatpsy.visualizer import (
    generate_vb_orbitals_visualization_data,
    show_vb_orbitals,
    visualize_vb_orbitals,
)


def test_cli_vb_json_stdout_and_output(tmp_path, capsys):
    ctrl = ControlConfig.localized_vb(basis="DZ", vb_wavefunction="cas")
    deck = build_diatomic("H", "H", 1.40, basis="DZ", spins=["up", "down"], control=ctrl)
    for atom in deck.geometry:
        atom.spin = None
        atom.spin_flip = False
    yaml_path = tmp_path / "h2_vb.yaml"
    json_out = tmp_path / "h2_vb_out.json"
    cube_out = tmp_path / "h2_vb_dens.cube"
    deck.to_yaml(yaml_path)

    rc = cli_main([
        "vb",
        str(yaml_path),
        "--compare-regimes",
        "--json-stdout",
        "-o", str(json_out),
        "--cube", str(cube_out),
    ])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["converged"] is True
    assert payload["method"] == "localized_vb"
    assert "regime_comparison" in payload
    assert set(("strict_1c", "bond_local", "delocalized")).issubset(payload["regime_comparison"])
    assert payload["total_energy"] < -1.10
    assert payload["binding_energy_ev"] > 2.8
    assert json_out.exists()
    assert cube_out.exists()
    assert "PHATPSY" in cube_out.read_text(encoding="utf-8")


def test_assembler_to_localized_vb_deck_and_cli_assemble(tmp_path):
    asm = HybridAssembler(root="O", hybridization="sp3_bent")
    asm.cap_all("H")
    deck = asm.to_localized_vb_deck(
        basis="DZ",
        domain_mode="custom",
        active_atoms=["O1", "H1", "H2"],
    )
    assert deck.control.method == "localized_vb"
    assert deck.control.spin_unrestricted is False
    assert deck.net_spin == 0
    assert len(deck.geometry) == 3
    assert all(a.spin is None for a in deck.geometry)
    assert all(a.is_active for a in deck.geometry)
    assert deck.geometry[0].orbital_domains == [["O1", "H1", "H2"]]

    out_yaml = tmp_path / "water_assembled_vb.yaml"
    rc = cli_main([
        "assemble",
        "-r", "O",
        "--hybridization", "sp3_bent",
        "-c", "H",
        "--localized-vb",
        "-o", str(out_yaml),
    ])
    assert rc == 0
    assert out_yaml.exists()
    reloaded = CalculationDeck.from_yaml(out_yaml)
    assert reloaded.control.method == "localized_vb"
    assert reloaded.control.spin_unrestricted is False
    assert all(a.spin is None for a in reloaded.geometry)


def test_cli_build_localized_vb_without_uhf_spins(tmp_path):
    xyz_path = tmp_path / "h2.xyz"
    xyz_path.write_text("2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n", encoding="utf-8")
    out_yaml = tmp_path / "h2_built_vb.yaml"
    rc = cli_main(["build", str(xyz_path), "--localized-vb", "-b", "DZ", "-o", str(out_yaml)])
    assert rc == 0
    assert out_yaml.exists()
    deck = CalculationDeck.from_yaml(out_yaml)
    assert deck.control.method == "localized_vb"
    assert deck.control.spin_unrestricted is False
    assert deck.control.default_basis == "DZ"
    assert all(a.spin is None for a in deck.geometry)


def test_volumetric_vb_orbitals_density_integral_and_cube_cli(tmp_path):
    ctrl = ControlConfig.localized_vb(basis="DZ", vb_wavefunction="cas")
    deck = build_diatomic("H", "H", 1.40, basis="DZ", spins=["up", "down"], control=ctrl)
    for atom in deck.geometry:
        atom.spin = None
        atom.spin_flip = False

    state = run_localized_vb(deck, quiet=True)
    grid_dens = compute_grid(state, grid_type="density", spacing=0.25, padding=4.5)
    assert grid_dens.integrate() == pytest.approx(2.0, abs=0.05)

    grid_loc = compute_vb_orbital_grid(state, orbital_index=1, orthonormal=False, spacing=0.25, padding=4.0)
    assert grid_loc.data.shape[0] >= 10
    assert grid_loc.max_value > 0.1

    grid_ortho = compute_vb_orbital_grid(state, orbital_index=1, orthonormal=True, spacing=0.25, padding=4.0)
    assert grid_ortho.min_value < -1e-3
    assert grid_ortho.max_value > 1e-3

    # CLI phatpsy cube --vb-orbital 1
    in_yaml = tmp_path / "h2_cube.yaml"
    out_cube = tmp_path / "h2_orb1.cube"
    deck.to_yaml(in_yaml)
    rc = cli_main([
        "cube", str(in_yaml),
        "--vb-orbital", "1",
        "--spacing", "0.35",
        "--padding", "3.5",
        "-o", str(out_cube),
    ])
    assert rc == 0
    assert out_cube.exists()
    cube_lines = out_cube.read_text(encoding="utf-8").splitlines()
    assert len(cube_lines) > 10


def test_visualize_vb_orbitals_html_and_top_level_exports(tmp_path):
    assert hasattr(phatpsy, "compute_vb_orbital_grid")
    assert hasattr(phatpsy, "visualize_vb_orbitals")
    assert hasattr(phatpsy, "show_vb_orbitals")

    ctrl = ControlConfig.localized_vb(basis="DZ", vb_wavefunction="cas")
    deck = build_diatomic("H", "H", 1.40, basis="DZ", spins=["up", "down"], control=ctrl)
    for atom in deck.geometry:
        atom.spin = None
        atom.spin_flip = False

    state = run_localized_vb(deck, quiet=True)
    html_path = tmp_path / "h2_vb_orbs.html"
    html_str = visualize_vb_orbitals(
        state,
        orthonormal=True,
        isovalue=0.04,
        spacing=0.3,
        padding=3.5,
        output_path=html_path,
        auto_open=False,
    )
    assert html_path.exists()
    assert "vb-orb-toggle" in html_str
    assert "Active / Spectator" in html_str
    assert "Three" in html_str or "THREE" in html_str
    # Orthonormalized localized orbital exhibits both + and - phase lobes
    assert "(+0.040)" in html_str
    assert "(-0.040)" in html_str


def test_show_vb_orbitals_inline_canvas_no_iframe_or_cdn():
    # 1. Diatomic H2 (no inactive core orbitals -> all default visible)
    ctrl_h2 = ControlConfig.localized_vb(basis="DZ", vb_wavefunction="cas")
    deck_h2 = build_diatomic("H", "H", 1.40, basis="DZ", control=ctrl_h2, spin_unrestricted=False, net_spin=0)
    state_h2 = run_localized_vb(deck_h2, quiet=True)
    disp_h2 = show_vb_orbitals(state_h2, orthonormal=True, isovalue=0.04, spacing=0.35, padding=3.5, height=500)
    html_h2 = disp_h2.data if hasattr(disp_h2, "data") else str(disp_h2)

    assert "<canvas" in html_h2
    assert "<script>" in html_h2
    assert "<iframe" not in html_h2.lower()
    assert "srcdoc=" not in html_h2.lower()
    assert "<script src=" not in html_h2.lower()
    assert "https://" not in re.findall(r"<script[^>]*>", html_h2.lower())
    assert "vb-orb-toggle" in html_h2
    assert "Active / Spectator" in html_h2
    assert "(+0.040)" in html_h2
    assert "(-0.040)" in html_h2

    m_h2 = re.search(r"var MOL_DATA = (\{.*?\});\n", html_h2, re.S)
    assert m_h2 is not None
    mol_h2 = json.loads(m_h2.group(1))
    assert len(mol_h2["surfaces"]) >= 2
    assert all(len(s["vertices"]) > 0 and len(s["faces"]) > 0 for s in mol_h2["surfaces"])
    assert all(item["visible"] is True for item in mol_h2["vb_orbital_items"])

    # 2. Polyatomic H2O (inactive O 1s core hidden by default; active valence orbitals visible)
    ctrl_h2o = ControlConfig.localized_vb(basis="DZ", domain_mode="bond_local")
    deck_h2o = build_molecule(
        [("O", 0.0, 0.0, 0.0), ("H", 1.430, 0.0, 1.107), ("H", -1.430, 0.0, 1.107)],
        basis="DZ",
        control=ctrl_h2o,
        spin_unrestricted=False,
        spins=None,
        net_spin=0,
    )
    state_h2o = run_localized_vb(deck_h2o, quiet=True)
    disp_h2o = show_vb_orbitals(
        state_h2o,
        isovalue=0.05,
        spacing=0.35,
        padding=3.5,
        height=550,
        title="H2O Localized VB Orbitals (Bond-Local MaskedVB)",
    )
    html_h2o = disp_h2o.data if hasattr(disp_h2o, "data") else str(disp_h2o)

    assert "<canvas" in html_h2o
    assert "<script>" in html_h2o
    assert "<iframe" not in html_h2o.lower()
    assert "srcdoc=" not in html_h2o.lower()
    assert "<script src=" not in html_h2o.lower()
    assert "H2O Localized VB Orbitals (Bond-Local MaskedVB)" in html_h2o
    assert "(+0.050)" in html_h2o
    assert "(-0.050)" in html_h2o

    m_h2o = re.search(r"var MOL_DATA = (\{.*?\});\n", html_h2o, re.S)
    assert m_h2o is not None
    mol_h2o = json.loads(m_h2o.group(1))
    items_h2o = mol_h2o["vb_orbital_items"]
    assert any(it.get("role") == "inactive" and it["visible"] is False for it in items_h2o)
    assert any(it.get("role") != "inactive" and it["visible"] is True for it in items_h2o)
    assert len(mol_h2o["surfaces"]) >= 4
    assert all(len(s["vertices"]) > 0 and len(s["faces"]) > 0 for s in mol_h2o["surfaces"])

    # Unescaped literal braces check: JS placeholder tokens must be substituted cleanly
    assert "__UID__" not in html_h2o
    assert "__HEIGHT__" not in html_h2o
    assert "__PAYLOAD__" not in html_h2o

    # Explicit orbital_indices overrides default inactive hiding
    data_explicit = generate_vb_orbitals_visualization_data(state_h2o, orbital_indices=[1], spacing=0.45, padding=3.5)
    assert data_explicit["vb_orbital_items"][0]["visible"] is True
    assert all(it["visible"] is False for it in data_explicit["vb_orbital_items"][1:])
