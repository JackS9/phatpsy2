# tests/test_spin_and_visualizer.py
import pytest
from pathlib import Path
import tempfile

from phatpsy import (
    CalculationDeck,
    AtomConfig,
    SymmetrySpec,
    build_molecule,
    build_diatomic,
    assign_alternating_spin,
    visualize_deck,
)
from phatpsy.spin import build_connectivity_graph, solve_alternating_spin
from phatpsy.visualizer import generate_visualization_data, generate_html
from phatpsy.cli import main as cli_main


def test_connectivity_graph_linear():
    # Linear H4 molecule with bond length 1.4 Bohr
    atoms = [
        AtomConfig(element="H", xyz=[0.0, 0.0, 0.0]),
        AtomConfig(element="H", xyz=[0.0, 0.0, 1.4]),
        AtomConfig(element="H", xyz=[0.0, 0.0, 2.8]),
        AtomConfig(element="H", xyz=[0.0, 0.0, 4.2]),
    ]
    adj = build_connectivity_graph(atoms, tolerance=1.3)
    assert len(adj) == 4
    # Atom 0 connected to 1
    assert 1 in adj[0] and len(adj[0]) == 1
    # Atom 1 connected to 0 and 2
    assert 0 in adj[1] and 2 in adj[1]
    # Atom 2 connected to 1 and 3
    assert 1 in adj[2] and 3 in adj[2]
    # Atom 3 connected to 2
    assert 2 in adj[3] and len(adj[3]) == 1


def test_solve_alternating_spin_bipartite():
    # Linear chain adjacency: 0 - 1 - 2 - 3
    adj = [[1], [0, 2], [1, 3], [2]]
    spins = solve_alternating_spin(adj, start_spin="up")
    assert spins == ["up", "down", "up", "down"]

    spins_down_start = solve_alternating_spin(adj, start_spin="down")
    assert spins_down_start == ["down", "up", "down", "up"]


def test_solve_alternating_spin_frustrated_ag3():
    # Equilateral triangle Ag3 (frustrated)
    # 0 - 1, 1 - 2, 2 - 0
    adj = [[1, 2], [0, 2], [0, 1]]
    spins = solve_alternating_spin(adj, start_spin="up")
    assert len(spins) == 3
    n_up = sum(1 for s in spins if s == "up")
    n_down = sum(1 for s in spins if s == "down")
    # For odd 3-cycle, best spin balance is 2 up / 1 down or 1 up / 2 down
    assert abs(n_up - n_down) == 1


def test_assign_alternating_spin_on_deck():
    # Build linear N4 cluster
    deck = build_molecule(
        geometry=[
            ("N", [0.0, 0.0, 0.0]),
            ("N", [0.0, 0.0, 2.1]),
            ("N", [0.0, 0.0, 4.2]),
            ("N", [0.0, 0.0, 6.3]),
        ],
        basis="SZ",
        auto_spin_flip=True,
    )
    assert deck.geometry[0].spin == "up"
    assert deck.geometry[1].spin == "down"
    assert deck.geometry[2].spin == "up"
    assert deck.geometry[3].spin == "down"

    # Verify to_stdin_text uses QFLIP=T for spin-down atoms
    stdin_text = deck.to_stdin_text()
    assert stdin_text.count("QFLIP=T") == 2


def test_explicit_spin_in_builder_and_xyz():
    # 1. Via tuple
    deck1 = build_molecule(
        geometry=[("N", [0.0, 0.0, 1.0], "up"), ("N", [0.0, 0.0, -1.0], "down")],
        basis="SZ",
    )
    assert deck1.geometry[0].spin == "up"
    assert deck1.geometry[1].spin == "down"
    assert deck1.geometry[1].is_spin_down is True

    # 2. Via XYZ string with 5th column
    xyz_str = """2
    N2 test
    N 0.0 0.0  1.0 up
    N 0.0 0.0 -1.0 down
    """
    deck2 = build_molecule(xyz_str, basis="SZ")
    assert deck2.geometry[0].spin == "up"
    assert deck2.geometry[1].spin == "down"


def test_visualization_data_generation():
    deck = CalculationDeck.from_diatomic("N", bond_length=2.068, basis="DZ", use_symmetry=True)
    # Add virtual and fixed attributes to test visualizer encoding
    deck.geometry[0].fixed = True
    deck.geometry.append(AtomConfig(element="H", xyz=[0.0, 1.5, 0.0], virtual=True))

    data = generate_visualization_data(deck)
    assert data["stats"]["total_atoms"] == 3
    assert data["stats"]["fixed"] == 1
    assert data["stats"]["virtual"] == 1
    assert data["atoms"][0]["fixed"] is True
    assert data["atoms"][2]["virtual"] is True
    assert data["atoms"][1]["symmetry"] is not None


def test_generate_html_content(tmp_path):
    deck = build_molecule(
        geometry=[
            ("C", [0.0, 0.0, 0.0], {"fixed": True, "spin": "up"}),
            ("O", [0.0, 0.0, 2.13], {"spin": "down"}),
            ("Gh.H", [0.0, 1.5, 0.0]),
        ],
        basis="SZ",
    )
    html_file = tmp_path / "view.html"
    html_text = visualize_deck(deck, output_path=html_file, auto_open=False)
    assert html_file.exists()
    assert "three.min.js" in html_text
    assert "OrbitControls.js" in html_text
    assert "Localized VB Spin Arrows" in html_text
    assert "Localized VB Spin Halos &amp; Glow" in html_text
    assert "color-mode-select" in html_text
    assert "TorusGeometry" in html_text
    assert "ArrowHelper" not in html_text
    assert "Fixed Atom Cages" in html_text
    assert "Virtual (Ghost) Centers" in html_text
    assert "0x38bdf8" in html_text  # Cyan spin up
    assert "0xfb923c" in html_text  # Orange spin down


def test_cli_view_and_auto_spin(tmp_path):
    xyz_file = tmp_path / "c2h4.xyz"
    xyz_file.write_text("""6
    Ethylene
    C  0.000000  0.000000  0.669500
    C  0.000000  0.000000 -0.669500
    H  0.000000  0.928900  1.232100
    H  0.000000 -0.928900  1.232100
    H  0.000000  0.928900 -1.232100
    H  0.000000 -0.928900 -1.232100
    """)

    # 1. Test build with --auto-spin
    yaml_out = tmp_path / "c2h4.yaml"
    ret = cli_main(["build", str(xyz_file), "--auto-spin", "-o", str(yaml_out)])
    assert ret == 0
    assert yaml_out.exists()
    deck = CalculationDeck.from_yaml(yaml_out)
    # The two carbons must have opposite spins in ethylene LVB representation
    assert deck.geometry[0].spin != deck.geometry[1].spin

    # 2. Test view command
    html_out = tmp_path / "c2h4.html"
    ret_view = cli_main(["view", str(yaml_out), "-o", str(html_out), "-n"])
    assert ret_view == 0
    assert html_out.exists()
    assert "PHATPSY 3D Viewer" in html_out.read_text()
