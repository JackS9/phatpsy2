"""
Unit and integration tests for the Stepwise Valence-Bond Molecular Assembler (Phase 14).
Tests Milestones M14.1 - M14.3:
- Topology, slots, and bond length heuristics.
- Stepwise atom additions with staggered, eclipsed, and double/triple/delta bonding.
- Auto-reservation of lone pairs on electronegative species (N, O).
- Quantum state seeding bridge to CalculationDeck.
- Real SCF convergence on assembled geometries.
"""
import math
import numpy as np
import pytest

from phatpsy.assembler import (
    HybridAssembler,
    HybridNode,
    ValenceSlot,
    ValenceSlotType,
    get_bond_length,
)
from phatpsy.spin import BOHR_PER_ANGSTROM
from phatpsy.driver import PhatpsyDriver


# ==============================================================================
# M14.1: Bond Lengths and Basic Topology
# ==============================================================================

def test_bond_length_heuristics():
    # Standard equilibrium values
    c_h = get_bond_length("C", "H", bond_order=1, units="angstrom")
    assert pytest.approx(c_h, abs=1e-3) == 1.090

    c_c_single = get_bond_length("C", "C", bond_order=1, units="angstrom")
    c_c_double = get_bond_length("C", "C", bond_order=2, units="angstrom")
    c_c_triple = get_bond_length("C", "C", bond_order=3, units="angstrom")
    assert pytest.approx(c_c_single, abs=1e-3) == 1.535
    assert pytest.approx(c_c_double, abs=1e-3) == 1.339
    assert pytest.approx(c_c_triple, abs=1e-3) == 1.203

    # Transition metal quadruple/delta bond
    cr_cr_4 = get_bond_length("Cr", "Cr", bond_order=4, units="angstrom")
    assert pytest.approx(cr_cr_4, abs=1e-3) == 1.970

    # In atomic units (Bohr)
    c_h_bohr = get_bond_length("C", "H", bond_order=1, units="bohr")
    assert pytest.approx(c_h_bohr, abs=1e-3) == 1.090 * BOHR_PER_ANGSTROM


def test_assembler_root_initialization():
    asm = HybridAssembler(root="C", hybridization="sp3")
    assert len(asm.nodes) == 1
    assert "C1" in asm.nodes

    root = asm.nodes["C1"]
    assert root.element == "C"
    assert root.hybridization == "sp3"
    assert len(root.slots) == 4
    assert len(root.open_slots) == 4

    # Check tetrahedral directions: angle between any pair should be ~109.47 degrees
    u0 = root.slots[0].direction
    u1 = root.slots[1].direction
    cos_angle = np.dot(u0, u1)
    assert pytest.approx(cos_angle, abs=1e-4) == -1.0 / 3.0


# ==============================================================================
# M14.2: Stepwise Growth, Capping, and Conformations
# ==============================================================================

def test_build_methane():
    # C(sp3) capped with 4 hydrogens -> CH4
    asm = HybridAssembler(root="C", hybridization="sp3")
    assert len(asm.get_open_slots()) == 4

    capped = asm.cap_all(element="H")
    assert len(capped) == 4
    assert len(asm.nodes) == 5
    assert len(asm.get_open_slots()) == 0

    c = asm.nodes["C1"]
    for h_idx in range(1, 5):
        h = asm.nodes[f"H{h_idx}"]
        dist_bohr = np.linalg.norm(c.position - h.position)
        dist_aa = dist_bohr / BOHR_PER_ANGSTROM
        assert pytest.approx(dist_aa, abs=1e-3) == 1.090


def test_build_ethane_staggered_vs_eclipsed():
    # 1. Staggered Ethane (default)
    asm_stag = HybridAssembler(root="C", hybridization="sp3")
    asm_stag.attach("C1", slot=0, element="C", hybridization="sp3", conformation="staggered")
    asm_stag.cap_all("H")
    assert len(asm_stag.nodes) == 8  # 2 C + 6 H

    c1 = asm_stag.nodes["C1"]
    c2 = asm_stag.nodes["C2"]
    cc_dist = np.linalg.norm(c1.position - c2.position) / BOHR_PER_ANGSTROM
    assert pytest.approx(cc_dist, abs=1e-3) == 1.535

    # Check dihedral angle between C1-H and C2-H arms around C-C axis
    # In staggered conformation, dihedral is ~60 degrees
    u_cc = (c2.position - c1.position) / np.linalg.norm(c2.position - c1.position)
    # Arms on C1 (excluding bond to C2)
    c1_arms = [s.direction for s in c1.slots if s.target_atom_id != "C2"]
    # Arms on C2 (excluding bond to C1)
    c2_arms = [s.direction for s in c2.slots if s.target_atom_id != "C1"]

    # Project first arm of each onto plane perpendicular to C-C
    p1 = c1_arms[0] - np.dot(c1_arms[0], u_cc) * u_cc
    p2 = c2_arms[0] - np.dot(c2_arms[0], u_cc) * u_cc
    p1_u = p1 / np.linalg.norm(p1)
    p2_u = p2 / np.linalg.norm(p2)
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, np.dot(p1_u, p2_u)))))
    assert pytest.approx(angle_deg, abs=2.0) == 60.0

    # 2. Eclipsed Ethane
    asm_ecl = HybridAssembler(root="C", hybridization="sp3")
    asm_ecl.attach("C1", slot=0, element="C", hybridization="sp3", conformation="eclipsed")
    c1_e = asm_ecl.nodes["C1"]
    c2_e = asm_ecl.nodes["C2"]
    c1_arms_e = [s.direction for s in c1_e.slots if s.target_atom_id != "C2"]
    c2_arms_e = [s.direction for s in c2_e.slots if s.target_atom_id != "C1"]
    p1_e = c1_arms_e[0] - np.dot(c1_arms_e[0], u_cc) * u_cc
    p2_e = c2_arms_e[0] - np.dot(c2_arms_e[0], u_cc) * u_cc
    angle_ecl = math.degrees(math.acos(max(-1.0, min(1.0, np.dot(p1_e / np.linalg.norm(p1_e), p2_e / np.linalg.norm(p2_e))))))
    assert pytest.approx(angle_ecl, abs=2.0) == 0.0


def test_build_ethylene_double_bond():
    # C(sp2) = C(sp2) double bond -> C2H4
    asm = HybridAssembler(root="C", hybridization="sp2")
    asm.attach("C1", slot=0, element="C", hybridization="sp2", bond_order=2)
    asm.cap_all("H")
    assert len(asm.nodes) == 6  # 2 C + 4 H

    c1 = asm.nodes["C1"]
    c2 = asm.nodes["C2"]
    cc_dist = np.linalg.norm(c1.position - c2.position) / BOHR_PER_ANGSTROM
    assert pytest.approx(cc_dist, abs=1e-3) == 1.339

    # Planar geometry: all 6 atoms should lie in the same z=0 plane
    z_coords = [n.position[2] for n in asm.nodes.values()]
    assert max(abs(z) for z in z_coords) < 1e-4


def test_build_acetylene_triple_bond():
    # C(sp) # C(sp) triple bond -> C2H2
    asm = HybridAssembler(root="C", hybridization="sp")
    asm.attach("C1", slot=0, element="C", hybridization="sp", bond_order=3)
    asm.cap_all("H")
    assert len(asm.nodes) == 4  # 2 C + 2 H

    c1 = asm.nodes["C1"]
    c2 = asm.nodes["C2"]
    cc_dist = np.linalg.norm(c1.position - c2.position) / BOHR_PER_ANGSTROM
    assert pytest.approx(cc_dist, abs=1e-3) == 1.203

    # Linear geometry: all atoms lie along the z axis (x=y=0)
    for n in asm.nodes.values():
        assert pytest.approx(n.position[0], abs=1e-4) == 0.0
        assert pytest.approx(n.position[1], abs=1e-4) == 0.0


def test_auto_lone_pairs_water_and_ammonia():
    # 1. Water via generic sp3: O(sp3) has 4 slots, 2 automatically reserved as lone pairs -> 2 open slots
    asm_w_sp3 = HybridAssembler(root="O", hybridization="sp3")
    o_sp3 = asm_w_sp3.nodes["O1"]
    assert len(o_sp3.slots) == 4
    assert len(o_sp3.open_slots) == 2
    assert sum(1 for s in o_sp3.slots if s.slot_type == ValenceSlotType.LONE_PAIR) == 2
    asm_w_sp3.cap_all("H")
    assert len(asm_w_sp3.nodes) == 3  # H2O

    # 2. Water via specialized sp3_bent: O(sp3_bent) already defines 2 bonding arms -> 2 open slots
    asm_w_bent = HybridAssembler(root="O", hybridization="sp3_bent")
    o_bent = asm_w_bent.nodes["O1"]
    assert len(o_bent.slots) == 2
    assert len(o_bent.open_slots) == 2
    asm_w_bent.cap_all("H")
    assert len(asm_w_bent.nodes) == 3  # H2O
    h1 = asm_w_bent.nodes["H1"]
    oh_dist = np.linalg.norm(o_bent.position - h1.position) / BOHR_PER_ANGSTROM
    assert pytest.approx(oh_dist, abs=1e-3) == 0.958

    # 3. Ammonia via generic sp3: N(sp3) has 4 slots, 1 automatically reserved as lone pair -> 3 open slots
    asm_n_sp3 = HybridAssembler(root="N", hybridization="sp3")
    n_sp3 = asm_n_sp3.nodes["N1"]
    assert len(n_sp3.slots) == 4
    assert len(n_sp3.open_slots) == 3
    assert sum(1 for s in n_sp3.slots if s.slot_type == ValenceSlotType.LONE_PAIR) == 1
    asm_n_sp3.cap_all("H")
    assert len(asm_n_sp3.nodes) == 4  # NH3

    # 4. Ammonia via specialized sp3_pyramidal: N(sp3_pyramidal) already defines 3 bonding arms -> 3 open slots
    asm_n_pyr = HybridAssembler(root="N", hybridization="sp3_pyramidal")
    n_pyr = asm_n_pyr.nodes["N1"]
    assert len(n_pyr.slots) == 3
    assert len(n_pyr.open_slots) == 3
    asm_n_pyr.cap_all("H")
    assert len(asm_n_pyr.nodes) == 4  # NH3


def test_transition_metal_delta_bond():
    # Cr(delta) quadruple bond
    asm = HybridAssembler(root="Cr", hybridization="delta")
    cr1 = asm.nodes["Cr1"]
    assert len(cr1.delta_directions) == 2

    # Attach second Cr with quadruple bond (order=4)
    cr2 = asm.attach("Cr1", slot=0, element="Cr", hybridization="delta", bond_order=4)
    dist_aa = np.linalg.norm(cr1.position - cr2.position) / BOHR_PER_ANGSTROM
    assert pytest.approx(dist_aa, abs=1e-3) == 1.970


def test_stepwise_ethanol_construction():
    # Build Ethanol: C1(sp3) -> C2(sp3) -> O1(sp3_bent) -> cap all H
    asm = HybridAssembler(root="C", hybridization="sp3")
    c2 = asm.attach("C1", slot=0, element="C", hybridization="sp3", conformation="staggered")
    o1 = asm.attach("C2", slot=1, element="O", hybridization="sp3_bent", conformation="staggered")
    asm.cap_all("H")

    # Formula: C2H6O (Ethanol = 9 atoms: 2 C, 1 O, 6 H)
    assert len(asm.nodes) == 9
    elem_counts = {}
    for n in asm.nodes.values():
        elem_counts[n.element] = elem_counts.get(n.element, 0) + 1
    assert elem_counts == {"C": 2, "O": 1, "H": 6}


# ==============================================================================
# M14.3: CalculationDeck Export & SCF Quantum Convergence
# ==============================================================================

def test_export_to_deck_and_starting_orbitals():
    # Build Water and export to deck
    asm = HybridAssembler(root="O", hybridization="sp3_bent")
    asm.cap_all("H")

    deck = asm.to_deck(title="Assembled Water", basis="DZ", spins="up", net_spin=3)
    assert deck.title == "Assembled Water"
    assert len(deck.geometry) == 3
    assert deck.geometry[0].element == "O"
    assert deck.geometry[0].hybridization == "sp3_bent"

    # Verify starting orbitals were initialized from hybrid seeds
    assert deck.geometry[0].starting_orbitals is not None
    assert len(deck.geometry[0].starting_orbitals) > 0


def test_scf_calculation_on_assembled_water():
    """Runs a full SCF calculation on the assembled water deck to verify physical invariants."""
    asm = HybridAssembler(root="O", hybridization="sp3_bent")
    asm.cap_all("H")

    deck = asm.to_deck(title="Water from Assembler", basis="DZ", spins="up", net_spin=3)
    driver = PhatpsyDriver()
    res = driver.run_deck(deck, quiet=True)

    assert res is not None
    assert -78.0 <= res.total_energy <= -72.0
    # Virial theorem sanity check: |-V/T - 2.0| <= 0.08
    assert abs(res.virial_ratio - 2.0) <= 0.08


# ==============================================================================
# M14.4: 3D Visualization, XYZ/ASE Export & CLI Subcommand
# ==============================================================================

def test_assembler_to_xyz_and_ase():
    asm = HybridAssembler(root="C", hybridization="sp3")
    asm.cap_all("H")

    # XYZ export
    xyz_str = asm.to_xyz()
    assert "5\n" in xyz_str
    assert "C " in xyz_str
    assert "H " in xyz_str

    # ASE export
    ase_atoms = asm.to_ase()
    assert len(ase_atoms) == 5
    assert ase_atoms.get_chemical_symbols() == ["C", "H", "H", "H", "H"]


def test_assembler_html_visualization(tmp_path):
    asm = HybridAssembler(root="C", hybridization="sp3")
    asm.attach("C1", slot=0, element="C", hybridization="sp3")

    html_file = tmp_path / "assembler.html"
    html_content = asm.visualize(output_path=html_file, auto_open=False)

    assert html_file.exists()
    assert "Three.js" in html_content or "three.min.js" in html_content
    assert "Molecular Assembler" in html_content
    assert "placeholderGroup" in html_content
    assert "btn-cap-all" in html_content


def test_assembler_cli_subcommand(tmp_path):
    from phatpsy.cli import main

    out_yaml = tmp_path / "methane.yaml"
    rc = main(["assemble", "-r", "C", "--hybridization", "sp3", "-o", str(out_yaml)])
    assert rc == 0
    assert out_yaml.exists()
    content = out_yaml.read_text(encoding="utf-8")
    assert "element: C" in content
    assert "element: H" in content

    # Test HTML export via CLI
    out_html = tmp_path / "methane.html"
    rc_html = main(["assemble", "-r", "C", "--hybridization", "sp3", "-o", str(out_html), "-n"])
    assert rc_html == 0
    assert out_html.exists()
    assert "Molecular Assembler" in out_html.read_text(encoding="utf-8")
