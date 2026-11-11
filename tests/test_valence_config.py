# tests/test_valence_config.py
"""
Acceptance test suite for orbital-level valence configuration (ValenceOrbital and AtomConfig.valence_config)
per valence-config-spec.md Section 9 and Meta Muse review amendments.
"""
import warnings
import pytest
import yaml

from phatpsy import (
    AtomConfig,
    CalculationDeck,
    ValenceOrbital,
    ATOMIC_NUMBERS,
    NOBLE_CORES,
    NOBLE_NEXT,
    core,
    core_next,
    total_valency_available,
    n_valence_orbitals,
    occupied_valency,
    orbital_counts,
    occupancy_counts,
    valence_net_spin,
    net_orbital_spin,
    build_molecule,
    propose_valence_config,
    get_element_basis,
    build_atom_orbitals,
)
from phatpsy.basis import InitialOrbital
from phatpsy.formats import PERIODIC_TABLE


# ==============================================================================
# 1. Formula Table & Periodic Table Cross-Check (Section 2 & 9.1)
# ==============================================================================

def test_atomic_numbers_cross_check_against_formats_periodic_table():
    """Verify ATOMIC_NUMBERS in schema.py matches formats.PERIODIC_TABLE for Z=1..118."""
    assert len(ATOMIC_NUMBERS) == 118
    for z in range(1, 119):
        sym = PERIODIC_TABLE[z]
        assert ATOMIC_NUMBERS[sym] == z


@pytest.mark.parametrize(
    "sym, z, exp_core, exp_next, exp_total, exp_norb, exp_occ",
    [
        ("H",  1,  0,  2,  2,  1,  1),
        ("He", 2,  2,  2,  0,  0,  0),
        ("C",  6,  2, 10,  8,  4,  4),
        ("N",  7,  2, 10,  8,  4,  5),
        ("O",  8,  2, 10,  8,  4,  6),
        ("Ar", 18, 18, 18, 0,  0,  0),
        ("K",  19, 18, 36, 18, 9,  1),
        ("Fe", 26, 18, 36, 18, 9,  8),
        ("Ga", 31, 18, 36, 18, 9, 13),  # Noble-gas core=18 (not 28)
        ("Rb", 37, 36, 54, 18, 9,  1),  # Catches missing-54 bug
        ("Xe", 54, 54, 54, 0,  0,  0),
        ("Cs", 55, 54, 86, 32, 16, 1),
        ("Rn", 86, 86, 86, 0,  0,  0),
        ("Fr", 87, 86, 118, 32, 16, 1),
    ],
)
def test_valence_formula_table(sym, z, exp_core, exp_next, exp_total, exp_norb, exp_occ):
    assert ATOMIC_NUMBERS[sym] == z
    assert core(z) == exp_core
    assert core_next(z) == exp_next
    assert total_valency_available(z) == exp_total
    assert n_valence_orbitals(z) == exp_norb
    assert occupied_valency(z, charge=0.0) == exp_occ


# ==============================================================================
# 2. Occupied Valency with Formal Charge (Section 2 & 9.2)
# ==============================================================================

@pytest.mark.parametrize(
    "sym, z, charge, exp_occ",
    [
        ("Fe", 26, +2.0, 6),   # Fe(II) -> 6
        ("Fe", 26, +3.0, 5),   # Fe(III) -> 5
        ("O",  8,  -1.0, 7),   # O- -> 8 - 2 - (-1) = 7
        ("N",  7,  +1.0, 4),   # N+ -> 7 - 2 - (+1) = 4
        ("B",  5,  -1.0, 4),   # B- -> 5 - 2 - (-1) = 4
        ("C",  6,  +1.0, 3),   # C+ -> 6 - 2 - (+1) = 3
    ],
)
def test_occupied_valency_with_formal_charges(sym, z, charge, exp_occ):
    assert occupied_valency(z, charge=charge) == exp_occ


# ==============================================================================
# 3. Worked Examples from Section 7 (Section 9.3)
# ==============================================================================

def test_eo_oxygen_mixed_spins_example():
    """
    EO oxygen: sp3 hybridization, 2 sigma bonds with opposite spins (antiferromagnetically
    bonded to C1 and C2), 2 lone pairs, and zero ferromagnetic links across the 3-membered ring.
    """
    from pathlib import Path

    yaml_str = """
    element: O
    id: O1
    hybridization: sp3
    valence_config:
      - {spin: down, orbital_type: sigma, occupation: 1, connect_to: C1}
      - {spin: up, orbital_type: sigma, occupation: 1, connect_to: C2}
      - {spin: none, orbital_type: lone_pair, occupation: 2, plane: [C1, C2], orientation: in_plane}
      - {spin: none, orbital_type: lone_pair, occupation: 2, plane: [C1, C2], orientation: out_of_plane}
    """
    atom = AtomConfig.from_dict(yaml.safe_load(yaml_str))
    assert atom.hybridization == "sp3"
    assert occupancy_counts(atom.valence_config) == (0, 2, 2)
    assert orbital_counts(atom.valence_config) == {"sigma": 2, "pi": 0, "delta": 0, "lone_pair": 2}
    assert valence_net_spin(atom.valence_config) == 0
    assert net_orbital_spin(atom.valence_config) == 0
    # Net orbital spin is 0, so atom.spin is not forced by valence_config
    assert atom.spin is None

    # Full Ethylene Oxide (EO.yaml) and Transition State (EOtoAA.yaml):
    # No ferromagnetic bond between C1 and C2; O1 has mixed-spin sp3 orbitals antiferromagnetically bonded to C1 and C2
    repo_root = Path(__file__).resolve().parent.parent
    for fname in ("EO.yaml", "EOtoAA.yaml"):
        deck_eo = CalculationDeck.from_yaml(repo_root / "examples" / fname)
        assert deck_eo.ferromagnetic_links == []
        assert deck_eo.physical_net_spin == 0
        c1, c2, o1 = deck_eo.geometry[0], deck_eo.geometry[1], deck_eo.geometry[2]
        assert c1.hybridization == "sp3" and c1.net_orbital_spin == 4
        assert c2.hybridization == "sp3" and c2.net_orbital_spin == -4
        assert o1.hybridization == "sp3" and o1.net_orbital_spin == 0
        # C1-C2 bond is antiferromagnetically paired (C1 up <-> C2 down)
        c1_to_c2 = next(vo for vo in c1.valence_config if vo.connect_to == "C2")
        c2_to_c1 = next(vo for vo in c2.valence_config if vo.connect_to == "C1")
        assert c1_to_c2.spin == "up" and c2_to_c1.spin == "down"
        # O1 is antiferromagnetically paired to both C1 (up <-> down) and C2 (down <-> up)
        o1_to_c1 = next(vo for vo in o1.valence_config if vo.connect_to == "C1")
        o1_to_c2 = next(vo for vo in o1.valence_config if vo.connect_to == "C2")
        assert o1_to_c1.spin == "down" and o1_to_c2.spin == "up"


def test_no_radical_non_alternant_pi_and_check_11():
    """
    NO radical (R = 2.1747 Bohr):
    Valence MO occupancy (sigma_2s)^2 (sigma*_2s)^2 (pi_2p)^4 (sigma_2p)^2 (pi*_2p)^1,
    formal MO bond order = (8 bonding - 3 antibonding) / 2 = 2.5,
    N has net orbital spin +1 (spin='up'), O has 0, molecule physical_net_spin = +1 (doublet).
    """
    yaml_str = """
    title: NO Radical
    net_spin: 0
    geometry:
      - element: N
        id: N1
        xyz: [0.0, 0.0, 0.0]
        hybridization: sp
        valence_config:
          - {spin: up, orbital_type: sigma, occupation: 1, connect_to: O1}
          - {spin: down, orbital_type: pi, occupation: 1, connect_to: O1}
          - {spin: up, orbital_type: pi, occupation: 1, connect_to: O1}
          - {spin: none, orbital_type: lone_pair, occupation: 2}
      - element: O
        id: O1
        xyz: [0.0, 0.0, 2.1747]
        spin: down
        hybridization: sp
        valence_config:
          - {spin: down, orbital_type: sigma, occupation: 1, connect_to: N1}
          - {spin: up, orbital_type: pi, occupation: 1, connect_to: N1}
          - {spin: none, orbital_type: lone_pair, occupation: 2}
          - {spin: none, orbital_type: lone_pair, occupation: 2}
    """
    deck = CalculationDeck.from_yaml(yaml_str)
    n_atom, o_atom = deck.geometry
    assert n_atom.spin == "up"
    assert o_atom.spin == "down"
    assert o_atom.xyz[2] == pytest.approx(2.1747)
    assert valence_net_spin(n_atom.valence_config) == 1
    assert valence_net_spin(o_atom.valence_config) == 0
    assert deck.net_spin == 0
    assert deck.physical_net_spin == 1
    # 11 valence electrons: 8 bonding, 3 antibonding -> formal MO bond order = 2.5
    total_val_e = sum(o.occupation for a in deck.geometry for o in a.valence_config)
    assert total_val_e == 11
    assert (8 - 3) / 2 == 2.5

    # Also test build_molecule with physical_net_spin assertion
    deck2 = build_molecule(
        deck.geometry,
        basis="DZ",
        net_spin=0,
        physical_net_spin=1,
    )
    assert deck2.physical_net_spin == 1

    with pytest.raises(ValueError, match="Physical net spin"):
        build_molecule(
            deck.geometry,
            basis="DZ",
            net_spin=0,
            physical_net_spin=0,
        )


def test_fe_ii_octahedral_high_spin_d6_example():
    """Fe(II) high-spin d6: Z=26, core=18, charge=+2 -> occupied=6, 9 orbitals."""
    yaml_str = """
    element: Fe
    id: Fe1
    charge: 2.0
    valence_config:
      - {spin: up, orbital_type: sigma, occupation: 1, connect_to: L1}
      - {spin: up, orbital_type: sigma, occupation: 1, connect_to: L2}
      - {spin: up, orbital_type: pi, occupation: 1, connect_to: L3, plane: [L3, L4], orientation: out_of_plane}
      - {spin: up, orbital_type: pi, occupation: 1, connect_to: L4, plane: [L3, L4], orientation: out_of_plane}
      - {spin: none, orbital_type: pi, occupation: 2, connect_to: L5, plane: [L5, L6], orientation: out_of_plane}
      - {spin: none, orbital_type: sigma, occupation: 0}
      - {spin: none, orbital_type: sigma, occupation: 0}
      - {spin: none, orbital_type: sigma, occupation: 0}
      - {spin: none, orbital_type: sigma, occupation: 0}
    """
    fe = AtomConfig.from_dict(yaml.safe_load(yaml_str))
    assert occupancy_counts(fe.valence_config) == (4, 4, 1)
    assert valence_net_spin(fe.valence_config) == 4
    assert fe.spin == "up"

    # Also test implicit padding of the 4 empty 4s/4p orbitals (Rule 7)
    fe_short = AtomConfig(
        element="Fe",
        id="Fe1",
        charge=2.0,
        valence_config=fe.valence_config[:5],
    )
    assert len(fe_short.valence_config) == 5
    assert occupancy_counts(fe_short.valence_config, n_orbitals=n_valence_orbitals(26)) == (4, 4, 1)


# ==============================================================================
# 4. Dative Pairing Checks (Section 6 & 9.4)
# ==============================================================================

def _nh3_bh3_atoms():
    return [
        AtomConfig.from_dict({
            "element": "N",
            "id": "N1",
            "xyz": [0.0, 0.0, 0.0],
            "spin": "up",
            "valence_config": [
                {"spin": "up", "orbital_type": "sigma", "occupation": 1, "connect_to": "H1"},
                {"spin": "down", "orbital_type": "sigma", "occupation": 1, "connect_to": "H2"},
                {"spin": "up", "orbital_type": "sigma", "occupation": 1, "connect_to": "H3"},
                {"spin": "none", "orbital_type": "lone_pair", "occupation": 2, "connect_to": "B1"},
            ],
        }),
        AtomConfig.from_dict({
            "element": "B",
            "id": "B1",
            "xyz": [0.0, 0.0, 3.1],
            "spin": "down",
            "valence_config": [
                {"spin": "down", "orbital_type": "sigma", "occupation": 1, "connect_to": "H4"},
                {"spin": "up", "orbital_type": "sigma", "occupation": 1, "connect_to": "H5"},
                {"spin": "down", "orbital_type": "sigma", "occupation": 1, "connect_to": "H6"},
                {"spin": "none", "orbital_type": "sigma", "occupation": 0, "connect_to": "N1"},
            ],
        }),
        AtomConfig(element="H", id="H1", xyz=[1.0, 1.0, -0.5], spin="down"),
        AtomConfig(element="H", id="H2", xyz=[-1.0, 1.0, -0.5], spin="up"),
        AtomConfig(element="H", id="H3", xyz=[0.0, -1.4, -0.5], spin="down"),
        AtomConfig(element="H", id="H4", xyz=[1.0, 1.0, 3.6], spin="up"),
        AtomConfig(element="H", id="H5", xyz=[-1.0, 1.0, 3.6], spin="down"),
        AtomConfig(element="H", id="H6", xyz=[0.0, -1.4, 3.6], spin="up"),
    ]


def test_dative_pairing_nh3_bh3():
    """NH3->BH3 passes check 9; removing B's acceptor raises; removing B's config warns."""
    atoms = _nh3_bh3_atoms()
    deck = build_molecule(atoms, basis="SZ", net_spin=0)
    assert len(deck.geometry) == 8

    # Removing B's empty acceptor entry (or its connect_to='N1') makes deck build raise ValueError
    atoms_unmatched = _nh3_bh3_atoms()
    atoms_unmatched[1].valence_config = atoms_unmatched[1].valence_config[:3]
    with pytest.raises(ValueError, match="Unmatched dative bond"):
        build_molecule(atoms_unmatched, basis="SZ", net_spin=0)

    # Removing B's whole valence_config downgrades it to a UserWarning
    atoms_no_b_cfg = _nh3_bh3_atoms()
    atoms_no_b_cfg[1].valence_config = None
    with pytest.warns(UserWarning, match="no valence_config"):
        build_molecule(atoms_no_b_cfg, basis="SZ", net_spin=0)


# ==============================================================================
# 5. Ferromagnetic Declaration Checks (Section 5 Check 10, Section 7 & 9.5)
# ==============================================================================

def _ni_pair_atoms():
    def _ni_cfg(partner_id: str):
        return [
            {"spin": "up", "orbital_type": "sigma", "occupation": 1, "connect_to": partner_id},
            {"spin": "up", "orbital_type": "sigma", "occupation": 1},
            {"spin": " none".strip(), "orbital_type": "lone_pair", "occupation": 2},
            {"spin": "none", "orbital_type": "lone_pair", "occupation": 2},
            {"spin": "none", "orbital_type": "lone_pair", "occupation": 2},
            {"spin": "none", "orbital_type": "lone_pair", "occupation": 2},
            {"spin": "none", "orbital_type": "sigma", "occupation": 0},
            {"spin": "none", "orbital_type": "sigma", "occupation": 0},
            {"spin": "none", "orbital_type": "sigma", "occupation": 0},
        ]

    return [
        AtomConfig.from_dict({
            "element": "Ni",
            "id": "Ni1",
            "xyz": [0.0, 0.0, 0.0],
            "valence_config": _ni_cfg("Ni2"),
        }),
        AtomConfig.from_dict({
            "element": "Ni",
            "id": "Ni2",
            "xyz": [0.0, 0.0, 4.7],
            "valence_config": _ni_cfg("Ni1"),
        }),
    ]


def test_ferromagnetic_declaration_ni_pair():
    """Two Ni atoms with same-spin facing sigma orbitals require ferromagnetic_links."""
    atoms = _ni_pair_atoms()
    with pytest.raises(ValueError, match="Undeclared ferromagnetic coupling"):
        build_molecule(atoms, basis="SZ", net_spin=2)

    deck = build_molecule(
        _ni_pair_atoms(),
        basis="SZ",
        net_spin=2,
        physical_net_spin=4,
        ferromagnetic_links=[("Ni1", "Ni2")],
    )
    assert deck.ferromagnetic_links == [("Ni1", "Ni2")]
    assert deck.physical_net_spin == 4


def test_ferromagnetic_link_orbital_level_validation_amendment_2():
    """
    Amendment 2: Two atoms with opposite net orbital spin labels (e.g. C1 down, C2 up)
    whose facing orbitals carry the SAME spin ('up' toward each other) must be allowed
    and required in ferromagnetic_links, whereas declaring a link on antiparallel facing
    orbitals must raise.
    """
    # C1 has 1 'up' toward C2 and 3 'down' toward H1, H2, H3 -> net_orbital_spin = -2 (label 'down')
    # C2 has 1 'up' toward C1 and 3 'up' toward H4, H5, H6 -> net_orbital_spin = +4 (label 'up')
    atoms = [
        AtomConfig.from_dict({
            "element": "C",
            "id": "C1",
            "xyz": [0.0, 0.0, 0.0],
            "valence_config": [
                {"spin": "up", "orbital_type": "sigma", "occupation": 1, "connect_to": "C2"},
                {"spin": "down", "orbital_type": "sigma", "occupation": 1, "connect_to": "H1"},
                {"spin": "down", "orbital_type": "sigma", "occupation": 1, "connect_to": "H2"},
                {"spin": "down", "orbital_type": "sigma", "occupation": 1, "connect_to": "H3"},
            ],
        }),
        AtomConfig.from_dict({
            "element": "C",
            "id": "C2",
            "xyz": [0.0, 0.0, 2.9],
            "valence_config": [
                {"spin": "up", "orbital_type": "sigma", "occupation": 1, "connect_to": "C1"},
                {"spin": "up", "orbital_type": "sigma", "occupation": 1, "connect_to": "H4"},
                {"spin": "up", "orbital_type": "sigma", "occupation": 1, "connect_to": "H5"},
                {"spin": "up", "orbital_type": "sigma", "occupation": 1, "connect_to": "H6"},
            ],
        }),
        AtomConfig(element="H", id="H1", xyz=[1.0, 0.0, -1.0], spin="up"),
        AtomConfig(element="H", id="H2", xyz=[-1.0, 0.0, -1.0], spin="up"),
        AtomConfig(element="H", id="H3", xyz=[0.0, 1.0, -1.0], spin="up"),
        AtomConfig(element="H", id="H4", xyz=[1.0, 0.0, 3.9], spin="down"),
        AtomConfig(element="H", id="H5", xyz=[-1.0, 0.0, 3.9], spin="down"),
        AtomConfig(element="H", id="H6", xyz=[0.0, 1.0, 3.9], spin="down"),
    ]
    assert atoms[0].spin == "down"
    assert atoms[1].spin == "up"

    # Without ferromagnetic_links, Check 10 raises
    with pytest.raises(ValueError, match="Undeclared ferromagnetic coupling"):
        build_molecule(atoms, basis="SZ", net_spin=0)

    # With ferromagnetic_links=[("C1", "C2")], passes even though C1.spin != C2.spin
    deck = build_molecule(atoms, basis="SZ", net_spin=0, ferromagnetic_links=[("C1", "C2")])
    assert deck.ferromagnetic_links == [("C1", "C2")]

    # Converse: declaring ferromagnetic_links on antiparallel facing orbitals raises
    h2_antiparallel = [
        AtomConfig.from_dict({
            "element": "H",
            "id": "H1",
            "xyz": [0.0, 0.0, 0.0],
            "valence_config": [{"spin": "up", "orbital_type": "sigma", "occupation": 1, "connect_to": "H2"}],
        }),
        AtomConfig.from_dict({
            "element": "H",
            "id": "H2",
            "xyz": [0.0, 0.0, 1.4],
            "valence_config": [{"spin": "down", "orbital_type": "sigma", "occupation": 1, "connect_to": "H1"}],
        }),
    ]
    with pytest.raises(ValueError, match="same-spin facing"):
        build_molecule(h2_antiparallel, basis="SZ", net_spin=0, ferromagnetic_links=[("H1", "H2")])


# ==============================================================================
# 6. Error Cases Raising ValueError (Section 9.6)
# ==============================================================================

def test_valence_orbital_and_atom_config_error_cases():
    # 1. Invalid spin: 'sideways'
    with pytest.raises(ValueError, match="spin"):
        ValenceOrbital(spin="sideways", orbital_type="sigma", occupation=1)

    # 2. Invalid orbital_type: 'empty'
    with pytest.raises(ValueError, match="orbital_type"):
        ValenceOrbital(spin="none", orbital_type="empty", occupation=0)

    # 3. lone_pair with occupation: 1
    with pytest.raises(ValueError, match="lone_pair"):
        ValenceOrbital(spin="up", orbital_type="lone_pair", occupation=1)

    # 4. occupation: 1 with spin: 'none'
    with pytest.raises(ValueError, match="occupation=1"):
        ValenceOrbital(spin="none", orbital_type="sigma", occupation=1)

    # 5. occupation: 0 with spin: 'up'
    with pytest.raises(ValueError, match="occupation=0"):
        ValenceOrbital(spin="up", orbital_type="sigma", occupation=0)

    # 6. occupation: 2 with spin: 'down'
    with pytest.raises(ValueError, match="occupation=2"):
        ValenceOrbital(spin="down", orbital_type="sigma", occupation=2)

    # 7. out_of_plane with no plane or 1-label plane
    with pytest.raises(ValueError, match="out_of_plane"):
        ValenceOrbital(spin="none", orbital_type="pi", occupation=0, orientation="out_of_plane")
    with pytest.raises(ValueError, match="out_of_plane"):
        ValenceOrbital(spin="none", orbital_type="pi", occupation=0, plane=["C1"], orientation="out_of_plane")

    # 8. List longer than n_valence_orbitals (e.g. 5 orbitals on O, which has 4)
    with pytest.raises(ValueError, match="exceeding n_valence_orbitals"):
        AtomConfig(
            element="O",
            id="O1",
            valence_config=[
                ValenceOrbital(spin="up", orbital_type="sigma", occupation=1),
                ValenceOrbital(spin="down", orbital_type="sigma", occupation=1),
                ValenceOrbital(spin="none", orbital_type="lone_pair", occupation=2),
                ValenceOrbital(spin="none", orbital_type="lone_pair", occupation=2),
                ValenceOrbital(spin="none", orbital_type="sigma", occupation=0),
            ],
        )

    # 9. Electron count mismatch (O with 5 counted valence electrons instead of 6)
    with pytest.raises(ValueError, match="occupied_valency"):
        AtomConfig(
            element="O",
            id="O1",
            valence_config=[
                ValenceOrbital(spin="up", orbital_type="sigma", occupation=1),
                ValenceOrbital(spin="none", orbital_type="lone_pair", occupation=2),
                ValenceOrbital(spin="none", orbital_type="lone_pair", occupation=2),
            ],
        )

    # 10. Non-integer formal charge when valence_config is present
    with pytest.raises(ValueError, match="integer-valued"):
        AtomConfig(
            element="O",
            id="O1",
            charge=0.5,
            valence_config=[
                ValenceOrbital(spin="up", orbital_type="sigma", occupation=1),
                ValenceOrbital(spin="down", orbital_type="sigma", occupation=1),
                ValenceOrbital(spin="none", orbital_type="lone_pair", occupation=2),
                ValenceOrbital(spin="none", orbital_type="lone_pair", occupation=2),
            ],
        )

    # 11. spin label contradicting a nonzero orbital-spin sum
    with pytest.raises(ValueError, match="contradicts"):
        AtomConfig(
            element="N",
            id="N1",
            spin="down",
            valence_config=[
                ValenceOrbital(spin="up", orbital_type="sigma", occupation=1),
                ValenceOrbital(spin="up", orbital_type="pi", occupation=1),
                ValenceOrbital(spin="up", orbital_type="pi", occupation=1),
                ValenceOrbital(spin="none", orbital_type="lone_pair", occupation=2),
            ],
        )

    # 12. connect_to naming self or nonexistent atom
    with pytest.raises(ValueError, match="itself"):
        AtomConfig(
            element="H",
            id="H1",
            valence_config=[ValenceOrbital(spin="up", orbital_type="sigma", occupation=1, connect_to="H1")],
        )

    with pytest.raises(ValueError, match="does not match any atom id"):
        build_molecule(
            [
                AtomConfig(
                    element="H",
                    id="H1",
                    xyz=[0.0, 0.0, 0.0],
                    valence_config=[ValenceOrbital(spin="up", orbital_type="sigma", occupation=1, connect_to="H99")],
                ),
                AtomConfig(element="H", id="H2", xyz=[0.0, 0.0, 1.4], spin="down"),
            ],
            basis="SZ",
            net_spin=0,
        )


def test_check_7_starting_orbitals_reconciliation_across_basis_sets():
    """Verify Check 7 agrees on matching starting_orbitals (including Fe SZ/DZ/ADF-TZP) and raises on mismatch."""
    # 1. Oxygen in SZ, DZ, DZP, ADF-TZP via build_atom_orbitals
    for basis in ("SZ", "DZ", "DZP", "ADF-TZP"):
        ed = get_element_basis("O", basis)
        start_orbs = build_atom_orbitals(ed, "up", hybridization="sp3_bent")
        atom = AtomConfig(
            element="O",
            id="O1",
            spin="up",
            starting_orbitals=start_orbs,
            valence_config=[
                ValenceOrbital(spin="up", orbital_type="sigma", occupation=1),
                ValenceOrbital(spin="up", orbital_type="sigma", occupation=1),
                ValenceOrbital(spin="none", orbital_type="lone_pair", occupation=2),
                ValenceOrbital(spin="none", orbital_type="lone_pair", occupation=2),
            ],
        )
        assert atom.spin == "up"

        # Conflict: valence_config has mixed spins (1 up, 1 down), while start_orbs has 2 up
        with pytest.raises(ValueError, match="starting_orbitals"):
            AtomConfig(
                element="O",
                id="O1",
                starting_orbitals=start_orbs,
                valence_config=[
                    ValenceOrbital(spin="up", orbital_type="sigma", occupation=1),
                    ValenceOrbital(spin="down", orbital_type="sigma", occupation=1),
                    ValenceOrbital(spin="none", orbital_type="lone_pair", occupation=2),
                    ValenceOrbital(spin="none", orbital_type="lone_pair", occupation=2),
                ],
            )

    # 2. Neutral Fe ground state in SZ, DZ, ADF-TZP (4s2 3d6: 2 doubly occupied + 4 singly occupied 'up')
    for basis in ("SZ", "DZ", "ADF-TZP"):
        ed_fe = get_element_basis("Fe", basis)
        fe_vc = [
            ValenceOrbital(spin="up", orbital_type="sigma", occupation=1),
            ValenceOrbital(spin="up", orbital_type="sigma", occupation=1),
            ValenceOrbital(spin="up", orbital_type="pi", occupation=1),
            ValenceOrbital(spin="up", orbital_type="pi", occupation=1),
            ValenceOrbital(spin="none", orbital_type="lone_pair", occupation=2),
            ValenceOrbital(spin="none", orbital_type="lone_pair", occupation=2),
        ]
        fe_atom = AtomConfig(
            element="Fe",
            id="Fe1",
            starting_orbitals=ed_fe.default_orbitals,
            valence_config=fe_vc,
        )
        assert fe_atom.spin == "up"


# ==============================================================================
# 7. Backward Compatibility & Serialization Round-Trip (Section 9.7)
# ==============================================================================

def test_atom_config_and_deck_yaml_round_trip():
    """AtomConfig and CalculationDeck round-trip through to_dict and YAML with valence_config present and absent."""
    # Without valence_config
    a_plain = AtomConfig(element="N", id="N1", xyz=[0.0, 0.0, 1.0], spin="up")
    d_plain = a_plain.to_dict()
    assert "valence_config" not in d_plain
    a_plain_rt = AtomConfig.from_dict(d_plain)
    assert a_plain_rt.valence_config is None
    assert a_plain_rt.spin == "up"

    # With valence_config
    a_vc = AtomConfig(
        element="O",
        id="O1",
        xyz=[0.0, 0.0, 0.0],
        spin="up",
        valence_config=[
            ValenceOrbital(spin="up", orbital_type="sigma", occupation=1, connect_to="C1"),
            ValenceOrbital(spin="down", orbital_type="sigma", occupation=1, connect_to="C2"),
            ValenceOrbital(spin="none", orbital_type="lone_pair", occupation=2, plane=["C1", "C2"], orientation="in_plane"),
            ValenceOrbital(spin="none", orbital_type="lone_pair", occupation=2, plane=["C1", "C2"], orientation="out_of_plane"),
        ],
    )
    yaml_text = yaml.dump(a_vc.to_dict(), sort_keys=False)
    a_vc_rt = AtomConfig.from_dict(yaml.safe_load(yaml_text))
    assert a_vc_rt.valence_config == a_vc.valence_config

    # Partial config physical_net_spin returns None (Amendment 4)
    deck_partial = build_molecule(
        [
            a_vc_rt,
            AtomConfig(element="C", id="C1", xyz=[1.2, 0.0, 1.0], spin="down"),
            AtomConfig(element="C", id="C2", xyz=[-1.2, 0.0, 1.0], spin="down"),
        ],
        basis="SZ",
        net_spin=-1,
    )
    assert deck_partial.physical_net_spin is None


# ==============================================================================
# 8. Rehybridization: BF3 + NH3 -> H3N->BF3 (Section 7 & 9.8)
# ==============================================================================

def test_bf3_rehybridization_free_and_adduct():
    """Free BF3 (sp2, empty pi) and H3N->BF3 adduct (sp3, empty sigma) validate; free BF3 labeled sp3 fails Check 5."""
    free_bf3_b = {
        "element": "B",
        "id": "B1",
        "hybridization": "sp2",
        "valence_config": [
            {"spin": "up", "orbital_type": "sigma", "occupation": 1, "connect_to": "F1"},
            {"spin": "down", "orbital_type": "sigma", "occupation": 1, "connect_to": "F2"},
            {"spin": "up", "orbital_type": "sigma", "occupation": 1, "connect_to": "F3"},
            {"spin": "none", "orbital_type": "pi", "occupation": 0, "plane": ["F1", "F2", "F3"], "orientation": "out_of_plane"},
        ],
    }
    b_free = AtomConfig.from_dict(free_bf3_b)
    assert b_free.hybridization == "sp2"
    assert b_free.spin == "up"

    # Free BF3 labeled sp3 fails Check 5 (has 3 sigma + 1 pi, whereas sp3 requires 4 sigma/lone_pair and 0 pi)
    bad_bf3 = dict(free_bf3_b, hybridization="sp3")
    with pytest.raises(ValueError, match="hybridization='sp3'"):
        AtomConfig.from_dict(bad_bf3)

    # Adduct H3N->BF3: B rehybridizes to sp3 (vacant orbital is now sigma, occupation=0, connect_to=N1)
    adduct_yaml = """
    title: H3N->BF3 Adduct
    net_spin: 0
    geometry:
      - element: B
        id: B1
        xyz: [0.0, 0.0, 0.0]
        hybridization: sp3
        spin: up
        valence_config:
          - {spin: up, orbital_type: sigma, occupation: 1, connect_to: F1}
          - {spin: down, orbital_type: sigma, occupation: 1, connect_to: F2}
          - {spin: up, orbital_type: sigma, occupation: 1, connect_to: F3}
          - {spin: none, orbital_type: sigma, occupation: 0, connect_to: N1}
      - element: N
        id: N1
        xyz: [0.0, 0.0, 3.0]
        hybridization: sp3
        spin: down
        valence_config:
          - {spin: down, orbital_type: sigma, occupation: 1, connect_to: H1}
          - {spin: up, orbital_type: sigma, occupation: 1, connect_to: H2}
          - {spin: down, orbital_type: sigma, occupation: 1, connect_to: H3}
          - {spin: none, orbital_type: lone_pair, occupation: 2, connect_to: B1}
      - {element: F, id: F1, xyz: [2.4, 0.0, -0.5], spin: down}
      - {element: F, id: F2, xyz: [-1.2, 2.1, -0.5], spin: up}
      - {element: F, id: F3, xyz: [-1.2, -2.1, -0.5], spin: down}
      - {element: H, id: H1, xyz: [1.8, 0.0, 3.6], spin: up}
      - {element: H, id: H2, xyz: [-0.9, 1.5, 3.6], spin: down}
      - {element: H, id: H3, xyz: [-0.9, -1.5, 3.6], spin: up}
    """
    adduct_deck = CalculationDeck.from_yaml(adduct_yaml)
    assert adduct_deck.geometry[0].hybridization == "sp3"
    assert adduct_deck.geometry[1].hybridization == "sp3"


# ==============================================================================
# 9. Automation: propose_valence_config() (Section 8 & 9.9)
# ==============================================================================

def test_propose_valence_config_automation():
    """
    propose_valence_config() returns an explicit list per atom; editing an entry and rebuilding
    uses the edited value; explicit configs override automated spin assignment; automation never
    creates a dative entry unprompted.
    """
    # Water geometry
    water_geom = [
        AtomConfig(element="O", id="O1", xyz=[0.0, 0.0, 0.0], spin="up"),
        AtomConfig(element="H", id="H1", xyz=[0.0, 1.43, 1.10], spin="down"),
        AtomConfig(element="H", id="H2", xyz=[0.0, -1.43, 1.10], spin="down"),
    ]
    proposals = propose_valence_config(water_geom)
    assert set(proposals.keys()) == {"O1", "H1", "H2"}
    assert len(proposals["O1"]) == 4
    assert len(proposals["H1"]) == 1
    assert len(proposals["H2"]) == 1

    # Automation never creates a dative lone_pair (connect_to is always None on lone_pair)
    for aid, vc_list in proposals.items():
        for orb in vc_list:
            if orb.orbital_type == "lone_pair":
                assert orb.connect_to is None

    # Editing an entry on O1 (making one sigma 'up' and one 'down') and rebuilding uses the edited value
    edited_o = list(proposals["O1"])
    edited_o[0] = ValenceOrbital(spin="up", orbital_type="sigma", occupation=1, connect_to="H1")
    edited_o[1] = ValenceOrbital(spin="down", orbital_type="sigma", occupation=1, connect_to="H2")
    water_geom[0].valence_config = edited_o
    water_geom[1].valence_config = [ValenceOrbital(spin="down", orbital_type="sigma", occupation=1, connect_to="O1")]
    water_geom[2].valence_config = [ValenceOrbital(spin="up", orbital_type="sigma", occupation=1, connect_to="O1")]
    water_geom[2].spin = "up"

    deck = build_molecule(water_geom, basis="SZ", net_spin=1, physical_net_spin=0)
    assert deck.geometry[0].valence_config[1].spin == "down"
    assert deck.physical_net_spin == 0

    # Re-running propose_valence_config on deck with explicit config preserves the edited config
    proposals_after = propose_valence_config(deck)
    assert proposals_after["O1"] == edited_o


def test_valence_config_hybrid_seeding_mixed_spin_and_dative():
    """
    Verify build_atom_orbitals_from_valence_config and apply_hybrid_seeds construct
    oriented hybrid starting_orbitals for mixed-spin EO oxygen, planar BF3, and dative H3N->BF3.
    """
    from phatpsy.hybrids import apply_hybrid_seeds, build_atom_orbitals_from_valence_config

    # 1. Ethylene oxide oxygen with mixed-spin sigma bonds and in_plane / out_of_plane lone pairs
    eo_o = AtomConfig(
        element="O",
        id="O1",
        xyz=[0.0, 0.0, 1.6],
        spin="up",
        valence_config=[
            ValenceOrbital(spin="up", orbital_type="sigma", occupation=1, connect_to="C1"),
            ValenceOrbital(spin="down", orbital_type="sigma", occupation=1, connect_to="C2"),
            ValenceOrbital(spin="none", orbital_type="lone_pair", occupation=2, plane=["C1", "C2"], orientation="in_plane"),
            ValenceOrbital(spin="none", orbital_type="lone_pair", occupation=2, plane=["C1", "C2"], orientation="out_of_plane"),
        ],
    )
    eo_c1 = AtomConfig(element="C", id="C1", xyz=[-1.38, 0.0, 0.0], spin="down")
    eo_c2 = AtomConfig(element="C", id="C2", xyz=[1.38, 0.0, 0.0], spin="up")
    ed_o = get_element_basis("O", "DZ")
    orbs_o = build_atom_orbitals_from_valence_config(
        ed_o,
        eo_o,
        atoms_by_id={"O1": eo_o, "C1": eo_c1, "C2": eo_c2},
    )
    val_orbs_o = orbs_o[1:]
    assert [o.occno for o in val_orbs_o] == [
        [1.0, 0.0],  # sigma to C1 (up)
        [0.0, 1.0],  # sigma to C2 (down)
        [1.0, 1.0],  # in_plane lone pair
        [1.0, 1.0],  # out_of_plane lone pair (pure py along normal to xz plane)
    ]
    # Out-of-plane lone pair is pure p_y (ml=-1 in engine convention) with no s component
    oop_mls = {c[1] for c in val_orbs_o[3].coefficients}
    assert oop_mls == {-1}

    # 2. Free planar BF3 vs H3N->BF3 adduct via apply_hybrid_seeds
    bf3_deck = build_molecule(
        [
            AtomConfig(
                element="B",
                id="B1",
                xyz=[0.0, 0.0, 0.0],
                hybridization="sp2",
                spin="up",
                valence_config=[
                    ValenceOrbital(spin="up", orbital_type="sigma", occupation=1, connect_to="F1"),
                    ValenceOrbital(spin="up", orbital_type="sigma", occupation=1, connect_to="F2"),
                    ValenceOrbital(spin="up", orbital_type="sigma", occupation=1, connect_to="F3"),
                    ValenceOrbital(spin="none", orbital_type="pi", occupation=0, plane=["F1", "F2", "F3"], orientation="out_of_plane"),
                ],
            ),
            AtomConfig(element="F", id="F1", xyz=[2.46, 0.0, 0.0], spin="down"),
            AtomConfig(element="F", id="F2", xyz=[-1.23, 2.1304, 0.0], spin="down"),
            AtomConfig(element="F", id="F3", xyz=[-1.23, -2.1304, 0.0], spin="down"),
        ],
        basis="DZ",
        net_spin=-2,
    )
    apply_hybrid_seeds(bf3_deck)
    b_val = bf3_deck.geometry[0].starting_orbitals[1:]
    assert [o.occno for o in b_val] == [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
    # 4th orbital on planar B1 is pure p_z (ml=0, l=1 only -> 2 STO functions in DZ)
    assert len(b_val[3].coefficients) == 2

    # 3. Multiple-bond inference in propose_valence_config for N2, C2H4, NO
    n2_prop = propose_valence_config(
        [
            AtomConfig(element="N", id="N1", xyz=[0.0, 0.0, 0.0], spin="up"),
            AtomConfig(element="N", id="N2", xyz=[0.0, 0.0, 2.0744], spin="down"),
        ]
    )
    assert orbital_counts(n2_prop["N1"]) == {"sigma": 1, "pi": 2, "delta": 0, "lone_pair": 1}
    assert orbital_counts(n2_prop["N2"]) == {"sigma": 1, "pi": 2, "delta": 0, "lone_pair": 1}

    no_prop = propose_valence_config(
        [
            AtomConfig(element="N", id="N1", xyz=[0.0, 0.0, 0.0], spin="up"),
            AtomConfig(element="O", id="O1", xyz=[0.0, 0.0, 2.1747], spin="down"),
        ]
    )
    assert orbital_counts(no_prop["N1"]) == {"sigma": 1, "pi": 2, "delta": 0, "lone_pair": 1}
    assert orbital_counts(no_prop["O1"]) == {"sigma": 1, "pi": 1, "delta": 0, "lone_pair": 2}

    # 4. Ethylene oxide (oxirane) automated proposal assigns mixed-spin sp3 on O1 with zero ferromagnetic links
    geom_eo = [
        ("C", [1.39, 0.00, -0.73]),
        ("C", [-1.39, 0.00, -0.73]),
        ("O", [0.00, 0.00, 1.56]),
        ("H", [2.30, 1.72, -1.18]),
        ("H", [2.30, -1.72, -1.18]),
        ("H", [-2.30, 1.72, -1.18]),
        ("H", [-2.30, -1.72, -1.18]),
    ]
    eo_prop = propose_valence_config(geom_eo)
    assert valence_net_spin(eo_prop["C1"]) == 4
    assert valence_net_spin(eo_prop["C2"]) == -4
    assert valence_net_spin(eo_prop["O1"]) == 0
    o1_sigmas = {vo.connect_to: vo.spin for vo in eo_prop["O1"] if vo.orbital_type == "sigma"}
    assert o1_sigmas == {"C1": "down", "C2": "up"}
    eo_deck_auto = build_molecule(
        geom_eo,
        spins=["up", "down", "up", "down", "down", "up", "up"],
        hybridization=["sp3", "sp3", "sp3", None, None, None, None],
        valence_configs=[eo_prop[k] for k in ("C1", "C2", "O1", "H1", "H2", "H3", "H4")],
        net_spin=1,
        physical_net_spin=0,
    )
    assert eo_deck_auto.ferromagnetic_links == []
    assert eo_deck_auto.physical_net_spin == 0

    # 5. Bulk Ni13 embedded cluster: all 13 Ni atoms have 2 unpaired spin-up electrons
    # ferromagnetically coupled to their equivalent neighbors (Ni2..Ni13 equivalent_to Ni1)
    from pathlib import Path
    import numpy as np
    repo_root = Path(__file__).resolve().parent.parent
    deck_ni13 = CalculationDeck.from_yaml(repo_root / "examples" / "ni13.yaml")
    deck_ni13.validate()
    assert len(deck_ni13.geometry) == 13
    assert deck_ni13.net_spin == 13
    assert deck_ni13.physical_net_spin == 2
    assert deck_ni13.geometry[0].id == "Ni1"
    assert deck_ni13.geometry[0].equivalent_to is None
    assert deck_ni13.geometry[0].virtual is False
    assert deck_ni13.geometry[0].cutoff == 5.0
    ni1_xyz = np.array(deck_ni13.geometry[0].xyz, dtype=float)
    for at in deck_ni13.geometry:
        assert at.spin == "up"
        assert at.net_orbital_spin == 2
        assert at.valence_config is not None
        assert sum(o.occupation for o in at.valence_config) == 10
        assert sum(1 for o in at.valence_config if o.occupation == 1 and o.spin == "up") == 2
    for at in deck_ni13.geometry[1:]:
        assert at.equivalent_to == "Ni1"
        assert at.virtual is True
        dist = float(np.linalg.norm(np.array(at.xyz, dtype=float) - ni1_xyz))
        assert abs(dist - 4.71) < 1e-2
    stdin_ni13 = deck_ni13.to_stdin_text()
    assert "MDIFAT=1" in stdin_ni13
    assert "NEQAT=12" in stdin_ni13
    assert "RMAX=5.00000" in stdin_ni13
    assert "QFLIP=T" not in stdin_ni13

    # 6. Clean Ni(100) 3-layer surface slab: 2 active atoms (Ni_surf_1, Ni_sub_1) with
    # cutoff=5.0 Bohr, 10 virtual equivalent neighbors, and 4 fixed bulk atoms from ni_bulk_embedded.json
    deck_slab = CalculationDeck.from_yaml(repo_root / "examples" / "ni_100_slab.yaml")
    deck_slab.validate()
    assert len(deck_slab.geometry) == 16
    assert deck_slab.net_spin == 16
    assert deck_slab.physical_net_spin == 4
    atoms_by_id = {at.id: at for at in deck_slab.geometry}
    surf1 = atoms_by_id["Ni_surf_1"]
    sub1 = atoms_by_id["Ni_sub_1"]
    assert surf1.virtual is False and surf1.fixed is False and surf1.cutoff == 5.0
    assert sub1.virtual is False and sub1.fixed is False and sub1.cutoff == 5.0
    surf1_xyz = np.array(surf1.xyz, dtype=float)
    sub1_xyz = np.array(sub1.xyz, dtype=float)
    surf1_nns = [
        at.id for at in deck_slab.geometry
        if at.id != "Ni_surf_1" and float(np.linalg.norm(np.array(at.xyz, dtype=float) - surf1_xyz)) <= 5.0
    ]
    sub1_nns = [
        at.id for at in deck_slab.geometry
        if at.id != "Ni_sub_1" and float(np.linalg.norm(np.array(at.xyz, dtype=float) - sub1_xyz)) <= 5.0
    ]
    assert len(surf1_nns) == 8
    assert len(sub1_nns) == 12
    stdin_slab = deck_slab.to_stdin_text()
    assert "MDIFAT=3" in stdin_slab
    assert "RMAX=5.00000" in stdin_slab

    # 7. CO dative-pi donor construction and CO/Ni(100) chemisorption deck
    deck_co = CalculationDeck.from_yaml(repo_root / "examples" / "CO.yaml")
    deck_co.validate()
    apply_hybrid_seeds(deck_co)
    c1_co, o1_co = deck_co.geometry[0], deck_co.geometry[1]
    assert c1_co.charge == 0.0 and o1_co.charge == 0.0
    # Verify O1's dative-pi lone_pair(connect_to='C1') builds an orthogonal ml=+1 pi orbital
    # rather than a duplicate ml=0 sigma hybrid
    o1_orbs = o1_co.starting_orbitals[1:]  # 4 valence orbitals
    o1_mls = [{c[1] for c in orb.coefficients} for orb in o1_orbs]
    assert {0} in o1_mls
    assert {-1} in o1_mls
    assert {1} in o1_mls

    deck_coni = CalculationDeck.from_yaml(repo_root / "examples" / "CO_ni_100.yaml")
    deck_coni.validate()
    assert len(deck_coni.geometry) == 11
    assert deck_coni.net_spin == 9
    coni_by_id = {at.id: at for at in deck_coni.geometry}
    assert coni_by_id["C1"].virtual is False and coni_by_id["C1"].fixed is False
    assert coni_by_id["O1"].virtual is False and coni_by_id["O1"].fixed is False
    assert coni_by_id["Ni1"].virtual is False and coni_by_id["Ni1"].fixed is False
    assert coni_by_id["Ni1"].cutoff == 5.0
    assert coni_by_id["C1"].embedded_state is not None
    assert coni_by_id["O1"].embedded_state is not None
    assert coni_by_id["Ni1"].embedded_state is not None
    for s_id in ("Ni2", "Ni3", "Ni4", "Ni5", "Ni6", "Ni7", "Ni8", "Ni9"):
        assert coni_by_id[s_id].fixed is True
        assert coni_by_id[s_id].virtual is True
    stdin_coni = deck_coni.to_stdin_text()
    assert "MDIFAT=3" in stdin_coni
    assert "RMAX=5.00000" in stdin_coni


def test_co_dative_pi_fractional_occupations_and_dz_parity():
    """
    Dedicated verification of the CO terminal sp dative-pi fractional occupation
    sharing in build_atom_orbitals_from_valence_config (phatpsy/hybrids.py:1298-1337):
    1. Group 14 acceptor (C, 4 valence e-) places its outward sigma lone-pair spin
       opposite its sigma_in/pi_y bonding spin ([0.0, 1.0] vs [1.0, 0.0]), and shares
       [1.0, 0.0] on both pi_x and pi_y (conserving 4.0 valence electrons).
    2. Group 16 donor (O, 6 valence e-) shares 0.5 electron from its dative pi_x
       donor across pi_x and pi_y ([0.5, 1.0] each), with [0.0, 1.0] on sigma_in
       and [1.0, 1.0] on its outward sigma lone pair (conserving 6.0 valence electrons),
       matching build_atom_orbitals(hybridization='sp') in both DZ and DZP.
    3. Verified DZ CO SCF convergence, total energy, and virial ratio are unchanged
       between valence_config and hybridization='sp' seeding.
    """
    from phatpsy.builder import build_diatomic
    from phatpsy.driver import PhatpsyDriver
    from phatpsy.schema import ValenceOrbital

    c_val = [
        ValenceOrbital(spin="up",   occupation=1, orbital_type="sigma",     connect_to="O2"),
        ValenceOrbital(spin="up",   occupation=1, orbital_type="pi",        connect_to="O2"),
        ValenceOrbital(spin="none", occupation=0, orbital_type="pi",        connect_to="O2"),
        ValenceOrbital(spin="none", occupation=2, orbital_type="lone_pair"),
    ]
    o_val = [
        ValenceOrbital(spin="down", occupation=1, orbital_type="sigma",     connect_to="C1"),
        ValenceOrbital(spin="down", occupation=1, orbital_type="pi",        connect_to="C1"),
        ValenceOrbital(spin="none", occupation=2, orbital_type="lone_pair", connect_to="C1"),
        ValenceOrbital(spin="none", occupation=2, orbital_type="lone_pair"),
    ]

    for basis_name in ("DZ", "DZP"):
        deck_vc = build_diatomic(
            "C", "O",
            bond_length=2.132,
            basis=basis_name,
            hybridization="sp",
            spins=["up", "down"],
            valence_configs=[c_val, o_val],
            net_spin=0,
            physical_net_spin=0,
            units="bohr",
        )
        deck_hyb = build_diatomic(
            "C", "O",
            bond_length=2.132,
            basis=basis_name,
            hybridization="sp",
            spins=["up", "down"],
            net_spin=0,
            units="bohr",
            seed_hybrids=True,
        )

        c_orbs_vc = deck_vc.geometry[0].starting_orbitals[1:5]
        o_orbs_vc = deck_vc.geometry[1].starting_orbitals[1:5]

        # Total valence electron count conserved on C (4.0) and O (6.0)
        assert sum(sum(o.occno) for o in c_orbs_vc) == pytest.approx(4.0)
        assert sum(sum(o.occno) for o in o_orbs_vc) == pytest.approx(6.0)

        # C valence occupations: three [1.0, 0.0] (sigma_in, pi_x, pi_y) and one [0.0, 1.0] (sigma_out LP)
        c_occs = sorted([list(o.occno) for o in c_orbs_vc])
        assert c_occs == [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]

        # O valence occupations: one [0.0, 1.0] (sigma_in), two [0.5, 1.0] (pi_x, pi_y), one [1.0, 1.0] (sigma_out LP)
        o_occs = sorted([list(o.occno) for o in o_orbs_vc])
        assert o_occs == [[0.0, 1.0], [0.5, 1.0], [0.5, 1.0], [1.0, 1.0]]

        # Matches build_atom_orbitals(hybridization="sp") occupation multiset on both atoms
        c_orbs_hyb = deck_hyb.geometry[0].starting_orbitals[1:5]
        o_orbs_hyb = deck_hyb.geometry[1].starting_orbitals[1:5]
        assert sorted([list(o.occno) for o in c_orbs_hyb]) == c_occs
        assert sorted([list(o.occno) for o in o_orbs_hyb]) == o_occs

        if basis_name == "DZ":
            driver = PhatpsyDriver(quiet=True)
            res_vc = driver.run_deck(deck_vc)
            res_hyb = driver.run_deck(deck_hyb)
            assert res_vc.converged and res_hyb.converged
            assert float(res_vc.total_energy) == pytest.approx(float(res_hyb.total_energy), abs=1e-6)
            assert float(res_vc.virial_ratio) == pytest.approx(float(res_hyb.virial_ratio), abs=1e-5)





