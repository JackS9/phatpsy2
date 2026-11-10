# tests/test_spin_api.py
"""
Unit and integration tests for the Explicit Spin Assignment API:
1. Hard error on None spin anywhere in the spin path.
2. Net spin conservation assertion ((n_up - n_down) == net_spin).
3. Non-bipartite pairing graph detection (frustrated rings like EO's 3-membered oxirane).
4. Heteronuclear diatomic defaults (no flip on CO unless requested).
5. Default spin list pinning for N2, CO, and H2.
6. Deprecation shims for legacy spin_flip and auto_spin_flip flags.
"""
import pytest
import warnings

from phatpsy.schema import CalculationDeck, AtomConfig, ControlConfig
from phatpsy.builder import build_diatomic, build_molecule
from phatpsy.spin import (
    NonBipartiteSpinGraphError,
    pp_vb_diatomic_spins,
    assign_alternating_spin,
    solve_alternating_spin,
    find_odd_cycle,
)


def test_none_spin_raises_at_deck_build_time():
    """Requirement 1: Every atom in a built deck carries spin 'up' or 'down'. None is a hard error."""
    # Direct CalculationDeck construction with None spin
    atoms_with_none = [
        AtomConfig(element="N", xyz=[0.0, 0.0, 1.0], spin="up"),
        AtomConfig(element="N", xyz=[0.0, 0.0, -1.0], spin=None),
    ]
    with pytest.raises(ValueError, match="forbidden"):
        CalculationDeck(geometry=atoms_with_none, net_spin=0)

    # CalculationDeck with unassigned (None) default
    atoms_unassigned = [
        AtomConfig(element="H", xyz=[0.0, 0.0, 0.0]),
        AtomConfig(element="H", xyz=[0.0, 0.0, 1.4]),
    ]
    with pytest.raises(ValueError, match="unassigned or invalid spin"):
        CalculationDeck(geometry=atoms_unassigned, net_spin=0)


def test_net_spin_mismatch_raises():
    """Requirement 2: (n_up - n_down) == net_spin; on mismatch raise, never silently adjust."""
    # n_up = 2, n_down = 0, but net_spin = 0
    atoms = [
        AtomConfig(element="C", xyz=[0.0, 0.0, 0.0], spin="up"),
        AtomConfig(element="O", xyz=[0.0, 0.0, 2.1], spin="up"),
    ]
    with pytest.raises(ValueError, match="Net spin mismatch"):
        CalculationDeck(geometry=atoms, net_spin=0)

    # build_molecule with mismatch between spins and net_spin
    with pytest.raises(ValueError, match="Net spin mismatch"):
        build_molecule(
            "C 0.0 0.0 0.0; O 0.0 0.0 2.132",
            basis="SZ",
            spins=["up", "up"],
            net_spin=0,  # Mismatch: 2 up - 0 down = 2 != 0
        )

    # Valid explicit net_spin passes cleanly
    deck = CalculationDeck(geometry=atoms, net_spin=2)
    assert deck.net_spin == 2
    assert deck.spins == ["up", "up"]


def test_frustrated_ring_raises_non_bipartite():
    """Requirement 3: Bipartite pairing check rejects odd cycles (e.g. EO 3-membered oxirane ring)."""
    # 1. Low-level BFS cycle detector on triangle graph
    adj_triangle = [[1, 2], [0, 2], [0, 1]]
    odd_cycle = find_odd_cycle(adj_triangle)
    assert odd_cycle is not None
    assert len(odd_cycle) == 3

    # Square graph (even cycle, bipartite) has no odd cycles
    adj_square = [[1, 3], [0, 2], [1, 3], [0, 2]]
    assert find_odd_cycle(adj_square) is None

    # 2. solve_alternating_spin with require_bipartite=True
    with pytest.raises(NonBipartiteSpinGraphError, match="odd cycle"):
        solve_alternating_spin(adj_triangle, require_bipartite=True)

    # 3. Explicit 3-membered Ethylene Oxide (oxirane) ring: C1 - C2 - O1
    # Bond lengths in Bohr: C-C ~ 2.78, C-O ~ 2.70
    eo_3ring_atoms = [
        AtomConfig(id="C1", element="C", xyz=[1.39, 0.0, -0.73]),
        AtomConfig(id="C2", element="C", xyz=[-1.39, 0.0, -0.73]),
        AtomConfig(id="O1", element="O", xyz=[0.0, 0.0, 1.47]),
    ]
    with pytest.raises(NonBipartiteSpinGraphError) as exc_info:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assign_alternating_spin(eo_3ring_atoms, require_bipartite=True)
    err_msg = str(exc_info.value)
    assert "odd cycle" in err_msg or "Non-bipartite" in err_msg
    # Offending ring atoms must be identified in error message
    assert "C1" in err_msg or "C2" in err_msg or "O1" in err_msg

    # 4. build_molecule with deprecated auto_spin_flip=True on frustrated ring with require_bipartite=True raises loudly
    with pytest.raises(NonBipartiteSpinGraphError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            build_molecule(
                "C 1.39 0.0 -0.73; C -1.39 0.0 -0.73; O 0.0 0.0 1.47",
                basis="SZ",
                auto_spin_flip=True,
                require_bipartite=True,
            )



def test_heteronuclear_diatomic_no_flip_unless_requested():
    """Requirement 4 & 6: Heteronuclear diatomic (CO) gets no flip unless explicitly requested."""
    # Default: closed-shell singlet (no spin flip on heteronuclear CO)
    deck_default = build_diatomic("C", "O", 2.132, basis="DZ")
    assert deck_default.spins == ["up", "up"]
    assert deck_default.net_spin == 2
    stdin_default = deck_default.to_stdin_text()
    assert "QFLIP=T" not in stdin_default

    # Explicitly requested spin flip
    deck_flipped = build_diatomic("C", "O", 2.132, basis="DZ", spins=["up", "down"], net_spin=0)
    assert deck_flipped.spins == ["up", "down"]
    assert deck_flipped.net_spin == 0


def test_builder_default_spin_lists_pinned():
    """Requirement 6: Builder's explicit default spin lists for N2, CO, and H2."""
    # N2: Homonuclear PP-VB alternating spin
    deck_n2 = build_diatomic("N", "N", 2.0744, basis="DZ")
    assert deck_n2.spins == ["up", "down"]
    assert deck_n2.net_spin == 0

    # H2: Homonuclear PP-VB alternating spin
    deck_h2 = build_diatomic("H", "H", 1.400, basis="SZ")
    assert deck_h2.spins == ["up", "down"]
    assert deck_h2.net_spin == 0

    # CO: Heteronuclear closed-shell
    deck_co = build_diatomic("C", "O", 2.132, basis="DZ")
    assert deck_co.spins == ["up", "up"]
    assert deck_co.net_spin == 2

    # Named PP-VB helper
    assert pp_vb_diatomic_spins() == ["up", "down"]


def test_deprecated_spin_flip_shims_emit_warnings():
    """Requirement 4: Deprecated spin_flip and auto_spin_flip shims emit DeprecationWarning."""
    # spin_flip=True in build_diatomic
    with pytest.deprecated_call(match="The 'spin_flip' parameter in build_diatomic is deprecated"):
        d_flip = build_diatomic("N", "N", 2.0744, basis="DZ", spin_flip=True)
        assert d_flip.spins == ["up", "down"]
        assert d_flip.net_spin == 0

    # spin_flip=False in build_diatomic
    with pytest.deprecated_call(match="The 'spin_flip' parameter in build_diatomic is deprecated"):
        d_noflip = build_diatomic("C", "O", 2.132, basis="DZ", spin_flip=False)
        assert d_noflip.spins == ["up", "up"]
        assert d_noflip.net_spin == 2

    # auto_spin_flip=True in build_molecule
    with pytest.deprecated_call(match="The 'auto_spin_flip' parameter in build_molecule is deprecated"):
        d_auto = build_molecule(
            "O 0.0 0.0 0.0; H 0.0 1.43 1.11; H 0.0 -1.43 1.11",
            basis="SZ",
            auto_spin_flip=True,
        )
        assert len(d_auto.spins) == 3


def test_symmetry_pairing_sanity_checks():
    """Requirement 3a: Verify symmetry equivalents must share element identity."""
    # Non-symmetry equivalent elements (e.g. C and O) linked with equivalent_to raise ValueError
    atoms = [
        AtomConfig(id="C1", element="C", xyz=[0.0, 0.0, 0.0], spin="up"),
        AtomConfig(id="O1", element="O", xyz=[0.0, 0.0, 2.1], equivalent_to="C1", symmetry_op=1, spin="down"),
    ]
    with pytest.raises(ValueError, match="different elements"):
        CalculationDeck(geometry=atoms, net_spin=0)


def test_frustrated_ring_detects_and_reports_ferromagnetic_link():
    """Requirement 3b: Automatic detection of odd cycles and reporting of ferromagnetic coupling."""
    eo_3ring_atoms = [
        AtomConfig(id="C1", element="C", xyz=[1.39, 0.0, -0.73]),
        AtomConfig(id="C2", element="C", xyz=[-1.39, 0.0, -0.73]),
        AtomConfig(id="O1", element="O", xyz=[0.0, 0.0, 1.47]),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        deck = build_molecule(
            eo_3ring_atoms,
            basis="SZ",
            auto_spin_flip=True,
        )

    # Odd cycle detected and ferromagnetic link identified on longest bond (C1-C2, d ~ 2.78 Bohr)
    assert deck.ferromagnetic_links == [("C1", "C2")]
    assert len(deck.ferromagnetic_reports) == 1
    rep = deck.ferromagnetic_reports[0]
    assert rep.atom_pair == ("C1", "C2")
    assert rep.ring_size == 3
    assert set(rep.ring_atoms) == {"C1", "C2", "O1"}
    assert abs(rep.distance_bohr - 2.78) < 0.01
    assert "longest bond" in rep.reason

    # C1 and C2 carry parallel spins (ferromagnetic link)
    c1 = next(a for a in deck.geometry if a.id == "C1")
    c2 = next(a for a in deck.geometry if a.id == "C2")
    o1 = next(a for a in deck.geometry if a.id == "O1")
    assert c1.spin == c2.spin == "up"
    assert o1.spin == "down"
    assert deck.net_spin == 1
    assert "Ferromagnetic Links" in deck.summary()


def test_frustrated_ring_user_specified_ferromagnetic_link():
    """Requirement 3b: User-specified placement of ferromagnetic coupling in frustrated ring."""
    eo_3ring_atoms = [
        AtomConfig(id="C1", element="C", xyz=[1.39, 0.0, -0.73]),
        AtomConfig(id="C2", element="C", xyz=[-1.39, 0.0, -0.73]),
        AtomConfig(id="O1", element="O", xyz=[0.0, 0.0, 1.47]),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        deck = build_molecule(
            eo_3ring_atoms,
            basis="SZ",
            auto_spin_flip=True,
            ferromagnetic_links=[("C1", "O1")],
        )

    # User-specified placement on (C1, O1)
    assert deck.ferromagnetic_links == [("C1", "O1")]
    assert len(deck.ferromagnetic_reports) == 1
    rep = deck.ferromagnetic_reports[0]
    assert rep.atom_pair == ("C1", "O1")
    assert "user-specified" in rep.reason

    # C1 and O1 carry parallel spins
    c1 = next(a for a in deck.geometry if a.id == "C1")
    c2 = next(a for a in deck.geometry if a.id == "C2")
    o1 = next(a for a in deck.geometry if a.id == "O1")
    assert c1.spin == o1.spin == "up"
    assert c2.spin == "down"
    assert deck.net_spin == 1


def test_ferromagnetic_links_deck_validation():
    """Requirement 3b & Deck Validation: Check validation of ferromagnetic links."""
    # Atoms with parallel spins validate cleanly with ferromagnetic_links
    atoms_parallel = [
        AtomConfig(id="C1", element="C", xyz=[1.39, 0.0, -0.73], spin="up"),
        AtomConfig(id="C2", element="C", xyz=[-1.39, 0.0, -0.73], spin="up"),
        AtomConfig(id="O1", element="O", xyz=[0.0, 0.0, 1.47], spin="down"),
    ]
    deck = CalculationDeck(geometry=atoms_parallel, net_spin=1, ferromagnetic_links=[("C1", "C2")])
    assert deck.ferromagnetic_links == [("C1", "C2")]

    # If ferromagnetic link specifies antiparallel spins, deck validation raises
    with pytest.raises(ValueError, match="antiparallel spins"):
        CalculationDeck(geometry=atoms_parallel, net_spin=1, ferromagnetic_links=[("C1", "O1")])

    # If ferromagnetic link specifies non-existent atom ID, deck validation raises
    with pytest.raises(ValueError, match="no atom with that ID"):
        CalculationDeck(geometry=atoms_parallel, net_spin=1, ferromagnetic_links=[("C1", "N99")])


def test_eo_mixed_spin_oxygen_eliminates_ferromagnetic_link():
    """
    Ethylene Oxide (EO): Oxygen has a mixed-spin sp3 configuration with one sp3 sigma orbital
    antiferromagnetically bonded to C1 and the other antiferromagnetically bonded to C2,
    so there is NO ferromagnetic bond between C1 and C2 (or anywhere in the ring).
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    deck = CalculationDeck.from_yaml(repo_root / "examples" / "EO.yaml")
    assert deck.ferromagnetic_links == []
    assert deck.ferromagnetic_reports == []
    assert deck.physical_net_spin == 0
    c1, c2, o1 = deck.geometry[0], deck.geometry[1], deck.geometry[2]
    assert c1.spin == "up" and c2.spin == "down"
    assert o1.hybridization == "sp3"
    assert o1.net_orbital_spin == 0


