# tests/test_atom_equivalency.py
import pytest
import numpy as np

from phatpsy import (
    CalculationDeck,
    AtomConfig,
    SymmetrySpec,
    build_molecule,
    build_diatomic,
)
from phatpsy.schema import (
    normalize_symmetry_op,
    normalize_symmetry_axis,
    apply_symmetry_operation,
)


def test_atom_config_intrinsic_properties():
    # Primary atom
    a1 = AtomConfig(id="Ni1", element="Ni", xyz=[0.0, 0.0, 1.5], spin="up", fixed=False, virtual=False)
    assert a1.id == "Ni1"
    assert a1.element == "Ni"
    assert a1.is_spin_up is True
    assert a1.is_spin_down is False
    assert a1.is_virtual is False
    assert a1.is_fixed is False
    assert a1.is_equivalent is False

    # Equivalent atom with intrinsic spin="down" and virtual=True
    a2 = AtomConfig(
        id="Ni2",
        element="Ni",
        xyz=[0.0, 0.0, -1.5],
        equivalent_to="Ni1",
        symmetry_op="reflection",
        symmetry_axis="z",
        spin="down",
        virtual=True,
    )
    assert a2.id == "Ni2"
    assert a2.equivalent_to == "Ni1"
    assert a2.is_equivalent is True
    assert a2.is_spin_down is True
    assert a2.is_virtual is True
    assert a2.symmetry_op == -1
    assert a2.symmetry_axis == 1
    assert a2.symmetry is not None
    assert a2.symmetry.isym == -1
    assert a2.symmetry.ixyz == 1


def test_symmetry_helpers():
    # Mnemonic normalizations
    assert normalize_symmetry_op("reflection") == -1
    assert normalize_symmetry_op("mirror") == -1
    assert normalize_symmetry_op("inversion") == -2
    assert normalize_symmetry_op("C4") == 4
    assert normalize_symmetry_op(2) == 2

    assert normalize_symmetry_axis("x") == 100
    assert normalize_symmetry_axis("y") == 10
    assert normalize_symmetry_axis("z") == 1

    # Transformations
    p = [1.0, 2.0, 3.0]
    # Reflection across XY plane (z -> -z)
    refl_z = apply_symmetry_operation(p, isym=-1, ixyz=1)
    assert np.allclose(refl_z, [1.0, 2.0, -3.0])

    # Inversion
    inv = apply_symmetry_operation(p, isym=-2)
    assert np.allclose(inv, [-1.0, -2.0, -3.0])

    # 4-fold rotation around Z (90 deg CCW: [1, 2, 3] -> [-2, 1, 3])
    c4_z = apply_symmetry_operation(p, isym=4, ixyz=1)
    assert np.allclose(c4_z, [-2.0, 1.0, 3.0], atol=1e-5)


def test_permuted_atom_ordering_and_deck_generation():
    """
    Ensures that when an equivalent atom is declared BEFORE its reference atom
    in the input list, group_atoms_by_equivalency correctly places the primary
    atom first and sets NEQAT appropriately in the Fortran stdin deck.
    """
    # Ni2 declared before Ni1
    geom = [
        AtomConfig(
            id="Ni2",
            element="Ni",
            equivalent_to="Ni1",
            symmetry_op=-1,
            symmetry_axis=1,
            spin="down",
        ),
        AtomConfig(
            id="Ni1",
            element="Ni",
            xyz=[0.0, 0.0, 1.5],
            spin="up",
        ),
    ]

    deck = CalculationDeck(title="Permuted Ni2 Test", geometry=geom)
    groups = deck.group_atoms_by_equivalency()

    assert "Ni" in groups
    assert len(groups["Ni"]) == 1
    grp = groups["Ni"][0]
    assert grp["primary"].id == "Ni1"
    assert len(grp["equivalent"]) == 1
    assert grp["equivalent"][0].id == "Ni2"

    stdin = deck.to_stdin_text()
    # Reference atom with NEQAT=1 must appear first
    assert "XYZ=0.00000,0.00000,1.50000, NEQAT=1" in stdin
    # Equivalent atom with ISYM=-1, IXYZ=001 and QFLIP=T (since Ni2 is down and Ni1 is up)
    assert "IXYZ=001, ISYM=-1, QFLIP=T" in stdin


def test_resolve_coordinates_for_equivalent_atoms():
    # Ni2 has [0, 0, 0] coordinates on initialization
    a1 = AtomConfig(id="Ni1", element="Ni", xyz=[1.2, 0.0, 2.5], spin="up")
    a2 = AtomConfig(
        id="Ni2",
        element="Ni",
        equivalent_to="Ni1",
        symmetry_op=-1,
        symmetry_axis=1,  # reflection through XY (z -> -z)
        spin="down",
    )
    deck = CalculationDeck(geometry=[a1, a2])
    assert a2.xyz == [0.0, 0.0, 0.0]

    deck.resolve_coordinates()
    assert np.allclose(a2.xyz, [1.2, 0.0, -2.5])


def test_yaml_roundtrip_with_atom_ids(tmp_path):
    a1 = AtomConfig(id="C1", element="C", xyz=[0.0, 0.0, 0.0], spin="up")
    a2 = AtomConfig(
        id="C2",
        element="C",
        xyz=[0.0, 0.0, 2.1],
        equivalent_to="C1",
        symmetry_op=-1,
        symmetry_axis=100,
        spin="down",
        virtual=True,
    )
    deck = CalculationDeck(title="YAML Test", geometry=[a1, a2])
    yaml_path = tmp_path / "test_atoms.yaml"
    deck.to_yaml(yaml_path)

    loaded = CalculationDeck.from_yaml(yaml_path)
    assert len(loaded.geometry) == 2
    g1 = loaded.geometry[0]
    g2 = loaded.geometry[1]

    assert g1.id == "C1"
    assert g1.spin == "up"

    assert g2.id == "C2"
    assert g2.equivalent_to == "C1"
    assert g2.symmetry_op == -1
    assert g2.symmetry_axis == 100
    assert g2.spin == "down"
    assert g2.virtual is True


def test_invalid_equivalent_reference_raises():
    deck = CalculationDeck(
        geometry=[
            AtomConfig(id="O2", element="O", equivalent_to="NonExistentAtom"),
        ]
    )
    with pytest.raises(ValueError, match="no reference atom with that ID was found"):
        deck.to_stdin_text()
