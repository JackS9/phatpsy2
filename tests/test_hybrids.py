"""Invariant tests for the prototype directed-hybrid seed builder (phatpsy.hybrids)."""
import numpy as np
import pytest

from phatpsy.basis import get_element_basis
from phatpsy.builder import build_molecule
from phatpsy.hybrids import (
    ML_TO_AXIS,
    apply_hybrid_seeds,
    atom_valence_vectors,
    build_atom_orbitals,
    valence_space,
)

TETRA = [np.array(v, float) / np.sqrt(3) for v in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]


def _water_deck(basis="ADF-DZ"):
    r, h = 1.809, np.radians(104.52 / 2)
    g = [("O", [0, 0, 0]), ("H", [0, r * np.sin(h), r * np.cos(h)]), ("H", [0, -r * np.sin(h), r * np.cos(h)])]
    deck = build_molecule(g, basis=basis, spins=["up", "down", "down"], net_spin=-1)
    return deck, {(0, 1): 1, (0, 2): 1}


def test_tetrahedral_sigma_hybrids_are_orthonormal_and_aligned():
    v = atom_valence_vectors(TETRA, [], n_lone_pairs=0)
    B = np.array(v["sigma"])
    assert np.allclose(B @ B.T, np.eye(4), atol=1e-12)
    for hybrid, d in zip(B, TETRA):
        # sp3: equal s weight 1/4 and the p part points along the bond
        assert hybrid[0] ** 2 == pytest.approx(0.25, abs=1e-9)
        assert np.allclose(hybrid[1:] / np.linalg.norm(hybrid[1:]), d, atol=1e-9)


def test_water_valence_vectors_span_the_valence_space_orthonormally():
    r, h = 1.809, np.radians(104.52 / 2)
    dirs = [np.array([0, np.sin(h), np.cos(h)]), np.array([0, -np.sin(h), np.cos(h)])]
    v = atom_valence_vectors(dirs, [], n_lone_pairs=2)
    allv = np.array(v["sigma"] + v["lone"])
    assert allv.shape == (4, 4)
    assert np.allclose(allv @ allv.T, np.eye(4), atol=1e-12)
    # bonding hybrids lean toward their bond (cosine with the bond direction is large)
    for hyb, d in zip(v["sigma"], dirs):
        assert hyb[1:] @ d / np.linalg.norm(hyb[1:]) > 0.99


def test_pi_orbital_is_perpendicular_to_the_sigma_plane():
    dirs = [np.array([1.0, 0, 0]), np.array([-0.5, np.sqrt(3) / 2, 0]), np.array([-0.5, -np.sqrt(3) / 2, 0])]
    v = atom_valence_vectors(dirs[:2], [np.array([0, 0, 1.0])], n_lone_pairs=0, n_extra=0)
    assert np.allclose(v["pi"][0], [0, 0, 0, 1], atol=1e-12)
    assert len(v["empty"]) == 1


@pytest.mark.parametrize("basis", ["DZ", "ADF-DZ", "ADF-TZ2P"])
def test_bond_pattern_electron_counts_and_spin(basis):
    elem = get_element_basis("O", basis)
    vs = valence_space(elem)
    assert vs.valence_electrons == 6
    d = [np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
    for spin, occ_bond in (("up", [1.0, 0.0]), ("down", [0.0, 1.0])):
        orbs = build_atom_orbitals(elem, spin, d, [])
        total = sum(sum(o.occno) for o in orbs)
        assert total == pytest.approx(8.0)  # neutral oxygen
        bonding = [o for o in orbs if o.occno == occ_bond]
        assert len(bonding) == 2
        assert sum(1 for o in orbs if o.occno == [1.0, 1.0]) == 3  # 1s core + 2 lone pairs


def test_hydrogen_gets_only_the_bond_electron_spin():
    elem = get_element_basis("H", "ADF-DZ")
    up = build_atom_orbitals(elem, "up", [np.array([0, 0, 1.0])], [])
    down = build_atom_orbitals(elem, "down", [np.array([0, 0, 1.0])], [])
    assert sum(o.occno[0] for o in up) == 1.0 and sum(o.occno[1] for o in up) == 0.0
    assert sum(o.occno[0] for o in down) == 0.0 and sum(o.occno[1] for o in down) == 1.0


def test_over_filled_valence_shell_is_rejected():
    elem = get_element_basis("O", "ADF-DZ")
    with pytest.raises(ValueError):
        build_atom_orbitals(elem, "up", [np.array(v, float) for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1))], [np.array([1.0, 0, 0])])


def test_apply_to_water_deck_sets_orbitals_and_deck_text_is_consistent():
    deck, bonds = _water_deck()
    apply_hybrid_seeds(deck, bonds)
    assert all(a.starting_orbitals for a in deck.geometry)
    text = deck.to_stdin_text()
    # custom orbitals are never combined with QFLIP (the engine skips &ATORB for a flipped atom)
    assert "QFLIP=T" not in text
    # total electrons implied by the seed occupations equals the nuclear charge sum
    total = sum(sum(o.occno) for a in deck.geometry for o in a.starting_orbitals)
    assert total == pytest.approx(10.0)


def test_engine_p_convention_mapping_is_the_documented_one():
    assert ML_TO_AXIS == {-1: 1, 0: 2, 1: 0}


def test_hybridization_registry_and_aliases():
    from phatpsy.hybrids import HYBRIDIZATION_REGISTRY, HybridizationType, get_hybridization_spec

    # Check key types exist in registry
    assert "sp3" in HYBRIDIZATION_REGISTRY
    assert "sp2" in HYBRIDIZATION_REGISTRY
    assert "sp" in HYBRIDIZATION_REGISTRY
    assert "dsp2" in HYBRIDIZATION_REGISTRY
    assert "d2sp3" in HYBRIDIZATION_REGISTRY
    assert "sd3" in HYBRIDIZATION_REGISTRY
    assert "sd" in HYBRIDIZATION_REGISTRY
    assert "delta" in HYBRIDIZATION_REGISTRY

    # Check Enum and alias lookups
    spec_sp3 = get_hybridization_spec(HybridizationType.SP3)
    assert spec_sp3.n_sigma == 4
    assert spec_sp3.geometry_name == "Tetrahedral (Td)"

    spec_alias = get_hybridization_spec("square_planar")
    assert spec_alias.name == "dsp2"
    assert spec_alias.n_sigma == 4
    assert spec_alias.has_d_orbitals is True

    spec_oct = get_hybridization_spec("octahedral")
    assert spec_oct.name == "d2sp3"
    assert spec_oct.n_sigma == 6

    spec_delta = get_hybridization_spec("delta")
    assert spec_delta.n_delta == 2
    assert spec_delta.has_d_orbitals is True


def test_d_orbital_rotation_matrix_orthonormality():
    from phatpsy.hybrids import d_orbital_rotation_matrix, valence_rotation_matrix
    from scipy.spatial.transform import Rotation

    # Check arbitrary 3D rotation
    R = Rotation.from_euler("zyx", [35.0, 45.0, 60.0], degrees=True).as_matrix()
    D = d_orbital_rotation_matrix(R)
    assert D.shape == (5, 5)
    assert np.allclose(D @ D.T, np.eye(5), atol=1e-12)
    assert np.linalg.det(D) == pytest.approx(1.0, abs=1e-9)

    R9 = valence_rotation_matrix(R, has_d=True)
    assert R9.shape == (9, 9)
    assert np.allclose(R9 @ R9.T, np.eye(9), atol=1e-12)


def test_kabsch_alignment_of_named_hybridization():
    from phatpsy.hybrids import align_hybrid_to_neighbors, get_hybridization_spec
    from scipy.spatial.transform import Rotation

    spec_sp3 = get_hybridization_spec("sp3")
    # Apply an arbitrary rotation to canonical tetrahedral vectors
    R_rot = Rotation.from_euler("zyx", [20.0, 50.0, 70.0], degrees=True).as_matrix()
    targets = [R_rot @ spec_sp3.canonical_sigma_dirs[0], R_rot @ spec_sp3.canonical_sigma_dirs[2]]

    rot_sigma, _, _, R_found = align_hybrid_to_neighbors(spec_sp3, targets)
    # Check that the rotated sigma vectors align with the targets
    assert any(np.dot(s, targets[0]) > 0.999 for s in rot_sigma)
    assert any(np.dot(s, targets[1]) > 0.999 for s in rot_sigma)


def test_dsp2_square_planar_hybrid_orthonormality():
    from phatpsy.hybrids import _construct_d_hybrid_vectors, get_hybridization_spec
    from scipy.spatial.transform import Rotation

    spec = get_hybridization_spec("dsp2")
    R = Rotation.from_euler("zyx", [15.0, 30.0, 45.0], degrees=True).as_matrix()
    vecs = _construct_d_hybrid_vectors(spec, R)

    sigmas = np.array(vecs["sigma"])
    assert sigmas.shape == (4, 9)
    # 4 sigma hybrids are mutually orthonormal
    assert np.allclose(sigmas @ sigmas.T, np.eye(4), atol=1e-12)

    all_vecs = np.array(vecs["sigma"] + vecs["lone"])
    assert all_vecs.shape == (9, 9)
    assert np.allclose(all_vecs @ all_vecs.T, np.eye(9), atol=1e-12)


def test_d2sp3_octahedral_hybrid_orthonormality():
    from phatpsy.hybrids import _construct_d_hybrid_vectors, get_hybridization_spec
    from scipy.spatial.transform import Rotation

    spec = get_hybridization_spec("d2sp3")
    R = Rotation.from_euler("zyx", [40.0, -25.0, 55.0], degrees=True).as_matrix()
    vecs = _construct_d_hybrid_vectors(spec, R)

    sigmas = np.array(vecs["sigma"])
    assert sigmas.shape == (6, 9)
    # 6 sigma hybrids are mutually orthonormal
    assert np.allclose(sigmas @ sigmas.T, np.eye(6), atol=1e-12)

    all_vecs = np.array(vecs["sigma"] + vecs["lone"])
    assert all_vecs.shape == (9, 9)
    assert np.allclose(all_vecs @ all_vecs.T, np.eye(9), atol=1e-12)


def test_build_atom_orbitals_with_named_hybridization():
    elem_c = get_element_basis("C", "SZ")
    orbs_sp3 = build_atom_orbitals(elem_c, "up", hybridization="sp3")
    assert len(orbs_sp3) == len(elem_c.default_orbitals)
    # 4 valence orbitals singly occupied with alpha spin
    valence_bonding = [o for o in orbs_sp3 if o.occno == [1.0, 0.0]]
    assert len(valence_bonding) == 4

    # Water with named bent hybridization
    elem_o = get_element_basis("O", "SZ")
    orbs_o = build_atom_orbitals(elem_o, "down", hybridization="sp3_bent")
    bonding = [o for o in orbs_o if o.occno == [0.0, 1.0]]
    lone_pairs = [o for o in orbs_o if o.occno == [1.0, 1.0]]
    assert len(bonding) == 2
    assert len(lone_pairs) == 3  # 1s core + 2 valence lone pairs


def test_transition_metal_nickel_dsp2_and_d2sp3():
    elem_ni = get_element_basis("Ni", "ADF-DZ")
    orbs_dsp2 = build_atom_orbitals(elem_ni, "up", hybridization="dsp2")
    assert len(orbs_dsp2) == len(elem_ni.default_orbitals)

    # Check total electron conservation (Ni Z=28)
    tot_e = sum(sum(o.occno) for o in orbs_dsp2)
    assert tot_e == pytest.approx(28.0)

    # 4 dsp2 sigma bonding orbitals with spin up
    bonding = [o for o in orbs_dsp2 if o.occno == [1.0, 0.0]]
    assert len(bonding) == 4

    # Octahedral Ni
    orbs_oct = build_atom_orbitals(elem_ni, "up", hybridization="d2sp3")
    tot_e_oct = sum(sum(o.occno) for o in orbs_oct)
    assert tot_e_oct == pytest.approx(28.0)


def test_deck_explicit_named_hybridization_override():
    from phatpsy.schema import AtomConfig, CalculationDeck

    deck = CalculationDeck(
        title="Ethylene with named sp2 hybrid override",
        geometry=[
            AtomConfig("C", [0.0, 0.0, 0.0], hybridization="sp2", spin="up"),
            AtomConfig("C", [1.34, 0.0, 0.0], hybridization="sp2", spin="down"),
            AtomConfig("H", [-0.5, 0.9, 0.0], hybridization="s", spin="down"),
            AtomConfig("H", [-0.5, -0.9, 0.0], hybridization="s", spin="down"),
            AtomConfig("H", [1.84, 0.9, 0.0], hybridization="s", spin="up"),
            AtomConfig("H", [1.84, -0.9, 0.0], hybridization="s", spin="up"),
        ],
        basis="ADF-DZ",
    )
    apply_hybrid_seeds(deck)
    for atom in deck.geometry:
        assert atom.starting_orbitals is not None
        assert len(atom.starting_orbitals) > 0
    # Total electrons for C2H4 = 2*6 + 4*1 = 16
    total_electrons = sum(sum(o.occno) for a in deck.geometry for o in a.starting_orbitals)
    assert total_electrons == pytest.approx(16.0)

