"""
Real-harmonic phase convention between the Python STO tooling and the Fortran core.

The Fortran two-center kernels (gensab/rotsab), and therefore the SCF MO coefficients,
use real harmonics carrying the Condon-Shortley phase (-1)^|m|.  Python code that
builds coefficients (hybrid seeds) or evaluates orbitals on grids (volumetric, cube,
viewer, EOM) must use the same convention, or orbitals mixing p_x/p_y/d_xz/d_yz with
other functions on the same atom come out mirrored.

The checks rely only on the Fortran overlap kernel and on rotational invariance.
"""
import math

import numpy as np
import pytest

import _phatpsy_core
from phatpsy.basis import get_element_basis
from phatpsy.builder import build_molecule
from phatpsy.gradients import DiatomicFrame, compute_diatomic_overlap_matrix_and_gradient
from phatpsy.volumetric import evaluate_real_sto


def _norm(n, z):
    return (2 * z) ** (n + 0.5) / math.sqrt(math.factorial(2 * n))


def _shell_arrays(ed):
    sh = [(f.n, f.l, f.zeta) for f in ed.functions]
    return (
        np.array([s[0] for s in sh], dtype=np.int32),
        np.array([s[1] for s in sh], dtype=np.int32),
        np.array([s[2] for s in sh]),
        np.array([_norm(s[0], s[2]) for s in sh]),
    )


def _coef_vector(coefficients, ed):
    """(isto, ml, c) list -> vector in Fortran basis order (shells, ml = -l..l)."""
    offsets, off = [], 0
    for f in ed.functions:
        offsets.append(off)
        off += 2 * f.l + 1
    v = np.zeros(off)
    for isto, ml, c in coefficients:
        f = ed.functions[isto - 1]
        v[offsets[isto - 1] + ml + f.l] += c
    return v


@pytest.fixture(scope="module", autouse=True)
def _init_core():
    _phatpsy_core.state.init_block_data_and_defaults()


@pytest.mark.parametrize("ml,axis", [(1, 0), (-1, 1), (0, 2)])
def test_grid_p_function_sign_matches_fortran_overlap(ml, axis):
    """Sign of <p_ml(A) | s(B)> from the Fortran kernel equals the sign of the grid
    p function evaluated at a point between A and B."""
    xa = np.zeros(3)
    xb = np.zeros(3)
    xb[axis] = 1.5
    ed_a = get_element_basis("O", "DZ")
    ed_b = get_element_basis("H", "DZ")
    s_f, _, _ = compute_diatomic_overlap_matrix_and_gradient(
        *_shell_arrays(ed_a), *_shell_arrays(ed_b), DiatomicFrame(xa, xb)
    )
    vec = _coef_vector([(len(ed_a.functions), ml, 1.0)], ed_a)   # outermost 2p shell
    s_fortran = float(vec @ s_f[:, 0])
    p = [f for f in ed_a.functions if f.l == 1][-1]
    mid = 0.5 * (xa + xb)
    val = evaluate_real_sto(np.array(mid[0]), np.array(mid[1]), np.array(mid[2]),
                            tuple(xa), p.n, 1, ml, p.zeta)
    assert abs(s_fortran) > 1e-3
    assert np.sign(s_fortran) == np.sign(float(val))


def _seeded_bond_overlap(h_xyz):
    deck = build_molecule([("F", [0.0, 0.0, 0.0]), ("H", list(h_xyz))], basis="DZ",
                          spins=["up", "down"], seed_hybrids=True)
    ed_f = get_element_basis("F", "DZ")
    ed_h = get_element_basis("H", "DZ")
    s_f, _, _ = compute_diatomic_overlap_matrix_and_gradient(
        *_shell_arrays(ed_f), *_shell_arrays(ed_h), DiatomicFrame(np.zeros(3), np.array(h_xyz))
    )
    # The singly occupied (alpha-only) F orbital is the sigma-bonding hybrid.
    bond = [o for o in deck.geometry[0].starting_orbitals if o.occno == [1.0, 0.0]]
    assert len(bond) == 1
    return float(_coef_vector(bond[0].coefficients, ed_f) @ s_f[:, 0])


@pytest.mark.parametrize("h_xyz", [[1.7328, 0.0, 0.0], [0.0, 1.7328, 0.0], [-1.7328, 0.0, 0.0],
                                   list(1.7328 / math.sqrt(3.0) * np.array([-1.0, 1.0, 1.0]))])
def test_seeded_bonding_hybrid_points_at_partner_in_any_orientation(h_xyz):
    """The seeded F sigma-bonding hybrid of HF has the same Fortran overlap with the
    H 1s for every orientation as on the z axis (where ml = 0 carries no phase)."""
    s_ref = _seeded_bond_overlap([0.0, 0.0, 1.7328])
    assert s_ref > 0.3
    assert _seeded_bond_overlap(h_xyz) == pytest.approx(s_ref, abs=1e-8)
