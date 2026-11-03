"""Behavioral tests for MAXOVL (src/linalg.f90), the maximum-overlap orbital matcher.

MAXOVL returns IORDER with IORDER(j) = i meaning "new vector j corresponds to old orbital i";
the SCF then applies NEWORD so that the occupation of slot j becomes the old occupation of
orbital IORDER(j). Its control flow reaches the "search the free vectors" branch from two
places on purpose, exactly as in legacy/maxovl.f: (a) the overlap of old orbital i with new
vector i is below one half, or (b) new vector i was already claimed by an earlier orbital.
These tests pin down both paths. They compute the expected ordering from the inputs
(a known permutation or rotation), so they are not calibrated against output values.
"""

import numpy as np
import pytest

import _phatpsy_core as core


def _identity_metric(n):
    packed = np.zeros(n * (n + 1) // 2)
    for k in range(1, n + 1):
        packed[k * (k + 1) // 2 - 1] = 1.0
    return packed


def _maxovl(evec1, evec2):
    n, nvec = evec1.shape
    iorder, qnword = core.linalg.maxovl(
        np.asfortranarray(evec1), np.asfortranarray(evec2), _identity_metric(n)
    )
    return np.asarray(iorder), bool(qnword)


def test_same_vectors_keep_their_order():
    evec = np.eye(5)
    iorder, qnword = _maxovl(evec, evec)
    assert list(iorder) == [1, 2, 3, 4, 5]
    assert qnword is False


def test_sign_flips_do_not_change_the_matching():
    evec = np.eye(4)
    iorder, qnword = _maxovl(evec, evec * np.array([1.0, -1.0, -1.0, 1.0]))
    assert list(iorder) == [1, 2, 3, 4]
    assert qnword is False


def test_pair_swap_is_detected():
    evec1 = np.eye(4)
    evec2 = evec1[:, [1, 0, 2, 3]]  # new vectors 1 and 2 are old orbitals 2 and 1
    iorder, qnword = _maxovl(evec1, evec2)
    assert list(iorder) == [2, 1, 3, 4]
    assert qnword is True


def test_three_cycle_exercises_the_target_already_taken_path():
    """new1 = old2, new2 = old3, new3 = old1.

    Old orbital 3 has its own slot (new vector 3) already claimed by old orbital 1, so it
    must fall through to the search over the remaining free vectors.
    """
    evec1 = np.eye(3)
    evec2 = evec1[:, [1, 2, 0]]
    iorder, qnword = _maxovl(evec1, evec2)
    assert list(iorder) == [2, 3, 1]
    assert qnword is True


@pytest.mark.parametrize("angle_deg, expected", [(20.0, [1, 2, 3]), (70.0, [2, 1, 3])])
def test_rotation_within_a_near_degenerate_pair_follows_the_larger_overlap(angle_deg, expected):
    """Rotating a degenerate pair (as an eigensolver may) swaps the labels once cos^2 < 1/2."""
    t = np.radians(angle_deg)
    evec1 = np.eye(3)
    evec2 = np.eye(3)
    evec2[:2, :2] = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]
    iorder, _ = _maxovl(evec1, evec2)
    assert list(iorder) == expected


def test_result_is_always_a_valid_permutation():
    """IORDER is a permutation of 1..n; QNWORD is set whenever the order is not the identity.

    The converse need not hold: QNWORD is raised on entering the free-vector search, which can
    still end by keeping an orbital in its own slot (best overlap below one half). The SCF
    therefore re-checks whether the occupations actually change before reordering them.
    """
    rng = np.random.default_rng(12345)
    for nvec in (3, 5, 8):
        identity = list(range(1, nvec + 1))
        for _ in range(25):
            q, _ = np.linalg.qr(rng.normal(size=(nvec, nvec)))
            iorder, qnword = _maxovl(np.eye(nvec), q)
            assert sorted(iorder) == identity
            if list(iorder) != identity:
                assert qnword is True
            if not qnword:
                assert list(iorder) == identity
