"""Spin assignment of the first atom of each element must reach the engine deck.

Library default orbitals are up-polarized.  A spin-down atom that is the first
(primary) atom of its element must therefore be emitted with alpha/beta swapped.
These are pure invariants of the deck, with no calibrated values.
"""
import re

import pytest

from phatpsy.builder import build_diatomic


def _atorb_blocks(deck):
    """Returns [(atom_line, [(occ_alpha, occ_beta), ...]), ...] from the stdin deck."""
    lines = deck.to_stdin_text().splitlines()
    blocks = []
    for line in lines:
        s = line.strip()
        if s.startswith("&ATOM"):
            blocks.append((s, []))
        elif "&ATORB" in s and blocks:
            m = re.search(r"OCCNO=([-\d.]+),([-\d.]+)", s)
            if m:
                blocks[-1][1].append((float(m.group(1)), float(m.group(2))))
    return blocks


def _totals(block):
    return sum(o[0] for o in block[1]), sum(o[1] for o in block[1])


def _deck(first, second, net_spin, a="C", b="O", r=2.132):
    return build_diatomic(a, b, r, basis="DZ", spins=[first, second], net_spin=net_spin, units="bohr")


def test_heteronuclear_antiparallel_total_spin_is_zero():
    blocks = _atorb_blocks(_deck("up", "down", 0))
    assert len(blocks) == 2
    alpha = sum(_totals(b)[0] for b in blocks)
    beta = sum(_totals(b)[1] for b in blocks)
    assert alpha == pytest.approx(beta)


def test_heteronuclear_parallel_total_spin_matches_net_spin():
    blocks = _atorb_blocks(_deck("up", "up", 2))
    alpha = sum(_totals(b)[0] for b in blocks)
    beta = sum(_totals(b)[1] for b in blocks)
    assert alpha - beta > 0


def test_spin_down_atom_is_mirror_of_spin_up_atom():
    up = _atorb_blocks(_deck("up", "up", 2))
    mixed = _atorb_blocks(_deck("up", "down", 0))
    # first atom unchanged; second atom has alpha and beta occupations swapped
    assert up[0][1] == mixed[0][1]
    assert [(b, a) for a, b in up[1][1]] == mixed[1][1]


def test_reversed_assignment_swaps_the_first_atom_instead():
    fwd = _atorb_blocks(_deck("up", "down", 0))
    rev = _atorb_blocks(_deck("down", "up", 0))
    assert [(b, a) for a, b in fwd[0][1]] == rev[0][1]
    assert [(b, a) for a, b in fwd[1][1]] == rev[1][1]


def test_equivalent_atoms_still_use_qflip_not_swap():
    deck = build_diatomic("N", "N", 2.0744, basis="DZ", spins=["up", "down"], net_spin=0, units="bohr")
    text = deck.to_stdin_text()
    assert "QFLIP=T" in text
    blocks = _atorb_blocks(deck)
    # only the primary atom carries orbital cards; it is not swapped
    assert len([b for b in blocks if b[1]]) == 1
    a, b = _totals([x for x in blocks if x[1]][0])
    assert a >= b


def test_spin_swapped_orbital_swaps_four_column_coefficients():
    from phatpsy.basis import InitialOrbital
    from phatpsy.schema import _spin_swapped_orbital

    orb = InitialOrbital(eigval=[-1.0, -2.0], occno=[1.0, 0.0],
                         coefficients=[(1, 0, 0.5, 0.25), (2, 0, 0.1)])
    sw = _spin_swapped_orbital(orb)
    assert list(sw.occno) == [0.0, 1.0]
    assert list(sw.eigval) == [-2.0, -1.0]
    assert tuple(sw.coefficients[0]) == (1, 0, 0.25, 0.5)
    assert tuple(sw.coefficients[1]) == (2, 0, 0.1)
    assert list(orb.occno) == [1.0, 0.0]  # original untouched

