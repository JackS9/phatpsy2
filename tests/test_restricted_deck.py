"""Restricted (spin_unrestricted=False) decks are closed-shell singlets without per-atom spins."""
import numpy as np
import pytest

from phatpsy.builder import build_diatomic, build_molecule
from phatpsy.schema import CalculationDeck, ControlConfig


def _water(**kw):
    r, ha = 1.81, np.radians(104.5 / 2)
    g = [("O", [0, 0, 0]), ("H", [0, float(r * np.sin(ha)), float(r * np.cos(ha))]),
         ("H", [0, float(-r * np.sin(ha)), float(r * np.cos(ha))])]
    return build_molecule(g, basis="DZ", spin_unrestricted=False, **kw)


def test_restricted_deck_needs_no_atom_spins():
    deck = _water()
    assert deck.net_spin == 0
    assert all(a.spin is None for a in deck.geometry)
    assert "QOPEN=F" in deck.to_stdin_text()


def test_restricted_deck_ignores_optional_spin_labels():
    deck = _water(spins=["up", "down", "down"], net_spin=-1)
    assert [a.spin for a in deck.geometry] == ["up", "down", "down"]
    assert "QOPEN=F" in deck.to_stdin_text()


def test_unrestricted_deck_still_requires_spins():
    r = 1.81
    g = [("O", [0, 0, 0]), ("H", [0, r, 0]), ("H", [0, -r, 0])]
    with pytest.raises(ValueError, match="explicit spin"):
        build_molecule(g, basis="DZ", spin_unrestricted=True)


def test_restricted_diatomic_homonuclear_and_heteronuclear():
    ctrl = ControlConfig.localized_vb(basis="DZ", domain_mode="bond_local")
    h2 = build_diatomic("H", "H", bond_length=1.427, basis="ADF-DZ", control=ctrl, spin_unrestricted=False, net_spin=0)
    assert h2.control.spin_unrestricted is False
    assert h2.net_spin == 0
    assert all(a.spin is None for a in h2.geometry)

    co = build_diatomic("C", "O", bond_length=2.132, basis="DZ", spin_unrestricted=False, net_spin=0)
    assert co.control.spin_unrestricted is False
    assert co.net_spin == 0
    assert all(a.spin is None for a in co.geometry)

    h2_cls = CalculationDeck.from_diatomic("H", "H", bond_length=1.427, basis="ADF-DZ", spin_unrestricted=False, net_spin=0)
    assert h2_cls.control.spin_unrestricted is False
    assert h2_cls.net_spin == 0
    assert all(a.spin is None for a in h2_cls.geometry)

