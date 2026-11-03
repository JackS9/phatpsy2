"""Selecting the model-potential exponent set per element (`elements: <sym>: potential_exponents`).

These are plumbing and validation tests: they check that a requested exponent set is validated,
written to the deck, applied only to the requested element, never mutates the shared basis
library, and reaches the Fortran core with the documented scaling. They make no claim about which
exponent set is physically best (see ROADMAP.md).
"""

import numpy as np
import pytest

from phatpsy.basis import get_element_basis
from phatpsy.builder import build_diatomic
from phatpsy.driver import PhatpsyDriver
from phatpsy.schema import CalculationDeck, ElementConfig

ZSCALE = 1.12956  # engine default model-potential length scale: ZSCALE * Z**(1/3) (src/atomic.f90)
EXPONENTS = [0.45, 0.9, 1.8, 3.6, 7.2]


def _co_deck(**elements):
    deck = build_diatomic("C", "O", 2.1322, basis="DZ", spins=["up", "down"], net_spin=0, units="bohr")
    deck.control.pk_scale = 1.0
    deck.control.max_cycles = 6
    for sym, cfg in elements.items():
        deck.elements[sym] = cfg
    return deck


def _element_block(text, symbol):
    """Lines of the &BASDIM/&ELEM header and basis block belonging to one element."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if f"LATOM='{symbol}'" in l)
    # BASDIM precedes ELEM by two lines; the block runs until the first &ATOM namelist
    first = start - 4
    stop = next(i for i in range(start, len(lines)) if lines[i].strip().startswith("&ATOM"))
    return lines[first:stop]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(potential_exponents=[1.0]), "between 2 and 19"),
        (dict(potential_exponents=list(np.linspace(0.1, 5.0, 20))), "between 2 and 19"),
        (dict(potential_exponents=[0.5, -1.0]), "finite and > 0"),
        (dict(potential_exponents=[0.5, 0.0]), "finite and > 0"),
        (dict(potential_exponents=[0.5, float("nan")]), "finite and > 0"),
        (dict(potential_exponents=[0.5, 0.5]), "distinct"),
        (dict(potential_exponents=[0.5, 1.0], potential_exponent_scale=0.0), "scale"),
        (dict(potential_exponent_scale=1.0), "only applies together"),
    ],
)
def test_invalid_exponent_requests_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ElementConfig(symbol="C", **kwargs)


def test_element_config_round_trips_through_dict():
    cfg = ElementConfig(symbol="C", potential_exponents=EXPONENTS, potential_exponent_scale=1.0)
    again = ElementConfig.from_dict(cfg.to_dict())
    assert again.potential_exponents == EXPONENTS
    assert again.potential_exponent_scale == 1.0
    assert "potential_exponents" not in ElementConfig(symbol="C").to_dict()


def test_exponents_round_trip_through_yaml():
    deck = _co_deck(C=ElementConfig(symbol="C", potential_exponents=EXPONENTS))
    reloaded = CalculationDeck.from_yaml(deck.to_yaml())
    assert reloaded.elements["C"].potential_exponents == EXPONENTS


def test_default_decks_are_unchanged():
    text = _co_deck().to_stdin_text()
    assert text.count("QTFPOT=T") == 2
    assert "QTFPOT=F" not in text
    assert "RSCALE" not in text
    assert text.count("  0  0   0.150") == 2


def test_selected_exponents_apply_to_the_requested_element_only():
    text = _co_deck(C=ElementConfig(symbol="C", potential_exponents=EXPONENTS)).to_stdin_text()
    carbon = "\n".join(_element_block(text, "C"))
    oxygen = "\n".join(_element_block(text, "O"))
    assert f"NVTERM={len(EXPONENTS) + 1}" in carbon
    assert "QTFPOT=F" in carbon
    assert "RSCALE" not in carbon  # default Z-based scale is kept
    assert "0.150" not in carbon
    for e in EXPONENTS:
        assert f"{e:10.6f}" in carbon
    # Oxygen keeps the built-in set
    assert "NVTERM=5" in oxygen
    assert "QTFPOT=T" in oxygen
    assert "  0  0   0.150" in oxygen


def test_absolute_scale_is_written_as_rscale():
    text = _co_deck(
        C=ElementConfig(symbol="C", potential_exponents=EXPONENTS, potential_exponent_scale=1.0)
    ).to_stdin_text()
    assert "RSCALE=1.000000" in "\n".join(_element_block(text, "C"))


def test_shared_basis_library_is_not_mutated():
    before = get_element_basis("C", "DZ")
    nvterm, terms = before.nvterm, list(before.model_potential_terms)
    _co_deck(C=ElementConfig(symbol="C", potential_exponents=EXPONENTS)).to_stdin_text()
    after = get_element_basis("C", "DZ")
    assert after.nvterm == nvterm
    assert list(after.model_potential_terms) == terms
    assert "QTFPOT=F" not in _co_deck().to_stdin_text()


def _vexp(result, atom_id):
    return np.array(result["embedded_atoms"][atom_id].model_potential["vexp"])


def test_selected_exponents_reach_the_core_with_the_default_z_scale():
    deck = _co_deck(C=ElementConfig(symbol="C", potential_exponents=EXPONENTS))
    res = PhatpsyDriver(quiet=True).run_deck(deck)
    vexp_c = _vexp(res, "C1")
    assert len(vexp_c) == len(EXPONENTS) + 1
    assert vexp_c[0] == 0.0  # the 1/r point-charge term
    assert vexp_c[1:] == pytest.approx(np.array(EXPONENTS) * ZSCALE * 6 ** (1.0 / 3.0), rel=1e-5)
    # Oxygen is untouched: it still has the five built-in terms
    assert len(_vexp(res, "O2")) == 5


def test_absolute_scale_reaches_the_core_unscaled():
    deck = _co_deck(C=ElementConfig(symbol="C", potential_exponents=[1.5, 6.0, 12.0], potential_exponent_scale=1.0))
    res = PhatpsyDriver(quiet=True).run_deck(deck)
    assert _vexp(res, "C1") == pytest.approx([0.0, 1.5, 6.0, 12.0], rel=1e-5)

