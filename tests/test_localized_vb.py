"""
Tests for the exact-integral localized-orbital VB layer (phatpsy/localized_vb.py).

References are independent of the code path under test:
  - Heitler-London and Weinbaum H2 energies from Sugiura closed-form integrals.
  - Full nonorthogonal-determinant CI (generalized Loewdin rules, including
    symmetry-singular overlap blocks) versus PySCF FCI in an orthonormalized basis.
  - RHF with custom exact-STO integrals versus the closed-shell determinant energy.
"""
from itertools import combinations

import numpy as np
import pytest

from phatpsy.localized_vb import (cas_energy, covalent_determinants, determinant_pair,
                                  diatomic_ao_integrals, generalized_ci_energy, pp_energy,
                                  pyscf_mean_field)
from phatpsy.two_center_integrals import STO
from tests.test_two_center_integrals import _sugiura


@pytest.fixture(scope="module")
def h2_ints():
    return diatomic_ao_integrals([STO((0, 0, 0), 1, 0, 0, 1.0)], [STO((0, 0, 1.4), 1, 0, 0, 1.0)], 1.0, 1.0)


def test_h2_pp_covalent_cas_match_closed_forms(h2_ints):
    R = 1.4
    ref = _sugiura(R)
    S = ref["S"]
    haa, hab = 0.5 + ref["Vaa"], ref["Tab"] + ref["Vab"]
    e_hl = (2 * haa + 2 * S * hab + ref["J"] + ref["K"]) / (1 + S * S) + 1 / R
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    none = np.zeros((2, 0))
    assert pp_energy(none, [(a, b)], h2_ints) == pytest.approx(e_hl, abs=1e-12)
    assert generalized_ci_energy(covalent_determinants(none, [a, b]), h2_ints)[0] == pytest.approx(e_hl, abs=1e-12)
    # Weinbaum (covalent + ionic) = sigma_g^2/sigma_u^2 CI (see test_vb_correction)
    assert cas_energy(none, [a, b], h2_ints) == pytest.approx(-1.106556606, abs=1e-8)


def test_full_nonorthogonal_ci_with_singular_blocks_equals_orthonormal_fci():
    """Two centers with 1s and 2p_x each, four electrons: s_A/p_x,B overlaps vanish by
    symmetry, so many determinant pairs have singular orbital-overlap blocks."""
    R = 1.6
    A, B = (0.0, 0.0, 0.0), (0.0, 0.0, R)
    ints = diatomic_ao_integrals(
        [STO(A, 1, 0, 0, 1.2), STO(A, 2, 1, 1, 1.0)], [STO(B, 1, 0, 0, 1.2), STO(B, 2, 1, 1, 1.0)], 2.0, 2.0
    )
    act = [np.eye(4)[:, k] for k in range(4)]
    dets = []
    for al in combinations(range(4), 2):
        for be in combinations(range(4), 2):
            dets.append((np.array([act[k] for k in al]).T, np.array([act[k] for k in be]).T, 1.0))
    e_noci = generalized_ci_energy(dets, ints)[0]
    e_fci = cas_energy(np.zeros((4, 0)), act, ints, nelec_act=(2, 2))
    assert e_noci == pytest.approx(e_fci, abs=1e-9)


def test_closed_shell_determinant_matches_pyscf_rhf():
    R = 2.0
    A, B = (0.0, 0.0, 0.0), (0.0, 0.0, R)
    ints = diatomic_ao_integrals(
        [STO(A, 1, 0, 0, 2.7), STO(A, 2, 0, 0, 0.65)], [STO(B, 1, 0, 0, 2.7), STO(B, 2, 0, 0, 0.65)], 3.0, 3.0
    )
    mf = pyscf_mean_field(ints, (3, 3), "RHF")
    C = mf.mo_coeff[:, :3]
    s, h = determinant_pair(C, C, C, C, ints)
    assert h / s == pytest.approx(mf.e_tot, abs=1e-10)

