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



def test_cas_analytic_orbital_gradient_matches_finite_differences():
    """Analytic gradient of the one-center CAS energy (mirror-symmetric N2 and general
    CO parametrizations) against central finite differences."""
    import sys
    sys.path.insert(0, "tools")
    from localized_vb_scan import NELEC, build_model, initial_orbitals
    from phatpsy.localized_vb import atom_basis

    for a, b, R in (("N", "N", 2.0744), ("C", "O", 2.132)):
        ints = diatomic_ao_integrals(atom_basis((0, 0, 0), a, "DZ"), atom_basis((0, 0, R), b, "DZ"),
                                     float(NELEC[a]), float(NELEC[b]))
        m = build_model(ints, a, b, a == b)
        m.method = "cas"
        x = m.pack(initial_orbitals(m)) + 0.05 * np.random.default_rng(1).standard_normal(m.nparam)
        e, g = m.energy_and_grad(x)
        assert e == pytest.approx(m.energy(x), abs=1e-10)
        for i in range(0, x.size, max(1, x.size // 6)):
            xp, xm = x.copy(), x.copy()
            xp[i] += 1e-5
            xm[i] -= 1e-5
            assert g[i] == pytest.approx((m.energy(xp) - m.energy(xm)) / 2e-5, abs=1e-7)


def test_own_atom_character_closed_form_and_gauge_invariance(h2_ints):
    """sigma_g = (a + b)/sqrt(2(1+S)) has own-atom character (1+S)/2 on either atom;
    a one-center orbital has 1; the measure is invariant to rotations within a subspace."""
    from phatpsy.localized_vb import own_atom_character

    S = h2_ints.S[0, 1]
    sg = np.array([[1.0], [1.0]])
    assert own_atom_character(h2_ints, sg, 0, 1)[0] == pytest.approx((1 + S) / 2, abs=1e-12)
    assert own_atom_character(h2_ints, np.array([[1.0], [0.0]]), 0, 1)[0] == pytest.approx(1.0, abs=1e-12)
    W = np.eye(2)
    U = np.array([[0.6, -0.8], [0.8, 0.6]])
    assert np.allclose(own_atom_character(h2_ints, W, 0, 1), own_atom_character(h2_ints, W @ U, 0, 1))


def test_masked_vb_unrestricted_domains_reproduce_casscf():
    """With every orbital allowed on all atoms, the masked model is CASSCF; with
    one-center domains it is variationally above it."""
    from pyscf import mcscf
    from phatpsy.localized_vb import MaskedVB, orthonormal_mo_guess

    R = 2.0
    A, B = (0.0, 0.0, 0.0), (0.0, 0.0, R)
    ints = diatomic_ao_integrals(
        [STO(A, 1, 0, 0, 2.7), STO(A, 2, 0, 0, 0.65), STO(A, 2, 1, 0, 0.65)],
        [STO(B, 1, 0, 0, 2.7), STO(B, 2, 0, 0, 0.65), STO(B, 2, 1, 0, 0.65)], 3.0, 3.0)
    roles = ["inactive", "inactive", "active", "active"]
    onA, onB, every = np.arange(3), np.arange(3, 6), np.arange(6)
    x_core = [np.eye(6)[0], np.eye(6)[3]]
    act = [np.eye(6)[1] + 0.3 * np.eye(6)[2], np.eye(6)[4] - 0.3 * np.eye(6)[5]]
    strict = MaskedVB(ints, roles, [onA, onB, onA, onB], (1, 1))
    free = MaskedVB(ints, roles, [every] * 4, (1, 1))
    rs = strict.optimize(strict.pack(x_core + act))
    mf = pyscf_mean_field(ints, (3, 3), "RHF")
    Xi, Xa = strict.spaces(strict.orbitals(rs.x))
    mc = mcscf.CASSCF(mf, 2, 2)
    mc.verbose = 0
    mc.conv_tol = 1e-11
    mc.conv_tol_grad = 1e-7
    mc.kernel(mo_coeff=orthonormal_mo_guess(Xi, Xa, ints.S))
    # At the CASSCF orbitals the unrestricted masked model reproduces E_CASSCF and is stationary
    C = mc.mo_coeff
    orbs = [C[:, 0], C[:, 1], C[:, 2], C[:, 3]]
    e, g = free.energy_and_grad(free.pack(orbs))
    assert e == pytest.approx(mc.e_tot, abs=1e-9)
    assert np.max(np.abs(g)) < 1e-5
    # Strict one-center domains are variationally above CASSCF
    assert rs.fun >= mc.e_tot - 1e-9
