"""
Tests for the exploratory two-center pair-energy layer (phatpsy/vb_correction.py).

Independent references only (no fitted values):
  - Heitler-London energy assembled from the Sugiura closed-form integrals.
  - Singlet CI in span{a, b} must equal the sigma_g^2 / sigma_u^2 two-configuration CI
    built from symmetry MOs (an independent route to the Weinbaum/Coulson-Fischer energy).
  - Classic literature results (Gate G0): Heitler-London zeta=1 minimum and the
    Weinbaum/Coulson-Fischer optimum, reproduced without adjustable parameters.
"""
import math

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from phatpsy.two_center_integrals import STO, ProlateGrid
from phatpsy.vb_correction import pair_energies_from_orbitals
from tests.test_two_center_integrals import _sugiura


def _bare(R, zeta=1.0):
    A, B = (0.0, 0.0, 0.0), (0.0, 0.0, R)
    g = ProlateGrid(A, B, zeta_min=zeta)
    return pair_energies_from_orbitals(
        g, [STO(A, 1, 0, 0, zeta)], np.array([1.0]), [STO(B, 1, 0, 0, zeta)], np.array([1.0]), 1.0, 1.0, lmax=20
    )


@pytest.mark.parametrize("R", [1.0, 1.4, 1.64, 2.5])
def test_heitler_london_matches_sugiura_closed_form(R):
    ref = _sugiura(R)
    S = ref["S"]
    haa = 0.5 + ref["Vaa"]
    hab = ref["Tab"] + ref["Vab"]
    e_hl = (2 * haa + 2 * S * hab + ref["J"] + ref["K"]) / (1 + S * S) + 1 / R
    pe = _bare(R)
    assert pe.E_HL == pytest.approx(e_hl, abs=1e-10)
    assert pe.E_HP == pytest.approx(2 * haa + ref["J"] + 1 / R, abs=1e-10)


@pytest.mark.parametrize("R", [1.4, 2.0])
def test_cas_equals_two_configuration_mo_ci(R):
    ref = _sugiura(R)
    S = ref["S"]
    haa, hab = 0.5 + ref["Vaa"], ref["Tab"] + ref["Vab"]
    J, K, hyb, aaaa = ref["J"], ref["K"], ref["hyb"], 0.625
    # sigma_g/u = (a +/- b)/sqrt(2(1 +/- S)); integrals by expansion over a, b
    hg = (haa + hab) / (1 + S)
    hu = (haa - hab) / (1 - S)
    gggg = (2 * aaaa + 2 * J + 4 * K + 8 * hyb) / (4 * (1 + S) ** 2)
    uuuu = (2 * aaaa + 2 * J + 4 * K - 8 * hyb) / (4 * (1 - S) ** 2)
    gugu = (2 * aaaa - 2 * J) / (4 * (1 + S) * (1 - S))
    H = np.array([[2 * hg + gggg, gugu], [gugu, 2 * hu + uuuu]])
    e_mo_ci = np.linalg.eigvalsh(H)[0] + 1 / R
    assert _bare(R).E_CAS == pytest.approx(e_mo_ci, abs=1e-10)


def test_gate_g0_classic_valence_bond_minima():
    # Heitler-London / Sugiura (zeta = 1): R_e ~ 1.64 Bohr, D_e ~ 3.15 eV
    r = minimize_scalar(lambda R: _bare(R).E_HL, bounds=(1.3, 2.0), method="bounded", options={"xatol": 1e-4})
    assert r.x == pytest.approx(1.6425, abs=2e-3)
    assert r.fun == pytest.approx(-1.115970, abs=2e-5)
    # Weinbaum / Coulson-Fischer at its optimum (zeta ~ 1.194, R ~ 1.43): E ~ -1.1479 Ha
    e = minimize_scalar(lambda z: _bare(1.4304, z).E_CAS, bounds=(1.0, 1.4), method="bounded").fun
    assert e == pytest.approx(-1.147937, abs=2e-5)
    assert e < _bare(1.4304, 1.194).E_HL  # ionic structures lower the energy

