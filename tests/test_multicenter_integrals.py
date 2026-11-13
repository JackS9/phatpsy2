"""
Tests for multi-center STO integrals from pair-density potentials
(phatpsy/multicenter_integrals.py and ProlateGrid.potentials_at).

References: closed-form one-center potentials; the exact two-center (single-frame)
path; and an independent PySCF rebuild with Gaussian-fitted STOs for genuinely
three-center integrals (agreement limited by the fit, ~1e-6).
"""
import math
import sys

import numpy as np
import pytest

from phatpsy.multicenter_integrals import molecular_ao_integrals
from phatpsy.two_center_integrals import STO, ProlateGrid
from tests.test_two_center_integrals import _sugiura


def test_potentials_at_match_closed_form_one_center_potential():
    R = 1.4
    A, B = np.zeros(3), np.array([0.0, 0.0, R])
    g = ProlateGrid(A, B, zeta_min=1.0)
    b = g.evaluate(STO(tuple(B), 1, 0, 0, 1.0))
    pts = np.random.default_rng(0).uniform(-3, 4, size=(60, 3))
    V = g.potentials_at(np.array([b * b]), pts, lmax=24)[0]
    r = np.linalg.norm(pts - B, axis=1)
    assert np.max(np.abs(V - (1 / r - np.exp(-2 * r) * (1 + 1 / r)))) < 1e-10


def test_cross_frame_coulomb_reproduces_two_center_hybrid():
    """(ab|aa) with rho_ab in the A-B frame and rho_aa integrated in an unrelated A-D
    frame equals the Sugiura closed form."""
    R = 1.4
    A, B, D = np.zeros(3), np.array([0.0, 0.0, R]), np.array([1.1, -0.8, -0.5])
    g = ProlateGrid(A, B, zeta_min=1.0)
    a, b = g.evaluate(STO(tuple(A), 1, 0, 0, 1.0)), g.evaluate(STO(tuple(B), 1, 0, 0, 1.0))
    g2 = ProlateGrid(A, D, zeta_min=1.0, n_eta=40, n_phi=24)
    pts = np.stack([g2.X.ravel(), g2.Y.ravel(), g2.Z.ravel()], axis=1)
    V = g.potentials_at(np.array([a * b]), pts, lmax=24)[0].reshape(g2.shape)
    val = g2.integrate(V * g2.evaluate(STO(tuple(A), 1, 0, 0, 1.0)) ** 2)
    assert val == pytest.approx(_sugiura(R)["hyb"], abs=5e-8)


def test_three_center_integrals_match_independent_gaussian_rebuild():
    sys.path.insert(0, "tools")
    from crosscheck_determinant_pyscf import fit_sto
    from pyscf import gto

    X = [np.array([0, 0, 0.0]), np.array([1.6, 0.3, 0.0]), np.array([0.5, 1.5, 0.4])]
    shells = [(1, 0, 1.0), (1, 0, 1.6)]
    atoms = [(1.0, x, [STO(tuple(x), n, l, 0, z) for (n, l, z) in shells]) for x in X]
    mi = molecular_ao_integrals(atoms, n_eta=32, n_phi=24, lmax=16)
    mol = gto.M(atom=[[f"H{k}", tuple(x)] for k, x in enumerate(X)],
                basis={f"H{k}": [[0] + [[float(a), float(c / gto.gto_norm(0, a))] for a, c in zip(*fit_sto(1, 0, z, K=18))]
                                 for (_, _, z) in shells] for k in range(3)},
                unit="Bohr", verbose=0, spin=1)
    assert np.max(np.abs(mi.S - mol.intor("int1e_ovlp"))) < 1e-7
    assert np.max(np.abs(mi.V - mol.intor("int1e_nuc"))) < 3e-5        # fit-limited (kinetic-like cusp)
    eri = mol.intor("int2e")
    centres = mi.atom_of
    three = np.array([[[[len({centres[p], centres[q], centres[r], centres[s]}) == 3
                         for s in range(mi.nao)] for r in range(mi.nao)] for q in range(mi.nao)] for p in range(mi.nao)])
    assert np.max(np.abs(mi.eri - eri)[three]) < 2e-6

