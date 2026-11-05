"""
PHATPSY 2.0: Unit Tests for Coupled-Perturbed (CPHF / Z-Vector) Orbital Response (Phase 12).
Verifies fully relaxed analytical Cartesian gradients against central finite differences
of the self-consistent field (SCF) total energy surface, translational zero-sum invariance,
and molecular point-group symmetry across benchmark systems (H2, HF, LiH, H2O).
"""
import numpy as np
import pytest

from phatpsy.builder import build_molecule
from phatpsy.driver import PhatpsyDriver
from phatpsy.gradients import (
    compute_polyatomic_analytical_gradient,
    compute_scf_finite_difference_gradient,
)
from phatpsy.cphf import (
    CPHFResponseSolver,
    compute_polyatomic_relaxed_gradient,
)


@pytest.fixture
def driver():
    return PhatpsyDriver(quiet=True)


class TestCPHFRelaxedGradients:
    """Test suite for Phase 12 CPHF / Z-vector orbital response and relaxed gradients."""

    def test_cphf_h2_variational_limit(self, driver):
        """
        On minimal-basis H2, all valence orbitals are occupied (no virtual/open-shell pairs).
        Z-vector vanishes identically, and the unrelaxed analytical gradient matches the
        relaxed gradient and the central finite difference SCF surface.
        """
        geom = [("H", [0.0, 0.0, 0.7]), ("H", [0.0, 0.0, -0.7])]
        deck = build_molecule(geom, basis="SZ", units="bohr", spins="up", net_spin=2, spin_unrestricted=False)
        deck.control.convergence = 1e-6
        deck.control.max_cycles = 100
        deck.control.use_diis = True

        st = driver.run_deck(deck)
        assert st["converged"]

        solver = CPHFResponseSolver(st, deck)
        assert solver.n_active == 0

        g_unrel = compute_polyatomic_analytical_gradient(st, deck, relaxed=False)
        g_rel = compute_polyatomic_analytical_gradient(st, deck, relaxed=True)

        # Relaxed and unrelaxed must be identical when n_active == 0
        np.testing.assert_allclose(g_rel, g_unrel, atol=1e-12)

        # Exact translational zero-sum invariance
        np.testing.assert_allclose(np.sum(g_rel, axis=0), np.zeros(3), atol=1e-12)

        # Axial symmetry: zero x and y forces
        np.testing.assert_allclose(g_rel[:, 0], np.zeros(2), atol=1e-12)
        np.testing.assert_allclose(g_rel[:, 1], np.zeros(2), atol=1e-12)

        # Compare to central finite difference
        fd_grad = compute_scf_finite_difference_gradient(driver, deck, step_size=0.002)
        np.testing.assert_allclose(g_rel, fd_grad, atol=1e-4)

    def test_cphf_hf_relaxed_gradient(self, driver):
        """
        Heteronuclear polar diatomic HF:
        Coupled-perturbed orbital response relaxes the fluorine valence shell,
        recovering exact agreement with the SCF energy surface.
        """
        geom = [("F", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, 1.733])]
        deck = build_molecule(geom, basis="SZ", units="bohr", spins="up", net_spin=2, spin_unrestricted=False)
        deck.control.convergence = 1e-6
        deck.control.max_cycles = 100
        deck.control.use_diis = True

        st = driver.run_deck(deck)
        assert st["converged"]

        solver = CPHFResponseSolver(st, deck)
        assert solver.n_active > 0

        # Skeleton Fock derivatives satisfy translational zero-sum
        F_deriv = solver.compute_fock_derivatives()
        assert F_deriv.shape == (solver.n_active, 2, 3)
        for p in range(solver.n_active):
            np.testing.assert_allclose(np.sum(F_deriv[p], axis=0), np.zeros(3), atol=1e-12)

        g_rel = compute_polyatomic_relaxed_gradient(st, deck)
        fd_grad = compute_scf_finite_difference_gradient(driver, deck, step_size=0.002)

        # Axial collinear symmetry and zero-sum
        np.testing.assert_allclose(g_rel[:, 0], np.zeros(2), atol=1e-12)
        np.testing.assert_allclose(g_rel[:, 1], np.zeros(2), atol=1e-12)
        np.testing.assert_allclose(np.sum(g_rel, axis=0), np.zeros(3), atol=1e-12)

        # Analytical agreement with SCF finite differences (< 1e-3 Ha/Bohr)
        np.testing.assert_allclose(g_rel, fd_grad, atol=1e-3)

    def test_cphf_lih_relaxed_gradient(self, driver):
        """
        Heteronuclear alkali hydride LiH:
        Tests active s-s core/valence rotation pair and relaxed gradient.
        """
        geom = [("Li", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, 3.015])]
        deck = build_molecule(geom, basis="SZ", units="bohr", spins="up", net_spin=2, spin_unrestricted=False)
        deck.control.convergence = 1e-6
        deck.control.max_cycles = 100
        deck.control.use_diis = True

        st = driver.run_deck(deck)
        assert st["converged"]

        g_rel = compute_polyatomic_relaxed_gradient(st, deck)
        fd_grad = compute_scf_finite_difference_gradient(driver, deck, step_size=0.002)

        # Axial collinear symmetry and zero-sum
        np.testing.assert_allclose(g_rel[:, 0], np.zeros(2), atol=1e-12)
        np.testing.assert_allclose(g_rel[:, 1], np.zeros(2), atol=1e-12)
        np.testing.assert_allclose(np.sum(g_rel, axis=0), np.zeros(3), atol=1e-12)

        # Analytical agreement with SCF finite differences (< 1e-3 Ha/Bohr)
        np.testing.assert_allclose(g_rel, fd_grad, atol=1e-3)

    def test_cphf_h2o_c2v_relaxed_gradient(self, driver):
        """
        Polyatomic non-linear H2O (C2v symmetry):
        Tests 2s-2pz polarization orbital response on Oxygen and verifies
        agreement with the central finite-difference SCF total energy surface
        and exact C2v point group symmetry.
        """
        geom = [
            ("O", [0.0, 0.0, 0.224]),
            ("H", [0.0, 1.45, -0.896]),
            ("H", [0.0, -1.45, -0.896]),
        ]
        deck = build_molecule(geom, basis="SZ", units="bohr", spins="up", net_spin=3, spin_unrestricted=False)
        deck.control.convergence = 1e-6
        deck.control.max_cycles = 100
        deck.control.use_diis = True

        st = driver.run_deck(deck)
        assert st["converged"]

        g_rel = compute_polyatomic_relaxed_gradient(st, deck)
        fd_grad = compute_scf_finite_difference_gradient(driver, deck, step_size=0.002)

        # Exact translational zero-sum
        np.testing.assert_allclose(np.sum(g_rel, axis=0), np.zeros(3), atol=1e-12)

        # C2v Point Group Symmetry Constraints:
        # 1. No forces out of yz plane (x = 0 for all atoms)
        np.testing.assert_allclose(g_rel[:, 0], np.zeros(3), atol=1e-12)
        # 2. Oxygen on C2 axis: y force must be zero
        assert abs(g_rel[0, 1]) < 1e-12
        # 3. Hydrogens related by sigma_v(yz): equal-and-opposite y forces
        np.testing.assert_allclose(g_rel[1, 1], -g_rel[2, 1], atol=1e-12)
        # 4. Hydrogens related by sigma_v(yz): equal z forces
        np.testing.assert_allclose(g_rel[1, 2], g_rel[2, 2], atol=1e-12)

        # Analytical agreement with SCF finite differences (< 1e-3 Ha/Bohr)
        np.testing.assert_allclose(g_rel, fd_grad, atol=1e-3)

    def test_driver_gradient_method_relaxed(self, driver):
        """Verifies driver.gradient(deck, method='relaxed') entrypoint."""
        geom = [("H", [0.0, 0.0, 0.7]), ("H", [0.0, 0.0, -0.7])]
        deck = build_molecule(geom, basis="SZ", units="bohr", spins="up", net_spin=2, spin_unrestricted=False)
        deck.control.convergence = 1e-5
        deck.control.max_cycles = 50
        deck.control.use_diis = True

        grad_ana = driver.gradient(deck, method="analytical")
        grad_rel = driver.gradient(deck, method="relaxed")

        assert grad_ana.shape == (2, 3)
        assert grad_rel.shape == (2, 3)
        np.testing.assert_allclose(grad_rel, grad_ana, atol=1e-10)

