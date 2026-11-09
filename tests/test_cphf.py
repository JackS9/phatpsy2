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

    def test_cphf_n2_heavy_atom_pk_relaxed_gradient_and_frequency(self, driver):
        """
        Homonuclear triple-bonded N2 (DZ, sp hybridization, orbital-resolved valence_configs):
        Both heavy atoms (Z=7 > 1.5) emit and receive Phillips-Kleinman (PK) pseudopotential
        and regularized pseudometric projectors (P1_A, P2_A). Verifies that CPHFResponseSolver
        includes the exact PK pseudopotential and pseudometric response, matching the SCF
        finite-difference gradient to < 1e-4 Ha/Bohr and yielding a positive real stretching
        frequency under use_grid_fit=True.
        """
        from phatpsy.builder import build_diatomic
        from phatpsy.schema import ValenceOrbital
        from phatpsy.vibrations import compute_vibrations

        n1_valence = [
            ValenceOrbital(spin="up",   occupation=1, orbital_type="sigma",     connect_to="N2"),
            ValenceOrbital(spin="up",   occupation=1, orbital_type="pi",        connect_to="N2"),
            ValenceOrbital(spin="up",   occupation=1, orbital_type="pi",        connect_to="N2"),
            ValenceOrbital(spin="none", occupation=2, orbital_type="lone_pair"),
        ]
        n2_valence = [
            ValenceOrbital(spin="down", occupation=1, orbital_type="sigma",     connect_to="N1"),
            ValenceOrbital(spin="down", occupation=1, orbital_type="pi",        connect_to="N1"),
            ValenceOrbital(spin="down", occupation=1, orbital_type="pi",        connect_to="N1"),
            ValenceOrbital(spin="none", occupation=2, orbital_type="lone_pair"),
        ]
        deck_n2 = build_diatomic(
            element1="N",
            element2="N",
            bond_length=2.0744,
            basis="DZ",
            hybridization="sp",
            spins=["up", "down"],
            valence_configs=[n1_valence, n2_valence],
            net_spin=0,
            physical_net_spin=0,
            units="bohr",
        )
        deck_n2.control.convergence = 1e-8
        deck_n2.control.max_cycles = 150

        st = driver.run_deck(deck_n2)
        assert st["converged"]

        g_rel = compute_polyatomic_relaxed_gradient(st, deck_n2)
        fd_grad = compute_scf_finite_difference_gradient(driver, deck_n2, step_size=1e-4)

        np.testing.assert_allclose(np.sum(g_rel, axis=0), np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(g_rel[:, :2], np.zeros((2, 2)), atol=1e-10)
        np.testing.assert_allclose(g_rel, fd_grad, atol=1e-4)

        vib_n2 = compute_vibrations(
            deck_n2, driver=driver, step_size=0.005, method="central", gradient_method="relaxed", quiet=True
        )
        stretch_mode = [m for m in vib_n2.modes if abs(m.frequency_cm) > 10.0][-1]
        assert not stretch_mode.is_imaginary
        assert stretch_mode.frequency_cm > 2000.0


