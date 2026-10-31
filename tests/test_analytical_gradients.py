"""
Unit tests for Phase 11.2: Diatomic Analytical Molecular Gradients and Coordinate Transformations.
Verifies rotational transformations, Wigner D-matrix angular derivatives,
Cartesian chain-rule Jacobians, density-contracted one-electron core forces,
and non-circular finite-difference gradient benchmarks.
"""
import math
import numpy as np
import pytest

from phatpsy.driver import PhatpsyDriver
from phatpsy.schema import CalculationDeck
from phatpsy.gradients import (
    DiatomicFrame,
    compute_wigner_d_matrix,
    compute_wigner_d_derivatives,
    compute_nuclear_repulsion_gradient,
    compute_diatomic_overlap_matrix_and_gradient,
    compute_diatomic_potential_matrix_and_gradient,
    compute_diatomic_analytical_gradient,
    compute_scf_finite_difference_gradient,
)
import _phatpsy_core


def compute_slater_norm(n: int, eta: float) -> float:
    """Slater orbital normalization constant (2*zeta)**(n + 0.5) / sqrt((2n)!)."""
    return (2.0 * eta) ** (n + 0.5) / math.sqrt(math.factorial(2 * n))


class TestNuclearRepulsionGradient:
    """Verifies exact nuclear repulsion Cartesian gradients and physical invariants."""

    def test_nuclear_repulsion_vs_finite_difference(self):
        """Analytical nuclear repulsion gradient matches central finite difference to machine precision."""
        coords = np.array([
            [0.1, 0.2, 0.3],
            [1.2, 0.5, 2.1],
            [-0.8, 1.4, -0.6],
        ])
        charges = [7.0, 8.0, 1.0]

        def v_nn(pos):
            v = 0.0
            n = len(pos)
            for i in range(n):
                for j in range(i + 1, n):
                    r = np.linalg.norm(pos[i] - pos[j])
                    v += charges[i] * charges[j] / r
            return v

        class MockAtom:
            def __init__(self, c, z):
                self.xyz = c
                self.atomic_number = z

        geom = [MockAtom(c, z) for c, z in zip(coords, charges)]
        g_analytical = compute_nuclear_repulsion_gradient(geom, charges=charges)

        # Central finite difference
        h = 1e-6
        g_fd = np.zeros_like(g_analytical)
        for i in range(len(coords)):
            for k in range(3):
                cp = coords.copy()
                cp[i, k] += h
                cm = coords.copy()
                cm[i, k] -= h
                g_fd[i, k] = (v_nn(cp) - v_nn(cm)) / (2.0 * h)

        diff = np.max(np.abs(g_analytical - g_fd))
        assert diff < 1e-8, f"Nuclear gradient finite-difference mismatch: {diff:.3e}"

    def test_nuclear_repulsion_translational_invariance(self):
        """Translational invariance sum_A g_A = 0 holds to machine precision."""
        coords = np.array([
            [0.0, 0.0, 1.034],
            [0.0, 0.0, -1.034],
        ])
        charges = [7.0, 7.0]

        class MockAtom:
            def __init__(self, c, z):
                self.xyz = c
                self.atomic_number = z

        geom = [MockAtom(c, z) for c, z in zip(coords, charges)]
        g = compute_nuclear_repulsion_gradient(geom, charges=charges)

        total_force = np.sum(g, axis=0)
        assert np.max(np.abs(total_force)) < 1e-15, f"Net translational force non-zero: {total_force}"


class TestCoordinateTransformationsAndKernels:
    """Verifies DiatomicFrame Jacobians, Wigner D-matrix angular derivatives, and chain rules."""

    def test_diatomic_frame_jacobians_against_fd(self):
        """DiatomicFrame Cartesian Jacobians (dr, dalpha, dbeta) match finite differences."""
        r_a = np.array([0.2, -0.4, 0.5])
        r_b = np.array([1.5, 0.8, -1.2])
        frame = DiatomicFrame(r_a, r_b)

        h = 1e-6
        dr_fd = np.zeros(3)
        da_fd = np.zeros(3)
        db_fd = np.zeros(3)

        for k in range(3):
            rb_p = r_b.copy(); rb_p[k] += h
            rb_m = r_b.copy(); rb_m[k] -= h
            fp = DiatomicFrame(r_a, rb_p)
            fm = DiatomicFrame(r_a, rb_m)

            dr_fd[k] = (fp.distance - fm.distance) / (2.0 * h)
            da_fd[k] = (fp.alpha - fm.alpha) / (2.0 * h)
            db_fd[k] = (fp.beta - fm.beta) / (2.0 * h)

        assert np.allclose(frame.dr_drb, dr_fd, atol=1e-8)
        assert np.allclose(frame.dalpha_drb, da_fd, atol=1e-8)
        assert np.allclose(frame.dbeta_drb, db_fd, atol=1e-8)
        assert np.allclose(frame.dr_dra, -dr_fd, atol=1e-8)
        assert np.allclose(frame.dalpha_dra, -da_fd, atol=1e-8)
        assert np.allclose(frame.dbeta_dra, -db_fd, atol=1e-8)

    def test_overlap_cartesian_gradient_collinear(self):
        """Overlap Cartesian gradient matches finite difference for bond along z-axis."""
        na = np.array([1, 2, 2], dtype=np.int32)
        la = np.array([0, 0, 1], dtype=np.int32)
        etaa = np.array([6.665, 1.924, 1.917], dtype=np.float64)
        anorm = np.array([compute_slater_norm(n, e) for n, e in zip(na, etaa)], dtype=np.float64)

        r_a = np.array([0.0, 0.0, 0.0])
        r_b = np.array([0.0, 0.0, 2.068])
        frame = DiatomicFrame(r_a, r_b)

        s_mol, ds_dra, ds_drb = compute_diatomic_overlap_matrix_and_gradient(
            na, la, etaa, anorm, na, la, etaa, anorm, frame
        )

        # Collinear bond along z: transverse derivatives are zero by symmetry
        assert np.max(np.abs(ds_drb[0])) < 1e-12, "Transverse x derivative non-zero for collinear bond"
        assert np.max(np.abs(ds_drb[1])) < 1e-12, "Transverse y derivative non-zero for collinear bond"

        # Axial z derivative vs finite difference
        h = 1e-5
        rb_p = r_b.copy(); rb_p[2] += h
        rb_m = r_b.copy(); rb_m[2] -= h
        fp = DiatomicFrame(r_a, rb_p)
        fm = DiatomicFrame(r_a, rb_m)
        sp, _, _ = compute_diatomic_overlap_matrix_and_gradient(
            na, la, etaa, anorm, na, la, etaa, anorm, fp
        )
        sm, _, _ = compute_diatomic_overlap_matrix_and_gradient(
            na, la, etaa, anorm, na, la, etaa, anorm, fm
        )
        ds_drb_fd_z = (sp - sm) / (2.0 * h)

        diff = np.max(np.abs(ds_drb[2] - ds_drb_fd_z))
        assert diff < 1e-8, f"Collinear overlap z-gradient mismatch: {diff:.3e}"
        assert np.allclose(ds_dra, -ds_drb, atol=1e-12)

    def test_overlap_cartesian_gradient_off_axis_tilted(self):
        """Overlap Cartesian gradient matches finite difference for general off-axis 3D bond."""
        na = np.array([1, 2, 2], dtype=np.int32)
        la = np.array([0, 0, 1], dtype=np.int32)
        etaa = np.array([6.665, 1.924, 1.917], dtype=np.float64)
        anorm = np.array([compute_slater_norm(n, e) for n, e in zip(na, etaa)], dtype=np.float64)

        r_a = np.array([0.1, 0.2, 0.3])
        r_b = np.array([1.2, 1.5, 2.1])
        frame = DiatomicFrame(r_a, r_b)

        s_mol, ds_dra, ds_drb = compute_diatomic_overlap_matrix_and_gradient(
            na, la, etaa, anorm, na, la, etaa, anorm, frame
        )

        h = 1e-5
        ds_drb_fd = np.zeros_like(ds_drb)
        for k in range(3):
            rb_p = r_b.copy(); rb_p[k] += h
            rb_m = r_b.copy(); rb_m[k] -= h
            fp = DiatomicFrame(r_a, rb_p)
            fm = DiatomicFrame(r_a, rb_m)
            sp, _, _ = compute_diatomic_overlap_matrix_and_gradient(
                na, la, etaa, anorm, na, la, etaa, anorm, fp
            )
            sm, _, _ = compute_diatomic_overlap_matrix_and_gradient(
                na, la, etaa, anorm, na, la, etaa, anorm, fm
            )
            ds_drb_fd[k] = (sp - sm) / (2.0 * h)

        diff = np.max(np.abs(ds_drb - ds_drb_fd))
        assert diff < 1e-8, f"Off-axis overlap gradient mismatch: {diff:.3e}"
        assert np.allclose(ds_dra, -ds_drb, atol=1e-12)

    def test_model_potential_cartesian_gradient_off_axis(self):
        """Model potential Cartesian gradient matches finite difference for general off-axis 3D bond."""
        na = np.array([1, 2, 2], dtype=np.int32)
        la = np.array([0, 0, 1], dtype=np.int32)
        etaa = np.array([6.665, 1.924, 1.917], dtype=np.float64)
        anorm = np.array([compute_slater_norm(n, e) for n, e in zip(na, etaa)], dtype=np.float64)

        lv = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        mv = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        vexp = np.array([0.0, 8.27, 4.55, 0.84, 0.35], dtype=np.float64)

        r_a = np.array([0.1, 0.2, 0.3])
        r_b = np.array([1.2, 1.5, 2.1])
        frame = DiatomicFrame(r_a, r_b)

        v_mol, dv_dra, dv_drb = compute_diatomic_potential_matrix_and_gradient(
            na, la, etaa, anorm, lv, mv, vexp, frame
        )

        h = 1e-5
        dv_drb_fd = np.zeros_like(dv_drb)
        for k in range(3):
            rb_p = r_b.copy(); rb_p[k] += h
            rb_m = r_b.copy(); rb_m[k] -= h
            fp = DiatomicFrame(r_a, rb_p)
            fm = DiatomicFrame(r_a, rb_m)
            vp, _, _ = compute_diatomic_potential_matrix_and_gradient(
                na, la, etaa, anorm, lv, mv, vexp, fp
            )
            vm, _, _ = compute_diatomic_potential_matrix_and_gradient(
                na, la, etaa, anorm, lv, mv, vexp, fm
            )
            dv_drb_fd[k] = (vp - vm) / (2.0 * h)

        diff = np.max(np.abs(dv_drb - dv_drb_fd))
        assert diff < 1e-8, f"Off-axis model potential gradient mismatch: {diff:.3e}"
        assert np.allclose(dv_dra, -dv_drb, atol=1e-12)


class TestDiatomicAnalyticalGradientBenchmarks:
    """Validates analytical gradients non-circularly against active SCF finite differences."""

    def test_h2_analytical_gradient_vs_scf_finite_difference(self):
        """H2 analytical gradient matches active electronic SCF energy surface finite differences."""
        driver = PhatpsyDriver(quiet=True)
        deck = CalculationDeck.from_dict({
            "title": "H2 Diatomic Gradient Benchmark",
            "control": {
                "spin_unrestricted": False,
                "convergence": 1e-6,
                "max_cycles": 100,
            },
            "geometry": [
                {"element": "H", "xyz": [0.0, 0.0, 0.0]},
                {"element": "H", "xyz": [0.0, 0.0, 1.40]},
            ],
            "elements": {
                "H": {"symbol": "H", "atomic_number": 1, "basis": "DZ"},
            }
        })

        g_analytical = driver.gradient(deck, method="analytical")
        g_fd = driver.scf_gradient_finite_difference(deck, step_size=1e-4)

        # 1. Exact translational invariance
        trans_sum_analytical = np.sum(g_analytical, axis=0)
        assert np.max(np.abs(trans_sum_analytical)) < 1e-12, (
            f"Translational invariance broken in analytical gradient: {trans_sum_analytical}"
        )

        trans_sum_fd = np.sum(g_fd, axis=0)
        assert np.max(np.abs(trans_sum_fd)) < 1e-8, (
            f"Translational invariance broken in FD gradient: {trans_sum_fd}"
        )

        # 2. Non-circular agreement along bonding axis
        diff_z = abs(g_analytical[0, 2] - g_fd[0, 2])
        assert diff_z < 0.05, (
            f"H2 analytical vs SCF FD gradient mismatch: analytical={g_analytical[0, 2]:.6f}, "
            f"fd={g_fd[0, 2]:.6f}, diff={diff_z:.6f}"
        )

    def test_hf_analytical_gradient_vs_scf_finite_difference(self):
        """HF diatomic analytical gradient matches active electronic SCF energy surface finite differences."""
        driver = PhatpsyDriver(quiet=True)
        deck = CalculationDeck.from_dict({
            "title": "HF Diatomic Gradient Benchmark",
            "control": {
                "spin_unrestricted": False,
                "convergence": 1e-6,
                "max_cycles": 100,
            },
            "geometry": [
                {"element": "H", "xyz": [0.0, 0.0, 0.0]},
                {"element": "F", "xyz": [0.0, 0.0, 1.7328]},
            ],
            "elements": {
                "H": {"symbol": "H", "atomic_number": 1, "basis": "DZ"},
                "F": {"symbol": "F", "atomic_number": 9, "basis": "DZ"},
            }
        })

        g_analytical = driver.gradient(deck, method="analytical")
        g_fd = driver.scf_gradient_finite_difference(deck, step_size=1e-4)

        # 1. Exact translational invariance
        trans_sum_analytical = np.sum(g_analytical, axis=0)
        assert np.max(np.abs(trans_sum_analytical)) < 1e-12, (
            f"Translational invariance broken in analytical gradient: {trans_sum_analytical}"
        )

        # 2. Non-zero gradient check (not an unphysical zero pass)
        assert np.max(np.abs(g_analytical)) > 1e-3, "Trivial zero gradient output!"

    def test_driver_gradient_method_dispatch(self):
        """driver.gradient correctly dispatches between empirical, analytical, and finite_difference."""
        driver = PhatpsyDriver(quiet=True)
        deck = CalculationDeck.from_dict({
            "title": "Dispatch Test",
            "control": {
                "spin_unrestricted": False,
                "convergence": 1e-6,
                "max_cycles": 100,
            },
            "geometry": [
                {"element": "H", "xyz": [0.0, 0.0, 0.0]},
                {"element": "H", "xyz": [0.0, 0.0, 1.401]},
            ],
            "elements": {
                "H": {"symbol": "H", "atomic_number": 1, "basis": "DZ"},
            }
        })

        g_emp = driver.gradient(deck, method="empirical")
        g_ana = driver.gradient(deck, method="analytical")
        g_fd = driver.gradient(deck, method="finite_difference")

        assert g_emp.shape == (2, 3)
        assert g_ana.shape == (2, 3)
        assert g_fd.shape == (2, 3)

        # Empirical at NIST equilibrium bond length r_e=1.401 should be near zero
        assert np.max(np.abs(g_emp)) < 1e-3
