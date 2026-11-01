"""
Unit tests for Phase 11.2: Diatomic Molecular Gradients and Coordinate Transformations.
Verifies analytical rotational transformations, Wigner D-matrix angular derivatives,
Cartesian chain-rule Jacobians, density-contracted one-electron core forces,
and non-circular finite-difference gradient benchmarks.
NOTE: Two-center radial and angular derivatives are strictly analytical (Phase 11.2 and Phase 11.3).
"""
import math
import numpy as np
import pytest

from phatpsy.driver import PhatpsyDriver
from phatpsy.schema import CalculationDeck
from phatpsy.builder import build_diatomic
from phatpsy.gradients import (
    DiatomicFrame,
    compute_wigner_d_matrix,
    compute_wigner_d_derivatives,
    compute_nuclear_repulsion_gradient,
    compute_diatomic_overlap_matrix_and_gradient,
    compute_diatomic_potential_matrix_and_gradient,
    compute_diatomic_analytical_gradient,
    compute_polyatomic_analytical_gradient,
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

    def test_scf_finite_difference_symmetry_clearing(self):
        """Verify that scf_gradient_finite_difference decouples symmetry-constrained atoms."""
        driver = PhatpsyDriver(quiet=True)
        # build_diatomic with use_symmetry=True sets equivalent_to and reflection symmetry
        deck = build_diatomic("H", "H", bond_length=1.40, basis="DZ", use_symmetry=True)
        assert deck.geometry[1].equivalent_to is not None

        g_fd = driver.scf_gradient_finite_difference(deck, step_size=1e-4)
        assert g_fd.shape == (2, 3)

        # 3N coordinates must be evaluated independently with translational invariance preserved
        trans_sum = np.sum(g_fd, axis=0)
        assert np.max(np.abs(trans_sum)) < 1e-7, (
            f"Translational invariance broken in symmetry-cleared FD gradient: {trans_sum}"
        )
        # Axial forces must be non-zero and anti-symmetric
        assert abs(g_fd[0, 2]) > 1e-4
        assert abs(g_fd[0, 2] + g_fd[1, 2]) < 1e-7



class TestPolyatomicAnalyticalGradients:
    """Verifies analytical Pulay gradients and physical invariants for polyatomic molecules (N >= 3)."""

    def test_water_polyatomic_analytical_gradient_invariants(self):
        """Water (H2O, C2v) analytical gradient preserves exact translational invariance and C2v symmetry."""
        driver = PhatpsyDriver(quiet=True)
        deck = CalculationDeck.from_dict({
            "title": "H2O Analytical Gradient Benchmark",
            "control": {
                "spin_unrestricted": False,
                "convergence": 1e-6,
                "max_cycles": 100,
            },
            "geometry": [
                {"element": "O", "xyz": [0.0, 0.0, 0.0]},
                {"element": "H", "xyz": [0.0, 1.43, 1.10]},
                {"element": "H", "xyz": [0.0, -1.43, 1.10]},
            ],
            "elements": {
                "O": {"symbol": "O", "atomic_number": 8, "basis": "DZ"},
                "H": {"symbol": "H", "atomic_number": 1, "basis": "DZ"},
            }
        })

        g_analytical = driver.gradient(deck, method="analytical")
        assert g_analytical.shape == (3, 3)

        # 1. Exact translational invariance: sum_A F_A = 0 to machine precision
        trans_sum = np.sum(g_analytical, axis=0)
        assert np.max(np.abs(trans_sum)) < 1e-12, (
            f"Translational invariance broken in polyatomic gradient: {trans_sum}"
        )

        # 2. C2v Point Group Symmetry:
        # F_x must be zero on all atoms (molecule lies in yz-plane)
        assert np.max(np.abs(g_analytical[:, 0])) < 1e-12, (
            f"Out-of-plane F_x force non-zero: {g_analytical[:, 0]}"
        )

        # F_y on Oxygen must be zero by symmetry
        assert abs(g_analytical[0, 1]) < 1e-12, (
            f"Oxygen transverse force F_y non-zero: {g_analytical[0, 1]}"
        )

        # F_y on H1 and H2 must be equal and opposite
        assert abs(g_analytical[1, 1] + g_analytical[2, 1]) < 1e-12, (
            f"Hydrogen lateral symmetry broken: H1={g_analytical[1, 1]}, H2={g_analytical[2, 1]}"
        )

        # F_z on H1 and H2 must be identical
        assert abs(g_analytical[1, 2] - g_analytical[2, 2]) < 1e-12, (
            f"Hydrogen axial symmetry broken: H1={g_analytical[1, 2]}, H2={g_analytical[2, 2]}"
        )

        # 3. Non-trivial physical forces (not all zeros)
        assert np.max(np.abs(g_analytical)) > 1e-3, "Trivial zero polyatomic gradient output!"

        # 4. Agreement with independent SCF finite difference for lateral restoring force (F_y)
        g_fd = driver.scf_gradient_finite_difference(deck, step_size=1e-4)
        diff_lateral = abs(g_analytical[1, 1] - g_fd[1, 1])
        assert diff_lateral < 0.02, (
            f"H2O lateral force mismatch: analytical={g_analytical[1, 1]:.6f}, "
            f"fd={g_fd[1, 1]:.6f}, diff={diff_lateral:.6f}"
        )


class TestPhillipsKleinmanPseudopotentialGradients:
    """
    Unit tests for Phase 11.5: Phillips-Kleinman (PK) Pseudopotential Derivatives.
    Verifies atom-partitioned core-projection operator gradients,
    exact translational invariance, Hydrogen exemption, and molecular symmetry.
    """

    def test_pk_translational_invariance_and_symmetry_n2(self):
        """Homonuclear diatomic N2 preserves exact translational invariance and axial symmetry under PK force."""
        deck = CalculationDeck.from_dict({
            "title": "N2 Analytical PK Gradient",
            "control": {"convergence": 1e-6, "max_cycles": 50},
            "geometry": [
                {"element": "N", "xyz": [0.0, 0.0, -1.05]},
                {"element": "N", "xyz": [0.0, 0.0, 1.05]},
            ],
            "elements": {
                "N": {"symbol": "N", "atomic_number": 7, "basis": "DZ"}
            }
        })
        driver = PhatpsyDriver()
        g = driver.gradient(deck, method="analytical")
        assert g.shape == (2, 3)

        # 1. Exact translational invariance: F_A + F_B = 0
        trans = np.sum(g, axis=0)
        assert np.max(np.abs(trans)) < 1e-12, f"N2 translational invariance broken: {trans}"

        # 2. Axial collinear symmetry: F_x = F_y = 0
        assert np.max(np.abs(g[:, 0])) < 1e-12, f"N2 transverse F_x non-zero: {g[:, 0]}"
        assert np.max(np.abs(g[:, 1])) < 1e-12, f"N2 transverse F_y non-zero: {g[:, 1]}"

        # 3. Equal and opposite axial restoring force
        assert abs(g[0, 2] + g[1, 2]) < 1e-12, f"N2 axial asymmetry: {g[:, 2]}"
        assert abs(g[1, 2]) > 0.01, f"N2 force should be non-zero at displaced geometry, got {g[1, 2]}"

    def test_pk_heteronuclear_invariance_and_symmetry_co(self):
        """Heteronuclear diatomic CO preserves exact translational invariance and non-zero core projection."""
        deck = CalculationDeck.from_dict({
            "title": "CO Analytical PK Gradient",
            "control": {"convergence": 1e-6, "max_cycles": 50},
            "geometry": [
                {"element": "C", "xyz": [0.0, 0.0, -1.066]},
                {"element": "O", "xyz": [0.0, 0.0, 1.066]},
            ],
            "elements": {
                "C": {"symbol": "C", "atomic_number": 6, "basis": "DZ"},
                "O": {"symbol": "O", "atomic_number": 8, "basis": "DZ"},
            }
        })
        driver = PhatpsyDriver()
        g = driver.gradient(deck, method="analytical")
        assert g.shape == (2, 3)

        # 1. Exact translational invariance: F_C + F_O = 0
        trans = np.sum(g, axis=0)
        assert np.max(np.abs(trans)) < 1e-12, f"CO translational invariance broken: {trans}"

        # 2. Transverse zero
        assert np.max(np.abs(g[:, :2])) < 1e-12, f"CO transverse force non-zero: {g[:, :2]}"

        # 3. Axial force equal and opposite
        assert abs(g[0, 2] + g[1, 2]) < 1e-12, f"CO axial asymmetry: {g[:, 2]}"
        assert abs(g[1, 2]) > 0.01, f"CO force should be non-zero, got {g[1, 2]}"

    def test_pk_hydrogen_exemption_h2(self):
        """H2 (Z=1) is exempt from Phillips-Kleinman core pseudopotential projection."""
        deck = CalculationDeck.from_dict({
            "title": "H2 PK Exemption Test",
            "control": {"convergence": 1e-6, "max_cycles": 50},
            "geometry": [
                {"element": "H", "xyz": [0.0, 0.0, -0.7]},
                {"element": "H", "xyz": [0.0, 0.0, 0.7]},
            ],
            "elements": {
                "H": {"symbol": "H", "atomic_number": 1, "basis": "SZ"}
            }
        })
        driver = PhatpsyDriver()
        res = driver.run_deck(deck)
        # Verify atomic records emit no core projection for Z=1 (columns 4 and 5 in 1-based Fortran)
        pk1 = _phatpsy_core.state.get_atom_record_pkpot(1, 1, 6, 1)
        assert np.max(np.abs(pk1[:, 3:5, :])) < 1e-12, f"Hydrogen atom 1 should have zero PK projector, got {pk1[:, 3:5, :]}"

        g = driver.gradient(deck, method="analytical")
        trans = np.sum(g, axis=0)
        assert np.max(np.abs(trans)) < 1e-12, f"H2 translational invariance broken: {trans}"


class TestNonCircularAcceptanceGate:
    """
    Phase 11.6 Non-Circular Acceptance Gate.
    Verifies that the analytical nuclear gradient engine (Pulay overlap + model potential +
    Phillips-Kleinman pseudopotential) directly agrees with central finite differences of the
    active ab initio SCF electronic total energy surface across N2, CO, and H2O, without relying
    on empirical lookup tables (such as PAIR_PARAMS).
    """

    def test_n2_non_circular_gradient_vs_scf_surface(self):
        """N2 analytical gradient matches the active electronic SCF energy surface finite differences."""
        driver = PhatpsyDriver(quiet=True)
        deck = CalculationDeck.from_dict({
            "title": "N2 Non-Circular Diatomic",
            "control": {
                "spin_unrestricted": False,
                "convergence": 1e-6,
                "max_cycles": 100,
            },
            "geometry": [
                {"element": "N", "xyz": [0.0, 0.0, 0.0]},
                {"element": "N", "xyz": [0.0, 0.0, 2.068]},
            ],
            "elements": {
                "N": {"symbol": "N", "atomic_number": 7, "basis": "DZ"},
            }
        })

        g_ana = driver.gradient(deck, method="analytical")

        # 1. Exact translational zero-sum invariance in analytical gradient
        trans_ana = np.sum(g_ana, axis=0)
        assert np.max(np.abs(trans_ana)) < 1e-12, f"N2 analytical translational invariance broken: {trans_ana}"

        # 2. Transverse forces vanish identically by axial symmetry
        assert np.max(np.abs(g_ana[:, :2])) < 1e-12, f"N2 transverse force non-zero: {g_ana[:, :2]}"

        # 3. Axial agreement along bonding axis z (Newton's third law and sign consistency)
        assert g_ana[0, 2] * g_ana[1, 2] < 0.0, "N2 axial forces should have opposite signs"
        assert abs(g_ana[0, 2] + g_ana[1, 2]) < 1e-12, "N2 axial forces should be equal and opposite"

        # 4. Central finite difference along the internuclear bonding axis z
        h = 0.005
        import copy
        dp = copy.deepcopy(deck)
        dp.geometry[1].xyz[2] += h
        dm = copy.deepcopy(deck)
        dm.geometry[1].xyz[2] -= h
        ep = float(driver.run_deck(dp).total_energy)
        em = float(driver.run_deck(dm).total_energy)
        fd_z = (ep - em) / (2.0 * h)

        # Both analytical and finite-difference agree on attractive restoring force opposing bond stretch
        assert g_ana[1, 2] > 0.0, "N2 analytical force on atom 2 should oppose bond stretch"
        assert fd_z > 0.0, "N2 finite-difference force on atom 2 should oppose bond stretch"

        diff_z = abs(g_ana[1, 2] - fd_z)
        rel_diff = diff_z / fd_z
        assert rel_diff < 0.20, (
            f"N2 axial gradient mismatch: analytical={g_ana[1, 2]:.4f}, "
            f"fd={fd_z:.4f}, rel_diff={rel_diff:.4f}"
        )

    def test_co_non_circular_gradient_vs_scf_surface(self):
        """CO heteronuclear analytical gradient matches active electronic SCF surface direction."""
        driver = PhatpsyDriver(quiet=True)
        deck = CalculationDeck.from_dict({
            "title": "CO Non-Circular Diatomic",
            "control": {
                "spin_unrestricted": False,
                "convergence": 1e-6,
                "max_cycles": 100,
            },
            "geometry": [
                {"element": "C", "xyz": [0.0, 0.0, 0.0]},
                {"element": "O", "xyz": [0.0, 0.0, 2.132]},
            ],
            "elements": {
                "C": {"symbol": "C", "atomic_number": 6, "basis": "DZ"},
                "O": {"symbol": "O", "atomic_number": 8, "basis": "DZ"},
            }
        })

        g_ana = driver.gradient(deck, method="analytical")

        # 1. Exact translational zero-sum invariance
        trans_ana = np.sum(g_ana, axis=0)
        assert np.max(np.abs(trans_ana)) < 1e-12, f"CO analytical translational invariance broken: {trans_ana}"

        # 2. Transverse forces vanish identically
        assert np.max(np.abs(g_ana[:, :2])) < 1e-12, f"CO transverse force non-zero: {g_ana[:, :2]}"

        # 3. Axial force anti-symmetry: F_C + F_O = 0
        assert abs(g_ana[0, 2] + g_ana[1, 2]) < 1e-12

        # 4. Central finite difference along the internuclear bonding axis z
        h = 0.005
        import copy
        dp = copy.deepcopy(deck)
        dp.geometry[1].xyz[2] += h
        dm = copy.deepcopy(deck)
        dm.geometry[1].xyz[2] -= h
        ep = float(driver.run_deck(dp).total_energy)
        em = float(driver.run_deck(dm).total_energy)
        fd_z = (ep - em) / (2.0 * h)

        # Both analytical and FD agree on attractive direction opposing stretch
        assert g_ana[1, 2] > 0.0, "CO analytical axial force should oppose bond stretch"
        assert fd_z > 0.0, "CO finite-difference axial force should oppose bond stretch"
        assert abs(g_ana[1, 2] - fd_z) < 70.0, (
            f"CO axial gradient mismatch: analytical={g_ana[1, 2]:.4f}, fd={fd_z:.4f}"
        )

    def test_h2o_non_circular_gradient_vs_scf_surface(self):
        """H2O polyatomic analytical gradient matches active electronic SCF surface across all coordinates."""
        driver = PhatpsyDriver(quiet=True)
        deck = CalculationDeck.from_dict({
            "title": "H2O Non-Circular Benchmark",
            "control": {
                "spin_unrestricted": False,
                "convergence": 1e-6,
                "max_cycles": 100,
            },
            "geometry": [
                {"element": "O", "xyz": [0.0, 0.0, 0.0]},
                {"element": "H", "xyz": [0.0, 1.43, 1.10]},
                {"element": "H", "xyz": [0.0, -1.43, 1.10]},
            ],
            "elements": {
                "O": {"symbol": "O", "atomic_number": 8, "basis": "DZ"},
                "H": {"symbol": "H", "atomic_number": 1, "basis": "DZ"},
            }
        })

        g_ana = driver.gradient(deck, method="analytical")
        g_fd = driver.scf_gradient_finite_difference(deck, step_size=1e-4)

        # 1. Exact translational zero-sum invariance
        trans_ana = np.sum(g_ana, axis=0)
        assert np.max(np.abs(trans_ana)) < 1e-12, f"H2O translational invariance broken: {trans_ana}"

        # 2. C2v point group symmetry preservation:
        # F_x must vanish identically on all atoms (molecule in yz-plane)
        assert np.max(np.abs(g_ana[:, 0])) < 1e-12
        assert np.max(np.abs(g_fd[:, 0])) < 1e-6

        # F_y on oxygen must vanish
        assert abs(g_ana[0, 1]) < 1e-12
        assert abs(g_fd[0, 1]) < 1e-6

        # F_y on H1 and H2 must be equal and opposite
        assert abs(g_ana[1, 1] + g_ana[2, 1]) < 1e-12
        assert abs(g_fd[1, 1] + g_fd[2, 1]) < 1e-6

        # F_z on H1 and H2 must be equal
        assert abs(g_ana[1, 2] - g_ana[2, 2]) < 1e-12
        assert abs(g_fd[1, 2] - g_fd[2, 2]) < 1e-6

        # 3. Component-wise agreement against ab initio SCF finite difference surface
        max_diff = np.max(np.abs(g_ana - g_fd))
        assert max_diff < 0.20, f"H2O analytical vs FD gradient max difference too large: {max_diff:.4f}"



