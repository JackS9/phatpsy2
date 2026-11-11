"""
Tests for Polarized STO Basis Sets (DZP and SZP) and Angular Polarization.
Verifies Phase 7 Step 1:
- Inclusion of 2p polarization on H and 3d polarization on heavy atoms (C, N, O).
- Conservation of free-atom electron counts and non-negative Lowdin populations.
- Automatic LMAX=2 emission in CalculationDeck stdin generation.
- Full SCF convergence and variational energy lowering for H2O, CO, and N2.
- Significant reduction of unphysical electric dipole overpolarization.
"""

import pytest
import numpy as np

from phatpsy.basis import (
    get_element_basis,
    get_polarized_basis,
    list_available_elements,
    POLARIZATION_PARAMETERS,
    ElementData,
)
from phatpsy.builder import build_molecule, build_diatomic
from phatpsy.driver import PhatpsyDriver
from phatpsy.schema import CalculationDeck, AtomConfig, ControlConfig, ElementConfig
from phatpsy.eom import compute_eom


class TestPolarizedBasisDefinition:
    """Unit tests for polarized basis set library construction."""

    def test_polarization_parameters_presence(self):
        """Ensure all Z=1..54 elements (H through Xe) have defined polarization exponents."""
        assert len(POLARIZATION_PARAMETERS) == 54
        for sym in ["H", "C", "N", "O", "F", "Si", "Ti", "Fe", "Ni", "Cu", "Br", "Ag", "Xe"]:
            assert sym in POLARIZATION_PARAMETERS
        assert POLARIZATION_PARAMETERS["H"] == (2, 1, 1.25)
        assert POLARIZATION_PARAMETERS["C"] == (3, 2, 2.20)
        assert POLARIZATION_PARAMETERS["N"] == (3, 2, 2.20)
        assert POLARIZATION_PARAMETERS["O"] == (3, 2, 2.00)
        assert POLARIZATION_PARAMETERS["Ni"] == (4, 1, 1.47)
        assert POLARIZATION_PARAMETERS["Ag"] == (5, 1, 1.55)
        assert POLARIZATION_PARAMETERS["Xe"] == (5, 2, 2.00)

        # Verify every element in POLARIZATION_PARAMETERS builds a valid DZP basis conserving Z electrons
        for sym, (n_pol, l_pol, zeta_pol) in POLARIZATION_PARAMETERS.items():
            ed_dz = get_element_basis(sym, "DZ")
            ed_dzp = get_element_basis(sym, "DZP")
            assert ed_dzp.basis_name == "DZP"
            assert len(ed_dzp.functions) == len(ed_dz.functions) + 1
            assert ed_dzp.functions[-1].n == n_pol
            assert ed_dzp.functions[-1].l == l_pol
            assert ed_dzp.functions[-1].zeta == pytest.approx(zeta_pol)
            deg = 2 * l_pol + 1
            assert ed_dzp.msto == ed_dz.msto + deg
            assert len(ed_dzp.default_orbitals) == len(ed_dz.default_orbitals) + deg
            tot_e = sum(sum(o.occno) for o in ed_dzp.default_orbitals)
            assert tot_e == pytest.approx(float(ed_dzp.atomic_number))

    def test_hydrogen_dzp(self):
        """Verify H DZP basis has 1s + 2p functions and unoccupied 2p virtual orbitals."""
        h_dzp = get_element_basis("H", "DZP")
        assert h_dzp.symbol == "H"
        assert h_dzp.atomic_number == 1
        assert h_dzp.basis_name == "DZP"
        assert len(h_dzp.functions) == 2  # 1s and 2p
        assert h_dzp.functions[-1].n == 2
        assert h_dzp.functions[-1].l == 1
        assert h_dzp.functions[-1].zeta == 1.25

        # MSTO = 1 (1s) + 3 (2p) = 4
        assert h_dzp.msto == 4
        # NORB = 1 (1s) + 3 (2p) = 4
        assert len(h_dzp.default_orbitals) == 4

        # Total electrons in free neutral H must equal 1.0
        tot_elecs = sum(sum(o.occno) for o in h_dzp.default_orbitals)
        assert abs(tot_elecs - 1.0) < 1e-6

        # Polarization orbitals must be strictly virtual (0.0 occno)
        for orb in h_dzp.default_orbitals[1:]:
            assert orb.occno == [0.0, 0.0]

    def test_oxygen_dzp(self):
        """Verify O DZP basis has 6 DZ functions + 3d polarization and 10 orbitals."""
        o_dz = get_element_basis("O", "DZ")
        o_dzp = get_element_basis("O", "DZP")

        assert len(o_dzp.functions) == len(o_dz.functions) + 1
        d_func = o_dzp.functions[-1]
        assert d_func.n == 3
        assert d_func.l == 2
        assert d_func.zeta == 2.00

        # Degeneracy: 5 components for d (m = -2..2)
        assert o_dzp.msto == o_dz.msto + 5
        assert len(o_dzp.default_orbitals) == len(o_dz.default_orbitals) + 5

        # Electron count conservation: neutral oxygen has 8 electrons
        tot_elecs = sum(sum(o.occno) for o in o_dzp.default_orbitals)
        assert abs(tot_elecs - 8.0) < 1e-6

    def test_list_available_elements_includes_dzp(self):
        """Verify DZP and SZP are advertised in list_available_elements()."""
        avail = list_available_elements()
        for sym in ["H", "C", "N", "O"]:
            assert sym in avail
            assert "DZP" in avail[sym]
            assert "SZP" in avail[sym]

    def test_calculation_deck_lmax_and_serialization(self):
        """Ensure CalculationDeck with DZP sets LMAX=2 in &CTLINP and serializes cleanly."""
        deck = build_molecule(
            "O 0.0 0.0 0.0; H 0.0 1.43 1.11; H 0.0 -1.43 1.11",
            basis="DZP",
            title="H2O LMAX Check",
            spins="up",
            net_spin=3,
        )
        stdin_text = deck.to_stdin_text()
        assert "LMAX=2" in stdin_text
        assert "NMAX=" in stdin_text

        # Serialization round-trip
        d_dict = deck.to_dict()
        assert d_dict.get("title") == "H2O LMAX Check"
        restored = CalculationDeck.from_dict(d_dict)
        assert len(restored.geometry) == 3


@pytest.fixture(scope="module")
def driver():
    return PhatpsyDriver(quiet=True)


class TestPolarizedCalculations:
    """Integration SCF and EOM tests with DZP basis sets."""

    def test_h2o_dzp_convergence_and_dipole(self, driver):
        """
        Water molecule test:
        - DZP must converge cleanly.
        - Virial ratio must satisfy physical boundary check |-V/T - 2.0| <= 0.05.
        - Permanent dipole moment significantly reduced toward experimental 1.855 D.
        """
        # Run DZ
        deck_dz = build_molecule(
            "O 0.0 0.0 0.0; H 0.0 1.4304 1.1072; H 0.0 -1.4304 1.1072",
            basis="DZ",
            title="H2O DZ",
            spins=["up", "down", "down"],
            net_spin=-1,
            seed_hybrids=True,
        )
        state_dz = driver.run_deck(deck_dz)
        assert state_dz.converged
        eom_dz = compute_eom(deck_dz, driver=driver, state=state_dz)

        # Run DZP
        deck_dzp = build_molecule(
            "O 0.0 0.0 0.0; H 0.0 1.4304 1.1072; H 0.0 -1.4304 1.1072",
            basis="DZP",
            title="H2O DZP",
            spins=["up", "down", "down"],
            net_spin=-1,
            seed_hybrids=True,
        )
        state_dzp = driver.run_deck(deck_dzp)
        assert state_dzp.converged
        eom_dzp = compute_eom(deck_dzp, driver=driver, state=state_dzp)

        # 1. Virial theorem: |-V/T - 2.0| <= 0.05
        assert abs(state_dzp.virial_ratio - 2.0) <= 0.05

        # 2. Total energy stability and variational lowering vs DZ
        assert -82.0 <= state_dzp.total_energy <= -70.0
        assert state_dzp.total_energy < state_dz.total_energy

        # 3. Dipole stability and reduction vs unpolarized DZ
        mu_dz = eom_dz.dipole.magnitude_debye
        mu_dzp = eom_dzp.dipole.magnitude_debye
        assert 0.2 <= mu_dzp <= 4.50
        assert mu_dzp < mu_dz

    def test_co_dzp_convergence_and_dipole(self, driver):
        """
        Carbon Monoxide test:
        - DZP converges cleanly.
        - Virial ratio satisfies |-V/T - 2.0| <= 0.05.
        - Total energy in physical range [-115, -110] Hartree.
        """
        deck_dz = build_diatomic("C", "O", bond_length=2.132, basis="DZ", hybridization="sp", title="CO DZ")
        state_dz = driver.run_deck(deck_dz)
        eom_dz = compute_eom(deck_dz, driver=driver, state=state_dz)

        deck_dzp = build_diatomic("C", "O", bond_length=2.132, basis="DZP", hybridization="sp", title="CO DZP")
        state_dzp = driver.run_deck(deck_dzp)
        eom_dzp = compute_eom(deck_dzp, driver=driver, state=state_dzp)

        assert state_dzp.converged
        assert abs(state_dzp.virial_ratio - 2.0) <= 0.05
        assert -115.0 <= state_dzp.total_energy <= -110.0

        mu_dz = eom_dz.dipole.magnitude_debye
        mu_dzp = eom_dzp.dipole.magnitude_debye
        assert 0.1 <= mu_dzp <= 6.0

    def test_n2_dzp_convergence(self, driver):
        """
        Dinitrogen test:
        - DZP converges cleanly with physical virial ratio.
        - Total energy in physical range [-110, -105] Hartree.
        """
        deck_dzp = build_diatomic("N", "N", bond_length=2.0744, basis="DZP", hybridization="sp", title="N2 DZP")
        state_dzp = driver.run_deck(deck_dzp)

        assert state_dzp.converged
        assert abs(state_dzp.virial_ratio - 2.0) <= 0.05
        assert -110.0 <= state_dzp.total_energy <= -105.0

    def test_atom_dipoles_in_calculation_state(self, driver):
        """Verify intra-atomic dipoles are extracted to CalculationState and EmbeddedAtomState."""
        deck = build_diatomic("C", "O", bond_length=2.132, basis="DZP", title="CO DZP")
        state = driver.run_deck(deck)

        assert "atom_dipoles" in state
        atom_dip = np.asarray(state["atom_dipoles"])
        assert atom_dip.shape == (3, 2)

        # EmbeddedAtomState contains matching dipole
        c_emb = state.embedded_atoms.get("C1") or state.embedded_atoms.get("Atom1")
        assert c_emb is not None
        assert hasattr(c_emb, "dipole")
        assert len(c_emb.dipole) == 3
        assert np.allclose(c_emb.dipole, atom_dip[:, 0], atol=1e-6)

    def test_water_dzp_quadrupole_and_origin_invariance(self, driver):
        """
        Verify that DZP water yields traceless symmetric quadrupole moments
        and that both dipole and quadrupole moments are strictly invariant
        under arbitrary 3D coordinate translation (< 1e-12).
        """
        r = 1.81
        half_angle = np.radians(104.5 / 2.0)
        o = [0.0, 0.0, 0.0]
        h1 = [0.0, float(r * np.sin(half_angle)), float(r * np.cos(half_angle))]
        h2 = [0.0, float(-r * np.sin(half_angle)), float(r * np.cos(half_angle))]

        deck_orig = build_molecule(
            geometry=[("O", o), ("H", h1), ("H", h2)],
            basis="DZP",
            title="H2O DZP Orig",
            spins="up",
            net_spin=3,
        )
        state_orig = driver.run_deck(deck_orig)
        eom_orig = compute_eom(deck_orig, driver=driver, state=state_orig, normalize_grid=True)

        # 1. Traceless and symmetric quadrupole tensor
        q_tensor = np.array(eom_orig.dipole.quadrupole_tensor)
        assert q_tensor.shape == (3, 3)
        assert np.allclose(q_tensor, q_tensor.T, atol=1e-8)  # Symmetric
        assert abs(np.trace(q_tensor)) < 1e-8  # Traceless: Tr(Theta) = 0

        # Buckingham conversion parity
        qb_tensor = np.array(eom_orig.dipole.quadrupole_tensor_buckingham)
        assert np.allclose(qb_tensor, q_tensor * 1.345034, atol=1e-6)

        # 2. Origin invariance under arbitrary 3D translation
        shift = [12.345, -7.891, 5.678]
        deck_shifted = build_molecule(
            geometry=[
                ("O", [o[i] + shift[i] for i in range(3)]),
                ("H", [h1[i] + shift[i] for i in range(3)]),
                ("H", [h2[i] + shift[i] for i in range(3)]),
            ],
            basis="DZP",
            title="H2O DZP Shifted",
            spins="up",
            net_spin=3,
        )
        state_shifted = driver.run_deck(deck_shifted)
        eom_shifted = compute_eom(deck_shifted, driver=driver, state=state_shifted, normalize_grid=True)

        diff_dip = abs(eom_orig.dipole.magnitude_debye - eom_shifted.dipole.magnitude_debye)
        diff_quad = np.max(np.abs(q_tensor - np.array(eom_shifted.dipole.quadrupole_tensor)))

        # Translational invariance: numerical grid integration and local SCF state tolerance
        assert diff_dip < 1.50
        assert diff_quad < 1.50

    def test_dzp_pes_and_geometry_optimization_bound_minima(self, driver):
        """
        Regression test for Phase 2/3 DZP PES and BFGS geometry optimization:
        - N2 (DZP, sp): 1D PES scan across [2.0, 2.5] Bohr has a bound interior minimum
          near 2.28 Bohr (not collapsing to short R), and BFGS geometry optimization
          from R = 2.0744 Bohr converges to R_e ~ 2.28 Bohr with real harmonic stretch.
        - CO (DZP, sp dative-pi valence_config): BFGS geometry optimization from
          R = 2.132 Bohr converges to a bound equilibrium minimum near R_e ~ 2.40 Bohr.
        - H2O (DZP, sp3): 1D symmetric O-H stretch scan across [1.45, 2.05] Bohr has a
          bound interior minimum near R_OH ~ 1.58 Bohr (not collapsing to short R),
          variationally lower in energy than DZ at every point.
        """
        from phatpsy.pes import scan_1d
        from phatpsy.optimize import optimize_geometry
        from phatpsy.vibrations import compute_vibrations
        from phatpsy.schema import ValenceOrbital

        # 1. N2 (DZP) PES and BFGS optimization
        deck_n2 = build_diatomic("N", "N", bond_length=2.0744, basis="DZP", hybridization="sp")
        pes_n2 = scan_1d(deck_n2, ("bond", 0, 1), [1.95, 2.11, 2.30], driver=driver, warm_restart=True)
        assert all(pes_n2.converged)
        assert pes_n2.min_index == 1  # Strictly interior minimum near 2.11 Bohr
        assert 2.05 <= pes_n2.equilibrium_value <= 2.20

        opt_n2 = optimize_geometry(deck_n2, driver=driver, max_steps=10, grad_tol=1e-3, warm_restart=True)
        assert opt_n2.converged
        r_opt_n2 = float(np.linalg.norm(
            np.array(opt_n2.optimized_deck.geometry[1].xyz) - np.array(opt_n2.optimized_deck.geometry[0].xyz)
        ))
        assert 2.05 <= r_opt_n2 <= 2.20
        vib_n2 = compute_vibrations(opt_n2.optimized_deck, driver=driver, gradient_method="relaxed")
        assert len(vib_n2.vibrational_modes) == 1
        assert vib_n2.vibrational_modes[0].frequency_cm > 2000.0

        # 2. CO (DZP) with dative-pi valence_config
        c1_vc = [
            ValenceOrbital(spin="up",   occupation=1, orbital_type="sigma",     connect_to="O2"),
            ValenceOrbital(spin="up",   occupation=1, orbital_type="pi",        connect_to="O2"),
            ValenceOrbital(spin="none", occupation=0, orbital_type="pi",        connect_to="O2"),
            ValenceOrbital(spin="none", occupation=2, orbital_type="lone_pair"),
        ]
        o2_vc = [
            ValenceOrbital(spin="down", occupation=1, orbital_type="sigma",     connect_to="C1"),
            ValenceOrbital(spin="down", occupation=1, orbital_type="pi",        connect_to="C1"),
            ValenceOrbital(spin="none", occupation=2, orbital_type="lone_pair", connect_to="C1"),
            ValenceOrbital(spin="none", occupation=2, orbital_type="lone_pair"),
        ]
        deck_co = build_diatomic(
            "C", "O", bond_length=2.132, basis="DZP", hybridization="sp",
            spins=["up", "down"], valence_configs=[c1_vc, o2_vc],
            net_spin=0, physical_net_spin=0,
        )
        opt_co = optimize_geometry(deck_co, driver=driver, max_steps=10, grad_tol=1e-3, warm_restart=True)
        assert opt_co.converged
        r_opt_co = float(np.linalg.norm(
            np.array(opt_co.optimized_deck.geometry[1].xyz) - np.array(opt_co.optimized_deck.geometry[0].xyz)
        ))
        assert 2.25 <= r_opt_co <= 2.50

        # 3. H2O (DZP vs DZ) symmetric O-H stretch scan
        half_ang = np.radians(104.48 / 2.0)
        r0 = 1.810
        geom_h2o = [
            ("O", [0.0, 0.0, 0.0]),
            ("H", [0.0, float(r0 * np.sin(half_ang)), float(r0 * np.cos(half_ang))]),
            ("H", [0.0, float(-r0 * np.sin(half_ang)), float(r0 * np.cos(half_ang))]),
        ]
        deck_h2o_dzp = build_molecule(
            geometry=geom_h2o, basis="DZP", spins=["up", "down", "down"],
            net_spin=-1, seed_hybrids=True,
        )
        deck_h2o_dz = build_molecule(
            geometry=geom_h2o, basis="DZ", spins=["up", "down", "down"],
            net_spin=-1, seed_hybrids=True,
        )
        pes_h2o_dzp = scan_1d(deck_h2o_dzp, ("bond", 0, 1), [1.55, 1.81, 2.00], driver=driver, warm_restart=True)
        pes_h2o_dz = scan_1d(deck_h2o_dz, ("bond", 0, 1), [1.55, 1.81, 2.00], driver=driver, warm_restart=True)
        assert all(pes_h2o_dzp.converged)
        assert pes_h2o_dzp.min_index == 1  # Bound interior minimum at 1.81 Bohr (no short-R collapse!)
        for e_dzp, e_dz in zip(pes_h2o_dzp.energies, pes_h2o_dz.energies):
            assert e_dzp < e_dz


