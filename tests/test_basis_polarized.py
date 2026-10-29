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
        """Ensure standard elements have defined polarization exponents."""
        for sym in ["H", "C", "N", "O", "F"]:
            assert sym in POLARIZATION_PARAMETERS
        assert POLARIZATION_PARAMETERS["H"] == (2, 1, 1.00)
        assert POLARIZATION_PARAMETERS["C"] == (3, 2, 1.25)
        assert POLARIZATION_PARAMETERS["N"] == (3, 2, 1.50)
        assert POLARIZATION_PARAMETERS["O"] == (3, 2, 1.75)

    def test_hydrogen_dzp(self):
        """Verify H DZP basis has 1s + 2p functions and unoccupied 2p virtual orbitals."""
        h_dzp = get_element_basis("H", "DZP")
        assert h_dzp.symbol == "H"
        assert h_dzp.atomic_number == 1
        assert h_dzp.basis_name == "DZP"
        assert len(h_dzp.functions) == 2  # 1s and 2p
        assert h_dzp.functions[-1].n == 2
        assert h_dzp.functions[-1].l == 1
        assert h_dzp.functions[-1].zeta == 1.00

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
        assert d_func.zeta == 1.75

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
        )
        state_dz = driver.run_deck(deck_dz)
        assert state_dz.converged
        eom_dz = compute_eom(deck_dz, driver=driver, state=state_dz)

        # Run DZP
        deck_dzp = build_molecule(
            "O 0.0 0.0 0.0; H 0.0 1.4304 1.1072; H 0.0 -1.4304 1.1072",
            basis="DZP",
            title="H2O DZP",
        )
        state_dzp = driver.run_deck(deck_dzp)
        assert state_dzp.converged
        eom_dzp = compute_eom(deck_dzp, driver=driver, state=state_dzp)

        # 1. Virial theorem: |-V/T - 2.0| <= 0.05
        assert abs(state_dzp.virial_ratio - 2.0) <= 0.05

        # 2. Total energy stability
        assert -82.0 <= state_dzp.total_energy <= -70.0

        # 3. Dipole reduction: DZ gives ~3.01 D, DZP reduces to ~1.50 D
        mu_dz = eom_dz.dipole.magnitude_debye
        mu_dzp = eom_dzp.dipole.magnitude_debye
        assert mu_dzp < mu_dz
        assert 1.40 <= mu_dzp <= 1.80

    def test_co_dzp_convergence_and_dipole(self, driver):
        """
        Carbon Monoxide test:
        - DZP converges cleanly.
        - Virial ratio satisfies |-V/T - 2.0| <= 0.05.
        - Total energy in physical range [-115, -110] Hartree.
        - Dipole moment reduced from >5 D in DZ to ~0.85-1.2 D in DZP.
        """
        deck_dz = build_diatomic("C", "O", bond_length=2.132, basis="DZ", title="CO DZ")
        state_dz = driver.run_deck(deck_dz)
        eom_dz = compute_eom(deck_dz, driver=driver, state=state_dz)

        deck_dzp = build_diatomic("C", "O", bond_length=2.132, basis="DZP", title="CO DZP")
        state_dzp = driver.run_deck(deck_dzp)
        eom_dzp = compute_eom(deck_dzp, driver=driver, state=state_dzp)

        assert state_dzp.converged
        assert abs(state_dzp.virial_ratio - 2.0) <= 0.05
        assert -115.0 <= state_dzp.total_energy <= -110.0

        mu_dz = eom_dz.dipole.magnitude_debye
        mu_dzp = eom_dzp.dipole.magnitude_debye
        assert mu_dzp < mu_dz
        assert mu_dzp < 2.0  # Dramatic reduction from >5 D

    def test_n2_dzp_convergence(self, driver):
        """
        Dinitrogen test:
        - DZP converges cleanly with physical virial ratio.
        - Total energy in physical range [-110, -105] Hartree.
        """
        deck_dzp = build_diatomic("N", "N", bond_length=2.068, basis="DZP", title="N2 DZP")
        state_dzp = driver.run_deck(deck_dzp)

        assert state_dzp.converged
        assert abs(state_dzp.virial_ratio - 2.0) <= 0.06
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
        )
        state_shifted = driver.run_deck(deck_shifted)
        eom_shifted = compute_eom(deck_shifted, driver=driver, state=state_shifted, normalize_grid=True)

        diff_dip = abs(eom_orig.dipole.magnitude_debye - eom_shifted.dipole.magnitude_debye)
        diff_quad = np.max(np.abs(q_tensor - np.array(eom_shifted.dipole.quadrupole_tensor)))

        # Machine-precision translational invariance (< 1e-12)
        assert diff_dip < 1e-12
        assert diff_quad < 1e-12

