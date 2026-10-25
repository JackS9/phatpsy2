# phatpsy/eom.py
"""
Electronic, Optical, and Magnetic (EOM) Properties Module for PHATPSY 2.0.

Provides rigorous evaluation of:
1. Permanent Electric Dipole Moment:
   - Nuclear and electronic dipole vectors in atomic units (e*a0) and Debyes.
   - Reference center of nuclear charge and dipole magnitude.
2. Electronic Excitations & UV-Vis Absorption Spectroscopy:
   - Single-particle transitions from occupied to virtual orbitals across spin channels.
   - Transition energies (Hartree, eV), absorption wavelengths (nm), transition dipole moments (a.u.),
     and dimensionless oscillator strengths (f_osc).
   - Continuous simulated UV-Vis absorption profile with Gaussian broadening.
3. Optical Polarizability Tensor:
   - Frequency-dependent dynamic polarizability alpha(omega) and static polarizability alpha(0)
     via Sum-Over-States (SOS) perturbation theory.
   - Isotropic mean polarizability (a.u., Angstrom^3) and optical anisotropy.
4. Magnetic Response Properties (Susceptibility):
   - Langevin-Van Vleck diamagnetic susceptibility tensor from electronic second moments.
   - Van Vleck temperature-independent paramagnetic susceptibility tensor from orbital angular momentum.
   - Curie paramagnetic susceptibility for open-shell spin states (S > 0).
   - Total magnetic susceptibility in atomic units and cgs molar units (cm^3/mol).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import json
import math
import numpy as np

from phatpsy.driver import PhatpsyDriver
from phatpsy.schema import CalculationDeck, AtomConfig
from phatpsy.state import CalculationState
from phatpsy.volumetric import prepare_grid_geometry, evaluate_real_sto


# Physical constants and unit conversions (CODATA 2018 / IUPAC)
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.509474
BOHR_TO_ANGSTROM = 0.529177210903
AU_TO_DEBYE = 2.54174623  # 1 a.u. dipole = e * a0 = 2.541746 Debye
AU_TO_ANG3 = BOHR_TO_ANGSTROM**3  # 0.148184711 Angstrom^3
SPEED_OF_LIGHT_AU = 137.035999084  # c in atomic units
AU_TO_CGS_MOLAR_SUSCEPTIBILITY = 0.792744e-6  # cm^3/mol per a.u. of chi
BOLTZMANN_AU_PER_K = 3.166811563e-6  # Hartree / K
PLANCK_CONSTANT_HC_EV_NM = 1239.84193  # eV * nm


@dataclass
class ElectronicTransition:
    """An individual electronic transition between occupied and virtual molecular orbitals."""
    index: int
    spin: str  # "alpha" or "beta"
    from_orbital: int  # 1-based index
    to_orbital: int    # 1-based index
    energy_hartree: float
    energy_ev: float
    wavelength_nm: float
    oscillator_strength: float
    transition_dipole: List[float]  # [mux, muy, muz] in a.u.
    transition_dipole_norm: float   # in a.u.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "spin": self.spin,
            "from_orbital": self.from_orbital,
            "to_orbital": self.to_orbital,
            "energy_hartree": round(float(self.energy_hartree), 8),
            "energy_ev": round(float(self.energy_ev), 4),
            "wavelength_nm": round(float(self.wavelength_nm), 2),
            "oscillator_strength": round(float(self.oscillator_strength), 6),
            "transition_dipole_au": [round(float(c), 6) for c in self.transition_dipole],
            "transition_dipole_norm_au": round(float(self.transition_dipole_norm), 6),
        }


@dataclass
class DipoleMoment:
    """Permanent molecular electric dipole moment."""
    total: List[float]         # [mux, muy, muz] in atomic units (e * a0)
    total_debye: List[float]   # in Debye
    magnitude_au: float
    magnitude_debye: float
    nuclear: List[float]       # nuclear dipole in a.u.
    electronic: List[float]    # electronic dipole in a.u.
    center_of_charge: List[float]  # reference origin in Bohr

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_au": [round(float(c), 6) for c in self.total],
            "total_debye": [round(float(c), 6) for c in self.total_debye],
            "magnitude_au": round(float(self.magnitude_au), 6),
            "magnitude_debye": round(float(self.magnitude_debye), 4),
            "nuclear_au": [round(float(c), 6) for c in self.nuclear],
            "electronic_au": [round(float(c), 6) for c in self.electronic],
            "center_of_charge_bohr": [round(float(c), 6) for c in self.center_of_charge],
        }


@dataclass
class PolarizabilityTensor:
    """Static and dynamic optical polarizability tensors."""
    static_tensor: List[List[float]]  # 3x3 matrix in atomic units (a0^3)
    mean_static_au: float
    mean_static_ang3: float
    anisotropy_au: float
    dynamic_tensors: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "static_tensor_au": [[round(float(v), 4) for v in row] for row in self.static_tensor],
            "mean_static_au": round(float(self.mean_static_au), 4),
            "mean_static_ang3": round(float(self.mean_static_ang3), 4),
            "anisotropy_au": round(float(self.anisotropy_au), 4),
            "dynamic_tensors": self.dynamic_tensors,
        }


@dataclass
class MagneticSusceptibility:
    """Ground-state magnetic susceptibility tensor."""
    diamagnetic_tensor: List[List[float]]    # 3x3 in a.u.
    paramagnetic_tensor: List[List[float]]   # 3x3 in a.u.
    curie_scalar: float                      # Curie term for S > 0 in a.u.
    total_tensor: List[List[float]]          # 3x3 in a.u.
    mean_susceptibility_au: float
    molar_susceptibility_cgs: float          # in cm^3/mol (or 10^-6 emu/mol)
    temperature_k: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diamagnetic_tensor_au": [[round(float(v), 8) for v in row] for row in self.diamagnetic_tensor],
            "paramagnetic_tensor_au": [[round(float(v), 8) for v in row] for row in self.paramagnetic_tensor],
            "curie_scalar_au": round(float(self.curie_scalar), 8),
            "total_tensor_au": [[round(float(v), 8) for v in row] for row in self.total_tensor],
            "mean_susceptibility_au": round(float(self.mean_susceptibility_au), 8),
            "molar_susceptibility_cgs": round(float(self.molar_susceptibility_cgs), 4),
            "temperature_k": round(float(self.temperature_k), 2),
        }


@dataclass
class EOMResult:
    """Container for Electronic, Optical, and Magnetic (EOM) response properties."""
    dipole: DipoleMoment
    transitions: List[ElectronicTransition]
    polarizability: PolarizabilityTensor
    magnetic_susceptibility: MagneticSusceptibility
    deck: Optional[CalculationDeck] = None
    state: Optional[CalculationState] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes EOM result into a structured dictionary."""
        return {
            "dipole_moment": self.dipole.to_dict(),
            "polarizability": self.polarizability.to_dict(),
            "magnetic_susceptibility": self.magnetic_susceptibility.to_dict(),
            "num_transitions": len(self.transitions),
            "transitions": [t.to_dict() for t in self.transitions],
        }

    def save_json(self, filepath: Union[str, Path]) -> None:
        """Saves EOM results to a JSON file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def generate_uvvis_spectrum(
        self,
        wmin_nm: float = 100.0,
        wmax_nm: float = 800.0,
        num_points: int = 1000,
        fwhm_nm: float = 20.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates simulated UV-Vis absorption spectrum with Gaussian peak broadening.

        Parameters
        ----------
        wmin_nm : float
            Minimum wavelength in nm (default: 100.0).
        wmax_nm : float
            Maximum wavelength in nm (default: 800.0).
        num_points : int
            Number of wavelength grid points (default: 1000).
        fwhm_nm : float
            Full-width at half-maximum broadening in nm (default: 20.0).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]:
            (wavelengths_nm, extinction_arbitrary_units)
        """
        wavelengths = np.linspace(wmin_nm, wmax_nm, num_points)
        intensity = np.zeros_like(wavelengths)
        sigma = fwhm_nm / (2.0 * math.sqrt(2.0 * math.log(2.0)))

        for t in self.transitions:
            if t.oscillator_strength > 1e-6:
                diff = wavelengths - t.wavelength_nm
                intensity += t.oscillator_strength * np.exp(-0.5 * (diff / sigma)**2) / (sigma * math.sqrt(2.0 * math.pi))

        return wavelengths, intensity

    def plot_uvvis(
        self,
        output_path: Optional[Union[str, Path]] = None,
        wmin_nm: float = 120.0,
        wmax_nm: float = 700.0,
        fwhm_nm: float = 20.0,
        title: Optional[str] = None,
    ) -> Any:
        """
        Plots simulated UV-Vis absorption spectrum with stick oscillator strengths.
        """
        import matplotlib.pyplot as plt

        wl, intens = self.generate_uvvis_spectrum(wmin_nm, wmax_nm, num_points=1000, fwhm_nm=fwhm_nm)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        ax.plot(wl, intens, color="#2563eb", lw=2, label="Gaussian Broadened Profile")
        ax.fill_between(wl, 0, intens, color="#3b82f6", alpha=0.25)

        # Draw stick transitions
        max_intens = float(np.max(intens)) if len(intens) > 0 and np.max(intens) > 0 else 1.0
        for t in self.transitions:
            if wmin_nm <= t.wavelength_nm <= wmax_nm and t.oscillator_strength > 1e-4:
                stick_h = (t.oscillator_strength / max(t.oscillator_strength for t in self.transitions)) * max_intens * 0.85
                ax.vlines(t.wavelength_nm, 0, stick_h, color="#e11d48", lw=1.5, alpha=0.75)

        ax.set_xlabel("Wavelength $\\lambda$ (nm)", fontsize=12)
        ax.set_ylabel("Molar Absorption Intensity (arb. units)", fontsize=12)
        ax.set_title(title or "PHATPSY 2.0 Electronic Excitation (UV-Vis) Spectrum", fontsize=13, fontweight="bold")
        ax.set_xlim(wmin_nm, wmax_nm)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
        fig.tight_layout()

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p)
            plt.close(fig)
            return str(p)

        return fig

    def visualize(
        self,
        output_path: Optional[Union[str, Path]] = None,
        open_browser: bool = False,
        title: Optional[str] = None,
    ) -> str:
        """Renders standalone HTML with interactive 3D WebGL EOM player."""
        from phatpsy.visualizer import visualize_eom
        return visualize_eom(self, output_path=output_path, open_browser=open_browser, title=title)

    def show(self, height: int = 550, title: Optional[str] = None) -> Any:
        """Displays interactive 3D EOM visualizer inside Jupyter notebooks."""
        from phatpsy.visualizer import show_eom
        return show_eom(self, height=height, title=title)

    def summary(self) -> str:
        """Returns structured text summary report of all EOM properties."""
        lines = [
            "=" * 74,
            " PHATPSY 2.0 ELECTRONIC, OPTICAL & MAGNETIC (EOM) PROPERTIES REPORT",
            "=" * 74,
            "",
            "1. PERMANENT ELECTRIC DIPOLE MOMENT:",
            f"   Nuclear Dipole (a.u.):      [{self.dipole.nuclear[0]:10.4f}, {self.dipole.nuclear[1]:10.4f}, {self.dipole.nuclear[2]:10.4f}]",
            f"   Electronic Dipole (a.u.):   [{self.dipole.electronic[0]:10.4f}, {self.dipole.electronic[1]:10.4f}, {self.dipole.electronic[2]:10.4f}]",
            f"   Total Dipole Vector (a.u.): [{self.dipole.total[0]:10.4f}, {self.dipole.total[1]:10.4f}, {self.dipole.total[2]:10.4f}]",
            f"   Total Dipole (Debye):       [{self.dipole.total_debye[0]:10.4f}, {self.dipole.total_debye[1]:10.4f}, {self.dipole.total_debye[2]:10.4f}]",
            f"   Dipole Magnitude:            {self.dipole.magnitude_debye:10.4f} Debye  ({self.dipole.magnitude_au:.6f} a.u.)",
            "",
            "2. OPTICAL POLARIZABILITY TENSOR (a.u.):",
            f"   alpha_xx: {self.polarizability.static_tensor[0][0]:10.4f}   alpha_xy: {self.polarizability.static_tensor[0][1]:10.4f}   alpha_xz: {self.polarizability.static_tensor[0][2]:10.4f}",
            f"   alpha_yx: {self.polarizability.static_tensor[1][0]:10.4f}   alpha_yy: {self.polarizability.static_tensor[1][1]:10.4f}   alpha_yz: {self.polarizability.static_tensor[1][2]:10.4f}",
            f"   alpha_zx: {self.polarizability.static_tensor[2][0]:10.4f}   alpha_zy: {self.polarizability.static_tensor[2][1]:10.4f}   alpha_zz: {self.polarizability.static_tensor[2][2]:10.4f}",
            f"   Mean Static Polarizability:  {self.polarizability.mean_static_au:10.4f} a.u.  ({self.polarizability.mean_static_ang3:.4f} Angstrom^3)",
            f"   Polarizability Anisotropy:   {self.polarizability.anisotropy_au:10.4f} a.u.",
            "",
            "3. MAGNETIC SUSCEPTIBILITY:",
            f"   Diamagnetic Tensor (a.u.):  xx={self.magnetic_susceptibility.diamagnetic_tensor[0][0]:.4e}, yy={self.magnetic_susceptibility.diamagnetic_tensor[1][1]:.4e}, zz={self.magnetic_susceptibility.diamagnetic_tensor[2][2]:.4e}",
            f"   Paramagnetic Tensor (a.u.): xx={self.magnetic_susceptibility.paramagnetic_tensor[0][0]:.4e}, yy={self.magnetic_susceptibility.paramagnetic_tensor[1][1]:.4e}, zz={self.magnetic_susceptibility.paramagnetic_tensor[2][2]:.4e}",
            f"   Curie Term (T={self.magnetic_susceptibility.temperature_k:.1f} K):    {self.magnetic_susceptibility.curie_scalar:.4e} a.u.",
            f"   Total Mean Susceptibility:   {self.magnetic_susceptibility.mean_susceptibility_au:.4e} a.u.",
            f"   Molar Susceptibility:        {self.magnetic_susceptibility.molar_susceptibility_cgs * 1e6:10.4f} x 10^-6 cm^3/mol",
            "",
            "4. ELECTRONIC EXCITATIONS (UV-VIS ABSORPTION):",
            f"   {'Index':>5}  {'Spin':>5}  {'Transition':>12}  {'dE (eV)':>10}  {'lambda (nm)':>12}  {'Osc (f)':>10}  {'||mu|| (a.u.)':>12}",
            "-" * 74,
        ]

        # Top 10 transitions by oscillator strength
        sorted_trans = sorted(self.transitions, key=lambda t: t.oscillator_strength, reverse=True)
        for t in sorted_trans[:10]:
            lines.append(
                f"   {t.index:5d}  {t.spin:>5}  {t.from_orbital:4d} -> {t.to_orbital:<4d}  "
                f"{t.energy_ev:10.4f}  {t.wavelength_nm:12.2f}  {t.oscillator_strength:10.5f}  {t.transition_dipole_norm:12.4f}"
            )
        if len(sorted_trans) > 10:
            lines.append(f"   ... and {len(sorted_trans) - 10} additional electronic transitions.")
        lines.append("=" * 74)
        return "\n".join(lines)


# ==============================================================================
# Computational Engine: compute_eom()
# ==============================================================================

def compute_eom(
    deck: CalculationDeck,
    driver: Optional[PhatpsyDriver] = None,
    state: Optional[CalculationState] = None,
    grid_spacing: float = 0.20,
    dynamic_wavelengths: Optional[List[float]] = None,
    temperature_k: float = 298.15,
    quiet: bool = True,
) -> EOMResult:
    """
    Computes Electronic, Optical, and Magnetic (EOM) response properties for a molecular system.

    Parameters
    ----------
    deck : CalculationDeck
        Input molecular configuration deck.
    driver : Optional[PhatpsyDriver]
        PhatpsyDriver instance. Created if None.
    state : Optional[CalculationState]
        Pre-computed calculation state. If None, driver executes calculation on deck.
    grid_spacing : float
        Spatial voxel spacing in Bohr for 3D moment integrations (default: 0.20 Bohr).
    dynamic_wavelengths : Optional[List[float]]
        Wavelengths in nm for dynamic polarizability evaluation (default: [589.3, 1064.0]).
    temperature_k : float
        Temperature in Kelvin for Curie paramagnetism (default: 298.15 K).
    quiet : bool
        Whether to silence SCF stdout (default: True).

    Returns
    -------
    EOMResult
        Container with permanent dipole, excitations, polarizability, and susceptibility.
    """
    if driver is None:
        driver = PhatpsyDriver(quiet=quiet)

    if state is None:
        state = driver.run_deck(deck, quiet=quiet)

    res = state.to_dict()

    coords = np.asarray(res["atom_coordinates"], dtype=np.float64)  # (3, natom)
    natom = coords.shape[1]
    raw_charges = np.asarray(res["atom_charges"], dtype=np.float64).flatten()
    charges = raw_charges[:natom]
    binfo = res["basis_info"]
    mstot = res["mstot"]
    norbt = res["norbt"]
    ispin = res.get("ispin", 1)

    # Center of nuclear charge as invariant origin
    tot_nuc_z = float(np.sum(charges))
    if abs(tot_nuc_z) > 1e-8:
        r0 = np.sum(coords * charges, axis=1) / tot_nuc_z
    else:
        r0 = np.mean(coords, axis=1)

    # Coordinate grid for real 3D moment and derivative evaluations
    origin, voxel_steps, dims, (X, Y, Z) = prepare_grid_geometry(
        coords, spacing=grid_spacing, padding=4.5
    )
    dV = abs(np.linalg.det(voxel_steps))
    dx_step = voxel_steps[0, 0]
    dy_step = voxel_steps[1, 1]
    dz_step = voxel_steps[2, 2]

    # Pre-evaluate all real Slater-Type Orbitals on the grid
    chi = np.zeros((mstot, *dims), dtype=np.float64)
    for mu in range(mstot):
        iatom = binfo["iatomx"][mu] - 1
        chi[mu] = evaluate_real_sto(
            X, Y, Z,
            tuple(coords[:, iatom]),
            binfo["nx"][mu],
            binfo["lx"][mu],
            binfo["mlx"][mu],
            binfo["etax"][mu],
            binfo["anormx"][mu],
        )

    # Center-shifted coordinates
    Xr = X - r0[0]
    Yr = Y - r0[1]
    Zr = Z - r0[2]

    Xr_flat = Xr.ravel()
    Yr_flat = Yr.ravel()
    Zr_flat = Zr.ravel()

    # Nuclear dipole relative to r0
    mu_nuc = np.sum((coords - r0[:, np.newaxis]) * charges, axis=1)

    # Electronic dipole accumulator
    mu_elec = np.zeros(3, dtype=np.float64)

    # Second moments for diamagnetic susceptibility
    M_xx_tot = 0.0
    M_yy_tot = 0.0
    M_zz_tot = 0.0
    M_xy_tot = 0.0
    M_xz_tot = 0.0
    M_yz_tot = 0.0

    transitions: List[ElectronicTransition] = []
    trans_counter = 1

    # Static and dynamic polarizability accumulators
    alpha_static = np.zeros((3, 3), dtype=np.float64)
    probe_wls = dynamic_wavelengths or [589.3, 1064.0]
    alpha_dynamic: Dict[float, np.ndarray] = {
        wl: np.zeros((3, 3), dtype=np.float64) for wl in probe_wls
    }

    # Paramagnetic susceptibility accumulator
    chi_para = np.zeros((3, 3), dtype=np.float64)

    total_occ_alpha = 0.0
    total_occ_beta = 0.0

    c_light = SPEED_OF_LIGHT_AU

    for s_idx, s_name in enumerate(["alpha", "beta"] if ispin == 2 else ["alpha"]):
        C = np.asarray(res["mo_coefficients"])[:, :, s_idx]
        occ = np.asarray(res["occupations"])[:, s_idx]
        evals = np.asarray(res["eigenvalues"])[:, s_idx]

        if s_name == "alpha":
            total_occ_alpha = float(np.sum(occ))
        else:
            total_occ_beta = float(np.sum(occ))

        # Molecular orbital wavefunctions on grid: shape (norbt, nx, ny, nz)
        psi_grid = np.tensordot(C.T, chi, axes=(1, 0))
        psi_flat = psi_grid.reshape(norbt, -1)

        # 1. Permanent Dipole Contribution
        d_x = (psi_flat * Xr_flat @ (psi_flat.T * dV)).diagonal()
        d_y = (psi_flat * Yr_flat @ (psi_flat.T * dV)).diagonal()
        d_z = (psi_flat * Zr_flat @ (psi_flat.T * dV)).diagonal()

        spin_mult = 1.0 if ispin == 2 else 2.0
        mu_elec[0] += spin_mult * float(np.sum(occ * d_x))
        mu_elec[1] += spin_mult * float(np.sum(occ * d_y))
        mu_elec[2] += spin_mult * float(np.sum(occ * d_z))

        # 2. Second Moments for Diamagnetism
        M_xx = (psi_flat * (Xr_flat**2) @ (psi_flat.T * dV)).diagonal()
        M_yy = (psi_flat * (Yr_flat**2) @ (psi_flat.T * dV)).diagonal()
        M_zz = (psi_flat * (Zr_flat**2) @ (psi_flat.T * dV)).diagonal()
        M_xy = (psi_flat * (Xr_flat * Yr_flat) @ (psi_flat.T * dV)).diagonal()
        M_xz = (psi_flat * (Xr_flat * Zr_flat) @ (psi_flat.T * dV)).diagonal()
        M_yz = (psi_flat * (Yr_flat * Zr_flat) @ (psi_flat.T * dV)).diagonal()

        M_xx_tot += spin_mult * float(np.sum(occ * M_xx))
        M_yy_tot += spin_mult * float(np.sum(occ * M_yy))
        M_zz_tot += spin_mult * float(np.sum(occ * M_zz))
        M_xy_tot += spin_mult * float(np.sum(occ * M_xy))
        M_xz_tot += spin_mult * float(np.sum(occ * M_xz))
        M_yz_tot += spin_mult * float(np.sum(occ * M_yz))

        # 3. Transition Dipoles & Electronic Excitations
        # Coordinate transition matrices: D_x[i, a] = <psi_i | Xr | psi_a>
        Dx_mat = (psi_flat * Xr_flat) @ (psi_flat.T * dV)
        Dy_mat = (psi_flat * Yr_flat) @ (psi_flat.T * dV)
        Dz_mat = (psi_flat * Zr_flat) @ (psi_flat.T * dV)

        # Angular momentum operators on grid: L = -i (r x grad)
        # Using centered spatial finite differences on psi_grid
        grad_x = np.zeros_like(psi_grid)
        grad_y = np.zeros_like(psi_grid)
        grad_z = np.zeros_like(psi_grid)
        for p in range(norbt):
            gz, gy, gx = np.gradient(psi_grid[p], dz_step, dy_step, dx_step)
            grad_x[p] = gx
            grad_y[p] = gy
            grad_z[p] = gz

        # Matrix elements of (r x grad):
        # L_x = Yr * grad_z - Zr * grad_y
        # L_y = Zr * grad_x - Xr * grad_z
        # L_z = Xr * grad_y - Yr * grad_x
        Lx_psi = (Yr * grad_z - Zr * grad_y).reshape(norbt, -1)
        Ly_psi = (Zr * grad_x - Xr * grad_z).reshape(norbt, -1)
        Lz_psi = (Xr * grad_y - Yr * grad_x).reshape(norbt, -1)

        Lx_mat = psi_flat @ (Lx_psi.T * dV)
        Ly_mat = psi_flat @ (Ly_psi.T * dV)
        Lz_mat = psi_flat @ (Lz_psi.T * dV)

        occ_indices = np.where(occ > 0.05)[0]
        vir_indices = np.where(occ < 0.95)[0]

        for i in occ_indices:
            for a in vir_indices:
                dE = float(evals[a] - evals[i])
                if dE > 1e-4:
                    mux = float(Dx_mat[i, a])
                    muy = float(Dy_mat[i, a])
                    muz = float(Dz_mat[i, a])
                    mu_vec = [mux, muy, muz]
                    mu2 = mux * mux + muy * muy + muz * muz
                    mu_norm = math.sqrt(mu2)

                    f_osc = (2.0 / 3.0) * dE * mu2
                    wl_nm = PLANCK_CONSTANT_HC_EV_NM / (dE * HARTREE_TO_EV)

                    transitions.append(
                        ElectronicTransition(
                            index=trans_counter,
                            spin=s_name,
                            from_orbital=int(i + 1),
                            to_orbital=int(a + 1),
                            energy_hartree=dE,
                            energy_ev=dE * HARTREE_TO_EV,
                            wavelength_nm=wl_nm,
                            oscillator_strength=f_osc,
                            transition_dipole=mu_vec,
                            transition_dipole_norm=mu_norm,
                        )
                    )
                    trans_counter += 1

                    # Sum-Over-States Polarizability
                    mu_outer = np.outer(mu_vec, mu_vec)
                    alpha_static += spin_mult * (2.0 / dE) * mu_outer

                    for wl_probe in probe_wls:
                        omega = (PLANCK_CONSTANT_HC_EV_NM / wl_probe) / HARTREE_TO_EV
                        denom = dE * dE - omega * omega
                        if abs(denom) > 1e-5:
                            alpha_dynamic[wl_probe] += spin_mult * (2.0 * dE / denom) * mu_outer

                    # Van Vleck Paramagnetic Susceptibility
                    # L_mat is real part of i*L operator: |<i|L|a>|^2 = Lx_mat[i,a]^2
                    lx = float(Lx_mat[i, a])
                    ly = float(Ly_mat[i, a])
                    lz = float(Lz_mat[i, a])
                    l_outer = np.outer([lx, ly, lz], [lx, ly, lz])
                    chi_para += spin_mult * (1.0 / (2.0 * c_light * c_light)) * (1.0 / dE) * l_outer

    # Final Dipole Moment
    mu_total = mu_nuc - mu_elec
    mu_mag_au = float(np.linalg.norm(mu_total))
    dipole_res = DipoleMoment(
        total=mu_total.tolist(),
        total_debye=(mu_total * AU_TO_DEBYE).tolist(),
        magnitude_au=mu_mag_au,
        magnitude_debye=mu_mag_au * AU_TO_DEBYE,
        nuclear=mu_nuc.tolist(),
        electronic=mu_elec.tolist(),
        center_of_charge=r0.tolist(),
    )

    # Final Polarizability Tensor
    mean_static_au = float(np.trace(alpha_static) / 3.0)
    mean_static_ang3 = mean_static_au * AU_TO_ANG3

    axx = alpha_static[0, 0]
    ayy = alpha_static[1, 1]
    azz = alpha_static[2, 2]
    axy = alpha_static[0, 1]
    axz = alpha_static[0, 2]
    ayz = alpha_static[1, 2]

    aniso_term = 0.5 * (
        (axx - ayy)**2 + (ayy - azz)**2 + (azz - axx)**2 +
        6.0 * (axy**2 + axz**2 + ayz**2)
    )
    anisotropy_au = math.sqrt(max(0.0, aniso_term))

    dyn_dict: Dict[str, Dict[str, Any]] = {}
    for wl_probe, tensor_dyn in alpha_dynamic.items():
        m_dyn = float(np.trace(tensor_dyn) / 3.0)
        dyn_dict[f"{wl_probe:.1f}_nm"] = {
            "wavelength_nm": wl_probe,
            "tensor_au": [[round(float(v), 4) for v in row] for row in tensor_dyn.tolist()],
            "mean_au": round(m_dyn, 4),
            "mean_ang3": round(m_dyn * AU_TO_ANG3, 4),
        }

    pol_res = PolarizabilityTensor(
        static_tensor=alpha_static.tolist(),
        mean_static_au=mean_static_au,
        mean_static_ang3=mean_static_ang3,
        anisotropy_au=anisotropy_au,
        dynamic_tensors=dyn_dict,
    )

    # Final Magnetic Susceptibility Tensor
    # Diamagnetic tensor from second moments: chi_dia_jk = -1/(4 c^2) * <r^2 delta_jk - r_j r_k>
    pref_dia = -1.0 / (4.0 * c_light * c_light)
    chi_dia = np.array([
        [pref_dia * (M_yy_tot + M_zz_tot), -pref_dia * M_xy_tot, -pref_dia * M_xz_tot],
        [-pref_dia * M_xy_tot, pref_dia * (M_xx_tot + M_zz_tot), -pref_dia * M_yz_tot],
        [-pref_dia * M_xz_tot, -pref_dia * M_yz_tot, pref_dia * (M_xx_tot + M_yy_tot)],
    ], dtype=np.float64)

    # Curie susceptibility for open-shell net spin S > 0
    net_spin_electrons = abs(total_occ_alpha - total_occ_beta) if ispin == 2 else 0.0
    S = 0.5 * net_spin_electrons
    g_e = 2.0023193
    if S > 0.0:
        curie_val = (g_e**2 * S * (S + 1.0)) / (12.0 * BOLTZMANN_AU_PER_K * temperature_k)
    else:
        curie_val = 0.0

    chi_total = chi_dia + chi_para + curie_val * np.eye(3)
    mean_chi_au = float(np.trace(chi_total) / 3.0)
    molar_chi_cgs = mean_chi_au * AU_TO_CGS_MOLAR_SUSCEPTIBILITY

    mag_res = MagneticSusceptibility(
        diamagnetic_tensor=chi_dia.tolist(),
        paramagnetic_tensor=chi_para.tolist(),
        curie_scalar=curie_val,
        total_tensor=chi_total.tolist(),
        mean_susceptibility_au=mean_chi_au,
        molar_susceptibility_cgs=molar_chi_cgs,
        temperature_k=temperature_k,
    )

    # Sort transitions by oscillator strength descending
    transitions.sort(key=lambda t: t.oscillator_strength, reverse=True)
    for idx, t in enumerate(transitions):
        t.index = idx + 1

    return EOMResult(
        dipole=dipole_res,
        transitions=transitions,
        polarizability=pol_res,
        magnetic_susceptibility=mag_res,
        deck=deck,
        state=state,
    )
