# Equation & Routine Mapping: Dissertation to Code

> **Status:** DRAFT (Pending User / Author Review)  
> **Source:** Jack A. Smith, *A Projected Hamiltonian Approach to Polyatomic Systems* (1980).  
> **Primary LaTeX Reference:** [`PhD_Thesis/PHATPSY.tex`](../../PhD_Thesis/PHATPSY.tex)

This document provides a line-of-sight mapping connecting key mathematical derivations in the PhD dissertation to original legacy FORTRAN routines and their modernized Fortran 2018 / Python equivalents.

---

## 1. Chapter 2: One-Electron Operator & Atom-in-Molecule Subspace Partitioning

| Dissertation Eq. | Mathematical Expression | Legacy Routine | Modernized Code | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Eq. (2.5)** | $F \phi = \varepsilon \phi$ | `src/oneint.f` | `src/state.f90`, `src/scf.f90` | Localized atomic spin-orbital secular equation |
| **Eq. (2.12)** | $\langle \phi_i^A \| \phi_j^B \rangle = 0$ | `src/scf.f` | `src/scf.f90` | Interatomic orbital orthogonality condition |
| **Eq. (2.25)** | $P = \sum_{j \in B} \|\phi_j^B\rangle \langle \phi_j^B\|$ | `src/oneint.f` | `src/scf.f90` | Projection operator onto neighboring atom ($B$) subspace |
| **Eq. (2.34)** | $T_{\mu\nu} = -\frac{1}{2} \langle \chi_\mu \| \nabla^2 \| \chi_\nu \rangle$ | `src/oneint.f` | `src/twocenter.f90`, `src/scf.f90` | Kinetic energy matrix in STO basis |

---

## 2. Chapter 3: Energy-Dependent Pseudopotential for Atom-Atom Partitioning

| Dissertation Eq. | Mathematical Expression | Legacy Routine | Modernized Code | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Eq. (3.12)** | $V_{\text{PK}} = \sum_{j \in B} (\varepsilon - \varepsilon_j^B) \|\phi_j^B\rangle \langle \phi_j^B\|$ | `src/scf.f`, `fokmat.f` | `src/scf.f90` | Interatomic Phillips-Kleinman Pauli exclusion pseudopotential |
| **Eq. (3.30)** | $M = 1 - \sum_k (1 - \lambda_k) \|\phi_k^B\rangle \langle \phi_k^B\|$ | `src/scf.f` | `src/scf.f90` | Approximate pseudometric operator for energy dependence |
| **Eq. (3.45)** | $F C = \varepsilon M C$ | `src/scf.f`, `eispak.f` | `src/scf.f90`, `src/linalg.f90` | Generalized secular equation (solved via LAPACK `DSPEVD`/`DSPEV`) |
| **Modern §1.2** | $V_{\text{PK}}(E) = \sum_{j} (E_{\text{ref}} - \varepsilon_j) \|\phi_j\rangle\langle\phi_j\|$ | — | `src/scf.f90` | Energy-dependent PK operator with unit scale `PKSCAL=1.0` |
| **Modern §1.2b** | $P_{\text{core}} = 0 \text{ for } Z \le 1$ | — | `src/scf.f90` | Hydrogen core-exemption invariant ($N_{\text{core}}=0$) enabling universal `PKSCAL=1.0` |

---

## 3. Chapter 4: Model Interatomic Coulomb Potential

| Dissertation Eq. | Mathematical Expression | Legacy Routine | Modernized Code | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Eq. (4.15)** | $V^B(r) = -Z_B^*/r + \sum_k c_k^B e^{-\zeta_k r}$ | `src/analys.f` | `src/analys.f90` | Spherical exponential model potential expansion |
| **Eq. (4.28)** | $\sum_k c_k^B = -Z_B + Z_B^*$ | `src/analys.f` | `src/analys.f90` | Nuclear cusp constraint at $r=0$ (`WGT0`) |
| **Eq. (4.35)** | $\int \nabla^2 V^B d^3r = 4\pi Q_B$ | `src/analys.f` | `src/analys.f90` | Poisson total electronic charge constraint (`WGT1`) |
| **Eq. (4.48)** | $\min \sum_i w_i (V_{\text{exact}}(r_i) - V_{\text{model}}(r_i))^2$ | `src/analys.f` | `src/analys.f90` (LAPACK `DGELSD`) | Constrained SVD least-squares model potential fitting |

---

## 4. Chapter 5: Nonlocal Model Exchange & Modern DFT Kernels

| Dissertation Eq. | Mathematical Expression | Legacy Routine | Modernized Code | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Eq. (5.10)** | $K_{\mu\nu} = \langle \chi_\mu \| V_X \| \chi_\nu \rangle$ | `src/twoint.f`, `fokmat.f` | `src/scf.f90` | Interatomic exchange potential integrals |
| **Eq. (5.24)** | $P_{X} = \sum_{k} \alpha_k \|\phi_k\rangle \langle \phi_k\|$ | `src/fokmat.f` | `src/scf.f90` | Projected local density exchange approximation |
| **Modern §1.3** | $V_X^{\text{LDA}}(\rho) = -\frac{3\alpha}{2} (3\rho/\pi)^{1/3}$ | — | `src/exchange_dft.f90`, `phatpsy/exchange_dft.py` | Local Slater exchange kernel ($N=64$ radial quadrature) |
| **Modern §1.3** | $F_x^{\text{B88}}(s), F_x^{\text{PBE}}(s)$ | — | `src/exchange_dft.f90`, `phatpsy/exchange_dft.py` | Becke88 and PBE GGA gradient-corrected exchange |

---

## 5. Chapter 6: Energy-Weighted Maximum Overlap, Propagator & Populations

| Dissertation Eq. | Mathematical Expression | Legacy Routine | Modernized Code | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Eq. (6.14)** | $G(E) = (E \cdot S - H)^{-1}$ | `src/ewmo.f` | `src/ewmo.f90`, `phatpsy/ewmo.py` | One-electron Green's function / Propagator |
| **Eq. (6.28)** | $\rho(E) = -\frac{1}{\pi} \text{Im} \operatorname{Tr} [G(E^+)]$ | `src/ewmo.f` | `src/ewmo.f90` | Spectral density of states decomposition |
| **Eq. (6.42)** | $S^{-1/2} H S^{-1/2} = U \Lambda U^\dagger$ | `src/lowdin.f`, `ewmo.f` | `src/ewmo.f90`, `src/linalg.f90` | Lowdin symmetric orthogonalization (LAPACK `DSPEVD`) |
| **Modern §1.4** | $\mathbf{P}_{\text{Lowdin}} = \mathbf{S}^{1/2} \mathbf{D} \mathbf{S}^{1/2}$ | — | `src/density.f90` (`lowdin_pop`) | Symmetrically orthogonalized Lowdin population analysis |

---

## 6. Chapters 7 & 8: Grand-Canonical Hartree-Fock, Equilibration & Subsystem Energies

| Dissertation Eq. | Mathematical Expression | Legacy Routine | Modernized Code | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Eq. (7.18)** | $n_i = [1 + \exp((\varepsilon_i - \mu)/\Theta)]^{-1}$ | `src/contrl.f`, `scf.f` | `src/contrl.f90`, `src/ewmo.f90` | Fermi-Dirac thermal occupation numbers (`solve_fermi_level`) |
| **Modern §1.4** | $\min \|\sum_k c_k \mathbf{e}^{(k)}\|^2$ | — | `phatpsy/diis.py`, `phatpsy/driver.py` | Pulay DIIS charge acceleration with total charge conservation |
| **Eq. (8.22)** | $\langle V \rangle / \langle T \rangle \to -2.0$ | `src/ewmo.f` | `src/contrl.f90`, `phatpsy/driver.py` | Virial theorem evaluation check |
| **Modern §1.1** | $E_{\text{total}} = E_{\text{cluster}} + E_{\text{env}}^{\text{bulk}} + E_{\text{embed}}$ | — | `phatpsy/subsystem.py` | Physical subsystem energy formulation & embedding |

---

## 7. Spatial Symmetry, Equivalent Atoms & Special Centers

| Operational Concept | Mathematical Operation | Legacy Routine | Modernized Code | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Equivalent Atom Spatial Mapping** | $\mathbf{R}_{\text{eq}} = \mathcal{R}_{\text{sym}} \mathbf{R}_{\text{gen}}$ | `src/symxyz.f` | `src/symxyz.f`, `phatpsy_atomic_mod.f90` | Generates XYZ coordinates for `NEQAT` atoms using `ISYM`/`IXYZ` |
| **Orbital Symmetry Transformation** | $C_{\mu i}^{(\text{eq})} = D(R) C_{\mu i}^{(\text{gen})}$ | `src/symop.f` | `src/symop.f`, `phatpsy_atomic_mod.f90` | Transforms STO coefficients and model potential via Wigner $D$-matrices |
| **Spin-Flipping on Symmetry** | $\alpha \leftrightarrow \beta$ | `src/flip.f` | `src/flip.f`, `phatpsy_atomic_mod.f90` | Exchanges spin manifolds when `QFLIP=T` (LVB singlet generation) |
| **Fixed (Frozen) Environment Atoms** | $\delta V_{\text{fixed}} = 0, \delta C_{\text{fixed}} = 0$ | `src/atomic.f`, `scf.f`, `analys.f` | `phatpsy_scf_mod.f90`, `phatpsy_analys_mod.f90` | Freezes orbitals and potential (`QFIXED=T`); bypasses SCF iterations |
| **Virtual Atoms (Coordination Completion)** | $V_{\text{ext}} \neq 0, \text{excluded at EWMO}$ | `src/atomic.f`, `bloop.f` | `phatpsy_atomic_mod.f90`, `phatpsy/schema.py` | Completes local coordination environment during atomic stage; excluded from molecular EWMO calculation (`QVIRTL=T`) |

---

## 8. Dispersion & Polarized Basis Sets (Phase 6)

| Operational Concept | Mathematical Expression | Legacy Routine | Modernized Code | Description |
| :--- | :--- | :--- | :--- | :--- |
| **DFT-D3(BJ) Dispersion Energy** | $E_{\text{disp}} = -\frac{1}{2}\sum [s_6 \frac{C_6}{R^6 + R_0^6} + s_8 \frac{C_8}{R^8 + R_0^8}]$ | — | `phatpsy/dispersion.py` | Grimme D3 with Becke-Johnson rational damping |
| **Analytical Dispersion Gradient** | $\mathbf{g}_{\text{disp}} = -\nabla E_{\text{disp}}$ | — | `phatpsy/dispersion.py` | Analytical Cartesian dispersion force vector |
| **DZP Polarized STOs** | $2p$ on $\text{H}$, $3d$ on $\text{Li}$--$\text{Ar}$, $4f$ on metals | — | `phatpsy/data/polarized_sto.json`, `phatpsy/basis.py` | Polarization basis functions for directional bonding |
| **Slater Spherical Normalization** | $N_n(\zeta) = (2\zeta)^{n+1/2}/\sqrt{(2n)!}$ | `src/genfac.f`, `src/gennrm.f` | `src/modelpot.f90`, `phatpsy/basis.py` | Normalization of high-$l$ spherical harmonics |

---

## 9. EOM Polarizability & Spectroscopic Properties (Phase 6)

| Operational Concept | Mathematical Expression | Legacy Routine | Modernized Code | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Invariant Permanent Dipole** | $\boldsymbol{\mu} = \sum Z_A (\mathbf{R}_A - \mathbf{R}_0) - \int (\mathbf{r}-\mathbf{R}_0)\rho d^3r$ | `src/prop.f` | `phatpsy/eom.py` | Center-of-charge referenced permanent dipole |
| **Traceless Quadrupole Tensor** | $\mathbf{\Theta} = \frac{1}{2}(3\mathbf{Q} - \operatorname{Tr}(\mathbf{Q})\mathbf{I})$ | — | `phatpsy/eom.py` | Real-space 3D grid second-moment tensor |
| **SOS Dynamic Polarizability** | $\boldsymbol{\alpha}(\omega) = 2 \sum \frac{\Delta \varepsilon}{\Delta \varepsilon^2 - \omega^2} \boldsymbol{\mu}_{ia} \otimes \boldsymbol{\mu}_{ia}$ | — | `phatpsy/eom.py` | Frequency-dependent optical response tensor |
| **Oscillator Strengths** | $f_{ia} = \frac{2}{3} \Delta \varepsilon_{ia} \|\boldsymbol{\mu}_{ia}\|^2$ | — | `phatpsy/eom.py` | Transition dipoles and UV-Vis absorption intensities |
| **Magnetic Susceptibility** | $\boldsymbol{\chi} = \boldsymbol{\chi}_{\text{dia}} + \boldsymbol{\chi}_{\text{para}}$ | — | `phatpsy/eom.py` | Van Vleck paramagnetic & diamagnetic tensors |

---

## 10. Analytical Gradients, Vibrations & BOMD (Phase 7)

| Operational Concept | Mathematical Expression | Legacy Routine | Modernized Code | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Analytical Nuclear Gradient** | $\mathbf{g}_A = -\mathbf{F}_A = \mathbf{g}_A^{\text{elec}} + \mathbf{g}_A^{\text{nuc}} + \mathbf{g}_A^{\text{disp}}$ | — | `phatpsy/driver.py` (`gradient`) | Conservative Cartesian forces ($\sum \mathbf{F} = \mathbf{0}$) |
| **Cartesian Hessian Matrix** | $H_{ij} = \partial^2 E / \partial x_i \partial x_j$ | — | `phatpsy/vibrations.py` | Central finite-difference energy curvature |
| **Eckart-Sayvetz Projector** | $\mathbf{P} = \mathbf{I} - \mathbf{D}(\mathbf{D}^T \mathbf{D})^{-1}\mathbf{D}^T$ | — | `phatpsy/vibrations.py` | Projects out 6 rigid translations and rotations |
| **Harmonic Vibrational Frequencies** | $\tilde{\nu} = \sqrt{\lambda} / (2\pi c)$ | — | `phatpsy/vibrations.py` | Mass-weighted normal modes and frequencies ($\text{cm}^{-1}$) |
| **Statistical Thermochemistry** | $H(T), S(T), G(T), E_{\text{ZPVE}}$ | — | `phatpsy/vibrations.py` | RRHO enthalpy, entropy, and Gibbs free energy |
| **Velocity Verlet BOMD** | $\mathbf{R}(t+\Delta t) = \mathbf{R} + \mathbf{v}\Delta t + \frac{1}{2}\mathbf{a}\Delta t^2$ | — | `phatpsy/md.py` | Microcanonical NVE dynamics with energy conservation |
| **Canonical Thermostats** | Berendsen ($\tau$), Andersen ($\nu_{\text{coll}}$) | — | `phatpsy/md.py` | NVT constant-temperature molecular dynamics |

---

## 11. Reaction Pathways, Saddle Points & IRC (Phase 8)

| Operational Concept | Mathematical Expression | Legacy Routine | Modernized Code | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Henkelman-Jónsson Tangents** | $\hat{\boldsymbol{\tau}}_i(V_{i+1}, V_i, V_{i-1})$ | — | `phatpsy/neb.py` | Energy-ordered tangents to prevent path kinking |
| **Climbing-Image NEB** | $\mathbf{F}_{i_{\max}} = \mathbf{F}^{\text{true}} - 2(\mathbf{F}^{\text{true}}\cdot\hat{\boldsymbol{\tau}})\hat{\boldsymbol{\tau}}$ | — | `phatpsy/neb.py` | Rigorous saddle point location without spring force |
| **QuickMin Optimizer** | $\mathbf{v} \to (\mathbf{v}\cdot\hat{\mathbf{F}})\hat{\mathbf{F}}$ | — | `phatpsy/neb.py`, `phatpsy/dimer.py` | Projected velocity damping path optimizer |
| **Dimer Method Saddle Refinement** | $\mathbf{F}^\dagger = -\mathbf{F} + 2(\mathbf{F}\cdot\hat{\mathbf{N}})\hat{\mathbf{N}}$ | — | `phatpsy/dimer.py` | Gradient-only saddle search along lowest eigenmode |
| **Intrinsic Reaction Coordinate** | $d\mathbf{x}/ds = -\mathbf{g}_x / \|\mathbf{g}_x\|$ | — | `phatpsy/irc.py` | Mass-weighted steepest descent from transition state |
| **Activation Barriers** | $\Delta E_f^\ddagger, \Delta E_r^\ddagger, \Delta E_{\text{rxn}}$ | — | `phatpsy/neb.py`, `phatpsy/irc.py` | Forward/reverse barriers and reaction thermodynamics |
