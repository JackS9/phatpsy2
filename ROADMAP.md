# PHATPSY 2.0 Development Roadmap & Goals

This roadmap defines the scientific, modeling, algorithmic, and architectural goals for the modernization of **PHATPSY 2.0**.

---

## 1. Scientific Methodology & Physics Kernels

- [x] **Improve TF Potential fitting [COMPLETED]**
  - **LAPACK Constrained Cusp Fitting (`DGELSD`):** Upgraded model potential parameterization in [`src/analys.f90`](file:///Users/jacksmith/phatpsy2/src/analys.f90) from heuristic matrix inversion to singular value decomposition (SVD) with analytical nuclear cusp conditions ($Z_p(0) = Z - N_{\text{core}}$, $(dZ_p/dr)_{r=0} = -2 Z (Z - N_{\text{core}})$), Poisson charge conservation, and valence matching.
  - **Dynamic STO Exponent Optimization:** Self-consistent gradient optimization of screening radial exponents $\{\zeta_k\}$, eliminating fixed empirical table lookups.
  - **Modern DFT Exchange Kernels:** Spherically symmetric radial quadrature evaluation of local Slater exchange ($\rho^{1/3}$), Becke88 gradient-corrected exchange, and Perdew-Burke-Ernzerhof (PBE) GGA exchange in [`src/exchange_dft.f90`](file:///Users/jacksmith/phatpsy2/src/exchange_dft.f90) and [`phatpsy/exchange_dft.py`](file:///Users/jacksmith/phatpsy2/phatpsy/exchange_dft.py) hooked into atomic SCF via `ControlConfig(exchange_kernel=...)`.
- [x] **Improve handling of energy-dependence in PK potential [COMPLETED]**
  - **Theoretical Lock `PKSCAL = 1.0`:** Eliminated arbitrary empirical scaling knobs ($0.40$ / $0.80$).
  - **Energy-Dependent Phillips-Kleinman Operator $V_{\text{PK}}(E)$:** Implemented dynamic valence eigenvalue projection in [`src/scf.f90`](file:///Users/jacksmith/phatpsy2/src/scf.f90) using reference valence level ($E_{\text{ref}} \approx -0.22\text{ Ha}$), stabilizing 12-coordinated bulk metals ($\text{Ni}_{13}$ FCC cluster) without core over-repulsion.
  - **Symmetrized Core Projection:** Reformulated non-local projection into symmetric matrix form preserving exact Hermiticity and physical virial ratios ($-V/T \approx 2.0$).
  - **Hydrogen Core-Exemption Invariant ($Z=1$):** Codified the physical principle that Hydrogen ($N_{\text{core}} = 0$) has no core states to project out ($V_{\text{PK}} = 0$, $M = \mathbf{S}$), enabling universal first-principles unit scaling (`pk_scale: 1.0`) across all elements without requiring empirical overrides on hydrides or C-H bonds.
- [x] **Improve charge equilibration after EWMO [COMPLETED]**
  - **Symmetrically Orthogonalized Löwdin Population Analysis:** Implemented `lowdin_pop` in [`src/density.f90`](file:///Users/jacksmith/phatpsy2/src/density.f90) ($\mathbf{P}_{\text{Lowdin}} = \mathbf{S}^{1/2}\mathbf{D}\mathbf{S}^{1/2}$), guaranteeing non-negative gross populations and strictly physical electron counts.
  - **Pulay DIIS Charge Acceleration:** Added built-in Direct Inversion in the Iterative Subspace in [`phatpsy/driver.py`](file:///Users/jacksmith/phatpsy2/phatpsy/driver.py) (`use_diis=True`) with total charge renormalization, suppressing charge-sloshing oscillations in polar heteronuclear systems ($\text{CO}$).
  - **Global Fermi Level ($\mu$) Equilibration:** Implemented `solve_fermi_level` in [`src/ewmo.f90`](file:///Users/jacksmith/phatpsy2/src/ewmo.f90) via hybrid Newton-Raphson/bisection under Fermi-Dirac distribution, guaranteeing exact particle conservation ($< 10^{-11}$) across degenerate d-bands and metallic clusters.
- [x] **Subsystem Energy Partitioning & Granular Diagnostics [COMPLETED]**
  - **ONIOM / DFT-in-DFT Physical Partitioning:** Implemented rigorous subsystem energy decomposition in [`phatpsy/subsystem.py`](file:///Users/jacksmith/phatpsy2/phatpsy/subsystem.py): $E_{\text{total}} = E_{\text{cluster}} + E_{\text{env}}^{\text{bulk}} + E_{\text{embed}}$ with frozen bulk references ($-1502.565\text{ Ha/atom}$ for $\text{Ni}$), Clementi & Roetti / Thakkar isolated atomic references, and exact energy conservation ($< 10^{-12}\text{ Ha}$).
  - **Granular Per-Cycle Diagnostics:** Real-time tracking of Phillips-Kleinman penalty energy ($dE_{\text{PK}}$), Coulomb fitting RMS, exchange fitting RMS, charge transfer ($\Delta Q$), and virial theorem in [`src/state.f90`](file:///Users/jacksmith/phatpsy2/src/state.f90) and `CalculationState.cycle_history`.
- [x] **Benchmark Validation & Parameter Retirement [COMPLETED]**
  - **Retirement of Legacy Baselines & Namelist Input:** Formally retired historical legacy baseline comparisons (`data/validation_baselines/`, `data/*_legacy.stdout`) and legacy Namelist input files (`*.stdin`). Calculations are specified via modern YAML/JSON decks (`CalculationDeck`) and typed Python API.
  - **External Experimental Benchmarks:** Built validation suite in [`phatpsy/benchmarks.py`](file:///Users/jacksmith/phatpsy2/phatpsy/benchmarks.py) evaluating error vectors against NIST CCCBDB, CRC Handbook, Herzberg, and Somorjai & Li across bond lengths, frequencies, dipoles, bulk metal stability ($Ni_{13}$), clean surface slab ($Ni(100)$ producing reusable surface and subsurface atoms with fixed bulk), and surface chemisorption ($E_{\text{ads}}$ for $CO/Ni(100)$).
  - **Retirement of Legacy Empirical Knobs:** Replaced manual damping parameters (`CDAMP`, `VDAMP`, `CACCEL`) with self-consistent first-principles defaults via `ControlConfig.first_principles()`.

### Continuous Methodology Improvement Program (High-Level Parity)
*Governed by [`Methodology_plan.md` (Part II)](file:///Users/jacksmith/phatpsy2/Methodology_plan.md) to systematically close error vectors against NIST CCCBDB, CRC, Herzberg, Somorjai & Li, and PySCF/CCSD(T):*
- [x] **Phase 6: Analytical Pulay Gradients & Vibrational Curvature Correction [COMPLETED]**
  - Implement analytical overlap derivatives ($\partial S/\partial R$) and model potential gradients ($\partial V/\partial R$) in [`src/twocenter.f90`](file:///Users/jacksmith/phatpsy2/src/twocenter.f90).
  - Add Pulay basis translation force corrections in [`src/analys.f90`](file:///Users/jacksmith/phatpsy2/src/analys.f90) and `phatpsy/driver.py` to correct the artificial stiffness in harmonic frequencies ($\omega_e = 2358.60\text{ cm}^{-1}$ on $N_2$ and $2169.24\text{ cm}^{-1}$ on $CO$, satisfying Milestone 6 criteria within $[2100, 2500]\text{ cm}^{-1}$ and $[1950, 2350]\text{ cm}^{-1}$). Upgrade Cartesian Hessian in `phatpsy/vibrations.py` to central differences of analytical gradients.
- [x] **Phase 7: Angular Polarization Basis Sets & Multipole Potential Fitting [COMPLETED]**
  - [x] **Polarized STO Basis Sets (`DZP` / `SZP`) [COMPLETED]:** Ingest d-polarization (C, N, O, Li–Ar) and p-polarization (H, He) STO functions into [`phatpsy/basis.py`](file:///Users/jacksmith/phatpsy2/phatpsy/basis.py) and [`phatpsy/schema.py`](file:///Users/jacksmith/phatpsy2/phatpsy/schema.py). Automatically configure `LMAX=2` and initialize $(2l+1)$ virtual polarization orbitals ($0.0 \uparrow, 0.0 \downarrow$) for variational mixing. Validated on $H_2O$ ($\mu$ reduced from $4.35\text{ D}$ to $1.18\text{ D}$), $CO$ ($\mu$ reduced from $>5\text{ D}$ to $0.96\text{ D}$ with correct $C^{\delta-} O^{\delta+}$ polarity), and $N_2$ across automated tests.
  - [x] **Multipole Model Potential Fitting & Traceless Quadrupoles [COMPLETED]:** Extended Fortran core ([`src/modelpot.f90`](file:///Users/jacksmith/phatpsy2/src/modelpot.f90), [`src/analys.f90`](file:///Users/jacksmith/phatpsy2/src/analys.f90), [`src/state.f90`](file:///Users/jacksmith/phatpsy2/src/state.f90)) to evaluate intra-atomic dipole transition moments ($s \leftrightarrow p$, $p \leftrightarrow d$) and interatomic reaction fields. Exposed `atom_dipoles` on `CalculationState` and `EmbeddedAtomState.dipole`. Upgraded [`phatpsy/eom.py`](file:///Users/jacksmith/phatpsy2/phatpsy/eom.py) to evaluate full traceless symmetric quadrupole tensors $\mathbf{\Theta}_{\alpha\beta}$ in atomic units and Buckingham ($1.345034\text{ B}$), yielding close parity with microwave spectroscopy on water ($-3.07, +2.76, +0.31\text{ B}$) and machine-precision translational origin invariance ($< 10^{-12}\text{ D}$).
  - **Basis Set Extension Program (ADF ZORA STO Ingest):**
    - [ ] **Phase 7.1: Literature Polarization Exponents (H–Ar) [IN PROGRESS]:** Replace placeholder arithmetic progression exponents in `POLARIZATION_PARAMETERS` with published ADF standard (ZORA-contracted) values (van Lenthe & Baerends 2003, Gasević et al. 2022) extracted from `zorabasis/TZP` vs `zorabasis/DZ`. Validated by shifting $H_2O$ dipole from $1.50\text{ D} \to 1.63\text{ D}$ toward experimental $1.855\text{ D}$ and $CO$ dipole from $1.19\text{ D} \to 1.08\text{ D}$.
    - [ ] **Phase 7.2: First-Class ADF STO Library (`phatpsy/data/adf_sto.json`):** Create automated ingest script `phatpsy/ingest_adf.py` to ingest all-electron basis definitions from `zorabasis/` for `ADF-DZ`, `ADF-TZP`, `ADF-TZ2P` (double polarization: $p+d$ on H, $d+f$ on heavy atoms), and `ADF-QZ4P` (quadruple polarization), with automatic initial orbital occupancy synthesizer.
    - [ ] **Phase 7.3: Transition Metal & Heavy Element Coverage (Z=1 to 118):** Ingest 3d/4d/5d transition metal sets (Fe, Co, Ni, Cu, Ag, Pt, Au) from `zorabasis/` to provide high-precision basis options for surface cluster and catalysis calculations (directly supporting Phase 9).
- [x] **Phase 8: Exchange-Correlation Functionals & Non-Covalent Dispersion (D3/D4) [COMPLETED]**
  - Integrated Perdew-Wang 92 (PW92) LDA and Perdew-Burke-Ernzerhof (PBE) GGA correlation functional kernels into [`src/exchange_dft.f90`](file:///Users/jacksmith/phatpsy2/src/exchange_dft.f90) and [`phatpsy/exchange_dft.py`](file:///Users/jacksmith/phatpsy2/phatpsy/exchange_dft.py), integrated into radial quadrature STO matrices via `ControlConfig(correlation_kernel="pbe")` and `CORR_TYPE` in namelist `/CTLINP/`.
  - Implemented Grimme DFT-D3(BJ) empirical dispersion in [`phatpsy/dispersion.py`](file:///Users/jacksmith/phatpsy2/phatpsy/dispersion.py) with analytical Cartesian nuclear gradients matching finite-difference forces to $< 10^{-8}\text{ Ha/Bohr}$ and machine-precision translational invariance ($< 10^{-12}$). Hooked into `ControlConfig(use_dispersion=True)` and `PhatpsyDriver.gradient()`.
  - Validated bond dissociation energies ($D_e = 9.759\text{ eV}$ for $\text{N}_2$ within $[9.0, 10.0]\text{ eV}$; $11.087\text{ eV}$ for $\text{CO}$ within $[10.5, 11.5]\text{ eV}$) and unimolecular ring-opening activation barrier on $\text{EO} \to \text{AA}$ ($\Delta E_f^\ddagger = 53.0\text{ kcal/mol}$ matching CCSD(T)/CBS benchmark within $[50.0, 56.0]\text{ kcal/mol}$). Verified across 244 passing tests in canonical build loop.
- [ ] **Phase 9: Transition-Metal Chemisorption & d-Band Coupling ($CO$ & $NO$ on $Ni(100)$)**
  - Directional $C_{4v}$ crystal field d-band projection ($e_g$ vs $t_{2g}$) for Blyholder $5\sigma / 2\pi^*$ coupling.
  - Multi-site 2D lateral potential energy surface reproducing experimental top-site preference ($E_{\text{ads}} = -30.0\text{ kcal/mol}$) over bridge and hollow, and vibrational red-shift ($\Delta\nu_{\text{CO}} \approx -78\text{ cm}^{-1}$).
  - Boundary atom fixation invariant: Fixed bulk (`Ni_bulk_fixed`), second-layer (`Ni_sub`), and surface-layer (`Ni_surf`) representations eliminating cluster boundary reflections.
- [ ] **Phase 10: Automated Multi-Level Computational Parity Suite (PySCF Bridge)**
  - Standalone comparative bridge script (`tools/benchmark_pyscf_bridge.py`) running PySCF on identical geometries.
  - Continuous multi-property $L_2$ error vector norm tracking in GitHub Actions CI to guarantee non-regression.

---

## 2. Modernized Atom Representation & Intrinsic Properties

- [x] **Improve/modernize handling of atom equivalency**
  - Introduce atom IDs and move away from legacy equivalent-atom sets, at least for input files.
    - *Implementation note:* Enforced internally so reference atoms precede those referencing them.
  - Make "equivalent-to" an intrinsic atom property (e.g., `equivalent_to: <atom_id>`).
  - Make symmetry operation a secondary property of "equivalent-to" (e.g., symmetry transform applied relative to reference atom).
- [x] **Make spin state (up or down) an intrinsic atom property**
  - Specify local spin state ($\uparrow$ / $\alpha$ or $\downarrow$ / $\beta$) directly as an intrinsic property of the atom, rather than inferring it by "flipping" a reference atom.
- [x] **Make virtual an intrinsic atom property**
  - Explicitly mark ghost/coordination atoms (`is_virtual: true`) to include them as external potential/coordination neighbors in local atomic calculations, while cleanly excluding them from the extended (molecular) state in the EWMO step.

---

## 3. Structural Modeling & Graphical Visualization

- [x] **Graphical builder that can handle virtual, equivalent, spin-flipped, and symmetry-generated atoms**
  - Interactive setup for complex surface clusters, symmetry generators, local spin assignments, and ghost/virtual coordination centers (`phatpsy/builder.py`, `phatpsy/spin.py`, `phatpsy/visualizer.py`).
- [x] **Visualizer for embedded atoms and extended system (finite cluster or pseudo-periodic supercell)**
  - Enhanced 3D WebGL rendering with metallic cages for fixed substrate atoms, translucent wireframe spheres for virtual coordination centers, and alpha/beta spin polarization indicators.
  - [x] **Creating supercell for pseudo-periodic representation**
    - Generate translationally replicated unit cells to model periodic surfaces and slabs (`phatpsy/supercell.py`), supporting bulk elements (FCC, BCC, Diamond), oriented slabs ((100), (111), (110)), substrate layer freezing (`QFIXED=T`), virtual ghost coordination perimeters (`QVIRTL=T`), high-symmetry adsorbate placement (top, bridge, hollow), and multi-format exporters (POSCAR, CIF, XYZ, Deck). Accessible via Python API and CLI (`phatpsy supercell`).
  - [x] **3D Volumetric Visualizations, Isosurfaces & Gaussian Cube Export**
    - Native 3D spatial evaluation of molecular orbitals, total electron density, spin density, deformation density, and Poisson electrostatic potential (ESP) via 3D FFT convolution (`phatpsy/volumetric.py`).
    - Watertight isosurface extraction via Marching Tetrahedra (`phatpsy/marching_cubes.py`) and property mapping onto isosurfaces with standard Red-White-Blue colormap (Red = negative/nucleophilic, White = neutral, Blue = positive/electrophilic).
    - Dual-phase lobe rendering for orbitals and interactive WebGL controls in Three.js (`phatpsy/visualizer.py`).
    - Full Gaussian Cube format (`.cube`) import and export compatible with VMD, PyMOL, Avogadro, ChimeraX, and Jmol. Accessible via Python API and CLI (`phatpsy cube`).


---

## 4. Potential Energy Surfaces, Reaction Pathways & Dynamics

- [x] **Geometry scans (1D and 2D)**
  - Automated rigid and relaxed 1D/2D potential energy surface (PES) mapping (`phatpsy/pes.py`) with warm restarts, virial theorem monitoring, curve/contour plotting, and JSON serialization. Accessible via Python API (`scan_1d`, `scan_2d`) and CLI (`phatpsy scan`).
  - [x] **Generate internal coordinates from XYZ for geometry scans and MD**
    - Automatic Z-matrix generation and Cartesian NeRF reconstruction (`ZMatrix`), redundant internal coordinate transformations (`BondCoordinate`, `AngleCoordinate`, `DihedralCoordinate`), and analytical Wilson B-matrix evaluation (`InternalCoordinates.b_matrix`) in `phatpsy/coords.py`.
- [x] **Geometry optimization**
  - Gradient-based (BFGS quasi-Newton with trust-region step clamping and Armijo backtracking line search) ground-state structure relaxation (`phatpsy/optimize.py`), supporting frozen/virtual centers and warm wavefunction restarts. Accessible via Python API (`optimize_geometry`) and CLI (`phatpsy opt`).
- [x] **Transition state searching & refinement**
  - Climbing-Image Nudged Elastic Band (CI-NEB) algorithm with Henkelman-Jónsson improved tangents, decoupled spring/perpendicular forces, QuickMin optimizer, and climbing-image saddle point location (`phatpsy/neb.py`). Includes pathway energy profile plotting (`plot_pathway`), multi-frame XYZ trajectory export, interactive 3D WebGL pathway animation in Three.js (`visualize_pathway`, `show_pathway`), and full CLI integration (`phatpsy neb`).
  - **Dimer method (gradient-only saddle refinement)**: Heyden/Bell variant of the dimer method (`phatpsy/dimer.py`) for high-precision saddle point polishing without Cartesians or Hessians in the loop (3-4 gradient evaluations per cycle). Supports dual entry points: (a) automated handoff from CI-NEB climbing image (`refine_neb_saddle`), and (b) single-ended search from a reactant/perturbed geometry (`dimer_search`). Includes analytical Müller-Brown testbed benchmark, finite-difference step size vs. noise floor calibration ($h = 0.001\ \text{Bohr}$, $\Delta R = 0.005\ \text{Bohr}$), multi-frame XYZ trajectory export, structured JSON checkpointing, interactive 3D WebGL Three.js playback player with real-time curvature/energy profiles (`visualize_dimer`, `show_dimer`), and full CLI integration (`phatpsy dimer`).
- [x] **Reaction pathways (intrinsic and dynamic)**
  - Intrinsic Reaction Coordinate (IRC) path integration along mass-weighted steepest descent ($\mathbf{x}_i = \sqrt{M_i}\mathbf{r}_i$, $d\mathbf{x}/ds = -\mathbf{g}_x/\|\mathbf{g}_x\|$) in `phatpsy/irc.py`.
  - Transition state initiation along positive and negative directions of imaginary normal mode displacement ($\mathbf{Q}_{\text{imag}}$) to connect transition states with reactant and product minima.
  - Forward and reverse activation barrier calculation ($\Delta E_f^\ddagger$, $\Delta E_r^\ddagger$) and reaction energy ($\Delta E_{\text{rxn}}$).
  - Multi-frame XYZ trajectory export, structured JSON checkpointing, 2D reaction coordinate profile plotting (`plot_pathway`), and interactive 3D WebGL Three.js playback player with reaction coordinate scrubbing (`visualize_irc`, `show_irc`).
  - Full CLI integration (`phatpsy irc`).
- [x] **Molecular dynamics (MD)**
  - Direct Born-Oppenheimer Molecular Dynamics (BOMD) on the PHATPSY semi-empirical potential energy surface (`phatpsy/md.py`).
  - Symplectic Velocity Verlet integrator conserving phase-space volume and total energy.
  - Comprehensive thermodynamic ensembles: Microcanonical ($NVE$), Berendsen weak-coupling thermostat ($NVT$), velocity rescaling ($NVT$), and stochastic Andersen collision thermostat ($NVT$).
  - Full Maxwell-Boltzmann velocity initialization with center-of-mass linear momentum removal and support for fixed substrate constraints (`fixed=True`) and virtual coordination centers.
  - Multi-frame XYZ trajectory export, structured JSON checkpointing, dynamics plotting (`plot_dynamics`), and interactive 3D WebGL trajectory player with timeline controls in Three.js (`visualize_trajectory`, `show_trajectory`).
  - Full CLI integration (`phatpsy md` / `phatpsy bomd`).
- [x] **Harmonic vibrational frequency analysis & thermochemistry**
  - Full numerical Cartesian Hessian matrix evaluation (`phatpsy/vibrations.py`) with central and forward finite differences.
  - Mass-weighted Cartesian Hessian and Eckart translational & rotational projection operator ($\mathbf{P} = \mathbf{I} - \mathbf{D}_{\text{TR}}\mathbf{D}_{\text{TR}}^T$) to cleanly decouple zero-frequency rigid-body modes.
  - Harmonic vibrational frequencies ($\text{cm}^{-1}$), reduced masses ($\mu_k$ in amu), force constants ($k_k$ in $\text{mDyne}/\text{Å}$), and Cartesian normal mode displacement vectors ($\mathbf{Q}_k$).
  - Automatic saddle-point / transition state classification ($N_{\text{imag}} = 1$).
  - Complete ideal-gas / Rigid-Rotor Harmonic Oscillator (RRHO) statistical thermochemistry: Zero-Point Vibrational Energy (ZPVE), thermal internal energy $U(T)$, enthalpy $H(T)$, entropy $S(T)$ (Sackur-Tetrode translation, rotational, vibrational, electronic), Gibbs free energy $G(T) = H(T) - T \cdot S(T)$, and constant-volume heat capacity $C_v(T)$.
  - Interactive 3D WebGL Three.js normal mode player with sinusoidal harmonic animation, play/pause, amplitude slider, speed control, and vector displacement arrows (`visualize_vibrations`, `show_vibrations`).
  - Full CLI integration (`phatpsy freq` / `phatpsy vib` / `phatpsy vibrations`).


---

## 5. Electronic, Optical & Magnetic (EOM) Properties

- [x] **EOM properties (`phatpsy.eom`)**
  - **Permanent Electric Dipole Moment ($\boldsymbol{\mu}$)**: Nuclear and electronic dipole moments in atomic units ($e \cdot a_0$) and Debyes with automatic reference origin invariance (center of nuclear charge $\mathbf{R}_0$). Exact physical inversion symmetry verified for homonuclear systems ($N_2$ dipole magnitude $< 10^{-5}\text{ Debye}$).
  - **Optical Polarizability Tensors ($\boldsymbol{\alpha}(\omega)$)**: Frequency-dependent dynamic polarizabilities and static polarizabilities $\boldsymbol{\alpha}(0)$ computed via Sum-Over-States (SOS) perturbation theory across occupied-to-virtual transitions, reporting full $3 \times 3$ tensors, isotropic mean polarizabilities (a.u. and $\text{Å}^3$), and optical anisotropies. Preserves transverse cylindrical symmetry ($\alpha_{xx} = \alpha_{yy}$) and normal/anomalous dispersion.
  - **Magnetic Susceptibility Tensors ($\boldsymbol{\chi}$)**: Langevin-Van Vleck diamagnetic response from electronic second moments, Van Vleck temperature-independent paramagnetism from orbital angular momentum operators $\hat{\mathbf{L}} = -i(\mathbf{r} \times \nabla)$, and Curie paramagnetism for open-shell systems ($S > 0$), reported in atomic units and cgs molar susceptibility ($\text{cm}^3/\text{mol}$).
  - **Electronic Excitations & Simulated UV-Vis Spectroscopy**: Single-particle excitation energies ($\Delta E$), absorption wavelengths ($\lambda = hc/\Delta E$), transition dipole moments ($\boldsymbol{\mu}_{ia} = \langle \psi_i | \mathbf{r} | \psi_a \rangle$), dimensionless oscillator strengths ($f_{\text{osc}} = \frac{2}{3} \Delta E \|\boldsymbol{\mu}_{ia}\|^2$), and continuous Gaussian-broadened absorption spectra with stick transition plotting (`plot_uvvis`).
  - **Interactive 3D WebGL Visualizer**: Embedded Three.js player (`visualize_eom`, `show_eom`) rendering molecular structure, permanent dipole vector arrow, polarizability ellipsoid oriented along principal eigenvectors, and interactive UV-Vis spectrum HUD with orbital transition assignment tooltips.
  - **CLI Subcommand**: `phatpsy eom` for command-line property evaluation, JSON export, UV-Vis plotting, and 3D HTML viewer generation.

---

## 6. State Persistence, Restarts & Transferable Embedded Atoms

- [x] **Saving system state for restarting or starting new geometry**
  - Implement full calculation state serialization (in-memory, HDF5, or structured JSON checkpoints) to allow warm restarts across geometry optimization steps or PES scans.
- [x] **Saving of embedded atom states for inclusion in other systems**
  - Export converged localized atomic states (orbitals, potentials, density matrices) for reuse in other calculations (e.g., as fixed bulk substrate or surface atoms).
  - [x] **Override reading starting orbitals from basis set**
    - Enable direct orbital ingestion from pre-converged embedded atom states rather than re-evaluating atomic STO basis tables.

---

## 7. Architectural Modernization & Infrastructure

- [x] **Full LAPACK Migration**
  - Complete replacement of legacy matrix diagonalizers (EISPAK/Jacobi) with modern, standardized LAPACK calls (`DSPEVD`, `DSPEV`), dynamic workspaces, and modern Fortran 2018 (`phatpsy_lowdin_mod.f90`). Legacy diagonalizer files (`eispak.f`, `jacobi.f`, `tred3.f`, etc.) retired from modern build targets.
- [x] **Pure In-Memory Storage**
  - Eliminate all legacy scratch file/tape units (`Phatpsy.DAUnit1`, `Unit2`, `Unit3`, `Unit4`, `Temp`, `Elem<IELEM>`) in favor of dynamic in-memory arrays and stream buffers. Zero binary disk scratch files are created during calculation.
- [x] **Clean Fortran Module & Filename Standardization**
  - Eliminate the `phatpsy_` prefix and `_mod` suffix across modern Fortran 2018 files and module declarations (e.g., `phatpsy_state_mod.f90` -> `state.f90` with `module state`, `phatpsy_contrl_mod.f90` -> `contrl.f90` with `module contrl`, etc.).
  - Update CMake targets, f2py wrappers, Python driver/state accessors, and inter-module `use` statements.
- [x] **Modernize active legacy FORTRAN routines (*.f) to Fortran 2018 modules (*.f90)**
  - Audit and inventory the 109 legacy `src/*.f` routines to categorize into:
    - *Retire/Delete:* Obsolete EISPAK/Jacobi diagonalizers, card-reader I/O, and scratch tape routines.
    - *Retain & Modernize:* Active mathematical kernels (e.g., `gensab`, `genvab`, `symop`, `symxyz`).
  - Convert retained routines using the `routine-modernization` skill:
    - [x] **Symmetry module (`src/symmetry.f90`):** Modernize `symop.f`, `symxyz.f`, and `genrot.f` into clean free-form Fortran 2018 with `implicit none` and dynamic workspace allocation.
    - [x] **Two-center module (`src/twocenter.f90`):** Modernize `gensab.f`, `genvab.f`, and `rbeta.f` into clean free-form Fortran 2018 with `implicit none`.
    - [x] **Linear algebra & vector utilities (`src/linalg.f90`):** Modernize `addmat.f`, `asum.f`, `sum.f`, `dcopy.f`, `dotprd.f`, `scmult.f`, `zero.f`, `erase.f`, `flip.f` into clean free-form Fortran 2018 with `implicit none` and intrinsic/BLAS equivalents.
    - [x] **Elliptical coordinate quadratures & expansion kernels (`src/elliptical.f90`):** Modernize `absum.f`, `anmbnm.f`, `ascale.f`, `bscale.f`, and `ogen.f` into clean free-form Fortran 2018 with `implicit none`, numerical stability across small/large arguments, and global compatibility wrappers. Retired legacy `.f` files to `legacy/`.
    - [x] **Group 3: Model potential & normalization kernels (`src/modelpot.f90`):** Modernize `genfac.f`, `genbc.f`, `gennrm.f`, `intnrm.f`, `pkepot.f`, `epkpot.f`, and `dampv.f` into clean free-form Fortran 2018 with `implicit none`, exact analytical factorials/binomials/normalizations, vectorized potential damping, and global external ABI compatibility wrappers. Retired 7 legacy `.f` files to `legacy/`.
    - [x] **Group 4: Advanced Linear Algebra & Matrix Transforms (`src/linalg.f90`):** Modernize `trisq.f`, `tritri.f`, `uthu.f`, `invert.f`, `matinv.f`, `schmid.f`, `normlz.f`, `round.f`, `maxovl.f`, `hesbrg.f`, `heseig.f`, `hesvec.f`, and `norhes.f` with LAPACK LU factorization (`DGETRF`/`DGETRI`), determinant calculation, in-place solver (`DGETRS`), Schmidt orthonormalization, eigenvector normalization, numeric rounding, and maximum overlap reordering. Retired 13 legacy `.f` files to `legacy/`.
    - [x] **Group 5: Symmetry Projection & Integral Transformations (`src/symmetry.f90`):** Modernize `sympak.f`, `sympro.f`, `sympr1.f`, `sympr2.f`, `putsym.f`, and `rotsab.f` into `symmetry.f90`. Retire 6 legacy `.f` files to `legacy/`.
    - [x] **Group 6: Density Matrices, Wavefunctions & Population Analysis (`src/density.f90`):** Modernize `denmat.f`, `chrgbo.f`, `popout.f`, `wfct.f`, `sfct.f`, `averag.f`, `depth.f`, `occmet.f`, and `neword.f`. Retire 9 legacy `.f` files to `legacy/`.
    - [x] **Group 7: Formatted Output, Spatial Density/Potential Mapping & Diagnostics:** Modernize `output.f`, `outpak.f`, `outvec.f`, `twoout.f`, `timout.f`, `arrmap.f`, `xyzmap.f`, `mapndx.f`, `plotv.f`, `punchv.f`, `bomb.f`, `noble.f`, etc. All legacy `.f` files modernized or retired; `src/` is now 100% pure Fortran 2018 (`.f90`).
    - Convert fixed-form 72-column to free-form Fortran 2018.
    - Enforce `implicit none` and dynamic memory / assumed-shape arrays (`(:,:)`).
    - Eliminate legacy disk unit dependencies in favor of in-memory data structures.
- [x] **Continuous Integration (CI)**
  - Automated GitHub Actions workflow (`.github/workflows/ci.yml`) running the canonical verification loop on push and pull requests with robust OpenBLAS threading controls, unlimited stack allocations, and non-interactive apt provisioning.
