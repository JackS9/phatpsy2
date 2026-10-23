# PHATPSY 2.0 Development Roadmap & Goals

This roadmap defines the scientific, modeling, algorithmic, and architectural goals for the modernization of **PHATPSY 2.0**.

---

## 1. Scientific Methodology & Physics Kernels

- [ ] **Improve TF Potential fitting**
  - Refine non-linear and linear least-squares parameterization of atomic Thomas-Fermi-Dirac model potentials ($Z_p(r)$, Coulomb, and exchange potentials).
- [ ] **Improve handling of energy-dependence in PK potential**
  - Advance the approximation for energy-dependent Phillips-Kleinman pseudopotentials used for atom-atom partitioning and interatomic Pauli exclusion.
- [ ] **Improve charge equilibration after EWMO**
  - Develop robust Mulliken/Lowdin population analysis and iterative charge-equilibration schemes following Extended Wolfsberg-Helmholtz Molecular Orbital initializations.

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

- [ ] **Graphical builder that can handle virtual, equivalent, spin-flipped, and symmetry-generated atoms**
  - Interactive setup for complex surface clusters, symmetry generators, local spin assignments, and ghost/virtual coordination centers.
- [ ] **Visualizer for embedded atoms and extended system (finite cluster or pseudo-periodic supercell)**
  - Enhanced 3D rendering for embedded active sites within bulk/surface substrate environments.
  - [ ] **Creating supercell for pseudo-periodic representation**
    - Generate translationally replicated unit cells to model periodic surfaces and slabs.

---

## 4. Potential Energy Surfaces, Reaction Pathways & Dynamics

- [ ] **Geometry scans (1D and 2D)**
  - Automated rigid and relaxed 1D/2D potential energy surface (PES) mapping.
  - [ ] **Generate internal coordinates from XYZ for geometry scans and MD**
    - Implement automatic Z-matrix / redundant internal coordinate transformations (bonds, angles, dihedrals) from Cartesian inputs.
- [ ] **Geometry optimization**
  - Gradient-based (BFGS / conjugate gradient) ground-state structure relaxation.
- [ ] **Transition state searching**
  - Saddle-point locating algorithms (e.g., Nudged Elastic Band (NEB) or Eigenvector Following).
- [ ] **Reaction pathways (intrinsic and dynamic)**
  - Intrinsic Reaction Coordinate (IRC) tracing and dynamic steepest-descent trajectories.
- [ ] **Molecular dynamics (MD)**
  - Born-Oppenheimer / direct molecular dynamics integration on the PHATPSY potential surface.

---

## 5. Excited States & Molecular Properties

- [ ] **EOM properties**
  - Equation-of-Motion (EOM) formalisms for excitation energies, transition dipole moments, and response properties.

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
- [ ] **Modernize active legacy FORTRAN routines (*.f) to Fortran 2018 modules (*.f90)**
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
    - [ ] **Group 6: Density Matrices, Wavefunctions & Population Analysis (`src/density.f90`):** Modernize `denmat.f`, `chrgbo.f`, `popout.f`, `wfct.f`, `sfct.f`, `averag.f`, `depth.f`, `occmet.f`, and `neword.f`. Retire 9 legacy `.f` files to `legacy/`.
    - [ ] **Group 7: Formatted Output, Spatial Density/Potential Mapping & Diagnostics:** Modernize `output.f`, `outpak.f`, `outvec.f`, `twoout.f`, `timout.f`, `arrmap.f`, `xyzmap.f`, `mapndx.f`, `plotv.f`, `punchv.f`, `bomb.f`, `noble.f`, etc. Retire remaining legacy `.f` files.
    - Convert fixed-form 72-column to free-form Fortran 2018.
    - Enforce `implicit none` and dynamic memory / assumed-shape arrays (`(:,:)`).
    - Eliminate legacy disk unit dependencies in favor of in-memory data structures.
- [x] **Continuous Integration (CI)**
  - Automated GitHub Actions workflow (`.github/workflows/ci.yml`) running the canonical verification loop on push and pull requests with robust OpenBLAS threading controls, unlimited stack allocations, and non-interactive apt provisioning.
