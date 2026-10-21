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

## 2. Structural Modeling & Graphical Visualization

- [ ] **Graphical builder that can handle virtual, equivalent, spin-flipped, and symmetry-generated atoms**
  - Interactive setup for complex surface clusters, symmetry generators, local spin assignments, and ghost/virtual coordination centers.
- [ ] **Visualizer for embedded atoms and extended system (finite cluster or pseudo-periodic supercell)**
  - Enhanced 3D rendering for embedded active sites within bulk/surface substrate environments.
  - [ ] **Creating supercell for pseudo-periodic representation**
    - Generate translationally replicated unit cells to model periodic surfaces and slabs.

---

## 3. Potential Energy Surfaces, Reaction Pathways & Dynamics

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

## 4. Excited States & Molecular Properties

- [ ] **EOM properties**
  - Equation-of-Motion (EOM) formalisms for excitation energies, transition dipole moments, and response properties.

---

## 5. Architectural Modernization & Infrastructure

- [ ] **Full LAPACK Migration**
  - Complete replacement of legacy matrix diagonalizers (EISPAK/Jacobi) and fitting routines with standardized LAPACK calls (`DSYEV`, `DGELS`).
- [ ] **Pure In-Memory Storage**
  - Eliminate all legacy scratch file/tape units (`Phatpsy.DAUnit1`, `Unit2`, `Unit3`, `Unit4`, `Temp`) in favor of dynamic in-memory arrays.
- [ ] **Continuous Integration (CI)**
  - Automated GitHub Actions workflow running tests across macOS and Linux runners on push and pull requests.
