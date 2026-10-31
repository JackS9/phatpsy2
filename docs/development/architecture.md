# PHATPSY 2.0 System Architecture

This document describes the software architecture, memory model, and language interoperability of **PHATPSY 2.0**.

---

## 1. High-Level Architecture

PHATPSY 2.0 employs a two-tier hybrid architecture:
1. **Computational Core (Modern Fortran 2018):** High-performance linear algebra, dynamic array allocation, Slater-Type Orbital (STO) evaluation, and quantum state updates.
2. **User Experience & Driver Layer (Python 3.10+):** Input parsing, format conversion (YAML/JSON/.stdin), molecular building, STO basis ingestion, 3D WebGL visualization, and execution orchestration.

```
                    +------------------------------------+
                    |  User Interface / Notebooks / CLI  |
                    |    (phatpsy.visualizer, Three.js)  |
                    +-----------------+------------------+
                                      |
                                      v
                    +-----------------+------------------+
                    |    Python Driver & Schema Layer    |
                    |    (phatpsy.driver, phatpsy.schema)|
                    +-----------------+------------------+
                                      |
                    +-----------------v------------------+
                    |       f2py C-Extension Bridge      |
                    |    (_phatpsy_core.pyf / .pyi)      |
                    +-----------------+------------------+
                                      |
                                      v
+-------------------------------------+--------------------------------------+
|                         Fortran 2018 Core                                  |
|                                                                            |
|  +--------------------+  +----------------------+  +--------------------+  |
|  |      state         |  |      contrl          |  |        scf         |  |
|  | Dynamic Memory     |  | Deck Parsing & Loop  |  | LAPACK DSYEV       |  |
|  +--------------------+  +----------------------+  +--------------------+  |
|                                                                            |
|  +--------------------+  +----------------------+  +--------------------+  |
|  |      analys        |  |       ewmo           |  |     integrals      |  |
|  | LAPACK DGELS Fit   |  | Spectral Density     |  | In-Memory Cache    |  |
|  +--------------------+  +----------------------+  +--------------------+  |
+-------------------------------------+--------------------------------------+
                                      |
                                      v
                    +-----------------+------------------+
                    |       BLAS / LAPACK Libraries      |
                    |    (OpenBLAS / Apple Accelerate)   |
                    +------------------------------------+
```

---

## 2. Memory Model & Elimination of Scratch Files

### Legacy Bottleneck:
The original FORTRAN 66 program partitioned matrix storage across static common blocks (`COMMON /CONTRL/`, `COMMON /WORK/`) and streamed two-center overlap and model potential integrals to direct-access tape/disk scratch files on Units 1, 2, 3, and 4.

### Modern In-Memory Model:
1. **Encapsulated Dynamic State (`state.f90`):**
   - All global state is encapsulated in the `MolecularWorkspace` derived type (`mol_ws`).
   - Dimension parameters (`natom`, `nelem`, `norbt`, `mstot`) are evaluated at runtime from the calculation deck.
   - Arrays are allocated dynamically via `compute_derived_dimensions_and_allocate()`.
2. **In-Memory Integral Cache (`integrals.f90`):**
   - Two-electron and model potential integrals are stored directly in dynamically sized allocatable arrays in RAM.
   - Scratch disk I/O on Units 1 and 2 is completely eliminated.

---

## 3. The f2py Bridge Layer

The Fortran core is compiled into a high-performance Python C-extension (`_phatpsy_core.so`) using `numpy.f2py`:
- **Interface Definition:** [`src/_phatpsy_core.pyf`](../../src/_phatpsy_core.pyf) maps Fortran module subroutines and state arrays to Python callable functions.
- **Type Stubs:** [`_phatpsy_core.pyi`](../../_phatpsy_core.pyi) provides full type hints for VS Code, Pylance, and Pyright, ensuring complete autocomplete and static verification.
- **Python Driver:** [`phatpsy/driver.py`](../../phatpsy/driver.py) interacts with the Fortran modules via memory buffers, allowing end-to-end calculations without writing scratch decks to disk.

---

## 4. Python Package Structure (`phatpsy/`)

- **`driver.py`:** High-level `PhatpsyDriver` orchestrator with cycle-by-cycle damping hooks.
- **`schema.py`:** Strongly typed data classes (`CalculationDeck`, `ControlConfig`, `AtomConfig`, `SymmetrySpec`).
- **`builder.py`:** Geometry parser (XYZ, Angstroms, Bohrs) and automated molecular deck synthesis.
- **`basis.py`:** Slater-Type Orbital (STO) basis library interface for Clementi-Roetti and Thakkar tables.
- **`spin.py`:** Bipartite graph alternating-spin solver for Localized Valence-Bond states.
- **`visualizer.py`:** Embedded Three.js WebGL 3D viewer for Jupyter notebooks and browser CLI.
- **`io.py`:** Two-way converter between legacy `.stdin` punch cards, modern YAML, and structured JSON.
- **`cli.py`:** Multi-subcommand CLI tool (`view`, `build`, `convert`, `elements`, `scan`, `opt`, `neb`, `md`, `freq`, `cube`, `supercell`, `chemistorption`, ...).
- **`coords.py`:** Internal coordinates (`ZMatrix`, `BondCoordinate`, `AngleCoordinate`, `DihedralCoordinate`) and PES scan coordinate machinery.
- **`optimize.py` / `pes.py` / `neb.py` / `dimer.py` / `irc.py` / `md.py`:** Geometry optimization, PES scans, CI-NEB transition states, dimer refinement, IRC pathways, Born-Oppenheimer MD.
- **`vibrations.py` / `eom.py`:** Harmonic vibrations, thermochemistry, and equation-of-motion polarizabilities.
- **`supercell.py` / `subsystem.py`:** Periodic supercells and surface slabs; ONIOM-style subsystem energy decomposition.
- **`volumetric.py` / `marching_cubes.py`:** 3D scalar-field grids (density, ESP, orbitals) and isosurface extraction.
- **`diis.py` / `ewmo.py` / `state.py`:** DIIS acceleration, energy-weighted maximum overlap, calculation state handling.
- **`dispersion.py` / `exchange_dft.py`:** Grimme D3(BJ) dispersion and DFT exchange-correlation kernels (Slater, B88, PBE).
- **`chemisorption.py`:** Transition-metal chemisorption — $C_{4v}$ d-band projection, Blyholder indices, lateral PES scans, vibrational shifts, surface core-level shifts.
- **`benchmarks.py`:** Literature benchmark suite and the composite $L_2$ error-norm metric.
- **`ingest_clementi.py` / `ingest_thakkar.py` / `ingest_adf.py`:** Basis-set ingest pipelines (protected generated data in `phatpsy/data/*.json`).
