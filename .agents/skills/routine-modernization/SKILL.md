---
name: routine-modernization
description: Repeatable step-by-step procedure for modernizing legacy FORTRAN 66 fixed-form routines in PHATPSY into clean Fortran 2018 modules with LAPACK linear algebra and in-memory data structures.
---

# Routine Modernization Skill

Use this skill whenever converting a remaining legacy fixed-form Fortran routine (`src/*.f`) into a modern Fortran 2018 module (`src/*.f90`) or modernizing an existing mathematical kernel.

---

## Step 1: Pre-Modernization Audit
1. **Identify Provenance & Role:**
   - Locate the routine in [`phatpsy_documentation/PHATPSY_Routine_Dictionary.md`](file:///Users/jacksmith/phatpsy2/phatpsy_documentation/PHATPSY_Routine_Dictionary.md) and [`phatpsy_documentation/PHATPSY_Flow_Diagram.md`](file:///Users/jacksmith/phatpsy2/phatpsy_documentation/PHATPSY_Flow_Diagram.md).
   - Look up the corresponding equation and theory in [`docs/science/equation_mapping.md`](file:///Users/jacksmith/phatpsy2/docs/science/equation_mapping.md) and [`PhD_Thesis/PHATPSY.tex`](file:///Users/jacksmith/phatpsy2/PhD_Thesis/PHATPSY.tex).
2. **Inspect Legacy State Dependencies:**
   - Catalog all `COMMON` blocks, `EQUIVALENCE` statements, and scratch disk units (`IUNIT(1)`, `IUNIT(2)`, `IUDUMP`, etc.).

---

## Step 2: Implementation Guidelines
1. **Module Structure:**
   - Create or update the modern `.f90` module using free-form Fortran 2018 with `implicit none`.
   - Import explicit standard types from `iso_fortran_env` (`real64`, `int32`).
2. **State & Memory Migration:**
   - Access global quantum state and dimensions via `use state`.
   - Never reintroduce static dimension limits or common blocks. Use dynamic arrays allocated in `mol_ws`.
3. **Linear Algebra Modernization (LAPACK Mandate):**
   - Replace legacy hand-rolled matrix algorithms (e.g. Jacobi iterations, EISPAK `tred3`/`imtqlv`) with standard LAPACK calls (`dsyev`, `dgels`, `dgemm`).
4. **Pure In-Memory Storage:**
   - Replace any disk I/O streaming to temporary units with in-memory buffers in [`integrals.f90`](file:///Users/jacksmith/phatpsy2/src/integrals.f90).

---

## Step 3: Build System & Interface Integration
1. **Update CMake:**
   - Add the new `.f90` module to `MODERN_F90_MODULES` in [`CMakeLists.txt`](file:///Users/jacksmith/phatpsy2/CMakeLists.txt).
   - Remove the old `.f` file from `REPLACED_LEGACY_FILES`.
2. **Update f2py Bridge (if exposed to Python):**
   - Update [`src/_phatpsy_core.pyf`](file:///Users/jacksmith/phatpsy2/src/_phatpsy_core.pyf) and [`_phatpsy_core.pyi`](file:///Users/jacksmith/phatpsy2/_phatpsy_core.pyi) if Python needs direct access to the new subroutine.

---

## Step 4: Verification & Regression Testing
1. **Compile & Build:**
   ```bash
   cd build && cmake .. && cmake --build . && cd ..
   ```
2. **Execute Test Suite:**
   ```bash
   .venv/bin/pytest tests/
   ```
3. **Verify Physical Invariants:**
   - Check total energy convergence and ensure virial ratio $-V/T \approx 2.00$ remains satisfied.
   - Confirm zero regression on reference decks (`examples/n2.stdin`, `examples/CO.stdin`, `examples/ag13dz.stdin`).
