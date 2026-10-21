# Code Conventions for PHATPSY 2.0

These conventions apply when authoring, refactoring, or reviewing code in the PHATPSY 2.0 repository.

---

## 1. Fortran 2018 Standards

1. **Free-Form Syntax:** All modernized code must use free-form Fortran 2018 (`.f90`) with standard indentation (2 or 4 spaces) and descriptive naming.
2. **Explicit Typing & ISO Kinds:**
   - Every module, subroutine, and function must include `implicit none`.
   - Numeric kinds must be drawn explicitly from `iso_fortran_env`:
     ```fortran
     use iso_fortran_env, only: real64, real32, int32, int64
     real(real64)  :: total_energy
     integer(int32) :: natoms
     ```
3. **No Legacy Common Blocks:**
   - Legacy `COMMON /.../` blocks and `EQUIVALENCE` statements are strictly forbidden in modern modules.
   - All state must reside in encapsulated module variables or derived types inside [`src/phatpsy_state_mod.f90`](file:///Users/jacksmith/phatpsy2/src/phatpsy_state_mod.f90).
4. **Dynamic Allocation:**
   - Do not use fixed dimension constants (e.g. `MXNATM`, `MXNORB`, `MXMSTO`) for workspace arrays.
   - Allocate arrays dynamically based on deck dimensions via `compute_derived_dimensions_and_allocate()`.
5. **LAPACK over Legacy Hand-Rolled Math:**
   - Call standard BLAS and LAPACK routines (`dsyev`, `dgemm`, `dgels`, etc.) rather than legacy custom matrix algorithms.

---

## 2. Python Architecture & Conventions

1. **Modern Python Standards:** Python 3.10+ syntax with strict type annotations (`typing.List`, `typing.Dict`, `Optional`, `numpy.typing.NDArray`).
2. **Data Model Validation:**
   - Calculation decks, atoms, and symmetry cards must be represented as structured dataclasses or Pydantic models ([`phatpsy/schema.py`](file:///Users/jacksmith/phatpsy2/phatpsy/schema.py)).
   - Do not pass raw unstructured dictionaries across internal package boundaries.
3. **Pure In-Memory Orchestration:**
   - The Python orchestrator ([`phatpsy/driver.py`](file:///Users/jacksmith/phatpsy2/phatpsy/driver.py)) communicates directly with the compiled f2py C-bridge (`_phatpsy_core`).
   - Do not invoke external shell subprocesses or temporary files unless explicitly requested by CLI file I/O commands.
4. **Testing Invariants:**
   - Unit tests go in [`tests/`](file:///Users/jacksmith/phatpsy2/tests).
   - All new features and refactored routines must include corresponding tests using `pytest`.
