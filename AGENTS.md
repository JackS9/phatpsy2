# PHATPSY 2.0: Agent Project Instructions

These instructions govern all autonomous and pair-programming development work on the **PHATPSY 2.0** project.

---

## 1. Project Mission & Invariants

**PHATPSY 2.0** modernizes the 1970s FORTRAN quantum chemistry package (*Projected Hamiltonian Approach to Polyatomic Systems*, Jack A. Smith, PhD Dissertation, Univ. of Florida, 1980) into a modern, high-performance **Fortran 2018** core with a typed **Python API**, **CMake** build system, and **Three.js** interactive visualization.

### Core Invariants:
1. **Prioritize Modern Standards Over Legacy Parity:** Full migration to **LAPACK** linear algebra (replacing EISPAK/Jacobi) and **in-memory storage** (eliminating scratch disk/tape units) are primary design objectives. Backward compatibility with legacy scratch files or punch-card quirks is not a priority and must not impede modernization.
2. **Scientific Precision & Physical Invariants:** Calculations operate in **atomic units** (Hartrees, Bohrs). Total energies, virial ratios ($-V/T \approx 2.0$), and orbital occupation counts must always satisfy physical boundary conditions.
3. **Clean Decoupling:** Fortran 2018 handles intensive quantum matrix operations and dynamic allocations. Python handles user workflows, geometry synthesis, format conversion, and visualization.
4. **Mandatory Documentation Consultation:** Before modifying any scientific routines or numerical algorithms, you **must** consult the relevant scientific and architectural references linked below.

---

## 2. When to Consult Project Documentation

| Task / Domain | Mandatory Document to Consult |
| :--- | :--- |
| **Physics, Operators, Pseudopotentials** | [Methodology Overview](docs/science/methodology_overview.md) |
| **Dissertation Equations vs Implementation** | [Equation Mapping](docs/science/equation_mapping.md) |
| **Model Potentials & Coulomb/Exchange Fitting** | [Model Potentials Reference](docs/science/model_potentials.md) |
| **Fortran/Python Architecture & Memory Model** | [System Architecture](docs/development/architecture.md) |
| **Regression Tests & Numerical Tolerances** | [Validation Standards](docs/development/validation_standards.md) |
| **Modernizing a Legacy `.f` Routine** | Use the `routine-modernization` skill ([`.agents/skills/routine-modernization/SKILL.md`](.agents/skills/routine-modernization/SKILL.md)) |

---

## 3. Development Workflow Rules

1. **No Silent Approximations:** Never change mathematical approximations, convergence criteria, or fitting formulas without documenting the scientific rationale and citing the relevant dissertation section.
2. **No Regression Without Notification:** All modifications to core Fortran modules or the Python driver must pass the test suite (`pytest tests/`). Any numerical deviation from established baselines must be reported explicitly.
3. **No Unrequested Tool/Dependency Installation:** Do not install global Python packages or system binaries without explicit user consent.
