# PHATPSY 2.0: Agent Project Instructions

These instructions govern all autonomous and pair-programming development work on the **PHATPSY 2.0** project.

---

## 1. Project Mission & Invariants

**PHATPSY 2.0** modernizes the 1970s FORTRAN quantum chemistry package (*Projected Hamiltonian Approach to Polyatomic Systems*, Jack A. Smith, PhD Dissertation, Univ. of Florida, 1980) into a modern, high-performance **Fortran 2018** core with a typed **Python API**, **CMake** build system, YAML configuration, and **Three.js** interactive visualization.

### Core Invariants:
1. **Abandon Legacy Parity & Namelist Input:** Backward compatibility with the legacy 1980 FORTRAN 66/77 implementation, legacy punch-card scratch units, and legacy Namelist input decks (`*.stdin`) is officially abandoned. The project has moved on to modern methodology, standard YAML/JSON input (`CalculationDeck`), typed Python API, and standard LAPACK linear algebra.
2. **Scientific Precision & Physical Invariants:** Calculations operate in **atomic units** (Hartrees, Bohrs). Validation relies on **basic physics-driven sanity checks** and physical boundary conditions:
   - Virial theorem: Neutral ground-state molecules must approach $-V/T \approx 2.00$ ($|-V/T - 2.0| \le 0.05$).
   - Particle conservation: Symmetrically orthogonalized Löwdin populations must yield strictly non-negative gross populations and physical total electron counts.
   - Variational bounds: Electronic energies and wavefunctions must be stable and self-consistent.
   - Translational/rotational invariance: Dipole moments and molecular properties must remain invariant under coordinate translation.
3. **External Scientific Benchmarks:** Rather than attempting to match frozen historical legacy outputs, scientific accuracy is judged against external experimental and theoretical benchmarks (NIST CCCBDB, CRC Handbook, Herzberg, Somorjai & Li, and standard quantum chemistry packages like PySCF/Gaussian) as implemented in `phatpsy/benchmarks.py`.
4. **Clean Decoupling:** Fortran 2018 handles intensive quantum matrix operations and dynamic allocations. Python handles user workflows, geometry synthesis, format conversion, and visualization.

---

## 2. When to Consult Project Documentation

| Task / Domain | Mandatory Document to Consult |
| :--- | :--- |
| **Physics, Operators, Pseudopotentials** | [Methodology Overview](docs/science/methodology_overview.md) |
| **Dissertation Equations vs Implementation** | [Equation Mapping](docs/science/equation_mapping.md) |
| **Model Potentials & Coulomb/Exchange Fitting** | [Model Potentials Reference](docs/science/model_potentials.md) |
| **Fortran/Python Architecture & Memory Model** | [System Architecture](docs/development/architecture.md) |
| **External Benchmarks & Physics Invariants** | [Validation Standards](docs/development/validation_standards.md) & [`phatpsy/benchmarks.py`](phatpsy/benchmarks.py) |
| **Project Milestones & Future Goals** | [Development Roadmap](ROADMAP.md) |
| **Modernizing a Legacy Routine** | Use the `routine-modernization` skill ([`.agents/skills/routine-modernization/SKILL.md`](.agents/skills/routine-modernization/SKILL.md)) |

---

## 3. Development Workflow Rules

1. **No Silent Approximations:** Never change mathematical approximations, convergence criteria, or fitting formulas without documenting the scientific rationale and citing the relevant dissertation section.
2. **No Regression Without Notification:** All modifications to core Fortran modules or the Python driver must pass the test suite (`pytest tests/`).
3. **No Unrequested Tool/Dependency Installation:** Do not install global Python packages or system binaries without explicit user consent.

---

## 4. Verification & Definition of Done

1. **Canonical verification loop:** `./make_build_test.sh` — builds the modern CLI and the Python extension, and executes the full pytest suite.
   *Healthy result:* the script exits 0 and the pytest summary reports all tests passed (0 failed, 0 errors).
2. **Quick loop (Python-only changes):**
   `PYTHONPATH=. .venv/bin/pytest tests/ -v`
   *Healthy result:* `N passed` with 0 failed.
3. **Evidence rule:** Work is not done until the verification loop has been run on the final state of the change. When reporting work complete, include the tail of the verification output (at minimum the pytest summary line) in the report. A claim that tests pass, without the output, is not evidence.
4. **Mistake-twice rule:** When the same mistake is made twice, the correction goes into this file (or the relevant file under `.agents/rules/`) in the same commit as the fix, so the next session does not repeat it.

---

## 5. Protected Paths

1. **Generated basis data — `phatpsy/data/*.json`**
   (e.g. `clementi_sto.json`, `thakkar_sto.json`): change these only by rerunning the corresponding ingest script (`phatpsy/ingest_clementi.py`, `phatpsy/ingest_thakkar.py`) and committing the regenerated output together with the ingest change. Never hand-edit values in these files.
2. **Tests during a bug fix — `tests/`:** while fixing a bug, do not edit the test that demonstrates the bug. If a test appears to be wrong, stop and flag it for review rather than changing it mid-fix.

