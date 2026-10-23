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
| **Project Milestones & Future Goals** | [Development Roadmap](ROADMAP.md) |
| **Modernizing a Legacy `.f` Routine** | Use the `routine-modernization` skill ([`.agents/skills/routine-modernization/SKILL.md`](.agents/skills/routine-modernization/SKILL.md)) |

---

## 3. Development Workflow Rules

1. **No Silent Approximations:** Never change mathematical approximations, convergence criteria, or fitting formulas without documenting the scientific rationale and citing the relevant dissertation section.
2. **No Regression Without Notification:** All modifications to core Fortran modules or the Python driver must pass the test suite (`pytest tests/`). Any numerical deviation from established baselines must be reported explicitly.
3. **No Unrequested Tool/Dependency Installation:** Do not install global Python packages or system binaries without explicit user consent.

---

## 4. Verification & Definition of Done

1. **Canonical verification loop:** `./make_build_test.sh` — it builds
   the legacy baseline, the modern CLI, and the Python extension, runs
   the N2 smoke tests, and finishes with the full pytest suite.
   *Healthy result:* the script exits 0 and the pytest summary reports
   all tests passed (0 failed, 0 errors).
2. **Quick loop (Python-only changes):**
   `PYTHONPATH=. .venv/bin/pytest tests/ -v`
   *Healthy result:* `N passed` with 0 failed.
3. **Evidence rule:** Work is not done until the verification loop has
   been run on the final state of the change. When reporting work
   complete, include the tail of the verification output (at minimum
   the pytest summary line) in the report. A claim that tests pass,
   without the output, is not evidence.
4. **Deviations:** Any numerical deviation from established baselines
   surfaced by verification is reported explicitly, per Section 3,
   Rule 2 — never silently absorbed.
5. **Mistake-twice rule:** When the same mistake is made twice, the
   correction goes into this file (or the relevant file under
   `.agents/rules/`) in the same commit as the fix, so the next
   session does not repeat it.

---

## 5. Protected Paths

These paths carry the project's baselines and generated data. The
restrictions below are the advisory layer; CI (`.github/workflows/ci.yml`)
is the deterministic backstop.

1. **Generated basis data — `phatpsy/data/*.json`**
   (e.g. `clementi_sto.json`, `thakkar_sto.json`): change these only
   by rerunning the corresponding ingest script
   (`phatpsy/ingest_clementi.py`, `phatpsy/ingest_thakkar.py`) and
   committing the regenerated output together with the ingest change.
   Never hand-edit values in these files.
2. **Frozen legacy baselines — `data/*_legacy.stdout`:** never modify.
   Modern reference outputs in `data/` (e.g. `n2_modern.stdout`) may
   change only alongside an explicit numerical-deviation report, per
   Section 3, Rule 2.
3. **Tests during a bug fix — `tests/`:** while fixing a bug, do not
   edit the test that demonstrates the bug. If a test appears to be
   wrong, stop and flag it for review rather than changing it mid-fix:
   a test that predates the fix and that the fix did not touch is the
   proof the bug is gone.

