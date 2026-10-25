# Prompt for Antigravity: dimer-based transition-state refinement (dimer-over-RFO)

Paste-ready prompt for the PHATPSY 2.0 PES/dynamics work. Context first, then the task, then acceptance criteria. Repo state referenced: master at 20c8566 (Oct 24, 2026), which already contains CI-NEB (`phatpsy/neb.py`, `phatpsy neb`) and BOMD (`phatpsy/md.py`, `phatpsy md`/`bomd`).

## Background and decision

PHATPSY cannot build analytic gradients or Hessians, so all gradients are numerical (finite differences). That rules out Hessian-based saddle-point search as the primary method: partitioned-RFO (P-RFO) demands a good starting Hessian, which here would have to come from an empirical model or from finite-differencing already-noisy gradients — expensive and unreliable. The Letter "Transition States Without Hessians" (Oct 24, 2026) benchmarked this: the dimer method kept converging on 24 small-to-medium reactions where Hessian-based P-RFO failed.

**Decision: dimer-over-RFO.** The saddle-refinement method for PHATPSY 2.0 is the dimer method (Henkelman), Heyden/Bell variant as the reference. The already-landed climbing-image NEB supplies the approximate saddle region; the dimer polishes it to the exact saddle. This does not replace CI-NEB — it completes the mapping: CI-NEB first (gradient-only), dimer second (gradient-only).

Target domain, as before: localized phenomena in extended systems modeled as finite clusters or supercells, tens of atoms at most, usually with local symmetry/equivalency to exploit. Cartesians are acceptable at this size.

## What to build

1. **Müller–Brown testbed first, before any dimer code.** Commit an analytic Müller–Brown test surface with its known minima and saddle, plus a test that runs the dimer on it and checks convergence to the known saddle. No dimer implementation lands without this testbed passing.
2. **Dimer implementation (gradient-only, no Hessian anywhere in the loop).**
   - Heyden/Bell variant as reference: 4 gradient evaluations per cycle, or 3 gradients + 1 energy, designed for noisy quantum-chemical forces.
   - Single-ended: must work from a reactant geometry alone, not only from a NEB pathway.
   - Two entry points: (a) from a CI-NEB climbing image / pathway (the standard handoff), and (b) standalone from a reactant geometry plus an initial displacement direction.
   - Reuse the repo's existing finite-difference gradient machinery (the same evaluation the optimizer uses). Respect frozen (`fixed=True`) atoms and virtual centers the same way the optimizer and NEB do.
   - Python API plus CLI following the existing pattern (`phatpsy neb`, `phatpsy md` → e.g. `phatpsy dimer`), and pathway/trajectory export consistent with the NEB outputs.
3. **Finite-difference step size vs SCF noise floor, measured not guessed.** Before or alongside the dimer, measure the step-size/noise relationship on a real PHATPSY case and document the chosen step sizes and convergence thresholds. This measurement decides everything downstream; do not hard-code a step size without it.
4. **Docs and roadmap.** Document the method in the user manual (docs/user/, MkDocs) following the docs-first rule for scientific routines, and record dimer as the saddle-refinement method under the roadmap's transition-state item.

## Acceptance criteria

- Müller–Brown: dimer converges to the known saddle from a nearby starting point; test committed and passing.
- Real-system check: starting from a CI-NEB climbing image on a small PHATPSY case, the dimer converges to a saddle with gradient norm below a documented threshold, and the result is consistent with the NEB pathway (same saddle region, not a different one).
- Single-ended check: dimer converges from a reactant geometry alone on at least one test case.
- Cost per cycle is as specified (4 gradients, or 3 gradients + 1 energy) and is reported, not assumed.
- No Hessian is constructed anywhere in the dimer path (no full finite-difference Hessian per step).
- Full suite green via the canonical loop (`./make_build_test.sh`), including new tests (e.g. `tests/test_dimer.py`) following the `test_neb.py` / `test_md.py` patterns. CI must pass.
- No silent approximations: step sizes, convergence thresholds, rotation/translation tolerances, and any failure to converge are reported explicitly, never swallowed.

## Constraints (repo rules, unchanged)

Atomic units throughout. LAPACK where linear algebra is needed. No silent approximations. Pytest baselines before new methods are trusted. Do not break existing `neb`, `md`, `opt`, or `scan` behavior. If any part of this spec conflicts with the repo's AGENTS.md or ROADMAP.md, flag the conflict rather than silently overriding it.

## Explicitly out of scope for this task

- Hessian-based P-RFO as the saddle method (decided against; see above).
- Reaction pathways (IRC tracing, dynamic trajectories) — still open, separate task.
- Any change to the CI-NEB or MD implementations except the interface the dimer needs to accept a climbing image as input.
