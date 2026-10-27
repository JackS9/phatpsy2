---
name: verification-review
description: Independent verification pass for PHATPSY 2.0 changes. Runs the canonical build/test loop and the physics sanity checks against the actual working tree, and reports findings without modifying code. Use after implementing a feature or fix, before declaring work complete, and whenever a numerical result or a data file needs an independent check.
---

# Verification Review Skill

Use this skill to verify a change independently of whoever (or whatever)
implemented it. The verdict must rest on the state of the working tree
and on command output — never on the implementer's summary of what
they did.

**Stance: verify only. Do not fix anything.** If a check fails, report
it with evidence; fixing is a separate task.

---

## Step 1: Run the Canonical Verification Loop

```bash
./make_build_test.sh
```

Record the exit status and the pytest summary line verbatim. If the
build fails, stop here and report the failure output.

## Step 2: Physics Sanity Checks

For changes touching scientific routines, data files, or the Python
driver, check the invariants from
`.agents/rules/scientific-integrity.md` against the produced outputs:

1. **Virial ratio:** converged closed-shell systems satisfy
   $-V/T \approx 2.00$; flag any $|-V/T - 2.0| > 0.05$.
2. **Physical energy bounds & electron conservation:** ground state
   energies are variational and negative; total electron counts and
   gross Löwdin populations are strictly non-negative and physically exact.
3. **External benchmark consistency:** calculations match external
   benchmark targets (NIST CCCBDB, CRC Handbook, Herzberg, PySCF)
   within documented tolerances in `phatpsy/benchmarks.py`.
4. **Basis-data changes** (any diff under `phatpsy/data/`): run the
   full-coverage basis tests and the standalone checker —
   ```bash
   PYTHONPATH=. .venv/bin/pytest tests/test_clementi_basis.py -v
   python3 check_clementi_orthonormality.py
   ```
   Orthonormality must hold for **all** elements and both SZ and DZ
   sets (no spot-checking known-good elements), stored energies must
   satisfy the variational ordering (e.g. $E_{SZ} > E_{DZ}$), and no
   placeholder values (e.g. `eigval = 0.0`) may be present.

## Step 3: Diff Audit

Inspect the change's diff (`git diff` / `git status`) and confirm:

1. No protected path was violated (AGENTS.md, Section 5): generated
   data files changed only via their ingest scripts, legacy baselines
   untouched, tests not edited in the course of a bug fix.
2. Any numerical deviation from baselines is explicitly reported in
   the change summary.
3. No new dependency or tool was installed without user consent
   (AGENTS.md, Section 3, Rule 3).

## Step 4: Report

Report in this form, and nothing else:

- **Verdict:** PASS or FAIL.
- **Evidence:** the commands run, with the pytest summary line and
  any failing output quoted verbatim.
- **Deviations:** every numerical deviation found, as numbers, or
  "none".
- **Concerns:** anything that could not be verified, and why.
