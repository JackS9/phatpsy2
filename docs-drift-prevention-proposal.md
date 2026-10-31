# Docs Drift Prevention — Proposed Rules for Antigravity

**Context:** Lux's Oct 30 docs audit (`docs-audit-findings.md`) found 24 stale/wrong items, 6 formatting issues, and 11 coverage gaps in the 17 hand-written manual pages — all drift between code and prose, at a checkout only days old. The build-generated reference pages (`tools/docs_gen.py`) were sound by design; every finding was in hand-written prose. Each rule below cites the finding IDs it would have prevented.

Proposed home: `AGENTS.md` (repo root) and/or `docs/development/validation_standards.md`. Wording is a proposal — Antigravity should adapt to repo conventions.

## The rules

### 1. Prose never duplicates generated reference (prevents A7–A10, A22, A24 class)
`tools/docs_gen.py` generates the API/CLI/YAML reference at build time from the code — it cannot drift. Hand-written pages must NEVER restate flags, defaults, field lists, or enum values; they link to the reference pages instead. If a prose page needs a flag table, it is generated, not hand-written.

### 2. Docs examples are executed in CI (prevents A1–A6, A19–A20 class — the severe ones)
Add `tests/test_docs_examples.py`, run in the canonical build loop:
- Extract every ```yaml block from `docs/` and load it with `CalculationDeck.from_yaml()` — fails the build on `AttributeError` (this alone catches the A1 "every YAML example unloadable" class).
- Extract every ```python block, AST-parse it, and resolve every `phatpsy.X` attribute access and call signature against the installed package (stub `phatpsy._phatpsy_core` as the audit did). Catches wrong attributes/kwargs (A2–A6, A19, A20).
- Collect every `phatpsy <subcommand> --flag` mention in docs and check each flag exists in the corresponding argparse parser in `phatpsy/cli.py`. Catches A7–A10 class.
- Executable documentation is the only durable fix: prose rots, tests don't.

### 3. No local-machine links (prevents F1)
Ban `file:///` links repo-wide (CI grep check). Source references use relative GitHub blob links (`../../src/...` won't survive the site build — use full `https://github.com/JackS9/phatpsy2/blob/master/...` URLs) or plain code spans.

### 4. Broken links fail the docs build (prevents F2; enables F1/F2 detection)
`mkdocs.yml`: set `validation.links.not_found: warn` **and** make the Deploy Docs workflow fail on warnings (`--strict` or grep the build log). The current `ignore` setting is what let broken links ship.

### 5. Prose style lint (prevents F3, F5, F6)
A lightweight CI grep check (or a `tools/lint_docs.py`):
- Only `!!!`-style admonitions (MkDocs `admonition` extension); reject GitHub-alert `> [!NOTE]` syntax, which renders as a plain blockquote.
- Every fenced code block carries a language tag.
- User-facing pages lead with YAML terms (`equivalent_to`, `symmetry_op`, `fixed`, `virtual`); legacy namelist names (`ISYM`, `IXYZ`, `QFIXED`, `CDAMP`, …) appear only on science/Fortran pages, always with their modern mapping.

### 6. Feature PRs update the manual (prevents G1–G9 class)
Checklist item in `AGENTS.md`: any commit adding a CLI subcommand, public API, YAML key, or basis-set name must update the corresponding hand-written docs page in the same commit, or state why not in the commit message. Reviewers (human or agent) reject feature PRs with no docs delta.

### 7. Methodology claims cite their source (prevents A13–A15 class)
Phase numbers, benchmark figures, and literature values in `docs/science/*` must cite `ROADMAP.md` or the code location — never memory. A claim without a citation is a draft, not documentation.

### 8. Terminology freeze on retirement (supports the "abandon legacy" invariant)
`AGENTS.md` already invariants legacy-output parity as abandoned. Extend: retired identifiers (`charge_damp`/`potential_damp` YAML keys, `CDAMP`/`VDAMP` namelist, legacy deck formats) must not appear in user-facing docs except in a clearly-marked "retired" callout. (Prevents A17/A20-class confusion where retired names linger as if current.)

### 9. Acceptance values must not be calibration constants (anti-circularity)
A phase/milestone acceptance test fails its purpose if the asserted values also appear as hardcoded fitted constants in the implementation. Rule: acceptance criteria must be *computed* by the code path under test. If a test's expected value appears verbatim — or as a calibration target — in the implementation, the test is circular and the phase is not verified, no matter what the commit message claims. Born from the Phase 6 incident (Oct 30): `PhatpsyDriver.gradient()` hardcoded NIST-calibrated force constants annotated with the very frequencies (N2 2358.6, CO 2169.8 cm⁻¹) the phase "verified," while its docstring claimed analytical Pulay corrections. The reviewer control is simple: for each accepted number, the reviewer must be able to point to the code path that *computes* it. (A CI assist is possible — flag test-expected floats that also appear as literals in the implementation — but reviewer discipline is the real gate.)

## Suggested rollout
1. Land the current fix patch (Lux's `docs-fixes-230903f.patch`, in progress) to get to a clean baseline.
2. Add rules 1–8 to `AGENTS.md` / `validation_standards.md` in one commit.
3. Implement rule 2 (`tests/test_docs_examples.py`) first — it is the highest-leverage single change: it converts the entire manual from prose-that-rots into prose-that-is-tested.
4. Rules 3–5 as a small `tools/lint_docs.py` wired into CI alongside the existing loop.
