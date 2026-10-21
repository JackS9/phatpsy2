<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# I’m working on development project in Antigravity IDE to modernize a very old FORTRAN program (from the 70s) called PHATPSY that implements a methodology developed for my PhD dissertation. I want to give Antigravity some persistent knowledge of the scientific methodology, as well as some coding guidelines. How do I capture that knowledge in a way that is persistent and automatically drawn on when needed? Is it a skill file or something different?

Use a combination of **rules**, reference documents, and—optionally—skills. For your project, the foundation should be persistent project rules that tell Antigravity what must always be respected and when to consult your scientific documentation. A skill is useful for a repeatable procedure, such as modernizing and validating one legacy routine, but it should not be the sole home for essential scientific constraints. Google’s documentation makes this distinction explicitly: rules capture constraints and invariants; skills capture task-specific procedures.[^1_1][^1_2]

## What belongs where

I recommend this division:


| Material | Where I would put it | How it becomes available |
| :-- | :-- | :-- |
| Essential scientific constraints and project-wide coding guidelines | Root-level `AGENTS.md` | Antigravity treats it as continuously active within its directory scope. |
| Detailed methodology, equations, notation, and explanations | Markdown documents under `docs/science/` | Rules instruct the agent when to read the relevant documents. Ordinary reference files are not automatically injected merely because they exist. |
| More detailed instructions for a particular scientific area | `.agents/rules/scientific-methodology.md` with `model_decision` activation | Antigravity sees the description and reads the full rule when it judges it relevant. |
| Repeatable modernization or validation procedure | `.agents/skills/modernize-fortran/SKILL.md` | Antigravity discovers the skill’s name and description, then loads its instructions when relevant. |

These loading mechanisms follow Antigravity’s current rules and skills documentation; the particular organization above is my recommendation for your project.[^1_2][^1_1]

The key distinction is between “persistently stored” and “always in the model’s current context.” Rules can be always active; detailed rules and skills can instead load on demand. That lets you preserve a substantial body of scientific knowledge without putting the entire dissertation into every interaction.[^1_1][^1_2]

## 1. Start with a small `AGENTS.md`

Put this file at the repository root. Antigravity supports both `AGENTS.md` and `GEMINI.md`; you do not need both. These files use plain Markdown, without YAML frontmatter.[^1_1]

Here is a proposed starting point—not a statement of policies you have already chosen:

```markdown
# Project instructions

## Purpose
Modernize the original dissertation-era FORTRAN implementation
while preserving its scientific methodology and validated behavior.

## Scientific integrity
- Distinguish implementation modernization from changes to the method.
- Before changing a scientific routine, read the relevant documents
  under docs/science/ and its entry in docs/science/code-map.md.
- Preserve documented units, normalization, signs, index conventions,
  approximations, and numerical precision requirements.
- Do not silently replace a specialized algorithm with a familiar
  textbook algorithm or library implementation.
- If the dissertation, source code, and reference results disagree,
  report the discrepancy rather than inventing a reconciliation.
- Clearly distinguish documented facts, inferred behavior, and
  unresolved questions.

## Change discipline
- Keep mechanical refactoring separate from mathematical changes.
- Identify the relevant reference tests before editing a routine.
- Do not change reference outputs or tolerances merely to make tests pass.
- Report tests run, numerical differences, and untested assumptions.
- Treat the original source and historical reference outputs as
  read-only unless explicitly authorized otherwise.

## Project references
- Scientific overview: docs/science/overview.md
- Mathematical conventions: docs/science/conventions.md
- Equations and algorithms: docs/science/methodology.md
- Mapping between mathematics and code: docs/science/code-map.md
- Validation criteria: docs/science/validation.md
- Coding standards: docs/development/coding-guidelines.md
```

Keep your most important coding requirements directly in this file as well. If something must apply to every edit, do not leave it solely in an optional skill.

For longer, mandatory coding standards, Antigravity also supports an explicit include inside a rule:

```markdown
@[Coding guidelines](docs/development/coding-guidelines.md)
```

That syntax inlines the referenced document. Merely writing its pathname does not inline it. Avoid inlining the entire scientific documentation: the current documentation specifies a 24,000-byte per-rule limit after includes and a shared 20,000-token budget for global and always-on rules.[^1_1]

## 2. Capture the methodology as a scientific reference set

For this project, I would make the reference documents an annotated bridge between your dissertation and the implementation, rather than simply a summary of the dissertation.

A useful proposed structure is:

```text
project/
├── AGENTS.md
├── .agents/
│   ├── rules/
│   │   └── scientific-methodology.md
│   └── skills/
│       └── modernize-fortran/
│           └── SKILL.md
├── docs/
│   ├── science/
│   │   ├── overview.md
│   │   ├── conventions.md
│   │   ├── methodology.md
│   │   ├── code-map.md
│   │   ├── validation.md
│   │   └── open-questions.md
│   └── development/
│       └── coding-guidelines.md
├── legacy/
├── src/
└── tests/
```

I would organize the scientific documents around these questions:

- `overview.md`: What problem does the methodology solve? What are its assumptions, approximations, and limits?
- `conventions.md`: What units, basis-function normalization, sign conventions, index orderings, and symbol definitions does it use?
- `methodology.md`: What are the defining equations and algorithms? Include dissertation page and equation references.
- `code-map.md`: Which routines implement which equations? How do mathematical symbols map to legacy variables, arrays, and COMMON blocks?
- `validation.md`: What constitutes an acceptable result? Identify reference cases, expected quantities, and the rationale for tolerances.
- `open-questions.md`: Which interpretations remain uncertain, and what evidence would resolve them?

For each important scientific statement, I recommend recording its provenance—for example, “dissertation equation 3.12,” “observed in routine X,” or “author clarification.” That helps prevent an agent’s plausible interpretation from quietly becoming accepted project knowledge.

Retain the original dissertation as primary evidence, but use reviewed Markdown for the operational guide. In particular, have the agent draft equation transcriptions and explanations for your review rather than treating its first extraction as authoritative.

## 3. Add an automatically selected methodology rule

Antigravity supports a `model_decision` rule specifically for detailed domain guidance. Its description is presented to the agent upfront, and the full rule is read on demand. Rules in `.agents/rules/` require valid YAML frontmatter; missing or invalid `trigger` values cause the rule to be discarded.[^1_1]

For `.agents/rules/scientific-methodology.md`, you could use:

```markdown
---
trigger: model_decision
description: "Use when explaining, modifying, translating, or testing scientific routines, equations, basis functions, matrix elements, numerical approximations, or dissertation methodology in this project."
---

# Scientific methodology guidance

Before proposing changes to scientific code:

1. Read docs/science/overview.md and docs/science/conventions.md.
2. Locate the routine in docs/science/code-map.md.
3. Read the relevant sections of docs/science/methodology.md.
4. Check docs/science/open-questions.md for unresolved issues.
5. Identify applicable validation cases in docs/science/validation.md.

In the proposed change, identify:
- The mathematical operation being implemented.
- The source equations and their provenance.
- The conventions and approximations that must be preserved.
- The evidence that the change preserves the intended method.

Do not infer missing scientific details without labeling them
as hypotheses and requesting review where they affect results.
```

This is where the “automatically drawn on when needed” behavior comes in. However, `model_decision` selection is contextual, not unconditional. That is why I would also keep the instruction “read the relevant scientific documentation before changing scientific routines” in the root `AGENTS.md`.[^1_1]

## 4. Use a skill for the modernization procedure

A skill becomes valuable when you want the same process applied repeatedly across many routines. The current workspace location is `.agents/skills/<skill-name>/SKILL.md`; the older `.agent/skills/` path remains supported. A skill’s description controls how the agent recognizes its relevance.[^1_2]

For example:

```markdown
---
name: modernize-fortran
description: "Modernizes dissertation-era FORTRAN routines while preserving scientific behavior. Use when refactoring, translating, or validating legacy numerical code in this project."
---

# Modernize a legacy scientific routine

1. Read the project rules and relevant scientific reference documents.
2. Explain the original routine before changing it:
   - inputs and outputs;
   - mathematical purpose;
   - variable types and precision;
   - shared state and side effects;
   - assumptions and unresolved questions.
3. Identify existing reference cases and validation criteria.
4. Propose a small change that separates structural modernization
   from algorithmic changes.
5. Implement the approved scope.
6. Run the relevant tests and compare numerical results.
7. Report differences, uncertainties, and any validation gaps.
8. Propose documentation updates for newly confirmed knowledge;
   do not present unverified inferences as established facts.
```

Skills can also contain supporting scripts, examples, and resources. You do not need to duplicate your methodology documents inside the skill; the skill can direct the agent to the shared documents. Antigravity automatically discovers relevant skills, and you can also mention a skill by name to ensure it is used.[^1_2]

## 5. Set it up and check it in a fresh conversation

In the IDE, Google documents the rules interface under the agent side panel’s `…` menu → Customizations → Rules. You can create workspace rules there, or author the files directly. Active skills can also be inspected through the Customizations menu.[^1_2][^1_1]

After adding the files, I suggest testing with a fresh conversation and a prompt such as:

> Before making changes, identify the project instructions that apply to this routine, read its relevant scientific references, and summarize the conventions, unresolved questions, and validation requirements. Cite the repository files you used. Do not edit anything yet.

Check whether its answer actually reflects your documents—not just whether it claims to have read them.

For your immediate needs, I would start with `AGENTS.md` plus a reviewed scientific reference set. Add the methodology rule for selective loading, then add a skill once you have settled on a repeatable modernization-and-validation process. That gives you durable project knowledge, explicit loading instructions, and a reusable procedure without conflating the three.

<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_3][^1_4][^1_5][^1_6][^1_7][^1_8][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: https://antigravity.google/docs/rules/

[^1_2]: https://antigravity.google/docs/skills/

[^1_3]: https://antigravity.google/docs/migration/workflows-to-skills/

[^1_4]: https://antigravity.google/docs/permissions/

[^1_5]: https://www.firecrawl.dev/blog/antigravity-skills

[^1_6]: https://codelabs.developers.google.com/getting-started-with-antigravity-skills

[^1_7]: https://atamel.dev/posts/2025/11-25_customize_antigravity_rules_workflows/

[^1_8]: https://www.reddit.com/r/google_antigravity/comments/1poybmm/how_are_you_providing_rules_system_instructions/

[^1_9]: https://antigravityskills.com/skills

[^1_10]: https://www.c-sharpcorner.com/article/what-are-awesome-claude-skills-and-how-to-install-them-in-google-antigravity

[^1_11]: https://discuss.ai.google.dev/t/bug-ag-is-not-seeing-the-workspace-rules-and-workflows/118065

[^1_12]: https://github.com/rmyndharis/antigravity-skills

[^1_13]: https://www.reddit.com/r/google_antigravity/comments/1qvrxxx/how_do_you_actually_set_up_antigravity_agent/

[^1_14]: https://javascript.plainenglish.io/advanced-antigravity-mastery-rules-mcps-workflows-and-agent-mode-8d2081fe64b2

[^1_15]: https://www.reddit.com/r/AISEOInsider/comments/1w8kt5r/google_antigravity_full_course_shows_5_agent/

[^1_16]: https://www.perplexity.ai/search/39bfee07-01db-4301-b3a4-40808aaeaf17

[^1_17]: https://www.perplexity.ai/search/b1b971a8-27f1-4f51-87e0-a5cd8e534753

[^1_18]: https://www.perplexity.ai/search/5760e8fa-cd3b-4400-8085-b3ca196f07d9


---

# Could I get Antigravity to do this itself based on this chat, my dissertation (stored in the project), the current codebase (including legacy code), and the history (knowledge) it already has about the modernization so far?

**Yes**—you can ask Antigravity to build this project knowledge structure itself from the dissertation, legacy and modernized code, existing project documents, and whatever modernization history is actually available to it. In fact, its documented `/learn` command is intended to turn lessons from recent interactions into persistent rules or skills. For your larger scientific-documentation task, I would use an explicit, staged request rather than rely on `/learn` alone.[^2_1]

The important qualification is “history it has.” Do not assume that everything discussed in previous sessions is available in the current one. Google explicitly distinguishes conversation context from durable customizations: lessons need to be written into rules or skills to benefit future sessions reliably.[^2_1]

## Give it this chat, then ask for an evidence-based reconstruction

Paste the relevant parts of this conversation into Antigravity, or save them in the project as a Markdown reference. Tell it that this conversation proposes an organizational approach—not an authoritative description of your scientific methodology.

Then have it inventory what it can actually access before it creates anything. That will distinguish:

- The dissertation and other primary scientific sources.
- The original implementation.
- The current implementation and tests.
- Documented modernization decisions.
- Available conversation context and saved artifacts.
- Missing history or unresolved scientific interpretations.

My recommendation is to separate discovery and planning from file creation. The principal risk is not that it cannot write the files; it is that it could turn a plausible inference about your science or past decisions into an apparently authoritative rule.

## A prompt you can paste into Antigravity

```text
I want you to establish durable, repository-based project knowledge
for this scientific-code modernization project (PHATPSY 2).

Use the discussion PHATPSY_Modernization_Agent_Environment.md as guidance for organizing the knowledge, not as an authoritative scientific source.

Sources to investigate:
- My dissertation, stored in this project at PhD_Thesis/PHATPSY.pdf
- The original legacy FORTRAN code.
- The current modernized implementation.
- Tests, reference inputs and outputs, build configuration, and
  existing documentation.
- Git history, commit messages, and recorded modernization decisions.
- Relevant conversation context, summaries, and saved artifacts that
  you can actually access.

Do not assume you can access all previous conversations. Identify
what history is available and what is missing.

PHASE 1 — READ-ONLY DISCOVERY

Do not modify files yet.

1. Inspect existing project instructions, rules, skills, and
   documentation before proposing anything new.

2. Locate the dissertation and determine whether its text and
   equations can be read reliably. Report extraction problems,
   ambiguous symbols, or inaccessible sections.

3. Inventory the legacy and modernized code, tests, reference
   results, and available modernization records.

4. Reconstruct:
   - The scientific purpose and scope of the methodology.
   - Its mathematical conventions, assumptions, and approximations.
   - The mapping between dissertation equations and legacy routines.
   - The mapping between legacy and modernized implementations.
   - Established coding and validation practices.
   - Confirmed modernization decisions and unresolved questions.

5. Distinguish carefully among:
   - Statements explicitly documented in the dissertation.
   - Behavior observed in the legacy code.
   - Behavior in the current implementation.
   - Decisions explicitly confirmed by me.
   - Your inferences or proposed recommendations.

6. For substantive claims, record provenance:
   dissertation chapter/page/equation, repository file and routine,
   relevant commit, test case, or identifiable conversation artifact.
   Do not invent references.

7. Do not assume either the dissertation or the code is infallible.
   Report discrepancies without silently resolving them.

8. Propose a minimal knowledge structure using:
   - Root AGENTS.md for essential always-applicable instructions.
   - docs/science/ for detailed scientific references.
   - docs/development/ for coding and development guidance.
   - A selectively activated scientific-methodology rule if useful.
   - A modernization/validation skill if there is a genuinely
     reusable procedure.

9. Prefer updating existing files over creating competing or
   duplicate instructions. Check the rule and skill formats supported
   by this installation.

Deliver an inventory, proposed file structure, a concise list of
scientific uncertainties, and questions that require my judgment.
Wait for approval before creating or changing files.

PHASE 2 — AFTER MY APPROVAL

Create the agreed documentation, rules, and skills.

Requirements:
- Keep always-active instructions short.
- Put detailed science in linked reference documents.
- State explicitly when the agent must consult those documents.
- Do not promote inferred or unverified claims into mandatory rules.
- Mark scientific documentation as draft pending my review.
- Separate scientific requirements from coding preferences.
- Separate preservation of the original method from any proposed
  algorithmic improvements.
- Do not modify source code, reference outputs, tests, numerical
  tolerances, or global agent settings during this task.
- Do not install tools or dependencies without asking.
- Report every file created or modified.

PHASE 3 — VERIFICATION

Demonstrate the setup using one representative scientific routine:
- Identify the applicable project instructions.
- Follow the documentation links to the relevant methodology.
- Explain the routine's scientific purpose and conventions.
- Identify its validation requirements.
- Identify unresolved issues without guessing.

Do not change the routine.
```

The instruction to prefer updates over duplicate files also matches the design of Antigravity’s `/learn` command, which looks for existing rules or skills to improve before creating new ones.[^2_1]

## Treat the different sources as evidence, not a single merged truth

For your project, I would ask Antigravity to preserve these distinctions throughout the documents:


| Source | What I would use it to establish |
| :-- | :-- |
| Dissertation | The stated mathematical method and its original justification |
| Legacy code | What the historical implementation actually computes |
| Current code | What the modernization currently implements |
| Reference calculations and tests | Which behavior has been checked, and under what conditions |
| Your explicit corrections | Authoritative clarifications of intent, with date and context |
| Git history and prior discussions | Why particular implementation decisions were made |
| Agent analysis | Proposed interpretations, clearly labeled until verified |

For example, “the legacy routine uses this normalization” and “the dissertation defines this normalization” should remain separate claims until their equivalence has been established. That is especially useful when documenting historical implementation quirks, suspected errors, or deliberate deviations from the written derivation.

## Use `/learn` for ongoing maintenance

Once the initial structure exists, `/learn` is a good fit for smaller, incremental lessons. Google documents it as reviewing recent interactions, identifying a durable lesson, and proposing a rule or skill in the appropriate scope.[^2_1]

After you settle a scientific or implementation issue, you could say:

```text
/learn

Capture the normalization convention we just confirmed.
Update the existing scientific documentation and applicable rule
rather than creating a duplicate.

Preserve the source reference and distinguish my confirmed
clarification from your earlier inference. Keep this project-scoped.
Show the proposed changes for review.
```

For a longer procedure—such as how to build the legacy executable, generate a reference case, and compare it with the modern version—a skill is appropriate. Antigravity discovers skill names and descriptions at conversation start and loads a skill’s full instructions when it judges the skill relevant.[^2_2]

Finally, perform the representative-routine check in a fresh conversation as well as the setup conversation. That is the practical test of whether you have created usable persistent project knowledge rather than merely a well-informed current chat.

<span style="display:none">[^2_10][^2_3][^2_4][^2_5][^2_6][^2_7][^2_8][^2_9]</span>

<div align="center">⁂</div>

[^2_1]: https://www.antigravity.google/docs/commands/learn/

[^2_2]: https://antigravity.google/docs/ide/skills/

[^2_3]: https://antigravity.google/docs/cli/conversations/

[^2_4]: https://discuss.ai.google.dev/t/bug-feedback-antigravity-2-2-1-models-not-following-global-nor-agents-rules-fake-using-skills-tools/172904

[^2_5]: https://dev.to/gde/google-antigravity-gemini-4-argon-how-to-load-your-global-agents-skills-2cek

[^2_6]: https://www.linkedin.com/pulse/setup-agent-skills-google-antigravity-chris-chen-1zqgc

[^2_7]: https://antigravity.google/docs/skills/

[^2_8]: https://antigravity.google/docs/rules/

[^2_9]: https://codelabs.developers.google.com/getting-started-with-antigravity-skills

[^2_10]: https://www.reddit.com/r/google_antigravity/comments/1poybmm/how_are_you_providing_rules_system_instructions/
