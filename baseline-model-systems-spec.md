# Baseline Model Systems Specification — PHATPSY 2.0

**Status:** DRAFT for Jack's review (2026-11-03). Physics spec from Jack's notes;
structure and API mapping by Lux. Nothing here is accepted until Jack signs off.
**Purpose:** define the model systems, spin states, and acceptance criteria for
the baseline re-establishment work (ROADMAP: "Baseline Re-establishment [ACTIVE
WORK ITEM]"). These replace the quarantined frozen baselines, which are presumed
invalid (false convergence, nondeterministic default SCF, ignored first-atom
spin="down").

## Global rules

1. **SCF settings for the baseline survey** (per ROADMAP survey): explicit spin
   lists (no `None`, no legacy `spin_flip`), `use_grid_fit`, `atomic_smearing=1e-3`,
   `convergence=1e-8`, 1500-cycle cap, no extension.
2. **Anti-Circularity Invariant applies**: energies/values observed in prior runs
   (noted below as *guardrails*) are sanity checks, NOT acceptance targets. The
   implementation must not be tuned to reproduce them. Acceptance is: SCF
   converges under the survey settings to the specified spin state, with virial
   |−V/T − 2| within a *justified* tolerance (freeze v1.1.0 used 0.05; the new
   tolerance needs justification, not widening — cf. the Nov 1 widen-to-fit kill).
3. **Baselines are captured via ingest scripts only.** `data/validation_baselines/*.json`
   is a protected path: regenerate via the ingest script, never hand-edit.
4. **Atom ordering** in spin lists follows the deck atom order. `net_spin` is
   asserted as (n_up − n_down); mismatch raises (spin API, 1ba3051).

## Systems

### 1. N2 — perfect-pairing reference
- **Purpose:** the canonical PP-VB triple-bond reference; guards the N2 analytical-gradient work (Phases 11.x).
- **Geometry:** TODO (Jack: confirm — existing benchmark deck geometry).
- **Spins:** 3 per N, alternating: N1 `["up","up","up"]`, N2 `["down","down","down"]`, `net_spin=0`.
- **Expect:** converge to the PP-VB basin; virial within tolerance. (Prior guardrail: N2 PP-VB basin unchanged through the spin redesign, 1ba3051.)

### 2. CO — heteronuclear triple bond, dative picture
- **Purpose:** the heteronuclear partner to N2; guards against the unphysical-basin bug class (8ba9ea6: `spin_flip=True` → −84.37 Ha MAX-CYCLES basin).
- **Geometry:** TODO (Jack: confirm).
- **Spins:** C `["up","up","up"]`, O `["down","down","down"]`, `net_spin=0`. C(−)O(+) triple bond, iso-electronic to N2. **No spin flip** (heteronuclear default).
- **Expect:** physical basin. Prior guardrails (NOT targets): E ≈ −112.69 Ha, −V/T ≈ 1.9702.

### 3. NO — doublet radical
- **Purpose:** the open-shell baseline; exercises explicit odd-electron spin assignment.
- **Geometry:** TODO (Jack: confirm).
- **Spins:** N `["up","up","up"]`, O `["down","down"]`, `net_spin=1`. Unpaired electron on N, double bond to O.
- **Expect:** converge with net_spin=1 asserted; virial within tolerance.

### 4. EO (ethylene oxide, oxirane) — frustrated ring, ferromagnetic anomaly
- **Purpose:** THE showcase for the spin-API redesign: a non-alternant 3-membered ring where the perfect-pairing picture breaks down and the anomaly must be *located and reported*.
- **Geometry:** TODO (Jack: confirm).
- **Spins:** each C: 4 assignments; each H: 1; O: **mixed** — 1 up toward C1, 1 down toward C2; all others alternating; `net_spin=0`. In API terms: explicit per-atom lists plus `ferromagnetic_links` identifying the forced parallel-spin pair (default: longest-bond rule; Jack to confirm the pair).
- **Expect:** closed-shell SCF converges (prior guardrail, NOT target: 25 cycles, E ≈ −134.79 Ha, −V/T ≈ 1.9098, per the 8ba9ea6 fix). The physically meaningful output is the *identified* ferromagnetic link.

### 5. AA (acetaldehyde) — relaxed isomerization endpoint
- **Purpose:** unfrustrated product of the EO→AA path; alternating-spin reference for C2H4O composition.
- **Geometry:** TODO (Jack: confirm).
- **Spins:** each C: 4; O: 2; each H: 1; regular alternating; `net_spin=0`.
- **Expect:** converge; virial within tolerance.

### 6. Ni bulk — ferromagnetic reference
- **Purpose:** the itinerant-ferromagnet baseline for the Ni program.
- **Geometry:** TODO (Jack: confirm — bulk lattice, supercell spec).
- **Spins:** all atoms spin-up, 2 unpaired spins per Ni (ferromagnetic).
- **Expect:** ferromagnetic SCF basin; virial within tolerance.

### 7. Ni slab — surface moment prediction target
- **Purpose:** the strongest physical target in the set: reproduce the reduced surface magnetic moment from the spin pattern.
- **Geometry:** TODO (Jack: confirm — slab construction, lattice constant, vacuum).
- **Real atoms (9):** 1 central surface atom (reference, up); 4 surface border atoms (equal to central by symmetry, **down** — antiferromagnetic surface coupling); 4 subsurface atoms (1 as subsurface reference, all equal, **up**).
- **Virtual atoms (3):** 2 virtual subsurface (ghost/embedding atoms completing the subsurface layer for the reference-atom calculation, **opposite spin** of the reference; results used recursively); 1 virtual fixed bulk atom (frozen boundary for the third layer of the 13-atom subsurface cluster, **same spin** as the reference).
- **Spins:** per-atom majority as above (exact per-Ni spin multiplicity — 2 each as in bulk, or otherwise — TODO, Jack to confirm). Net: 5 of 9 real atoms spin-up ≈ **0.56 net spin/atom**, vs experimental **0.6 μB/atom**.
- **Expect:** the 0.56-vs-0.6 correspondence is the validation target. **Dependency:** virtual/ghost and fixed/frozen atom support in the builder (graphical-builder subgoal) — flag if not yet implemented when baselines are captured.
- **Note:** Jack's "5 of 9 ≈ 0.56" is stated as the up-fraction proxy for μB/atom; the precise moment mapping is Jack's to confirm, not Lux's to derive.

## Parked: EO → AA reaction path (PES/dynamics subgoal)
The spin narrative — diradical (C1, O) → ionic (C2+, O−) → (C1+, O−) → (C1+, H−, C=O) → hydride migration (C2→C1) → AA, with the O→H spin-down transfer / C2→C1 spin-up exchange closing the bookkeeping — is an **initial-spin-guess schedule for a minimum-energy-path calculation**, not a baseline item. Park under the PES and dynamics subgoal (goal_271d00f5c371).

## Open questions for Jack
1. Geometries for all seven systems (or pointers to the canonical decks/notebooks).
2. Ni slab: per-atom spin multiplicity (2 each as in bulk?).
3. EO: confirm the ferromagnetic link pair (or accept the longest-bond default).
4. New virial/convergence acceptance tolerances (justified, not widened).
5. data/*.stdout + walkthrough notebook captured outputs still show the old Union Carbide banner — regenerate or leave as historical captures?

---

## Paste-ready implementation prompt (Antigravity, after Jack signs off)

```
Task: capture the new PHATPSY 2.0 validation baselines per the Baseline Model
Systems Specification (7 systems: N2, CO, NO, EO, AA, Ni bulk, Ni slab).

Inputs: geometries and confirmed spin lists from Jack (see spec TODOs). SCF
settings: explicit spin lists, use_grid_fit, atomic_smearing=1e-3,
convergence=1e-8, 1500-cycle cap, no extension.

For each system: build the deck with the specified explicit spins
(net_spin asserted; ferromagnetic_links for EO), run to convergence, record
E, -V/T, cycles, net spin. Acceptance per system is convergence to the
specified spin state with virial within the justified tolerance — the prior
guardrail energies in the spec are sanity checks, NOT targets (Anti-Circularity
Invariant: never tune the implementation to reproduce them).

Write baselines ONLY via the repo's ingest scripts (data/validation_baselines/
is protected — never hand-edit JSON). test_co_frozen_baseline stays xfail until
its baseline is regenerated.

Guardrails: no physics changes to the SCF itself — this is measurement, not
tuning. No widening of tolerances to fit (Nov 1 rule). Run ./make_build_test.sh
on the final state and show output. Do NOT touch the EO→AA path (parked under
PES/dynamics).
```
