# SCF Convergence Robustness Plan (Heteronuclear Diatomics — Legacy `phatpsy_scf` Path)

!!! info "Scope & Architectural Boundary"
    PHATPSY 2.0's primary electronic-structure architecture is the exact-integral localized valence-bond engine (`control.method: localized_vb` / `masked_vb`; Phase 17–18 in `ROADMAP.md`). **This plan covers only the legacy single-center model-potential / Phillips–Kleinman SCF engine (`control.method: phatpsy_scf`),** which survives as a backward-compatible option and as the designated non-bonded spectator-environment treatment (`environment_embedding: model_potential`).

    **Nothing in the localized VB engine or its active milestones is gated by this plan.** Execution is worth doing cleanly and reproducibly, but it must remain strictly isolated from the VB codebase and obey all invariants in `AGENTS.md` and `.agents/rules/scientific-integrity.md`.

---

## 0. Governing Invariants, Scope Discipline & Sequencing Preconditions

### Governing Rules
Execution of this plan is governed by `AGENTS.md`, `.agents/rules/scientific-integrity.md`, and [Validation Standards](validation_standards.md):

1. **Atomic units** throughout (Hartrees, Bohrs, elementary charge).
2. **Honest convergence only:** Every convergence claim must use the landed consecutive-cycle machinery (`|ΔE| < CONVRG` for `SCF_NCONV_REQ = 3` cycles plus the `QLAST` confirmation cycle, `converged=True`, `stop_reason="converged"`). No stage may define convergence down, loosen tolerances to mask oscillation, or treat a cycle-limit exit as converged.
3. **Anti-Circularity Invariant (`AGENTS.md` §1 item 5; [Validation Standards §9](validation_standards.md#9-anti-circularity-standard-executable-documentation)):** Acceptance values must be computed by the active physical code path under test. Fitted parameters or literature reference targets must never be hard-coded into the implementation or used as circular acceptance gates.
4. **Docs-first for physics changes (`AGENTS.md` §3 Rule 1):** Any change to a mathematical approximation, regularization, occupation rule, or fitting procedure must be documented in [Model Potentials Reference](../science/model_potentials.md) with scientific rationale and thesis citation before or alongside code changes.
5. **Protected paths (`AGENTS.md` §5 & §6):** Basis data under `phatpsy/data/*.json` changes only via ingest scripts; frozen baselines under `data/validation_baselines/` are never hand-edited or re-frozen except through `tools/generate_validation_baselines.py` under the `AGENTS.md` §6 procedure (and never from an unconverged run).
6. **Canonical verification loop (`AGENTS.md` §4):** Every execution PR must pass `./make_build_test.sh` (0 failed, 0 errors), including `tests/test_docs_examples.py`.

### Scope Discipline
- **In scope:** Legacy single-center SCF Fortran kernels (`src/scf.f90`, `src/analys.f90`, `src/bloop.f90`, `src/aloop.f90`, `src/modelpot.f90`, `src/state.f90`, `src/contrl.f90`), legacy SCF Python driver/control/response (`phatpsy/scf_control.py`, `phatpsy/driver.py`, `phatpsy/cphf.py`, `phatpsy/schema.py` limited to `phatpsy_scf` controls), diagnostic scripts under `tools/`, legacy SCF tests, and corresponding documentation.
- **Strictly out of scope:** Zero changes to `phatpsy/localized_vb.py`, `phatpsy/vb_driver.py`, `phatpsy/two_center_integrals.py`, `phatpsy/multicenter_integrals.py`, `phatpsy/embedding.py`, or deck-vocabulary renaming (tracked separately in `ROADMAP.md` under *Deck vocabulary*).

### Sequencing Preconditions
Execution of Stages 0–5 **begins only after both preconditions have landed on `master`**:

1. **Phase 18 Stages 3–4** ($\text{NO}^* \to \text{N}^* + \text{O}^*$ dissociation and $\text{N}_2$ catalytic formation network on ferromagnetic $\text{Ni}(100)$) have landed on `master`.
2. **The test/baseline triage branch** has landed on `master`. In particular, `tests/test_correlation_and_dispersion.py::test_milestone8_diatomic_energy_stability_benchmark` (currently quarantined in `tests/conftest.py:26`) and `tests/test_validation_baselines.py::test_co_frozen_baseline` (`tests/test_validation_baselines.py:169-210`) are dispositioned on that triage branch; execution of this plan conforms to that triage outcome rather than re-deciding it.
3. **Branch & merge workflow:** Execution lands on a dedicated feature branch off updated `master` with a pull request opened against `master`. **Jack merges.**

---

## 1. Audit Against `master` Tip (`891f64e`)

Every "already landed" claim and open item from `ROADMAP.md` (Part II, *SCF Convergence Robustness for Heteronuclear Diatomics (CO, N₂)*) was verified directly against the source tree at commit `891f64e` (`Merge pull request #3 from JackS9/docs/roadmap-open-model-items`). Discrepancies between documentation/brief assumptions and the actual code are flagged explicitly below rather than silently glossed over.

### 1.1 Verification of "Already Landed" Claims

| # | Claim | Audit Verdict | Code Evidence at `master` (`891f64e`) | Audit Notes / Discrepancies |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Honest convergence reporting** (`|ΔE| < CONVRG` for 3 consecutive cycles; cycle-limit exits return `converged=False` with `stop_reason`; `phatpsy/scf_control.py` arbitrates extension) | **Verified — holds** | `src/state.f90:96-99, 376` (`SCF_STOP_REASON`, `SCF_NCONSEC`, `SCF_NCONV_REQ = 3`, `SCF_TEDIFF`); `src/contrl.f90:286, 297-330`; `phatpsy/scf_control.py:25-182` (`ScfExtensionPolicy`, `judge_continuation`); `phatpsy/driver.py:211-253, 303-305`; `tests/test_scf_control.py:1-167`. | Two nuances: (a) reaching `SCF_NCONSEC >= 3` sets `QLAST = .true.` (`src/contrl.f90:325`), and the next entry updates `SCF_NCONSEC` before checking `QLAST` (`lines 300-311`), resetting `QLAST = .false.` if the confirmation cycle breaks the streak (`line 323`) — so convergence effectively requires **3 + 1 confirmation = 4** consecutive cycles with `|ΔE| < CONVRG`; (b) defaults are now `convergence=1e-6`, `max_cycles=200` (`src/state.f90:373-374`, `phatpsy/schema.py:198-199`, commit `ce8c658`), and `AdaptiveDamping` is also implemented (`phatpsy/scf_control.py:186-237`, default `False`). |
| 2 | **`run_analys` real-to-integer `nelec` bug fix** | **Verified — holds** | `src/analys.f90:445, 463-471` (`integer(int32) :: nelec_int`, `nelec_int = int(nelec, int32)`, `ncore = NOBLE(nelec_int)`); `src/diagnostics.f90:603-620, 861-868` (`integer(int32) function noble(nucz)`). | Matches `NOBLE`'s `integer(int32)` ABI cleanly. |
| 3 | **Grid-fit model potential** (`use_grid_fit`, default ON) with simplex screening fit and analytical CPHF response | **Verified — holds** | Default ON: `src/state.f90:108, 377`, `src/contrl.f90:441, 464` (`QGRDFT = .true.`), `phatpsy/schema.py:209, 1939` (`use_grid_fit: bool = True`). Simplex NNLS fit: `src/analys.f90:21-100` (`nnls_small`), `119-205` (`fit_screening_simplex`), `648-656` (fallback to `solve_model_potential_fit`). Analytical CPHF response: `phatpsy/cphf.py:77-198` (`_compute_grid_fit_dvcoef`), `533-539` (`atom_uses_grid_fit`), `797-808`. | Two notes: (a) `ROADMAP.md:86` and [Model Potentials §5.3](../science/model_potentials.md) still claim `tests/test_grid_fit.py` tests CO, whereas `tests/test_grid_fit.py:18-24` (`_co_deck`) actually tests closed-shell **HF**; (b) `QGRDFT` is also checked on `Z > 18` heavy-atom branches in `src/scf.f90:469, 608, 728, 812, 927`, `src/atomic.f90:373`, `src/aloop.f90:49`, and `src/bloop.f90:340, 343, 355` (C, N, O have $Z \le 18$ and do not hit those branches). |
| 4 | **`atomic_smearing` (opt-in, default `0.0`)** and claim that **overlap-tracking remains the default when unspecified** | **DOES NOT HOLD AS STATED — FLAGGED** | `src/scf.f90:95-128` (`smear_atomic_occupations`), `131-215` (`effective_atomic_smearing`), `592-602, 625, 684`; `src/state.f90:106, 109, 377`; `phatpsy/schema.py:210, 228-232, 1939`; `tests/test_grid_fit.py:50-52, 105-153`; `docs/user/yaml/overview.md:47`. | **Major code/doc divergence** (commits `8dcd601`, `e393edd`, `ce8c658`): `SMEARW == 0.0d0` is **no longer** "smearing off / pure overlap-tracking" on unseeded UHF atoms (`QNWVEC=F, QOPEN=T, QGFERM=F`). Instead, `effective_atomic_smearing` runs automatic early-cycle annealing on cycles 2–4 ($w = 0.02 \times 2^{-(\text{NCYCL}-2)}\text{ Ha}$ for $\text{gap}_{\text{HL}} < 0.15\text{ Ha}$), a HOMO–LUMO gap floor ($w = \max(10^{-4}, 0.5(10^{-3} - \text{gap}_{\text{HL}}))$ for $\text{gap}_{\text{HL}} < 10^{-3}\text{ Ha}$), and a cycle-5 snap ($10^{-4}\text{ Ha}$), suppressing `MAXOVL`/`NEWORD` whenever $w_{\text{eff}} > 0$. Pure overlap-tracking on unseeded UHF requires `SMEARW < 0` (`src/scf.f90:176`), which **`ControlConfig.__post_init__` rejects with `ValueError`** (`phatpsy/schema.py:228-232`, pinned by `tests/test_grid_fit.py:50-52`). Pure overlap-tracking remains default only on seeded (`QNWVEC=T`), RHF, or `QGFERM` decks. Meanwhile, `docs/user/yaml/overview.md:47` describes automatic mode, whereas `ROADMAP.md:85, 92`, [Model Potentials §6](../science/model_potentials.md), `docs/user/python/setup-and-run.md:241`, `src/state.f90:106`, and `phatpsy/schema.py:210, 230` still say `0` disables smearing. |
| 5 | **First-atom spin fix** (spin-down first atoms start with swapped α/β occupations) | **Verified — holds** | `phatpsy/schema.py:1274-1291` (`_spin_swapped_orbital`), `2039-2041, 2082-2087, 2150-2167` (`to_stdin_text`); `tests/test_spin_first_atom.py:1-90`. | Swaps `occno`, `eigval`, and 4-column `coefficients` for `group_idx == 0` spin-down primary atoms using library defaults; symmetry-equivalent atoms and `group_idx > 0` primaries continue using `QFLIP`. |

### 1.2 Audit of the Five Open Items at `master` Tip (`891f64e`)

Direct inspection and read-only diagnostic runs at `891f64e` establish four additional discrepancies relative to the older `ROADMAP.md` text:

1. **Stark unseeded vs. seeded split on $\text{CO}$ and $\text{N}_2$ (Open Items 1 & 2):**
   Observed behavior at `891f64e` (`use_grid_fit=True`, `spins=["up", "down"]`, `net_spin=0`, `max_cycles=200`, `convergence=1e-6`, `R(CO)=2.132 Bohr`, `R(N2)=2.0744 Bohr`):

   | System | Basis | Seeding (`QNWVEC`) | Occupation Mode | `converged` | Cycles | $E_{\text{tot}}$ (Ha) | $-V/T$ | Last-20 Spread (Ha) | `stop_reason` |
   | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
   | $\text{CO}$ | `DZ` | Unseeded (`F`) | `atomic_smearing=0.0` (auto) | `False` | 200 | `-112.350981` | `1.9742` | `4.88e-3` | `not_converged: energy changes not decreasing` |
   | $\text{CO}$ | `DZ` | Unseeded (`F`) | `atomic_smearing=1e-3` | `False` | 200 | `-113.401482` | `1.9688` | `3.57e-1` | `not_converged: periodic limit cycle` |
   | $\text{CO}$ | `DZ` | Unseeded (`F`) | `SMEARW=-1.0` (pure `MAXOVL`) | `False` | 200 | `-112.351971` | `1.9742` | `4.89e-3` | `not_converged: energy changes not decreasing` |
   | $\text{CO}$ | `DZP` | Unseeded (`F`) | `atomic_smearing=0.0` (auto) | `False` | 200 | `-111.202365` | `1.9658` | `1.08e-1` | `not_converged: predicted 352 > limit 200` |
   | $\text{CO}$ | `DZP` | Unseeded (`F`) | `atomic_smearing=1e-3` | `False` | 200 | `-114.077723` | `1.9885` | `4.29e-1` | `not_converged: predicted 215 > limit 200` |
   | $\text{CO}$ | `DZ` | `hybridization="sp"` (`T`) | `atomic_smearing=0.0` (`MAXOVL`) | **`True`** | 72 | `-112.703665` | `1.9874` | `2.13e-4` | `converged` |
   | $\text{CO}$ | `DZ` | `hybridization="sp"` (`T`) | `atomic_smearing=1e-3` | **`True`** | 101 | `-112.838605` | `1.9882` | `6.68e-5` | `converged` |
   | $\text{CO}$ | `DZP` | `hybridization="sp"` (`T`) | `atomic_smearing=0.0` (`MAXOVL`) | **`True`** | 60 | `-112.576316` | `1.9775` | `1.05e-4` | `converged` |
   | $\text{CO}$ | `DZP` | `hybridization="sp"` (`T`) | `atomic_smearing=1e-3` | **`True`** | 57 | `-112.744118` | `1.9784` | `1.16e-4` | `converged` |
   | $\text{N}_2$ | `DZ` | Unseeded (`F`) | `atomic_smearing=0.0` (auto) | `False` | 200 | `-108.966995` | `1.9503` | `1.91e-3` | `not_converged: energy changes not decreasing` |
   | $\text{N}_2$ | `DZ` | Unseeded (`F`) | `atomic_smearing=1e-3` | `False` | 200 | `-109.510170` | `1.9433` | `2.30e-1` | `not_converged: energy changes not decreasing` |
   | $\text{N}_2$ | `DZ` | Unseeded (`F`) | `SMEARW=-1.0` (pure `MAXOVL`) | `False` | 200 | `-108.968674` | `1.9503` | `1.91e-3` | `not_converged: energy changes not decreasing` |
   | $\text{N}_2$ | `DZP` | Unseeded (`F`) | `atomic_smearing=0.0` (auto) | `False` | 200 | `-107.450154` | `1.9380` | `1.13e-2` | `not_converged: energy changes not decreasing` |
   | $\text{N}_2$ | `DZP` | Unseeded (`F`) | `atomic_smearing=1e-3` | `False` | 200 | `-109.407014` | `1.9558` | `1.08e-1` | `not_converged: periodic limit cycle` |
   | $\text{N}_2$ | `DZ` | `hybridization="sp"` (`T`) | `atomic_smearing=0.0` / `1e-3` | **`True`** | 73 / 76 | `-109.387336` | `1.9775` | `5.25e-5` | `converged` |
   | $\text{N}_2$ | `DZP` | `hybridization="sp"` (`T`) | `atomic_smearing=0.0` / `1e-3` | **`True`** | 51 / 54 | `-109.527717` | `1.9688` | `1.12e-4` | `converged` |

   Notably, `hybridization="sp"` triggers `apply_hybrid_seeds` (`phatpsy/schema.py:1879-1886`), setting `QNWVEC=T` (and using the fractional dative-$\pi$ seed `[0.5, 1.0]/[1.0, 0.5]` on CO, tracked separately at `ROADMAP.md:210`). Five concrete code-level mechanisms separate non-converging `QNWVEC=F` from converging `QNWVEC=T` on C/N/O ($Z \le 18$):
   - Unoccupied orbital eigenvalues included in `pkpot(:,5)` (`w_pk5 = eval(1:NORB)` vs `occ * eval`; `src/scf.f90:728-733, 812-817, 927-932`, `src/atomic.f90:371-373`);
   - Pseudometric regularization cap $\delta_{\text{reg}} = 10^{-3}$ (nearly uncapped `cap = 0.999`) vs `0.42` (`cap = 0.58`, smooth $C^1$ ramp; `src/scf.f90:467-470`, `src/modelpot.f90:332-523`);
   - Active BLOOP `CDAMP` occupation blending every cycle (`src/scf.f90:442-449`, `src/bloop.f90:337-376`) vs frozen occupations under `QNWVEC=T`;
   - Joint `run_lowdin` back-projection orthonormalization (`src/scf.f90:658-666`) vs occupied/unoccupied partitioned `orthonormalize_projected_orbitals` (`src/scf.f90:227-317, 633-643`);
   - Active-subspace restriction on two-center attraction (`src/bloop.f90:324`) for `DZP`.

   Also note: `pkepot` and `epkpot` in `src/modelpot.f90:174-303` are **not called** anywhere in the active SCF loop; the active energy-weighted PK feedback flows entirely through `pkpot(:,5)` → `src/bloop.f90:299-315` → `pkpot(:,2)` → `stfock - pkpot(:,2)` (`src/scf.f90:564`) with `VDAMP` mixing (`src/aloop.f90:130-135`).

2. **The Hydrogen `stevec = 0` hole (Open Item 3) is wider than single-STO (`MSTO=1, NORB=1`):**
   The old `if (MSTO /= 1 .or. NORB /= 1)` guard cited in `ROADMAP.md:89` no longer exists. Instead, `src/scf.f90:667-669` (introduced in `e393edd` to eliminate an uninitialized `projec` read on H, then narrowed from unconditional `else` to `else if (NELEM > 1)` in `5342b53` to fix unseeded $\text{H}_2$) explicitly executes `stevec(:, 1:NORB, isp) = 0.0d0` for **every unseeded (`QNWVEC=F`) Hydrogen atom in every heteronuclear molecule (`NELEM > 1`), regardless of basis size**:
   - Confirmed at `891f64e`: `max|stevec| = 0.0000` and `convex = False` (mixed-sign fallback in `solve_model_potential_fit`) on unseeded `HF` (`SZ`, `DZ`, `DZP` with `MSTO=4, NORB=4`) and unseeded `H2O` (`DZ`, `DZP` with `MSTO=4, NORB=4`, `ADF-DZ` with `MSTO=2, NORB=1`).
   - Because `NORMLZ` (`src/linalg.f90:457`) skips zero-norm vectors, `stevec` remains identically zero, causing `eval_atomic_rho` (`src/exchange_dft.f90:238-339`) to return zero, `fit_screening_simplex` (`src/analys.f90:162`) to fail (`ngrid <= 1e-8`), `run_analys` to fall back to mixed-sign `solve_model_potential_fit`, and H's one-center kinetic/potential matrix elements in `run_analys` to vanish.
   - Homonuclear $\text{H}_2$ (`NELEM = 1`, `SZ`/`DZ`/`DZP`) is already fixed (`max|stevec| = 1.0`, convex weights, CPHF parity $< 10^{-6}\text{ Ha/Bohr}$, pinned in `tests/test_basis_polarized.py:380-417`), as is seeded (`QNWVEC=T`) heteronuclear H.
   - Several passing tests (`tests/test_grid_fit.py:133`, `tests/test_basis_polarized.py::test_h2o_dzp_convergence_and_dipole`, and unseeded `HF`/`LiH`/`H2O` in `tests/test_cphf.py`) currently execute on this `stevec = 0` branch.

3. **Model-potential exponent plumbing (Open Item 4):**
   Verified as described: default exponents (`TFEXP` in `src/state.f90:300` plus the `0.150` extra term at `phatpsy/schema.py:2019`, scaled by `1.12956 * Z**(1/3)` in `src/atomic.f90:33, 106, 175`) are unchanged; per-element `ElementConfig.potential_exponents` and `potential_exponent_scale` are implemented (`phatpsy/schema.py:1199-1271, 1768-1783, 2004-2019`) and tested (`tests/test_potential_exponents.py:1-132`). No committed script currently dumps $S(r)$ or reproduces the scratch-study RMS / $|\Delta V(R)|$ figures in `ROADMAP.md:94-100`.

4. **Quarantined tests and stale metadata:**
   - `tests/conftest.py:20-45` holds **24** entries in `BASELINE_QUARANTINE` (`ROADMAP.md:54` still reads 26).
   - `test_milestone8_diatomic_energy_stability_benchmark` (`tests/test_correlation_and_dispersion.py:225-239`, quarantined at `tests/conftest.py:26`) calls `evaluate_dissociation_energies_benchmark` (`phatpsy/benchmarks.py:377-422`), which runs **antiparallel** (`spins=["up", "down"]`) PBE-D3 on unseeded `DZ` $\text{N}_2$ and seeded (`hybridization="sp"`) `DZ` $\text{CO}$ — not a "high-spin" run as `ROADMAP.md:92` describes it.
   - `tests/test_validation_baselines.py:169-210` (`test_co_frozen_baseline`, standalone `@pytest.mark.xfail`) runs unseeded antiparallel `DZ` $\text{CO}$ and accepts four reference energies (with `-99.308811` listed twice via `ref_e` and literal).

---

## 2. Staged Execution Plan

Each stage is self-contained, carries a fixed acceptance gate defined in advance, and states an explicit stop condition so execution halts rather than improvising ad-hoc tuning or weakening convergence definitions.

### Stage 0 — Reproducible Diagnostic Harness & Code/Doc Audit Reconciliation (Zero Physics Change)

- **Goal:** Establish version-controlled, bit-reproducible diagnostic tooling for SCF trajectories and radial screening residuals, and eliminate internal documentation contradictions discovered in §1 (without changing any SCF physics or default behavior).
- **Preconditions:** Sequencing preconditions (Phase 18 Stages 3–4 + test/baseline triage branch on `master`).
- **Expected files touched:**
  - `tools/diagnose_scf_diatomics.py` (new): runs controlled legacy-SCF sweeps on $\text{N}_2$ and $\text{CO}$ (`DZ`, `DZP`; unseeded and `hybridization="sp"`; `auto`, explicit `atomic_smearing`, and pure `MAXOVL`), recording per-cycle $E_k$, $|\Delta E_k|$, $-V/T$, `stop_reason`, per-atom charges/spins, orbital eigenvalues/occupations, $\lambda_{\max}(S_A^{-1/2} P_{1,A} S_A^{-1/2})$, $\|P_{2,A}\|_F$, and last-window spectral oscillation diagnostics to JSON.
  - `tools/diagnose_screening_fit.py` (new): computes the exact atomic radial screening function $S_{\text{exact}}(r) = 1 - r V_{\text{el}}(r)/N$ on the 400-point logarithmic grid of `fit_screening_simplex` (`src/analys.f90:126-181`, mirroring the verified Poisson integrator in `phatpsy/cphf.py:128-169`), compares it against $S_{\text{fit}}(r) = \sum_k w_k e^{-\zeta_k r}$ from the engine's `vcoef`/`vexp`, and reports weighted RMS residual and implied $|\Delta V(R)| = N |S_{\text{fit}}(R) - S_{\text{exact}}(R)| / R$.
  - `tests/test_scf_diagnostics.py` (new): non-circular unit tests verifying Poisson charge normalization ($Q(R_{\max}) = N$ to $10^{-6}$), exact reproduction of Fortran `fit_screening_simplex` weights on a multi-electron atom (`< 10^{-10}`), and JSON schema determinism.
  - Documentation factual fixes only: update stale CO-vs-HF citation in [Model Potentials §5.3](../science/model_potentials.md) and `ROADMAP.md:86`, stale `BASELINE_QUARANTINE` count (`24`) and `test_milestone8` description in `ROADMAP.md:54, 92`, and record the §1.1 `atomic_smearing=0.0` audit finding accurately without prejudging Decision 3.
- **Acceptance gate:**
  1. Both diagnostic scripts execute deterministically: two independent invocations produce identical JSON outputs (`max|Δ| == 0.0` on identical platform).
  2. `tools/diagnose_screening_fit.py` reproduces Fortran `vcoef` weights from `fit_screening_simplex` to `< 10^{-10}` on C, N, O, and F.
  3. `./make_build_test.sh` exits `0` with `0 failed, 0 errors` (zero physics or test-behavior changes).
- **Stop condition:** If fixed-smearing or `QNWVEC=T` diagnostic runs are **not** bit-reproducible across consecutive invocations on the same platform, stop and isolate the nondeterminism source (uninitialized Fortran buffer or BLAS tie-breaking) before starting Stage 1.

---

### Stage 1 — Causal Diagnosis of $\text{N}_2$ Residual Oscillation (Test Attribution Before Any Fix)

- **Goal:** Empirically test the attribution that unseeded $\text{N}_2$ residual oscillation is driven by feedback through the energy-weighted Phillips–Kleinman projector (`pkpot(:,5)` → `pkpot(:,2)`), isolating all five mechanisms that distinguish non-converging `QNWVEC=F` from converging `QNWVEC=T`.
- **Preconditions:** Stage 0 merged.
- **Expected files touched:**
  - `tools/diagnose_scf_diatomics.py` (extended with the five-ablation matrix; any temporary Fortran ablation toggles used for controlled experiments remain strictly opt-in / diagnostic-only and never alter defaults);
  - `data/diagnostics/n2_oscillation_ablation.json` (new diagnostic dataset, outside `data/validation_baselines/`);
  - [Model Potentials Reference](../science/model_potentials.md) (new diagnostic record of the causal attribution).
- **Diagnostic ablation protocol (unseeded antiparallel $\text{N}_2$, `DZ` and `DZP`, $R = 2.0744\text{ Bohr}$, `use_grid_fit=True`, `ScfExtensionPolicy(enabled=False)`, `max_cycles=1000`, `convergence=1e-6` and `1e-8`):**
  1. **Ablation A (Energy-weighted PK projector `pkpot(:,5)` / `pkpot(:,2)`):**
     - *A1:* Change `w_pk5` from `eval(1:NORB)` to `occ * eval` (removing unoccupied virtual eigenvalues from `pkpot(:,5)`).
     - *A2:* Freeze `pkpot(:,2)` after cycle $k_0 = 20$ while allowing `pkpot(:,1)`, `cpot2`, `xpot2`, `vcoef`, and `occ` to evolve normally.
     - *A3:* Damp `pkpot(:,2)` independently (`VDAMP_PK2` $\in \{0.5, 0.8, 0.9, 0.95\}$) vs uniform `VDAMP` and `AdaptiveDamping`.
  2. **Ablation B (Pseudometric regularization cap $\delta_{\text{reg}}$):** Sweep $\delta_{\text{reg}} \in \{10^{-3}, 0.10, 0.25, 0.42\}$ and log $\lambda_{\max}(S_A^{-1/2} P_{1,A} S_A^{-1/2})$ per cycle.
  3. **Ablation C (BLOOP `CDAMP` occupation feedback):** Compare `CDAMP = 0.5`, `0.8`, and `1.0` (freezing atomic occupations against BLOOP charge transfer as `QNWVEC=T` does).
  4. **Ablation D (Back-projection orthonormalization):** Compare unpartitioned `run_lowdin` (`src/scf.f90:658-666`) against occupied/unoccupied partitioned `orthonormalize_projected_orbitals` (`src/scf.f90:227-317`).
  5. **Ablation E (Late-cycle mode spectrum):** Compute the dominant oscillation period, empirical contraction factor $|\mu|$, and relative phase/amplitude of $\|\Delta P_2\|_F$, $\|\Delta P_1\|_F$, $\|\Delta \text{occ}\|_2$, and $\|\Delta \text{vcoef}\|_2$ over cycles 100–300.
- **Acceptance gate:**
  1. A committed, reproducible ablation table (`data/diagnostics/n2_oscillation_ablation.json`) quantifying which specific mechanism(s) drive the unstable/limit-cycle eigenmode ($|\mu| \ge 1$) on both `DZ` and `DZP`.
  2. Either (a) at least one physically principled mechanism ablation achieves honest convergence (`converged=True`, `stop_reason="converged"` at `convergence=1e-6` and `1e-8`, `|-V/T - 2.0| <= 0.05`, $\mu < 10^{-10}\text{ D}$), or (b) all five ablations are exhaustively accounted for with exact residual amplitudes.
- **Stop condition:** If freezing `pkpot(:,2)` (A2) fails to eliminate oscillation **and** none of Ablations A–D yields an honestly converged, virial-compliant (`|-V/T - 2.0| <= 0.05`) unseeded $\text{N}_2$ state without sacrificing physical invariants, **stop** solver engineering on the unseeded $\text{N}_2$ path, document the root cause, and escalate to Jack to record unseeded $\text{N}_2$ as an inherent limitation of the legacy unseeded path (leaving seeded `hybridization="sp"` as the supported legacy SCF mode).

---

### Stage 2 — Targeted $\text{N}_2$ Fix & Systematic Antiparallel $\text{CO}$ Attractor Survey

- **Goal:** (1) Implement the minimal, docs-first, CPHF-consistent fix established by Stage 1 for $\text{N}_2$; (2) systematically characterize all reachable converged and non-converged attractors of antiparallel $\text{CO}$ (`DZ` and `DZP`, unseeded and seeded) to supply the evidence required for **Decision 1** and **Decision 3**.
- **Preconditions:** Stage 1 accepted with a positive causal fix candidate.
- **Expected files touched:**
  - [Model Potentials Reference](../science/model_potentials.md) (docs-first rationale + thesis citation before code change);
  - Minimal targeted Fortran edit in `src/scf.f90` (and `src/atomic.f90`, `src/bloop.f90`, or `src/modelpot.f90` **only** where proved necessary by Stage 1);
  - `phatpsy/cphf.py` (exact analytical derivative mirror of any changed SCF operator — zero finite-difference fitting);
  - `tools/diagnose_scf_diatomics.py` + `data/diagnostics/co_attractor_survey.json` (new): maps antiparallel $\text{CO}$ (`DZ`, `DZP`; unseeded and `hybridization="sp"` / `valence_config` seeded; `VDAMP` $\in \{0.5, 0.7, 0.85, 0.92\}$ and `AdaptiveDamping`; smearing modes `auto`, `1e-3`, `SMEARW=-1.0`), recording for every converged basin: $E_{\text{PHATPSY}}$, exact determinant energy $E_{\text{det}}$ and non-variational gap (`phatpsy/vb_correction.py::interatomic_decomposition`), $-V/T$, per-atom $(N_\alpha, N_\beta)$, dipole $\mu$ (magnitude and polarity), occupation integrality, and CPHF gradient parity;
  - `tests/test_scf_convergence.py` (new): non-circular tests asserting honest convergence, `|-V/T - 2.0| <= 0.05`, $\mu(\text{N}_2) < 10^{-10}\text{ D}$, bit reproducibility, and CPHF vs finite-difference gradient parity (`< 1e-4 Ha/Bohr`, `< 1e-5 Ha/Bohr` on `DZP`) — never hardcoding calibrated energies.
- **Acceptance gate:**
  1. Unseeded and seeded antiparallel $\text{N}_2$ (`DZ` and `DZP`) achieve honest convergence (`converged=True`, `stop_reason="converged"`) at `convergence=1e-6` and `1e-8`, `|-V/T - 2.0| <= 0.05`, $\mu < 10^{-10}\text{ D}$, bit-reproducible across repeated runs, and `max|g_rel - g_fd| < 1e-4 Ha/Bohr` (`< 1e-5 Ha/Bohr` on `DZP`).
  2. Complete, reproducible CO attractor evidence table committed in `data/diagnostics/co_attractor_survey.json`, ready for Jack's Decision 1 and Decision 3 rulings.
  3. `./make_build_test.sh` exits `0` with `0 failed, 0 errors` and zero regressions across existing `tests/test_cphf.py`, `tests/test_basis_polarized.py`, `tests/test_grid_fit.py`, and localized VB tests.
- **Stop condition:**
  - Any CPHF gradient parity regression (`> 1e-4 Ha/Bohr`), virial violation (`|-V/T - 2.0| > 0.05`), or convergence regression on previously green legacy systems ($\text{H}_2$, $\text{HF}$, $\text{H}_2\text{O}$, $\text{EO}$) halts the change immediately.
  - If antiparallel unseeded $\text{CO}$ yields **zero** honestly converged physical basins across the full sweep (or only negative-occupation / virial-violating basins), stop unseeded-CO solver engineering and present that empirical result directly in Decision 1 (favoring seeded `hybridization="sp"` / `valence_config` for heteronuclear multiple bonds).

---

### Stage 3 — Scoped Resolution (or Explicit Deferral) of the Unseeded Heteronuclear Hydrogen `stevec = 0` Defect

- **Goal:** Close Audit Item 3 (`src/scf.f90:667-669`), where `else if (NELEM > 1) then stevec(:, 1:NORB, isp) = 0.0d0` wipes out the Hydrogen orbital vector on **all** unseeded (`QNWVEC=F`) heteronuclear hydrides (`SZ`, `DZ`, `DZP`, `ADF-DZ`), preventing `fit_screening_simplex` from running and zeroing H's one-center electronic energy.
- **Preconditions:** Stage 0 merged (can run independently of Stages 1–2, or sequentially after Stage 2 to keep PRs small and bisectable).
- **Scoped Fix Plan (Primary Path):**
  1. Remove the `else if (NELEM > 1) then stevec(:, 1:NORB, isp) = 0.0d0` wipeout at `src/scf.f90:667-669` so unseeded heteronuclear H retains its normalized eigenvector from `DMPAB(tranx, onevec)` (remaining exempt from `projec` back-projection just as `src/scf.f90:467` already exempts it from `TRISQ`).
  2. Verify `phatpsy/cphf.py` grid-fit response on H (`atom_uses_grid_fit` at `phatpsy/cphf.py:534-539`, `_compute_grid_fit_dvcoef` at `lines 77-198`) matches central finite differences on unseeded `HF`, `LiH`, and `H2O` (`SZ`, `DZ`, `DZP`).
  3. Generate an explicit numerical-deviation report comparing old vs new energies, virials, dipoles, and CPHF parity on unseeded `HF`, `LiH`, and `H2O` (required by `AGENTS.md` §3 Rule 2 before touching `tests/test_grid_fit.py:133` or `tests/test_basis_polarized.py::test_h2o_dzp_convergence_and_dipole`).
  4. Remove the one-electron convexity exemption in `tests/test_grid_fit.py:84-87` and add explicit `max|stevec| > 0`, $T_{\text{H}} > 0.4\text{ Ha}$, and convex-weight assertions for H in `HF` and `H2O`.
- **Expected files touched:**
  - *If fixed:* `src/scf.f90:653-670`, `phatpsy/cphf.py` (only if H grid-fit derivative requires adjustment), `tests/test_grid_fit.py`, `tests/test_basis_polarized.py` (only with Jack's approval on the deviation report), [Model Potentials Reference](../science/model_potentials.md), `ROADMAP.md`.
  - *If deferred:* `src/scf.f90:667` (comment clarifying exact scope), `tests/test_grid_fit.py:84-87` (comment clarifying scope), [Model Potentials Reference](../science/model_potentials.md), `ROADMAP.md`.
- **Acceptance gate (Primary Path):**
  1. Across unseeded `HF` (`SZ`, `DZ`, `DZP`), `LiH` (`SZ`, `DZ`), and `H2O` (`SZ`, `DZ`, `DZP`, `ADF-DZ`): `max|stevec| > 0`, $T_{\text{H}} > 0.4\text{ Ha}$, H screening weights satisfy $w_k \in [0, 1]$ and $\sum_k w_k = 1$ to $10^{-10}$ (zero fallback to `solve_model_potential_fit`), honest convergence (`converged=True`, `stop_reason="converged"`), and `|-V/T - 2.0| <= 0.05`.
  2. CPHF relaxed gradient parity `max|g_rel - g_fd| < 1e-4 Ha/Bohr` across all systems in `tests/test_cphf.py`.
  3. Jack signs off on the numerical-deviation report for any shifted unseeded hydride reference value; `./make_build_test.sh` exits `0`.
- **Stop condition (Explicit Deferral Trigger):** If removing `src/scf.f90:667-669` causes unseeded `HF`, `LiH`, or `H2O` to lose SCF convergence, violate `|-V/T - 2.0| <= 0.05`, suffer unphysical bond/dipole collapse requiring new empirical H–X workarounds, or break `tests/test_cphf.py` parity without a clean first-principles derivative fix, **defer fixing unseeded heteronuclear H**: leave `src/scf.f90:667-669` untouched, document the true scope (`NELEM > 1`, `.not. QNWVEC`, all bases — with `QNWVEC=T`, homonuclear $\text{H}_2$, and `localized_vb` unaffected) in `ROADMAP.md` and [Model Potentials Reference](../science/model_potentials.md), and close Stage 3 without perturbing legacy CPHF parity.

---

### Stage 4 — Reproducible Model-Potential Exponent Study (Measurement Only — No Default Change)

- **Goal:** Reproduce, extend, and place under version control the model-potential exponent study summarized in `ROADMAP.md:94-100`, explicitly testing (a) Hydrogen, (b) exact two-center Coulomb accuracy across both bonded and non-bonded distances ($R = 4\text{--}8\text{ Bohr}$), and (c) correlation between $S(r)$ residual and $|E_{\text{PHATPSY}} - E_{\text{det}}|$, providing complete evidence for **Decision 2**.
- **Preconditions:** Stage 0 merged (Stage 2 preferred so $\text{N}_2$/$\text{CO}$ convergence behavior is evaluated on the stabilized SCF).
- **Expected files touched:**
  - `tools/run_exponent_study.py` (new): drives `tools/diagnose_screening_fit.py` + `phatpsy/two_center_integrals.py` + `phatpsy/vb_correction.py` across elements **H, C, N, O, F** and molecules **$\text{H}_2$, $\text{HF}$, $\text{H}_2\text{O}$, $\text{N}_2$, $\text{CO}$, $\text{NO}$** (`DZ` and `DZP`, unseeded and seeded), comparing:
    1. Built-in Thomas–Fermi-derived default (`TFEXP` + `0.150`, $1.12956\,Z^{1/3}$ scaling);
    2. Candidate Z-scaled even-tempered sets ($K = 5$, $\zeta_k = 0.45\,\text{rscale}\cdot \beta^k$ for $\beta \in \{1.8, 2.0, 2.2\}$, plus H-specific exponents);
    3. Element-specific 2- and 3-exponential free-exponent fits to each isolated atom's own Hartree–Fock density (strictly internal target — zero experimental calibration).
  - `data/diagnostics/exponent_study.json` (new committed study output, outside `data/validation_baselines/` and `phatpsy/data/`).
  - `tests/test_potential_exponents.py`: add non-circular invariant tests for the screening residual calculator (default exponents unchanged).
  - [Model Potentials §8](../science/model_potentials.md) and `ROADMAP.md`: replace unverified scratch numbers with citations to `tools/run_exponent_study.py` and `data/diagnostics/exponent_study.json`.
- **Acceptance gate:**
  1. `PYTHONPATH=. .venv/bin/python tools/run_exponent_study.py` reproduces `data/diagnostics/exponent_study.json` to `< 10^{-10}` across consecutive runs.
  2. Study reports for every candidate set: weighted $S(r)$ RMS, $|\Delta V(R)|$, exact two-center Coulomb error at $R_e$ and at $R \in \{4, 6, 8\}\text{ Bohr}$, $|E_{\text{PHATPSY}} - E_{\text{det}}|$, honest convergence status, cycle count, and $-V/T$.
  3. Default exponents (`src/state.f90:300`, `phatpsy/schema.py:2019`) remain untouched; `./make_build_test.sh` exits `0`.
- **Stop condition:** If all lower-RMS exponent candidates degrade SCF convergence on $\text{CO}$ or $\text{N}_2$ **and** provide $< 10^{-3}\text{ Ha}$ improvement in non-bonded ($R \ge 4\text{ Bohr}$) exact two-center Coulomb error, stop further exponent-set engineering and present the negative result in Decision 2 (recommending Option 2A: keep current defaults).

---

### Stage 5 — Apply Jack's Decisions, Conform to Baseline Triage & Close Out

- **Goal:** Implement Jack's rulings on Decisions 1, 2, and 3, reconcile all documentation and tests, and conform strictly to the test/baseline triage branch outcome.
- **Preconditions:** Stages 1–4 complete (or stopped at documented stop conditions) **and** Jack has decided Decisions 1, 2, and 3.
- **Expected files touched (conditional on Jack's decisions):**
  - Legacy SCF code (`src/scf.f90`, `src/state.f90`, `src/analys.f90`, `phatpsy/schema.py`, `phatpsy/cphf.py`) strictly as authorized by Decisions 1–3;
  - Tests (`tests/test_grid_fit.py`, `tests/test_potential_exponents.py`, `tests/test_scf_convergence.py`, and `tests/conftest.py` conforming to the landed triage disposition of `test_milestone8_diatomic_energy_stability_benchmark` and `test_co_frozen_baseline`);
  - `tools/generate_validation_baselines.py` and `data/validation_baselines/*.json` **only** if re-freezing is authorized under `AGENTS.md` §6 (recording `converged=True`, `stop_reason="converged"`, explicit spins, and settings; never freezing unconverged runs);
  - User and science docs (`docs/science/model_potentials.md`, `docs/user/yaml/overview.md`, `docs/user/python/setup-and-run.md`, `ROADMAP.md`).
- **Acceptance gate:**
  1. `./make_build_test.sh` exits `0` with `0 failed, 0 errors`, including `tests/test_docs_examples.py` and `tools/lint_docs.py`.
  2. Every targeted legacy SCF system passes honest convergence (`converged=True`, `stop_reason="converged"`), `|-V/T - 2.0| <= 0.05`, cross-run bit reproducibility, and CPHF gradient parity `< 1e-4 Ha/Bohr`.
  3. `git diff --stat` confirms zero edits to `phatpsy/localized_vb.py`, `phatpsy/vb_driver.py`, `phatpsy/two_center_integrals.py`, `phatpsy/multicenter_integrals.py`, and `phatpsy/embedding.py`.
- **Stop condition:** Any regression in the localized VB suite, any CPHF parity failure (`> 1e-4 Ha/Bohr`), or any baseline failing honest convergence blocks the PR until resolved or explicitly recorded as a blocked `xfail` in `ROADMAP.md`.

---

## 3. Decisions for Jack (Posed — Not Decided by This Plan)

Three design and methodology decisions belong to Jack. Each is posed below with its options, the evidence required from Stages 1–4, and an explicitly labeled recommendation.

### Decision 1 — Selection Rule Between Competing Converged States (Antiparallel $\text{CO}$ and Heteronuclear Diatomics)

- **Why this is open:** The earlier converged CO states ($\approx -112.50, -112.61\text{ Ha}$) came from an erroneous high-spin start and are invalidated. Moreover, Stage 3 determinant diagnostics (`ROADMAP.md:213`, [Model Potentials §10](../science/model_potentials.md)) proved that the legacy single-center functional $E_{\text{PHATPSY}}$ is **non-variational** ($E_{\text{PHATPSY}} - E_{\text{det}} \approx -38\text{ eV}$ on $\text{CO}$ `DZ`), so "lowest $E_{\text{PHATPSY}}$" can reward model-potential over-binding rather than physical accuracy. Meanwhile, seeded (`hybridization="sp"`) antiparallel CO already converges honestly on `DZ` and `DZP`, whereas unseeded antiparallel CO does not.
- **Options:**
  - **Option 1A — Lowest $E_{\text{PHATPSY}}$ subject to physical admissibility gates** (honest convergence, non-negative occupations, `|-V/T - 2.0| <= 0.05`, $\langle S_z \rangle = 0$).
  - **Option 1B — Lowest exact bare-Hamiltonian determinant energy $E_{\text{det}} = \langle \Psi_{\text{PHATPSY}} | \hat{H} | \Psi_{\text{PHATPSY}} \rangle$** (`phatpsy/vb_correction.py::interatomic_decomposition`) among honestly converged, virial- and spin-admissible attractors.
  - **Option 1C — Declared valence seed defines the canonical state (`hybridization` / `valence_config`):** designate directed valence seeding (`QNWVEC=T`) as the canonical legacy-SCF path for heteronuclear multiple bonds rather than defining an unseeded multi-basin search rule (contingent on separate adjudication of the fractional dative-$\pi$ seed item at `ROADMAP.md:210`).
  - **Option 1D — Multi-criterion screen adding qualitative dipole polarity ($\text{C}^{\delta-}\text{O}^{\delta+}$):** apply Option 1B plus qualitative polarity verification (strictly excluding experimental dipole *magnitude* as a selection criterion under the Anti-Circularity Invariant).
- **Evidence required:**
  - *For 1A vs 1B vs 1D:* Stage 2 `data/diagnostics/co_attractor_survey.json` showing whether multiple distinct honestly converged physical basins exist on antiparallel CO, and whether ranking by $E_{\text{PHATPSY}}$ vs $E_{\text{det}}$ picks different basins (along with their virial ratios, spin populations, and dipole signs).
  - *For 1C:* Stage 2 confirmation of whether unseeded antiparallel CO converges at all after Stage 1's fix, and whether seeded `hybridization="sp"` CO converges uniquely across damping schedules and basis tiers (`DZ`, `DZP`).
- **Recommendation *(recommendation only — Jack decides)*:** **Adopt Option 1C as the default workflow for heteronuclear multiple bonds in `phatpsy_scf`, paired with Option 1B ($E_{\text{det}}$ ranking) whenever competing honestly converged physical basins are compared on the same deck.** *Reasoning:* Because $E_{\text{PHATPSY}}$ is non-variational, minimizing it can pick an over-bound artifact; $E_{\text{det}}$ is strictly variational by Rayleigh–Ritz, internal to the exact Hamiltonian (non-circular), and already implemented in the repository.

---

### Decision 2 — Default Model-Potential Screening Exponent Set

- **Why this is open:** The unreproduced scratch study (`ROADMAP.md:94-100`) found that the default Thomas–Fermi-derived exponent table has large $S(r)$ residuals (RMS `0.0246`, $|\Delta V(R)|$ up to `0.28 Ha`) because its largest exponent ($3.52\,\text{rscale}$) misses the sharp $1s$ core of C/N/O/F, whereas Z-scaled even-tempered sets cut RMS to `0.0044` (`0.04 Ha`). However, replacing defaults shifted total energies by $-0.63$ to $+0.08\text{ Ha}$, **failed to improve (and worsened) CO convergence**, did not test Hydrogen properly, and did not verify whether lower $S(r)$ RMS reduces exact-Coulomb error or $|E_{\text{PHATPSY}} - E_{\text{det}}|$.
- **Options:**
  - **Option 2A — Keep the current default exponent table unchanged;** retain `elements.<sym>.potential_exponents` / `potential_exponent_scale` as the supported per-element override.
  - **Option 2B — Replace defaults with a universal Z-scaled even-tempered set** (e.g., $K = 5$, $\zeta_k = 0.45\,\text{rscale}\cdot 2^k$, with a documented H rule).
  - **Option 2C — Replace defaults with element-specific 2- or 3-exponential tables** fitted strictly to each isolated atom's own Hartree–Fock density via a committed generator script (zero experimental calibration).
- **Evidence required:**
  - Stage 4 committed dataset (`data/diagnostics/exponent_study.json`) across H, C, N, O, F and $\text{H}_2, \text{HF}, \text{H}_2\text{O}, \text{N}_2, \text{CO}, \text{NO}$ comparing Options 2A, 2B, and 2C on: (1) $S(r)$ weighted RMS and $|\Delta V(R)|$; (2) exact two-center Coulomb error at both $R_e$ and non-bonded spectator distances ($R = 4\text{--}8\text{ Bohr}$); (3) $|E_{\text{PHATPSY}} - E_{\text{det}}|$; (4) SCF convergence rates and CPHF gradient parity.
- **Recommendation *(recommendation only — Jack decides)*:** **Hold defaults unchanged through Stage 4, and favor Option 2A unless Stage 4 demonstrates a clear improvement in non-bonded ($R \ge 4\text{ Bohr}$) exact-Coulomb error *without* regressing SCF convergence on $\text{CO}$ and $\text{N}_2$.** *Reasoning:* Because covalent bonding now resides in exact-integral localized VB, the practical payoff of changing legacy defaults lies in non-bonded spectator electrostatics; a global default change that shifts every legacy energy by $\sim 0.1\text{--}0.6\text{ Ha}$ and degrades convergence would hurt more than it helps.

---

### Decision 3 — Default Atomic Occupation Mode (`atomic_smearing` vs. Automatic Annealing vs. Pure `MAXOVL`)

- **Why this is open:** As flagged in §1.1 item 4, `master` (`891f64e`) currently has a silent semantic contradiction: `atomic_smearing = 0.0` runs **automatic early-cycle annealing + HOMO–LUMO gap floor + cycle-5 integer snap** on unseeded UHF atoms (`src/scf.f90:131-215`), pure `MAXOVL` requires `SMEARW < 0` which `ControlConfig.__post_init__` (`phatpsy/schema.py:228-232`) rejects, seeded (`QNWVEC=T`) decks still default to pure `MAXOVL`, and most docs still describe `0.0` as "off". Meanwhile, fixed finite smearing (`1e-3 Ha`) shifts seeded CO energy by $0.135\text{--}0.168\text{ Ha}$ (`DZ`/`DZP`), does not converge unseeded CO/N₂, and introduces an unmodeled $\partial n_i / \partial \mathbf{R}$ occupation-response term in CPHF whenever final occupations remain fractional.
- **Options:**
  - **Option 3A — Ratify automatic annealing + gap floor + integer snap as the default for unseeded UHF (preserving pure `MAXOVL` on seeded/RHF), and expose a clean explicit opt-out to pure `MAXOVL`:** keep the current `src/scf.f90` stabilization on unseeded UHF, make pure `MAXOVL` reachable from Python/YAML without `extra_namelist` workarounds (e.g., permitting an explicit opt-out sentinel or documented `None` = auto / `0.0` = strictly off distinction, without altering unrelated deck vocabulary), and align all docstrings and manual pages.
  - **Option 3B — Restore pure overlap-tracking (`MAXOVL`/`NEWORD`) as the true default when `atomic_smearing == 0.0`:** remove automatic activation at `0.0` and require explicit opt-in for annealing/smearing, restoring strict historical `0.0 = off` semantics everywhere.
  - **Option 3C — Make fixed finite smearing (e.g., `atomic_smearing = 1e-3`) the global default** on unseeded UHF decks (always reporting the width in run metadata).
- **Evidence required:**
  - Stage 1–2 convergence, energy-bias, cross-BLAS reproducibility (macOS Accelerate vs Linux OpenBLAS), and CPHF gradient parity comparisons across all three modes (`auto`, pure `MAXOVL`, fixed `1e-3`) on unseeded and seeded `H2O`, `HF`, `N2`, `CO`, `NO`, `EO`, and `Ni13`;
  - Verification of whether `tests/test_grid_fit.py::test_automatic_annealing_and_gap_floor_on_unseeded_and_seeded_atoms` (unseeded `HF` reproducibility and `H2O DZP` ground state) remains cross-platform deterministic under Option 3B.
- **Recommendation *(recommendation only — Jack decides)*:** **Adopt Option 3A (ratify automatic annealing with cycle-5 integer snap on unseeded UHF, expose an explicit supported opt-out to pure `MAXOVL`, and reconcile all stale docs/comments), and reject Option 3C.** *Reasoning:* Option 3C introduces large ($> 0.13\text{ Ha}$) occupation-width bias on CO, breaks exact CPHF parity when occupations stay fractional, and fails to converge unseeded CO/N₂; Option 3B re-exposes unseeded `HF` and `H2O DZP` to $10^{-14}$ eigensolver-noise nondeterminism across BLAS backends. Option 3A preserves cross-platform reproducibility and snaps back to exact integer occupations once the gap opens (keeping CPHF exact), while fixing the current silent overload and `ControlConfig` opt-out block.

---

## 4. Summary of Stage Gates & Decision Dependencies

| Stage | Primary Subject | Depends On | Fixed Gate | Stop Trigger |
| :--- | :--- | :--- | :--- | :--- |
| **0** | Diagnostic harness & doc reconciliation | Preconditions (Phase 18 Stages 3–4 + triage) | Bit-identical JSON runs; Python vs Fortran $S(r)$ weights `< 1e-10`; `./make_build_test.sh` green | Nondeterminism on fixed-smearing / `QNWVEC=T` runs |
| **1** | Causal diagnosis of $\text{N}_2$ oscillation | Stage 0 | Quantitative 5-ablation table isolating $|\mu| \ge 1$ mode in `data/diagnostics/n2_oscillation_ablation.json` | No single/pair ablation yields honest, virial-compliant unseeded $\text{N}_2$ |
| **2** | Targeted $\text{N}_2$ fix + antiparallel $\text{CO}$ survey | Stage 1 | $\text{N}_2$ honest convergence (`1e-6`/`1e-8`), `|-V/T-2| <= 0.05`, CPHF `< 1e-4` (`< 1e-5` DZP); complete CO attractor table | CPHF or virial regression on existing green systems; or zero physical unseeded CO basins |
| **3** | Unseeded heteronuclear H `stevec=0` defect | Stage 0 | Unseeded `HF`/`LiH`/`H2O`: `max|stevec| > 0`, convex H weights (`1e-10`), `|-V/T-2| <= 0.05`, CPHF `< 1e-4`, approved deviation report | Breaks convergence/virial/CPHF on `HF`/`LiH`/`H2O` → trigger explicit deferral |
| **4** | Reproducible model-potential exponent study | Stage 0 (Stage 2 preferred) | Reproducible `data/diagnostics/exponent_study.json` (`< 1e-10`); covers H + bonded & $R=4\text{--}8\text{ Bohr}$ exact Coulomb + $E_{\text{det}}$ | Lower-RMS sets degrade $\text{CO}$/$\text{N}_2$ convergence and fail to improve far-field Coulomb |
| **5** | Implement Decisions 1–3 & triage closeout | Stages 1–4 + Jack's rulings | `./make_build_test.sh` green; zero VB/embedding edits; any baseline re-freeze via `AGENTS.md` §6 only | Any VB regression, CPHF parity failure, or unconverged baseline |
