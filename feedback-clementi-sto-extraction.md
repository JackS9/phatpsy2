# Feedback: verification of `phatpsy/data/clementi_sto.json` (Clementi & Roetti Tables 42/43)

Prepared by Lux (Jack's assistant) after an independent verification pass, Oct 21, 2026.
Companion script: `check_clementi_orthonormality.py` (stdlib-only; runs in <1 s).

## The check to add

Total-energy spot checks only cover a handful of atoms. A cheaper test covers **all** of them:
orthonormality of the stored orbitals, **CᵀSC = I** per angular-momentum block, computed from
the stored (n, ζ) pairs and coefficients alone — no SCF, no PHATPSY build.

For normalized STOs of the same *l*, R(r) = N r^(n−1) e^(−ζr) with N = (2ζ)^(n+½)/√((2n)!), the
overlap is S_ij = N_i·N_j·(n_i+n_j)! / (ζ_i+ζ_j)^(n_i+n_j+1). The tables print 5 decimals, so a
clean parse lands at worst-residual ≲ 1×10⁻³, typically ~10⁻⁵. Anything larger is a parse defect,
not print rounding. Recommend wiring the attached script into the test suite and gating
`ingest_clementi.py` output on it.

## Headline result

**Only 8 of 106 element/set combinations pass** at 1×10⁻³:
SZ — B (6×10⁻⁶), C (9×10⁻⁶), F (4×10⁻⁴), Ca (4×10⁻⁵); DZ — C (3×10⁻⁴), O (1×10⁻⁵), F (4×10⁻⁴), Cl (7×10⁻⁴).
(He SZ is numerically exact — ζ = 1.6875, c = 1.0 — and is only flagged for a placeholder eigval, below.)

## Failure classes

1. **Row/column misalignment beyond the light atoms.** Residuals run 10⁻¹ → 10¹⁴ for most of
   Z > 10. Smoking guns in the stored values themselves: coefficients of 35.0 and 45.0 (Cu SZ 1S/2S),
   1012240.0 (Cu SZ 3S), 108301.0 (Al SZ 2P), 10101504.0 (P DZ 1S), and an exponent ζ = 5119095.0
   (P DZ). These look like decimal points and/or Fortran D-exponents from the OCR text layer being
   dropped or merged during parsing, plus rows slipping against their atom/orbital labels.
2. **A recurring phantom exponent: ζ = 25.0 exactly**, stored as the first s function in 29 entries
   (Li, Be, Cu, P, …). Systematic artifact — worth hunting in the ingest's column handling.
3. **Li and Be are garbled even in the light block** (both sets): Li's 2S never references the n = 2
   function; Be SZ 2S carries c = 0.2 on the wrong function (a single-zeta orbital is one STO with
   c = 1). Norm deviations up to 0.96.
4. **Energies:** mostly variationally sane (just above the full-basis RHF values), but with exceptions
   that suggest the energy column slips too: He DZ energy is −3.286167, which is *below* the exact He
   energy (−2.9037) and therefore cannot be a total energy; Al SZ shows −324.115376 vs the known
   Al HF energy ≈ −241.9.
5. **Orbital energies unpopulated:** 239 orbitals carry eigval = 0.0 (essentially all p and d orbitals).

## Reference point

An independent extraction of the full RHF expansions (the paper's main tables) for He–Ne, verified
by the same orthonormality test plus the tables' printed energies, is available from Jack
(`clementi-roetti_he-ne_orbital-major.json`). Its He–Ne energies — He −2.861680, Li −7.432726,
Be −14.573021, B −24.529057, C −37.688612, N −54.400924, O −74.809370, F −99.409300, Ne −128.547050 —
are variational floors the SZ/DZ energies must sit above; useful as regression anchors while the
ingest is fixed.

## Suggested order of work

1. Fix number parsing (D-exponent/decimal handling) and row alignment in `ingest_clementi.py`.
2. Re-run the attached checker; iterate until all 106 pass at 1×10⁻³.
3. Cross-check the energy column against variational ordering (SZ ≥ DZ ≥ full-basis RHF ≥ exact).
4. Populate or explicitly null the orbital energies rather than 0.0 placeholders.
5. Spot-verify a sample of repaired entries against the printed pages (the scan's 4↔6 confusion
   is documented — e.g. F's ζ printed by OCR as 6.90066 is really 4.90066 in the main tables).
