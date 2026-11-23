# Spin-Catalysis Flagships — Acceptance Specification
## PHATPSY 2.0 Model Systems 8 and 9

**Status:** Approved — adopted by Jack, November 22, 2026, with the
model-system criterion amended the same day (either an experimental or an
independent computational comparison suffices; both preferred; the report
states which when only one is present). This document defines what
"done" means for the two spin-catalysis flagships. It is an acceptance
document only — it does not implement code, and meeting it is judged against
the checks stated here, in order, with the guardrails below in force
throughout.

**Staging:** Systems 8 and 9 come after the seven baseline systems in the
Baseline Model Systems Specification (N2, CO, NO, EO, AA, Ni bulk, Ni slab).
They are the next rung, not part of the baseline capture.

---

## Global guardrails (apply to both flagships)

1. **Atomic units.** All energies, geometries, and gradients are computed and
   stored in atomic units (Hartree, Bohr). eV or kcal/mol conversions may be
   reported alongside, labeled as conversions.
2. **Method of record.** The localized VB approach (migration decision,
   November 13, 2026): exact integrals for bonded regions; the legacy model
   potential / PK functional is used only for non-bonded interactions.
   pk_scale = 1.0 identically in the VB/embedding layer.
3. **Representation.** YAML is the reference representation. Every system
   ships as a YAML deck with an explicit `valence_config`, plus an equivalent
   Python API construction and a CLI run that reproduce the same single-point
   energies. Modern YAML inputs throughout; no legacy .stdin decks.
4. **Spin bookkeeping.** Spins are explicit per atom and per orbital
   (valence_config). The deck `net_spin` is an atom-label balance, not the
   physical 2S; where the two differ, both are reported. Frustrated or
   ferromagnetic configurations are legitimate inputs — they are treated,
   never refused or silently reassigned.
5. **Anti-Circularity Invariant.** No acceptance value in this spec may
   appear as a constant, calibration target, or fitted parameter in the
   implementation. Every acceptance check names its independent oracle —
   experiment, PySCF, finite differences, or a published value — in advance.
6. **No silent approximations.** Cluster truncation, basis tier, active
   space, and every frozen / fixed-state / clamped choice are stated in the
   deck and in the report. A result whose approximations are not stated is
   not accepted, however good the number looks.
7. **The model-system criterion (Jack, November 8, 2026; amended by Jack,
   November 22, 2026).** A flagship counts as a model system when it has an
   experimental comparison OR an independent computational comparison; both
   are preferred. When only one side is present, the report states which
   side it is.
8. **Failures are recorded as failures.** A transition-state search that
   does not produce a verified first-order saddle, a run that lands in the
   wrong spin sector, and a check that misses its anchor are all written into
   the record as what they are — never relabeled, averaged away, or dropped.

## CO encoding guardrail (applies wherever CO appears)

CO is a **dative pi bond with no formal charges**: the O lone pair donates
into the empty C pi orbital. Spins: C `["up","up"]`, O `["down","down"]`,
`net_spin = 0`, no spin flip. The valence_config encoding, per the baseline
spec:

- C: sigma bond (occ 1, up, →O), pi bond (occ 1, up, →O), pi acceptor
  (occ 0, →O), sigma lone pair (occ 2). Total 4 electrons.
- O: sigma bond (occ 1, down, →C), pi bond (occ 1, down, →C), pi donor lone
  pair (occ 2, →C), sigma lone pair (occ 2). Total 6 electrons.

Guardrail values (sanity checks only, never targets): E ≈ −112.69 Ha,
−V/T ≈ 1.9702. Any CO deck that introduces formal charges, a spin flip, or a
different orbital assignment fails this guardrail regardless of its energy.

## Parked systems — explicitly out of scope

The RuN4 nitrate-reduction system and the La-promoted Co ammonia-synthesis
system (Lu et al., Acc. Mater. Res. 2026) are **parked**. They are not
flagships, they are not Systems 8 or 9, and they must not be promoted,
substituted, or started under this spec. They are revisited only when both
flagships below are in working order.

---

## System 8 — NO dissociation on Ni(100)

**Definition.** Nitric oxide chemisorbed on a ferromagnetic Ni(100) cluster
model, followed through N–O bond cleavage to adsorbed N* + O*. Builds on
baseline System 3 (NO radical: unpaired electron on N, N `["up","up","up"]`,
O `["down","down"]`, deck net_spin 1) and System 7 (Ni slab spin pattern).
The working model is the Ni4-active / Ni9-spectator plaquette in Ni13, with
the 4-fold hollow as the primary site — the dissertation's site (1978,
Ch. IX: NO chemisorption on Ni(100) was the PHATPSY method test case;
Ch. VIII's spin-orbital assignments are the ancestor of valence_config).
Both adsorbate orientations, N-down and O-down, are computed, revisiting
dissertation Table 7.

**Experimental anchors.** Brundle's XPS core binding energies for adsorbed
NO (dissertation Table 7 experimental column: N 1s 400/403 eV, O 1s
531/532 eV); UHV XPS N 1s ≈ 399.8 eV for hollow-site N-down NO.

**Computational comparisons.** PySCF CASSCF/CASCI oracles for the VB
energies; the 1978 dissertation values as historical comparison (its
Koopmans core levels run ~15 eV high — the orientation splitting is the
signal, not the absolute values); published N–O cleavage barriers
(J. Phys. Chem. C 2023) and the Ni85 nanocluster result (Mahata et al.,
Sci. Rep. 2016) as literature context.

**Where it stands (November 21, 2026).** Chemisorption stages signed off:
the 4-fold hollow N-down is the global molecular binding site, and the
CVS-ΔCASSCF N 1s for hollow N-down lands at the XPS anchor. Dissociation
endpoints are final — the dissociated N*+O* minimum sits 1.83 eV below the
tilted molecular minimum. The open item is the dissociation transition
state: the first NEB band slid laterally without stretching the N–O bond
and is recorded as a failed saddle search.

**Acceptance checks.**

1. **Site ordering.** The 4-fold hollow N-down is the lowest molecular
   minimum among the sites computed (top, bridge, hollow × N-down, O-down),
   with all energies reported in Hartree and site differences in eV.
2. **Orientation sign.** N-down lies below O-down on every model size
   computed (Ni2 bridge screen, Ni5 dissertation geometry, Ni4/Ni9
   plaquette). One reversal on any model fails this check.
3. **Core levels.** CVS-ΔCASSCF N 1s for hollow N-down within 1 eV of the
   399.8 eV XPS anchor; O-down shifts both N 1s and O 1s upward relative to
   N-down (the Table 7 sign). Koopmans values, where reported, are labeled
   orbital-energy shifts, never binding energies.
4. **Verified saddle.** The dissociation transition state has exactly one
   imaginary frequency, whose mode is predominantly N–O stretch; the NEB is
   converged at the 1.2e-4 Ha/Bohr gradient target; the saddle N–O distance
   is substantially elongated against the 2.80 Bohr molecular value. A band
   maximum without these properties is a failed search, recorded as such.
5. **Connected path.** An IRC from the verified saddle reaches the tilted
   molecular NO* minimum in one direction and the dissociated N*+O*
   minimum in the other. The barrier is reported against the tilted
   molecular minimum.
6. **Energetics.** The dissociated products lie below molecular NO*
   (dissociation exothermic on the plaquette), with the signed difference
   reported in Ha and eV.
7. **Spin fate.** The spin state of adsorbed NO* and of the N* and O*
   products is reported per atom, and the fate of NO's unpaired electron
   through adsorption and cleavage is stated explicitly, including the
   spin-quenching profile across the spin sectors computed.
8. **Comparison test.** At least one experimental anchor (check 3) OR at
   least one independent computational comparison (PySCF oracle or a
   published barrier) is reported for the dissociation step; both are
   preferred. If only one side is present, the report states which.

---

## System 9 — Ag-catalyzed ethylene epoxidation: EO, and EO → AA

**Definition.** Ethylene + O2 over a silver cluster model → ethylene oxide
(EO), with the competing isomerization EO → acetaldehyde (AA). Builds on
baseline Systems 4 (EO, mixed-spin ring oxygen) and 5 (AA). The spin story
starts at triplet O2 on a diamagnetic Ag surface and runs through the
spin-forbidden steps. PHATPSY is a cluster code: the Ag surface is a
cluster model, and the truncation is stated, never implied to be a
periodic slab. The mechanism runs through the surface oxametallacycle
(OMC), which branches: ring closure → EO, or H-transfer → AA.

**The EO desorption constraint (hard requirement).** AA is the gateway to
CO2 in both routes: AA formed on the surface burns rapidly, and EO that
isomerizes to AA instead of desorbing feeds the same combustion channel.
EO must therefore desorb before it isomerizes or is deeply oxidized, and
the selectivity question is a race between EO desorption, EO → AA
isomerization, and combustion. Any pathway account that lets adsorbed EO
isomerize without comparing the EO–Ag desorption energy against the
isomerization barrier from the same model fails acceptance on its face.

**Experimental anchors.** EO selectivity ≈ 50% on unpromoted Ag and
≈ 90% on promoted Ag; the AA branch barrier ≈ 0.4 eV below the epoxidation
barrier on oxygen-reconstructed Ag (J. Phys. Chem. C); the ACS Catal. 2024
microkinetic model (doi:10.1021/acscatal.4c04521), in which AA is the CO2
precursor and the OMC pathway alone favors combustion.

**Computational comparisons.** The gas-phase EO → AA localized VB pathway
(already computed and signed off, November 15, 2026: EO → TS1 → biradical
minimum → TS2 → AA; ΔE‡ forward 55.88 kcal/mol, ΔG‡(298) 49.50,
ΔE_rxn −26.89 kcal/mol) against the CCSD(T)/CBS reference barrier of
53.0 kcal/mol; published DFT OMC branching barriers; microkinetic
selectivity trends.

**Where it stands (November 21, 2026).** The gas-phase EO → AA spine is
done at CAS(4e,4o) with the biradical intermediate characterized (natural
occupations ≈ [1.973, 1.000, 0.9996, 0.027]); the level sensitivity at DZ
is disclosed in that record. The Ag surface model — O2 activation, the
OMC, and the branching — has not been started.

**Acceptance checks.**

1. **Gas-phase spine.** The EO → TS1 → biradical → TS2 → AA pathway is
   reproduced at a stated level (basis tier and active space in the deck),
   each transition state a verified first-order saddle with its imaginary
   mode assigned, and the biradical's natural occupations reported. The
   forward barrier is compared to the CCSD(T)/CBS 53.0 kcal/mol reference;
   a deviation is explained by level, never tuned away.
2. **OMC branching.** Ring-closure (→ EO) and H-transfer (→ AA) barriers
   are computed from the same OMC geometry with explicit per-orbital
   spins. The sign of the AA-vs-EO barrier difference, and its size
   against the published ≈ 0.4 eV, are reported.
3. **The desorption race, quantified.** The EO–Ag desorption energy and
   the EO → AA isomerization barrier from the same Ag model are reported
   side by side, and the selectivity conclusion drawn follows from those
   two numbers and no others.
4. **Spin bookkeeping.** The triplet-O2 activation path is stated; the
   spin state at every stationary point is reported; any spin-forbidden
   step is identified as spin-forbidden, not smoothed over.
5. **Coverage probe.** A one-oxygen vs. two-nearby-oxygen cluster
   comparison (the multi-oxygen collision channel to CO2) is reported —
   or explicitly deferred, with the reason recorded.
6. **Comparison test.** At least one experimental anchor (selectivity or
   the ≈ 0.4 eV branching difference) OR at least one independent
   computational comparison (CCSD(T)/CBS or published DFT/MKM) is
   reported; both are preferred. If only one side is present, the report
   states which.

---

## Reporting standard (both flagships)

Every reported number carries its method, basis tier, active space, spin
assignment, and the oracle it was checked against. Notebooks use the YAML
decks as their inputs and demonstrate the Python API and CLI equivalents.
Deviations from an anchor are stated with their size and the level they
are attributed to. Nothing in either flagship's record is tuned to an
acceptance value in this document.
