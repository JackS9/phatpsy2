# PHATPSY 2.0 Validation Standards & Benchmarks

This document specifies the numerical validation standards, physical invariants, reference datasets, and test execution procedures for **PHATPSY 2.0**.

---

## 1. Physical Invariants & Quality Gates

Every calculation and modernized kernel must satisfy the following physical quality gates:

1. **Virial Ratio Test ($-V/T$):**
   - For an isolated neutral molecule in its ground state at equilibrium geometry, the quantum virial theorem requires:
     $$\frac{-\langle V \rangle}{\langle T \rangle} \approx 2.000$$
   - In modern PHATPSY 2.0, standard benchmark systems ($N_2$, $CO$, bulk $Ni_{13}$) must converge with $|-V/T - 2.0| \le 0.05$.
2. **Monotonic Total Energy Convergence:**
   - Under standard clean defaults (`CDAMP = 0.5`, `VDAMP = 0.5`), the cycle-by-cycle total electronic energy must converge monotonically after initial atomic preparation.
3. **Spin Conservation in Localized Valence-Bond (LVB) States:**
   - For closed-shell molecules and embedded slabs treated with alternating spin (`spin="up"` / `spin="down"`), the net molecular spin must satisfy physical boundary conditions:
     $$\langle S_z \rangle = \frac{1}{2} (N_\alpha - N_\beta) = 0.000 \quad \text{(for singlets)}$$

---

## 2. Validation Methodology: Physics Invariants & External Benchmarks

Historical legacy baselines (`data/validation_baselines/`) and legacy namelist input decks (`*.stdin`) have been formally retired. Rather than chasing exact numerical parity with 1980s legacy scratch-disk outputs, modern PHATPSY 2.0 validates against **fundamental physical invariants** and **external scientific benchmarks**.

In adherence to first-principles theory, **`PKSCAL = 1.0` is adopted as the standard default** across the codebase. Eliminating empirical scaling factors (like the legacy $0.80$ and $0.40$ settings) via the energy-dependent formulation $V_{PK}(E)$ and canonical Lowdin regularization provides numerical stability without hand-tuned knobs.

### Core Benchmark Systems & Physical Gates:

| System | Domain / Capability Validated | Physical Quality Gates |
| :--- | :--- | :--- |
| **$N_2$ (DZ)** | Homonuclear anchor & dual-geometry | Converged ground state, $-V/T \approx 2.00$, zero permanent dipole ($\mu = 0$) |
| **$H_2O$** | Non-linear polar triatomic & Eckart modes | Converged ground state, $-V/T \approx 2.00$, non-zero dipole ($\mu > 1\text{ D}$), origin invariance |
| **Bulk $Ni_{13}$** | Bulk metal coordination ($Z=12$) | Converged ground state under theoretical $\text{PKSCAL}=1.0$, $-V/T \approx 2.00$ |
| **$Ni(100)$ Slab** | 3-layer embedded surface slab | Produces reusable surface and subsurface (second layer) atoms with third layer fixed as bulk atoms |

---

## 3. Subsystem Embedding & ONIOM Parity

For surface and solid-state calculations, PHATPSY supports embedding active subsystems within spectator or frozen substrate environments:

1. **Subsystem Partitioning:**
   $$E_{\text{tot}} = E_{\text{active}} + E_{\text{env}} + E_{\text{int}}$$
2. **Embedded Excess Energy:**
   $$E_{\text{embed}} = E_{\text{tot}}(\text{active} + \text{env}) - E_{\text{env}}(\text{isolated})$$
   This decouples active surface and chemisorption processes from the massive, extensive core self-energy of deep bulk layers, maintaining compatibility with ONIOM and DFT-in-DFT schemes.
3. **Site-Resolved Atomic Energy Partitioning:**
   $$\varepsilon_A = \text{toten}_A - \frac{1}{2} \sum_{B \ne A} \frac{Z_A Z_B}{R_{AB}}$$
   Enables direct tracking of surface undercoordination shifts, relaxation decay, and work function perturbations.

---

## 4. Layer B: Experimental Literature Benchmarks

Documented in [`data/validation_baselines/experimental_benchmarks.json`](../../data/validation_baselines/experimental_benchmarks.json), compiled from NIST CCCBDB, CRC Handbook, and Herzberg:

- **$N_2$:** $R_e = 1.0977\text{ \AA}$, $\omega_e = 2358.57\text{ cm}^{-1}$, $D_0 = 9.759\text{ eV}$ ($225.06\text{ kcal/mol}$), $I_p = 15.58\text{ eV}$.
- **$CO$:** $R_e = 1.1283\text{ \AA}$, $\omega_e = 2169.81\text{ cm}^{-1}$, $\mu = 0.112\text{ D}$ ($C^-O^+$), $D_0 = 11.087\text{ eV}$, $I_p = 14.01\text{ eV}$.
- **$H_2O$:** $R_{OH} = 0.9578\text{ \AA}$, $\theta = 104.48^\circ$, $\mu = 1.8546\text{ D}$, $\nu = [3657, 1595, 3756]\text{ cm}^{-1}$.
- **$EO \to AA$:** $\Delta H_{\text{rxn}} = -27.2\text{ kcal/mol}$, $E_a^{\text{fwd}} = 57.2\text{ kcal/mol}$.
- **Bulk $Ni$:** FCC $a = 3.524\text{ \AA}$, $d_{\text{NN}} = 2.492\text{ \AA}$, $E_{\text{coh}} = 4.44\text{ eV/atom}$ ($102.4\text{ kcal/mol}$), $\mu = 0.616\ \mu_B$.
- **$CO / Ni(100)$:** Top site $E_{\text{ads}} = -30.0\text{ kcal/mol}$ ($-1.30\text{ eV}$), $\Delta\nu_{CO} = -78\text{ cm}^{-1}$.

---

---

## 5. Benchmark Validation Engine & Error Vector Evaluation

Phase 5 establishes an automated literature benchmark evaluation engine in [`phatpsy/benchmarks.py`](../../phatpsy/benchmarks.py) and [`tests/test_benchmarks.py`](../../tests/test_benchmarks.py):
- Compares calculated values under `ControlConfig.first_principles(pk_scale=1.0)` against external Layer B literature (NIST CCCBDB, CRC Handbook, Somorjai & Li).
- Reports quantitative error vectors on bond lengths ($\Delta R_e$), harmonic vibrational frequencies ($\Delta \omega_e$), permanent dipole moments ($\Delta \mu$), virial theorem compliance ($-V/T \to 2.0$), and chemisorption energy ($E_{\text{ads}}$).
- Formally retires empirical damping parameters (`CDAMP`, `VDAMP`, `CACCEL`) in favor of self-consistent Pulay DIIS, constrained LAPACK cusp fitting, and dynamic STO radial exponent optimization.

---

## 6. Numerical Tolerances & Update Rule

| Metric | Tolerance | Rationale |
| :--- | :--- | :--- |
| **Total Energy ($E_{\text{tot}}$)** | $\text{atol} = 2.5 \times 10^{-3}\text{ Hartree}$ | Modern LAPACK matrix diagonalizations and cross-BLAS portability (macOS Accelerate vs Linux OpenBLAS). |
| **Virial Ratio ($-V/T$)** | $\text{atol} = 0.05$ | Physical virial boundary condition. |
| **Harmonic Frequency ($\omega$)** | $\text{atol} = 10.0\text{ cm}^{-1}$ | Finite-difference numerical Hessian accuracy. |
| **Dipole Moment ($\mu$)** | $\text{atol} = 0.05\text{ Debye}$ | EOM density and origin invariance. |
| **Polarizability ($\bar{\alpha}$)** | $\text{atol} = 0.10\text{ a}_0^3$ | Sum-over-states dipole response. |
| **Reaction Barrier ($E_a$)** | $\text{atol} = 1.0\text{ kcal/mol}$ | CI-NEB climbing image energy convergence. |

### Baseline Update Rule:
All reference files in `data/validation_baselines/` are **frozen**. In accordance with **AGENTS.md §3 Rule 2**, any modification to these values requires an explicit numerical-deviation report documenting the scientific rationale and citing the relevant dissertation section. Baselines must never be silently adjusted.

### Perturbation Detector ("Tests That Bite"):
`tests/test_validation_baselines.py::test_baseline_perturbation_detector` deliberately perturbs baseline values by $> 5\times$ tolerance and asserts that the suite fails loudly with an actionable diff, proving that tests cannot pass silently on corrupted physics.

---

## 7. Test Execution Instructions

Run the full automated test suite using pytest:
```bash
# Run the complete test suite
./make_build_test.sh

# Run specifically the frozen validation baseline tests
PYTHONPATH=. .venv/bin/pytest tests/test_validation_baselines.py -v

# Run the Layer B benchmark validation suite
PYTHONPATH=. .venv/bin/pytest tests/test_benchmarks.py -v

# Generate benchmark summary table directly
PYTHONPATH=. .venv/bin/python -m phatpsy.benchmarks
```
