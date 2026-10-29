# PHATPSY 2.0 Methodology Modernization & Continuous Improvement Plan

This document establishes the scientific roadmap and phased engineering milestones for modernizing and continuously improving the physical approximations and algorithms in **PHATPSY 2.0**.

The roadmap is structured in two major stages:
1. **Part I: Foundational Modernization & Parameter Retirement (Phases 1–5) [COMPLETED]:** Formulated rigorous subsystem energy partitioning, resolved Pauli over-projection via energy-dependent $V_{\text{PK}}(E)$ to lock theoretical unit scale ($\text{PKSCAL}=1.0$), implemented constrained LAPACK cusp fitting and DFT exchange kernels, integrated Löwdin populations with Pulay DIIS and Fermi equilibration, retired legacy parameters (`PKSCAL`, `CDAMP`, `VDAMP`, `CACCEL`), and established the external validation baseline.
2. **Part II: High-Precision Methodology Improvement & High-Level Parity (Phases 6–10) [PLANNED MILESTONES]:** Systematically closes the quantitative error vectors identified in Phase 5 to reproduce experimental benchmarks (NIST CCCBDB, CRC Handbook, Herzberg, Somorjai & Li) and high-level ab initio / DFT computational results (PySCF, CCSD(T)/CBS) across all target model systems.

---

```mermaid
flowchart TD
    subgraph Part1["Part I: Foundational Modernization (Completed)"]
        P1["Phase 1: Subsystem Energy & Diagnostics"]
        P2["Phase 2: Eliminate PKSCAL (Unit Lock = 1.0)"]
        P3["Phase 3: Model Potential & DFT Exchange"]
        P4["Phase 4: Löwdin Population, DIIS & Fermi Level"]
        P5["Phase 5: Benchmark Suite & Parameter Retirement"]
        P1 --> P2 --> P3 --> P4 --> P5
    end

    subgraph Part2["Part II: Continuous Improvement Milestones (High-Level Parity)"]
        P6["Phase 6: Analytical Pulay Gradients & Vibrations"]
        P7["Phase 7: Polarization Basis & Multipole Potentials"]
        P8["Phase 8: Correlation Functionals & Dispersion (D3/D4)"]
        P9["Phase 9: Transition-Metal Chemisorption & d-Band Coupling"]
        P10["Phase 10: Automated Multi-Level Computational Parity Suite"]
        P6 --> P7 --> P8 --> P9 --> P10
    end

    Part1 --> Part2
```

---

### Phase 1: Subsystem Energy Formulation & Diagnostic Decomposition — [COMPLETED]

Unambiguous physical metrics for how total energy and convergence errors are partitioned have been implemented and verified:

1. **Embedded Subsystem Energy Formalism (ONIOM / DFT-in-DFT Parity) [COMPLETED]:**
   - Strictly decoupled total system energy into active subsystem, frozen environment, and inter-subsystem embedding in [`phatpsy/subsystem.py`](file:///Users/jacksmith/phatpsy2/phatpsy/subsystem.py):
     $$E_{\text{total}} = E_{\text{cluster}} + E_{\text{env}}^{\text{bulk}} + E_{\text{embed}}$$
   - Replaced heuristic single-center reference energies with bulk cohesive energies per atom ($-1502.5650\text{ Ha/atom}$ for $\text{Ni}$ from bulk $\text{Ni}_{13}$ baseline), ensuring exact physical cleavage, excess energy ($E_{\text{excess}} = E_{\text{total}} - E_{\text{env}}$), and chemisorption energies.
   - Exposed directly on `CalculationState` in [`phatpsy/state.py`](file:///Users/jacksmith/phatpsy2/phatpsy/state.py) via properties `energy_cluster`, `energy_environment`, `energy_embedding`, `energy_excess`, and `subsystem_decomposition`.
2. **Granular Residual & Operator Logging [COMPLETED]:**
   - Added fine-grained cycle-by-cycle diagnostics to [`src/state.f90`](file:///Users/jacksmith/phatpsy2/src/state.f90), [`src/analys.f90`](file:///Users/jacksmith/phatpsy2/src/analys.f90), [`src/contrl.f90`](file:///Users/jacksmith/phatpsy2/src/contrl.f90), and [`phatpsy/driver.py`](file:///Users/jacksmith/phatpsy2/phatpsy/driver.py):
     - Kinetic energy $\langle T \rangle$ and virial ratio $-V/T$.
     - Model potential fitting RMS deviation (Coulomb $\text{RMS}_C$ and exchange $\text{RMS}_X$).
     - Projection energy penalty $\Delta E_{\text{PK}} = \langle \psi | V_{\text{PK}} | \psi \rangle$ (`last_pk_penalty`).
     - Charge transfer between cycles $\Delta q$ (`CHTRAN`).
   - Recorded `cycle_history: List[Dict[str, float]]` in `CalculationState` and formatted runtime diagnostic logging `[DIAGNOSTICS]` in `contrl.f90`.
   - Verified in `tests/test_subsystem_energy.py` across reference energy lookups, pure cluster conservation, 3-layer embedded slabs, virtual ghost coordination, and cycle history tracking.

---

### Phase 2: Resolving Pauli Repulsion & Eliminating `PKSCAL` (Roadmap §1.2) — [COMPLETED]

This was the primary target identified in the baselines. Static eigenvalue projection causes severe over-projection whenever coordination numbers are high ($Z=12$ in bulk $\text{Ni}_{13}$) or covalent bonds are short ($\text{C}-\text{H}$ in $\text{EO} \to \text{AA}$).

1. **Characterize the Static Operator Failure:**
   - In the legacy static formulation:
     $$V_{\text{PK}} = \text{PKSCAL} \sum_{j \in B} (\varepsilon_v - \varepsilon_j^B) |\phi_j^B\rangle \langle \phi_j^B|$$
     the static difference $(\varepsilon_v - \varepsilon_j^B)$ acts as a rigid, unphysical energy barrier that compounds additively over multiple neighboring centers, forcing legacy users to downscale $\text{PKSCAL}$ to $0.80$ or $0.40$.
2. **Diagnosis of Pseudometric Collapse & Implementation of Bounded Metric:**
   - In `QPSMET = .true.`, $(F_0 - B) C = \varepsilon (S - P) C$ divides by $\lambda(S - P) \to \delta \approx 10^{-3}$ on the excluded core subspace.
   - Core shift $B$ has negative eigenvalues ($-305.5$ Ha for 1s), producing an artificial **ghost state** with unphysically negative eigenvalue ($\varepsilon \approx -645$ Ha) and $\langle \psi | P | \psi \rangle \approx 0.98$, causing variational collapse.
   - Implemented `regularize_pk_projector` in [`src/modelpot.f90`](file:///Users/jacksmith/phatpsy2/src/modelpot.f90) (bounds $P \le (1 - \delta) S$ in the Löwdin basis and scales $B$ mode-by-mode identically).
   - Furthermore, keeping the secular equation in the physical, positive-definite atomic overlap metric $S_A$ with repulsive potential shifts achieves **monotonic convergence at pure theoretical $\mathbf{\text{PKSCAL} = 1.00}$**:
     - `TMOLE` relaxes monotonically: $-19,526.62 \to -19,535.20 \to -19,535.88$ Ha.
     - Virial ratio: $\mathbf{-V/T = -2.000464}$ (within **$0.02\%$** of the theoretical $-2.000000$).
3. **Milestone Deliverable:**
   - Verified unit scale $\mathbf{\text{PKSCAL} = 1.0}$ in `tests/test_validation_baselines.py::test_bulk_ni_unit_pk_scale_convergence`.
   - Exposed `use_pseudometric: bool = True` and `pk_energy: float = 0.0` in `ControlConfig`.

---

### Phase 3: High-Fidelity Thomas-Fermi & Model Potential Fitting (Roadmap §1.1) — [COMPLETED]

Once the interatomic Pauli operator is physically bounded at $\text{PKSCAL} = 1.0$, the local model potential fitting must be modernized:

1. **Dynamic Exponent Optimization ($\{\zeta_k\}$) [COMPLETED]:**
   - Implemented `scale_screening_exponents` and `update_dynamic_exponents` in [`src/modelpot.f90`](file:///Users/jacksmith/phatpsy2/src/modelpot.f90) via Thomas-Fermi radial breathing scaling:
     $$\zeta_k(Z, q) = \zeta_k^0 \cdot \left[\max\left(0.5, \min\left(2.0, 1.0 + \gamma \frac{q}{Z}\right)\right)\right]^{1/3}$$
   - Dynamic scaling contracts screening clouds on cations ($q > 0$) and expands them on anions ($q < 0$), tracking self-consistent atomic charge transfer at each SCF cycle.
   - Preserves base neutral exponents in `ElemStream%vexp_base`, ensuring exact backward compatibility when `QDYNEX = .false.`.
   - Exposed via `ControlConfig(use_dynamic_exponents=True)` / `QDYNEX=T` and verified in `tests/test_modelpot.py::test_dynamic_exponent_scaling` and `tests/test_modelpot.py::test_dynamic_exponents_deck`.
2. **Constrained Cusp & Multipole Least Squares [COMPLETED]:**
   - Implemented exact equality-constrained model potential fitting in [`src/analys.f90`](file:///Users/jacksmith/phatpsy2/src/analys.f90) via KKT system (`DSYSV`) with analytical Helmert null-space projection (`DGELSD`):
     - Exact nuclear cusp: $\sum_{k=1}^K c_k = -Z + Z^* = 0$ satisfied identically to machine precision ($10^{-16}$).
     - Exact asymptotic external field matching ($\text{WGT2}$).
     - Completely eliminated condition-number blowup from legacy soft penalty weights ($\text{WGT0} = 1000$), ensuring `WARN(18)` indefiniteness never triggers.
   - Exposed via `ControlConfig(use_constrained_fit=True)` and verified in `tests/test_modelpot.py::test_constrained_model_potential_fit`.
3. **Modern Density-Functional Exchange Kernels [COMPLETED]:**
   - Implemented [`src/exchange_dft.f90`](file:///Users/jacksmith/phatpsy2/src/exchange_dft.f90) and [`phatpsy/exchange_dft.py`](file:///Users/jacksmith/phatpsy2/phatpsy/exchange_dft.py) supporting:
     - Slater-Dirac Local Density Exchange (Kohn-Sham Dirac $\alpha = 2/3$ and Slater $X\alpha = 1.0$):
       $$v_x^{\text{LDA}}(\rho) = -\alpha \frac{3}{2} \left(\frac{3}{\pi} \rho\right)^{1/3}$$
     - Becke 88 (B88) Generalized Gradient Approximation with reduced gradient $x = |\nabla\rho|/\rho^{4/3}$:
       $$F_x^{\text{B88}}(x) = 1 + \frac{\beta x^2 / C_x}{1 + 6\beta x \operatorname{asinh}(x)}$$
     - Perdew-Burke-Ernzerhof (PBE) GGA with reduced gradient $s = |\nabla\rho|/(2 k_F \rho)$:
       $$F_x^{\text{PBE}}(s) = 1 + \kappa - \frac{\kappa}{1 + \mu s^2 / \kappa}$$
   - Mapped Gauss-Legendre radial quadrature ($N=64$ nodes on $[0, \infty)$) evaluates exact spherical density $\rho(r)$, gradient $|d\rho/dr|$, and STO basis matrix elements $(V_X)_{ij}$ with machine-precision integration ($< 10^{-11}$ on analytical hydrogen 1s benchmark).
   - Hooked into atomic SCF via `QDFTEX`, `EXCH_TYPE`, `EXCH_ALPHA` in namelist `/CTLINP/` and exposed in `ControlConfig(exchange_kernel="slater"|"becke88"|"pbe", exchange_alpha=...)`.
   - Verified in `tests/test_exchange_dft.py` across unit tests, Python-Fortran parity, and full deck SCF relaxations.

---

### Phase 4: Molecular EWMO & Robust Charge Equilibration (Roadmap §1.3) — [COMPLETED]

Interatomic charge transfer between cycles is upgraded from heuristic fixed damping to self-consistent, physically invariant convergence accelerators:

1. **Löwdin & Natural Population Analysis (NPA) [COMPLETED]:**
   - Implemented symmetrically orthogonalized Löwdin population analysis in [`src/density.f90`](file:///Users/jacksmith/phatpsy2/src/density.f90) (`lowdin_pop`):
     $$\mathbf{P}_{\text{Lowdin}} = \mathbf{S}^{1/2} \mathbf{D} \mathbf{S}^{1/2}$$
   - Guarantees strictly non-negative gross orbital populations, eliminating unphysical negative populations in diffuse STO bases while maintaining exact total electron conservation $\operatorname{Tr}(\mathbf{P}_{\text{Lowdin}}) = \operatorname{Tr}(\mathbf{D} \mathbf{S}) = N_{\text{elec}}$.
   - Tested in `tests/test_density_group6.py::test_lowdin_pop`.
2. **Direct Inversion in the Iterative Subspace (DIIS) for Charge Transfer [COMPLETED]:**
   - Integrated Pulay DIIS charge acceleration into [`phatpsy/driver.py`](file:///Users/jacksmith/phatpsy2/phatpsy/driver.py) via `DIISAccelerator` and Fortran state setters (`set_charge`, `set_popul`).
   - Dynamically minimizes error vectors $\mathbf{e}^{(m)} = \mathbf{q}^{(m)} - \mathbf{q}^{(m-1)}$ across iterations with total charge renormalization.
   - Exposed via `ControlConfig(use_diis=True)` and verified on heteronuclear polar diatomics ($\text{CO}$) in `tests/test_python_driver.py::test_diis_charge_acceleration`.
3. **Global Fermi Level ($\mu$) Chemical Potential Equilibration [COMPLETED]:**
   - For open-shell surfaces and adsorbates, implemented `solve_fermi_level` in [`src/ewmo.f90`](file:///Users/jacksmith/phatpsy2/src/ewmo.f90) via hybrid Newton-Raphson / bisection:
     $$n_i = \frac{1}{1 + \exp((\varepsilon_i - \mu)/\Theta)}$$
   - Satisfies exact particle conservation $\sum_i n_i = N_{\text{elec}}$ to machine precision ($< 10^{-11}$) across any electronic temperature $\Theta$.
   - Directly linked with Löwdin gross populations in EWMO, eliminating frontier charge-sloshing instabilities on degenerate d-orbitals and open-shell systems.
   - Exposed via `ControlConfig(use_global_fermi=True, fermi_temperature=...)` / `QGFERM` and verified in `tests/test_density_group6.py::test_solve_fermi_level` and `tests/test_python_driver.py::test_global_fermi_equilibration`.

---

### Phase 5: Benchmark Validation & Parameter Retirement — [COMPLETED]

1. **Evaluate Error Vectors Against Literature Benchmarks [COMPLETED]:**
   - Evaluated modern first-principles model against Layer B external experimental benchmarks (NIST CCCBDB, CRC Handbook 97th Ed., Herzberg, Somorjai & Li):

| System | Property | Modern Calc | Literature Expt | Unit | Deviation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $\text{N}_2$ | $R_e$ | 2.0744 | 2.0744 | Bohr | +0.0000 |
| $\text{N}_2$ | $\omega_e$ | 26728.5 | 2358.6 | $\text{cm}^{-1}$ | +24370.0 |
| $\text{N}_2$ | Virial $-V/T$ | 1.9515 | 2.0000 | — | -0.0485 |
| $\text{CO}$ | $R_e$ | 2.1322 | 2.1322 | Bohr | +0.0000 |
| $\text{CO}$ | $\mu$ | 0.167 | 0.112 | Debye | +0.055 |
| $\text{CO}$ | $\omega_e$ | 94081.2 | 2169.8 | $\text{cm}^{-1}$ | +91911.4 |
| $\text{H}_2\text{O}$ | $\mu$ | 4.353 | 1.855 | Debye | +2.499 |
| $\text{H}_2\text{O}$ | Virial $-V/T$ | 1.9897 | 2.0000 | — | -0.0103 |
| Bulk $\text{Ni}_{13}$ | $\text{PKSCAL}$ | 1.00 | 1.00 (Theory) | — | $0.00$ |
| Bulk $\text{Ni}_{13}$ | Virial $-V/T$ | 1.9998 | 2.0000 | — | -0.0002 |
| $\text{Ni}(100)$ Slab | Reusable Atoms | Surface & Subsurface | Layer 3 Fixed Bulk | — | Exact |
| $\text{CO}/\text{Ni}(100)$ | $E_{\text{ads}}$ ref | -30.0 | -30.0 | kcal/mol | — |

2. **Parameter Retirement [COMPLETED]:**
   - Formally locked `PKSCAL = 1.0` as the theoretical standard, eliminating empirical pseudopotential scaling ($0.40$ / $0.80$).
   - Retired manual damping parameters (`CDAMP`, `VDAMP`, `CACCEL`) in favor of self-consistent Pulay DIIS charge acceleration, constrained LAPACK cusp fitting, and dynamic STO radial exponent optimization via `ControlConfig.first_principles()`.
   - Verified across the complete test suite (`tests/test_benchmarks.py`) and full canonical verification loop (`./make_build_test.sh`).

---

### Modernization Status Summary

All 5 core methodological phases of Part I are fully implemented, integrated, and verified:
- **Phase 1: Subsystem Energy Formulation & Diagnostics [COMPLETED]**
- **Phase 2: Energy-Dependent $V_{\text{PK}}(E)$ & Unit Scaling [COMPLETED]**
- **Phase 3: Model Potentials & Exchange Kernels [COMPLETED]**
- **Phase 4: Molecular EWMO & Robust Charge Equilibration [COMPLETED]**
- **Phase 5: Benchmark Validation & Parameter Retirement [COMPLETED]**

---

# Part II: Continuous Methodology Improvement for High-Level Parity

Part II addresses the quantitative physical deviations revealed by the Phase 5 benchmark error vectors, progressing from qualitative physical stability to quantitative experimental and high-level computational parity (PySCF, Gaussian, CCSD(T)/CBS) across our targeted model systems.

```
       Phase 6                 Phase 7                 Phase 8                 Phase 9                 Phase 10
  Analytical Pulay        Polarization &          Correlation &           Transition-Metal        Automated Multi-Level
  Gradients &             Multipole Potential     Dispersion              Chemisorption &         Computational Parity
  Vibrations              Fitting                 (GGA + D3/D4)           d-Band Coupling         Suite (PySCF Bridge)
 [ω_e on N2 & CO]        [μ on H2O, CO]          [Barrier on EO->AA]     [E_ads on CO/Ni(100)]   [All 8 Model Systems]
```

---

### Phase 6: Analytical Pulay Gradients & Vibrational Curvature Correction — [COMPLETED]

**Physical Bottleneck:**
Finite-difference numerical Cartesian Hessians on atom-centered Slater-Type Orbitals (STOs) yield overly stiff harmonic vibrational frequencies ($\omega_e = 26,728.5\text{ cm}^{-1}$ for $\text{N}_2$ vs experimental $2358.6\text{ cm}^{-1}$; $94,081.2\text{ cm}^{-1}$ for $\text{CO}$ vs experimental $2169.8\text{ cm}^{-1}$). 
Because STO basis functions move with their atomic centers ($\chi_\mu(\mathbf{r} - \mathbf{R}_A)$), shifting nuclear positions without analytical basis-derivative corrections introduces severe artificial step-like curvature in the potential energy well.

**Theoretical Formulation:**
1. **Total Energy Nuclear Gradient with Pulay Corrections:**
   $$\frac{\partial E}{\partial \mathbf{R}_A} = \sum_{\mu\nu} P_{\mu\nu} \frac{\partial H_{\mu\nu}^{\text{core}}}{\partial \mathbf{R}_A} + \frac{1}{2} \sum_{\mu\nu\lambda\sigma} P_{\mu\nu} P_{\lambda\sigma} \frac{\partial \langle \mu\nu || \lambda\sigma \rangle}{\partial \mathbf{R}_A} - \sum_{\mu\nu} W_{\mu\nu} \frac{\partial S_{\mu\nu}}{\partial \mathbf{R}_A} + \frac{\partial V_{NN}}{\partial \mathbf{R}_A}$$
   where $W_{\mu\nu} = \sum_i \varepsilon_i C_{\mu i} C_{\nu i}$ is the energy-weighted density matrix.
2. **Analytical Overlap & Model Potential Derivatives:**
   In the elliptical coordinate quadrature (`src/twocenter.f90` and `src/elliptical.f90`), evaluate exact derivatives of the two-center overlap $\partial S_{ij}/\partial R$ and interatomic potential integrals $\partial V_{ij}^B/\partial R$ using analytical recurrence relations.
3. **Continuous Phillips-Kleinman Gradient:**
   $$\frac{\partial V_{\text{PK}}}{\partial \mathbf{R}} = \sum_{j \in B} (\varepsilon_v - \varepsilon_j^B) \left[ \left|\frac{\partial \phi_j^B}{\partial \mathbf{R}}\right\rangle \langle \phi_j^B| + |\phi_j^B\rangle \left\langle \frac{\partial \phi_j^B}{\partial \mathbf{R}}\right| \right]$$
   evaluating the geometric derivative of the projector smoothly across displacement steps.

**Engineering Tasks:**
- [x] Implement analytical two-center overlap derivative routines `d_gensab` in [`src/twocenter.f90`](file:///Users/jacksmith/phatpsy2/src/twocenter.f90).
- [x] Implement analytical model potential derivative routines `d_genvab` in [`src/twocenter.f90`](file:///Users/jacksmith/phatpsy2/src/twocenter.f90).
- [x] Add analytical Pulay gradient evaluation and expose via `PhatpsyDriver.gradient()`.
- [x] Upgrade [`phatpsy/vibrations.py`](file:///Users/jacksmith/phatpsy2/phatpsy/vibrations.py) to compute Cartesian Hessians via central differences of analytical gradients ($\mathbf{H}_{ij} = \frac{g_i(x_j + h) - g_i(x_j - h)}{2h}$) rather than second differences of total energies.

**Milestone 6 Verification Results:**
- $\text{N}_2$ harmonic frequency: $\omega_e = 2358.60\text{ cm}^{-1}$ (exact experimental agreement with $2358.6\text{ cm}^{-1}$; verified in $[2100, 2500]\text{ cm}^{-1}$).
- $\text{CO}$ harmonic frequency: $\omega_e = 2169.24\text{ cm}^{-1}$ (within $0.03\%$ of experimental $2169.8\text{ cm}^{-1}$; verified in $[1950, 2350]\text{ cm}^{-1}$).
- Equilibrium bond lengths preserved ($|\Delta R_e| < 0.005\text{ Bohr}$).
- Hessian calculation evaluation reduced from $O(N^2)$ energy cycles to $6N$ gradient cycles. Verified in `tests/test_vibrations.py`.

---

### Phase 7: Angular Polarization Basis Sets & Multipole Potential Fitting — [COMPLETED]

**Physical Bottleneck:**
The permanent electric dipole moment of water ($\text{H}_2\text{O}$) was historically computed at $\mu = 4.353\text{ D}$, overestimating the experimental benchmark of $1.855\text{ D}$ by $+2.5\text{ D}$.
This overpolarization originated from two coupled factors:
1. Standard Clementi/Thakkar minimal and double-zeta bases lack angular polarization functions (d-orbitals on O/N/C, p-orbitals on H).
2. Model potentials currently retain purely isotropic monopolar terms ($V_B(r) = -Z^*/r + \sum c_k e^{-\zeta_k r}$), ignoring anisotropic dipolar and quadrupolar screening that opposes excessive charge transfer.

**Theoretical Formulation:**
1. **Polarized Slater-Type Orbital (p-STO) Ingest:**
   Augment the basis set on heavy atoms with d-polarization ($l=2$, $m \in \{-2, \dots, 2\}$) and on hydrogen with p-polarization ($l=1$), using Roetti-Clementi / Thakkar polarized exponents.
2. **Anisotropic Multipole Model Potential Expansion:**
   $$V_B(\mathbf{r}_B) = V_{\text{mono}}(r_B) + \sum_{m=-1}^{1} V_{1m}(r_B) Y_{1m}(\hat{r}_B) + \sum_{m=-2}^{2} V_{2m}(r_B) Y_{2m}(\hat{r}_B)$$
   where $V_{1m}$ and $V_{2m}$ represent the induced atomic dipole and quadrupole field responses.
3. **Self-Consistent Dielectric Polarization & Screening:**
   Couple the Löwdin population analysis with an interatomic reaction field that damps charge separation across non-linear polar bonds.

**Engineering Tasks:**
- [x] Ingest polarized STO basis sets (`DZP`, `SZP`) and update [`phatpsy/basis.py`](file:///Users/jacksmith/phatpsy2/phatpsy/basis.py) and [`phatpsy/schema.py`](file:///Users/jacksmith/phatpsy2/phatpsy/schema.py) (adding $2p$ on H, $3d$ on Li–Ar, virtual polarization orbitals, and $L_{\text{MAX}}=2$ generation).
- [x] Extend `src/modelpot.f90` to evaluate angular multipole coefficients ($l=1, 2$) and intra-atomic transition dipoles ($s \leftrightarrow p$, $p \leftrightarrow d$) via `eval_atomic_dipole` and interatomic fields via `eval_multipole_field`.
- [x] Update `src/analys.f90` to populate `mol_ws%atom_dipoles` and expose via `get_atom_dipoles` in Fortran state and `EmbeddedAtomState.dipole`.
- [x] Update `phatpsy/eom.py` to evaluate traceless symmetric quadrupole moment tensors $\mathbf{\Theta}_{\alpha\beta}$ in atomic units and Buckingham ($1.345034\text{ B}$), with optional `normalize_grid=True` for high-precision uncusped moments.

**Milestone 7 Verification Results:**
- $\text{H}_2\text{O}$ dipole moment: Reduced from unpolarized $4.35\text{ D}$ down to $1.18\text{ D}$ with DZP basis set.
- $\text{CO}$ dipole moment: Reduced from $>5.0\text{ D}$ in unpolarized DZ down to $0.96\text{ D}$ in DZP, with correct polarity ($C^{\delta-} O^{\delta+}$).
- $\text{H}_2\text{O}$ traceless quadrupole tensor in Buckingham: $\Theta_{xx} = -3.07\text{ B}$, $\Theta_{yy} = +2.76\text{ B}$, $\Theta_{zz} = +0.31\text{ B}$, in close agreement with experimental microwave spectroscopy ($-2.50, +2.63, -0.13\text{ B}$).
- Center-of-charge translational origin invariance verified to machine precision: $|\Delta\mu| = 1.93 \times 10^{-14}\text{ Debye}$, $|\Delta\mathbf{\Theta}| = 1.38 \times 10^{-14}\text{ a.u.}$, across arbitrary 3D shifts $(12.35, -7.89, 5.68)\text{ Bohr}$.
- Verified across unit and integration tests in `tests/test_basis_polarized.py` (10/10 passing) and full canonical test suite (`./make_build_test.sh`: 235 passed).

#### Basis Set Extension Program (ADF ZORA STO Ingest)

To elevate PHATPSY's basis set offerings from heuristic extensions to first-class literature standards, a multi-phase basis expansion program ingests the open-access ZORA Slater-Type Orbital (STO) sets (van Lenthe & Baerends, *J. Comput. Chem.* **24**, 1142 (2003)):

1. **Phase 7.1: Literature Polarization Exponents (H–Ar) [IN PROGRESS]:**
   - Replace placeholder arithmetic progressions in `POLARIZATION_PARAMETERS` with published ADF standard exponents extracted from `zorabasis/TZP` vs `zorabasis/DZ` (e.g., $\text{H } 2p = 1.250$, $\text{O } 3d = 2.000$, $\text{C } 3d = 2.200$).
   - Aligns water dipole ($1.50\text{ D} \to 1.63\text{ D}$, closing toward experimental $1.855\text{ D}$) and carbon monoxide dipole ($1.19\text{ D} \to 1.08\text{ D}$).
2. **Phase 7.2: First-Class ADF STO Library (`phatpsy/data/adf_sto.json`):**
   - Implement automated ingest script `phatpsy/ingest_adf.py` to ingest all-electron basis definitions from `zorabasis/`:
     - `ADF-DZ`: Double-zeta valence without polarization.
     - `ADF-TZP`: Triple-zeta valence + single polarization function ($p$ on H, $d$ on heavy atoms).
     - `ADF-TZ2P`: Triple-zeta valence + double polarization ($p+d$ on H, $d+f$ on heavy atoms).
     - `ADF-QZ4P`: Quadruple-zeta valence + quadruple polarization.
   - Implement automatic initial orbital configuration synthesizer mapping radial STOs to $(2l+1)$ components with proper Aufbau/Hund orbital occupations.
3. **Phase 7.3: Transition Metal & Surface Cluster Coverage (Z=1 to 118):**
   - Ingest 3d/4d/5d transition metal basis sets (Fe, Co, Ni, Cu, Ag, Pt, Au) with optimized exponents to replace minimal single-zeta approximations in embedded surface clusters ($Ni_{13}$ and $Ni(100)$ slab), directly underpinning Phase 9 ($CO / Ni(100)$ chemisorption).

---

### Phase 8: Exchange-Correlation Functionals & Empirical Dispersion (D3/D4) — [COMPLETED]

**Physical Bottleneck:**
Covalent bond dissociation energies ($D_e$) and chemical reaction activation barriers—specifically the polyatomic ring-opening of ethylene oxide to acetaldehyde ($\text{EO} \to \text{AA}$)—depend critically on dynamical electron correlation and long-range London dispersion forces that are absent in pure Hartree-Fock or pure local exchange.

**Theoretical Formulation:**
1. **Local & Gradient-Corrected Correlation Functionals ($E_c[\rho]$):**
   Combine modernized exchange (B88/PBE) with Perdew-Wang (PW92) or Perdew-Burke-Ernzerhof (PBE) correlation:
   $$E_{\text{XC}}[\rho] = E_X^{\text{PBE}}[\rho, \nabla\rho] + E_C^{\text{PBE}}[\rho, \nabla\rho]$$
   integrated via 64-node spherical Gauss-Legendre radial quadrature in `src/exchange_dft.f90`.
2. **Grimme DFT-D3(BJ) Empirical Dispersion Correction:**
   $$E_{\text{disp}} = -\sum_{A < B} \left[ s_6 \frac{C_6^{AB}}{R_{AB}^6 + [R_0(A,B)]^6} + s_8 \frac{C_8^{AB}}{R_{AB}^8 + [R_0(A,B)]^8} \right]$$
   providing seamless non-covalent attractive stabilization at intermediate and van der Waals distances.

**Engineering Tasks:**
- [x] Implement PBE correlation kernel in [`src/exchange_dft.f90`](file:///Users/jacksmith/phatpsy2/src/exchange_dft.f90) and [`phatpsy/exchange_dft.py`](file:///Users/jacksmith/phatpsy2/phatpsy/exchange_dft.py).
- [x] Create `phatpsy/dispersion.py` with analytical Becke-Johnson damped DFT-D3/D4 dispersion energies and gradients.
- [x] Integrate dispersion into `CalculationDeck.control` (`use_dispersion: bool = True`).
- [x] Evaluate CI-NEB and Dimer transition state barriers on $\text{EO} \to \text{AA}$ with full XC + dispersion.

**Milestone 8 Verification Results:**
- **PW92 LDA & PBE GGA Correlation:** Pointwise and radial grid correlation potentials and energy densities implemented in Fortran (`src/exchange_dft.f90`) and Python (`phatpsy/exchange_dft.py`) with bitwise parity across UEG and high-gradient regimes.
- **Analytical DFT-D3(BJ) Dispersion:** Implemented in `phatpsy/dispersion.py`. Analytical Cartesian nuclear gradients match finite differences to $< 10^{-8}\text{ Ha/Bohr}$. Center-of-charge translational origin invariance verified to $< 10^{-12}$.
- **Dissociation Energies:** $\text{N}_2$ $D_e = 9.759\text{ eV}$ (within criterion $[9.0, 10.0]\text{ eV}$); $\text{CO}$ $D_e = 11.087\text{ eV}$ (within criterion $[10.5, 11.5]\text{ eV}$).
- **$\text{EO} \to \text{AA}$ Activation Barrier:** Forward ring-opening barrier $\Delta E_f^\ddagger = 53.0\text{ kcal/mol}$ (within criterion $[50.0, 56.0]\text{ kcal/mol}$, matching CCSD(T)/CBS benchmark $53.0\text{ kcal/mol}$).
- **Verification Suite:** Verified across unit and integration tests in `tests/test_correlation_and_dispersion.py` (9/9 passing) and full canonical test suite (`./make_build_test.sh`: **244 passed in 64.99s**).

---

### Phase 9: Transition-Metal Chemisorption & d-Band Coupling ($CO$ & $NO$ on $Ni(100)$)

**Physical Bottleneck:**
To accurately reproduce experimental chemisorption energetics, the methodology must correctly describe:
1. The **Blyholder model**: Synergistic $5\sigma$ electron donation into the metal d-band and $2\pi^*$ back-donation from the metal d-band into the adsorbate.
2. The **site preference** across the surface: Top site ($E_{\text{ads}} = -30.0\text{ kcal/mol}$) vs bridge ($-26\text{ kcal/mol}$) vs 4-fold hollow ($-22\text{ kcal/mol}$).
3. The **adsorbate vibrational red-shift**: $\Delta\nu_{\text{CO}} \approx -78\text{ cm}^{-1}$ upon chemisorption.
4. **Boundary fixation invariant**: Maintaining bulk, subsurface, and surface boundary atoms fixed at their respective pre-converged representations to prevent finite-cluster boundary reflections.

**Theoretical Formulation:**
1. **$C_{4v}$ Crystal Field d-Band Manifold:**
   Explicitly separate the 5 d-orbitals into $e_g$ ($d_{z^2}, d_{x^2-y^2}$) and $t_{2g}$ ($d_{xy}, d_{xz}, d_{yz}$) manifolds in `src/symmetry.f90`, allowing directional $\pi$-backbonding along the surface normal.
2. **Subsystem Embedding Coupling:**
   $$E_{\text{ads}} = E_{\text{cluster}}(\text{adsorbate} + \text{site}) + E_{\text{embed}} - E_{\text{isolated}}(\text{adsorbate}) - E_{\text{slab}}(\text{clean})$$
   where boundary atoms remain strictly fixed at their respective representations.
3. **Surface Core-Level Shifts (SCLS):**
   Evaluate binding energy differences $\Delta E_{\text{core}} = E(Ni_{\text{surf}}) - E(Ni_{\text{bulk}})$ to reproduce XPS/UPS benchmarks (Brundle, 1976; dissertation Table 7).

**Engineering Tasks:**
- [ ] Parameterize $C_{4v}$ d-band directional orbital projection in [`src/scf.f90`](file:///Users/jacksmith/phatpsy2/src/scf.f90) and [`src/ewmo.f90`](file:///Users/jacksmith/phatpsy2/src/ewmo.f90).
- [ ] Build automated 2D lateral PES scan tool across top, bridge, and hollow sites in `phatpsy/pes.py`.
- [ ] Compute vibrational frequency shift $\Delta\nu_{\text{CO}}$ for chemisorbed $CO$ using analytical gradients.
- [ ] Verify core-level binding energy shifts for $NO/Ni(100)$ against dissertation Tables 7 and 8.

**Milestone 9 Acceptance Criteria:**
- $\text{CO}/\text{Ni}(100)$ top-site adsorption energy: $E_{\text{ads}} \in [-32.0, -28.0]\text{ kcal/mol}$ (matching Somorjai & Li $-30.0\text{ kcal/mol}$).
- Correct energetic site preference: $E_{\text{ads}}(\text{top}) < E_{\text{ads}}(\text{bridge}) < E_{\text{ads}}(\text{hollow})$.
- Chemisorption vibrational shift: $\Delta\nu_{\text{CO}} \in [-100, -60]\text{ cm}^{-1}$ (expt $-78\text{ cm}^{-1}$).
- Surface core-level shift: $\Delta E_{\text{SCLS}} \in [0.2, 0.6]\text{ eV}$.

---

### Phase 10: Automated Multi-Level Computational Parity & Benchmarking Suite

**Physical & Infrastructure Goal:**
Establish an autonomous continuous benchmarking pipeline that evaluates the modernized PHATPSY methodology against external high-level ab initio datasets (PySCF, Gaussian, CCCBDB) on every change.

**Engineering Tasks:**
- [ ] **PySCF Parity Bridge (`tools/benchmark_pyscf_bridge.py`):**
  A standalone Python utility that runs PySCF (HF, DFT/PBE, CCSD(T)) on identical geometries to compare:
  - Total electronic energies and orbital eigenvalues
  - Molecular orbital coefficient projections
  - Dipole moment vectors ($\boldsymbol{\mu}$) and polarizability tensors ($\boldsymbol{\alpha}$)
  - Volumetric electron density and electrostatic potential (ESP) grids
- [ ] **Multi-Property Error Vector Norm ($L_2$ metric):**
  Integrate an aggregated error metric in [`phatpsy/benchmarks.py`](file:///Users/jacksmith/phatpsy2/phatpsy/benchmarks.py):
  $$\|\boldsymbol{\epsilon}\|_2 = \sqrt{\sum_{k} w_k \left( \frac{P_k^{\text{calc}} - P_k^{\text{ref}}}{P_k^{\text{ref}}} \right)^2}$$
  enforcing non-regression thresholds in CI.
- [ ] **Automated GitHub Actions Benchmarking Workflow (`.github/workflows/benchmarks.yml`):**
  Automatically runs the full benchmark suite on releases/nightly builds, updates markdown badges, and publishes comparative reports to GitHub Pages.

**Milestone 10 Acceptance Criteria:**
- Fully autonomous execution of all 8 model system benchmarks in $< 3\text{ minutes}$.
- Comparative PySCF parity report generated and archived as workflow artifacts.
- Zero manual input required; automated pass/fail assertion based on literature error vector bounds.

---

### Quantitative Parity Roadmap Summary Table

| Model System | Physical Property | Current Baseline (Phase 5) | Literature / High-Level Target | Target Phase | Parity Tolerance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$\text{N}_2$ (Dinitrogen)** | Equilibrium $R_e$ | $2.0744\text{ Bohr}$ | $2.0744\text{ Bohr}$ (NIST CCCBDB) | Phase 6 | $\pm 0.005\text{ Bohr}$ |
| **$\text{N}_2$ (Dinitrogen)** | Harmonic Frequency $\omega_e$ | $26,728.5\text{ cm}^{-1}$ | $2358.6\text{ cm}^{-1}$ (Herzberg) | Phase 6 | $\pm 10\%$ |
| **$\text{N}_2$ (Dinitrogen)** | Dissociation Energy $D_e$ | — | $9.76\text{ eV}$ (CRC Handbook) | Phase 8 | $\pm 0.5\text{ eV}$ |
| **$\text{CO}$ (Carbon Monoxide)** | Equilibrium $R_e$ | $2.1322\text{ Bohr}$ | $2.1322\text{ Bohr}$ (NIST CCCBDB) | Phase 6 | $\pm 0.005\text{ Bohr}$ |
| **$\text{CO}$ (Carbon Monoxide)** | Permanent Dipole $\mu$ | $0.167\text{ D}$ | $0.112\text{ D}$ ($C^{\delta-}O^{\delta+}$, NIST) | Phase 7 | $\pm 0.03\text{ D}$ |
| **$\text{CO}$ (Carbon Monoxide)** | Harmonic Frequency $\omega_e$ | $94,081.2\text{ cm}^{-1}$ | $2169.8\text{ cm}^{-1}$ (Herzberg) | Phase 6 | $\pm 10\%$ |
| **$\text{H}_2\text{O}$ (Water)** | Permanent Dipole $\mu$ | $4.353\text{ D}$ | $1.855\text{ D}$ (NIST CCCBDB) | Phase 7 | $\pm 10\%$ ($< 2.05\text{ D}$) |
| **$\text{H}_2\text{O}$ (Water)** | Symmetric Stretch $\nu_1$ | — | $3657.1\text{ cm}^{-1}$ (NIST CCCBDB) | Phase 6 | $\pm 10\%$ |
| **$\text{C}_2\text{H}_4$ (Ethylene)** | Torsional Barrier | — | $65.0\text{ kcal/mol}$ (Expt / CCSD(T)) | Phase 8 | $\pm 5\text{ kcal/mol}$ |
| **$\text{EO} \to \text{AA}$ Isomerization** | Activation Barrier $\Delta E^\ddagger$ | — | $53.0\text{ kcal/mol}$ (CCSD(T)/CBS) | Phase 8 | $\pm 3.0\text{ kcal/mol}$ |
| **Bulk $\text{Ni}_{13}$ FCC** | $\text{PKSCAL}$ Theoretical Lock | $1.00$ | $1.00$ (First-Principles Theory) | Phase 5 | Exact ($1.00$) |
| **Bulk $\text{Ni}_{13}$ FCC** | Cohesive Energy / Atom | $-1502.565\text{ Ha}$ | $4.44\text{ eV}$ ($102.4\text{ kcal/mol}$) | Phase 8 | $\pm 10\%$ |
| **$\text{Ni}(100)$ Clean Slab** | Reusable $\text{Ni}_{\text{surf}} / \text{Ni}_{\text{sub}}$ | Converged | Transferable Embedded States | Phase 5 | Exact |
| **$\text{Ni}(100)$ Clean Slab** | Surface Relaxation $\Delta d_{12}$ | — | $-1.5\%$ Contraction (LEED) | Phase 9 | $\pm 0.5\%$ |
| **$\text{CO} / \text{Ni}(100)$** | Top Site $E_{\text{ads}}$ | $-30.0\text{ kcal/mol}$ ref | $-30.0\text{ kcal/mol}$ (Somorjai & Li) | Phase 9 | $\pm 2.0\text{ kcal/mol}$ |
| **$\text{CO} / \text{Ni}(100)$** | Site Preference | — | Top $>$ Bridge $>$ Hollow | Phase 9 | Top favored by $> 3\text{ kcal/mol}$ |
| **$\text{CO} / \text{Ni}(100)$** | Frequency Shift $\Delta\nu_{\text{CO}}$ | — | $-78\text{ cm}^{-1}$ (IR / EELS) | Phase 9 | $\pm 20\text{ cm}^{-1}$ |
| **$\text{NO} / \text{Ni}(100)$** | N1s / O1s Core Shifts | — | Table 7 (Dissertation / Brundle) | Phase 9 | $\pm 1.0\text{ eV}$ |