# PHATPSY 2.0 Validation Standards & Benchmarks

This document specifies the numerical validation standards, physical invariants, reference datasets, and test execution procedures for **PHATPSY 2.0**.

---

## 1. Physical Invariants & Quality Gates

Every calculation and modernized kernel must satisfy the following physical quality gates:

1. **Virial Ratio Test ($-V/T$):**
   - For an isolated neutral molecule in its ground state at equilibrium geometry, the quantum virial theorem requires:
     $$\frac{-\langle V \rangle}{\langle T \rangle} \approx 2.000$$
   - In modern PHATPSY 2.0, standard benchmark systems ($N_2$, $CO$) must converge with $|-V/T - 2.0| \le 0.05$.
2. **Monotonic Total Energy Convergence:**
   - Under standard damping (`CDAMP \approx 0.5`, `VDAMP \approx 0.5`), the cycle-by-cycle total electronic energy must converge monotonically after initial atomic preparation.
3. **Spin Conservation in Localized Valence-Bond (LVB) States:**
   - For closed-shell molecules treated with alternating spin (`spin="up"` / `spin="down"`), the net molecular spin must remain strictly zero:
     $$\langle S_z \rangle = \frac{1}{2} (N_\alpha - N_\beta) = 0.000$$

---

## 2. Numerical Tolerances

Because PHATPSY 2.0 migrates from legacy 1970s algorithms (EISPAK, manual matrix inversion) to modern IEEE-754 64-bit precision with standard **LAPACK** (`dsyev`, `dgels`), minor numerical divergence at the 3rd or 4th decimal place is normal and expected:

| Comparison Metric | Acceptable Tolerance | Rationale |
| :--- | :--- | :--- |
| **Cycle Total Energies** | $\text{atol} = 1.0 \times 10^{-2} \text{ Hartree}$ | Accounts for LAPACK eigenvector phase and QR least-squares versus legacy normal equations. |
| **Final Total Molecular Energy** | $\text{atol} = 1.0 \times 10^{-2} \text{ Hartree}$ | Agreement across full SCF cycles. |
| **Orbital Eigenvalues (MO Spectrum)** | $\text{atol} = 5.0 \times 10^{-3} \text{ Hartree}$ | LAPACK `dsyev` produces orthonormalized degenerate spaces without legacy Jacobi drift. |
| **Gross Atomic Populations** | $\text{atol} = 1.0 \times 10^{-2} \text{ electrons}$ | Mulliken/EWMO population projection stability. |
| **Internal Python Data Model Roundtrip** | $\text{atol} = 1.0 \times 10^{-6}$ | YAML/JSON/dataclass conversion fidelity. |

---

## 3. Standard Benchmark Test Cases

The following test calculations in [`examples/`](../../examples) and reference outputs in [`data/`](../../data) serve as standard regression benchmarks:

1. **Nitrogen Molecule ($N_2$):**
   - Deck: [`examples/n2.stdin`](../../examples/n2.stdin) / [`examples/n2.yaml`](../../examples/n2.yaml)
   - Baseline Output: [`data/n2_modern.stdout`](../../data/n2_modern.stdout)
   - Benchmark Values:
     - Total Molecular Energy: $\approx -109.5689 \text{ Hartree}$
     - Total Kinetic Energy: $\approx +108.2264 \text{ Hartree}$
     - Total Potential Energy: $\approx -217.7952 \text{ Hartree}$
     - Virial Ratio ($-V/T$): $\approx 2.0124$
2. **Carbon Monoxide ($CO$):**
   - Deck: [`examples/CO.stdin`](../../examples/CO.stdin) / [`examples/CO.yaml`](../../examples/CO.yaml)
   - Baseline Output: [`data/CO_modern.stdout`](../../data/CO_modern.stdout)
3. **Silver Cluster ($Ag_{13}$ Double-Zeta):**
   - Deck: [`examples/ag13dz.stdin`](../../examples/ag13dz.stdin)
   - Baseline Output: [`data/ag13dz_modern.stdout`](../../data/ag13dz_modern.stdout)
   - Demonstrates transition-metal localized pseudopotential scaling and GCHF convergence.

---

## 4. Test Execution Instructions

Run the full automated test suite using pytest:
```bash
# Run all unit and regression tests
.venv/bin/pytest tests/

# Run specifically the Fortran/Python numerical regression tests
.venv/bin/pytest tests/test_phatpsy_regression.py tests/test_python_driver.py -v

# Run Python model and visualizer tests
.venv/bin/pytest tests/test_builder.py tests/test_clementi_basis.py tests/test_spin_and_visualizer.py -v
```
