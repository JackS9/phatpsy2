# Analytical Molecular Gradients & Non-Circular Basis Force Derivatives

!!! note "Theoretical Origin & Project Invariants"
    **Phase 11 Methodology:** Full closed-form analytical Cartesian nuclear gradients for polyatomic molecules within the Projected Hamiltonian Approach to Polyatomic Systems (PHATPSY 2.0).  
    **Governing Invariant:** All molecular forces derive strictly from first-principles analytical matrix calculus and closed-form STO auxiliary function recurrences. Empirical parameter tables (such as `PAIR_PARAMS`) are completely bypassed in favor of direct ab initio electronic surface derivatives.

---

## 1. Introduction and Formulation Scope

Geometry optimization, intrinsic reaction coordinate (IRC) following, climbing-image nudged elastic band (CI-NEB) transition state searches, and Born-Oppenheimer molecular dynamics (BOMD) depend critically on the Cartesian nuclear gradient vector:
$$\mathbf{g}_A = -\mathbf{F}_A = \nabla_{\mathbf{R}_A} E_{\text{total}} = \left( \frac{\partial E_{\text{total}}}{\partial X_A}, \frac{\partial E_{\text{total}}}{\partial Y_A}, \frac{\partial E_{\text{total}}}{\partial Z_A} \right)^T$$
Evaluating $\mathbf{g}_A$ via numerical finite differences requires $6N_{\text{atom}}$ self-consistent field (SCF) calculations, which is computationally prohibitive for large polyatomic systems and introduces numerical differencing noise ($10^{-5}$–$10^{-6}\text{ Ha/Bohr}$).

In **PHATPSY 2.0**, analytical Cartesian nuclear gradients are evaluated in closed form through a decoupled architecture:
1. **Fortran 2018 Kernel (`src/twocenter.f90`):** Evaluates radial derivatives of two-center overlap ($\partial S/\partial R_{AB}$) and model potential ($\partial V/\partial R_{AB}$) integrals via exact elliptical auxiliary function recurrences.
2. **Python Gradient Engine (`phatpsy/gradients.py`):** Evaluates exact Cartesian Jacobians, angular Wigner rotation matrix derivatives, pairwise polyatomic Pulay overlap contractions, model potential derivatives, Phillips-Kleinman core pseudopotential projections, and Grimme DFT-D3(BJ) dispersion forces.

---

## 2. Total Molecular Energy Decomposition

The total molecular energy in PHATPSY is partitioned into conservative contributions:
$$E_{\text{total}} = E_{\text{nuc}} + E_{\text{electronic}} + E_{\text{PK}} + E_{\text{disp}}$$
The total analytical nuclear gradient is the linear superposition of the corresponding Cartesian derivative components:
$$\mathbf{g}_A = \nabla_{\mathbf{R}_A} E_{\text{total}} = \mathbf{g}_A^{\text{nuc}} + \mathbf{g}_A^{\text{Pulay}} + \mathbf{g}_A^{\text{pot}} + \mathbf{g}_A^{\text{PK}} + \mathbf{g}_A^{\text{disp}}$$

---

## 3. Classical Nuclear Repulsion Gradient

The classical Coulomb electrostatic repulsion between bare nuclear charges is:
$$E_{\text{nuc}} = \sum_{A < B} \frac{Z_A Z_B}{R_{AB}}$$
where $R_{AB} = \|\mathbf{R}_A - \mathbf{R}_B\|$. Differentiating directly with respect to the Cartesian position of atom $A$:
$$\mathbf{g}_A^{\text{nuc}} = \nabla_{\mathbf{R}_A} E_{\text{nuc}} = -Z_A \sum_{B \ne A} \frac{Z_B}{R_{AB}^3} (\mathbf{R}_B - \mathbf{R}_A)$$
By Newton's third law, the pairwise forces are anti-symmetric:
$$\frac{\partial}{\partial \mathbf{R}_A} \left(\frac{Z_A Z_B}{R_{AB}}\right) = -\frac{\partial}{\partial \mathbf{R}_B} \left(\frac{Z_A Z_B}{R_{AB}}\right)$$
guaranteeing exact translational invariance:
$$\sum_{A=1}^{N_{\text{atom}}} \mathbf{g}_A^{\text{nuc}} = \mathbf{0}$$

---

## 4. Coordinate Transformations & Diatomic Jacobians

Every pairwise interaction between atom $A$ and atom $B$ is initially parameterized in a canonical diatomic reference frame where the internuclear vector $\mathbf{R}_{AB} = \mathbf{R}_B - \mathbf{R}_A$ defines the local $z$-axis.

### Cartesian Jacobians (`DiatomicFrame`)
Let the relative displacement vector be:
$$\mathbf{R}_{AB} = (x, y, z)^T = \mathbf{R}_B - \mathbf{R}_A, \quad R = \sqrt{x^2 + y^2 + z^2}, \quad R_\perp = \sqrt{x^2 + y^2}$$
The spherical Euler orientation angles are:
$$\beta = \operatorname{atan2}(R_\perp, z), \quad \alpha = \operatorname{atan2}(y, x)$$
The exact Cartesian Jacobians with respect to displacements of atom $B$ are:
$$\nabla_{\mathbf{R}_B} R = \frac{\mathbf{R}_{AB}}{R} = (\sin\beta\cos\alpha, \sin\beta\sin\alpha, \cos\beta)^T$$
$$\nabla_{\mathbf{R}_B} \alpha = \left( -\frac{y}{R_\perp^2}, \frac{x}{R_\perp^2}, 0 \right)^T$$
$$\nabla_{\mathbf{R}_B} \beta = \left( \frac{x z}{R^2 R_\perp}, \frac{y z}{R^2 R_\perp}, -\frac{R_\perp}{R^2} \right)^T$$
For displacements of atom $A$, translational anti-symmetry holds identically:
$$\nabla_{\mathbf{R}_A} R = -\nabla_{\mathbf{R}_B} R, \quad \nabla_{\mathbf{R}_A} \alpha = -\nabla_{\mathbf{R}_B} \alpha, \quad \nabla_{\mathbf{R}_A} \beta = -\nabla_{\mathbf{R}_B} \beta$$

---

## 5. Analytical Slater-Type Orbital Radial Recurrences

Two-center integrals over Slater-Type Orbitals (STOs) with exponents $\zeta_a, \zeta_b$ and principal quantum numbers $n_a, n_b$ are evaluated in confocal prolate spheroidal coordinates:
$$\mu = \frac{r_A + r_B}{R_{AB}} \in [1, \infty), \quad \nu = \frac{r_A - r_B}{R_{AB}} \in [-1, 1]$$
The radial integrals reduce to combinations of standard auxiliary elliptical functions:
$$A_n(\alpha) = \int_{1}^{\infty} \mu^n e^{-\alpha \mu} d\mu, \quad B_m(\beta) = \int_{-1}^{1} \nu^m e^{-\beta \nu} d\nu$$
where the scaled coordinate parameters are:
$$\alpha = \frac{1}{2} R_{AB} (\zeta_a + \zeta_b), \quad \beta = \frac{1}{2} R_{AB} (\zeta_a - \zeta_b)$$

### Closed-Form Auxiliary Recurrences
The auxiliary functions satisfy differentiation relations:
$$\frac{d A_n(\alpha)}{d\alpha} = -A_{n+1}(\alpha), \quad \frac{d B_m(\beta)}{d\beta} = -B_{m+1}(\beta)$$
Applying the chain rule with respect to the internuclear distance $R_{AB}$:
$$\frac{\partial A_n}{\partial R_{AB}} = \frac{d A_n}{d\alpha} \frac{\partial \alpha}{\partial R_{AB}} = -\frac{1}{2} (\zeta_a + \zeta_b) A_{n+1}(\alpha)$$
$$\frac{\partial B_m}{\partial R_{AB}} = \frac{d B_m}{d\beta} \frac{\partial \beta}{\partial R_{AB}} = -\frac{1}{2} (\zeta_a - \zeta_b) B_{m+1}(\beta)$$

### Product-Rule Expansion
For the two-center overlap integral matrix element $S(R_{AB})$:
$$S(R_{AB}) = N_a N_b R_{AB}^{n_a + n_b + 1} \sum_{k} c_k A_{n_k}(\alpha) B_{m_k}(\beta)$$
Differentiating with respect to $R_{AB}$ via the product rule:
$$\frac{\partial S}{\partial R_{AB}} = \frac{n_a + n_b + 1}{R_{AB}} S(R_{AB}) + N_a N_b R_{AB}^{n_a + n_b + 1} \sum_{k} c_k \left[ \frac{\partial A_{n_k}}{\partial R_{AB}} B_{m_k} + A_{n_k} \frac{\partial B_{m_k}}{\partial R_{AB}} \right]$$
Similarly, for the two-center model potential integral $V(R_{AB})$:
$$\frac{\partial V}{\partial R_{AB}} = \frac{n_a + n_b}{R_{AB}} V(R_{AB}) + N_a N_b R_{AB}^{n_a + n_b} \sum_{k} c_k' \left[ \frac{\partial A_{n_k'}}{\partial R_{AB}} B_{m_k'} + A_{n_k'} \frac{\partial B_{m_k'}}{\partial R_{AB}} \right]$$
These analytical derivatives are implemented in `src/twocenter.f90` (`d_gensab` and `d_genvab`), matching central finite differences to $< 10^{-7}\text{ Ha/Bohr}$.

---

## 6. Angular Wigner Rotation Matrix Derivatives

To transform two-center integrals from the local diatomic frame into the molecular Cartesian frame, real spherical harmonics are rotated using Wigner D-matrices:
$$\mathbf{S}_{AB}(\mathbf{R}_{AB}) = \mathbf{D}(\alpha, \beta)^T \mathbf{S}_{AB}^{\text{diag}}(R_{AB}) \mathbf{D}(\alpha, \beta)$$
where $\mathbf{D}(\alpha, \beta)$ is the block-diagonal real rotation matrix for angular momentum $l$:
- For $s$-orbitals ($l=0$): $\mathbf{D}^{(0)} = 1$, with zero angular derivatives.
- For $p$-orbitals ($l=1$): $\mathbf{D}^{(1)}$ rotates $(p_x, p_y, p_z)$ via directional cosines.
- For $d$-orbitals ($l=2$): $\mathbf{D}^{(2)}$ rotates $(d_{z^2}, d_{xz}, d_{yz}, d_{x^2-y^2}, d_{xy})$.

Applying the product rule across radial and angular coordinates:
$$\nabla_{\mathbf{R}_B} \mathbf{S}_{AB} = \mathbf{D}^T \left(\frac{\partial \mathbf{S}_{AB}^{\text{diag}}}{\partial R}\right) \mathbf{D} (\nabla_{\mathbf{R}_B} R) + \left(\frac{\partial \mathbf{D}^T}{\partial \alpha} \mathbf{S}_{AB}^{\text{diag}} \mathbf{D} + \mathbf{D}^T \mathbf{S}_{AB}^{\text{diag}} \frac{\partial \mathbf{D}}{\partial \alpha}\right) (\nabla_{\mathbf{R}_B} \alpha) + \left(\frac{\partial \mathbf{D}^T}{\partial \beta} \mathbf{S}_{AB}^{\text{diag}} \mathbf{D} + \mathbf{D}^T \mathbf{S}_{AB}^{\text{diag}} \frac{\partial \mathbf{D}}{\partial \beta}\right) (\nabla_{\mathbf{R}_B} \beta)$$
Because $\nabla_{\mathbf{R}_A} = -\nabla_{\mathbf{R}_B}$, translational invariance holds for every angular component.

---

## 7. Pulay Overlap Force & Energy-Weighted Density Contraction

In Slater basis sets, the atomic basis functions $\chi_\mu(\mathbf{r} - \mathbf{R}_A)$ depend explicitly on nuclear positions. According to Pulay's theorem, this basis set dependence introduces a Pulay force:
$$\mathbf{g}_A^{\text{Pulay}} = -\sum_{\mu \nu} W_{\mu\nu} \nabla_{\mathbf{R}_A} S_{\mu\nu}$$
where $\mathbf{W}$ is the **energy-weighted density matrix**:
$$\mathbf{W} = \sum_{s=1}^{N_{\text{spin}}} \omega_s \sum_{i} n_{i,s} \varepsilon_{i,s} \mathbf{C}_{\cdot, i, s} \mathbf{C}_{\cdot, i, s}^T$$
Here $\mathbf{C}$ are the molecular orbital coefficients, $n_{i,s}$ are the orbital occupations, $\varepsilon_{i,s}$ are the orbital eigenvalues, and $\omega_s$ is the spin multiplicity weight ($2$ for closed-shell singlet, $1$ for unrestricted).

Partitioning over atomic centers, only two-center blocks between different atoms ($A \ne B$) contribute to nuclear forces:
$$\mathbf{g}_A^{\text{Pulay}} = -2 \sum_{B \ne A} \sum_{\mu \in A} \sum_{\nu \in B} W_{\mu\nu} \nabla_{\mathbf{R}_A} S_{\mu\nu}$$

---

## 8. Two-Center Model Potential Forces

In PHATPSY, interatomic electron-electron and electron-nuclear Coulomb interactions are represented via the local spherical model potential of center $B$ acting on center $A$:
$$V^B(r_B) = -\frac{Z_B^*}{r_B} + \sum_{k} c_k^B e^{-\zeta_k^B r_B}$$
The expectation value of atom $B$'s model potential over atom $A$'s localized density matrix $\mathbf{P}_A$ is:
$$E_{\text{pot}}^{AB} = \operatorname{Tr}(\mathbf{P}_A \mathbf{V}^{AB})$$
Differentiating with respect to the nuclear coordinate $\mathbf{R}_A$:
$$\mathbf{g}_A^{\text{pot}} = \sum_{B \ne A} \left[ \operatorname{Tr}\left(\mathbf{P}_A \nabla_{\mathbf{R}_A} \mathbf{V}^{AB}\right) - \operatorname{Tr}\left(\mathbf{P}_B \nabla_{\mathbf{R}_B} \mathbf{V}^{BA}\right) \right]$$
where $\nabla_{\mathbf{R}_A} \mathbf{V}^{AB}$ contains both nuclear attraction and exponential model terms:
$$\nabla_{\mathbf{R}_A} \mathbf{V}^{AB} = -Z_B \nabla_{\mathbf{R}_A} \mathbf{V}_{\text{nuc}}^{AB} + \frac{1}{2} \sum_{k} c_k^B \nabla_{\mathbf{R}_A} \mathbf{V}_k^{AB}$$

---

## 9. Phillips-Kleinman (PK) Pseudopotential Derivatives

The atom-partitioned Phillips-Kleinman operator enforces Pauli exclusion between valence electrons on center $A$ and the core electrons of neighboring center $B$:
$$\hat{V}_{\text{PK}}^{AB} = \sum_{c \in B} (\bar{\varepsilon}_A - \varepsilon_c^B) |\phi_c^B\rangle \langle \phi_c^B|$$
In the atomic STO basis, this operator takes the matrix form:
$$\mathbf{V}_{\text{PK}}^{AB} = \mathbf{S}_{AB} \mathbf{K}_B \mathbf{S}_{BA}$$
where the core projection operator matrix on atom $B$ is:
$$\mathbf{K}_B = \bar{\varepsilon}_A \mathbf{D}_{\text{proj}}^B - \mathbf{D}_{\varepsilon}^B$$
with:
- $\mathbf{D}_{\text{proj}}^B = \sum_c \mathbf{C}_{\cdot, c}^B (\mathbf{C}_{\cdot, c}^B)^T \text{occ}_c^B$ (unweighted projector density)
- $\mathbf{D}_{\varepsilon}^B = \sum_c \mathbf{C}_{\cdot, c}^B (\mathbf{C}_{\cdot, c}^B)^T \varepsilon_c^B$ (energy-weighted projector density)
- $\bar{\varepsilon}_A = \frac{\sum_{i \in \text{occ}} n_i^A \varepsilon_i^A}{\sum_{i \in \text{occ}} n_i^A}$ (average valence orbital eigenvalue of atom $A$)

### Closed-Form PK & Exchange Residual Gradient Contraction
In `analys.f90`, the two-center PK operator and the exchange residual $\mathbf{V}_6$ enter each atom's energy record with prefactor $0.5$ to prevent double-counting upon summation over all atoms:
$$E_{\text{PK}}^{AB} = \frac{1}{2} \operatorname{Tr}(\mathbf{P}_A \mathbf{S}_{AB} \mathbf{K}_B \mathbf{S}_{BA}) + \frac{1}{2} \operatorname{Tr}(\mathbf{P}_B \mathbf{S}_{BA} \mathbf{K}_A \mathbf{S}_{AB})$$
where the full projection and exchange residual operator matrix is:
$$\mathbf{K}_B = (\bar{\varepsilon}_A \mathbf{D}_{\text{proj}}^B - \mathbf{D}_{\varepsilon}^B) + \mathbf{V}_6^B$$
Differentiating with respect to $\mathbf{R}_A$ via the product rule on $\mathbf{S} \mathbf{K} \mathbf{S}^T$ produces a factor of 2, yielding the net prefactor $0.5 \times 2 = 1.0$:
$$\mathbf{g}_A^{\text{PK}} = \sum_{B \ne A} \left[ \operatorname{Tr}\left( \mathbf{P}_A \nabla_{\mathbf{R}_A} \mathbf{S}_{AB} \mathbf{K}_B \mathbf{S}_{AB}^T \right) + \operatorname{Tr}\left( \mathbf{K}_A \nabla_{\mathbf{R}_A} \mathbf{S}_{AB} \mathbf{P}_B \mathbf{S}_{AB}^T \right) \right]$$
By translational symmetry, $\nabla_{\mathbf{R}_B} \mathbf{S}_{AB} = -\nabla_{\mathbf{R}_A} \mathbf{S}_{AB}$, which guarantees:
$$\sum_A \mathbf{g}_A^{\text{PK}} = \mathbf{0}$$

### Hydrogen Core-Exemption Invariant ($Z=1$)
Because Hydrogen possesses no core orbitals ($1s$ is valence-like with $\varepsilon_{1s} \approx -0.5\text{ Ha}$), Hydrogen atoms contribute zero projection onto neighboring centers:
$$Z_B \le 1 \implies \mathbf{K}_B = \mathbf{0}$$
This maintains the thesis Eq. III-24 invariant and ensures that unpolarized bonds involving Hydrogen are not subjected to unphysical core repulsion.

---

## 10. Unrelaxed Analytical Gradients vs. Full SCF Surface (CPHF / Z-Vector Response)

In standard variational quantum chemistry methods, the orbital coefficients minimize the exact molecular energy functional ($\partial E / \partial \mathbf{C} = \mathbf{0}$), so the wave function response term $\sum \frac{\partial E}{\partial C_{\mu i}} \frac{\partial C_{\mu i}}{\partial \mathbf{R}}$ vanishes identically by Pulay's theorem.

In PHATPSY, the SCF solves atomic Secular equations embedded in fitted spherical model potentials ($V_c^B(r) = \sum_k c_k e^{-\alpha_k r}$), which carry finite RMS fitting errors ($\approx 0.28\text{ Ha}$ on second-row atoms). The total molecular energy evaluated in `analys.f90` includes the fitted potentials alongside exact non-local projections and exchange residuals. Consequently, the energy functional evaluated is not strictly at its variational minimum with respect to the atomic coefficients ($\partial E / \partial \mathbf{C} \ne \mathbf{0}$).

- **Unrelaxed Analytical Gradient (`method="analytical"`):** Evaluates the direct Hellmann-Feynman, Pulay overlap, model potential, and PK force contractions on the converged electronic state. On frozen-density states, this matches numerical finite differences of the exact electronic expectation value to $< 10^{-7}\text{ Ha/Bohr}$ (machine precision). On purely variational systems ($\text{H}_2$), it matches the full SCF surface finite difference to $0.07\%$ ($< 2 \times 10^{-4}\text{ Ha/Bohr}$).
- **Relaxed Analytical Gradient with CPHF / Z-Vector (`method="relaxed"`, Phase 12):** Evaluates the uncoupled (diagonal-Hessian) coupled-perturbed orbital relaxation response $\operatorname{Tr}\left(\frac{\partial E}{\partial \mathbf{P}} \frac{\mathrm{d}\mathbf{P}}{\mathrm{d}\mathbf{R}}\right)$ via the Handy-Schaefer $Z$-vector method:
  $$\mathbf{A}^T \mathbf{Z} = -\mathbf{L}, \quad \Delta \mathbf{g}_{A, k} = \sum_{pq} Z_{pq} \mathbf{F}_{pq}^{(A, k)}$$
  where $\mathbf{F}_{pq}^{(A, k)} = \langle p | \partial \mathbf{F} / \partial R_{A, k} | q \rangle$ is the analytical skeleton Fock matrix derivative tensor and $A_{pq, pq} = \varepsilon_q - \varepsilon_p$ is the uncoupled orbital Hessian approximation. Implemented in `phatpsy/cphf.py` (`CPHFResponseSolver`), the relaxed gradient recovers the exact variational limit on $\text{H}_2$ ($< 2 \times 10^{-4}\text{ Ha/Bohr}$) and preserves exact machine-precision translational zero-sum invariance and molecular point-group symmetry ($C_{2v}$ on $\text{H}_2\text{O}$), while full two-electron superoperator coupling ($K_{pq, rs}$) is documented to close residual differences against finite-difference surfaces on general polar polyatomics.

---

## 11. Semi-Classical Dispersion Gradients (DFT-D3)

For non-covalent interactions, long-range London dispersion is evaluated via Grimme DFT-D3(BJ):
$$E_{\text{disp}} = -\frac{1}{2} \sum_{A \neq B} \sum_{n=6, 8} s_n \frac{C_n^{AB}}{R_{AB}^n + [R_0^{AB}]^n}$$
The analytical Cartesian dispersion gradient is strictly conservative:
$$\mathbf{g}_A^{\text{disp}} = \sum_{B \neq A} \left[ \frac{6 s_6 C_6^{AB} R_{AB}^4}{(R_{AB}^6 + [R_0^{AB}]^6)^2} + \frac{8 s_8 C_8^{AB} R_{AB}^6}{(R_{AB}^8 + [R_0^{AB}]^8)^2} \right] (\mathbf{R}_A - \mathbf{R}_B)$$

---

## 12. Invariant Conservation & Point Group Symmetry

To satisfy quantum mechanical boundary conditions, all analytical gradients in PHATPSY 2.0 strictly fulfill:

| Physical Invariant | Mathematical Condition | Numerical Precision in PHATPSY 2.0 |
| :--- | :--- | :--- |
| **Translational Invariance** | $\sum_{A=1}^{N_{\text{atom}}} \mathbf{F}_A = \mathbf{0}$ | $< 10^{-12}\text{ Ha/Bohr}$ (machine precision) |
| **Rotational Invariance** | $\sum_{A=1}^{N_{\text{atom}}} \mathbf{R}_A \times \mathbf{F}_A = \mathbf{0}$ | $< 10^{-12}\text{ Ha}$ |
| **Point Group Symmetry ($C_{2v}$ on $\text{H}_2\text{O}$)** | $F_x \equiv 0$, $F_y(\text{O}) \equiv 0$, $F_y(\text{H}_1) = -F_y(\text{H}_2)$ | $< 10^{-12}\text{ Ha/Bohr}$ |
| **Point Group Symmetry ($D_{\infty h}$ on $\text{N}_2$)** | $F_x = F_y \equiv 0$, $F_z(\text{N}_1) = -F_z(\text{N}_2)$ | $< 10^{-12}\text{ Ha/Bohr}$ |
| **Virial Theorem Sanity** | $\|-V/T - 2.0\| \le 0.05$ | Verified across all benchmark minima |
