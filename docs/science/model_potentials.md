# Model Potentials: Theory and Constrained Fitting

> **Status:** DRAFT (Pending User / Author Review)  
> **Source:** Jack A. Smith, *A Projected Hamiltonian Approach to Polyatomic Systems* (1980), Chapter 4: *A Model Approximation to the Effective One-Electron Interatomic Coulomb Potential*.  
> **Implementation:** [`src/phatpsy_analys_mod.f90`](../../src/phatpsy_analys_mod.f90) (Modern LAPACK `dgels`) | Legacy: `src/analys.f`

---

## 1. Physical Principle

In all-electron polyatomic calculations, evaluating the interatomic Coulomb potential involves computing two-center charge distribution potentials over non-spherical overlap densities.

In the **Projected Hamiltonian Approach**, the electrostatic potential field produced by an atomic center $B$ on electrons centered around another atom $A$ is approximated by a **spherically symmetric model potential**:
$$V_B^{\text{model}}(r_B) = -\frac{Z_B^*}{r_B} + \sum_{k=1}^{K_B} c_k^B e^{-\zeta_k^B r_B}$$
where:
- $r_B = |\mathbf{r} - \mathbf{R}_B|$ is the radial distance from nucleus $B$.
- $Z_B^*$ is the effective nuclear charge (nuclear charge minus core electrons if core-projected).
- $\{\zeta_k^B\}$ are fixed radial screening exponents.
- $\{c_k^B\}$ are expansion coefficients determined dynamically by linear least-squares fitting at every SCF cycle.

---

## 2. Boundary Constraints & Weighting Scheme

The fitting must reproduce both short-range nuclear physics and long-range electrostatic asymptotics. PHATPSY enforces these through five weighted conditions (`WGT0` through `WGT2` in namelist `&CTLINP`):

1. **Nuclear Cusp Condition (`WGT0`):**
   As $r_B \to 0$, the potential must approach $-Z_B/r_B$:
   $$\lim_{r_B \to 0} \left( V_B^{\text{model}}(r_B) + \frac{Z_B^*}{r_B} \right) = \sum_{k=1}^{K_B} c_k^B = -Z_B + Z_B^*$$
   This ensures the nuclear potential cusp is exact at the nucleus.

2. **Poisson Total Charge Conservation (`WGT1`):**
   Integrating the Laplacian of the model potential over all space must yield the exact net electronic charge $Q_B$ on center $B$:
   $$\int \nabla^2 V_B^{\text{model}} d^3r = 4\pi \sum_{k=1}^{K_B} \frac{c_k^B}{(\zeta_k^B)^2} = 4\pi Q_B$$

3. **Core Shell Potential Matching (`WGTC`):**
   Matches the model potential against the exact Hartree-Fock potential within the core orbital radii.

4. **Valence Shell Potential Matching (`WGTV`):**
   Matches the potential in the bonding/valence region where chemical overlap occurs.

5. **External Field Matching (`WGT2`):**
   Matches the asymptotic decaying potential in the outer tail of the atomic density.

---

## 3. Modernized Fitting Implementation via LAPACK

In the legacy code (`src/analys.f`), the constrained linear system was solved using custom matrix inversion and heuristic pivots.

In **PHATPSY 2.0** ([`src/phatpsy_analys_mod.f90`](../../src/phatpsy_analys_mod.f90)), the fitting is solved as an overdetermined linear least-squares problem with equality/penalty constraints using standard LAPACK:
```fortran
call dgels('N', m_rows, n_cols, 1, fit_matrix, lda, rhs_vector, ldb, work, lwork, info)
```
- **Stability:** QR factorization with Householder reflections avoids the numerical instability of forming the normal equations ($A^T A$).
- **In-Memory Storage:** The fitting grid and matrices are allocated dynamically in heap memory (`mol_ws`), removing all scratch disk files.
