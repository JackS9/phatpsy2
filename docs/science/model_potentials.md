# Model Potentials: Theory and Constrained Fitting

> **Status:** DRAFT (Pending User / Author Review)  
> **Source:** Jack A. Smith, *A Projected Hamiltonian Approach to Polyatomic Systems* (1980), Chapter 4: *A Model Approximation to the Effective One-Electron Interatomic Coulomb Potential*.  
> **Implementation:** [`src/analys.f90`](https://github.com/JackS9/phatpsy2/blob/master/src/analys.f90) (Modern LAPACK `dgels`) | Legacy: `src/analys.f`

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

## 3. Modernized Fitting Implementation via LAPACK SVD (`DGELSD`)

In the legacy code (`src/analys.f`), the constrained linear system was solved using custom matrix inversion and heuristic pivots.

In **PHATPSY 2.0** ([`src/analys.f90`](https://github.com/JackS9/phatpsy2/blob/master/src/analys.f90)), the fitting is solved as a rank-deficient or ill-conditioned linear least-squares problem using standard LAPACK singular value decomposition (SVD):
```fortran
call DGELSD(nfit, nfit, 1, a_full, nfit, rhs, nfit, s_svd, 1.0d-10, &
            rank, work, lwork, iwork, info)
```
- **Stability:** Minimum-norm SVD solution avoids the severe numerical instability of near-singular overlap matrices for diffuse STO bases.
- **Analytical Cusp Constraints:** Enforces exact nuclear cusp ($Z_p(0) = Z - N_{\text{core}}$) and derivative slope ($(dZ_p/dr)_{r=0} = -2 Z (Z - N_{\text{core}})$).
- **Dynamic STO Exponent Optimization:** Screening exponents $\{\zeta_k^B\}$ are refined via gradient optimization, retiring fixed tables.
- **In-Memory Storage:** The fitting grid and matrices are allocated dynamically in heap memory (`mol_ws`), removing all scratch disk files.

---

## 4. Spherically Symmetric DFT Exchange Kernels

In addition to Hartree-Fock exchange fitting, PHATPSY 2.0 incorporates modern density functional exchange kernels in [`src/exchange_dft.f90`](https://github.com/JackS9/phatpsy2/blob/master/src/exchange_dft.f90) and [`phatpsy/exchange_dft.py`](https://github.com/JackS9/phatpsy2/blob/master/phatpsy/exchange_dft.py):
- **Local Slater Exchange:** $V_X^{\text{LDA}}(\rho) = -\alpha \frac{3}{2} \left(\frac{3}{\pi} \rho(r)\right)^{1/3}$
- **Becke 1988 (B88) Gradient Correction:** $F_x^{\text{B88}}(s) = 1 + \frac{\beta s^2}{1 + 6\beta s \sinh^{-1}(s)}$
- **Perdew-Burke-Ernzerhof (PBE) GGA:** $F_x^{\text{PBE}}(s) = 1 + \kappa - \frac{\kappa}{1 + \mu s^2 / \kappa}$

Radial integrals over STO basis functions are evaluated using Gauss-Legendre quadrature ($N=64$ nodes on $[0, \infty)$) with $< 10^{-11}\text{ Ha}$ analytical precision. Exposed via `ControlConfig(exchange_kernel="slater"|"becke88"|"pbe")`.

---

## 5. Opt-in Grid-Based Screening Fit (`use_grid_fit`)

!!! note "Status"
    Off by default. Implemented in `fit_screening_simplex` and `nnls_small` in `src/analys.f90`; exercised by `tests/test_grid_fit.py`. It changes the model potential, so energies differ from the default fit.

### 5.1 Motivation

The default fit determines all coefficients $\{c_k\}$ of Section 1 from the Coulomb expectation values of the atom's few orbitals. Measured on $\text{CO}$/DZ with the default exponents, the fit matrix is close to rank-deficient: after unit-diagonal scaling its four eigenvalues span $\sim 10^{-7}$ to $4$, and from the second SCF cycle (when the external-field term is switched on) its smallest eigenvalue is negative. The solved coefficients reach $|c_k| \approx 6$ with $\sum_k c_k$ fixed near $-N$, i.e. they cancel heavily, and small changes in the density swing the potential. Because the same fit is repeated every SCF cycle, this noise is fed back into the iteration.

### 5.2 Formulation

Write the electronic part of the model potential as a screening function:

$$V_{\text{el}}(r) = \frac{N}{r}\left[1 - \sum_{k} w_k\, e^{-\zeta_k r}\right], \qquad w_k \ge 0, \quad \sum_k w_k = 1$$

This is the form of Section 1 with $c_k = -N w_k$. The far field $N/r$ (charge conservation) and a finite potential at $r=0$ (the nuclear cusp sum rule) then hold exactly, and every weight lies in $[0, 1]$, so no cancellation is possible. The weights are fitted to the screening function $S(r) = 1 - r V_{\text{el}}(r)/N$ computed from the current density on a logarithmic radial grid (400 points, $10^{-4}$ to $40$ bohr):

1. The spherically averaged density comes from `eval_atomic_rho` (`src/exchange_dft.f90`).
2. $V_{\text{el}} = Q(r)/r + U(r)$ with $Q(r) = \int_0^r 4\pi r'^2\rho\, dr'$ and $U(r) = \int_r^\infty 4\pi r'\rho\, dr'$ (trapezoid rule in $\ln r$).
3. A weighted non-negative least-squares fit (Lawson-Hanson) with grid weights $h\, r\, e^{-r/10}$, a heavily weighted sum-to-one row, and a tiny ridge toward a uniform mixture so that the solution is unique when exponentials are nearly collinear. The weights are renormalized to sum to exactly 1.

The exponents $\{\zeta_k\}$ are unchanged (the element's existing table, scaled by $Z^{1/3}$); the choice of exponent set is separate from this change. The fit weighting and grid are first choices and have not been tuned.

### 5.3 What was and was not verified

- Verified by `tests/test_grid_fit.py`: weights in $[0,1]$, $\sum w_k = 1$ and $\sum_k c_k = -c_1$ to $10^{-10}$, electron-count conservation, honest convergence, and bit-for-bit reproducibility, for $\text{CO}$ with `atomic_smearing = 1.0e-3`.
- Observed but not asserted: with both options on, $\text{CO}$ converged to two different states ($\approx -112.50$ and $\approx -112.61\text{ Ha}$) depending on the damping history, and $\text{N}_2$ still shows a slow residual oscillation (see `ROADMAP.md`).

---

## 6. Opt-in Smeared Atomic Occupations (`atomic_smearing`)

!!! note "Status"
    Off by default (`atomic_smearing: 0`). Implemented in `smear_atomic_occupations` in `src/scf.f90`.

In the default atomic SCF, occupation numbers belong to orbital slots and are re-assigned after every iteration by maximum-overlap matching of the new eigenvectors with the previous ones (`MAXOVL`/`NEWORD` in `src/scf.f90` and `src/linalg.f90`). When orbitals are nearly degenerate or cross during the iteration, this assignment flips on eigensolver noise of order $10^{-14}$, and identical inputs then give different SCF trajectories from the second cycle onward.

With `atomic_smearing` $= w > 0$ (Hartree), each spin channel instead uses Fermi-Dirac occupations $f_i = [1 + e^{(\varepsilon_i - \mu)/w}]^{-1}$ with $\mu$ chosen by bisection so that the channel keeps its electron count. Occupations are then smooth functions of the eigenvalues, and the overlap-based reordering is skipped. This is a finite-width approximation to aufbau filling: it changes the atomic reference occupations by an amount that grows with $w$, so the width should be reported with any result that uses it.

---

## 7. Known Inconsistency: the `WGT1` Condition

Section 2 lists a Poisson total-charge condition weighted by `WGT1`. In the code, `WGT1` is read from the namelist and printed (`src/contrl.f90`, `src/state.f90`) but is not used in either the modern `src/analys.f90` or the legacy `legacy/analys.f` (where the corresponding terms are commented out). Total charge is enforced in the default fit only through the leading $1/r$ coefficient, and exactly in the grid fit of Section 5.

---

## 8. Selecting the Exponent Set

The exponents $\{\zeta_k\}$ of Section 1 are a per-element input. By default each element uses the built-in set (`TFEXP` in `src/state.f90`, scaled by $1.12956\,Z^{1/3}$, plus one extra exponential read from the deck). A deck can replace it with `elements.<symbol>.potential_exponents`, optionally with `potential_exponent_scale` (see `docs/user/yaml/overview.md`). The deck writer then emits `QTFPOT=F` and all model-potential terms for that element, and `RSCALE` when a scale is given (`phatpsy/schema.py`, `src/atomic.f90`). How to *choose* a set is not settled; the comparison criterion used so far (residual of the screening function against the Poisson potential of the atomic density) and its findings are recorded under the SCF convergence item in `ROADMAP.md`.
