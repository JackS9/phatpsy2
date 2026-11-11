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

## 5. Default Grid-Based Screening Fit (`use_grid_fit`)

!!! note "Status & Author Approval"
    Enabled by default (`use_grid_fit: true`, `QGRDFT = .true.`, approved 2026-11-09). Implemented in `fit_screening_simplex` and `nnls_small` in `src/analys.f90`; exercised by `tests/test_grid_fit.py` and differentiated analytically in `phatpsy/cphf.py`. Passing `use_grid_fit: false` reverts to the orbital-expectation linear fit of Section 2.

### 5.1 Motivation

The orbital-expectation fit of Section 2 (`use_grid_fit: false`) determines all coefficients $\{c_k\}$ of Section 1 from the Coulomb expectation values of the atom's few orbitals. Measured on $\text{CO}$/DZ with the default exponents, that fit matrix is close to rank-deficient: after unit-diagonal scaling its four eigenvalues span $\sim 10^{-7}$ to $4$, and from the second SCF cycle (when the external-field term is switched on) its smallest eigenvalue is negative. The solved coefficients reach $|c_k| \approx 6$ with $\sum_k c_k$ fixed near $-N$, i.e. they cancel heavily, and small changes in the density swing the potential. Because the same fit is repeated every SCF cycle, this noise is fed back into the iteration.

By fitting the screening function directly on a radial grid under non-negativity and sum-to-one constraints (`use_grid_fit: true`), both charge conservation and the nuclear cusp condition hold by construction and coefficient cancellation is eliminated.

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

---

## 9. Polarized Basis (`DZP`) Potential Energy Surfaces & Phillips-Kleinman Regularization

!!! note "Diagnostic Script & Verification"
    Executable diagnostic script: `tools/diagnose_dzp_pes.py`. Exercised by `tests/test_basis_polarized.py` (`test_dzp_pes_and_geometry_optimization_bound_minima`) and `tests/test_cphf.py` (`test_cphf_dzp_polarization_separation_and_gradient_parity`).

### 9.1 Phase 1 Root-Cause Diagnosis

When polarization shells ($3d$ on $\text{B}\text{--}\text{Ne}$, $2p$ on $\text{H}\text{--}\text{He}$) are added to form the default `DZP` basis, three coupled effects occur if the two-center operators and Phillips-Kleinman (PK) projectors are applied uniformly across the full $M_{\text{STO}}$ basis:

1. **Two-Center Polarization Delocalization (`src/bloop.f90`, `src/scf.f90`, `phatpsy/cphf.py`):**
   Unoccupied polarization functions ($l > l_{\max,\text{val}}$, i.e. $3d$ on first-row atoms and $2p$ on $\text{H}$) have diffuse radial tails that overlap strongly with neighboring cores. Allowing two-center nuclear attraction (`zpot2a`), model-potential screening (`cpot2a`), and two-center exchange/Coulomb residuals (`pkpot(:, 6)`) to couple directly into the polarization shell ($i > n_{\text{act,sto}}$) causes unphysical charge delocalization into the neighbor's core. Restricting two-center attraction/screening and `pkpot(:, 6)` to the active core+valence STO subspace ($1 \dots n_{\text{act,sto}}$, while retaining full $M_{\text{STO}} \times M_{\text{STO}}$ intra-atomic Fock diagonalization in `solve_orthonormal_fock`) preserves physical intra-atomic $s\text{--}p\text{--}d$ polarization while eliminating inter-atomic basis-set superposition over-attraction. For Hydrogen ($Z \le 1.5$), $2p$ polarization two-center attraction is retained only in homonuclear $\text{H}_2$ ($Z_A \le 1.5$ and $Z_B \le 1.5$), where it provides the $\sigma_g$ polarization lowering required for $R_e(\text{H}_2) \approx 1.401\text{ Bohr}$.
2. **Phillips-Kleinman Pseudometric Regularization (`src/scf.f90:regularize_pk_projector`, `phatpsy/cphf.py`):**
   In multi-bonded diatomics ($\text{N}_2$, $\text{CO}$) at short internuclear distances ($R < 2.2\text{ Bohr}$), the raw environment overlap projector $P_{1,A}^{\text{raw}} = \sum_{B \ne A} S_{AB} D_{4,B} S_{BA}$ develops generalized eigenvalues $\lambda_{\max}(S_A^{-1/2} P_{1,A}^{\text{raw}} S_A^{-1/2}) \in [0.60, 0.96]$. Because the PK pseudometric $\tilde{S}_A = S_A - P_{1,A}$ enters as $\tilde{S}_A^{-1/2}$, an unregularized eigenvalue approaching $1$ amplifies short-range Pauli repulsion as $(1 - \lambda)^{-1}$ and creates a false barrier or repulsive wall. Smooth $C^1$ cubic Hermite capping at $\lambda_{\max} = 1 - \delta_{\text{reg}} = 0.58$ ($\delta_{\text{reg}} = 0.42$ when `QNWVEC` is active) bounds the pseudometric condition number while preserving continuous analytical CPHF derivatives.
3. **Seeded Hydrogen PK Rules (`src/bloop.f90`, `src/scf.f90`, `phatpsy/cphf.py`, `phatpsy/schema.py`):**
   - *Heteronuclear $\text{H}\text{--}\text{X}$ exemption — PHATPSY 2.0 implementation rule (motivated by the PK construction in Thesis Chapter III):* Hydrogen has no inner core shell, and in a covalent $\text{X}\text{--}\text{H}$ bond ($\text{O}\text{--}\text{H}$, $\text{N}\text{--}\text{H}$, $\text{C}\text{--}\text{H}$) the localized $\text{H}$ $1s$ orbital overlaps the heavy-atom lone pairs ($S \sim 0.4\text{--}0.5$). Applying PK projection between $\text{H}$ and $\text{X}$ unphysically projects out the $\text{X}\text{--}\text{H}$ $\sigma$ bond, so heteronuclear $\text{H}\text{--}\text{X}$ PK transfer is zero.
   - *Homonuclear $\text{H}\text{--}\text{H}$ Pauli repulsion under `QNWVEC` (empirically calibrated $w_{\text{HH}} = 0.40$):* When explicit spin-paired `valence_config` seeds are used in bent polyatomics such as $\text{H}_2\text{O}$ ($\text{O}$ $\alpha$ bonded to $\text{H}_1(\beta)$ and $\text{H}_2(\beta)$), the two non-bonded Hydrogen atoms carry parallel spins ($\beta, \beta$). Without homonuclear $\text{H}\text{--}\text{H}$ PK projection ($w_{\text{HH}} = 0$), $\text{H}_1(\beta)$ and $\text{H}_2(\beta)$ experience zero Pauli repulsion and the $\text{H}\text{--}\text{O}\text{--}\text{H}$ bond angle collapses toward $\le 90^\circ$. Scaling homonuclear $\text{H}\text{--}\text{H}$ PK transfer by $w_{\text{HH}} = 0.40$ when `QNWVEC` is active is an **empirical calibration** chosen to restore the $\text{H}_2\text{O}$ equilibrium bond angle ($\theta_e = 105.45^\circ$ in `DZP`); accordingly, the $\text{H}_2\text{O}$ bond angle in Section 9.2 is a calibration target rather than an independent validation. At $(R_{\text{OH}} = 1.810\text{ Bohr}, \theta = 104.48^\circ)$, the non-bonded $\text{H}\cdots\text{H}$ overlap is $S_{\text{HH}} \approx 0.298$ and the pseudometric eigenvalue $\lambda_{\max}(S_{\text{H}}^{-1/2} P_{1,\text{H}} S_{\text{H}}^{-1/2})$ scales linearly as $0.2950\,w_{\text{HH}}$ ($0.0885$ at $w_{\text{HH}} = 0.30$, $0.1180$ at $w_{\text{HH}} = 0.40$, $0.1475$ at $w_{\text{HH}} = 0.50$; `tools/diagnose_dzp_pes.py` Section 3). Full SCF re-optimization across $w_{\text{HH}} \ne 0.40$ requires recompiling `src/bloop.f90:312` and is not swept at runtime. *(Flagged for Chapter III author adjudication.)*

### 9.2 Verified `DZP` Benchmark Summary and Open Items

All rows below are reproduced by `PYTHONPATH=. .venv/bin/python tools/diagnose_dzp_pes.py` and match the executed benchmark notebooks (`notebooks/01_N2_dinitrogen.ipynb`, `notebooks/02_CO_carbon_monoxide.ipynb`, `notebooks/03_H2O_water.ipynb`):

| Molecule | Basis | $R_e$ (Bohr) | Expt $R_e$ (Bohr) | $E(R_e)$ (Ha) | $-V/T$ | $\lambda_{\max}(P_1)$ at $R_{\text{disp}}$ | $\max \|g_{\text{CPHF}} - g_{\text{FD}}\|$ (Ha/Bohr) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| $\text{H}_2$ | `DZP` | $1.6034$ | $1.401$ | $-1.072591$ | $2.0640$ | $0.3551$ ($R=1.45$) | $4.72 \times 10^{-7}$ |
| $\text{N}_2$ | `DZP` | $2.1129$ | $2.0744$ | $-109.530037$ | $1.9706$ | $0.7703$ ($R=2.15$) | $4.74 \times 10^{-6}$ |
| $\text{CO}$ | `DZP` | $2.4495$ | $2.1322$ | $-112.652935$ | $1.9857$ | $0.7093$ ($R=2.25$) | $1.69 \times 10^{-6}$ |
| $\text{H}_2\text{O}$ | `DZP` | $1.7685$ ($\theta_e = 105.45^\circ$*) | $1.809$ ($104.5^\circ$) | $-76.336507$ | $2.0076$ | $0.1123$ ($R=1.85$) | $3.88 \times 10^{-6}$ |

*\*Note:* $\theta_e(\text{H}_2\text{O}) = 105.45^\circ$ is a calibration target of the empirical homonuclear $\text{H}\text{--}\text{H}$ PK weight $w_{\text{HH}} = 0.40$ (Section 9.1 item 3), not an independent validation.

- **Provenance & Settings (`tools/diagnose_dzp_pes.py`):**
  - **CPHF Gradient Parity (Section 1 of `tools/diagnose_dzp_pes.py`, `tests/test_cphf.py::test_cphf_dzp_polarization_separation_and_gradient_parity`):** Evaluated with `convergence=1e-10`, `max_cycles=200`, `use_grid_fit=True`, central finite-difference step `step_size=1e-4` Bohr at displaced geometries $\text{H}_2$ ($R=1.45\text{ Bohr}$), $\text{N}_2$ ($R=2.15\text{ Bohr}$), $\text{CO}$ ($R=2.25\text{ Bohr}$), and $\text{H}_2\text{O}$ ($R_{\text{OH}}=1.85\text{ Bohr}, \theta=104.5^\circ$).
  - **BFGS Geometry Optimization (Section 2 of `tools/diagnose_dzp_pes.py`, matching `notebooks/01_N2_dinitrogen.ipynb`, `02_CO_carbon_monoxide.ipynb`, `03_H2O_water.ipynb`):** Evaluated with `optimize_geometry(deck, max_steps=20, grad_tol=1e-3, finite_diff_step=0.005, warm_restart=True)` using `convergence=1e-6`, `max_cycles=200`:
    - $\text{H}_2$: `build_diatomic("H", "H", bond_length=1.401, basis="DZP", hybridization="s", spins=["up", "down"], valence_configs=[...], net_spin=0, physical_net_spin=0)` (converges in 3 BFGS steps to $R_e = 1.6034\text{ Bohr}$, $E = -1.072591\text{ Ha}$).
    - $\text{N}_2$: `build_diatomic("N", "N", bond_length=2.0744, basis="DZP", hybridization="sp", spins=["up", "down"], valence_configs=[...], net_spin=0, physical_net_spin=0)` (converges in 3 BFGS steps to $R_e = 2.1129\text{ Bohr}$, $E = -109.530037\text{ Ha}$, $-V/T = 1.9706$; 1D 7-point parabolic scan gives $R_e = 2.1140\text{ Bohr}$).
    - $\text{CO}$: `build_diatomic("C", "O", bond_length=2.132, basis="DZP", hybridization="sp", spins=["up", "down"], valence_configs=[...], net_spin=0, physical_net_spin=0)` with dative-$\pi$ fractional occupation sharing (converges in 6 BFGS steps to $R_e = 2.4495\text{ Bohr}$, $E = -112.652935\text{ Ha}$, $-V/T = 1.9857$; at $R_{\text{expt}} = 2.132\text{ Bohr}$, $E = -112.576316\text{ Ha}$, $\mu = 0.0311\text{ D}$).
    - $\text{H}_2\text{O}$: `build_molecule([("O", ...), ("H", ...), ("H", ...)], basis="DZP", spins=["up", "down", "down"], hybridization=["sp3", "s", "s"], valence_configs=[...], net_spin=-1, physical_net_spin=0)` starting from $R_0 = 1.810\text{ Bohr}, \theta_0 = 104.48^\circ$ ($E_0 = -76.335319\text{ Ha}$, $-V/T = 2.0078$, $\mu = 0.876\text{ D}$; converges in 3 BFGS steps to $R_e = 1.7685\text{ Bohr}, \theta_e = 105.45^\circ$, $E = -76.336507\text{ Ha}$, $-V/T = 2.0076$).
- **Known Open Item — Seeded `valence_config` $\text{H}_2\text{O}$ (`DZP`) Dipole Moment:**
  On unseeded default-orbital $\text{H}_2\text{O}$ (`DZP`), the molecular dipole moment is $1.633\text{ D}$ (`tests/test_basis_polarized.py::test_h2o_dzp_convergence_and_dipole`, within $[1.0, 4.50]\text{ D}$ vs experimental $1.855\text{ D}$). On seeded `valence_configs` $\text{H}_2\text{O}$ (`DZP`), restricting $\text{H}$ $2p$ two-center nuclear/Coulomb attraction toward Oxygen (`znucb > 1.5d0`, required to prevent $\text{O}\text{--}\text{H}$ bond-length collapse to $1.55\text{ Bohr}$) reduces the intra-atomic $\text{H}$ $1s\text{--}2p$ polarization dipole contribution, yielding $\mu = 0.876\text{ D}$. Reconciling $\text{H}$ $2p$ two-center polarization with the $\text{O}\text{--}\text{H}$ short-range repulsion in seeded `valence_config` $\text{H}_2\text{O}$ is tracked under "Open Model Items" in `ROADMAP.md` (Part II) and the Quantitative Parity Summary table.
- **Known Open Item — Homonuclear $\text{H}_2$ (`ncore = 0`) Equilibrium Bond Length ($R_e = 1.6034\text{ Bohr}$ in `DZP` vs $1.401\text{ Bohr}$ Expt):**
  Because Hydrogen has no inner core shell (`ncore = 0`) and each atom in spin-paired singlet $\text{H}_2$ ($\text{H}_1^\alpha, \text{H}_2^\beta$) carries a single occupied electron in opposite spin channels, three structural properties of the embedded-atom expansion govern the $\text{H}_2$ potential energy curve:
  1. **Zero Occupied-Channel PK & Two-Center Exchange Projection:** Both the Phillips-Kleinman pseudopotential projectors (`pkpot(:, 4:5, isp)`) and the two-center exchange / Coulomb-fit residual projector (`pkpot(:, 6, isp)`) are diagonal in spin (`src/analys.f90:530-565`, `src/bloop.f90:330-358`). For $\text{H}_1^\alpha$ (`occ_alpha = [1, 0]`) interacting with $\text{H}_2^\beta$ (`occ_alpha = [0, 0]`), the neighbor's $\alpha$-spin density is identically zero ($P_{1,\text{H}_1}^\alpha = 0$, $P_{2,\text{H}_1}^\alpha = 0$, $X_{2,\text{H}_1}^\alpha = 0$). The $\lambda_{\max}(P_1) = 0.3551$ entry in the table above resides in the *unoccupied* minority-spin ($\beta$) channel of $\text{H}_1$, so the empirical homonuclear $\text{H}\text{--}\text{H}$ PK scale $w_{\text{HH}} = 0.40$ has zero effect on the occupied orbitals or total energy of singlet $\text{H}_2$.
  2. **Single-Center Hartree-Product Energy Functional (Missing Two-Center Heitler-London Resonance $J_{\text{HL}}(R)$):** In the embedded-atom ansatz (Thesis Chapter II, p. 305; Chapter VIII, p. 936), each occupied orbital $\phi_A^\alpha$ is expanded purely in the single-center STO basis on atom $A$, and intra-atomic self-interaction cancels exactly for a one-electron shell ($J^{AA} - K^{AA} = 0$). Consequently, the total energy in `src/analys.f90` reduces to the single-center Hartree product expectation value $E_{\text{PHATPSY}}(R) = 2\langle\phi_A|-\tfrac{1}{2}\nabla^2 - r_A^{-1}|\phi_A\rangle + 2\langle\phi_A|-r_B^{-1}|\phi_A\rangle + \langle\phi_A|V_B^{\text{model}}(r_B)|\phi_A\rangle + R^{-1} = 2E_A^{(1)} + Q(R)$. In true covalent $\text{H}_2$, $\sim 85\%$ of the binding energy at $R_e = 1.401\text{ Bohr}$ comes from the two-center Heitler-London interference/exchange integral $J_{\text{HL}}(R) \approx -0.163\text{ Ha}$ (electron kinetic-energy lowering in the internuclear region), whereas the classical Sugiura electrostatic penetration integral $Q(R)$ alone has a shallow minimum near $R \approx 1.97\text{ Bohr}$ ($E = -1.033\text{ Ha}$ in `SZ`/`DZ`, $D_e \approx 0.033\text{ Ha}$). Adding the $2p_z(\zeta=1.25)$ polarization shell in `DZP` permits one-center dipolar distortion toward the neighbor ($\phi_A = 0.990|1s_A\rangle + 0.138|2p_{z,A}\rangle$), recovering $\sim 0.068\text{ Ha}$ of electrostatic stabilization and contracting $R_e$ from $1.97\text{ Bohr} \to 1.6034\text{ Bohr}$ ($E = -1.072591\text{ Ha}$), but a single-center $l \le 1$ expansion on $A$ cannot form the nuclear cusp at $B$ or capture the two-center interference term $J_{\text{HL}}(R)$.
  3. **Single-Zeta $1s(\zeta=1.0)$ Fallback in Hydrogen `"DZ"` / `"DZP"` (Missing Radial Contraction):** Because the Clementi-Roetti (1974) double-zeta tables begin at Helium ($Z = 2$), `get_element_basis("H", "DZ")` (`phatpsy/basis.py:581`) falls back to `get_thakkar_basis("H")`, which contains only a single $1s$ STO with $\zeta = 1.0$. Thus `"SZ"` and `"DZ"` are identical for Hydrogen (`msto = 1`), and `"DZP"` is $1s(\zeta=1.0) + 2p(\zeta=1.25)$ (`SZP`) with no split radial $1s, 1s'$ freedom. Without a tight $1s'$ component, Hydrogen's $1s$ shell cannot radially contract toward the Wang (1928) optimal exponent $\zeta \approx 1.17$ at $R_e = 1.401\text{ Bohr}$: the kinetic energy remains frozen near $T \approx 1.008\text{--}1.011\text{ Ha}$ across $R \in [1.40, 1.60]\text{ Bohr}$, yielding $-V/T = 2.0640$ at $R_e = 1.6034\text{ Bohr}$. Switching to the split-valence `ADF-TZP` basis ($1s$ exponents $0.69, 0.92, 1.58$ plus $2p(\zeta=1.25)$) restores radial contraction ($T = 1.103\text{ Ha}$ at $R=1.401\text{ Bohr}$, $-V/T = 2.0036$ at minimum, $R_e = 1.576\text{ Bohr}$). Tracked under "Open Model Items" in `ROADMAP.md`.


