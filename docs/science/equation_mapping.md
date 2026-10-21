# Equation & Routine Mapping: Dissertation to Code

> **Status:** DRAFT (Pending User / Author Review)  
> **Source:** Jack A. Smith, *A Projected Hamiltonian Approach to Polyatomic Systems* (1980).  
> **Primary LaTeX Reference:** [`PhD_Thesis/PHATPSY.tex`](../../PhD_Thesis/PHATPSY.tex)

This document provides a line-of-sight mapping connecting key mathematical derivations in the PhD dissertation to original legacy FORTRAN routines and their modernized Fortran 2018 / Python equivalents.

---

## 1. Chapter 2: One-Electron Operator & Atom-in-Molecule Subspace Partitioning

| Dissertation Eq. | Mathematical Expression | Legacy Routine | Modernized Code | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Eq. (2.5)** | $F \phi = \varepsilon \phi$ | `src/oneint.f` | `phatpsy_state_mod.f90` | Localized atomic spin-orbital secular equation |
| **Eq. (2.12)** | $\langle \phi_i^A \| \phi_j^B \rangle = 0$ | `src/scf.f` | `phatpsy_scf_mod.f90` | Interatomic orbital orthogonality condition |
| **Eq. (2.25)** | $P = \sum_{j \in B} \|\phi_j^B\rangle \langle \phi_j^B\|$ | `src/oneint.f` | `phatpsy_integrals_mod.f90` | Projection operator onto neighboring atom ($B$) subspace |
| **Eq. (2.34)** | $T_{\mu\nu} = -\frac{1}{2} \langle \chi_\mu \| \nabla^2 \| \chi_\nu \rangle$ | `src/oneint.f` | `src/oneint.f`, `gensab.f` | Kinetic energy matrix in STO basis |

---

## 2. Chapter 3: Energy-Dependent Pseudopotential for Atom-Atom Partitioning

| Dissertation Eq. | Mathematical Expression | Legacy Routine | Modernized Code | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Eq. (3.12)** | $V_{\text{PK}} = \sum_{j \in B} (\varepsilon - \varepsilon_j^B) \|\phi_j^B\rangle \langle \phi_j^B\|$ | `src/scf.f`, `fokmat.f` | `phatpsy_scf_mod.f90` | Interatomic Phillips-Kleinman Pauli exclusion pseudopotential |
| **Eq. (3.30)** | $M = 1 - \sum_k (1 - \lambda_k) \|\phi_k^B\rangle \langle \phi_k^B\|$ | `src/scf.f` | `phatpsy_scf_mod.f90` | Approximate pseudometric operator for energy dependence |
| **Eq. (3.45)** | $F C = \varepsilon M C$ | `src/scf.f`, `eispak.f` | `phatpsy_scf_mod.f90` | Generalized secular equation (solved via LAPACK `dsyev`) |

---

## 3. Chapter 4: Model Interatomic Coulomb Potential

| Dissertation Eq. | Mathematical Expression | Legacy Routine | Modernized Code | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Eq. (4.15)** | $V^B(r) = -Z_B^*/r + \sum_k c_k^B e^{-\zeta_k r}$ | `src/analys.f` | `phatpsy_analys_mod.f90` | Spherical exponential model potential expansion |
| **Eq. (4.28)** | $\sum_k c_k^B = -Z_B + Z_B^*$ | `src/analys.f` | `phatpsy_analys_mod.f90` | Nuclear cusp constraint at $r=0$ (`WGT0`) |
| **Eq. (4.35)** | $\int \nabla^2 V^B d^3r = 4\pi Q_B$ | `src/analys.f` | `phatpsy_analys_mod.f90` | Poisson total electronic charge constraint (`WGT1`) |
| **Eq. (4.48)** | $\min \sum_i w_i (V_{\text{exact}}(r_i) - V_{\text{model}}(r_i))^2$ | `src/analys.f` (inversion) | `phatpsy_analys_mod.f90` (LAPACK `dgels`) | Constrained weighted least-squares fitting |

---

## 4. Chapter 5: Nonlocal Model Exchange Potential

| Dissertation Eq. | Mathematical Expression | Legacy Routine | Modernized Code | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Eq. (5.10)** | $K_{\mu\nu} = \langle \chi_\mu \| V_X \| \chi_\nu \rangle$ | `src/twoint.f`, `fokmat.f` | `phatpsy_integrals_mod.f90`, `phatpsy_scf_mod.f90` | Interatomic exchange potential integrals |
| **Eq. (5.24)** | $P_{X} = \sum_{k} \alpha_k \|\phi_k\rangle \langle \phi_k\|$ | `src/fokmat.f` | `phatpsy_scf_mod.f90` | Projected local density exchange approximation |

---

## 5. Chapter 6: Energy-Weighted Maximum Overlap & Propagator

| Dissertation Eq. | Mathematical Expression | Legacy Routine | Modernized Code | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Eq. (6.14)** | $G(E) = (E \cdot S - H)^{-1}$ | `src/ewmo.f` | `phatpsy_ewmo_mod.f90`, `phatpsy/ewmo.py` | One-electron Green's function / Propagator |
| **Eq. (6.28)** | $\rho(E) = -\frac{1}{\pi} \text{Im} \operatorname{Tr} [G(E^+)]$ | `src/ewmo.f` | `phatpsy_ewmo_mod.f90` | Spectral density of states decomposition |
| **Eq. (6.42)** | $S^{-1/2} H S^{-1/2} = U \Lambda U^\dagger$ | `src/lowdin.f`, `ewmo.f` | `phatpsy_ewmo_mod.f90` | Lowdin symmetric orthogonalization |

---

## 6. Chapters 7 & 8: Grand-Canonical Hartree-Fock & Charge Equilibration

| Dissertation Eq. | Mathematical Expression | Legacy Routine | Modernized Code | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Eq. (7.18)** | $n_i = [1 + \exp((\varepsilon_i - \mu)/\Theta)]^{-1}$ | `src/contrl.f`, `scf.f` | `phatpsy_contrl_mod.f90` | Fermi-Dirac thermal occupation numbers |
| **Eq. (8.12)** | $\Delta q_A^{(m+1)} = (1 - \alpha) \Delta q_A^{(m)} + \alpha \Delta q_A^{\text{new}}$ | `src/contrl.f` | `phatpsy_contrl_mod.f90` (`CDAMP`, `VDAMP`) | Relative charge transfer (RCT) damping |
| **Eq. (8.22)** | $\langle V \rangle / \langle T \rangle \to -2.0$ | `src/ewmo.f` | `phatpsy_ewmo_mod.f90`, `phatpsy/driver.py` | Virial theorem evaluation check |

---

## 7. Spatial Symmetry, Equivalent Atoms & Special Centers

| Operational Concept | Mathematical Operation | Legacy Routine | Modernized Code | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Equivalent Atom Spatial Mapping** | $\mathbf{R}_{\text{eq}} = \mathcal{R}_{\text{sym}} \mathbf{R}_{\text{gen}}$ | `src/symxyz.f` | `src/symxyz.f`, `phatpsy_atomic_mod.f90` | Generates XYZ coordinates for `NEQAT` atoms using `ISYM`/`IXYZ` |
| **Orbital Symmetry Transformation** | $C_{\mu i}^{(\text{eq})} = D(R) C_{\mu i}^{(\text{gen})}$ | `src/symop.f` | `src/symop.f`, `phatpsy_atomic_mod.f90` | Transforms STO coefficients and model potential via Wigner $D$-matrices |
| **Spin-Flipping on Symmetry** | $\alpha \leftrightarrow \beta$ | `src/flip.f` | `src/flip.f`, `phatpsy_atomic_mod.f90` | Exchanges spin manifolds when `QFLIP=T` (LVB singlet generation) |
| **Fixed (Frozen) Environment Atoms** | $\delta V_{\text{fixed}} = 0, \delta C_{\text{fixed}} = 0$ | `src/atomic.f`, `scf.f`, `analys.f` | `phatpsy_scf_mod.f90`, `phatpsy_analys_mod.f90` | Freezes orbitals and potential (`QFIXED=T`); bypasses SCF iterations |
| **Virtual Atoms (Coordination Completion)** | $V_{\text{ext}} \neq 0, \text{excluded at EWMO}$ | `src/atomic.f`, `bloop.f` | `phatpsy_atomic_mod.f90`, `phatpsy/schema.py` | Completes local coordination environment during atomic stage; excluded from molecular EWMO calculation (`QVIRTL=T`) |
