# Scientific Methodology Overview: The Projected Hamiltonian Approach

> **Status:** DRAFT (Pending User / Author Review)  
> **Source:** Jack A. Smith, *A Projected Hamiltonian Approach to Polyatomic Systems*, Ph.D. Dissertation, Department of Chemistry & Quantum Theory Project, University of Florida (1980).  
> **Reference Text:** [`PhD_Thesis/PHATPSY.tex`](../../PhD_Thesis/PHATPSY.tex) | Archival Scanned: [`PhD_Thesis/projectedhamilto00smit.pdf`](../../PhD_Thesis/projectedhamilto00smit.pdf)

---

## 1. Scientific Motivation and Scope

Non-empirical electronic structure calculations on large polyatomic molecules, chemisorption systems, and transition-metal clusters (e.g., $CO$ on $Ni(100)$, $Ag_{13}$, $Ni_{16}$) are conventionally hindered by:
1. The enormous computational cost of four-center two-electron integrals $\langle \mu \nu | \lambda \sigma \rangle$.
2. The presence of tightly bound, chemically inert atomic core electrons that dominate total electronic energy but contribute little to chemical bonding.

**PHATPSY** (Projected Hamiltonian Approach to Polyatomic Systems) implements a non-empirical self-consistent field (SCF) methodology that circumvents these bottlenecks through:
- **Atom-in-Molecule Subspace Partitioning:** Partitioning the molecular Hilbert space into local atomic subspaces centered on individual nuclei ($\mathcal{H} = \mathcal{H}_A \oplus \mathcal{H}_B \oplus \dots$).
- **Interatomic Phillips-Kleinman Pseudopotentials:** Enforcing orthogonality and Pauli exclusion between atomic orbitals on neighboring atoms (atom-atom partitioning) using generalized **Phillips-Kleinman pseudopotentials** and **pseudometrics**.
- **Model Interatomic Potentials:** Replacing costly multi-center electron-electron integrals with rigorous, self-consistently fitted local spherical model potentials.
- **Slater-Type Orbital (STO) Basis Functions:** Employing analytical exponential basis functions rather than contracted Gaussians for accurate cusp behavior and long-range asymptotic decay.

---

## 2. Atom-in-Molecule Subspace Partitioning (Chapter 2)

Rather than forcing molecular orbitals to be symmetry-adapted canonical functions delocalized over the entire molecule, PHATPSY formulates the problem using **localized atomic orbitals** centered on individual atomic sites:
$$\mathcal{H} = \mathcal{H}_A \oplus \mathcal{H}_B \oplus \mathcal{H}_{B'} \dots$$
When solving for the electronic structure on a chosen atomic center $A$, all remaining atoms in the system ($B, B', \dots$) are treated as its chemical environment.

To ensure that spin-orbitals on center $A$ and center $B$ are simultaneous eigenfunctions of the total effective one-electron operator $F$, they must be mutually non-interacting through $F$:
$$\langle \phi_i^A | F | \phi_j^B \rangle = 0$$
A sufficient condition for this non-interaction is that the orbitals on different atomic centers be kept **strictly orthogonal**:
$$\langle \phi_i^A | \phi_j^B \rangle = 0$$

Let $P$ represent the projector onto the orbital subspace of all other atoms in the environment ($B, B', \dots$):
$$P = \sum_{j \in \{B, B', \dots\}} |\phi_j^B\rangle \langle \phi_j^B|$$
with the complementary projector onto atom $A$'s localized subspace:
$$Q = 1 - P$$

The true localized orbital $|\psi_i^A\rangle$ is related to a smooth pseudowavefunction $|\tilde{\psi}_i^A\rangle$ via:
$$|\psi_i^A\rangle = Q |\tilde{\psi}_i^A\rangle = (1 - P) |\tilde{\psi}_i^A\rangle$$

---

## 3. The Phillips-Kleinman Pseudopotential for Atom-Atom Partitioning (Chapter 3)

The key conceptual insight in PHATPSY is utilizing the **Phillips-Kleinman pseudopotential formulation for atom-atom partitioning** rather than traditional core-valence separation. 

Substituting the projected atomic orbital $(1 - P)|\tilde{\psi}_i^A\rangle$ into the secular equation yields:
$$(1 - P) (F - \varepsilon_i) (1 - P) |\tilde{\psi}_i^A\rangle = 0$$
which can be expressed as a modified secular equation containing an interatomic repulsive pseudopotential:
$$(F + V_{\text{PK}} - \varepsilon_i) |\tilde{\psi}_i^A\rangle = 0$$
where the Phillips-Kleinman operator projects out the states of neighboring atoms:
$$V_{\text{PK}} = \sum_{j \in \{B\}} (\varepsilon_i - \varepsilon_j^B) |\phi_j^B\rangle \langle \phi_j^B|$$
Physically, $V_{\text{PK}}$ acts as an effective interatomic repulsive potential that enforces the **Pauli Exclusion Principle** between interacting atoms, preventing electrons on center $A$ from collapsing into the occupied orbital space of center $B$.

### Energy-Dependent $V_{\text{PK}}(E)$ & Theoretical Unit Scaling (`PKSCAL = 1.0`):
Unlike static local potentials, $V_{\text{PK}}(\varepsilon_i)$ depends explicitly on the eigenvalue $\varepsilon_i$ of the state being sought. In legacy PHATPSY, static approximations caused severe core over-repulsion in high-coordination systems, necessitating arbitrary empirical dampening ($0.40$ or $0.80$).

In **PHATPSY 2.0**:
1. **Theoretical Lock `PKSCAL = 1.0`:** The arbitrary empirical scaling parameter has been eliminated.
2. **Dynamic Valence Reference Projection:** The operator is evaluated using the valence reference energy $E_{\text{ref}}$ (e.g. `PKE = -0.22 Ha` for transition metal d-bands), dynamically projecting out core states while preserving valence bonding.
3. **Symmetrized Operator:** Evaluated as a symmetric matrix in [`src/scf.f90`](file:///Users/jacksmith/phatpsy2/src/scf.f90), preserving exact Hermiticity, exact total energy conservation, and physical virial ratios ($-V/T \approx 2.0$). This completely stabilizes 12-coordinated bulk metals ($\text{Ni}_{13}$ FCC cluster) under pure theory `PKSCAL = 1.0`.

---

## 4. Local Model Potentials & Modern Exchange Kernels (Chapters 4 & 5)

To eliminate two-electron multi-center integrals, the Coulomb potential generated by a neighboring atomic charge distribution on center $B$ is represented as a fitted spherical model potential:
$$V_{\text{Coulomb}}^B(r_B) = -\frac{Z_B^*}{r_B} + \sum_{k} c_k^B e^{-\zeta_k^B r_B}$$
In **PHATPSY 2.0**:
1. **LAPACK SVD Constrained Fitting (`DGELSD`):** Implemented in [`src/analys.f90`](file:///Users/jacksmith/phatpsy2/src/analys.f90) with exact analytical cusp boundary constraints:
   - Nuclear cusp: $Z_p(0) = Z - N_{\text{core}}$
   - Nuclear slope: $(dZ_p/dr)_{r=0} = -2 Z (Z - N_{\text{core}})$
   - Poisson total charge conservation: $\int \nabla^2 V^B d^3r = 4\pi Q_B$
2. **Dynamic STO Exponent Optimization:** Screening exponents $\{\zeta_k^B\}$ are optimized along the negative gradient of the potential residual, retiring fixed empirical tables.
3. **Modern DFT Exchange Kernels:** Spherically symmetric radial quadrature evaluation of local Slater exchange ($\rho^{1/3}$), Becke88 gradient exchange, and Perdew-Burke-Ernzerhof (PBE) GGA exchange in [`src/exchange_dft.f90`](file:///Users/jacksmith/phatpsy2/src/exchange_dft.f90) and [`phatpsy/exchange_dft.py`](file:///Users/jacksmith/phatpsy2/phatpsy/exchange_dft.py), exposed via `ControlConfig(exchange_kernel="slater"|"becke88"|"pbe")`.

---

## 5. Molecular EWMO, Robust Charge Equilibration & Subsystem Energies

For open-shell surface clusters, polar diatomics, and chemisorption complexes:
1. **Symmetrically Orthogonalized Löwdin Population Analysis:**
   $$\mathbf{P}_{\text{Lowdin}} = \mathbf{S}^{1/2} \mathbf{D} \mathbf{S}^{1/2}$$
   Implemented in [`src/density.f90`](file:///Users/jacksmith/phatpsy2/src/density.f90), guaranteeing non-negative gross orbital populations and strict electron conservation $\operatorname{Tr}(\mathbf{P}_{\text{Lowdin}}) = N_{\text{elec}}$ without diffuse basis unphysical artifacts.
2. **Pulay DIIS Charge Acceleration:**
   Integrated into [`phatpsy/driver.py`](file:///Users/jacksmith/phatpsy2/phatpsy/driver.py) via `DIISAccelerator`, dynamically minimizing error vectors $\mathbf{e}^{(m)} = \mathbf{q}^{(m)} - \mathbf{q}^{(m-1)}$ across SCF iterations with total charge renormalization.
3. **Global Fermi Level ($\mu$) Equilibration:**
   Implemented in [`src/ewmo.f90`](file:///Users/jacksmith/phatpsy2/src/ewmo.f90) (`solve_fermi_level`) via hybrid Newton-Raphson/bisection under Fermi-Dirac distribution:
   $$n_i = \frac{1}{1 + \exp((\varepsilon_i - \mu)/\Theta)}$$
   Satisfies particle conservation $\sum_i n_i = N_{\text{elec}}$ to machine precision ($< 10^{-11}\text{ Ha}$) across any electronic temperature $\Theta$.
4. **Subsystem Energy Partitioning:**
   Physical embedding formulation in [`phatpsy/subsystem.py`](file:///Users/jacksmith/phatpsy2/phatpsy/subsystem.py):
   $$E_{\text{total}} = E_{\text{cluster}} + E_{\text{env}}^{\text{bulk}} + E_{\text{embed}}$$
   with frozen bulk references ($-1502.565\text{ Ha/atom}$ for $\text{Ni}$), isolated atomic ground-state references from Clementi & Roetti / Thakkar, exact energy conservation ($< 10^{-12}\text{ Ha}$), and excess energy metrics ($E_{\text{excess}} = E_{\text{total}} - E_{\text{env}}^{\text{bulk}}$).
5. **Parameter Retirement:**
   Legacy empirical knobs (`CDAMP`, `VDAMP`, `CACCEL`) are formally retired in favor of first-principles self-consistency via `ControlConfig.first_principles()`.

---

## 6. Localized Valence-Bond (LVB) Alternating Spin

For chemisorption and metal clusters with radical character but overall zero net molecular spin:
- Orbitals are treated in an Unrestricted Hartree-Fock (UHF) framework with local spin polarization.
- Alpha ($\alpha$) and Beta ($\beta$) spins are assigned to alternating atomic centers (`QFLIP=T` or `spin="down"`).
- Automated via a bipartite graph solver in [`phatpsy/spin.py`](../../phatpsy/spin.py), yielding decoupled electron pairs with zero net spin ($S_z = 0$).

---

## 7. Special Atomic Centers, Symmetry Operations & Atom Equivalency

PHATPSY provides specialized atomic center classifications and spatial symmetry operators tailored for large molecular complexes, crystalline lattices, and chemisorption surfaces:

### A. Atom Equivalency & Spatial Symmetry (`NEQAT`, `ISYM`, `IXYZ`)
- **Primary Generator Atoms:** An independent unique atom is defined with its coordinates and Slater orbital basis.
- **Symmetry-Equivalent Atoms (`NEQAT`):** Rather than repeating calculations for all symmetry-related sites, a generator atom specifies `NEQAT` equivalent centers generated via spatial point-group operations:
  - **Reflection (`ISYM = -1`):** Mirror reflection through the Cartesian coordinate plane perpendicular to axis `IXYZ` ($1=x, 2=y, 3=z$).
  - **Inversion (`ISYM = -2`):** Inversion through the coordinate origin ($\mathbf{r} \to -\mathbf{r}$).
  - **Rotation (`ISYM = N`, $N \ge 2$):** $N$-fold proper rotation ($C_N$) about coordinate axis `IXYZ`.
- **Orbital Transformation (`SYMOP`):** Atomic orbital coefficients and model potentials are transformed into equivalent centers using Clebsch-Gordan coefficients and Wigner rotation matrices (`SYMOP`, `SYMXYZ`), eliminating redundant integral evaluations and reducing SCF dimension.

### B. Spin-Flipping on Symmetry Operations (`QFLIP = T`)
- Symmetry operations can simultaneously flip the spin quantization axis (`QFLIP = T`).
- Calls `FLIP` to exchange $\alpha$ and $\beta$ Fock matrices, eigenvectors, and eigenvalues on the generated center.
- Enables constructing rigorous antiferromagnetic / Localized Valence-Bond singlet states (e.g. $N_2$, ethylene, transition-metal dimers) using spatial point-group symmetry.

### C. Fixed / Frozen Environment Atoms (`QFIXED = T`, `fixed=True`)
- **Physical Purpose:** Models chemisorption on infinite or semi-infinite substrate surfaces (e.g., $CO$ adsorbed on a $Ni_{16}(100)$ cluster).
- **Behavior:** The atomic orbitals, eigenvalues, and model potentials of fixed atoms are computed initially (or read from restart units) and held **completely frozen** throughout the molecular calculation (`qdone = QFIXED`).
- **Computational Benefit:** Eliminates unnecessary SCF iterations on deep substrate or buffer layers, focusing variational degrees of freedom entirely on the active chemisorption site and adsorbate.

### D. Virtual Atoms (`QVIRTL = T`, `virtual=True`)
- **Physical Purpose:** Used to **complete the local neighborhood and coordination environment (external potentials)** of an atom during its local atomic calculation. For surface clusters, edge atoms, or chemisorption models, real physical atoms on the perimeter would otherwise experience an artificial vacuum truncation; virtual atoms provide the surrounding crystalline/lattice potential field so the active atom experiences a realistic coordination environment during its local phase.
- **Exclusion at the Molecular (EWMO) Level:** Virtual centers are explicitly **excluded at the molecular / Energy-Weighted Maximum Overlap (EWMO) calculation level** (`if (.not. QVIRTL)` in `phatpsy_atomic_mod.f90`). This preserves a finite, tractable cluster model of the system without expanding the molecular secular matrix.
- **Important Distinction:** In PHATPSY, virtual atoms are **not** used for Basis Set Superposition Error (BSSE) counterpoise corrections. They act as environment/coordination completion potentials that drop out of the final molecular secular equation.
- In the Fortran core (`phatpsy_atomic_mod.f90`), virtual centers do not increment the count of real physical molecular atoms (`nratom`), have a zero cutoff radius (`rmax = 0`), and do not contribute to classical molecular nuclear repulsion energy ($Z_A Z_B / R_{AB}$).
