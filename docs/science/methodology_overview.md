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

### The Energy-Dependence Challenge (Active Work in Progress):
Unlike static local potentials, $V_{\text{PK}}(\varepsilon_i)$ depends explicitly on the eigenvalue $\varepsilon_i$ of the state being sought. Because of this energy-dependence:
1. It **cannot be solved exactly** as a single fixed static potential for all molecular states simultaneously.
2. PHATPSY must treat this operator in an **approximate fashion**—for example, by reformulating into a generalized pseudometric ($F |\tilde{\psi}_i\rangle = \varepsilon_i M |\tilde{\psi}_i\rangle$), truncating interatomic overlap expansions to second-order, or using reference eigenvalue approximations.
3. Establishing the most robust and computationally effective way to approximate this energy-dependent atom-atom projector remains an active research investigation and a work in progress within the PHATPSY methodology.

---

## 4. Local Model Potentials for Interatomic Coulomb Interactions (Chapter 4)

To eliminate two-electron multi-center integrals, the Coulomb potential generated by a neighboring atomic charge distribution on center $B$ is represented as a fitted spherical model potential:
$$V_{\text{Coulomb}}^B(r_B) = -\frac{Z_B^*}{r_B} + \sum_{k} c_k^B e^{-\zeta_k^B r_B}$$
The expansion coefficients $\{c_k^B\}$ are fit self-consistently every SCF cycle under constrained boundary conditions:
1. **Nuclear Cusps:** Reproduces exact $-Z/r$ divergence at $r \to 0$.
2. **Poisson Screening:** Guarantees total integrated electronic charge matches gross atomic populations.
3. **Core and Valence Potential Matching:** Fits model potential values at characteristic radial shells.

In modern PHATPSY 2.0, this constrained fitting is standardized on LAPACK linear least squares (`dgels`) in [`src/phatpsy_analys_mod.f90`](../../src/phatpsy_analys_mod.f90).

---

## 5. Grand-Canonical Hartree-Fock (GCHF) & Fermi Temperature (Chapters 7 & 8)

For open systems and chemisorption surfaces with fractional charge transfer:
- Orbital occupations $n_i$ are assigned via a Fermi-Dirac distribution characterized by a pseudo-temperature parameter `THETA`:
  $$n_i = \frac{1}{1 + e^{(\varepsilon_i - \mu) / \Theta}}$$
- As $\Theta \to 0$, standard integer Aufbau occupations are recovered.
- Inter-atomic charge transfer between cycles is damped using relative charge transfer (RCT) damping (`CDAMP`, `VDAMP`) to guarantee numerical convergence.

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
