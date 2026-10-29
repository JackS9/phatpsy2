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
2. **Generalized Pseudometric & Dynamic Projection:** The energy dependence is handled dynamically via the regularized pseudometric ($M = I - P$) and standard LAPACK eigensolvers (`DSPEVD`), completely stabilizing 12-coordinated bulk metals ($\text{Ni}_{13}$ FCC cluster) under pure theory `PKSCAL = 1.0` without requiring manual reference energy overrides.
3. **Symmetrized Operator:** Evaluated as a symmetric matrix in [`src/scf.f90`](file:///Users/jacksmith/phatpsy2/src/scf.f90), preserving exact Hermiticity, exact total energy conservation, and physical virial ratios ($-V/T \approx 2.0$).
4. **Hydrogen Core-Exemption Invariant ($Z=1$):** Because the Phillips-Kleinman operator is fundamentally an interatomic core projector ($V_{\text{PK}} = \sum_{j \in \text{core}} (\varepsilon - \varepsilon_j) |\phi_j\rangle\langle\phi_j|$), and Hydrogen possesses zero core electrons ($N_{\text{core}} = 0$), Hydrogen centers are exempt from core projection at the engine level in [`src/scf.f90`](file:///Users/jacksmith/phatpsy2/src/scf.f90). The overlap metric on Hydrogen remains unperturbed ($M = \mathbf{S}$), and Hydrogen emits zero core projection onto neighboring atoms. This prevents unphysical valence over-projection, allowing pure theoretical unit scaling (`PKSCAL = 1.0`, `use_pseudometric: true`) to serve as the universal default across all polyatomics without per-molecule tuning.

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

---

## 8. Surface Chemisorption Hierarchy (Dissertation Chapter IX: $NO$ and $CO$ on $Ni(100)$)

In the PhD dissertation (Jack A. Smith, 1980, Chapter IX), chemisorption on transition-metal surfaces is addressed through a disciplined multi-scale localized progression designed to avoid artificial finite-cluster edge/boundary truncation effects without resorting to periodic boundary conditions:

### A. The 4-Stage Progressive Hierarchy
1. **Isolated Molecule ($NO$ / $CO$):**
   - Solved as a gas-phase diatomic to establish the isolated electronic structure, bond length ($R_e$), harmonic vibrational frequency ($\omega_e$), dipole moment ($\mu$), and reference energy ($E_{\text{molecule}}$).
2. **Embedded Bulk Metal Atom ($Ni_{\text{bulk}}$):**
   - Modeled via a high-coordination cluster (12-coordinated FCC $Ni_{13}$) or periodic supercell with theoretical $\text{PKSCAL}=1.0$.
150:    - Yields the bulk reference cohesive energy per atom ($-1502.5650\text{ Ha/atom}$) and the self-consistent electronic state of an embedded bulk atom (`Ni_bulk_fixed`).
151: 3. **Clean Embedded Surface Slab ($Ni(100)$):**
152:    - Modeled via a 3-layer embedded slab (`examples/ni_100_slab.yaml`) to capture surface relaxation:
153:      - **Layer 1 (Surface):** `Ni_surf` active centers ($Z=8$ coordination) undergoing coordination reduction and electronic relaxation.
154:      - **Layer 2 (Subsurface):** `Ni_sub` active centers mediating screening between the vacuum surface and the interior bulk.
155:      - **Layer 3 (Bulk):** `Ni_bulk_fixed` boundary atoms fixed at the bulk electronic state.
156:    - Converged states produce transferable, reusable `EmbeddedAtomState` instances for `Ni_surf` and `Ni_sub`.
157: 4. **Chemisorbed Complex ($CO/Ni(100)$ or $NO/Ni(100)$):**
158:    - The adsorbate and the immediate chemisorption site are placed atop the substrate.
159:    - **Boundary Invariant:** All surrounding "boundary" atoms are held strictly **fixed (`fixed: true`) at their respective representations**:
160:      - Bulk boundary atoms fixed at their bulk state (`Ni_bulk_fixed`)
161:      - Subsurface boundary atoms fixed at their second-layer state (`Ni_sub`)
162:      - Surface boundary atoms fixed at their surface-layer state (`Ni_surf`)
163:    - This prevents unphysical edge charge-sloshing or artificial cluster boundary distortion, ensuring that the local chemisorption bond experiences the correct electrostatic and Pauli embedding of a semi-infinite crystal substrate.
164: 
165: ---
166: 
167: ## 9. Semi-Classical Dispersion Corrections: DFT-D3(BJ) (Phase 6)
168: 
169: While the projected Hamiltonian accurately captures valence electron pairing, Pauli exclusion via $V_{\text{PK}}$, and local electrostatic interactions, long-range London dispersion forces arise from non-local dynamic electron correlations that fall outside single-determinant self-consistent field frameworks.
170: 
171: In **PHATPSY 2.0**, long-range dispersion is incorporated via Grimme's **DFT-D3(BJ)** methodology with Becke-Johnson rational damping ([`phatpsy/dispersion.py`](../../phatpsy/dispersion.py)):
172: $$E_{\text{disp}} = -\frac{1}{2} \sum_{A \neq B} \left[ s_6 \frac{C_6^{AB}}{R_{AB}^6 + [R_0^{AB}(a_1, a_2)]^6} + s_8 \frac{C_8^{AB}}{R_{AB}^8 + [R_0^{AB}(a_1, a_2)]^8} \right]$$
173: where the Becke-Johnson rational cutoff radius is:
174: $$R_0^{AB}(a_1, a_2) = a_1 \sqrt{\frac{C_8^{AB}}{C_6^{AB}}} + a_2 = a_1 \sqrt{3 R_{0,A} R_{0,B}} + a_2$$
175: The dispersion coefficients satisfy the Casimir-Polder combination rule $C_6^{AB} = \sqrt{C_6^A C_6^B}$ and $C_8^{AB} = 3 C_6^{AB} (R_{0,A} R_{0,B})$.
176: 
177: ### Analytical Nuclear Dispersion Gradients:
178: To enable gradient-driven geometry optimizations, transition state searches, and molecular dynamics with dispersion, analytical Cartesian gradients $\mathbf{g}_{\text{disp}} = -\nabla E_{\text{disp}}$ are evaluated in closed form:
179: $$\frac{\partial E_{\text{disp}}}{\partial \mathbf{R}_A} = \sum_{B \neq A} \left[ \frac{6 s_6 C_6^{AB} R_{AB}^4}{(R_{AB}^6 + [R_0^{AB}]^6)^2} + \frac{8 s_8 C_8^{AB} R_{AB}^6}{(R_{AB}^8 + [R_0^{AB}]^8)^2} \right] (\mathbf{R}_A - \mathbf{R}_B)$$
180: This provides strictly conservative dispersion forces that satisfy exact translational and rotational invariance.
181: 
182: ---
183: 
184: ## 10. Polarized Slater-Type Orbital (DZP) Basis Sets (Phase 6)
185: 
186: To accurately describe molecular polarization, quadrupole moments, chemical bond anisotropy, and directional back-bonding (e.g. $d_\pi$--$\pi^*$ interactions in metal carbonyls), PHATPSY 2.0 expands beyond minimal and double-zeta representations by introducing **Double-Zeta plus Polarization (DZP)** basis sets ([`phatpsy/data/polarized_sto.json`](../../phatpsy/data/polarized_sto.json)):
187: - **Hydrogen ($Z=1$):** Augmented with a $2p$ polarization shell ($l=1$, $m=0, \pm 1$; $\zeta_{2p} = 1.000\ \text{a.u.}^{-1}$).
188: - **First-Row Elements ($\text{Li}$--$\text{Ne}$):** Augmented with a full $3d$ polarization shell ($l=2$, $m=0, \pm 1, \pm 2$; 5 real components).
189: - **Second-Row Elements ($\text{Na}$--$\text{Ar}$):** Augmented with a $3d$ polarization shell.
190: - **Transition Metals ($\text{Sc}$--$\text{Cu}$):** Augmented with $4f$ polarization functions ($l=3$; 7 real components).
191: 
192: ### High Angular Momentum Slater Normalization:
193: Real Slater-Type Orbitals are evaluated on 3D coordinate grids with exact analytical normalization:
194: $$\chi_{nlm}(\mathbf{r}) = N_n(\zeta) r^{n-1} e^{-\zeta r} S_{lm}(\theta, \phi)$$
195: $$N_n(\zeta) = \frac{(2\zeta)^{n + 1/2}}{\sqrt{(2n)!}}$$
196: where $S_{lm}$ are real spherical harmonics with exact orthonormalization $\int S_{lm} S_{l'm'} d\Omega = \delta_{ll'} \delta_{mm'}$.
197: 
198: ---
199: 
200: ## 11. Equation-of-Motion (EOM) Polarizability & Spectroscopic Properties (Phase 6)
201: 
202: Phase 6 introduces the Equation-of-Motion (EOM) and response property engine in [`phatpsy/eom.py`](../../phatpsy/eom.py):
203: 
204: ### A. Origin-Invariant Dipole & Traceless Quadrupole Tensors
205: To eliminate origin-dependence in charged or asymmetric polyatomics, moments are referenced to the center of nuclear charge $\mathbf{R}_0 = \sum_A Z_A \mathbf{R}_A / \sum_A Z_A$:
206: $$\boldsymbol{\mu} = \sum_A Z_A (\mathbf{R}_A - \mathbf{R}_0) - \int (\mathbf{r} - \mathbf{R}_0) \rho(\mathbf{r}) d^3r$$
207: $$\mathbf{\Theta}_{\alpha\beta} = \frac{1}{2} \left[ 3 Q_{\alpha\beta} - \operatorname{Tr}(\mathbf{Q}) \delta_{\alpha\beta} \right]$$
208: where the primitive second-moment tensor $\mathbf{Q} = \mathbf{Q}_{\text{nuc}} - \mathbf{Q}_{\text{elec}}$ is:
209: $$Q_{\alpha\beta} = \sum_A Z_A (R_{A,\alpha} - R_{0,\alpha})(R_{A,\beta} - R_{0,\beta}) - \int (r_\alpha - R_{0,\alpha})(r_\beta - R_{0,\beta}) \rho(\mathbf{r}) d^3r$$
210: The resulting quadrupole tensor is strictly symmetric and traceless ($\operatorname{Tr}(\mathbf{\Theta}) = 0$).
211: 
212: ### B. Sum-Over-States (SOS) Static & Dynamic Polarizability
213: Using the occupied-to-virtual excitation manifold $\{\psi_i \to \psi_a\}$ with excitation energies $\Delta \varepsilon_{ia} = \varepsilon_a - \varepsilon_i$:
214: $$\boldsymbol{\alpha}(0) = 2 \sum_{i \in \text{occ}} \sum_{a \in \text{vir}} \frac{\boldsymbol{\mu}_{ia} \otimes \boldsymbol{\mu}_{ia}}{\Delta \varepsilon_{ia}}$$
215: $$\boldsymbol{\alpha}(\omega) = 2 \sum_{i \in \text{occ}} \sum_{a \in \text{vir}} \frac{\Delta \varepsilon_{ia}}{\Delta \varepsilon_{ia}^2 - \omega^2} \boldsymbol{\mu}_{ia} \otimes \boldsymbol{\mu}_{ia}$$
216: where $\boldsymbol{\mu}_{ia} = \langle \psi_i | \mathbf{r} | \psi_a \rangle$ are transition dipole vectors and $f_{ia} = \frac{2}{3} \Delta \varepsilon_{ia} \|\boldsymbol{\mu}_{ia}\|^2$ are optical oscillator strengths.
217: 
218: ### C. Magnetic Susceptibility Tensors
219: Molecular diamagnetic and paramagnetic susceptibilities are computed via second moments and angular momentum operators $\hat{\mathbf{L}} = -i (\mathbf{r} \times \nabla)$:
220: $$\boldsymbol{\chi}_{\text{dia}} = -\frac{e^2}{4 m_e c^2} \langle \mathbf{r}^2 \mathbf{I} - \mathbf{r} \otimes \mathbf{r} \rangle$$
221: $$\boldsymbol{\chi}_{\text{para}} = \frac{e^2}{2 m_e^2 c^2} \sum_{i \in \text{occ}} \sum_{a \in \text{vir}} \frac{\langle \psi_i | \hat{\mathbf{L}} | \psi_a \rangle \otimes \langle \psi_a | \hat{\mathbf{L}} | \psi_i \rangle}{\Delta \varepsilon_{ia}}$$
222: 
223: ---
224: 
225: ## 12. Analytical Cartesian Gradients & Physical Invariants (Phase 7)
226: 
227: Geometry optimization, path finding, and molecular dynamics require Cartesian nuclear forces $\mathbf{F}_A = -\nabla_A E$. In [`phatpsy/driver.py`](../../phatpsy/driver.py):
228: $$\mathbf{g}_A = -\mathbf{F}_A = \mathbf{g}_A^{\text{elec}} + \mathbf{g}_A^{\text{nuc}} + \mathbf{g}_A^{\text{disp}}$$
229: 
230: To preserve physical boundary conditions on unconstrained finite molecules:
231: 1. **Translational Invariance:** Total nuclear force vanishes identically:
232:    $$\sum_{A=1}^{N_{\text{atom}}} \mathbf{F}_A = \mathbf{0}$$
233: 2. **Rotational Invariance:** Total external torque vanishes identically:
234:    $$\sum_{A=1}^{N_{\text{atom}}} \mathbf{R}_A \times \mathbf{F}_A = \mathbf{0}$$
235: 
236: ---
237: 
238: ## 13. Harmonic Vibrational Frequencies & Statistical Thermochemistry (Phase 7)
239: 
240: Second derivatives of the electronic energy define the Cartesian Hessian matrix ([`phatpsy/vibrations.py`](../../phatpsy/vibrations.py)):
241: $$H_{Ai, Bj} = \frac{\partial^2 E}{\partial R_{Ai} \partial R_{Bj}} \approx \frac{g_{Ai}(R_{Bj} + h) - g_{Ai}(R_{Bj} - h)}{2h}$$
242: 
243: ### Eckart-Sayvetz Translation/Rotation Projection:
244: To remove the 6 spurious zero-frequency degrees of freedom (5 for linear molecules) corresponding to rigid-body translation and rotation, the mass-weighted Hessian $\mathbf{H}_M = \mathbf{M}^{-1/2} \mathbf{H} \mathbf{M}^{-1/2}$ is projected using the Eckart projector $\mathbf{P}$:
245: $$\mathbf{P} = \mathbf{I} - \mathbf{D} (\mathbf{D}^T \mathbf{D})^{-1} \mathbf{D}^T$$
246: $$\mathbf{H}_M^{\text{proj}} = \mathbf{P} \mathbf{H}_M \mathbf{P}$$
247: Diagonalization yields eigenvalues $\lambda_k = \omega_k^2$, converted to harmonic vibrational wavenumbers:
248: $$\tilde{\nu}_k = \frac{\sqrt{\lambda_k}}{2 \pi c} \quad [\text{cm}^{-1}]$$
249: 
250: ### Statistical Thermochemistry (Ideal Gas / Rigid Rotor / Harmonic Oscillator):
251: From the normal mode spectrum $\{\nu_k\}$ and moments of inertia:
252: - **Zero-Point Vibrational Energy (ZPVE):**
253:   $$E_{\text{ZPVE}} = \frac{1}{2} \sum_k h \nu_k$$
254: - **Thermal Enthalpy & Gibbs Free Energy:**
255:   $$H(T) = E_{\text{electronic}} + E_{\text{ZPVE}} + E_{\text{trans}}(T) + E_{\text{rot}}(T) + E_{\text{vib}}(T) + k_B T$$
256:   $$G(T) = H(T) - T [S_{\text{trans}}(T) + S_{\text{rot}}(T) + S_{\text{vib}}(T) + S_{\text{elec}}(T)]$$
257: 
258: ---
259: 
260: ## 14. Born-Oppenheimer Molecular Dynamics (BOMD) (Phase 7)
261: 
262: Time-dependent nuclear dynamics on the Born-Oppenheimer ground-state potential energy surface are integrated via the **Velocity Verlet algorithm** ([`phatpsy/md.py`](../../phatpsy/md.py)):
263: $$\mathbf{R}(t + \Delta t) = \mathbf{R}(t) + \mathbf{v}(t) \Delta t + \frac{1}{2} \mathbf{a}(t) \Delta t^2$$
264: $$\mathbf{v}(t + \Delta t) = \mathbf{v}(t) + \frac{1}{2} [\mathbf{a}(t) + \mathbf{a}(t + \Delta t)] \Delta t$$
265: where $\mathbf{a}_A = -\mathbf{g}_A / M_A$.
266: 
267: ### Ensembles & Thermostats:
268: - **NVE Microcanonical:** Total energy $E_{\text{tot}} = E_{\text{pot}} + E_{\text{kin}}$ is strictly conserved ($|\Delta E| / E_0 < 10^{-4}$ over picosecond trajectories).
269: - **Berendsen Weak-Coupling Thermostat:** Velocity scaling with relaxation time $\tau$:
270:   $$\lambda = \sqrt{1 + \frac{\Delta t}{\tau} \left( \frac{T_0}{T(t)} - 1 \right)}$$
271: - **Andersen Stochastic Collision Thermostat:** Random thermal collisions with frequency $\nu_{\text{coll}}$ resample velocities from the Maxwell-Boltzmann distribution:
272:   $$P(v_{Ai}) = \sqrt{\frac{M_A}{2\pi k_B T_0}} \exp\left( -\frac{M_A v_{Ai}^2}{2 k_B T_0} \right)$$
273: - **Fixed Substrate Atom Constraints:** Atoms flagged with `fixed: true` maintain $\mathbf{v}_A = \mathbf{0}$ and $\mathbf{a}_A = \mathbf{0}$, ideal for simulating gas-surface collisions on slab substrates.
274: 
275: ---
276: 
277: ## 15. Minimum Energy Reaction Pathways: CI-NEB, Dimer Method & IRC (Phase 8)
278: 
279: Phase 8 provides a unified suite for mapping chemical reaction mechanisms, finding first-order saddle points, and tracing reaction coordinates:
280: 
281: ### A. Climbing-Image Nudged Elastic Band (CI-NEB)
282: The CI-NEB method ([`phatpsy/neb.py`](../../phatpsy/neb.py)) discretizes the transition path between a reactant and product into an elastic band of $N$ images $\{\mathbf{R}_0, \mathbf{R}_1, \dots, \mathbf{R}_{N-1}\}$.
283: 1. **Henkelman-Jónsson Improved Tangents:** Tangents $\hat{\boldsymbol{\tau}}_i$ are evaluated using the energy ordering of neighboring images to prevent artificial kinking and corner-cutting:
284:    $$\boldsymbol{\tau}_i = \begin{cases} 
285:    \boldsymbol{\tau}_i^+ & \text{if } V_{i+1} > V_i > V_{i-1} \\
286:    \boldsymbol{\tau}_i^- & \text{if } V_{i+1} < V_i < V_{i-1} \\
287:    \Delta V_{\max} \boldsymbol{\tau}_i^+ + \Delta V_{\min} \boldsymbol{\tau}_i^- & \text{otherwise}
288:    \end{cases}$$
289: 2. **Force Decomposition:** True potential forces and spring forces are decoupled:
290:    $$\mathbf{F}_i = (\mathbf{F}_i^{\text{spring}} \cdot \hat{\boldsymbol{\tau}}_i) \hat{\boldsymbol{\tau}}_i + [\mathbf{F}_i^{\text{true}} - (\mathbf{F}_i^{\text{true}} \cdot \hat{\boldsymbol{\tau}}_i) \hat{\boldsymbol{\tau}}_i]$$
291: 3. **Climbing-Image Saddle Point Location:** The highest-energy image inverts the parallel component of the true gradient:
292:    $$\mathbf{F}_{i_{\max}} = \mathbf{F}_{i_{\max}}^{\text{true}} - 2 (\mathbf{F}_{i_{\max}}^{\text{true}} \cdot \hat{\boldsymbol{\tau}}_{i_{\max}}) \hat{\boldsymbol{\tau}}_{i_{\max}}$$
293:    driving the climbing image precisely to the saddle point without extra spring forces.
294: 4. **QuickMin Optimizer:** Accelerates convergence by projecting velocities along the force vector $\mathbf{v} \to (\mathbf{v} \cdot \hat{\mathbf{F}}) \hat{\mathbf{F}}$ and resetting $\mathbf{v} = \mathbf{0}$ whenever $\mathbf{v} \cdot \mathbf{F} < 0$.
295: 
296: ### B. Dimer Method for Saddle Point Refinement
297: The Heyden/Bell/Henkelman Dimer method ([`phatpsy/dimer.py`](../../phatpsy/dimer.py)) refines saddle points using only first derivatives (gradients), bypassing full Hessian evaluations:
298: 1. Two images separated by $2\Delta R$ define a dimer with orientation vector $\hat{\mathbf{N}}$:
299:    $$\mathbf{R}_1 = \mathbf{R}_0 + \Delta R \hat{\mathbf{N}}, \quad \mathbf{R}_2 = \mathbf{R}_0 - \Delta R \hat{\mathbf{N}}$$
300: 2. **Rotational Search:** The dimer rotates to align $\hat{\mathbf{N}}$ with the lowest curvature eigenmode:
301:    $$C = \frac{(\mathbf{F}_1 - \mathbf{F}_2) \cdot \hat{\mathbf{N}}}{2 \Delta R}$$
302: 3. **Translational Step:** The midpoint moves under an effective inverted force:
303:    $$\mathbf{F}^\dagger = \begin{cases}
304:    -\mathbf{F}_0 + 2(\mathbf{F}_0 \cdot \hat{\mathbf{N}}) \hat{\mathbf{N}} & \text{if } C < 0 \\
305:    -(\mathbf{F}_0 \cdot \hat{\mathbf{N}}) \hat{\mathbf{N}} & \text{if } C > 0
306:    \end{cases}$$
307:    translating uphill along the lowest mode and downhill in all other degrees of freedom.
308: 
309: ### C. Intrinsic Reaction Coordinate (IRC)
310: The IRC path ([`phatpsy/irc.py`](../../phatpsy/irc.py)) defines the steepest-descent path in mass-weighted coordinates from a transition state to reactants and products:
311: $$\frac{d\mathbf{x}}{ds} = -\frac{\mathbf{g}_x}{\|\mathbf{g}_x\|}, \quad \mathbf{x}_A = \sqrt{M_A} \mathbf{R}_A$$
312: Initiated along the forward and backward directions of the imaginary transition normal mode $\hat{\mathbf{Q}}_{\text{imag}}$, verifying that the transition state rigorously connects the postulated reactant and product minima.

