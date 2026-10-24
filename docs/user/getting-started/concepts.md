# Core Concepts

This chapter introduces the fundamental scientific and computational concepts employed by PHATPSY 2.0. For complete mathematical formulations, consult the [Science & Methodology](../../science/methodology_overview.md) documentation.

---

## 1. Projected Hamiltonian SCF

Traditional *ab initio* Hartree-Fock calculations compute four-center two-electron repulsion integrals whose evaluation scales as $\mathcal{O}(N^4)$. The **Projected Hamiltonian Approach** (developed by Dr. Jack A. Smith) replaces four-center integrals with projection-operator representations of atomic and molecular charge distributions:

$$\hat{F} = \hat{h} + \sum_i \hat{J}_i - \sum_i \hat{K}_i + \hat{V}_{\text{proj}}$$

Where:

* $\hat{h}$ is the one-electron core Hamiltonian (kinetic energy and electron-nuclear attraction).
* $\hat{J}_i$ and $\hat{K}_i$ are the Coulomb and Exchange operators, represented via optimized model density fittings.
* $\hat{V}_{\text{proj}}$ is a non-local projection operator that enforces orthogonality to core or substrate states without requiring explicit core basis functions.

This formulation allows fast self-consistent field (SCF) calculations on large polyatomic clusters, transition metal complexes, and chemisorption slabs.

---

## 2. Slater-Type Orbitals (STOs)

Unlike standard quantum chemistry programs that use Gaussian-Type Orbitals (GTOs), PHATPSY 2.0 uses **Slater-Type Orbitals (STOs)**:

$$\chi_{nlm}(\mathbf{r}) = N_n r^{n-1} e^{-\zeta r} Y_{lm}(\theta, \phi)$$

STOs possess two critical physical properties that GTOs lack:

1. **Nuclear Cusp Condition:** The electron density has a correct finite derivative cusp at the nucleus ($\lim_{r \to 0} \frac{d\psi}{dr} = -Z\psi(0)$).
2. **Proper Asymptotic Decay:** Orbitals decay exponentially as $e^{-\zeta r}$ at long range, ensuring accurate long-range overlap and tail interactions.

### Built-in Basis Libraries

PHATPSY 2.0 includes compiled basis sets derived from Clementi & Roetti and Thakkar:

* **SZ (Single-Zeta):** Minimal basis (one STO per occupied subshell). Fast exploratory calculations.
* **DZ (Double-Zeta):** Two STOs per valence subshell. Standard for quantitative molecular and cluster studies.
* **THAKKAR:** Extended basis sets with polarized valence functions.

You can inspect all available elements and orbital exponents using `phatpsy elements -b DZ`.

---

## 3. Embedded Atoms & Chemisorption

A core capability of PHATPSY is modeling surface adsorption (e.g., carbon monoxide on a nickel surface). In such systems, treating an entire metal bulk *ab initio* is computationally prohibitive.

PHATPSY models substrate atoms using **embedded states**:

* **Substrate Projectors:** Deep or bulk substrate atoms can be designated with `embedded_state: 1`. Their atomic projectors are pre-computed in their bulk coordination state and frozen during the chemisorption cycle.
* **Active Valence Region:** Only the adsorbate and immediate surface atoms undergo active SCF relaxation.

---

## 4. Local Valence Bond (LVB) Alternating Spin

Standard Unrestricted Hartree-Fock (UHF) calculations often converge to closed-shell symmetry-adapted solutions even when bond breaking or biradical character is present.

PHATPSY utilizes the **Local Valence Bond (LVB)** initial spin assignment:

* Adjacent bonded atoms are seeded with opposite spin orientations (`spin: up` and `spin: down`).
* This breaks artificial spin symmetry in the initial density matrix, guiding the SCF optimizer toward the true open-shell, antiferromagnetic, or dissociation ground states.
* You can automate this across any molecule using `CalculationDeck.assign_alternating_spin()` or `phatpsy build -s`.

---

## 5. Fixed and Virtual Centers

Atoms in the `geometry:` block can be assigned special physical roles:

* **`fixed: true`:** Prevents the optimizer from moving these atoms during geometry optimization or potential energy scans. Ideal for anchoring bottom layers of a crystal slab.
* **`virtual: true`:** Virtual (ghost) atoms carry projector potentials and basis functions to eliminate boundary edge effects or model Madelung potentials, but do not contribute active nuclear charges or electrons to the system.

---

## 6. Symmetry Reduction

PHATPSY 2.0 can exploit molecular point symmetry to eliminate redundant integral computations. In the `geometry:` list, equivalent atoms can reference a master atom:

```yaml
geometry:
  - element: N
    id: N1
    xyz: [0.0, 0.0, 1.034]
    spin: up
  - element: N
    id: N2
    xyz: [0.0, 0.0, -1.034]
    equivalent_to: N1
    symmetry_op: -1
    symmetry_axis: 1
    spin: down
```

Here, `N2` is generated from `N1` via inversion along the z-axis (`symmetry_axis: 1`, `symmetry_op: -1`).
