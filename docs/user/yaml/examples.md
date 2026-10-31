# Annotated YAML Examples

This chapter presents production-ready calculation decks from the `examples/` directory with detailed scientific annotations.

---

## 1. Diatomic Nitrogen ($N_2$) with Double-Zeta Basis

This calculation computes the closed-shell $N_2$ molecule at its equilibrium experimental bond distance ($R_e = 2.068\text{ Bohrs} \approx 1.094\text{ \AA}$).

```yaml
title: "Nitrogen Molecule (N2) - Double-Zeta Basis"
control:
  spin_unrestricted: true
  max_cycles: 100
  convergence: 1.0e-4
  separate_atoms_first_cycle: true

geometry:
  - element: N
    id: N1
    xyz: [0.0, 0.0, 1.034]
    spin: up
  - element: N
    id: N2
    xyz: [0.0, 0.0, -1.034]
    spin: down

elements:
  N:
    symbol: N
    basis: DZ
```

### Key Features:
* `separate_atoms_first_cycle: true`: First solves isolated nitrogen atoms to generate atomic projectors, establishing rapid SCF convergence (typically $\le 10$ cycles).
* `spin: up` and `spin: down`: Seeds the initial density with Local Valence Bond (LVB) character.
* Converged energy: $-109.8930\text{ Hartree}$, Virial ratio: $2.0441$.

---

## 2. Carbon Monoxide (CO)

A heteronuclear diatomic molecule demonstrating unequal nuclear charges and dipole polarization.

```yaml
title: "Carbon Monoxide (CO) - Double-Zeta Basis"
control:
  spin_unrestricted: true
  max_cycles: 100
  convergence: 1.0e-4
  separate_atoms_first_cycle: true

geometry:
  - element: C
    id: C1
    xyz: [0.0, 0.0, -1.218]
    spin: up
  - element: O
    id: O1
    xyz: [0.0, 0.0, 0.914]
    spin: down

elements:
  C:
    symbol: C
    basis: DZ
  O:
    symbol: O
    basis: DZ
```

---

## 3. Transition Metal Cluster: $\text{Ag}_{13}$

A 13-atom silver nanocluster consisting of a central silver atom surrounded by 12 shell atoms in cuboctahedral coordination.

```yaml
title: "Ag13 Nanocluster (Cuboctahedral)"
control:
  spin_unrestricted: true
  max_cycles: 80
  convergence: 1.0e-3
  use_global_fermi: true  # Fermi smearing stabilizes metallic d-band oscillations
  fermi_temperature: 300.0
  separate_atoms_first_cycle: true

geometry:
  - element: Ag
    id: Ag_center
    xyz: [0.0, 0.0, 0.0]
    spin: up
  - element: Ag
    id: Ag_1
    xyz: [3.86, 3.86, 0.0]
    spin: down
  - element: Ag
    id: Ag_2
    xyz: [-3.86, 3.86, 0.0]
    spin: up
  # Additional coordination shell atoms...

elements:
  Ag:
    symbol: Ag
    basis: DZ
```

### Scientific Note:
In transition metal clusters, high density of states near the Fermi level can cause cycle-to-cycle charge sloshing. Setting `use_global_fermi: true` with finite temperature smoothly guides convergence.

---

## 4. Chemisorption: $\text{CO}$ Adsorbed on $\text{Ni}(100)$

This deck models carbon monoxide adsorbed in an on-top site over a 3-layer nickel slab with a virtual perimeter.

```yaml
title: "CO Adsorption on Ni(100) (2x2 Slab, 3 Layers)"
control:
  spin_unrestricted: true
  max_cycles: 120
  convergence: 1.0e-4
  separate_atoms_first_cycle: true

geometry:
  # Adsorbate: Carbon Monoxide
  - element: C
    id: C_ads
    xyz: [0.0, 0.0, 3.50]
    spin: up
  - element: O
    id: O_ads
    xyz: [0.0, 0.0, 5.67]
    spin: down

  # Surface Layer 1 (Active)
  - element: Ni
    id: Ni_top_1
    xyz: [0.0, 0.0, 0.0]
    spin: up
  - element: Ni
    id: Ni_top_2
    xyz: [4.71, 0.0, 0.0]
    spin: down

  # Bottom Layer 3 (Fixed substrate anchor)
  - element: Ni
    id: Ni_bot_1
    xyz: [0.0, 0.0, -6.66]
    fixed: true
    spin: up

  # Virtual boundary perimeter atoms
  - element: Ni
    id: Ni_virt_1
    xyz: [9.42, 0.0, 0.0]
    virtual: true

elements:
  C:
    symbol: C
    basis: DZ
  O:
    symbol: O
    basis: DZ
  Ni:
    symbol: Ni
    basis: DZ
```

### Key Features:
* `fixed: true` on bottom-layer nickel atoms ensures structural stability during surface relaxations.
* `virtual: true` boundary atoms carry projector operators to prevent unphysical edge dipole build-ups.

---

## 5. Symmetry-Reduced $N_2$

Demonstrates point-symmetry reduction along the molecular $z$-axis:

```yaml
title: "Nitrogen Molecule with Inversion Symmetry"
control:
  spin_unrestricted: true
  max_cycles: 100
  convergence: 1.0e-4

geometry:
  - element: N
    id: N1
    xyz: [0.0, 0.0, 1.034]
    spin: up
  - element: N
    id: N2
    xyz: [0.0, 0.0, -1.034]
    equivalent_to: N1
    symmetry_op: -2
    symmetry_axis: 1
    spin: down

elements:
  N:
    symbol: N
    basis: DZ
```

The Fortran computational core uses the `equivalent_to` relationship to evaluate only non-redundant matrix blocks, saving CPU time and memory.
