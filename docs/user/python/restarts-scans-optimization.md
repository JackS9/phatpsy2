# Restarts, Scans & Geometry Optimization

This chapter explains how to perform warm-restart calculations, run 1D and 2D potential energy surface (PES) scans, configure internal coordinates, and optimize molecular geometries.

---

## Warm Restarts & Checkpoints

Re-converging an SCF calculation from scratch for slightly perturbed geometries is inefficient. PHATPSY supports **warm restarts**, which seed the initial density from a previous `CalculationState`:

```python
# 1. Initial calculation
res1 = driver.run_deck(deck)

# 2. Modify geometry (e.g. stretch bond)
new_geometry = [
    ("N", [0.0, 0.0,  1.100]),
    ("N", [0.0, 0.0, -1.100]),
]

# 3. Warm restart reusing converged molecular orbitals
res2 = driver.restart(res1, new_geometry=new_geometry)
```

You can also generate an independent restart YAML deck containing the converged orbital coefficients:

```python
restart_deck = res1.create_restart_deck()
restart_deck.to_yaml("warm_restart.yaml")
```

---

## Internal Coordinates & Z-Matrices

PHATPSY provides internal coordinate primitives for manipulating geometry without raw Cartesian math:

```python
from phatpsy import InternalCoordinates, BondCoordinate, AngleCoordinate, DihedralCoordinate

# Define internal coordinates
bond = BondCoordinate(atom1=0, atom2=1)
angle = AngleCoordinate(atom1=0, atom2=1, atom3=2)
dihedral = DihedralCoordinate(atom1=0, atom2=1, atom3=2, atom4=3)

coords = InternalCoordinates([bond, angle, dihedral])
```

You can also construct systems using `ZMatrix`:

```python
from phatpsy import ZMatrix, ZMatrixEntry

zmat = ZMatrix([
    ZMatrixEntry("O"),
    ZMatrixEntry("H", bond_to=0, bond_length=0.957),
    ZMatrixEntry("H", bond_to=0, bond_length=0.957, angle_to=1, angle_val=104.5),
])

# Convert to Cartesian CalculationDeck
deck = zmat.to_deck(title="Water Molecule", basis="DZ")
```

---

## Potential Energy Surface (PES) Scans

### 1D Scans (`scan_1d`)

Scan a bond, valence angle, or dihedral angle along a discrete grid with automatic warm restarting:

```python
import numpy as np
import matplotlib.pyplot as plt
from phatpsy import scan_1d

# Define grid of bond distances in Bohrs
r_grid = np.linspace(1.8, 3.2, 15)

# Coordinate specification: ("bond", atom_i, atom_j)
bond_coord = ("bond", 0, 1)

# Execute scan with warm restarts enabled
pes_result = scan_1d(deck, coordinate=bond_coord, values=r_grid, warm_restart=True)

# Plot potential energy curve
plt.figure(figsize=(7, 4.5))
plt.plot(pes_result["values"], pes_result["energies"], marker="o", color="navy")
plt.xlabel("Bond Distance $R$ (Bohrs)")
plt.ylabel("Total Energy (Hartrees)")
plt.title("N2 Bond Dissociation Energy Curve")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
```

### 2D Scans (`scan_2d`)

For studying coupling between multiple degrees of freedom (e.g. stretching two bonds simultaneously):

```python
from phatpsy import scan_2d

r1_grid = np.linspace(1.9, 2.5, 7)
r2_grid = np.linspace(1.9, 2.5, 7)

pes2d = scan_2d(
    deck,
    coord1=("bond", 0, 1),
    values1=r1_grid,
    coord2=("bond", 1, 2),
    values2=r2_grid,
    warm_restart=True,
)
```

---

## Geometry Optimization (`optimize_geometry`)

Automated Cartesian geometry optimization uses the BFGS quasi-Newton algorithm with finite-difference numerical gradients:

```python
from phatpsy import optimize_geometry

opt_result = optimize_geometry(
    deck,
    max_steps=30,
    grad_tol=1.0e-3,     # RMS gradient threshold
    energy_tol=1.0e-5,   # Energy change threshold
    finite_diff_step=0.005,
)

print(f"Optimization Converged: {opt_result.converged}")
print(f"Total Steps Taken     : {len(opt_result.steps)}")
print(f"Final Energy          : {opt_result.final_energy:.6f} Hartrees")

# Relaxed geometry as a new CalculationDeck
relaxed_deck = opt_result.final_deck
relaxed_deck.to_yaml("relaxed_geometry.yaml")
```

### Inspecting Optimization Trajectory

The `OptimizationResult` stores intermediate steps with energies, gradients, and Cartesian coordinates:

```python
for step in opt_result.steps:
    print(f"Step {step.step_num:2d} | Energy: {step.energy:.6f} Eh | Grad Norm: {step.gradient_norm:.3e}")
```

---

## Transition State Searching (Climbing-Image NEB)

To locate reaction transition states (first-order saddle points) and calculate activation energy barriers, PHATPSY 2.0 provides the **Climbing-Image Nudged Elastic Band (CI-NEB)** method via `neb_search()`:

```python
from phatpsy import neb_search, build_diatomic

# 1. Define reactant and product states
deck_reactant = build_diatomic("N", "N", bond_length=2.068, basis="DZ")
deck_product = build_diatomic("N", "N", bond_length=2.800, basis="DZ")

# 2. Run CI-NEB calculation across 7 pathway images
neb_res = neb_search(
    initial_deck=deck_reactant,
    final_deck=deck_product,
    num_images=7,
    k_spring=0.1,        # Spring constant in Hartree/Bohr^2
    max_iterations=40,
    force_tol=0.003,     # Force threshold in Hartree/Bohr
    climb=True,          # Inverts tangent force on highest image to locate exact saddle point
    warm_restart=True,
)

print(f"NEB Converged      : {neb_res.converged}")
print(f"Iterations         : {neb_res.num_iterations}")
print(f"Forward Barrier    : {neb_res.barrier_forward_kcal:.2f} kcal/mol ({neb_res.barrier_forward:.6f} Eh)")
print(f"Reverse Barrier    : {neb_res.barrier_reverse_kcal:.2f} kcal/mol")
print(f"Reaction Energy    : {neb_res.reaction_energy_kcal:.2f} kcal/mol")
print(f"Transition State   : Image {neb_res.transition_state_index}")

# Transition state geometry as a new CalculationDeck
ts_deck = neb_res.transition_state_deck
ts_deck.to_yaml("transition_state.yaml")
```

### Reaction Profile Plotting

Plot the potential energy reaction profile with automatic cubic-spline interpolation:

```python
# Plot in kcal/mol (default) or Hartree
neb_res.plot_pathway("reaction_barrier.png")
```

### Interactive 3D Pathway Playback in Notebooks

Scrub through reaction pathway images and watch animated bond rearrangements directly inside Jupyter cells:

```python
neb_res.show(height=550)
```

You can also export the reaction trajectory to multi-frame XYZ format:

```python
neb_res.save_xyz("reaction_pathway.xyz")
```
