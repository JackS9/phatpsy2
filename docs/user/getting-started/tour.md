# Five-Minute Tour: $N_2$ Three Ways

This quick-start tutorial demonstrates how to perform a quantum self-consistent field (SCF) calculation on the nitrogen molecule ($N_2$) using three different approaches:

1. **Direct YAML Deck Execution via CLI**
2. **Automated Conversion from XYZ Coordinate File**
3. **Interactive Python API in a Jupyter Notebook**

---

## Method 1: Direct YAML Execution

The simplest way to run PHATPSY 2.0 is from an existing YAML calculation deck.

### 1. Inspect the Deck (`examples/n2.yaml`)

```yaml
title: "Nitrogen Molecule (N2) - Double-Zeta Basis"
control:
  spin_unrestricted: true
  max_cycles: 100
  convergence: 1.0e-4
  charge_damp: 0.5
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
  - symbol: N
    basis: DZ
```

### 2. Run the Calculation

```bash
python -m phatpsy examples/n2.yaml -o n2_results.json
```

### 3. Review the Output

PHATPSY streams convergence progress to the terminal and saves full structured output to `n2_results.json`:

```text
==================================================
PHATPSY 2.0: Projected Hamiltonian SCF
==================================================
Title: Nitrogen Molecule (N2) - Double-Zeta Basis
Atoms: 2 | Basis functions: 10
Cycle   1: Energy = -108.921345 Eh | Delta = 1.089e+02
Cycle   5: Energy = -109.891204 Eh | Delta = 1.240e-03
Cycle  10: Energy = -109.893041 Eh | Delta = 4.210e-05
--------------------------------------------------
SCF Converged in 10 cycles.
Total Energy    : -109.893041 Hartree
Virial Ratio    : 2.0441
```

---

## Method 2: Build from XYZ

If you have molecular coordinates in standard XYZ format, construct a valid calculation deck using `phatpsy build`:

### 1. Create an XYZ File (`n2.xyz`)

```text
2
Nitrogen molecule in Angstroms
N  0.0  0.0   0.5472
N  0.0  0.0  -0.5472
```

### 2. Convert and Run

```bash
# Generate the YAML deck with Double-Zeta basis and auto-assigned alternating spin
python -m phatpsy build n2.xyz -b DZ -s -o n2_built.yaml

# Execute the generated deck
python -m phatpsy n2_built.yaml -o n2_built_res.json
```

---

## Method 3: Interactive Python API

The Python API allows seamless calculation execution, data manipulation, and visualization directly inside Jupyter notebooks.

```python
from phatpsy import PhatpsyDriver, build_diatomic, compute_grid, show_grid

# 1. Synthesize diatomic deck (equilibrium bond distance = 2.068 Bohrs)
deck = build_diatomic("N", "N", bond_length=2.068, units="bohr", basis="DZ")

# 2. Preview 3D molecular structure
deck.show()

# 3. Instantiate driver and run SCF calculation
driver = PhatpsyDriver()
res = driver.run_deck(deck)

# 4. Inspect physical observables
print(f"Total Energy   : {res['total_energy']:.6f} Hartrees")
print(f"Kinetic Energy : {res['kinetic_energy']:.6f} Hartrees")
print(f"Potential E    : {res['potential_energy']:.6f} Hartrees")
print(f"Virial Ratio   : {res['virial_ratio']:.4f} (-V/T ≈ 2.0)")

# 5. Compute and render 3D total electron density isosurface
density_grid = compute_grid(res, grid_type="density", spacing=0.2, padding=3.0)
show_grid(density_grid, isovalue=0.05, title="N2 Total Electron Density")
```

---

## Exporting Volumetric Grids

Once the calculation has run, you can extract 3D scalar fields to Gaussian Cube format or view them interactively:

```bash
# Export the Highest Occupied Molecular Orbital (HOMO, orbital 7) to Cube format
python -m phatpsy cube examples/n2.yaml --mo 7 -o n2_homo.cube

# Launch interactive 3D browser isosurface viewer
python -m phatpsy cube examples/n2.yaml --mo 7 --view --isovalue 0.08
```
