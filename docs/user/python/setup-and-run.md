# Setup & Running Calculations

The PHATPSY 2.0 Python API provides an interactive, typed interface for preparing calculation decks, executing SCF cycles, and manipulating molecular objects in Jupyter notebooks and automated Python scripts.

---

## Notebook Environment Setup

When working inside a Jupyter notebook:

1. **Kernel Selection:** Ensure your Jupyter notebook kernel uses the project virtual environment (`.venv`).
2. **One Calculation Per Process:** The legacy Fortran 2018 core maintains global state per driver run. For multiple independent calculations, instantiate a fresh `PhatpsyDriver` or run separate processes.
3. **Suppressing Fortran Standard Output:** The Fortran engine writes convergence tables directly to C file descriptor `1` (`stdout`). If you prefer completely silent runs in a notebook cell, you can redirect file descriptor 1:

```python
import os
import sys
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = os.dup(1)
        os.dup2(devnull.fileno(), 1)
        try:
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.close(old_stdout)
```

---

## Constructing Calculation Decks

PHATPSY provides high-level builder utilities to synthesize `CalculationDeck` instances quickly:

### 1. Diatomic Molecules (`build_diatomic`)

```python
from phatpsy import build_diatomic

# Build N2 at 2.068 Bohrs with Double-Zeta basis
deck = build_diatomic("N", "N", bond_length=2.068, units="bohr", basis="DZ")
```

### 2. Multi-Atom Molecules (`build_molecule`)

```python
from phatpsy import build_molecule

geometry = [
    ("C", [0.0, 0.0, -1.218]),
    ("O", [0.0, 0.0,  0.914]),
]
deck = build_molecule(geometry, title="Carbon Monoxide", basis="DZ", units="bohr")
```

### 3. Parsing XYZ Strings or Files (`parse_geometry`)

```python
from phatpsy import CalculationDeck, parse_geometry

xyz_block = """
O   0.000000   0.000000   0.1174
H   0.000000   0.757000  -0.4696
H   0.000000  -0.757000  -0.4696
"""
deck = CalculationDeck.from_geometry(xyz_block, basis="DZ", units="angstrom")
```

---

## Inspecting and Modifying Decks

`CalculationDeck` is a typed dataclass that can be inspected and modified programmatically:

```python
# Adjust control parameters
deck.control.convergence = 1.0e-5
deck.control.use_diis = True
deck.control.max_cycles = 80

# Assign alternating spin states (Local Valence Bond seed)
deck.assign_alternating_spin()

# Export modified deck to YAML or JSON
deck.to_yaml("custom_deck.yaml")
deck.to_json("custom_deck.json")
```

### Interactive 3D Preview in Notebooks

Render interactive Three.js views directly inside Jupyter notebook output cells:

```python
deck.show(height=500)
```

---

## Running Calculations with `PhatpsyDriver`

The central execution engine is `PhatpsyDriver`:

```python
from phatpsy import PhatpsyDriver

driver = PhatpsyDriver()

# Run from a CalculationDeck object
res = driver.run_deck(deck)

# Or run directly from a YAML file
res = driver.run_file("examples/n2dz.yaml")
```

### Custom Damping & Acceleration Hooks

You can supply custom damping hooks to the driver to monitor or modify density damping dynamically:

```python
def my_damping_hook(cycle, delta_e, current_damp):
    # Dynamically increase damping if energy oscillates
    if delta_e > 0.0:
        return max(0.1, current_damp * 0.8)
    return current_damp

res = driver.run_deck(deck, custom_damping_hook=my_damping_hook)
```

---

## First-Principles Modern Execution & Subsystem Decomposition

### 1. First-Principles Control (`ControlConfig.first_principles`)

To run under theoretical unit scaling (`PKSCAL = 1.0`) with Pulay DIIS charge acceleration, constrained LAPACK cusp fitting, and dynamic STO radial exponent optimization (retiring empirical manual damping knobs):

```python
from phatpsy.schema import ControlConfig

# Standard first-principles configuration
deck.control = ControlConfig.first_principles(
    pk_scale=1.0,
    use_diis=True,
    use_constrained_fit=True,
    use_dynamic_exponents=True,
    exchange_kernel="pbe",  # "hartree-fock" | "slater" | "becke88" | "pbe"
)
```

### 2. Granular Per-Cycle Diagnostics

Each cycle's physical metrics are automatically recorded in `res.cycle_history`:

```python
for cycle_info in res.cycle_history:
    print(
        f"Cycle {cycle_info['cycle']}: E={cycle_info['total_energy']:.6f} Ha, "
        f"-V/T={cycle_info['virial_ratio']:.4f}, PK_penalty={cycle_info['pk_penalty']:.4e}, "
        f"RMS_Coulomb={cycle_info['rms_coulomb']:.4e}, Charge_transfer={cycle_info['charge_transfer']:.4e}"
    )
```

### 3. Subsystem Energy Decomposition (ONIOM Parity)

For chemisorption surfaces, metal slabs, and complex active-site clusters:

```python
decomp = res.subsystem_decomposition
print(f"Total Energy:       {decomp.energy_total:.6f} Ha")
print(f"Active Cluster:     {decomp.energy_cluster:.6f} Ha")
print(f"Bulk Environment:   {decomp.energy_environment:.6f} Ha")
print(f"Embedding Energy:   {decomp.energy_embedding:.6f} Ha")
print(f"Excess Energy:      {decomp.energy_excess:.6f} Ha")
```
