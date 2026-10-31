# Results & Scientific Analysis

When an SCF calculation finishes, `PhatpsyDriver.run_deck()` returns a `CalculationState` object. This chapter details how to inspect observables, unpack quantum matrices, and generate publication-quality figures.

---

## The `CalculationState` Object

`CalculationState` is a typed subclass of Python's standard `dict`. It provides direct dictionary access to all computed physical quantities and quantum matrices:

```python
res = driver.run_deck(deck)

# Status and convergence
print(f"Converged   : {res['converged']}")
print(f"Cycles      : {res['cycles']}")
print(f"Final Delta : {res['delta_energy']:.3e} Hartree")

# Physical energies
print(f"Total Energy: {res['total_energy']:.6f} Hartree")
print(f"Kinetic (T) : {res['kinetic_energy']:.6f} Hartree")
print(f"Potential(V): {res['potential_energy']:.6f} Hartree")
print(f"Nuclear Rep : {res['nuclear_repulsion']:.6f} Hartree")
print(f"Virial Ratio: {res['virial_ratio']:.4f} (-V/T)")
```

### Complete Key Reference

| Key | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `converged` | `bool` | scalar | `True` if energy change reached `convergence` threshold. |
| `cycles` | `int` | scalar | Total number of SCF cycles completed. |
| `total_energy` | `float` | scalar | Electronic total energy $E_{\text{tot}}$ in Hartrees. |
| `kinetic_energy`| `float` | scalar | Total electronic kinetic energy $T$ in Hartrees. |
| `potential_energy`|`float`| scalar | Total electronic and nuclear potential energy $V$ in Hartrees. |
| `nuclear_repulsion`|`float`| scalar | Classical nuclear-nuclear repulsion energy $V_{nn}$. |
| `virial_ratio` | `float` | scalar | Virial ratio $-V / T$. Equilibrium systems satisfy $\approx 2.0$. |
| `norbt` | `int` | scalar | Total number of molecular orbitals. |
| `mstot` | `int` | scalar | Total number of atomic basis functions. |
| `ispin` | `int` | scalar | Number of spin channels ($1$ for RHF, $2$ for UHF). |
| `eigenvalues` | `ndarray` | `(norbt, ispin)` | Orbital eigenvalues in Hartrees. |
| `occupations` | `ndarray` | `(norbt, ispin)` | Orbital occupation numbers ($1.0$ or $0.0$). |
| `mo_coefficients`|`ndarray`| `(mstot, norbt, ispin)` | Molecular orbital eigenvectors (LCAO coefficients). |
| `overlap_packed`| `ndarray` | `(mstot*(mstot+1)//2,)` | Atomic orbital overlap matrix $S$ in packed lower-triangular format. |
| `atom_charges` | `ndarray` | `(natoms,)` | Net atomic partial charges. |
| `atom_populations`|`ndarray`| `(natoms,)` | Gross atomic orbital electron populations. |

---

## Orbital Spectrum & HOMO-LUMO Gap

Extracting orbital eigenvalues and computing the frontier orbital gap:

```python
import numpy as np

# Extract alpha spin eigenvalues and occupations
alpha_evals = res["eigenvalues"][:, 0]
alpha_occ = res["occupations"][:, 0]

# Identify occupied and virtual indices
occupied_indices = np.where(alpha_occ > 0.5)[0]
virtual_indices = np.where(alpha_occ < 0.5)[0]

homo_idx = occupied_indices[-1]
lumo_idx = virtual_indices[0]

homo_energy = alpha_evals[homo_idx]
lumo_energy = alpha_evals[lumo_idx]
gap_hartree = lumo_energy - homo_energy
gap_ev = gap_hartree * 27.211386

print(f"HOMO (Orbital {homo_idx + 1}): {homo_energy:.4f} Eh")
print(f"LUMO (Orbital {lumo_idx + 1}): {lumo_energy:.4f} Eh")
print(f"HOMO-LUMO Gap: {gap_hartree:.4f} Eh ({gap_ev:.2f} eV)")
```

---

## Unpacking Quantum Matrices

Because atomic basis overlaps and projected Hamiltonian matrices are symmetric, the Fortran engine stores them in packed lower-triangular format:

$$\text{index}(i, j) = \frac{i(i - 1)}{2} + j \quad (i \ge j)$$

Use `phatpsy.unpack_symmetric` to reconstitute full $N \times N$ symmetric NumPy matrices:

```python
from phatpsy import unpack_symmetric

# Unpack packed lower-triangular overlap matrix into full (mstot, mstot) array
S_matrix = unpack_symmetric(res["overlap_packed"], res["mstot"])

print(f"Overlap matrix shape: {S_matrix.shape}")
print(f"Diagonal (self-overlaps): {np.diag(S_matrix)}")
```

---

## Visualizing Results in Matplotlib

### 1. Overlap Matrix Heatmap

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))
plt.imshow(S_matrix, cmap="viridis", interpolation="nearest")
plt.colorbar(label="Overlap Integral $S_{ij}$")
plt.title("Atomic Orbital Overlap Matrix")
plt.xlabel("Basis Function Index")
plt.ylabel("Basis Function Index")
plt.tight_layout()
plt.show()
```

### 2. MO Energy Level Diagram

```python
plt.figure(figsize=(4, 6))
for i, energy in enumerate(alpha_evals):
    occ = alpha_occ[i]
    color = "tab:blue" if occ > 0.5 else "tab:red"
    label = "Occupied" if (occ > 0.5 and i == 0) else ("Virtual" if (occ < 0.5 and i == lumo_idx) else "")
    plt.hlines(energy, 0.2, 0.8, colors=color, linewidth=2, label=label)

plt.ylabel("Energy (Hartrees)")
plt.title("Molecular Orbital Spectrum")
plt.xlim(0.0, 1.0)
plt.legend(loc="upper right")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
```

---

## Saving and Loading Results

Save full calculation states to JSON or NumPy archive format:

```python
# Save to JSON
res.save("n2_calculation.json")

# Load saved state from JSON
from phatpsy import CalculationState
saved_state = CalculationState.load("n2_calculation.json")
```
