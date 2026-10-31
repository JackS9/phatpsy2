# 3D Volumetric Fields & Visualization

PHATPSY 2.0 features a 3D volumetric field engine capable of evaluating atomic orbitals, molecular orbitals, electron densities, spin densities, deformation densities, and electrostatic potentials (ESP) on regular Cartesian grids.

---

## 1. Computing 3D Scalar Fields

Use `compute_grid()` to evaluate quantum mechanical fields from a converged `CalculationState`:

```python
from phatpsy import compute_grid

# 1. Total Electron Density rho(r) = rho_alpha + rho_beta
density_grid = compute_grid(res, grid_type="density", spacing=0.2, padding=3.0)

# 2. Net Spin Density rho_s(r) = rho_alpha - rho_beta
spin_grid = compute_grid(res, grid_type="spin_density", spacing=0.2, padding=3.0)

# 3. Molecular Orbital (HOMO, 0-based index 6)
homo_grid = compute_grid(res, grid_type="mo", mo_index=6, spin="alpha", spacing=0.2)

# 4. Atomic Orbital (AO 0 on atom 1)
ao_grid = compute_grid(res, grid_type="ao", ao_index=0, spacing=0.2)
```

### Specialized Field Functions

* **`compute_deformation_density(res, ...)`:**
  $$\Delta \rho(\mathbf{r}) = \rho_{\text{molecule}}(\mathbf{r}) - \sum_{A} \rho_{\text{atom } A}(\mathbf{r})$$
  Highlights charge accumulation (bonding) and charge depletion regions upon chemical bond formation.
* **`compute_esp_grid(res, ...)`:**
  Evaluates the Molecular Electrostatic Potential (MEP):
  $$V_{\text{ESP}}(\mathbf{r}) = \sum_{A} \frac{Z_A}{|\mathbf{r} - \mathbf{R}_A|} - \int \frac{\rho(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} d^3\mathbf{r}'$$

---

## 2. The `VolumetricGrid` Class

All grid functions return a `VolumetricGrid` object encapsulating the 3D data array and coordinate system:

```python
print(f"Grid dimensions : {density_grid.dimensions}")
print(f"Grid origin     : {density_grid.origin}")
print(f"Voxel steps     : {density_grid.voxel_steps} Bohrs")
```

### Grid Arithmetic

You can perform arithmetic operations directly on matching grids:

```python
# Spin polarization difference
diff_grid = alpha_density - beta_density

# Scaled grid
scaled_grid = density_grid * 2.0
```

### Numerical Integration (Physical Sanity Check)

To verify physical normalization, call `.integrate()` to compute the volume integral $\int f(\mathbf{r}) d^3\mathbf{r}$:

```python
total_electrons = density_grid.integrate()
print(f"Integrated Electron Count: {total_electrons:.4f} (expected: 14.0 for N2)")
```

---

## 3. Gaussian Cube I/O

`VolumetricGrid` seamlessly reads and writes standard Gaussian Cube files compatible with VMD, PyMOL, Avogadro, and ChimeraX:

```python
# Save grid to Gaussian Cube format
density_grid.to_cube("n2_density.cube")

# Load grid from Cube file, StringIO, or raw string text
from phatpsy import VolumetricGrid

loaded_grid = VolumetricGrid.from_cube("n2_density.cube")
```

---

## 4. Interactive 3D Visualization in Jupyter (`show_grid`)

Visualize isosurfaces directly inside notebook cells using Three.js and real-time marching-tetrahedra polygonization:

### Molecular Orbital (Two-Lobe Phase Coloring)

Molecular orbitals display positive (blue) and negative (orange/red) phase lobes:

```python
from phatpsy import show_grid

show_grid(homo_grid, isovalue=0.08, height=500, title="N2 HOMO Orbital")
```

### Density Colored by Electrostatic Potential (ESP)

Map electrostatic potential values onto the total electron density isosurface using a standard Red-White-Blue color scheme:

```python
esp_grid = compute_esp_grid(res, spacing=0.2, padding=3.0)

# Map ESP values onto the 0.05 e/bohr^3 electron density isosurface
show_grid(
    density_grid,
    isovalue=0.05,
    property_grid=esp_grid,
    height=550,
    title="N2 Density Mapped with ESP",
)
```

---

## 5. Choosing Grid Parameters

* **`spacing` (Grid Resolution):**
  * `0.25` Bohrs ($\approx 0.13\text{ \AA}$): Fast previews and interactive visualization.
  * `0.15`–`0.20` Bohrs ($\approx 0.08$–$0.10\text{ \AA}$): Standard publication quality.
  * `0.10` Bohrs: High-precision integrals and fine orbital nodal planes.
* **`padding` (Boundary Buffer):**
  * `3.0`–`4.0` Bohrs: Standard molecular density and valence orbitals.
  * `5.0`–`6.0` Bohrs: Diffuse Rydberg orbitals and asymptotic ESP decay.
* **`isovalue` (Surface Threshold):**
  * Total electron density: `0.02` to `0.05` $\text{e/bohr}^3$ (approximates the van der Waals molecular surface).
  * Molecular orbitals: `0.05` to `0.10` $\text{a.u.}^{-3/2}$.
  * Deformation density: `0.005` to `0.02` $\text{e/bohr}^3$.
