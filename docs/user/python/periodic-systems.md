# Periodic Systems & Surface Slabs

PHATPSY 2.0 includes a comprehensive crystal and surface generation toolkit designed for modeling chemisorption, metal clusters, and thin films.

---

## 1. Bulk Crystals (`PeriodicCell.bulk`)

Generate standard crystallographic unit cells for pure elements:

```python
from phatpsy import PeriodicCell

# Generate an FCC Nickel unit cell (a = 3.52 Angstroms)
ni_bulk = PeriodicCell.bulk("Ni", crystal_structure="fcc", a=3.52, units="angstrom")

print(f"Lattice Vectors:\n{ni_bulk.lattice.matrix}")
print(f"Number of atoms in cell: {len(ni_bulk.atoms)}")
```

### Supported Crystal Structures

* `fcc`: Face-Centered Cubic
* `bcc`: Body-Centered Cubic
* `diamond`: Diamond cubic
* `sc`: Simple Cubic

(`PeriodicCell.bulk` supports these four; for hexagonal lattices use `Lattice.hexagonal(a, c)` directly.)

---

## 2. Supercells (`build_supercell`)

Expand a unit cell into a periodic supercell:

```python
# Create a 3x3x3 supercell with automatic alternating LVB spin
supercell = ni_bulk.build_supercell(repeat=(3, 3, 3), auto_spin=True, center=True)
print(f"Total atoms in supercell: {len(supercell.atoms)}")
```

---

## 3. Surface Slabs (`build_slab`)

The `build_slab()` function automates crystallographic surface cleavage, layer stacking, and vacuum slab construction:

```python
from phatpsy import build_slab

# Build a 3-layer (2x2) Ni(100) slab with 12 Angstrom vacuum
slab = build_slab(
    element="Ni",
    crystal_structure="fcc",
    a=3.52,
    miller_index=(1, 0, 0),
    layers=3,
    vacuum=12.0,
    repeat=(2, 2),
    fixed_layers=1,       # Freeze bottom layer
)
```

### Key Parameters

* `miller_index`: Surface plane indices (e.g., `(1, 0, 0)`, `(1, 1, 0)`, `(1, 1, 1)`).
* `layers`: Number of atomic planes along the surface normal.
* `vacuum`: Vacuum spacing in Ångströms to isolate slabs.
* `repeat`: Lateral supercell repetition $(n_x, n_y)$.
* `fixed_layers`: Number of subsurface layers frozen with `fixed: true`.

---

## 4. Attaching Adsorbates (`attach_adsorbate`)

Attach molecules to high-symmetry surface binding sites:

```python
from phatpsy import attach_adsorbate

# Attach carbon monoxide (CO) to an on-top site at 2.0 Angstrom height
complex = attach_adsorbate(
    slab=slab,
    adsorbate="CO",
    site_type="top",      # "top", "bridge", or "hollow"
    site_index=0,
    height=2.0,
    tilt_deg=0.0,
)
```

### Adsorption Sites

* **`top`:** Directly above a surface metal atom.
* **`bridge`:** Centered between two adjacent surface metal atoms.
* **`hollow`:** Centered in the 3-fold or 4-fold hollow coordination site between surface atoms.

---

## 5. Exporting to Chemistry & Crystallography Formats

Slabs and periodic cells can be exported to multiple formats:

```python
# Convert directly to a PHATPSY CalculationDeck
deck = complex.to_deck(title="CO on Ni(100)", basis="DZ")

# Export to standard crystallographic and visualization formats
complex.to_xyz("system.xyz")
complex.to_poscar("POSCAR")
complex.to_cif("system.cif")
```

---

## 6. Chemisorption Analysis (Phase 9)

The `phatpsy.chemisorption` module provides transition-metal chemisorption energetics and spectroscopy for CO/NO on Ni(100)-type surfaces:

| Function | Purpose |
| :--- | :--- |
| `evaluate_chemisorption_energy` | Adsorption energy $E_{ads} = E(\text{slab+ads}) - E(\text{slab}) - E(\text{ads})$ |
| `project_c4v_dband` | C4v crystal-field d-band projection (irreps $a_1$:$d_{z^2}$, $e$:{$d_{xz}$,$d_{yz}$}, $b_1$:$d_{x^2-y^2}$, $b_2$:$d_{xy}$), spin-resolved |
| `scan_lateral_chemisorption_pes` | 2D lateral PES scan across top/bridge/hollow sites |
| `compute_chemisorption_vibrational_shift` | Adsorbate vibrational frequency red-shift on adsorption |
| `compute_surface_core_level_shifts` | Surface core-level shifts (metal and adsorbate) |

```python
from phatpsy.chemisorption import evaluate_chemisorption_energy

result = evaluate_chemisorption_energy(
    slab_with_ads="ni100_co.yaml",   # slab + adsorbate deck
    clean_slab="ni100_clean.yaml",   # bare slab deck
    isolated_ads="co.yaml",          # isolated molecule deck
    site_name="top",
)
print(f"E_ads = {result.energy_kcal_mol:.1f} kcal/mol")
```

**CLI:** `phatpsy chemisorption` runs the same analyses from the command line:

```bash
phatpsy chemisorption ni100_co.yaml --site top --c4v --atom Ni1 -o chemi.json
phatpsy chemisorption ni100_co.yaml --sites --lattice 4.71    # lateral PES scan
phatpsy chemisorption ni100_co.yaml --vib --scls              # red-shift + SCLS
phatpsy chemisorption ni100_co.yaml --all -o chemi_full.json  # everything
```
