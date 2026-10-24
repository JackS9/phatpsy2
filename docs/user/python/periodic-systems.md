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
* `hcp`: Hexagonal Close-Packed (requires $a$ and $c$)
* `diamond`: Diamond cubic
* `sc`: Simple Cubic

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
