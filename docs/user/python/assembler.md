# Stepwise Valence-Bond Molecular Assembler

The **Valence-Bond Molecular Assembler** (`phatpsy.assembler`) provides an intuitive, mathematically grounded engine to build complete 3D molecular structures one atom at a time.

Instead of hand-crafting Cartesian coordinates or Z-matrices, you assemble molecules by replacing open valence placeholder slots on hybridized atoms with new real atoms, with automatic covalent bond lengths, hybrid frame orientation, conformational dihedral control (staggered vs. eclipsed), and direct quantum starting orbital seeding.

---

## 1. Quickstart: Building Molecules Stepwise

### Methane ($CH_4$)

```python
from phatpsy import HybridAssembler

# Initialize central sp3 carbon (4 tetrahedral placeholder arms)
asm = HybridAssembler(root="C", hybridization="sp3")

# Cap all open placeholder arms with hydrogens
asm.cap_all(element="H")

# Export directly to a validated CalculationDeck
deck = asm.to_deck(title="Methane", basis="DZ")
```

### Ethanol ($CH_3CH_2OH$)

```python
from phatpsy import HybridAssembler

# 1. Root methyl carbon
asm = HybridAssembler(root="C", hybridization="sp3")

# 2. Attach methylene carbon to slot 0 (staggered conformation)
c2 = asm.attach("C1", slot=0, element="C", hybridization="sp3", conformation="staggered")

# 3. Attach bent oxygen to C2 slot 1
o1 = asm.attach("C2", slot=1, element="O", hybridization="sp3_bent", conformation="staggered")

# 4. Cap all remaining open slots with hydrogens (yields C2H6O)
asm.cap_all("H")

# 5. Export to CalculationDeck ready for SCF
deck = asm.to_deck(title="Ethanol", basis="DZ")
```

---

## 2. Multiple Bonds & Conformations

### Ethylene ($C_2H_4$) with Coplanar $\pi$-Orbitals

When attaching $sp^2$ hybridized atoms with `bond_order=2`, the assembler automatically aligns out-of-plane $\pi$ orbital vectors, enforcing planar geometry:

```python
from phatpsy import HybridAssembler

asm = HybridAssembler(root="C", hybridization="sp2")
asm.attach("C1", slot=0, element="C", hybridization="sp2", bond_order=2)
asm.cap_all("H")

deck = asm.to_deck(title="Ethylene", basis="DZ")
```

### Acetylene ($C_2H_2$) Linear Triple Bond

```python
from phatpsy import HybridAssembler

asm = HybridAssembler(root="C", hybridization="sp")
asm.attach("C1", slot=0, element="C", hybridization="sp", bond_order=3)
asm.cap_all("H")

deck = asm.to_deck(title="Acetylene", basis="DZ")
```

### Transition-Metal Delta Bond (Quadruple Bond)

The assembler supports $d$-orbital face-to-face overlap ($\delta$ bonding) for transition metal centers:

```python
from phatpsy import HybridAssembler

asm = HybridAssembler(root="Cr", hybridization="delta")
asm.attach("Cr1", slot=0, element="Cr", hybridization="delta", bond_order=4)

deck = asm.to_deck(title="Dichromium Quadruple Bond", basis="DZ", apply_seeds=False)
```

---

## 3. Automated Lone-Pair Handling

For electronegative elements (N, O, halogens), the assembler auto-reserves non-bonding valence slots:

- **$sp^3$ Oxygen / $sp^3\text{\_bent}$:** Two arms for bonding, two lone pairs.
- **$sp^3$ Nitrogen / $sp^3\text{\_pyramidal}$:** Three arms for bonding, one lone pair.
- **Halogens (F, Cl, Br, I):** One arm for bonding, three lone pairs.

```python
from phatpsy import HybridAssembler

# Water: automatically leaves 2 open slots for H
asm_water = HybridAssembler(root="O", hybridization="sp3_bent")
asm_water.cap_all("H")

# Ammonia: automatically leaves 3 open slots for H
asm_nh3 = HybridAssembler(root="N", hybridization="sp3_pyramidal")
asm_nh3.cap_all("H")
```

---

## 4. Interactive 3D "Click-to-Build" WebGL UI

Render an interactive Three.js viewport directly in Jupyter notebooks or standalone in the browser:

```python
# Inside a Jupyter notebook:
asm.show(height=600)

# Or save to standalone HTML file:
asm.visualize("builder.html", auto_open=True)
```

The interactive 3D viewer displays:
- Real atoms as solid CPK spheres.
- Open placeholder slots as pulsating translucent cyan spheres labeled `(+)`.
- Clicking any `(+)` sphere opens an interactive modal to choose elements, hybridizations, and conformational dihedrals with live updates.
- One-click buttons to "Cap All with H" and download the resulting YAML calculation deck or XYZ file.

---

## 5. Command-Line Interface (`phatpsy assemble`)

You can assemble molecules directly from the command line:

```bash
# Generate methane calculation deck
phatpsy assemble -r C --hybridization sp3 -o methane.yaml

# Launch interactive 3D builder in your browser
phatpsy assemble -r C --hybridization sp3 --interactive
```

