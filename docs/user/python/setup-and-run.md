# Setup & Running Calculations

The PHATPSY 2.0 Python API provides an interactive, typed interface for preparing calculation decks, executing SCF cycles, and manipulating molecular objects in Jupyter notebooks and automated Python scripts.

---

## Notebook Environment Setup

When working inside a Jupyter notebook:

1. **Kernel Selection:** Ensure your Jupyter notebook kernel uses the project virtual environment (`.venv`).
2. **One Calculation Per Process:** The legacy Fortran 2018 core maintains global state per driver run. For multiple independent calculations, instantiate a fresh `PhatpsyDriver` or run separate processes.
3. **Suppressing Fortran Standard Output:** `PhatpsyDriver` is quiet by default (`quiet=True`). If you need to silence raw C-level `stdout` (file descriptor 1) around other calls in a notebook cell, use the built-in `silence_stdout` context manager:

```python
from phatpsy import silence_stdout

with silence_stdout():
    ...  # any calls that write to C stdout
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

### 4. Importing PDB, Gaussian, and VASP POSCAR Structures

PHATPSY natively parses standard chemical and crystal structure files without external dependencies:

```python
from phatpsy import build_molecule, parse_geometry

# 1. Protein Data Bank (.pdb) - parses ATOM/HETATM records and charges
atoms_pdb = parse_geometry("molecule.pdb", units="angstrom")

# 2. Gaussian Input (.gjf, .com) - parses title, charge, multiplicity, and coordinates
atoms_gjf = parse_geometry("calc.gjf")

# 3. VASP POSCAR / CONTCAR (POSCAR, .vasp) - parses lattice and selective dynamics
atoms_poscar = parse_geometry("POSCAR")

# 4. Tripos Mol2 (.mol2) - extracts coordinates, elements, and Sybyl atom types
atoms_mol2 = parse_geometry("ligand.mol2")

# 5. Chemical Markup Language (.cml) - pure-Python zero-dependency XML parser
atoms_cml = parse_geometry("structure.cml")
```

### 5. Atomic Simulation Environment (ASE) & OpenBabel Ecosystem Bridges

PHATPSY provides bidirectional integration with the Atomic Simulation Environment (ASE) and OpenBabel to unlock seamless interoperability with hundreds of chemical and materials formats (CIF, GROMACS `.gro`, MDL `.mol`/`.sdf`, LAMMPS `data.*`, Quantum ESPRESSO, CASTEP, SMILES, InChI, etc.):

```python
from phatpsy import CalculationDeck, from_ase, to_ase, parse_geometry, parse_mol2, to_mol2, parse_cml, to_cml
import ase.io

# 1. Build a CalculationDeck directly from an ASE Atoms object
atoms_ase = ase.io.read("crystal.cif")
deck = CalculationDeck.from_ase(atoms_ase, basis="DZ", spins="up", net_spin=2)

# 2. Export a PHATPSY CalculationDeck to an ASE Atoms object
atoms_out = deck.to_ase()
atoms_out.write("exported.cif")

# 3. Transparently load any ASE-supported format via parse_geometry
atoms_gro = parse_geometry("solvation_box.gro")

# 4. Direct Mol2 and CML round-trip export
mol2_string = deck.to_mol2()
cml_string = deck.to_cml()
```

The bridge transfers:
- **Positions & Lattice:** Automatically scaled between ASE Ångströms and PHATPSY atomic units (Bohrs).
- **Formal Charges:** Initial charges in ASE map directly to `AtomConfig.charge`.
- **Magnetic Moments & Spin:** ASE magnetic moments map to PHATPSY explicit per-atom spins (`"up"` / `"down"`) and flip states.
- **Constraints:** `ase.constraints.FixAtoms` constraints translate directly to `AtomConfig.fixed = True`.

---

## Inspecting and Modifying Decks

`CalculationDeck` is a typed dataclass that can be inspected and modified programmatically:

```python
# Adjust control parameters
deck.control.convergence = 1.0e-5
deck.control.max_cycles = 80

# Assign alternating spin states (Local Valence Bond seed)
deck.assign_alternating_spin()

# Export modified deck to YAML or JSON
deck.to_yaml("custom_deck.yaml")
deck.to_json("custom_deck.json")
```

### Directed Hybrid Starting Orbitals (Valence-Bond Seeding)

In standard quantum calculations, atoms initialize in spherical ground-state configurations ($1s^2 2s^2 2p^2$ for C). In molecular systems, atoms enter chemical bonds in directed valence states ($sp^3, sp^2, sp, dsp^2, d^2sp^3$). PHATPSY provides the `phatpsy.hybrids` module to construct on-the-fly directed-hybrid starting orbitals:

```python
from phatpsy import apply_hybrid_seeds

# 1. Automated seeding from geometry, valency, and any AtomConfig.valence_config
apply_hybrid_seeds(deck)

# 2. Or apply seeds from an explicit bond skeleton
bonds = {(0, 1): 1, (0, 2): 1}  # Atom index pairs -> bond order
apply_hybrid_seeds(deck, bonds=bonds)

# 3. Or assign predefined named hybridizations to individual atoms
deck.geometry[0].hybridization = "sp3"
apply_hybrid_seeds(deck)
```

You can also pass `seed_hybrids=True` directly to `build_molecule` or `build_diatomic` to seed all atoms automatically upon deck construction. When an atom carries an explicit `valence_config`, `build_atom_orbitals_from_valence_config` and `apply_hybrid_seeds` construct oriented hybrid `starting_orbitals` honoring each orbital's spin (`up`, `down`, or `none`), occupation, `connect_to` bond direction, and `plane`/`orientation` specifiers.

The hybrid frame automatically aligns with bonded neighbor directions via the Kabsch algorithm. Supported named types include main-group hybrids (`"sp3"`, `"sp2"`, `"sp"`, `"sp3_pyramidal"`, `"sp3_bent"`, `"pure_p"`) and transition-metal coordination geometries (`"dsp2"`, `"d2sp3"`, `"sd3"`, `"sd"`, `"delta"`).

### Orbital-Level Valence Configuration (`ValenceOrbital` & `propose_valence_config`)

To record explicit orbital-level electron assignments—including mixed spins in frustrated rings (e.g., $\text{O}$ in ethylene oxide), open-shell radicals ($\text{NO}$), dative/coordinate bonds ($\text{NH}_3 \to \text{BH}_3$), and transition-metal crystal-field states—attach a list of `ValenceOrbital` records to `AtomConfig.valence_config` or generate an editable proposal via `propose_valence_config`:

```python
from phatpsy import AtomConfig, ValenceOrbital, propose_valence_config

# Explicit mixed-spin valence configuration on Oxygen in ethylene oxide
o_atom = AtomConfig(
    element="O",
    id="O1",
    xyz=[0.0, 0.0, 0.0],
    valence_config=[
        ValenceOrbital(spin="up", orbital_type="sigma", occupation=1, connect_to="C1"),
        ValenceOrbital(spin="down", orbital_type="sigma", occupation=1, connect_to="C2"),
        ValenceOrbital(spin="none", orbital_type="lone_pair", occupation=2, plane=["C1", "C2"], orientation="in_plane"),
        ValenceOrbital(spin="none", orbital_type="lone_pair", occupation=2, plane=["C1", "C2"], orientation="out_of_plane"),
    ],
)

# Or materialize an automated proposal for review and editing
proposals = propose_valence_config(deck)
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
res = driver.run_file("examples/n2.yaml")
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

### Convergence Reporting & Cycle Limits

A run is reported as converged only when the energy change stays below `control.convergence` for several consecutive cycles. Reaching `control.max_cycles` is never reported as convergence. Inspect the result:

```python
res = driver.run_deck(deck)
print(res["converged"], res["stop_reason"], res["cycles_extended"])
```

`stop_reason` is `"converged"`, or starts with `"not_converged:"` followed by the reason (cycle limit, oscillation, divergence, or too-slow convergence). When the cycle limit is reached, `ScfExtensionPolicy` decides from the energy history whether the run is still contracting fast enough to be worth extending, or should stop and return `converged=False`. Pass your own policy to change the limits, or enable adaptive potential damping:

```python
from phatpsy import PhatpsyDriver
from phatpsy.scf_control import ScfExtensionPolicy

policy = ScfExtensionPolicy(max_extension_cycles=100, adaptive_damping=True)
driver = PhatpsyDriver(scf_policy=policy)
```

For systems whose SCF does not settle (some heteronuclear diatomics), `control.use_grid_fit` is enabled by default (`True`), and `control.atomic_smearing` (off by default, e.g. `1.0e-3`) can be enabled to smooth near-degenerate atomic orbital occupations (see the YAML key reference).

---

## First-Principles Modern Execution & Subsystem Decomposition

### 1. First-Principles Control (`ControlConfig`)

By default, `ControlConfig()` runs under theoretical unit scaling (`pk_scale = 1.0`, `use_pseudometric = True`) with convex radial-grid Coulomb potential fitting (`use_grid_fit = True`) and tight convergence (`convergence = 1.0e-6`, `max_cycles = 200`):

```python
from phatpsy.schema import ControlConfig

# Standard first-principles configuration (with optional DFT exchange-correlation)
deck.control = ControlConfig(
    exchange_kernel="pbe",  # "hartree-fock" | "slater" | "becke88" | "pbe"
    correlation_kernel="pbe",
    use_dispersion=True,
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
print(f"Total Energy:       {decomp.total_energy:.6f} Ha")
print(f"Active Cluster:     {decomp.energy_cluster:.6f} Ha")
print(f"Bulk Environment:   {decomp.energy_environment:.6f} Ha")
print(f"Embedding Energy:   {decomp.energy_embedding:.6f} Ha")
print(f"Excess Energy:      {decomp.energy_excess:.6f} Ha")
```
