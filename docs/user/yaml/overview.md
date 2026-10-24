# YAML Input Format

PHATPSY 2.0 uses structured YAML (or JSON) calculation decks to define chemical systems, basis sets, and numerical control parameters. This replaces legacy FORTRAN 66 fixed-format punch cards and Namelist inputs with modern, validated schemas.

---

## Top-Level Structure

A valid PHATPSY calculation deck contains four primary keys:

```yaml
title: "Molecule or System Description"

control:
  # Convergence and numerical iteration options
  spin_unrestricted: true
  max_cycles: 100
  convergence: 1.0e-4

geometry:
  # List of atomic centers and coordinates
  - element: N
    id: N1
    xyz: [0.0, 0.0, 1.034]
    spin: up

elements:
  # Basis set assignment per chemical element
  - symbol: N
    basis: DZ
```

---

## The `control:` Block

The `control:` section configures the SCF convergence criteria, numerical damping, acceleration, and debugging flags:

| Key | Type | Default | Scientific Meaning & Tuning |
| :--- | :--- | :--- | :--- |
| `spin_unrestricted` | `bool` | `true` | Enables Unrestricted Hartree-Fock (UHF). Solves separate $\alpha$ and $\beta$ Fock equations. |
| `max_cycles` | `int` | `100` | Maximum number of SCF iterations before aborting. |
| `convergence` | `float` | `1.0e-4` | Convergence threshold on root-mean-square energy change ($\Delta E_h$). |
| `time_limit` | `float` | `360.0` | Maximum allowed calculation wall time in seconds. |
| `charge_damp` | `float` | `0.5` | Fractional mixing factor for atomic charge densities ($0 < \lambda \le 1$). Lower values ($\approx 0.2$) stabilize oscillating calculations. |
| `charge_accel` | `float` | `1.0` | Charge acceleration factor across consecutive cycles. |
| `potential_damp`| `float` | `0.5` | Damping factor applied to potential matrix updates. |
| `potential_accel`| `float`| `1.0` | Acceleration factor applied to potential matrix updates. |
| `pk_scale` | `float` | `0.0` | Scale factor for projected exchange fitting potentials. |
| `separate_atoms_first_cycle` | `bool` | `true` | Computes isolated atom states during cycle 1 to form a high-quality initial molecular density guess. Highly recommended. |
| `debug` | `bool` | `false` | Enables verbose matrix logging in the core Fortran engine. |
| `trace` | `bool` | `false` | Prints cycle-by-cycle execution timestamps and diagnostic traces. |
| `extra_namelist`| `dict` | `{}` | Key-value escape hatch passed directly to legacy Namelist variables if necessary. |

---

## The `geometry:` Block

The `geometry:` key contains a list of atomic specifications. Each entry defines a single atomic center:

### Identity & Position

* **`element`** (`str`, required): Chemical symbol (e.g., `H`, `C`, `Ni`, `Ag`).
* **`id`** (`str`, optional): Unique label for the center (e.g., `Ni_surf_1`, `N_top`). If omitted, defaults to `<element><index>`.
* **`xyz`** (`list[float]`, required): Cartesian coordinates $[x, y, z]$ in **Bohrs** ($a_0$).

### Electronic Properties

* **`charge`** (`float`, default `0.0`): Net formal charge on the atom.
* **`spin`** (`str` or `int`, optional): Initial spin projection for the atom:
  * Accepts: `up`, `down`, `alpha`, `beta`, `+1`, `-1`, or `None`.
  * Used by the local valence bond (LVB) starting projector.

### Special Center Roles

* **`fixed`** (`bool`, default `false`): If `true`, freezes this atom's position during geometry optimizations and potential energy scans.
* **`virtual`** (`bool`, default `false`): If `true`, marks this center as a ghost/virtual projector site (carries model potential projectors without contributing nuclear charges or electrons).

### Symmetry Relations

To exploit molecular point symmetry, an atom can reference another atom:

* **`equivalent_to`** (`str`): `id` of the parent atom.
* **`symmetry_op`** (`int`): Symmetry transformation operator ($\pm 1$).
* **`symmetry_axis`** (`int`): Coordinate axis index for reflection/inversion ($1 = z, 2 = y, 3 = x$).

### Warm Starts & Embedded States

* **`starting_orbitals`** (`list[dict]`): Initial orbital populations and coefficients for warm restarting a calculation without recomputing from scratch.
* **`embedded_state`** (`int`): Index of pre-calculated embedded substrate atom state (used in surface slab calculations).

---

## The `elements:` Block

The `elements:` block maps each chemical element present in the geometry to a basis set or explicit basis functions:

```yaml
elements:
  - symbol: N
    basis: DZ
  - symbol: H
    basis: SZ
```

### Supported Basis Sets

* `SZ`: Single-Zeta Slater-Type Orbitals.
* `DZ`: Double-Zeta Slater-Type Orbitals.
* `THAKKAR`: Thakkar polarized basis sets.

### Explicit Custom Basis Functions

For specialized calculations, basis functions can be defined explicitly:

```yaml
elements:
  - symbol: C
    description: "Custom polarized STO basis"
    functions:
      - n: 1
        l: 0
        zeta: 5.6727   # 1s
      - n: 2
        l: 0
        zeta: 1.6083   # 2s
      - n: 2
        l: 1
        zeta: 1.5679   # 2p
```

---

## Validation & Common Errors

Calculation decks are validated on load by `phatpsy.schema.CalculationDeck`:

* **Coordinate Dimensions:** Every `xyz` entry must contain exactly 3 numerical coordinates.
* **Element Recognition:** Elements must correspond to valid entries in the periodic table and have valid basis assignments.
* **Symmetry References:** If `equivalent_to: AtomName` is specified, an atom with `id: AtomName` must exist earlier in the `geometry:` list.

---

## Format Conversion

You can convert legacy `.stdin` punch-card files to modern YAML using `phatpsy convert`:

```bash
python -m phatpsy convert examples/n2.stdin -o n2.yaml
```

You can also convert between YAML and JSON:

```bash
python -m phatpsy convert n2.yaml -o n2.json
```
