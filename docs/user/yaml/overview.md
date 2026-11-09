1# YAML Input Format

PHATPSY 2.0 uses structured YAML (or JSON) calculation decks to define chemical systems, basis sets, and numerical control parameters. This replaces legacy FORTRAN 66 fixed-format punch cards and Namelist inputs with modern, validated schemas.

---

## Top-Level Structure

A valid PHATPSY calculation deck contains four primary keys:

```yaml
title: "Molecule or System Description"
net_spin: 1

control:
  # Convergence and numerical iteration options
  spin_unrestricted: true
  max_cycles: 200
  convergence: 1.0e-6

geometry:
  # List of atomic centers and coordinates
  - element: N
    id: N1
    xyz: [0.0, 0.0, 1.034]
    spin: up

elements:
  # Basis set assignment per chemical element
  N:
    symbol: N
    basis: DZ
```

---

## The `control:` Block

The `control:` section configures the SCF convergence criteria, numerical damping, acceleration, and debugging flags:

| Key | Type | Default | Scientific Meaning & Tuning |
| :--- | :--- | :--- | :--- |
| `spin_unrestricted` | `bool` | `true` | Enables Unrestricted Hartree-Fock (UHF). Solves separate $\alpha$ and $\beta$ Fock equations. Setting it to `false` runs a restricted closed-shell calculation: per-atom `spin` labels are optional and ignored. |
| `max_cycles` | `int` | `200` | Maximum number of SCF iterations before aborting. |
| `convergence` | `float` | `1.0e-6` | Convergence threshold on root-mean-square energy change ($\Delta E_h$). |
| `use_grid_fit` | `bool` | `true` | Fits each atom's Coulomb potential on a radial grid as a convex mixture of screening exponentials (weights in $[0,1]$ summing to 1), so charge conservation and the nuclear cusp hold exactly and the fitted coefficients cannot cancel. See `docs/science/model_potentials.md`, Section 5. |
| `atomic_smearing` | `float` | `0.0` | Fermi-Dirac width in Hartree for the atomic orbital occupations. `0` enables automatic early-cycle annealing and gap-floor smearing on unseeded atoms; a positive value sets a fixed width. |
| `use_global_fermi` | `bool` | `false` | Enables finite-temperature Fermi-Dirac orbital occupation smearing for metallic clusters. |
| `fermi_temperature` | `float` | `null` | Electron temperature (Kelvin) for Fermi smearing when enabled. |
| `exchange_kernel` | `str` | `"hartree-fock"` | Exchange-correlation kernel (`hartree-fock`, `slater`, `becke88`/`b88`, `pbe`). |
| `exchange_alpha` | `float` | `0.6667` | Slater $X\alpha$ exchange scaling parameter ($\alpha = 2/3$). |
| `correlation_kernel` | `str`/`null` | `null` | Correlation kernel: `pw92`/`lda` (LDA) or `pbe`/`gga` (GGA). Auto-set to `pbe` when `exchange_kernel: pbe`. `null` disables correlation. |
| `use_dispersion` | `bool` | `false` | Enables Grimme D3 dispersion correction (Becke-Johnson damping). |
| `dispersion_model` | `str` | `"d3bj"` | Dispersion model variant (`d3bj`). |
| `default_basis` | `str`/`null` | `null` | Cascading basis default for all atoms (e.g. `DZ`, `DZP`); individual atoms can override via their own `basis:` key. |
| `separate_atoms_first_cycle` | `bool` | `true` | Computes isolated atom states during cycle 1 to form a high-quality initial molecular density guess. Highly recommended. |
| `debug` | `bool` | `false` | Enables verbose matrix logging in the core Fortran engine. |
| `trace` | `bool` | `false` | Prints cycle-by-cycle execution timestamps and diagnostic traces. |
| `extra_namelist`| `dict` | `{}` | Key-value escape hatch passed directly to core engine variables if necessary. |

!!! note "Retired Control Parameters"
    The legacy parameters `time_limit`, `charge_damp`, `charge_accel`, `potential_damp`, `potential_accel`, `pk_scale`, `use_pseudometric`, `pk_energy`, `use_constrained_fit`, `use_diis`, and `use_dynamic_exponents` should **not** appear in modern YAML input files. They are handled internally by default radial-grid convex potential fitting (`use_grid_fit: true`), first-principles projection scale (`pk_scale: 1.0`, `use_pseudometric: true`), and LAPACK solvers.

!!! tip "Theoretical First-Principles Defaults & Hydrogen Core Exemption"
    In PHATPSY 2.0, the Phillips-Kleinman operator operates under strict first-principles unit scaling (`pk_scale: 1.0`, `use_pseudometric: true`) across all multi-electron elements. Because the energy-weighted projector (thesis Eq. III-24) dynamically suppresses valence-like contributions and Hydrogen's sole occupied orbital ($1s$) is valence-like, Hydrogen is governed by the physical **Hydrogen Core-Exemption Invariant**: Hydrogen centers neither receive nor contribute Phillips-Kleinman projection ($V_{\text{PK}} = 0$), and maintain an unperturbed overlap metric ($M = \mathbf{S}$) — keeping H's bonding $1s$ out of the unweighted pseudometric prevents unphysical valence over-projection. Standard input decks require no empirical tuning or scaling parameters.

---

## The `geometry:` Block

The `geometry:` key contains a list of atomic specifications. Each entry defines a single atomic center:

### Identity & Position

* **`element`** (`str`, required): Chemical symbol (e.g., `H`, `C`, `Ni`, `Ag`).
* **`id`** (`str`, optional): Unique label for the center (e.g., `Ni_surf_1`, `N_top`). If omitted, defaults to `<element><index>`.
* **`xyz`** (`list[float]`, required): Cartesian coordinates $[x, y, z]$ in **Bohrs** ($a_0$).

### Electronic Properties

* **`charge`** (`float`, default `0.0`): Net formal charge on the atom (must be integer-valued when `valence_config` is specified).
* **`spin`** (`str` or `int`, optional): Initial spin projection for the atom:
  * Accepts: `up`, `down`, `alpha`, `beta`, `+1`, `-1`, or `None`.
  * Used by the local valence bond (LVB) starting projector.
* **`hybridization`** (`str`, optional): Predefined directed-hybrid valence state type. Overrides automatic skeletal hybridization perception. Options include:
  * Main-group ($s + p$): `"s"`, `"sp"`, `"sp2"`, `"sp2_bent"`, `"sp3"`, `"sp3_pyramidal"`, `"sp3_bent"`, `"pure_p"`.
  * Transition-metal ($s + p + d$): `"dsp2"` (square planar), `"d2sp3"` (octahedral), `"dsp3"` (trigonal bipyramidal), `"sd3"`, `"sd"`, `"delta"` ($\delta$-bonding pair).
  The local hybrid frame automatically rotates via the Kabsch algorithm to maximize directional overlap with bonded neighbors.
* **`valence_config`** (`list[dict]`, optional): Explicit orbital-level valence electron and orientation assignment (`ValenceOrbital` records). Each entry supports:
  * `spin`: `"up"`, `"down"`, or `"none"` (consistent with `occupation`: `1` $\iff$ `"up"`/`"down"`; `0` or `2` $\iff$ `"none"`).
  * `orbital_type`: `"sigma"`, `"pi"`, `"delta"`, or `"lone_pair"` (orbital role/symmetry; emptiness is encoded via `occupation: 0`, and `"lone_pair"` requires `occupation: 2`).
  * `occupation`: `0` (empty), `1` (singly occupied), or `2` (doubly occupied).
  * `connect_to` (`str`, optional): Target atom `id` this orbital points toward. A dative/coordinate bond is encoded as a matched pair: a donor `lone_pair` (`occupation: 2`, `connect_to: <acceptor>`) and a vacant acceptor orbital on the partner (`occupation: 0`, `connect_to: <donor>`).
  * `plane` (`list[str]`, optional): 1–3 atom `id` labels defining a reference plane, paired with `orientation: "in_plane"` (default) or `"out_of_plane"` (requires $\ge 2$ plane labels).

### Special Center Roles

* **`fixed`** (`bool`, default `false`): If `true`, freezes this atom in two senses: (1) its position is excluded from geometry optimizations, MD, and PES scans; (2) its electronic state is frozen during SCF — the atom emits `QFIXED=T`, so its orbitals, eigenvalues, and model potential are computed once and never iterated (`qdone = QFIXED` in `src/scf.f90`). Ideal for anchoring bottom layers of a crystal slab or freezing buffer atoms.
* **`virtual`** (`bool`, default `false`): If `true`, marks this center as a ghost/virtual projector site (carries model potential projectors without contributing nuclear charges or electrons).

### Symmetry Relations

To exploit molecular point symmetry, an atom can reference another atom:

* **`equivalent_to`** (`str`): `id` of the parent atom.
* **`symmetry_op`** (`int`): Symmetry transformation operator: `1`/`0` = identity,
  `-1` = reflection through the plane perpendicular to the axis, `-2` = inversion
  through the origin, `N > 1` = $N$-fold proper rotation ($C_N$) about the axis,
  `-N < -2` = $N$-fold roto-reflection ($S_N$). String aliases are also accepted
  (`"reflection"`, `"inversion"`, `"c4"`, `"s6"`, ...).
* **`symmetry_axis`** (`int` or `str`): Axis for the symmetry operation —
  `1` = $z$, `10` = $y$, `100` = $x$ (legacy Fortran `IXYZ` encoding; the strings
  `"x"`, `"y"`, `"z"` are also accepted).

### Warm Starts & Embedded States

* **`starting_orbitals`** (`list[dict]`): Initial orbital populations and coefficients for warm restarting a calculation without recomputing from scratch.
* **`embedded_state`** (`int`): Index of pre-calculated embedded substrate atom state (used in surface slab calculations).

---

## The `elements:` Block

The `elements:` block maps each chemical element present in the geometry to a basis set or explicit basis functions:

```yaml
elements:
  N:
    symbol: N
    basis: DZ
  H:
    symbol: H
    basis: SZ
```

### Supported Basis Sets

| Name | Description | Elements |
| :--- | :--- | :--- |
| `SZ` | Single-Zeta Slater-Type Orbitals (Clementi–Roetti) | Z = 2–54 |
| `DZ` | Double-Zeta Slater-Type Orbitals (Clementi–Roetti) | Z = 2–54 |
| `SZP` | SZ plus polarization functions (2p on H/He, 3d on Li–Ar, 4p/4d on K–Kr, 5p/5d on Rb–Xe) | Z = 1–54 |
| `DZP` | DZ plus polarization functions (default; 2p on H/He, 3d on Li–Ar, 4p/4d on K–Kr, 5p/5d on Rb–Xe) | Z = 1–54 |
| `THAKKAR` | Thakkar even-tempered STO sets | Z = 1–103 |
| `ADF-DZ` | ADF ZORA-contracted STO, double-zeta | Z = 1–118 |
| `ADF-TZP` | ADF ZORA-contracted STO, triple-zeta + 1 polarization | Z = 1–118 |
| `ADF-TZ2P` | ADF ZORA-contracted STO, triple-zeta + 2 polarization | Z = 1–118 |
| `ADF-QZ4P` | ADF ZORA-contracted STO, quadruple-zeta + 4 polarization | Z = 1–118 |

The ADF tiers (van Lenthe & Baerends, *J. Comput. Chem.* **24**, 1142 (2003)) provide a systematic hierarchy for convergence studies: `DZ < TZP < TZ2P < QZ4P`. H/He polarization exponents are literature values (H 2p ζ = 1.25).

### Cascading Basis Hierarchy

`control.default_basis` sets the basis for all atoms; any atom can override it with its own `basis:` key. This enables mixed-basis calculations (e.g. a large basis on the active site, a compact basis on the environment):

```yaml
control:
  default_basis: DZ        # fallback for all atoms
net_spin: 0
geometry:
  - element: Ni
    xyz: [0.0, 0.0, 0.0]
    spin: up
    basis: ADF-TZP          # override: larger basis on the active center
  - element: Ni
    xyz: [0.0, 0.0, 4.0]
    spin: down
    # inherits default_basis: DZ
```

### Explicit Custom Basis Functions

For specialized calculations, basis functions can be defined explicitly:

```yaml
elements:
  C:
    symbol: C
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

### Selecting the Model-Potential Exponents

Each atom's spherically averaged Coulomb potential is represented as a $1/r$ term plus a sum of screening exponentials (see `docs/science/model_potentials.md`). By default every element uses the built-in exponent set. To choose a different set for an element, list the exponentials in `potential_exponents`:

```yaml
elements:
  C:
    symbol: C
    potential_exponents: [0.45, 0.9, 1.8, 3.6, 7.2]
  O:
    symbol: O
    basis: DZ        # keeps the built-in exponent set
```

* The point-charge $1/r$ term is always present and must **not** be listed; `potential_exponents` holds only the exponentials.
* The listed values are multiplied by the engine's Z-based length scale, $1.12956\,Z^{1/3}$, so one set can be reused across elements. To give absolute exponents (in inverse bohr), also set `potential_exponent_scale: 1.0`; any other positive value replaces the Z-based scale by that multiplier.
* Between 2 and 19 distinct, positive values are accepted. Starting weights are uniform; every coefficient is refit from the first SCF cycle.
* The selection applies to the named element only (all of its basis variants), and it never alters the built-in library.

!!! note "Which set to use"
    The default set is unchanged. How to choose an exponent set is not settled: in studies on first-row atoms the default fits the atomic screening function noticeably worse than a Z-scaled even-tempered set, but a better fit did not by itself make the SCF converge (see `ROADMAP.md`). Report the exponent set together with any result that uses a non-default one.

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
