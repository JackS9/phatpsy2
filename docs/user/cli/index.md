# Command-Line Suite

The `phatpsy` command-line interface provides high-throughput execution, conversion, surface generation, volumetric evaluation, and geometry optimization directly from the shell.

---

## General Invocation

Because PHATPSY 2.0 is developed as an in-tree Python package, invoke commands using Python's module launcher:

```bash
python -m phatpsy [subcommand] [arguments]
```

Or configure a shell alias:

```bash
alias phatpsy="python -m phatpsy"
```

### Standard Streams & Exit Codes

* **Standard Output (`stdout`):** Streams real-time SCF iteration logs, progress bars, and formatted results summaries.
* **Standard Error (`stderr`):** Reserved for validation warnings, driver diagnostics, and runtime errors.
* **Exit Codes:**
  * `0`: Calculation or command completed successfully.
  * `1`: General error (file not found, syntax error, or unhandled exception).
  * `2`: Convergence failure (SCF exceeded `max_cycles` without reaching convergence criteria).
* **Piping & Stdin:** Most commands accept `-` in place of a file path to read YAML or JSON from standard input.

---

## Subcommands Overview

### 1. `phatpsy` (Default Run)

Runs an SCF calculation on a YAML or JSON calculation deck:

```bash
python -m phatpsy input.yaml -o results.json
```

* **Options:**
  * `input`: Path to input YAML/JSON deck, or `-` for stdin.
  * `-o`, `--output`: Path to write structured results JSON file.
  * `--json-stdout`: Suppress human-readable progress logs and write pure JSON output to `stdout` (ideal for shell pipelines).

---

### 2. `phatpsy build`

Constructs a valid calculation deck from a standard Cartesian XYZ coordinate file:

```bash
python -m phatpsy build molecule.xyz -b DZ -s -o deck.yaml
```

* **Options:**
  * `-b`, `--basis`: Basis set name (`SZ`, `DZ`, or `THAKKAR`). Default: `DZ`.
  * `-u`, `--units`: Input coordinate units (`angstrom` or `bohr`). Default: `angstrom`.
  * `-s`, `--auto-spin`: Automatically assign alternating Local Valence Bond (LVB) spins (`up`/`down`).
  * `-t`, `--title`: Calculation title string.
  * `-o`, `--output`: Output YAML file path.

---

### 3. `phatpsy convert`

Converts between legacy `.stdin` punch-card files, modern YAML decks, and JSON formats:

```bash
python -m phatpsy convert legacy.stdin -o modern.yaml
```

* **Options:**
  * `input`: Input file path (`.stdin`, `.yaml`, `.yml`, or `.json`).
  * `-o`, `--output`: Output file path. Format is auto-detected from file extension.

---

### 4. `phatpsy elements`

Lists available elements in the database or inspects STO basis functions:

```bash
# List all supported elements
python -m phatpsy elements

# Inspect Double-Zeta basis functions and orbital exponents for Nickel
python -m phatpsy elements Ni -b DZ
```

---

### 5. `phatpsy view`

Renders an interactive 3D molecular structure using Three.js:

```bash
python -m phatpsy view deck.yaml --view
```

* **Options:**
  * `-o`, `--output`: Save standalone HTML visualizer file.
  * `--view`: Automatically open the visualizer in your default web browser.
  * `-s`, `--spin`: Color atomic spheres by spin state (red = $\alpha$, blue = $\beta$).
  * `-n`, `--no-bonds`: Disable automated covalent bond inference.

---

### 6. `phatpsy supercell` (alias `slab`)

Constructs periodic crystal structures, surface slabs, and adsorbate complexes:

```bash
# Build a 3-layer (2x2) Ni(100) slab with CO adsorbed on a top site
python -m phatpsy supercell -e Ni -m 1,0,0 -l 3 -r 2,2 \
  --adsorbate CO --site top --height 2.0 \
  -o ni100_co.yaml
```

* **Surface Flags:**
  * `-e`, `--element`: Metal element symbol (default: `Ni`).
  * `-c`, `--crystal`: Crystal lattice type (`fcc`, `bcc`, `hcp`, `diamond`, `sc`).
  * `-a`, `--lattice-constant`: Lattice constant $a$ in Ångströms.
  * `-m`, `--miller`: Miller indices for surface cleavage (e.g., `1,0,0` or `1,1,1`).
  * `-l`, `--layers`: Number of atomic substrate layers (default: 3).
  * `-v`, `--vacuum`: Vacuum spacing in Ångströms (default: 15.0).
  * `-r`, `--repeat`: Supercell lateral repetition (e.g., `2,2` for $2 \times 2$).
  * `-f`, `--fixed-layers`: Number of bottom substrate layers frozen in place.
  * `--virtual-perimeter`: Automatically add virtual perimeter boundary atoms.
* **Adsorbate Flags:**
  * `--adsorbate`: Molecular formula or XYZ file (e.g., `CO`, `NO`, `H2O`).
  * `--site`: Adsorption site (`top`, `bridge`, `hollow`).
  * `--height`: Adsorbate height above surface in Ångströms.
  * `--tilt`: Tilt angle in degrees.
* **Output Flags:**
  * `--format`: Output format (`yaml`, `json`, `poscar`, `cif`, `xyz`).

---

### 7. `phatpsy cube` (aliases `grid`, `volumetric`)

Evaluates 3D volumetric scalar fields on Cartesian grids and exports to Gaussian Cube format or HTML isosurfaces:

```bash
# Total electron density
python -m phatpsy cube deck.yaml --density -o density.cube

# Highest Occupied Molecular Orbital (HOMO, 1-based index)
python -m phatpsy cube deck.yaml --mo 7 -o homo.cube

# Interactive 3D isosurface viewer in browser
python -m phatpsy cube deck.yaml --density --view --isovalue 0.05
```

* **Field Selectors:**
  * `--density`: Total electron charge density $\rho(\mathbf{r}) = \rho_\alpha + \rho_\beta$.
  * `--spin-density`: Net spin density $\rho_s(\mathbf{r}) = \rho_\alpha - \rho_\beta$.
  * `--deformation`: Deformation density $\Delta \rho(\mathbf{r}) = \rho_{\text{mol}} - \sum \rho_{\text{atoms}}$.
  * `--esp`: Electrostatic potential field $V(\mathbf{r})$.
  * `--mo <int>`: Molecular orbital index (1-based).
  * `--ao <int>`: Atomic orbital index (1-based).
* **Grid Parameters:**
  * `--spacing <float>`: Grid resolution in Bohrs (default: `0.2` Bohrs $\approx 0.1\text{ \AA}$).
  * `--padding <float>`: Grid boundary padding beyond atomic coordinates in Bohrs (default: `3.0`).
  * `--isovalue <float>`: Default isosurface threshold value for rendering.

---

### 8. `phatpsy scan` (alias `pes`)

Performs automated 1D or 2D Potential Energy Surface (PES) scans:

```bash
# 1D bond dissociation scan between atom 0 and atom 1 from 1.8 to 3.0 Bohrs
python -m phatpsy scan examples/n2.yaml -b 0,1 -r 1.8:3.0:0.1 -o n2_pes.json -p n2_pes.png
```

* **Coordinate Flags:**
  * `-b`, `--bond`: Bond stretch atom indices `i,j` (0-based).
  * `-a`, `--angle`: Valence angle atom indices `i,j,k` (0-based).
  * `-c`, `--dihedral`: Dihedral angle atom indices `i,j,k,l` (0-based).
  * `-r`, `--range`: Scan range formatted as `start:stop:step` (in Bohrs or degrees).
  * `--bond2`, `--coord2`, `--range2`: Second coordinate specification for 2D PES surfaces.
* **Execution Flags:**
  * `-p`, `--plot`: Save PES energy curve plot to file.
  * `--no-warm-restart`: Disable warm-restart density wave propagation between scan steps.

---

### 9. `phatpsy opt` (alias `optimize`)

Executes automated Cartesian geometry optimization using the BFGS algorithm:

```bash
python -m phatpsy opt input.yaml -o relaxed.yaml -p opt_trajectory.png
```

* **Optimization Flags:**
  * `--max-steps <int>`: Maximum optimization cycles (default: 30).
  * `--grad-tol <float>`: Convergence tolerance on RMS gradient (default: $1.0 \times 10^{-3}$).
  * `--energy-tol <float>`: Convergence tolerance on energy change (default: $1.0 \times 10^{-5}$).
  * `-t`, `--trajectory`: Save all optimization trajectory steps to file.
  * `-p`, `--plot`: Plot energy and gradient convergence profiles.

---

### 10. `phatpsy neb` (aliases `ci-neb`, `path`)

Finds transition states (saddle points) and calculates reaction activation energy barriers using the Climbing-Image Nudged Elastic Band (CI-NEB) method:

```bash
# Calculate minimum energy path and activation barrier between reactant and product
python -m phatpsy neb reactant.yaml product.yaml -n 7 -o pathway.json --xyz pathway.xyz -p barrier.png --view
```

* **Pathway Flags:**
  * `-n`, `--images <int>`: Total number of pathway images including endpoints (default: 7).
  * `-k`, `--spring <float>`: Inter-image spring constant in Hartree/Bohr$^2$ (default: 0.1).
  * `--max-iter <int>`: Maximum NEB optimization iterations (default: 40).
  * `--force-tol <float>`: Force convergence tolerance in Hartree/Bohr (default: 0.003).
  * `--no-climb`: Disable climbing image (runs standard NEB instead of CI-NEB).
  * `--no-warm-restart`: Disable warm-restarting between image SCF calculations.
* **Output Flags:**
  * `-o`, `--output`: Save pathway JSON data (energies, barriers in kcal/mol, and geometries).
  * `--xyz`: Save reaction pathway as a multi-frame standard XYZ trajectory.
  * `-p`, `--plot`: Save reaction profile plot ($E$ vs. reaction coordinate).
  * `--view`: Open an interactive 3D WebGL pathway viewer with playback controls and timeline slider in browser.
