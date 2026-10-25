# User Manual

Welcome to the PHATPSY 2.0 User Manual. This manual provides comprehensive documentation for preparing calculation decks in modern YAML format, running calculations and generating volumetric grids via the command line, and performing interactive quantum chemistry workflows using the Python API in Jupyter notebooks.

---

## Conventions & Units

To ensure physical consistency across all calculations, PHATPSY 2.0 strictly adheres to the following conventions:

### Units of Measurement

* **Energy:** All energies (total energy, kinetic energy, potential energy, orbital eigenvalues) are reported in **atomic units** (Hartrees, $E_h$). $1\text{ Hartree} \approx 27.211386\text{ eV} \approx 627.509\text{ kcal/mol}$.
* **Lengths & Coordinates:** 
    * The internal computational core and Cartesian matrices operate strictly in **Bohrs** ($a_0$). $1\text{ Bohr} = 0.529177210903\text{ \AA}$.
    * In YAML calculation decks, coordinates are assumed to be in **Bohrs** by default unless explicitly converted.
    * When using Python builders such as `build_molecule()`, `build_diatomic()`, or `build_slab()`, you can specify `units="angstrom"` or `units="bohr"`.
* **Virial Ratio:** For stable, bound molecular systems at equilibrium, the virial ratio satisfies:
  $$\eta = -\frac{V}{T} \approx 2.0$$
  PHATPSY 2.0 outputs this metric in all calculation summaries.

### Indexing Conventions

* **Command-Line Interface (CLI):**
    * `--mo <index>`: **1-based index** (e.g., `--mo 1` refers to the lowest-energy molecular orbital).
    * `--ao <index>`: **1-based index** corresponding to the atomic basis function order.
* **Python API:**
    * NumPy arrays (`res["mo_coefficients"]`, `res["eigenvalues"]`) follow standard Python **0-based indexing**.
    * Potential energy scan atom indices (e.g., `coord=("bond", 0, 1)`) are **0-based** indices into the atom list.

### Spin Conventions

PHATPSY 2.0 uses spin-unrestricted Extended Wheeler/Hartree-Fock formulations (UHF / GCHF):

* Spin states per atom can be assigned as `up` ($\alpha$) or `down` ($\beta$).
* The YAML parser accepts `up`, `down`, `alpha`, `beta`, `+1`, and `-1`.
* Alternating local valence bond (LVB) spin configurations can be automatically assigned via `CalculationDeck.assign_alternating_spin()` or CLI flags.

---

## Document Map: "I Want To..."

| Task | Recommended Starting Point | Key Tools |
| :--- | :--- | :--- |
| **Install and test PHATPSY 2.0** | [Installation & Setup](getting-started/install.md) | `./make_build_test.sh`, `pytest` |
| **Run a quick test calculation** | [Five-Minute Tour](getting-started/tour.md) | `python -m phatpsy`, `build_diatomic()` |
| **Understand the core concepts** | [Concepts You Need](getting-started/concepts.md) | Basis sets, LVB spin, embedded atoms |
| **Write a YAML calculation deck** | [YAML Input Overview](yaml/overview.md) | `control:`, `geometry:`, `elements:` |
| **Inspect all YAML keys & defaults** | [YAML Reference](yaml/reference.md) | `schema.py` dataclass reference |
| **Convert legacy `.stdin` or XYZ to YAML** | [Format Conversion](cli/index.md#3-phatpsy-convert) | `phatpsy convert`, `phatpsy build` |
| **Run batch jobs via the CLI** | [Command Line Overview](cli/index.md) | `phatpsy input.yaml -o results.json` |
| **Construct a crystal slab or surface** | [Periodic Systems](python/periodic-systems.md) | `phatpsy supercell`, `build_slab()` |
| **Calculate & visualize 3D orbitals or density** | [3D Volumetric Fields](python/volumetric.md) | `phatpsy cube`, `compute_grid()`, `show_grid()` |
| **Optimize molecular geometry** | [Geometry Optimization](python/restarts-scans-optimization.md#geometry-optimization-optimize_geometry) | `phatpsy opt`, `optimize_geometry()` |
| **Calculate vibrational frequencies & thermochemistry** | [Vibrations & Thermochemistry](python/restarts-scans-optimization.md#harmonic-vibrational-frequency-analysis-thermochemistry) | `phatpsy freq`, `compute_vibrations()` |
| **Interactive work in Jupyter Notebooks** | [Python Notebook Setup](python/setup-and-run.md) | `PhatpsyDriver`, `CalculationDeck` |
| **Diagnose SCF convergence failures** | [Troubleshooting Guide](guide/troubleshooting.md) | Damping, `separate_atoms_first_cycle` |



