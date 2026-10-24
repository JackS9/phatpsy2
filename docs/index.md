# PHATPSY 2.0

**PHATPSY 2.0** (*Projected Hamiltonian Approach to Polyatomic Systems*) modernizes the 1970s quantum chemistry package developed by Dr. Jack A. Smith (Ph.D. Dissertation, University of Florida, 1980) into a modern, high-performance computational platform.

The system combines:

* **Fortran 2018 computational core** powered by standard LAPACK linear algebra and dynamic in-memory state management.
* **Modern YAML & JSON calculation decks**, replacing punch-card Namelist inputs with human-readable, schema-validated inputs.
* **Comprehensive Python API**, designed from the ground up for interactive research, automated potential energy scans, cluster/slab generation, and Jupyter notebook workflows.
* **Command-line suite (`phatpsy`)** for high-throughput batch execution, format conversion, supercell construction, and volumetric field generation.
* **Interactive 3D visualization** utilizing Three.js and marching-tetrahedra isosurfaces in browser windows and embedded notebook cells.

---

## Quick Navigation

<div class="grid cards" markdown>

-   :material-book-open-page-variant:{ .lg .middle } **[User Manual](user/index.md)**

    ---

    Overview, conventions, Getting Started guide, and direct task-oriented recipes.

-   :material-code-json:{ .lg .middle } **[YAML Input Format](user/yaml/overview.md)**

    ---

    Structured input reference, geometry configuration, symmetry, basis sets, and annotated examples.

-   :material-console:{ .lg .middle } **[Command Line](user/cli/index.md)**

    ---

    Complete CLI command guide: execution, slab building, volumetric cube extraction, and geometry optimization.

-   :material-language-python:{ .lg .middle } **[Python API](user/python/setup-and-run.md)**

    ---

    Driver execution, results inspection, PES scans, internal coordinates, slabs, and 3D volumetric fields.

-   :material-atom:{ .lg .middle } **[Science & Theory](science/methodology_overview.md)**

    ---

    Theoretical foundation: projected Hamiltonians, effective core potentials, exchange approximations, and equation mappings.

-   :material-tools:{ .lg .middle } **[System Architecture](development/architecture.md)**

    ---

    Fortran 2018 module structure, Python C-extension boundary, validation standards, and continuous integration.

</div>

---

## Quick Example: Diatomic Nitrogen ($N_2$)

=== "Command Line"

    ```bash
    # Run from a modern YAML deck
    python -m phatpsy examples/n2.yaml -o n2_results.json

    # Generate a 3D total electron density grid and export to Gaussian Cube
    python -m phatpsy cube examples/n2.yaml --density -o n2_density.cube

    # Launch interactive Three.js isosurface visualizer
    python -m phatpsy cube examples/n2.yaml --density --view --isovalue 0.05
    ```

=== "Python API"

    ```python
    from phatpsy import PhatpsyDriver, build_diatomic, compute_grid, show_grid

    # Build an N2 molecule with Double-Zeta (DZ) basis
    deck = build_diatomic("N", "N", bond_length=2.068, units="bohr", basis="DZ")

    # Run SCF calculation
    driver = PhatpsyDriver()
    res = driver.run_deck(deck)

    print(f"Total Energy: {res['total_energy']:.6f} Hartrees")
    print(f"Virial Ratio: {res['virial_ratio']:.4f}")

    # Compute and visualize total electron density isosurface
    grid = compute_grid(res, grid_type="density", spacing=0.2, padding=3.0)
    show_grid(grid, isovalue=0.05, title="N2 Total Density")
    ```

=== "YAML Deck"

    ```yaml
    title: "Nitrogen Molecule (N2) - Double-Zeta Basis"
    control:
      spin_unrestricted: true
      max_cycles: 100
      convergence: 1.0e-4
      charge_damp: 0.5
      separate_atoms_first_cycle: true

    geometry:
      - element: N
        id: N1
        xyz: [0.0, 0.0, 1.034]
        spin: up
      - element: N
        id: N2
        xyz: [0.0, 0.0, -1.034]
        spin: down

    elements:
      - symbol: N
        basis: DZ
    ```

---

## Installation Summary

PHATPSY 2.0 requires a Fortran compiler (`gfortran` $\ge 9$), a C compiler, CMake ($\ge 3.20$), OpenBLAS/LAPACK, and Python $\ge 3.10$.

```bash
# Clone the repository
git clone https://github.com/JackS9/phatpsy2.git
cd phatpsy2

# Run the canonical build and verification suite
./make_build_test.sh
```

For complete details, see [Installation & Setup](user/getting-started/install.md).
