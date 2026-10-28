# PHATPSY 2.0: Projected Hamiltonian Approach to Polyatomic Systems

**PHATPSY** is a quantum chemistry package designed for non-empirical self-consistent field (SCF) calculations on polyatomic molecules, chemisorption complexes, and surface clusters using localized pseudopotentials and projected core/valence basis sets.

Originally developed by **Jack A. Smith** at the Quantum Theory Project (University of Florida) and Union Carbide Corporation, **PHATPSY 2.0** modernizes the original codebase into a high-performance **Fortran 2018** core with a modern **Python API**, **CMake** build system, **Three.js 3D WebGL visualization**, and interactive **Jupyter** workflows.

---

## Key Features

### 1. Modernized Fortran 2018 Core
- **Modular Architecture**: Legacy FORTRAN 66 global common blocks have been migrated into ISO Fortran 2018 modules (`phatpsy_state_mod`, `phatpsy_contrl_mod`, `phatpsy_scf_mod`, etc.).
- **Dynamic Memory Allocation**: Rigid dimension limits (`MXNATM`, `MXNORB`, `MXMSTO`) replaced with dynamic array allocation sized to the molecular deck.
- **BLAS & LAPACK Acceleration**: Modern optimized linear algebra routines integrated for diagonalization and fitting.
- **f2py C/Python Bridge**: Zero-overhead Fortran-Python bindings exposing internal state arrays and computation stages.

### 2. High-Level Python Driver (`phatpsy`)
- **Direct Orchestration**: `PhatpsyDriver` runs calculations in-memory without scratch files or subprocess pipes.
- **Dual Deck Formats**: Full support for both human-readable **YAML** (`.yaml`) decks and legacy 80-column punch-card **Namelist** (`.stdin`) decks.
- **Built-in STO Basis Library**: Automated retrieval of Clementi-Roetti and Thakkar Slater-Type Orbital (STO) basis sets across the periodic table.
- **Localized Valence-Bond (LVB) Alternating-Spin Engine**: Automated bipartite graph solver assigning decoupled electron pairs with alternating spin vectors ($\uparrow$ / $\downarrow$).
- **Structured JSON Export**: Complete extraction of quantum observables (eigenvalues, MO coefficients, density matrices, virial ratios).

### 3. Interactive 3D WebGL Visualizer
- **In-Notebook 3D Viewing**: Call `deck.show()` inside Jupyter notebooks for interactive rotating 3D molecular structures.
- **Visual Encoding**:
  - **Spin Vectors**: Color-coded cyan ($\uparrow$, $\alpha$) and orange ($\downarrow$, $\beta$) arrows.
  - **Virtual Atoms**: Translucent wireframe spheres representing coordination/neighborhood completion centers.
  - **Fixed Atoms**: Metallic cage bounding representations for frozen matrix environments.
  - **Symmetry Elements**: Reflection mirror planes, rotation axes, and generator linkage lines.
- **CLI & Standalone Viewer**: Run `phatpsy view deck.stdin` to launch a full-screen 3D viewer in your browser.

---

## Installation & Setup

### Prerequisites
- **Fortran Compiler**: `gfortran` (GCC 10+)
- **Build Tools**: `cmake` (3.18+) and `ninja` or `make`
- **Libraries**: BLAS and LAPACK (e.g., OpenBLAS or Apple Accelerate)
- **Python**: Python 3.10+ with `numpy`, `scipy`, `matplotlib`, `pyyaml`, `ipython`

### Quick Build
```bash
# Clone the repository
git clone https://github.com/JackS9/phatpsy2.git
cd phatpsy2

# Set up Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib pyyaml ipykernel pytest

# Build Fortran core, CLI, and Python extension
mkdir build && cd build
cmake ..
cmake --build .
cd ..
```

---

## Usage

### Command-Line Interface (CLI)
```bash
# View interactive 3D model in your browser
phatpsy view examples/CO_ni_100.stdin

# View with automatic alternating-spin assignment
phatpsy view -s examples/n2.yaml

# Convert between formats (legacy .stdin to modern YAML)
phatpsy convert examples/CO.stdin -o examples/CO.yaml

# Inspect available basis sets for an element
phatpsy elements C
```

### Python API
```python
from phatpsy import PhatpsyDriver, CalculationDeck, build_molecule

# Build a molecule directly from coordinates
deck = build_molecule(
    geometry="examples/CO.stdin",
    basis="DZ",
    auto_spin_flip=True
)

# Render 3D viewer inside Jupyter Notebook
deck.show()

# Run SCF calculation
driver = PhatpsyDriver()
results = driver.run_deck(deck)

print(f"Total Energy: {results['total_energy']:.6f} Hartree")
print(f"Virial Ratio: {results['virial_ratio']:.6f}")
```

### Interactive Walkthrough
Open and run the complete Jupyter Notebook tutorial:
```bash
jupyter notebook notebooks/phatpsy_driver_walkthrough.ipynb
```

---

## User Manual & Documentation

Comprehensive documentation, including the YAML input reference, CLI guide, Python API reference, and scientific methodology, is built with MkDocs Material:

* **Online Manual:** [https://jacks9.github.io/phatpsy2/](https://jacks9.github.io/phatpsy2/)
* **Build & Serve Locally:**
  ```bash
  pip install -r docs/requirements.txt
  mkdocs serve
  ```
  Then navigate to `http://127.0.0.1:8000/`.

---

## Repository Structure

```
phatpsy2/
├── CMakeLists.txt              # CMake build configuration
├── src/                        # Fortran 2018 modules and legacy mathematical routines
│   ├── phatpsy_state_mod.f90   # Core dynamic state and memory allocation
│   ├── phatpsy_contrl_mod.f90  # Deck input, element loading, and SCF orchestration
│   ├── phatpsy_cli.f90         # Standalone Fortran executable entry point
│   ├── _phatpsy_core.pyf       # f2py extension signature definition
│   └── ...
├── phatpsy/                    # Modern Python package
│   ├── driver.py               # PhatpsyDriver orchestrator
│   ├── schema.py               # CalculationDeck, AtomConfig, SymmetrySpec models
│   ├── builder.py              # Molecular builder and deck synthesis
│   ├── basis.py                # STO basis set parser and database interface
│   ├── spin.py                 # Localized Valence-Bond alternating-spin graph solver
│   ├── visualizer.py           # Three.js 3D WebGL renderer and notebook embedding
│   ├── io.py                   # Legacy punch-card namelist parser and JSON exporter
│   └── cli.py                  # Multi-command CLI tool
├── notebooks/                  # Interactive Jupyter notebooks and tutorials
├── examples/                   # Sample calculation decks (.stdin, .yaml)
├── tests/                      # Automated test suite
└── PhD_Thesis/                 # Theoretical foundation and original documentation
```

---

## Project Roadmap & Goals
Detailed modernization objectives, planned physics improvements, and structural modeling milestones are tracked in [ROADMAP.md](ROADMAP.md).

---

## License & Citation
Developed by **Jack A. Smith**.
Theoretical details and mathematical derivations are available in the accompanying documentation and [PhD Thesis](PhD_Thesis/PHATPSY.pdf).
