# Installation & Setup

This guide covers setting up the development environment, compiling the Fortran 2018 computational engine, building the Python C-extension module, and verifying the installation.

---

## Prerequisites

Before building PHATPSY 2.0, ensure your environment has the following tools installed:

| Tool | Minimum Version | Notes |
| :--- | :--- | :--- |
| **Fortran Compiler** | `gfortran` $\ge 9.0$ | Supports Fortran 2018 standard (OpenMP optional) |
| **C/C++ Compiler** | `gcc` or `clang` | Required for building Python C-API extensions |
| **CMake** | $\ge 3.20$ | Generates native build systems and targets |
| **Linear Algebra** | OpenBLAS / LAPACK | macOS: Apple Accelerate framework or Homebrew `openblas` |
| **Python** | $\ge 3.10$ | Tested against Python 3.10, 3.11, and 3.12 |

### macOS Installation (Homebrew)

```bash
brew install gcc cmake openblas python@3.12
```

### Ubuntu/Debian Installation

```bash
sudo apt-get update
sudo apt-get install -y gfortran gcc g++ cmake libopenblas-dev python3 python3-venv python3-dev
```

---

## Canonical Build & Verification

PHATPSY 2.0 provides an all-in-one build and verification script: `make_build_test.sh`.

```bash
./make_build_test.sh
```

This script performs the following actions deterministically:

1. Creates or activates the local Python virtual environment (`.venv`).
2. Installs dependencies from `requirements.txt`.
3. Runs CMake to configure and compile the native binaries:
   * The modern Fortran 2018 library and standalone CLI: `bin/phatpsy_modern`.
   * The Python extension module: `_phatpsy_core.*.so`.
4. Executes the validation smoke tests comparing legacy and modern outputs.
5. Runs the complete `pytest` test suite.

A healthy build will end with:

```text
======================= 136 passed in 11.41s =======================
```

---

## Manual Step-by-Step Build

If you prefer building each component manually:

### 1. Set Up the Python Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure and Compile Native Targets

```bash
# Configure CMake in build directory
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# Compile all targets (core library, CLI, and f2py Python extension)
cmake --build build -j
```

Upon completion, the compiled extension `_phatpsy_core` (e.g., `_phatpsy_core.cpython-312-darwin.so`) will reside in the project root or build tree, readily importable by the Python package.

### 3. Verify with Pytest

```bash
PYTHONPATH=. .venv/bin/pytest tests/ -v
```

---

## Running the CLI

Because PHATPSY 2.0 is developed as an integrated in-tree scientific package, execute the CLI using Python's module runner:

```bash
# Activate your virtual environment
source .venv/bin/activate

# Execute the CLI
python -m phatpsy --help
```

You can also create an alias in your shell configuration (`~/.zshrc` or `~/.bashrc`):

```bash
alias phatpsy="python -m phatpsy"
```
