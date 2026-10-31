#!/bin/bash
set -e

# Increase stack size limit for quantum chemical array evaluations
ulimit -s $(ulimit -H -s 2>/dev/null || echo 65536) 2>/dev/null || ulimit -s unlimited 2>/dev/null || true

# Prevent multi-threaded BLAS/LAPACK runtime stack overflow in OpenBLAS / OMP
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OMP_STACKSIZE=1G

# 1. Configure and build legacy baseline, modern CLI, and Python f2py extension
PY_EXE=""
if [ -f ".venv/bin/python3" ]; then
    PY_EXE="-DPython3_EXECUTABLE=$(pwd)/.venv/bin/python3"
fi
cmake -B build -DCMAKE_BUILD_TYPE=Release $PY_EXE
cmake --build build -j1

# 2. Run modern Python driver smoke test on N2 YAML
mkdir -p data
if [ -f ".venv/bin/python3" ]; then
    PYTHONPATH=. .venv/bin/python3 -m phatpsy examples/n2.yaml -o data/n2_python_driver.json
else
    PYTHONPATH=. python3 -m phatpsy examples/n2.yaml -o data/n2_python_driver.json
fi

# 3. Run the automated pytest regression suite
if [ -f ".venv/bin/pytest" ]; then
    PYTHONPATH=. .venv/bin/pytest tests/ -v -s --tb=short
else
    PYTHONPATH=. pytest tests/ -v -s --tb=short
fi