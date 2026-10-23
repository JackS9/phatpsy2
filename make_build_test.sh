#!/bin/bash
set -e

# Increase stack size limit for quantum chemical array evaluations
ulimit -s unlimited 2>/dev/null || true

# 1. Configure and build legacy baseline, modern CLI, and Python f2py extension
PY_EXE=""
if [ -f ".venv/bin/python3" ]; then
    PY_EXE="-DPython3_EXECUTABLE=$(pwd)/.venv/bin/python3"
fi
cmake -B build -DCMAKE_BUILD_TYPE=Release $PY_EXE
cmake --build build -j1

# 2. Run direct smoke tests on N2
mkdir -p data
# Modern Fortran 2018 CLI output (traditional formatted text stdout)
./build/bin/phatpsy < examples/n2.stdin > data/n2_modern.stdout

# Python driver output (structured JSON output)
if [ -f ".venv/bin/python3" ]; then
    PYTHONPATH=. .venv/bin/python3 -m phatpsy examples/n2.stdin -o data/n2_python_driver.json
else
    PYTHONPATH=. python3 -m phatpsy examples/n2.stdin -o data/n2_python_driver.json
fi

# 3. Run the automated pytest regression suite
if [ -f ".venv/bin/pytest" ]; then
    PYTHONPATH=. .venv/bin/pytest tests/ -v
else
    PYTHONPATH=. pytest tests/ -v
fi