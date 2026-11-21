# syntax=docker/dockerfile:1

# ------------------------------------------------------------------------------
# Stage 1: Builder — compile Fortran 2018 core, CLI executables, and f2py module
# ------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    OMP_STACKSIZE=1G \
    MPLBACKEND=Agg

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gfortran \
        cmake \
        build-essential \
        libopenblas-dev \
        liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtualenv for consistent f2py / runtime NumPy ABI
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --upgrade pip \
    && pip install \
        numpy \
        scipy \
        matplotlib \
        pyyaml \
        pytest \
        meson \
        ninja \
        pypdf \
        setuptools \
        pyscf \
        ase

WORKDIR /app
COPY . /app

# Replicate the canonical CMake Release (-j1) build from make_build_test.sh
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/opt/venv/bin/python3 \
    && cmake --build build -j1

# ------------------------------------------------------------------------------
# Stage 2: Runtime — minimal image with runtime BLAS/LAPACK only (no compiler)
# ------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app" \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    OMP_STACKSIZE=1G \
    MPLBACKEND=Agg

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libopenblas0 \
        liblapack3 \
        libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/phatpsy /app/phatpsy
COPY --from=builder /app/_phatpsy_core*.so /app/
COPY --from=builder /app/_phatpsy_core.pyi /app/_phatpsy_core.pyi
COPY --from=builder /app/build/bin /app/build/bin
COPY --from=builder /app/examples /app/examples
COPY --from=builder /app/data /app/data
COPY --from=builder /app/tests /app/tests
COPY --from=builder /app/tools /app/tools
COPY --from=builder /app/docs /app/docs
COPY --from=builder /app/pytest.ini /app/pytest.ini
COPY --from=builder /app/mkdocs.yml /app/mkdocs.yml
COPY --from=builder /app/README.md /app/README.md
COPY --from=builder /app/ROADMAP.md /app/ROADMAP.md

# Provide `phatpsy` CLI wrapper on PATH
RUN printf '#!/bin/sh\nexec /opt/venv/bin/python3 -m phatpsy "$@"\n' > /usr/local/bin/phatpsy \
    && chmod +x /usr/local/bin/phatpsy

ENTRYPOINT ["phatpsy"]
CMD ["--help"]
