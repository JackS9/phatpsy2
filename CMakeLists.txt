cmake_minimum_required(VERSION 3.18)
project(PHATPSY VERSION 2.0.0 LANGUAGES Fortran C)

enable_testing()

# Locate BLAS and LAPACK (supports Intel MKL or OpenBLAS)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# 1. Collect all original FORTRAN 66/77 files in src/
file(GLOB ALL_LEGACY_F_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.f")

# 2. List the 10 legacy files replaced by Modern Fortran (.f90) modules
set(REPLACED_LEGACY_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/main.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/contrl.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/atomic.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/aloop.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bloop.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/scf.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/analys.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ewmo.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/fokmat.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/twoint.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/oneint.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/lowdin.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/eispak.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/jacobi.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tred3.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/imtqlv.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tinvit.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/trbak3.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/eig.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/teindx.f"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/excite.f"
)

# Remaining low-level math/quadrature/utility .f files (gensab, genvab, lowdin, eispak, etc.)
set(RETAINED_UTILITY_F_SOURCES ${ALL_LEGACY_F_SOURCES})
list(REMOVE_ITEM RETAINED_UTILITY_F_SOURCES ${REPLACED_LEGACY_FILES})

# Modernized Fortran 2018 module sources (in dependency order)
set(MODERN_F90_MODULES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/lowdin.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/state.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/integrals.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/analys.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bloop.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/aloop.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/atomic.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/scf.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ewmo.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/contrl.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/modern_bridge.f90"
)

# Compiler flags for legacy fixed-form (.f) vs modern free-form (.f90)
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    set_source_files_properties(${ALL_LEGACY_F_SOURCES} PROPERTIES
        COMPILE_FLAGS "-std=legacy -fno-automatic -O2"
    )
    set_source_files_properties(${MODERN_F90_MODULES} "${CMAKE_CURRENT_SOURCE_DIR}/src/cli.f90" PROPERTIES
        COMPILE_FLAGS "-std=gnu -ffree-line-length-none -Wall -Wno-compare-reals -Wno-unused-variable -fcheck=bounds -O2 -fPIC"
    )
    set_source_files_properties(${RETAINED_UTILITY_F_SOURCES} PROPERTIES
        COMPILE_FLAGS "-std=legacy -fno-automatic -O2 -fPIC"
    )
endif()

# ----------------------------------------------------------------------
# Target 1: Baseline Legacy Executable (bin/phatpsy_legacy)
# ----------------------------------------------------------------------
add_executable(phatpsy_legacy ${ALL_LEGACY_F_SOURCES})
target_link_libraries(phatpsy_legacy PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
set_target_properties(phatpsy_legacy PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ----------------------------------------------------------------------
# Target 2: Modernized Core Static Library & CLI Executable (bin/phatpsy)
# ----------------------------------------------------------------------
add_library(phatpsy_core_lib SHARED
    ${MODERN_F90_MODULES}
    ${RETAINED_UTILITY_F_SOURCES}
)
target_link_libraries(phatpsy_core_lib PUBLIC ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

add_executable(phatpsy
    ${MODERN_F90_MODULES}
    ${RETAINED_UTILITY_F_SOURCES}
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cli.f90"
)
target_link_libraries(phatpsy PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
set_target_properties(phatpsy PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ----------------------------------------------------------------------
# Target 3: Python Hybrid Extension (_phatpsy_core via f2py)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
    set(Python3_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
endif()
set(Python3_FIND_VIRTUALENV FIRST)
set(Python3_FIND_STRATEGY LOCATION)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
if(Python3_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE Python3_NumPy_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/.f2py_f2cmap" DESTINATION "${CMAKE_BINARY_DIR}")
    add_custom_target(python_extension ALL
        COMMAND ${Python3_EXECUTABLE} -m numpy.f2py
                -c "${CMAKE_CURRENT_SOURCE_DIR}/src/_phatpsy_core.pyf"
                ${MODERN_F90_MODULES}
                ${RETAINED_UTILITY_F_SOURCES}
                "--f90flags=-std=gnu -ffree-line-length-none -O2 -fPIC"
                "--f77flags=-std=legacy -fno-automatic -O2 -fPIC"
                -llapack -lblas
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS phatpsy_core_lib
        COMMENT "Building Python f2py extension _phatpsy_core..."
    )
    add_custom_command(TARGET python_extension POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different _phatpsy_core*.so "${CMAKE_CURRENT_SOURCE_DIR}/"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Copying _phatpsy_core extension to root workspace..."
    )
endif()