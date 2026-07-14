cmake_minimum_required(VERSION 3.22)
project(mumag_matrixfree_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find active Conda Prefix or default to .pixi env
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX_DIR $ENV{CONDA_PREFIX})
else()
    set(CONDA_PREFIX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.pixi/envs/default")
endif()

# Get dynamic Python version name (e.g. python3.12)
execute_process(
    COMMAND python -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}', end='')"
    OUTPUT_VARIABLE PYTHON_VERSION_NAME
)

# Get dynamic JAX include directory
execute_process(
    COMMAND python -c "import os, jaxlib; print(os.path.join(jaxlib.__path__[0], 'include'), end='')"
    OUTPUT_VARIABLE JAX_INCLUDE_DIR
)

# Set up include directories
include_directories(
    "${CONDA_PREFIX_DIR}/include"
    "${CONDA_PREFIX_DIR}/include/${PYTHON_VERSION_NAME}"
    "${JAX_INCLUDE_DIR}"
)

# Locate libraries
set(MKL_LIBRARIES "${CONDA_PREFIX_DIR}/lib/libmkl_rt.so")
set(PYTHON_LIBRARIES "${CONDA_PREFIX_DIR}/lib/lib${PYTHON_VERSION_NAME}.so")

# Target 1: mkl_ffi_lib
add_library(mkl_ffi_lib SHARED mkl_ffi.cpp)
target_link_libraries(mkl_ffi_lib PRIVATE 
    ${MKL_LIBRARIES}
    ${PYTHON_LIBRARIES}
)
set_target_properties(mkl_ffi_lib PROPERTIES 
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

# Target 2: cpp_mkl_minimizer
find_package(OpenMP REQUIRED)

add_library(cpp_mkl_minimizer SHARED cpp_mkl_minimizer.cpp)
target_compile_options(cpp_mkl_minimizer PRIVATE -O3 -march=native -ffast-math -funroll-loops)
target_link_libraries(cpp_mkl_minimizer PRIVATE 
    mkl_ffi_lib
    ${MKL_LIBRARIES}
    ${PYTHON_LIBRARIES}
    OpenMP::OpenMP_CXX
)
set_target_properties(cpp_mkl_minimizer PROPERTIES 
    PREFIX "lib"
    OUTPUT_NAME "cpp_mkl_minimizer"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
