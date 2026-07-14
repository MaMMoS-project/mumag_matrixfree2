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

# Set up MKL include directory
include_directories(
    "${CONDA_PREFIX_DIR}/include"
)

# Locate MKL library
set(MKL_LIBRARIES "${CONDA_PREFIX_DIR}/lib/libmkl_rt.so")

# Target: cpp_mkl_minimizer
find_package(OpenMP REQUIRED)

add_library(cpp_mkl_minimizer SHARED cpp_mkl_minimizer.cpp)
target_compile_options(cpp_mkl_minimizer PRIVATE -O3 -march=native -ffast-math -funroll-loops)
target_link_libraries(cpp_mkl_minimizer PRIVATE 
    ${MKL_LIBRARIES}
    OpenMP::OpenMP_CXX
)
set_target_properties(cpp_mkl_minimizer PROPERTIES 
    PREFIX "lib"
    OUTPUT_NAME "cpp_mkl_minimizer"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../lib"
)
