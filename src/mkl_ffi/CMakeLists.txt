cmake_minimum_required(VERSION 3.22)
project(mkl_ffi_lib CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure Python paths manually to guarantee linking against the Pixi CPU environment
set(Python_INCLUDE_DIRS "/ceph/home/schrefl/jax_dev/mumag_matrixfree2/.pixi/envs/cpu/include/python3.12")
set(Python_LIBRARIES "/ceph/home/schrefl/jax_dev/mumag_matrixfree2/.pixi/envs/cpu/lib/libpython3.12.so")

# JAX_INCLUDE_DIR will be passed from the command line
include_directories(${JAX_INCLUDE_DIR})
include_directories(${Python_INCLUDE_DIRS})

# Include MKL directory from the default environment
set(MKL_LIB_DIR "/ceph/home/schrefl/jax_dev/mumag_matrixfree2/.pixi/envs/default/lib")
include_directories("/ceph/home/schrefl/jax_dev/mumag_matrixfree2/.pixi/envs/default/include")

# Define target shared library
add_library(mkl_ffi_lib SHARED mkl_ffi.cpp)

# Link against MKL and Python
target_link_libraries(mkl_ffi_lib PRIVATE 
    ${MKL_LIB_DIR}/libmkl_rt.so
    ${Python_LIBRARIES}
)

# Output library directly to the parent directory (src) for easy Python imports
set_target_properties(mkl_ffi_lib PROPERTIES 
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
