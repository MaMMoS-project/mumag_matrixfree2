cmake_minimum_required(VERSION 3.10)
project(MicromagneticsCpp)

set(CMAKE_CXX_STANDARD 17)

# Include paths
include_directories(
    /usr/include/eigen3
    deps/vexcl
    deps/amgcl
    ${CMAKE_CURRENT_SOURCE_DIR}/src_cpp
)

# OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

# Boost (header-only is enough for AMGCL, but some components might need libs)
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# cnpy
find_package(ZLIB REQUIRED)
link_libraries(cnpy z)

# Source files
set(SOURCES
    src_cpp/fem_utils.cpp
    src_cpp/poisson_solve.cpp
    src_cpp/energy_kernels.cpp
)

# Poisson test
add_executable(test_poisson_convergence src_cpp/test_poisson_convergence.cpp ${SOURCES})
target_link_libraries(test_poisson_convergence ${OpenCL_LIBRARIES} ${Boost_LIBRARIES} cnpy z)

# Energy test
add_executable(test_energy src_cpp/test_energy.cpp ${SOURCES})
target_link_libraries(test_energy ${OpenCL_LIBRARIES} ${Boost_LIBRARIES} cnpy z)

