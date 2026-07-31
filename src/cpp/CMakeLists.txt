cmake_minimum_required(VERSION 3.22)
project(mumag_matrixfree_cpp CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    message(FATAL_ERROR "The tommos native extension supports only Linux x86-64")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import sys; print(sys.prefix)"
    RESULT_VARIABLE PYTHON_PREFIX_RESULT
    OUTPUT_VARIABLE PYTHON_PREFIX
    ERROR_VARIABLE PYTHON_PREFIX_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT PYTHON_PREFIX_RESULT EQUAL 0 OR PYTHON_PREFIX STREQUAL "")
    message(FATAL_ERROR "Could not obtain the build Python prefix: ${PYTHON_PREFIX_ERROR}")
endif()

set(MKL_ROOT "${PYTHON_PREFIX}" CACHE PATH
    "oneMKL root in the build Python prefix" FORCE)
set(MKL_DIR "${PYTHON_PREFIX}/lib/cmake/mkl" CACHE PATH
    "oneMKL CMake configuration in the build Python prefix" FORCE)
set(MKL_LINK "sdl")
set(MKL_INTERFACE "lp64")

find_package(MKL CONFIG REQUIRED NO_DEFAULT_PATH)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(cpp_mkl_minimizer SHARED cpp_mkl_minimizer.cpp)
target_include_directories(cpp_mkl_minimizer PRIVATE "${MKL_INCLUDE}")
target_compile_options(cpp_mkl_minimizer PRIVATE -O3)
target_link_libraries(cpp_mkl_minimizer PRIVATE
    MKL::mkl_rt
    OpenMP::OpenMP_CXX
)

set_target_properties(cpp_mkl_minimizer PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "cpp_mkl_minimizer"
    SKIP_BUILD_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "$ORIGIN/../../../.."
)

install(
    TARGETS cpp_mkl_minimizer
    LIBRARY DESTINATION tommos/_native
)
