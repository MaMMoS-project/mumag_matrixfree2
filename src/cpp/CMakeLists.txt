cmake_minimum_required(VERSION 3.22)
project(mumag_matrixfree_cpp CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    message(FATAL_ERROR "The tommos native extension supports only Linux x86-64")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MKL_LINK "sdl")
set(MKL_INTERFACE "lp64")
set(MKL_THREADING "gnu_thread")

find_package(Python REQUIRED COMPONENTS Interpreter)
execute_process(
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/find_mkl.py" --cmake-dir
    RESULT_VARIABLE MKL_DISCOVERY_RESULT
    OUTPUT_VARIABLE MKL_CMAKE_DIR
    ERROR_VARIABLE MKL_DISCOVERY_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT MKL_DISCOVERY_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not locate distribution-owned oneMKL CMake metadata: ${MKL_DISCOVERY_ERROR}")
endif()
get_filename_component(MKL_ROOT_FROM_METADATA "${MKL_CMAKE_DIR}/../../.." ABSOLUTE)
set(MKL_ROOT "${MKL_ROOT_FROM_METADATA}" CACHE PATH
    "oneMKL root derived from mkl-devel distribution metadata" FORCE)

find_package(MKL CONFIG REQUIRED PATHS "${MKL_CMAKE_DIR}" NO_DEFAULT_PATH)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(cpp_mkl_minimizer SHARED cpp_mkl_minimizer.cpp)
target_include_directories(cpp_mkl_minimizer PRIVATE ${MKL_INCLUDE})
target_compile_options(cpp_mkl_minimizer PRIVATE -O3)
target_link_options(cpp_mkl_minimizer PRIVATE "LINKER:--enable-new-dtags")
target_link_libraries(cpp_mkl_minimizer PRIVATE
    MKL::mkl_rt
    OpenMP::OpenMP_CXX
)

set_target_properties(cpp_mkl_minimizer PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "cpp_mkl_minimizer"
    SKIP_BUILD_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE
    INSTALL_RPATH "$ORIGIN/../../../.."
)

install(
    TARGETS cpp_mkl_minimizer
    LIBRARY DESTINATION tommos/_native
)
