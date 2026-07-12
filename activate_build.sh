#!/bin/bash

# Only build if the libraries are missing
if [ ! -f "$PIXI_PROJECT_ROOT/src/libcpp_mkl_minimizer.so" ] || [ ! -f "$PIXI_PROJECT_ROOT/src/mkl_ffi_lib.so" ]; then
    echo "======================================================"
    echo "Compiling C++ shared libraries for the new environment..."
    echo "======================================================"
    
    cd "$PIXI_PROJECT_ROOT/src/mkl_ffi"
    cmake .
    make
    
    cd "$PIXI_PROJECT_ROOT"
    bash compile_cpp_minimizer.sh
    
    echo "Compilation complete!"
fi
