#!/bin/bash

# Only build if the libraries are missing
if [ ! -f "$PIXI_PROJECT_ROOT/lib/libcpp_mkl_minimizer.so" ]; then
    echo "======================================================"
    echo "Compiling C++ shared libraries for the new environment..."
    echo "======================================================"
    
    # Run CMake compilation
    mkdir -p "$PIXI_PROJECT_ROOT/src/cpp/build"
    cd "$PIXI_PROJECT_ROOT/src/cpp/build"
    cmake ..
    make
    
    echo "Compilation complete!"
fi
