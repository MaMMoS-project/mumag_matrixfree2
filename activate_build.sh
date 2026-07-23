#!/bin/bash

# Only build if the libraries are missing
if [ ! -f "$PIXI_PROJECT_ROOT/lib/libcpp_mkl_minimizer.so" ]; then
    if [ -n "$SLURM_JOB_ID" ]; then
        echo "SLURM_JOB_ID detected: Skipping global auto-build."
        echo "Please ensure you have 'pixi run compile' in your .slurm script!"
    else
        echo "======================================================"
        echo "Compiling C++ shared libraries for the new environment..."
        echo "======================================================"
        
        bash "$PIXI_PROJECT_ROOT/compile_local.sh"
        
        echo "Compilation complete!"
    fi
fi
