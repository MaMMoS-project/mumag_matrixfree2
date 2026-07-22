#!/bin/bash
set -e

# Use Slurm job ID to create a unique local folder, or fallback to PID
if [ -n "$SLURM_JOB_ID" ]; then
    LOCAL_BUILD_DIR="/tmp/mumag_build_${SLURM_JOB_ID}"
    # Slurm users build output directly into the isolated tmp folder
    export MUMAG_LIB_OUT="$LOCAL_BUILD_DIR"
else
    LOCAL_BUILD_DIR="/tmp/mumag_build_$$"
    # Desktop/head-node users still build the source safely in /tmp to avoid 
    # cluttering the repository, but the final compiled .so library 
    # is placed permanently in the global lib/ folder!
    if [ -n "$PIXI_PROJECT_ROOT" ]; then
        export MUMAG_LIB_OUT="$PIXI_PROJECT_ROOT/lib"
        mkdir -p "$MUMAG_LIB_OUT"
    fi
fi

echo "Compiling locally in $LOCAL_BUILD_DIR..."

# Create local directory and copy CMake files and source
mkdir -p "$LOCAL_BUILD_DIR/src/cpp"
# Copy from the original source location to the local temp directory
cp -r src/cpp/* "$LOCAL_BUILD_DIR/src/cpp/"

cd "$LOCAL_BUILD_DIR/src/cpp"
rm -rf build && mkdir build && cd build

# We need to tell CMake where to find the conda prefix manually because CMAKE_CURRENT_SOURCE_DIR changed
if [ -z "$CONDA_PREFIX" ]; then
    # Pixi sets PIXI_PROJECT_ROOT automatically. We use that to reliably find the env.
    if [ -n "$PIXI_PROJECT_ROOT" ]; then
        export CONDA_PREFIX="$PIXI_PROJECT_ROOT/.pixi/envs/default"
    else
        echo "Error: Neither CONDA_PREFIX nor PIXI_PROJECT_ROOT are set."
        exit 1
    fi
fi

cmake ..
make -j

echo "Successfully compiled library to $LOCAL_BUILD_DIR/libcpp_mkl_minimizer.so"
