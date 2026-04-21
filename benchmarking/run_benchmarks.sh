#!/bin/bash
set -e

# Configuration
ENV_NAME="mfree-mumag-gpu"
MESH_FILE="cube_60nm_shell.npz"


PY_OUT="py_bench.txt"
CPP_OUT="cpp_bench.txt"

echo "=== Micromagnetics Poisson Benchmark Driver ==="

# 1. Run Python Benchmark
echo "Running Python (JAX) benchmark..."
micromamba run -n $ENV_NAME python3 ../src/test_poisson_convergence.py | tee $PY_OUT

# 2. Build C++ Project
echo "Building C++ project..."
mkdir -p build
cd build
cmake ..
make -j$(nproc)
cd ..

# 3. Run C++ Benchmark
echo "Running C++ (VexCL/AMGCL) benchmark..."
# Ensure we use the mesh created by Python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/deps/cnpy/build
./build/test_poisson_convergence $MESH_FILE | tee $CPP_OUT

# 4. Generate Report
echo "Generating benchmark report..."
python3 generate_report.py $PY_OUT $CPP_OUT

echo "Done. See BENCHMARK_REPORT.md"
