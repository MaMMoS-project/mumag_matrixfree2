#!/bin/bash
set -e

# Configuration
L=60
H=2
LAYERS=4
K=2.0
H0_AIR=3.0
OUT_DIR="bench_remanence"
ENV_NAME="mfree-mumag-cpu"

echo "=== Micromagnetics Performance Benchmark (Remanence State) ==="
echo "Core Mesh: ${L}x${L}x${L} nm cube, h=${H} nm"
echo "Airbox: ${LAYERS} layers, K=${K}, h0_air=${H0_AIR} nm"
echo "Environment: ${ENV_NAME}"

# 1. Generate Mesh
echo "Step 1: Generating core cube mesh..."
micromamba run -n ${ENV_NAME} python3 src/mesh.py --geom box --extent ${L},${L},${L} --h ${H} --backend grid --out-name cube_${L}nm_2nm --no-vis

# 2. Run Micromagnetics Simulation
echo "Step 2: Running relaxation to remanence state..."
micromamba run -n ${ENV_NAME} python3 src/loop.py --mesh cube_${L}nm_2nm.npz --materials cube.krn \
    --B-start 0 --B-end 0 --dB 1 --h-dir 0,0,1 \
    --out-dir ${OUT_DIR} \
    --add-shell --layers ${LAYERS} --K ${K} --h0 ${H0_AIR} \
    --precond-type amgcl --cg-tol 1e-8 --eps-a 1e-10 --verbose

echo "=== Benchmark Complete ==="
