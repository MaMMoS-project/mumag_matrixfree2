#!/bin/bash
set -e

L=20
H=2
LAYERS=4
K=2.0
H0_AIR=${H}
OUT_DIR="bench_demag_curve"
ENV_NAME="mfree-mumag-cpu"

echo "=== Micromagnetics Performance Benchmark (Hysteresis Loop) ==="
echo "Core Mesh: ${L}x${L}x${L} nm cube, h=${H} nm"
echo "Airbox: ${LAYERS} layers, K=${K}, h0_air=${H0_AIR} nm"
echo "Environment: ${ENV_NAME}"
echo "Parameters from cube_20nm.p2"

# 1. Generate Mesh
# echo "Step 1: Generating core cube mesh..."
# pixi run python ../src/mesh.py --geom box --extent ${L},${L},${L} --h ${H} --backend grid --out-name cube_${L}nm --no-vis

# 2. Run Micromagnetics Simulation
echo "Step 2: Running hysteresis loop simulation..."
mkdir -p ${OUT_DIR}
pixi run python ../src/loop.py cube_${L}nm \
    --out-dir ${OUT_DIR} \
    --add-shell --layers ${LAYERS} --K ${K} --h0 ${H0_AIR} \
    --method cohen \
    --tau0 1.0 \
    --operator-mode assembled \
    --verbose 2>&1 | tee ${OUT_DIR}/simulation.log

echo "=== Benchmark Complete ==="
