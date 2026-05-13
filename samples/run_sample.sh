#!/bin/bash
set -e

L=20
H=2
LAYERS=4
K=2.0
H0_AIR=${H}
ENV_NAME="mfree-mumag-cpu"

run_simulation() {
    local MODE=$1
    local OUT_DIR="bench_${MODE}"
    
    echo "--------------------------------------------------------"
    echo "Running Simulation: ${MODE}"
    echo "--------------------------------------------------------"
    
    # 1. Generate Mesh
    echo "Step 1: Generating core cube mesh..."
    python3 ../src/mesh.py --geom box --extent ${L},${L},${L} --h ${H} --backend grid --out-name cube_${L}nm --no-vis
    
    # 2. Run Micromagnetics Simulation
    # Note: the script uses the .p2 file from the current directory (MODE)
    cd ${MODE}
    echo "Step 2: Running hysteresis loop simulation in ${MODE}..."
    python3 ../../src/loop.py cube_${L}nm \
        --out-dir ../${OUT_DIR} \
        --add-shell --layers ${LAYERS} --K ${K} --h0 ${H0_AIR} \
        --verbose
    cd ..
}

echo "=== Micromagnetics Performance Benchmark (Hysteresis Loop) ==="
echo "Core Mesh: ${L}x${L}x${L} nm cube, h=${H} nm"
echo "Environment: ${ENV_NAME}"

# Run both modes
run_simulation "demag"
run_simulation "no_demag"

echo "--------------------------------------------------------"
echo "=== All Samples Complete ==="
echo "Results available in samples/bench_demag/ and samples/bench_no_demag/"
