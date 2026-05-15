#!/bin/bash
set -e

L=20
H=2
LAYERS=4
K=2.0
H0_AIR=${H}
ENV_NAME="mfree-mumag-cpu"

# 1. Establish core mesh for all runs
echo "Generating core cube mesh..."
python3 ../src/mesh.py --geom box --extent ${L},${L},${L} --h ${H} --backend grid --out-name cube_${L}nm --no-vis

run_standard() {
    local MODE=$1
    local OUT_DIR="bench_${MODE}"
    local EXTRA_ARGS=""
    if [ "$MODE" == "no_demag" ]; then
        EXTRA_ARGS="--no-demag"
    fi

    echo "--------------------------------------------------------"
    echo "Running Standard Simulation: ${MODE}"
    echo "--------------------------------------------------------"
    
    cd ${MODE}
    python3 ../../src/loop.py cube_${L}nm \
        --mesh ../cube_${L}nm.npz \
        --out-dir ../${OUT_DIR} \
        --add-shell --layers ${LAYERS} --K ${K} --h0 ${H0_AIR} \
        ${EXTRA_ARGS} \
        --verbose
    cd ..
}

run_magnetoelastic() {
    local MODE=$1
    local OUT_DIR="bench_me_${MODE}"
    local EXTRA_ARGS=""
    if [ "$MODE" == "no_demag" ]; then
        EXTRA_ARGS="--no-demag"
    fi

    echo "--------------------------------------------------------"
    echo "Running Magnetoelastic Verification: ${MODE}"
    echo "--------------------------------------------------------"
    
    # We use the .inp file with half K1 and the .krn with half K1.
    # Total anisotropy should equal the standard case.
    # Note: we still use the .p2 from demag/ for consistent field sweep params.
    cd ${MODE}
    python3 ../../src/loop.py cube_${L}nm \
        --mesh ../cube_20nm_halfK1me.inp \
        --materials ../cube_20nm_halfK1.krn \
        --out-dir ../${OUT_DIR} \
        --add-shell --layers ${LAYERS} --K ${K} --h0 ${H0_AIR} \
        ${EXTRA_ARGS} \
        --verbose
    cd ..
}

echo "=== Micromagnetics Performance Benchmark & ME Verification ==="
echo "Core Mesh: ${L}x${L}x${L} nm cube, h=${H} nm"

# Original Runs
run_standard "demag"
run_standard "no_demag"

# New Verification Runs
run_magnetoelastic "demag"
run_magnetoelastic "no_demag"

echo "--------------------------------------------------------"
echo "=== Running Automated Consistency Checks ==="
python3 check_results.py

echo "--------------------------------------------------------"
echo "=== All Samples Complete ==="
echo "Results available in samples/bench_*/"
