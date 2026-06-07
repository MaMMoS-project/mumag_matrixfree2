#!/bin/bash
# benchmarking/run_comparison_benchmark.sh
set -e

# Dimensions and resolution for a small test mesh
L=10
H=2
LAYERS=4

echo "=== Micromagnetics Minimizer Comparison Benchmark ==="
echo "Mesh: ${L}x${L}x${L} nm cube, h=${H} nm"
echo "Airbox: ${LAYERS} layers"

# 1. Generate a small mesh
echo "Step 1: Generating small core mesh..."
pixi run python src/mesh.py --geom box --extent ${L},${L},${L} --h ${H} --backend grid --out-name bench_mesh --no-vis

# 2. Run the comparison benchmark
echo "Step 2: Running minimizer comparison benchmark with airbox..."
pixi run python benchmarking/compare_minimizers.py --mesh bench_mesh.npz --max-iter 200 --add-shell --layers ${LAYERS}

# 3. Clean up
rm bench_mesh.npz

echo "=== Benchmark Complete ==="
