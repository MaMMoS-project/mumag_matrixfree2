#!/bin/bash
#SBATCH --job-name=demag_multigpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm_multigpu.out
#SBATCH --error=logs/slurm_multigpu.err

# Ensure we are in the submission directory (tests/)
cd "$SLURM_SUBMIT_DIR" || exit 1

OUT_DIR="tmp_test_out"
LOG_DIR="logs"

# Disable NCCL P2P due to PHB topology issues on this motherboard
export NCCL_P2P_DISABLE=1

echo "Running Test 7 on Pm..."
rm -f "$OUT_DIR"/cube_20nm.*
pixi run -e cuda python ../src/loop.py cube_20nm --add-shell --benchmark --out-dir "$OUT_DIR" > "$LOG_DIR/run7_multigpu.log" 2>&1

echo "Running Test 8 on Pm..."
rm -f "$OUT_DIR"/cube_20nm.*
pixi run -e cuda python ../src/loop.py cube_20nm --add-shell --method tr --benchmark --out-dir "$OUT_DIR" > "$LOG_DIR/run8_multigpu_tr.log" 2>&1
