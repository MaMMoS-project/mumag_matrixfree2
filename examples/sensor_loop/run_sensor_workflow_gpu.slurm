#!/bin/bash
#SBATCH --job-name=sensor_loop
#SBATCH --partition=dissSims
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=24:00:00
#SBATCH --output=sensor_loop_gpu_%j.log

echo "Starting Sensor Loop Workflow..."
cd /ceph/home/schrefl/jax_dev/mumag_matrixfree2/examples/sensor_loop

WORKSPACE="run_gpu_${SLURM_JOB_ID}"
mkdir -p "${WORKSPACE}"
echo "Running in isolated workspace: ${WORKSPACE}"

echo "Running step-by-step simulation..."
pixi run -e cuda python sensor_loop_step_by_step.py --workspace "${WORKSPACE}"

echo "Running evaluation..."
pixi run -e cuda python sensor_loop_evaluation.py --sensor-loop-dir "${WORKSPACE}"

echo "Plotting comparison..."
pixi run python plot_comparison.py --workspace "${WORKSPACE}"

echo "Workflow complete."
