#!/bin/bash
#SBATCH --job-name=case_a_down
#SBATCH --partition=dissSims
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=24:00:00
#SBATCH --output=case_a_down_%j.log

echo "Starting Case A Down simulation..."
cd /ceph/home/schrefl/jax_dev/mumag_matrixfree2/examples/sensor_loop/sensor_case-a_down

# Run the single hysteresis branch
pixi run -e cuda python ../../../src/loop.py sensor

echo "Copying hysteresis data to case directory..."
cp hyst_sensor/sensor.mh sensor.mh

echo "Evaluating and updating files..."
cd /ceph/home/schrefl/jax_dev/mumag_matrixfree2/examples/sensor_loop
pixi run -e cuda python sensor_loop_evaluation.py

echo "Regenerating comparison plot..."
pixi run python plot_comparison.py

echo "Job complete!"
