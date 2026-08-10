#!/bin/bash
#SBATCH --job-name=sensor_loop_cpu
#SBATCH --partition=dissSims
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=sensor_loop_cpu_%j.log

echo "Starting Sensor Loop Workflow on CPU..."
cd /ceph/home/schrefl/jax_dev/mumag_matrixfree2/examples/sensor_loop

export JAX_ENABLE_X64=True

# Force MKL and OpenMP to use exactly the number of cores allocated by Slurm
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Pin threads to physical cores to maximize cache hits and memory bandwidth
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# Keep threads awake between rapid matrix multiplications to avoid sleep/wake latency
export OMP_WAIT_POLICY=ACTIVE

echo "Compiling the C++ library..."
pixi run --manifest-path /ceph/home/schrefl/jax_dev/mumag_matrixfree2/pixi.toml compile

WORKSPACE="run_cpu_${SLURM_JOB_ID}"
mkdir -p "${WORKSPACE}"
echo "Running in isolated workspace: ${WORKSPACE}"

echo "Running step-by-step simulation..."
pixi run -e cpu python sensor_loop_step_by_step.py --workspace "${WORKSPACE}"

echo "Running evaluation..."
pixi run -e cpu python sensor_loop_evaluation.py --sensor-loop-dir "${WORKSPACE}"

echo "Plotting comparison..."
pixi run -e cpu python plot_comparison.py --workspace "${WORKSPACE}"

# Clean tmp compilation files to avoid leaving artifacts on the compute node
rm -rf /tmp/mumag_build_${SLURM_JOB_ID}

echo "Workflow complete."
