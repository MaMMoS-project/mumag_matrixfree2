#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1-00:00
#SBATCH --job-name=mumag-sample
#SBATCH --partition=dissSims
#SBATCH --gpus=1
#SBATCH --constraint=cuda
#SBATCH --export=NONE

set -e

# Define project source directory
SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
PROJ_DIR=$(realpath "$(dirname "$SCRIPT_PATH")/..")

# Define Pixi installation directory in home
export PIXI_HOME="$HOME/.pixi"
export PATH="$PIXI_HOME/bin:$PATH"

# Install Pixi if not found
if ! command -v pixi &> /dev/null; then
    echo "Pixi not found. Installing..."
    curl -fsSL https://pixi.sh/install.sh | bash
fi

# Use a temporary directory for the simulation
SIMDIR="$HOME/slurm_sims/job_$SLURM_JOB_ID"
mkdir -p "$SIMDIR"
# The trap now only cleans up the temporary SIMDIR
trap 'echo "Cleaning up $SIMDIR..."; rm -rf "$SIMDIR"' EXIT

# Copy project files from the source directory to SIMDIR
echo "Copying project from $PROJ_DIR to $SIMDIR..."
cp -r "$PROJ_DIR"/* "$SIMDIR/"
cd "$SIMDIR"

echo "JOBNAME: $SLURM_JOB_NAME"
echo "NODELIST: $SLURMD_NODENAME"
echo "SIMDIR: $PWD"
echo "SUBMIT_DIR: $SLURM_SUBMIT_DIR"

# Define parameters consistent with samples/run_sample.sh
L=20
H=2
LAYERS=4
K=2.0
H0_AIR=${H}

echo "Starting simulation steps (Direct Python calls)..."

# Ensure double precision for JAX
export JAX_ENABLE_X64=True

cd samples

# 1. Generate core mesh
echo "Generating core cube mesh..."
pixi run -e cuda python3 ../src/mesh.py --geom box --extent ${L},${L},${L} --h ${H} --backend grid --out-name cube_${L}nm --no-vis

# 2. Standard Demag
echo "Running Standard Simulation: demag"
cd demag
pixi run -e cuda python3 ../../src/loop.py cube_${L}nm \
    --mesh ../cube_${L}nm.npz \
    --out-dir ../bench_demag \
    --add-shell --layers ${LAYERS} --K ${K} --h0 ${H0_AIR} \
    --verbose
cd ..

# 3. Standard No-Demag
echo "Running Standard Simulation: no_demag"
cd no_demag
pixi run -e cuda python3 ../../src/loop.py cube_${L}nm \
    --mesh ../cube_${L}nm.npz \
    --out-dir ../bench_no_demag \
    --add-shell --layers ${LAYERS} --K ${K} --h0 ${H0_AIR} \
    --no-demag \
    --verbose
cd ..

# 4. Magnetoelastic Demag
echo "Running Magnetoelastic Verification: demag"
cd demag
pixi run -e cuda python3 ../../src/loop.py cube_${L}nm \
    --mesh ../cube_20nm_halfK1me.inp \
    --materials ../cube_20nm_halfK1.krn \
    --out-dir ../bench_me_demag \
    --add-shell --layers ${LAYERS} --K ${K} --h0 ${H0_AIR} \
    --verbose
cd ..

# 5. Magnetoelastic No-Demag
echo "Running Magnetoelastic Verification: no_demag"
cd no_demag
pixi run -e cuda python3 ../../src/loop.py cube_${L}nm \
    --mesh ../cube_20nm_halfK1me.inp \
    --materials ../cube_20nm_halfK1.krn \
    --out-dir ../bench_me_no_demag \
    --add-shell --layers ${LAYERS} --K ${K} --h0 ${H0_AIR} \
    --no-demag \
    --verbose
cd ..

# 6. Automated Consistency Checks
echo "Running Automated Consistency Checks..."
pixi run -e cuda python3 check_results.py

# Create results directory in submit dir and copy results back
RESULTS_DIR="$SLURM_SUBMIT_DIR/results_$SLURM_JOB_ID"
echo "Copying results back to $RESULTS_DIR..."
mkdir -p "$RESULTS_DIR"

# Copy all created bench directories
cp -r bench_* "$RESULTS_DIR/"

echo "Job $SLURM_JOB_ID complete. Results saved in $RESULTS_DIR"
