#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1-00:00
#SBATCH --job-name=mumag-custom
#SBATCH --partition=dissSims
#SBATCH --gpus=1
#SBATCH --constraint=cuda
#SBATCH --exclude=Ce,Nd,Tb,Sc
#SBATCH --export=NONE

set -e

# ==============================================================================
# USER CONFIGURATION
# Set the base name of your model files (e.g., if you have my_mesh.inp, 
# my_mesh.krn, and my_mesh.p2, set MODEL_NAME="my_mesh")
# ==============================================================================
MODEL_NAME="model"
# ==============================================================================

# Define project source directory (assuming this script is in PROJ_DIR/slurm/)
SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
PROJ_DIR=$(realpath "$(dirname "$SCRIPT_PATH")/..")

# Define Pixi installation directory
export PIXI_HOME="$HOME/.pixi"
export PATH="$PIXI_HOME/bin:$PATH"

# Install Pixi if not found
if ! command -v pixi &> /dev/null; then
    echo "Pixi not found. Installing..."
    curl -fsSL https://pixi.sh/install.sh | bash
fi

# Use a temporary directory for the simulation on the local node
SIMDIR="$HOME/slurm_sims/job_$SLURM_JOB_ID"
mkdir -p "$SIMDIR"
trap 'echo "Cleaning up $SIMDIR..."; rm -rf "$SIMDIR"' EXIT

# 1. Copy essential project source code
echo "Copying essential project files from $PROJ_DIR to $SIMDIR..."
mkdir -p "$SIMDIR/src"
cp -r "$PROJ_DIR/src"/* "$SIMDIR/src/"
cp "$PROJ_DIR/pixi.toml" "$SIMDIR/"
if [ -f "$PROJ_DIR/pixi.lock" ]; then
    cp "$PROJ_DIR/pixi.lock" "$SIMDIR/"
fi

# 2. Copy user input files from the submission directory to the execution directory
echo "Copying input files for '$MODEL_NAME' from $SLURM_SUBMIT_DIR..."
cp "$SLURM_SUBMIT_DIR/${MODEL_NAME}.inp" "$SIMDIR/"
cp "$SLURM_SUBMIT_DIR/${MODEL_NAME}.krn" "$SIMDIR/"
cp "$SLURM_SUBMIT_DIR/${MODEL_NAME}.p2" "$SIMDIR/"

cd "$SIMDIR"

echo "JOBNAME: $SLURM_JOB_NAME"
echo "NODELIST: $SLURMD_NODENAME"
echo "SIMDIR: $PWD"
echo "SUBMIT_DIR: $SLURM_SUBMIT_DIR"

# 3. Run the simulation
# - JAX_ENABLE_X64=True ensures double precision
# - modelname (first arg) triggers auto-discovery of .inp, .krn, and .p2
echo "Starting simulation (CUDA)..."
export JAX_ENABLE_X64=True
pixi run -e cuda python3 src/loop.py "$MODEL_NAME" --no-demag --verbose

# 4. Copy the results back to the submission directory
# loop.py defaults to output directory 'hyst_[MODEL_NAME]'
OUT_DIR="hyst_${MODEL_NAME}"
RESULTS_DEST="$SLURM_SUBMIT_DIR/results_${MODEL_NAME}_$SLURM_JOB_ID"

echo "Copying results back to $RESULTS_DEST..."
if [ -d "$OUT_DIR" ]; then
    mkdir -p "$RESULTS_DEST"
    cp -r "$OUT_DIR"/* "$RESULTS_DEST/"
    echo "Success: Results saved to $RESULTS_DEST"
else
    echo "Error: Output directory $OUT_DIR not found. Check simulation logs."
    exit 1
fi
