#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=mumag-tests
#SBATCH --partition=dissSims
#SBATCH --exclude=Pr,Ce,Nd,Tb,Ho
#SBATCH --export=NONE

# Define project source directory
# SLURM copies the script to a spool directory, so $0 is not the original path.
# We use scontrol to find the original Command path.
SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
PROJ_DIR=$(realpath "$(dirname "$SCRIPT_PATH")/..")

# Define Pixi installation directory in home
export PIXI_HOME="$HOME/.pixi"
export PATH="$PIXI_HOME/bin:$PATH"

# Install Pixi if not found
if ! command -v pixi &> /dev/null; then
    echo "Pixi not found. Installing to $PIXI_HOME..."
    curl -fsSL https://pixi.sh/install.sh | bash
fi

# Navigate to the project source directory to run tests
cd "$PROJ_DIR"

echo "JOBNAME: $SLURM_JOB_NAME"
echo "NODELIST: $SLURMD_NODENAME"
echo "PROJECT DIR: $PWD"
echo "SUBMIT DIR: $SLURM_SUBMIT_DIR"

# Run tests using Pixi
# We use 'pixi run test' which is defined in pixi.toml
echo "Starting tests..."
pixi run test

# Cleanup
echo "Cleaning up temporary files..."
pixi run clean
rm -rf .pytest_cache
