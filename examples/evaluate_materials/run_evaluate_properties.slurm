#!/bin/bash
#SBATCH --job-name=eval_props
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

cd $SLURM_SUBMIT_DIR

# Load necessary modules/environments here if required by your cluster
# e.g., module load python/3.x...
# source /path/to/venv/bin/activate

# Example values, you can pass these via the sbatch command or loop over them
K1=${1:-700000}
Js=${2:-0.8}
A=${3:-7.6e-11}
HSTART=${4:-2.0}
HFINAL=${5:--2.0}
HSTEP=${6:-0.01}

echo "Starting evaluation for K1=$K1, Js=$Js, A=$A..."
pixi run python evaluate_properties.py --K1 $K1 --Js $Js --A $A --hstart $HSTART --hfinal $HFINAL --hstep $HSTEP
echo "Done."
