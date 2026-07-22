#!/bin/bash
#SBATCH --job-name=gen_structs
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

cd $SLURM_SUBMIT_DIR

# Load necessary modules/environments here if required by your cluster
# e.g., module load python/3.x...
# source /path/to/venv/bin/activate

echo "Starting structure generation..."
pixi run python generate_structures.py --extent 80,80,80 --grains 8 --num-structures 10
echo "Done."
