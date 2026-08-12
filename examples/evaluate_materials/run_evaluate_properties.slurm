#!/bin/bash
#SBATCH --job-name=eval_props
#SBATCH --partition=dissSims
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=2
#SBATCH --gres=gpu:1 # a100 currently blocked
#SBATCH --mem=128G
#SBATCH --time=04:00:00

cd $SLURM_SUBMIT_DIR

# Load necessary modules/environments here if required by your cluster
# e.g., module load python/3.x...
# source /path/to/venv/bin/activate

# Example values, you can pass these via the sbatch command or loop over them
# (prepared) here I took a few different material combinations
# Coey book (page 377) materials:
# mat name    K1 (MJ/m³)  Js (T)   A (pJ/m)
# Nd2Fe14B    4.9         1.61      8
# Sm2Fe17N3   8.6         1.54     12
# SmCo5      17.2         1.07     12
# Sm2Co17     4.2         1.25     16
# CoPt        4.9         0.99     10

# NOTE ':-' means: 
# if the script does not get a first command line input it'll take 700000 instead 
# NOW NEW no exchange stiffness constant is given, it will compute it based on
# the power law explained below 
# MAYBE this <c> has to be controlled, because if we look at the above material
# properties, they do not follow the same c as Nd2Fe14B 

K1=${1:-700000}
Js=${2:-0.8}

# Compute exchange stiffness A from Nd2Fe14B reference scaling if not passed as $3
# Nd2Fe14B reference: A0 = 8e-12 J/m, Js0 = 1.61 T => c = 8e-12 / (1.61^2)
if [ -z "${3:-}" ]; then
    A=$(python3 -c "c = 8e-12 / (1.61**2); print(f'{c * ($Js**2):.6e}')")
else
    A=$3
fi

# Compute anisotropy field H_ani (in Tesla) = 2 * mu_0 * K1 / Js
H_ANI=$(python3 -c "import math; mu0 = 4 * math.pi * 1e-7; print(f'{(2 * mu0 * $K1 / $Js):.6f}')")

# Field bounds: +1.5 * H_ANI to -1.5 * H_ANI
# Field step: proportional step size for 400 steps across total sweep range
if [ -z "${4:-}" ]; then
    HSTART=$(python3 -c "print(f'{(1.5 * float($H_ANI)):.6f}')")
else
    HSTART=$4
fi

if [ -z "${5:-}" ]; then
    HFINAL=$(python3 -c "print(f'{(-1.5 * float($H_ANI)):.6f}')")
else
    HFINAL=$5
fi

if [ -z "${6:-}" ]; then
    HSTEP=$(python3 -c "print(f'{(-3.0 * float($H_ANI) / 400.0):.6f}')")
else
    HSTEP=$6
fi

echo "Calculated parameters: H_ANI=${H_ANI}T, HSTART=${HSTART}T, HFINAL=${HFINAL}T, HSTEP=${HSTEP}T"

echo "Starting evaluation for K1=$K1, Js=$Js, A=$A..."
export JAX_ENABLE_X64=True
#export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
pixi run -e cuda python evaluate_properties.py --K1 $K1 --Js $Js --A $A --hstart $HSTART --hfinal $HFINAL --hstep $HSTEP
echo "Done."
