#!/bin/bash
cd "$(dirname "$0")/.."

echo "Cleaning CPU run outputs..."

# Remove generated mesh files
rm -f cube_20nm.npz cube_20nm.vtu

# Remove generated configuration files
rm -f cube_20nm.krn cube_20nm.p2

# Remove plotting script and image
rm -f plot_demag_cpu.py demag_curve_cpu.png

# Remove simulation output directory
rm -rf hyst_cube_20nm

# Remove slurm output logs from the slurm directory
rm -f slurm/slurm-*.out

echo "Done."
