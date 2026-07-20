#!/bin/bash
cd "$(dirname "$0")/.."

echo "Cleaning GPU run outputs..."

# Remove generated mesh files
rm -f poly_gb_100nm.npz poly_gb_100nm.vtu

# Remove generated python script, parameter, and configuration files
rm -f create_krn.py poly_gb_100nm.krn poly_gb_100nm.p2

# Remove plotting script and image
rm -f plot_demag_gpu.py demag_curve_gpu.png

# Remove simulation output directory
rm -rf hyst_poly_gb_100nm

# Remove slurm output logs from the slurm directory
rm -f slurm/slurm-*.out

echo "Done."
