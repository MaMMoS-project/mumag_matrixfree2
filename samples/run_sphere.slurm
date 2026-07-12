#!/bin/bash
#SBATCH --job-name=sphere_40nm
#SBATCH --partition=dissSims
#SBATCH --nodelist=Pm
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=sphere_%j.log

echo "=== Running Sphere Sample on Pm ==="
export JAX_ENABLE_X64=True

echo "=== Creating the sphere mesh ==="
pixi run --manifest-path ../pixi.toml -e cuda python3 ../src/mesh.py --geom ellipsoid --extent 40,40,40 --h 0.83 --out-name sphere_40nm

echo "=== Running the simulation ==="
pixi run --manifest-path ../pixi.toml -e cuda python3 ../src/loop.py sphere_40nm --add-shell --layers 4 --out-dir hyst_sphere_40nm --verbose || echo "=== WARNING: simulation failed! ==="

echo "=== Plotting the results ==="
pixi run --manifest-path ../pixi.toml -e cuda python3 ../src/plot_hysteresis.py hyst_sphere_40nm/hysteresis.csv hyst_sphere_40nm/sphere_hyst.png

echo "=== Extracting and reporting the nucleation field ==="
pixi run --manifest-path ../pixi.toml -e cuda python3 ../src/extract_nucleation.py hyst_sphere_40nm/hysteresis.csv

echo "=== Sphere simulation finished ==="
