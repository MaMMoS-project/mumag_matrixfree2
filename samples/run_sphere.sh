#!/bin/bash
# run_sphere.sh
# Run this from the samples/ directory
set -e

# Run the simulation
pixi run python ../src/loop.py sphere_40nm --add-shell --layers 4 --out-dir hyst_sphere_40nm --verbose

# Plot the results
pixi run python ../src/plot_hysteresis.py hyst_sphere_40nm/hysteresis.csv hyst_sphere_40nm/sphere_hyst.png
