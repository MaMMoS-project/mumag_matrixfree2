#!/bin/bash
set -e

# Run the simulation
~/.pixi/bin/pixi run python src/loop.py sphere_40nm --add-shell --layers 4 --out-dir hyst_sphere_40nm --verbose

# Plot the results
~/.pixi/bin/pixi run python src/plot_hysteresis.py hyst_sphere_40nm/hysteresis.csv --out sphere_hyst.png
