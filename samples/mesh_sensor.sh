#!/bin/bash
set -e

echo "=== Generating High Aspect Ratio Eye Mesh (Sensor) ==="
python ../src/mesh.py --geom eye --extent 2000,100,10 --h 5 --add-shell --out-name eye_mesh_2000

echo "=== Mesh Generation Complete ==="
