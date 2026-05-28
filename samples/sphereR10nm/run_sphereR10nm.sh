#!/usr/bin/env bash
set -euo pipefail

export JAX_ENABLE_X64=True

echo "Start mesh generation."
python ../../src/mesh.py --geom ellipsoid --extent 20,20,20 --h 2.0 --backend meshpy --out-name sphereR10nm
echo "Finished mesh generation."

echo "Start simulation."
python ../../src/loop.py sphereR10nm --add-shell --layers 4 --K 2.0 --h0 2.0 --out-dir sphereR10nm_out
