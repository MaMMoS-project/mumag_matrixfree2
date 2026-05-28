#!/usr/bin/env bash
set -euo pipefail

export JAX_ENABLE_X64=True

echo "Start mesh generation."
python ../../src/mesh.py --geom ellipsoid --extent 40,40,40 --h 2.0 --backend meshpy --out-name sphereR20nm
echo "Finished mesh generation."

echo "Start simulation."
python ../../src/loop.py sphereR20nm --add-shell --layers 4 --K 2.0 --h0 2.0 --out-dir sphereR20nm_out
