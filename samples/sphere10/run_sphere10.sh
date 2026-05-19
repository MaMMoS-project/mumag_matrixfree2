#!/usr/bin/env bash
set -euo pipefail

export JAX_ENABLE_X64=True

echo "Start mesh generation."
python ../../src/mesh.py --geom ellipsoid --extent 20,20,20 --h 2.0 --backend meshpy --out-name sphere10
echo "Finished mesh generation."

echo "Start simulation."
python ../../src/loop.py sphere10 --add-shell --layers 4 --K 2.0 --h0 2.0 --out-dir sphere10_out
