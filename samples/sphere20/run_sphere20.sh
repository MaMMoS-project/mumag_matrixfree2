#!/usr/bin/env bash
set -euo pipefail

export JAX_ENABLE_X64=True

echo "Start mesh generation."
python ../../src/mesh.py --geom ellipsoid --extent 40,40,40 --h 2.0 --backend meshpy --out-name sphere20
echo "Finished mesh generation."

echo "Start simulation."
python ../../src/loop.py sphere20 --add-shell --layers 4 --K 2.0 --h0 2.0 --out-dir sphere20_out
