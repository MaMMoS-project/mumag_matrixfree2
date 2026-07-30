#!/bin/bash
set -euo pipefail

if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Skipping C++ minimizer compilation on non-Linux platform."
    exit 0
fi

PROJECT_ROOT="${PIXI_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"

echo "Rebuilding the editable tommos installation through isolated PEP 517..."
python -m pip install --no-deps --force-reinstall --no-cache-dir --editable "$PROJECT_ROOT"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    LOCAL_BUILD_DIR="/tmp/mumag_build_${SLURM_JOB_ID}"
    mkdir -p "$LOCAL_BUILD_DIR"
    NATIVE_LIBRARY="$(
        python - <<'PY'
from importlib.metadata import distribution

native_entries = [
    entry
    for entry in distribution("tommos").files or ()
    if str(entry).endswith("tommos/_native/libcpp_mkl_minimizer.so")
]
if len(native_entries) != 1:
    raise SystemExit(
        f"expected one installed libcpp_mkl_minimizer.so, found {len(native_entries)}"
    )
print(distribution("tommos").locate_file(native_entries[0]).resolve(strict=True))
PY
    )"
    cp "$NATIVE_LIBRARY" "$LOCAL_BUILD_DIR/libcpp_mkl_minimizer.so"
    echo "Copied Slurm compatibility library to $LOCAL_BUILD_DIR/libcpp_mkl_minimizer.so"
fi

echo "Successfully rebuilt the editable tommos installation."
