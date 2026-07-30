#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "usage: repair_linux_wheel.sh WHEEL DEST_DIR" >&2
    exit 2
fi

INPUT_WHEEL="$1"
DEST_DIR="$2"
if [[ ! -f "$INPUT_WHEEL" ]]; then
    echo "input wheel does not exist: $INPUT_WHEEL" >&2
    exit 2
fi

mkdir -p "$DEST_DIR"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tommos-wheel-repair.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

AUDITWHEEL_DIR="$WORK_DIR/auditwheel"
UNPACK_DIR="$WORK_DIR/unpacked"
mkdir -p "$AUDITWHEEL_DIR" "$UNPACK_DIR"

auditwheel repair \
    --plat manylinux_2_28_x86_64 \
    --exclude libmkl_rt.so.3 \
    --wheel-dir "$AUDITWHEEL_DIR" \
    "$INPUT_WHEEL"

shopt -s nullglob
REPAIRED_WHEELS=("$AUDITWHEEL_DIR"/*.whl)
if [[ "${#REPAIRED_WHEELS[@]}" -ne 1 ]]; then
    echo "expected exactly one auditwheel output, found ${#REPAIRED_WHEELS[@]}" >&2
    exit 1
fi

python -m wheel unpack --dest "$UNPACK_DIR" "${REPAIRED_WHEELS[0]}"
UNPACKED_ROOTS=("$UNPACK_DIR"/*/)
if [[ "${#UNPACKED_ROOTS[@]}" -ne 1 ]]; then
    echo "expected exactly one unpacked wheel root, found ${#UNPACKED_ROOTS[@]}" >&2
    exit 1
fi

UNPACKED_ROOT="${UNPACKED_ROOTS[0]}"
NATIVE_EXTENSION="$UNPACKED_ROOT/tommos/_native/libcpp_mkl_minimizer.so"
if [[ ! -f "$NATIVE_EXTENSION" ]]; then
    echo "repaired wheel is missing $NATIVE_EXTENSION" >&2
    exit 1
fi

patchelf \
    --add-rpath '$ORIGIN/../../../..' \
    --force-rpath \
    "$NATIVE_EXTENSION"

EXPECTED_RPATH='$ORIGIN/../../tommos.libs:$ORIGIN/../../../..'
ACTUAL_RPATH="$(patchelf --print-rpath "$NATIVE_EXTENSION")"
if [[ "$ACTUAL_RPATH" != "$EXPECTED_RPATH" ]]; then
    echo "unexpected repaired RPATH: $ACTUAL_RPATH" >&2
    exit 1
fi

python -m wheel pack --dest-dir "$DEST_DIR" "$UNPACKED_ROOT"
