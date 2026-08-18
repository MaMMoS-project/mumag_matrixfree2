#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUT_DIR="$DIR/tmp_test_out"
LOG_DIR="$DIR/logs"

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

rm -f "$LOG_DIR"/run*.log
rm -f "$LOG_DIR"/slurm_multigpu.out "$LOG_DIR"/slurm_multigpu.err

cd "$DIR"

check_switching_field() {
    local mh_file=$1
    echo "Checking $mh_file"
    if [ ! -f "$mh_file" ]; then
        echo "File $mh_file not found!"
        return 1
    fi
    awk '
    /^#/ { next }
    $2 < 0 { 
        # Check if B_ext is around -6.0
        if ($1 > -6.1 && $1 < -5.9) {
            print "Pass: Switching field is " $1 " T"
            exit 0
        } else {
            print "Fail: Expected switching field ~ -6.0 T, got " $1 " T"
            exit 1
        }
    }' "$mh_file" || return 1
}

# Check hardware capabilities
HAS_GPU=false
if command -v nvidia-smi &> /dev/null; then
    if pixi run -e cuda python -c "import jax; assert jax.devices()[0].platform == 'gpu'" &> /dev/null; then
        HAS_GPU=true
    fi
fi

HAS_CPP=false
if [ "$(uname -s)" = "Linux" ]; then
    if pixi run compile > "$LOG_DIR/compile.log" 2>&1; then
        HAS_CPP=true
    fi
fi

if [ "$HAS_GPU" = true ]; then
    echo "=== Test 1: GPU with --benchmark ==="
    rm -f "$OUT_DIR"/cube_20nm.*
    pixi run -e cuda python ../src/loop.py cube_20nm --add-shell --benchmark --out-dir "$OUT_DIR" > "$LOG_DIR/run1_gpu.log" 2>&1
    check_switching_field "$OUT_DIR/cube_20nm.mh"
else
    echo "=== Test 1: GPU unavailable, skipping ==="
fi

if [ "$HAS_CPP" = true ]; then
    echo "=== Test 2: CPU with C++ minimizer ==="
    rm -f "$OUT_DIR"/cube_20nm.*
    JAX_PLATFORMS=cpu pixi run python ../src/loop.py cube_20nm --add-shell --out-dir "$OUT_DIR" > "$LOG_DIR/run2_cpu_cpp.log" 2>&1
    check_switching_field "$OUT_DIR/cube_20nm.mh"
else
    echo "=== Test 2: C++ MKL minimizer unavailable, skipping ==="
fi

echo "=== Test 3: CPU with scipy (Universal Reference) ==="
rm -f "$OUT_DIR"/cube_20nm.*
JAX_PLATFORMS=cpu pixi run python ../src/loop.py cube_20nm --add-shell --out-dir "$OUT_DIR" --cpu-spmv-backend scipy --no-cpp-mkl --poisson-solver jax > "$LOG_DIR/run3_cpu_scipy.log" 2>&1
check_switching_field "$OUT_DIR/cube_20nm.mh"

if [ "$HAS_GPU" = true ]; then
    echo "=== Test 4: GPU with --benchmark and --method tr ==="
    rm -f "$OUT_DIR"/cube_20nm.*
    pixi run -e cuda python ../src/loop.py cube_20nm --add-shell --benchmark --method tr --out-dir "$OUT_DIR" > "$LOG_DIR/run4_gpu_tr.log" 2>&1
    check_switching_field "$OUT_DIR/cube_20nm.mh"
else
    echo "=== Test 4: GPU unavailable, skipping ==="
fi

if [ "$HAS_CPP" = true ]; then
    echo "=== Test 5: CPU with C++ minimizer and --method tr ==="
    rm -f "$OUT_DIR"/cube_20nm.*
    JAX_PLATFORMS=cpu pixi run python ../src/loop.py cube_20nm --add-shell --method tr --out-dir "$OUT_DIR" > "$LOG_DIR/run5_cpu_cpp_tr.log" 2>&1
    check_switching_field "$OUT_DIR/cube_20nm.mh"
else
    echo "=== Test 5: C++ MKL minimizer unavailable, skipping ==="
fi

echo "=== Test 6: CPU with scipy and --method tr ==="
rm -f "$OUT_DIR"/cube_20nm.*
JAX_PLATFORMS=cpu pixi run python ../src/loop.py cube_20nm --add-shell --method tr --out-dir "$OUT_DIR" --cpu-spmv-backend scipy --no-cpp-mkl --poisson-solver jax > "$LOG_DIR/run6_cpu_scipy_tr.log" 2>&1
check_switching_field "$OUT_DIR/cube_20nm.mh"

TARGET_NODE="Pm"
TARGET_NODE_AVAILABLE=false
if sinfo -h -n "$TARGET_NODE" -o "%T" | grep -qE "mixed|allocated|idle|reserved"; then
    TARGET_NODE_AVAILABLE=true
fi

if [ "$TARGET_NODE_AVAILABLE" = true ]; then
    echo "=== Submitting Tests 7 and 8 to Slurm on $TARGET_NODE (waiting for completion) ==="
    sbatch --wait --nodelist="$TARGET_NODE" --reservation=pm_exclusive run_multi_gpu.slurm
    
    echo "=== Checking Test 7 ==="
    check_switching_field "$OUT_DIR/cube_20nm.mh"

    echo "=== Checking Test 8 ==="
    check_switching_field "$OUT_DIR/cube_20nm.mh"
else
    echo "=== Test 7: Multi-GPU (2-device) skipped ($TARGET_NODE unavailable) ==="
    echo "=== Test 8: Multi-GPU (2-device) skipped ($TARGET_NODE unavailable) ==="
fi


echo "=== Comparing Logs ==="
pixi run python -c "
import os
import sys

def parse_log(log_path):
    res = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if line.startswith('step '):
                    parts = line.split()
                    b_val = float(parts[2].split('=')[1])
                    j_val = float(parts[4].split('=')[1])
                    res.append((b_val, j_val))
    except Exception as e:
        print(f'Error reading {log_path}: {e}')
    return res

ref_log = '$LOG_DIR/run3_cpu_scipy.log'
all_candidates = [
    '$LOG_DIR/run1_gpu.log',
    '$LOG_DIR/run2_cpu_cpp.log',
    '$LOG_DIR/run4_gpu_tr.log',
    '$LOG_DIR/run5_cpu_cpp_tr.log',
    '$LOG_DIR/run6_cpu_scipy_tr.log',
    '$LOG_DIR/run7_multigpu.log',
    '$LOG_DIR/run8_multigpu_tr.log'
]
logs_to_check = [log for log in all_candidates if os.path.exists(log)]

ref_data = parse_log(ref_log)
if not ref_data:
    print('Failed to parse reference log from CPU scipy run (Test 3).')
    sys.exit(1)

fail = False
for log in logs_to_check:
    print(f'Comparing {os.path.basename(log)} against CPU scipy reference run (Test 3)...')
    data = parse_log(log)
    if len(data) != len(ref_data):
        print(f'  -> Lengths differ: {len(data)} vs {len(ref_data)}')
        fail = True
        continue
    
    max_diff = 0.0
    for (b1, j1), (b2, j2) in zip(ref_data, data):
        if abs(b1 - b2) > 1e-6:
            print(f'  -> B fields mismatch: {b1} vs {b2}')
            fail = True
            break
        max_diff = max(max_diff, abs(j1 - j2))
    
    print(f'  -> Max J_par difference: {max_diff:.3e}')
    if max_diff > 2e-6:
        print('  -> Result: Not passed (difference > 2e-6)')
        fail = True
    else:
        print('  -> Result: Passed')

if fail:
    sys.exit(1)
" || exit 1

echo "=== Performance Report ==="
for log in "$LOG_DIR"/run*.log; do
    if [ -f "$log" ]; then
        echo "--- $(basename "$log") ---"
        grep -E "Hysteresis loop finished|Total minimizer iterations|Total preconditioner iterations|Total function evaluations|Total Poisson" "$log" || echo "No timing data found."
        echo ""
    fi
done

rm -rf "$OUT_DIR"

echo "All tests finished!"
