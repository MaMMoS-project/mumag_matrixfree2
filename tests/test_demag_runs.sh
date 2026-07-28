#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUT_DIR="$DIR/tmp_test_out"
LOG_DIR="$DIR/logs"

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

rm -f "$LOG_DIR"/run*.log

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

echo "=== Test 1: GPU with --benchmark ==="
rm -f "$OUT_DIR"/cube_20nm.*
pixi run -e cuda python ../src/loop.py cube_20nm --add-shell --benchmark --out-dir "$OUT_DIR" > "$LOG_DIR/run1_gpu.log" 2>&1
check_switching_field "$OUT_DIR/cube_20nm.mh"

echo "=== Test 2: CPU with C++ minimizer ==="
rm -f "$OUT_DIR"/cube_20nm.*
pixi run compile > "$LOG_DIR/compile.log" 2>&1
JAX_PLATFORMS=cpu pixi run python ../src/loop.py cube_20nm --add-shell --out-dir "$OUT_DIR" > "$LOG_DIR/run2_cpu_cpp.log" 2>&1
check_switching_field "$OUT_DIR/cube_20nm.mh"

echo "=== Test 3: CPU with scipy ==="
rm -f "$OUT_DIR"/cube_20nm.*
JAX_PLATFORMS=cpu pixi run python ../src/loop.py cube_20nm --add-shell --out-dir "$OUT_DIR" --cpu-spmv-backend scipy --no-cpp-mkl --poisson-solver jax > "$LOG_DIR/run3_cpu_scipy.log" 2>&1
check_switching_field "$OUT_DIR/cube_20nm.mh"

echo "=== Test 4: GPU with --benchmark and --method tr ==="
rm -f "$OUT_DIR"/cube_20nm.*
pixi run -e cuda python ../src/loop.py cube_20nm --add-shell --benchmark --method tr --out-dir "$OUT_DIR" > "$LOG_DIR/run4_gpu_tr.log" 2>&1
check_switching_field "$OUT_DIR/cube_20nm.mh"

echo "=== Test 5: CPU with C++ minimizer and --method tr ==="
rm -f "$OUT_DIR"/cube_20nm.*
JAX_PLATFORMS=cpu pixi run python ../src/loop.py cube_20nm --add-shell --method tr --out-dir "$OUT_DIR" > "$LOG_DIR/run5_cpu_cpp_tr.log" 2>&1
check_switching_field "$OUT_DIR/cube_20nm.mh"

echo "=== Test 6: CPU with scipy and --method tr ==="
rm -f "$OUT_DIR"/cube_20nm.*
JAX_PLATFORMS=cpu pixi run python ../src/loop.py cube_20nm --add-shell --method tr --out-dir "$OUT_DIR" --cpu-spmv-backend scipy --no-cpp-mkl --poisson-solver jax > "$LOG_DIR/run6_cpu_scipy_tr.log" 2>&1
check_switching_field "$OUT_DIR/cube_20nm.mh"

echo "=== Comparing Logs ==="
pixi run python -c "
import sys

def parse_log(log_path):
    res = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if line.startswith('step '):
                    parts = line.split()
                    # Example: step 00016 B=-6.000000e+00 T J_par=+1.576612e+00 T
                    b_val = float(parts[2].split('=')[1])
                    j_val = float(parts[4].split('=')[1])
                    res.append((b_val, j_val))
    except Exception as e:
        print(f'Error reading {log_path}: {e}')
    return res

ref_log = '$LOG_DIR/run2_cpu_cpp.log'
logs_to_check = [
    '$LOG_DIR/run1_gpu.log',
    '$LOG_DIR/run3_cpu_scipy.log',
    '$LOG_DIR/run4_gpu_tr.log',
    '$LOG_DIR/run5_cpu_cpp_tr.log',
    '$LOG_DIR/run6_cpu_scipy_tr.log'
]

ref_data = parse_log(ref_log)
if not ref_data:
    print('Failed to parse reference log from CPU C++ run.')
    sys.exit(1)

fail = False
for log in logs_to_check:
    print(f'Comparing {log} against CPU C++ run...')
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
for log in "$LOG_DIR"/run1_gpu.log "$LOG_DIR"/run2_cpu_cpp.log "$LOG_DIR"/run3_cpu_scipy.log "$LOG_DIR"/run4_gpu_tr.log "$LOG_DIR"/run5_cpu_cpp_tr.log "$LOG_DIR"/run6_cpu_scipy_tr.log; do
    echo "--- $(basename "$log") ---"
    grep -E "Hysteresis loop finished|Total minimizer iterations|Total preconditioner iterations|Total function evaluations|Total Poisson" "$log" || echo "No timing data found."
    echo ""
done

rm -rf "$OUT_DIR"

echo "All tests finished!"
