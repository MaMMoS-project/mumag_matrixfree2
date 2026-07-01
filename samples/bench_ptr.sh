#!/bin/bash
export L=20
m="ptr"
OUT_DIR="bench_${m}"
mkdir -p ${OUT_DIR}
mkdir -p ${OUT_DIR}_warmup

echo "Warmup run for $m (compiling XLA...)"
pixi run env PYTHONUNBUFFERED=1 python ../src/loop.py cube_${L}nm \
    --out-dir ${OUT_DIR}_warmup \
    --add-shell --layers 4 --K 2.0 --h0 2 \
    --method $m \
    --max-iter 200 \
    --operator-mode assembled > ${OUT_DIR}_warmup/warmup.log 2>&1

echo "Actual benchmark for $m..."
pixi run env PYTHONUNBUFFERED=1 python ../src/loop.py cube_${L}nm \
    --out-dir ${OUT_DIR} \
    --add-shell --layers 4 --K 2.0 --h0 2 \
    --method $m \
    --max-iter 200 \
    --operator-mode assembled \
    --verbose > ${OUT_DIR}/simulation.log 2>&1

echo "Done! Parsing stats..."
iters=$(grep -oP "          number of iterations   : \K\d+" ${OUT_DIR}/simulation.log | awk '{s+=$1} END {print s}')
preco=$(grep -oP "number of iterations for preco   : \K\d+" ${OUT_DIR}/simulation.log | awk '{s+=$1} END {print s}')
evals=$(grep -oP "number of function evaluations   : \K\d+" ${OUT_DIR}/simulation.log | awk '{s+=$1} END {print s}')
demag=$(grep -oP "number of iterations for demag   : \K\d+" ${OUT_DIR}/simulation.log | awk '{s+=$1} END {print s}')
timing=$(grep "Hysteresis loop finished in" ${OUT_DIR}/simulation.log)
echo "Method: $m | Iterations: $iters | Preco/Inner: $preco | Func Evals: $evals | Demag: $demag | $timing"

echo "Checking physics output (Energy and Polarization for last step):"
tail -n 1 ${OUT_DIR}/hysteresis.csv
