#!/bin/bash
#SBATCH --job-name=mumag_cpu
#SBATCH --partition=dissSims
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd $SLURM_SUBMIT_DIR/..

export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu

# Optimize Intel MKL threading
export MKL_INTERFACE_LAYER=LP64
export MKL_THREADING_LAYER=GNU
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_DYNAMIC=FALSE

# Thread affinity for OpenMP
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_WAIT_POLICY=ACTIVE
export OMP_SCHEDULE=static

# Optimize other BLAS/Vector engines
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK

# Optimize JAX/XLA CPU thread pool
export JAX_XLA_CPU_INTRA_OP_PARALLELISM_THREADS=$SLURM_CPUS_PER_TASK
export JAX_XLA_CPU_INTER_OP_PARALLELISM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"

echo "0. Compiling C++ backend..."
pixi run -e cpu compile

echo "1. Creating mesh..."
pixi run -e cpu env MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=GNU python3 src/mesh.py --geom box --extent 20,20,20 --h 2.0 --out-name cube_20nm

echo "2. Creating krn file..."
cat << 'EOF' > cube_20nm.krn
0.0 0.0 4.9e6 0.0 1.61 8e-12
EOF

echo "3. Creating p2 file..."
# Field aligned 1 degree off the z-axis (easy axis)
# sin(1 deg) = 0.0174524, cos(1 deg) = 0.9998477
cat << 'EOF' > cube_20nm.p2
[mesh]
size = 1e-9

[initial state]
mx = 0.0174524
my = 0.0
mz = 0.9998477

[field]
hstart = 2.0
hfinal = -8.0
hstep = -0.05
hx = 0.0174524
hy = 0.0
hz = 0.9998477

[minimizer]
method = pcohen_hs
tol_fun = 1e-8
eps_a = 1e-12
EOF

echo "4. Running simulation..."
pixi run -e cpu env MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=GNU python3 src/loop.py cube_20nm --add-shell --verbose

echo "5. Plotting demagnetization curve..."
cat << 'EOF' > plot_demag_cpu.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('hyst_cube_20nm/hysteresis.csv')
plt.figure()
plt.plot(df['B_ext_T'], df['J_par_T'], 'o-', linewidth=2)
plt.xlabel('Applied Field (T)')
plt.ylabel('Magnetization (T)')
plt.title('Demagnetization Curve (Nd2Fe14B, CPU)')
plt.grid(True)
plt.savefig('demag_curve_cpu.png')
EOF

pixi run -e cpu env MKL_INTERFACE_LAYER=LP64 MKL_THREADING_LAYER=GNU python3 plot_demag_cpu.py
echo "Done! Plot saved to demag_curve_cpu.png"
