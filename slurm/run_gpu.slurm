#!/bin/bash
#SBATCH --job-name=mumag_gpu
#SBATCH --partition=dissSims
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

cd $SLURM_SUBMIT_DIR/..

export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

echo "1. Creating poly_gb mesh..."
pixi run -e cuda python3 src/mesh.py --geom poly_gb --extent 100,100,100 --h 3.0 --n 10 --gb-thickness 2.0 --neper-timeout 30 --out-name poly_gb_100nm

echo "2. Creating krn file..."
cat << 'EOF' > create_krn.py
import numpy as np

n_grains = 10
# Main phase: Nd2Fe14B
K1 = 4.9e6
Js = 1.61
A = 8.0e-12

# Grain boundary phase: Nd0.5Fe0.5
Js_gb = 0.273
A_gb = 4.59e-12

np.random.seed(42)
lines = []

# Generate grains
for i in range(n_grains):
    theta = np.abs(np.random.normal(0, 20)) * np.pi / 180.0
    phi = np.random.uniform(0, 2 * np.pi)
    lines.append(f"{theta:.6f} {phi:.6f} {K1} 0.0 {Js} {A}")

# Generate GB phase
lines.append(f"0.0 0.0 0.0 0.0 {Js_gb} {A_gb}")



with open('poly_gb_100nm.krn', 'w') as f:
    f.write("\n".join(lines) + "\n")
EOF

pixi run -e cuda python3 create_krn.py

echo "3. Creating p2 file..."
# Field aligned 1 degree off the z-axis
# sin(1 deg) = 0.0174524, cos(1 deg) = 0.9998477
cat << 'EOF' > poly_gb_100nm.p2
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
pixi run -e cuda python3 src/loop.py poly_gb_100nm --add-shell --verbose

echo "5. Plotting demagnetization curve..."
cat << 'EOF' > plot_demag_gpu.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('hyst_poly_gb_100nm/hysteresis.csv')
plt.figure()
plt.plot(df['B_ext_T'], df['J_par_T'], 'o-', linewidth=2)
plt.xlabel('Applied Field (T)')
plt.ylabel('Magnetization (T)')
plt.title('Demagnetization Curve (10-grain Nd2Fe14B + GB, GPU)')
plt.grid(True)
plt.savefig('demag_curve_gpu.png')
EOF

pixi run -e cuda python3 plot_demag_gpu.py
echo "Done! Plot saved to demag_curve_gpu.png"
