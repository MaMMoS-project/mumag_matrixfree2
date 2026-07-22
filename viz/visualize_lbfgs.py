import matplotlib.pyplot as plt
import numpy as np

pcohen_csv = "/ceph/home/schrefl/jax_dev/test_matrixfree2/Nd0.5Fe0.5/results_pcohen_hs_assembled_tau1/hysteresis.csv"
lbfgs_csv = "/ceph/home/schrefl/jax_dev/test_matrixfree2/Nd0.5Fe0.5/results_dplbfgs_assembled_tau1/hysteresis.csv"

df_pcohen = np.loadtxt(pcohen_csv, delimiter=",", skiprows=1)
df_lbfgs = np.loadtxt(lbfgs_csv, delimiter=",", skiprows=1)

plt.figure(figsize=(8, 6))

# Plot pcohen (correct physics)
# Column 1 is B_ext_T, Column 2 is J_par_T
plt.plot(df_pcohen[:, 1], df_pcohen[:, 2], "o-", label="pcohen (Correct, Bc = -2.246 T)", color="blue", markersize=5)

# Plot lbfgs (premature flipping)
plt.plot(df_lbfgs[:, 1], df_lbfgs[:, 2], "s--", label="L-BFGS (Vaults Barrier at -2.174 T)", color="red", markersize=5)

plt.xlabel("Applied Field B (T)", fontsize=14)
plt.ylabel("Magnetization $J_{\\parallel}$ (T)", fontsize=14)
plt.title("Metastable Well Vaulting: L-BFGS Failure", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(fontsize=12)

# Annotate the premature flip
plt.annotate(
    "Premature reversal",
    xy=(-2.174, -0.5),
    xytext=(-1.8, -0.2),
    arrowprops=dict(facecolor="black", shrink=0.05),
    fontsize=12,
    color="red",
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="red", facecolor="white"),
)

plt.savefig("/ceph/home/schrefl/jax_dev/mumag_matrixfree2/lbfgs_failure.png", dpi=300, bbox_inches="tight")
print("Saved lbfgs_failure.png")
