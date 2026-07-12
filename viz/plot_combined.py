import numpy as np
import matplotlib.pyplot as plt

pcohen_hs_csv = '/ceph/home/schrefl/jax_dev/test_matrixfree2/Nd0.5Fe0.5/results_pcohen_hs_assembled_tau1/hysteresis.csv'
lbfgs_csv = '/ceph/home/schrefl/jax_dev/test_matrixfree2/Nd0.5Fe0.5/results_dplbfgs_assembled_tau1/hysteresis.csv'
pcohen_csv = '/ceph/home/schrefl/jax_dev/test_matrixfree2/Nd0.5Fe0.5/results_pcohen_assembled_tau1_181774/hysteresis.csv'

df_pcohen_hs = np.loadtxt(pcohen_hs_csv, delimiter=',', skiprows=1)
df_lbfgs = np.loadtxt(lbfgs_csv, delimiter=',', skiprows=1)
df_pcohen = np.loadtxt(pcohen_csv, delimiter=',', skiprows=1)

plt.figure(figsize=(8, 6))

# Plot pcohen_hs (correct physics)
plt.plot(df_pcohen_hs[:, 1], df_pcohen_hs[:, 2], 'o-', label='CG (Hestenes-Stiefel) [Correct]', color='blue', markersize=5)

# Plot lbfgs (premature flipping)
plt.plot(df_lbfgs[:, 1], df_lbfgs[:, 2], 's--', label='L-BFGS [Vaults Barrier]', color='red', markersize=5)

# Plot pcohen PR (premature flipping)
plt.plot(df_pcohen[:, 1], df_pcohen[:, 2], '^:', label='CG (Polak-Ribière) [Vaults Barrier]', color='orange', markersize=6)

plt.xlabel('Applied Field B (T)', fontsize=14)
plt.ylabel('Magnetization $J_{\\parallel}$ (T)', fontsize=14)
plt.title('Metastable Well Vaulting: Minimizer Failures', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Annotate the premature flip
plt.annotate('Premature reversal\nat -2.174 T', 
             xy=(-2.174, -0.5), xytext=(-1.8, -0.2),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=12, color='black', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

plt.savefig('/ceph/home/schrefl/jax_dev/mumag_matrixfree2/minimizer_failures.png', dpi=300, bbox_inches='tight')
print("Saved minimizer_failures.png")
