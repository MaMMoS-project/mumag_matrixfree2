import matplotlib.pyplot as plt
import numpy as np
import os

# Increase global font size
plt.rcParams.update({'font.size': 18})

def read_csv(path):
    angles = []
    b_sw = []
    with open(path, 'r') as f:
        next(f) # header
        for line in f:
            if not line.strip(): continue
            a, b = line.strip().split(',')
            angles.append(float(a))
            b_sw.append(float(b))
    return np.array(angles), np.array(b_sw)

# Values from test_stoner_wohlfarth_me.py
Js_si = 1.6
K1_si = 4.3e6
mu0 = 4.0 * np.pi * 1e-7
Hk_tesla = mu0 * 2.0 * K1_si / Js_si # Anisotropy field in Tesla

files = {
    "phi=0": "greene_results/sw_summary_me_phi0.csv",
    "phi=45": "greene_results/sw_summary_me_phi45.csv",
    "phi=90": "greene_results/sw_summary_me_phi90.csv"
}
markers = ['o', 's', '^'] # Different symbols for different phi

plt.figure(figsize=(10, 8))

for (label, path), marker in zip(files.items(), markers):
    if os.path.exists(path):
        theta, b_sw = read_csv(path)
        plt.plot(theta, b_sw / Hk_tesla, marker=marker, linestyle='None', label=label, markersize=8)

# Stoner-Wohlfarth reference
theta_sw = np.linspace(0.1, 89.9, 100)
theta_rad = np.deg2rad(theta_sw)
# H_sw(theta) = Hk / (cos(theta)^(2/3) + sin(theta)^(2/3))^(3/2)
h_sw_norm = 1.0 / (np.cos(theta_rad)**(2/3) + np.sin(theta_rad)**(2/3))**(3/2)
plt.plot(theta_sw, h_sw_norm, 'k--', label='Stoner-Wohlfarth', linewidth=2)

plt.xlabel(r'Angle $\theta$ (deg)')
plt.ylabel('normalized switching field')
plt.title(r'Switching Field vs Angle $\theta$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.ylim(0, 1.3)
plt.tight_layout()
plt.savefig('switching_field_plot.png', dpi=300)
print("Plot saved to switching_field_plot.png")
