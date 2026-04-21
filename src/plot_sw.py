"""plot_sw.py

Plotting script for Stoner-Wohlfarth verification.
Reads sw_summary.csv using numpy and creates a comparison plot.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_results():
    """Plot Stoner-Wohlfarth switching field comparison.

    Reads 'sw_summary.csv', compares simulated switching fields with 
    analytical theory, and saves the plot as 'sw_plot.png'.
    """
    csv_path = Path("hyst_no_demag/sw_summary.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    # Skip header
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    angle_deg = data[:, 0]
    B_sw_exp_T = data[:, 1]
    B_sw_theory_T = data[:, 2]
    
    plt.figure(figsize=(10, 6))
    plt.plot(angle_deg, B_sw_theory_T, 'r-', label='Theory (Stoner-Wohlfarth)')
    plt.plot(angle_deg, B_sw_exp_T, 'bo', label='Experiment (Simulation)')
    
    plt.xlabel('Field Angle (degrees)')
    plt.ylabel('Switching Field (Tesla)')
    plt.title('Stoner-Wohlfarth Switching Field vs. Angle')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_plot = "hyst_no_demag/sw_plot.png"
    plt.savefig(output_plot, dpi=300)
    print(f"Plot saved to {output_plot}")

if __name__ == "__main__":
    plot_results()
