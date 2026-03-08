import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_hysteresis(csv_path, output_image):
    # Load the data using numpy
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    
    # Extract B_ext_T and J_par_T columns
    B_ext = data[:, 0]
    J_par = data[:, 1]
    
    # Plot J_par_T vs B_ext_T
    plt.figure(figsize=(10, 6))
    plt.plot(B_ext, J_par, 'o-', label='Magnetization Curve')
    
    # Add labels and title
    plt.xlabel('Applied Field B_ext (T)')
    plt.ylabel('Magnetic Polarization J_par (T)')
    plt.title('Hard-Axis Magnetization Curve (X-axis)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    plt.savefig(output_image)
    print(f"Plot saved to {output_image}")

if __name__ == "__main__":
    csv_file = "test_hyst_x_out/hysteresis.csv"
    output_png = "test_hyst_x_out/hysteresis_plot.png"
    
    if Path(csv_file).exists():
        plot_hysteresis(csv_file, output_png)
    else:
        print(f"Error: {csv_file} not found.")
