import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_hysteresis(csv_path: str | Path, output_image: str | Path):
    """Plot a magnetization curve from a CSV file and save as an image.

    Args:
        csv_path (str | Path): Path to the input CSV file.
        output_image (str | Path): Path where the output plot will be saved.

    Example:
        >>> plot_hysteresis("hysteresis.csv", "plot.png")
    """
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
