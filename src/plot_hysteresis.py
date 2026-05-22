#!/usr/bin/env python3
"""Module for plotting magnetization curves from CSV data."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_hysteresis(csv_path: str | Path, output_image: str | Path):
    """Plot a magnetization curve from a CSV file and save as an image."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]

    B_ext = data[:, 0]
    J_par = data[:, 1]

    plt.figure(figsize=(10, 6))
    plt.plot(B_ext, J_par, "o-", label="Magnetization Curve")

    plt.xlabel("Applied Field B_ext (T)")
    plt.ylabel("Magnetic Polarization J_par (T)")
    plt.title("Magnetization Curve")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    plt.savefig(output_image)
    print(f"Plot saved to {output_image}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/plot_hysteresis.py <csv_file> [output_image]")
        sys.exit(1)

    csv_file = sys.argv[1]
    output_png = sys.argv[2] if len(sys.argv) > 2 else Path(csv_file).with_suffix(".png")

    if Path(csv_file).exists():
        plot_hysteresis(csv_file, output_png)
    else:
        print(f"Error: {csv_file} not found.")
