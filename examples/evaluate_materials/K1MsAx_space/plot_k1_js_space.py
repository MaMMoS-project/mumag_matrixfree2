"""Module to calculate, sample, export, and visualize the K1-Js search space corridor."""

from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import mammos_entity as me


def compute_corridor_boundaries(
    k1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the lower and upper saturation polarization boundaries.

    Args:
        k1: Array of magnetocrystalline anisotropy constant values in MJ/m^3.

    Returns:
        A tuple containing (js_lower, js_upper) boundary arrays in Tesla.
    """
    js_lower = 0.1 * k1 + 0.65
    js_upper = 0.1 * k1 + 1.22
    return js_lower, js_upper


def sample_corridor_points(
    k1_min: float = 0.9,
    k1_max: float = 8.5,
    num_k1: int = 12,
    num_js: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates an evenly spaced grid of (K1, Js) points within the corridor.

    Args:
        k1_min: Minimum K1 value in MJ/m^3.
        k1_max: Maximum K1 value in MJ/m^3.
        num_k1: Number of sampling steps along the K1 axis.
        num_js: Number of sampling steps between lower and upper Js boundaries.

    Returns:
        A tuple of 1D arrays (k1_points, js_points) representing sample pairs.
    """
    k1_samples = np.linspace(k1_min, k1_max, num_k1)
    k1_list = []
    js_list = []

    for k1_val in k1_samples:
        js_low = 0.1 * k1_val + 0.65
        js_high = 0.1 * k1_val + 1.22
        js_samples = np.linspace(js_low, js_high, num_js)
        for js_val in js_samples:
            k1_list.append(k1_val)
            js_list.append(js_val)

    return np.array(k1_list), np.array(js_list)


def export_samples_to_csv(
    k1_pairs: np.ndarray,
    js_pairs: np.ndarray,
    csv_filename: str = "corridor_samples.csv",
) -> None:
    """Exports sample (K1, Js) pairs to a CSV file with unique index IDs.

    Args:
        k1_pairs: 1D array of K1 values in MJ/m^3.
        js_pairs: 1D array of Js values in Tesla.
        csv_filename: Path to save the output CSV file.
    """
    indices = np.arange(len(k1_pairs))
    k1_j_m3 = k1_pairs * 1e6

    collection = me.EntityCollection(
        "K1-Js corridor samples for material evaluation.",
        index=me.Entity("Number", indices, ""),
        K1_MJ_m3=me.Entity("MagnetocrystallineAnisotropyConstantK1", k1_pairs, "MJ / m^3"),
        K1_J_m3=me.Entity("MagnetocrystallineAnisotropyConstantK1", k1_j_m3, "J / m^3"),
        Js_T=me.Entity("SpontaneousMagneticPolarization", js_pairs, "T")
    )
    collection.to_csv(csv_filename)


def plot_corridor(
    k1_min: float = 0.9,
    k1_max: float = 8.5,
    num_points: int = 21,
    num_sample_k1: int = 12,
    num_sample_js: int = 5,
    output_filename: str = "k1_js_corridor.png",
    csv_filename: str = "corridor_samples.csv",
) -> None:
    """Generates and saves a matplotlib plot of the K1-Js search space corridor.

    Args:
        k1_min: Minimum value for K1 in MJ/m^3.
        k1_max: Maximum value for K1 in MJ/m^3.
        num_points: Number of points for continuous boundary plotting.
        num_sample_k1: Number of grid steps along K1 for sampled pairs.
        num_sample_js: Number of grid steps along Js for sampled pairs.
        output_filename: File path to save the generated plot image.
        csv_filename: File path to save exported CSV sample data.
    """
    plt.rcParams.update({"font.size": 14})

    k1 = np.linspace(k1_min, k1_max, num_points)
    js_lower, js_upper = compute_corridor_boundaries(k1)
    k1_pairs, js_pairs = sample_corridor_points(
        k1_min=k1_min,
        k1_max=k1_max,
        num_k1=num_sample_k1,
        num_js=num_sample_js,
    )

    export_samples_to_csv(k1_pairs, js_pairs, csv_filename)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Boundary 1: Lower limit Js = 0.1 * K1 + 0.65
    ax.plot(
        k1,
        js_lower,
        color="#1f77b4",
        linestyle="--",
        marker="s",
        markevery=3,
        linewidth=2,
        markersize=7,
        label=r"Lower limit: $J_s = 0.1 K_1 + 0.65$",
    )

    # Boundary 2: Upper limit Js = 0.1 * K1 + 1.22
    ax.plot(
        k1,
        js_upper,
        color="#d62728",
        linestyle="-",
        marker="o",
        markevery=3,
        linewidth=2,
        markersize=7,
        label=r"Upper limit: $J_s = 0.1 K_1 + 1.22$",
    )

    # Highlight corridor
    ax.fill_between(
        k1,
        js_lower,
        js_upper,
        color="#17becf",
        alpha=0.25,
        label="Search Corridor",
    )

    # Overlay sample pairs
    ax.scatter(
        k1_pairs,
        js_pairs,
        color="#2ca02c",
        edgecolor="black",
        s=45,
        zorder=5,
        label=f"Sample Pairs ($N={len(k1_pairs)}$)",
    )

    # Annotate point index numbers
    for idx, (x_val, y_val) in enumerate(zip(k1_pairs, js_pairs)):
        ax.annotate(
            str(idx),
            (x_val, y_val),
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=8,
            fontweight="bold",
            color="#111111",
            zorder=6,
        )

    ax.set_xlabel(r"$K_1\ (\mathrm{MJ/m}^3)$")
    ax.set_ylabel(r"$J_s\ (\mathrm{T})$")
    ax.set_xlim(k1_min - 0.2, k1_max + 0.4)
    ax.set_ylim(0.6, 2.2)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_corridor()
