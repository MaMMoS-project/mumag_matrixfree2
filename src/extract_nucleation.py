import sys

import numpy as np


def extract_nucleation(csv_file: str) -> None:
    """Extract and print the nucleation field from a hysteresis CSV file.

    Args:
        csv_file (str): Path to the CSV file containing hysteresis data.
    """
    data = np.genfromtxt(csv_file, delimiter=",", names=True)
    b_ext = data["B_ext_T"]
    j_par = data["J_par_T"]

    # Calculate first derivative dM/dH
    dM_dH = np.gradient(j_par, b_ext)

    # Calculate second derivative d^2M/dH^2
    d2M_dH2 = np.gradient(dM_dH, b_ext)

    # We find where dM_dH exceeds 1% of its maximum to avoid numerical noise.
    max_dM_dH = np.max(dM_dH)

    if max_dM_dH < 1e-3:
        print("Nucleation not observed in the given field range.")
        return

    threshold = 0.01 * max_dM_dH

    nucleation_indices = np.where(dM_dH > threshold)[0]

    if len(nucleation_indices) > 0:
        idx = nucleation_indices[0]
        hnuc = b_ext[idx]
        print(f"Computed Nucleation Field (dM/dH > 1% max): {hnuc:.4f} T")

        # Also find the peak of the second derivative
        idx_2nd_deriv = np.argmax(np.abs(d2M_dH2))
        hnuc_2nd = b_ext[idx_2nd_deriv]
        print(f"Computed Nucleation Field (max |d²M/dH²|): {hnuc_2nd:.4f} T")

        print("Theoretical Nucleation Field: +0.5282 T")
    else:
        print("Nucleation not observed in the given field range.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_nucleation.py <csv_file>")
        sys.exit(1)
    extract_nucleation(sys.argv[1])
