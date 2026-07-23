"""Convert Salome meshes to .npz format."""

import numpy as np

""" Created with Microsoft Copilot. """


def knt_ijk_to_npz(knt_path: str, ijk_path: str, npz_path: str) -> None:
    """Convert KNT/IJK mesh files to an NPZ file.

    Args:
        knt_path (str): Path to the .knt file (points).
        ijk_path (str): Path to the .ijk file (connectivity + optional data).
        npz_path (str): Path to output .npz file.

    Raises:
        ValueError: If input format is invalid.
    """
    # ---- Load points (.knt) ----
    try:
        pts = np.loadtxt(knt_path, dtype=np.float64)
    except Exception as e:
        raise ValueError(f"Failed to read KNT file: {e}")  # noqa: B904

    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("KNT file must contain at least 3 columns (x y z)")

    pts = pts[:, :3]  # ensure exactly 3 coords

    # ---- Load connectivity (.ijk) ----
    try:
        ijk_raw = np.loadtxt(ijk_path, dtype=np.int32)
    except Exception as e:
        raise ValueError(f"Failed to read IJK file: {e}")  # noqa: B904

    if ijk_raw.ndim == 1:
        ijk_raw = ijk_raw.reshape(1, -1)

    if ijk_raw.shape[1] < 4:
        raise ValueError("IJK file must have at least 4 columns (tetra indices)")

    # ---- Extract tetra connectivity ----
    tets = ijk_raw[:, :4].astype(np.int32)

    # ---- Detect indexing (1-based vs 0-based) ----
    if tets.min() == 1:
        # Convert to 0-based indexing
        tets -= 1

    # ---- Optional material IDs ----
    mat = None
    if ijk_raw.shape[1] >= 5:
        mat = ijk_raw[:, 4].astype(np.int32)

    # ---- Combine like your VTU version ----
    ijk = np.column_stack([tets, mat]) if mat is not None else tets

    # ---- Save NPZ ----
    np.savez(npz_path, knt=pts, ijk=ijk)


if __name__ == "__main__":
    # Example usage
    knt_ijk_to_npz("cube.knt", "cube.ijk", "cube.npz")
