"""Tools for visualizing 2D slices of 3D meshes."""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def slice_tet_mesh_z0(knt, ijk):
    """Slices a 3D tetrahedral mesh at z = 0 using a custom marching-tets method.

    Returns:
        x, y: coordinates of slice vertices
        triangles: triangle connectivity (M, 3)
        mat_ids: material IDs of each triangle (M,).
    """
    # Perturb z slightly to avoid degeneracies (nodes exactly on the slice plane)
    z = knt[:, 2] + 1e-10 * np.sin(knt[:, 0] * 123.45 + knt[:, 1] * 678.90)

    slice_pts = []
    triangles = []
    mat_ids = []

    # Map from mesh edge (u, v) with u < v to index in slice_pts
    edge_to_slice_idx = {}
    edges_list = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    # Extract connectivity and material IDs
    tets = ijk[:, :4]
    tets_mat = ijk[:, 4]

    for t_idx in range(tets.shape[0]):
        v_idx = tets[t_idx]
        mat = tets_mat[t_idx]

        zs = z[v_idx]
        signs = np.sign(zs)

        # If all vertices are on the same side of the plane, no intersection
        if np.all(signs > 0) or np.all(signs < 0):
            continue

        tet_intersect_pts = []
        for a, b in edges_list:
            sa, sb = signs[a], signs[b]
            if sa != sb:
                u, v = v_idx[a], v_idx[b]
                edge_key = (u, v) if u < v else (v, u)

                if edge_key in edge_to_slice_idx:
                    p_idx = edge_to_slice_idx[edge_key]
                else:
                    # Linearly interpolate to find the intersection point
                    zu, zv = z[u], z[v]
                    t = -zu / (zv - zu)
                    pu, pv = knt[u], knt[v]
                    p_int = pu + t * (pv - pu)

                    slice_pts.append(p_int[:2])  # keep only x, y
                    p_idx = len(slice_pts) - 1
                    edge_to_slice_idx[edge_key] = p_idx

                tet_intersect_pts.append(p_idx)

        # A plane-tetrahedron intersection is either a triangle (3 points) or a quad (4 points)
        if len(tet_intersect_pts) == 3:
            triangles.append(tet_intersect_pts)
            mat_ids.append(mat)
        elif len(tet_intersect_pts) == 4:
            # Sort the 4 vertices of the quad angularly to split it into two triangles
            q_pts = np.array([slice_pts[idx] for idx in tet_intersect_pts])
            centroid = q_pts.mean(axis=0)
            angles = np.arctan2(q_pts[:, 1] - centroid[1], q_pts[:, 0] - centroid[0])
            order = np.argsort(angles)
            ordered = [tet_intersect_pts[idx] for idx in order]

            # Split quad into two triangles: (0, 1, 2) and (0, 2, 3)
            triangles.append([ordered[0], ordered[1], ordered[2]])
            mat_ids.append(mat)
            triangles.append([ordered[0], ordered[2], ordered[3]])
            mat_ids.append(mat)

    return np.array(slice_pts), np.array(triangles), np.array(mat_ids)


def main():
    """CLI entry point to slice and visualize a 3D mesh."""
    ap = argparse.ArgumentParser(
        description="Slice a 3D tetrahedral mesh (.npz) at z=0 and plot material IDs and mesh grid."
    )
    ap.add_argument(
        "--mesh", type=str, default="box_with_shell.npz", help="Path to input NPZ mesh containing 'knt' and 'ijk'."
    )
    ap.add_argument("--out", type=str, default="viz/mesh_slice.png", help="Path to save the output slice image.")
    args = ap.parse_args()

    print(f"Loading {args.mesh}...")
    data = np.load(args.mesh)
    knt = data["knt"]
    ijk = data["ijk"]

    print(f"Slicing mesh with {len(knt)} nodes and {len(ijk)} tets...")
    pts_2d, triangles, mat_ids = slice_tet_mesh_z0(knt, ijk)
    print(f"Extracted slice: {len(pts_2d)} nodes, {len(triangles)} triangles.")

    if len(pts_2d) == 0:
        print("Error: Extracted slice is empty. Verify that the mesh center is around z=0.")
        return

    x = pts_2d[:, 0]
    y = pts_2d[:, 1]

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Material IDs (filled triangles)
    ax = axes[0]
    unique_mats = np.unique(mat_ids)
    tc = ax.tripcolor(x, y, triangles, facecolors=mat_ids, cmap="tab10", edgecolors="none", vmin=0, vmax=9)
    fig.colorbar(tc, ax=ax, label="Material ID", ticks=unique_mats)
    ax.set_title("Material IDs (z = 0 Slice)", fontsize=14)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Mesh Grid (wireframe showing grading)
    ax = axes[1]
    ax.triplot(x, y, triangles, color="black", linewidth=0.2, alpha=0.7)
    ax.scatter(x, y, color="red", s=0.5, alpha=0.8)
    ax.set_title("Mesh Grid / Grading (z = 0 Slice)", fontsize=14)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(f"Tetrahedral Mesh Slice at z = 0 ({args.mesh})", fontsize=16)

    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved visualization to {args.out}")


if __name__ == "__main__":
    main()
