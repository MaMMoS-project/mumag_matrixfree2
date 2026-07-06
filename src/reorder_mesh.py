"""reorder_mesh.py.

Spatially or topologically reorder elements and nodes of a tetrahedral NPZ mesh.
Supports Morton (Z-order) space-filling curve (optimal for GPU cuSPARSE CSR) and
Reverse Cuthill-McKee (RCM) graph bandwidth minimization (optimal for CPU MKL).

Author: Antigravity
License: MIT
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def morton_encode(q_coords: np.ndarray) -> np.ndarray:
    """Interleave bits of three 10-bit integer coordinates into a 30-bit Morton key."""
    x = q_coords[:, 0].astype(np.int64)
    y = q_coords[:, 1].astype(np.int64)
    z = q_coords[:, 2].astype(np.int64)

    code = np.zeros(q_coords.shape[0], dtype=np.int64)
    for i in range(10):
        code |= ((x >> i) & 1) << (3 * i + 2)
        code |= ((y >> i) & 1) << (3 * i + 1)
        code |= ((z >> i) & 1) << (3 * i)
    return code


def reorder_mesh(in_path: str, out_path: str, target: str = "gpu") -> None:
    print(f"Loading original mesh from {in_path}...")
    data = np.load(in_path)
    knt = np.asarray(data["knt"], dtype=np.float64)
    ijk = np.asarray(data["ijk"])

    E = ijk.shape[0]
    N = knt.shape[0]
    print(f"Mesh has {N} nodes and {E} elements.")

    # 1. Bounding box calculation for quantization
    min_knt = np.min(knt, axis=0)
    max_knt = np.max(knt, axis=0)
    denom = max_knt - min_knt
    denom = np.where(denom == 0, 1.0, denom)

    if target == "cpu":
        print("Sorting nodes using Reverse Cuthill-McKee (RCM) on graph adjacency...")
        import scipy.sparse as sp
        from scipy.sparse.csgraph import reverse_cuthill_mckee

        # Build node-to-node adjacency graph from element connectivity (tetrahedron edges)
        conn = ijk[:, :4].astype(np.int64)
        e0 = conn[:, [0, 1]]
        e1 = conn[:, [0, 2]]
        e2 = conn[:, [0, 3]]
        e3 = conn[:, [1, 2]]
        e4 = conn[:, [1, 3]]
        e5 = conn[:, [2, 3]]
        all_edges = np.vstack([e0, e1, e2, e3, e4, e5])

        rows = all_edges[:, 0]
        cols = all_edges[:, 1]
        vals = np.ones(all_edges.shape[0], dtype=bool)

        # Build symmetric adjacency matrix
        adj = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()

        # Run RCM reordering
        node_perm = reverse_cuthill_mckee(adj, symmetric_mode=True)

        # Construct inverse node mapping to update element connectivity
        inv_node_perm = np.zeros(N, dtype=np.int32)
        inv_node_perm[node_perm] = np.arange(N, dtype=np.int32)

        # Sort elements by their average mapped node index
        print("Sorting elements by average mapped node index...")
        mapped_nodes = inv_node_perm[conn]
        mean_mapped = np.mean(mapped_nodes, axis=1)
        elem_perm = np.argsort(mean_mapped)

    else:
        print("Sorting nodes and elements using 3D Morton spatial curve...")
        # 2. Sort nodes by Morton code
        q_knt = ((knt - min_knt) / denom * 1023).astype(np.int32)
        morton_nodes = morton_encode(q_knt)
        node_perm = np.argsort(morton_nodes)

        # Construct inverse node mapping
        inv_node_perm = np.zeros(N, dtype=np.int32)
        inv_node_perm[node_perm] = np.arange(N, dtype=np.int32)

        # 3. Sort elements by their centroids' Morton code
        conn = ijk[:, :4].astype(np.int64)
        centroids = np.mean(knt[conn], axis=1)
        q_centroids = ((centroids - min_knt) / denom * 1023).astype(np.int32)
        morton_elements = morton_encode(q_centroids)
        elem_perm = np.argsort(morton_elements)

    # 4. Apply reordering
    print("Applying permutations...")
    knt_sorted = knt[node_perm]

    # Map connectivity elements using element permutation and node inverse mapping
    ijk_sorted = ijk[elem_perm].copy()
    ijk_sorted[:, :4] = inv_node_perm[ijk_sorted[:, :4].astype(np.int64)]

    # 5. Measure node footprint reduction (for element chunking verification)
    chunk_size = 200_000
    num_chunks = int(np.ceil(E / chunk_size))

    unique_orig = []
    unique_sorted = []
    for i in range(num_chunks):
        s = i * chunk_size
        e = min((i + 1) * chunk_size, E)
        unique_orig.append(len(np.unique(ijk[s:e, :4])))
        unique_sorted.append(len(np.unique(ijk_sorted[s:e, :4])))

    print(f"Original average unique nodes per 200k chunk: {np.mean(unique_orig):.1f}")
    print(f"Reordered average unique nodes per 200k chunk: {np.mean(unique_sorted):.1f}")
    print(f"Reduction factor: {np.mean(unique_orig) / np.mean(unique_sorted):.2f}x")

    # 6. Save reordered mesh
    print(f"Saving reordered mesh to {out_path}...")
    np.savez(
        out_path,
        knt=knt_sorted,
        ijk=ijk_sorted,
        node_perm=node_perm,
        inv_node_perm=inv_node_perm,
    )
    print("[ok] Finished reordering mesh.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Spatially or topologically reorder mesh nodes and elements.")
    ap.add_argument("--in-mesh", required=True, help="Input NPZ mesh file.")
    ap.add_argument("--out-mesh", help="Output NPZ mesh file. Defaults to [in-mesh]_[target_suffix].npz")
    ap.add_argument(
        "--target",
        type=str,
        default="gpu",
        choices=["cpu", "gpu"],
        help="Optimization target. 'gpu' uses 3D Morton curve. 'cpu' uses Reverse Cuthill-McKee (RCM) graph sorting.",
    )
    args = ap.parse_args()

    in_path = Path(args.in_mesh)
    if not in_path.exists():
        ap.error(f"Input mesh not found: {in_path}")

    out_path = args.out_mesh
    if not out_path:
        suffix = "sorted" if args.target == "gpu" else "rcm"
        out_path = in_path.with_name(f"{in_path.stem}_{suffix}.npz")
    else:
        out_path = Path(out_path)

    reorder_mesh(str(in_path), str(out_path), target=args.target)


if __name__ == "__main__":
    main()
