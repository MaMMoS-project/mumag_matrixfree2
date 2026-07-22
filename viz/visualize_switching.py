import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

files = [
    "/ceph/home/schrefl/jax_dev/test_matrixfree2/Nd0.5Fe0.5/results_pcohen_hs_assembled_tau1/state_cfg00000_B+2.0000e+00T.vtu",
    "/ceph/home/schrefl/jax_dev/test_matrixfree2/Nd0.5Fe0.5/results_pcohen_hs_assembled_tau1/state_cfg00001_B-1.8020e+00T.vtu",
    "/ceph/home/schrefl/jax_dev/test_matrixfree2/Nd0.5Fe0.5/results_pcohen_hs_assembled_tau1/state_cfg00002_B-2.2460e+00T.vtu",
]

titles = ["B = +2.0000 T", "B = -1.8020 T", "B = -2.2460 T"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, f in enumerate(files):
    mesh = pv.read(f)

    # Filter out airbox (highest material id)
    if "mat_id" in mesh.cell_data:
        mat_ids = mesh.cell_data["mat_id"]
        max_id = np.max(mat_ids)
        mesh = mesh.extract_cells(mat_ids < max_id)
    elif "mat_id" in mesh.point_data:
        mat_ids = mesh.point_data["mat_id"]
        max_id = np.max(mat_ids)
        mesh = mesh.extract_points(mat_ids < max_id)

    # Get points and magnetization
    pts = mesh.points
    if "m" in mesh.point_data:
        m = mesh.point_data["m"]
    elif "m" in mesh.cell_data:
        # Interpolate cell data to point data for easy scatter plotting
        mesh = mesh.cell_data_to_point_data()
        m = mesh.point_data["m"]

    m_z = m[:, 2]

    # Extract a slice near the middle of Z
    z_min, z_max = np.min(pts[:, 2]), np.max(pts[:, 2])
    z_mid = (z_min + z_max) / 2.0

    # Take points within a thin slice
    mask = np.abs(pts[:, 2] - z_mid) < 5.0
    x_slice = pts[mask, 0]
    y_slice = pts[mask, 1]
    mz_slice = m_z[mask]

    ax = axes[i]
    sc = ax.scatter(x_slice, y_slice, c=mz_slice, cmap="coolwarm", vmin=-1, vmax=1, s=1)
    ax.set_title(titles[i], fontsize=14)
    ax.set_aspect("equal")
    ax.axis("off")

cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
cbar.set_label("Magnetization $m_z$", fontsize=12)

plt.suptitle("Magnetization Switching in Nd0.5Fe0.5 Cube (Mid-plane Slice)", fontsize=16)
plt.savefig("/ceph/home/schrefl/jax_dev/mumag_matrixfree2/switching_process.png", dpi=300, bbox_inches="tight")
print("Saved switching_process.png")
