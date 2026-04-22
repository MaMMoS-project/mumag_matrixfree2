"""test_hysteresis_x_axis.py

Test for magnetization curve along the hard axis (X).
Easy axis is Z. Field is applied along X.
Uses AMGCL preconditioner on a 20nm cube (NdFeB-like properties).
"""

from __future__ import annotations

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import time
from pathlib import Path

import jax.numpy as jnp

import add_shell
import mesh
from fem_utils import TetGeom, compute_node_volumes
from hysteresis_loop import LoopParams, run_hysteresis_loop
from io_utils import ensure_dir
from loop import compute_grad_phi_from_JinvT, compute_volume_JinvT


def write_inp(path: str, nodes_arr: np.ndarray, elements_arr: np.ndarray) -> None:
    """Write a tetrahedral mesh to an AVS UCD (.inp) file.

    Args:
        path (str): target output path.
        nodes_arr (np.ndarray): Node coordinates (N, 3).
        elements_arr (np.ndarray): Element connectivity and material ID (E, 5).
    """
    num_nodes = nodes_arr.shape[0]
    num_elems = elements_arr.shape[0]

    print(f"Writing INP file: {path}")
    with open(path, "w") as f:
        # header: nodes elements 0 0 0
        f.write(f"{num_nodes} {num_elems} 0 0 0\n")

        # Write nodes (1-based ID)
        for i in range(num_nodes):
            x, y, z = nodes_arr[i]
            f.write(f"{i + 1} {x} {y} {z}\n")

        # Write elements (1-based ID, tetra type)
        for i in range(num_elems):
            eid = i + 1
            mat_id = int(elements_arr[i, 4])
            nids = elements_arr[i, :4].astype(int) + 1  # 1-based node IDs
            nids_str = " ".join(map(str, nids))
            f.write(f"{eid} {mat_id} tet {nids_str}\n")


def run_benchmark(precond_type="amgcl", order=3, L_cube=20.0, h=2.0, layers=8) -> float:
    """Run a hard-axis magnetization benchmark for a given preconditioner.

    Creates a cube mesh, adds an airbox, and executes a field sweep along the X-axis.

    Args:
        precond_type (str, optional): solver preconditioner. Defaults to 'amgcl'.
        order (int, optional): Chebyshev order. Defaults to 3.
        L_cube (float, optional): physical side length. Defaults to 20.0.
        h (float, optional): target core mesh size. Defaults to 2.0.
        layers (int, optional): airbox layers. Defaults to 8.

    Returns:
        float: total simulation time in seconds.
    """
    print(f"\n=== Running Hysteresis with {precond_type.upper()} (order={order}) ===")

    print(f"Creating mesh: {L_cube}nm cube, h={h}nm, layers={layers}...")
    knt0, ijk0, _, _ = mesh.run_single_solid_mesher(
        geom="box",
        extent=f"{L_cube},{L_cube},{L_cube}",
        h=h,
        backend="grid",
        no_vis=True,
        return_arrays=True,
    )

    tmp_path = f"tmp_mesh_{precond_type}.npz"
    np.savez(tmp_path, knt=knt0, ijk=ijk0)
    knt, ijk = add_shell.run_add_shell_pipeline(
        in_npz=tmp_path, layers=layers, K=1.4, h0=h, verbose=False
    )
    if Path(tmp_path).exists():
        Path(tmp_path).unlink()

    # Save to INP
    out_dir = ensure_dir(f"hyst_{precond_type}")
    inp_path = out_dir / "mesh_20nm.inp"
    write_inp(str(inp_path), knt, ijk)

    tets = ijk[:, :4].astype(np.int64)
    mat_id = ijk[:, 4].astype(np.int32)
    conn32, volume, JinvT = compute_volume_JinvT(knt, tets)
    grad_phi = compute_grad_phi_from_JinvT(JinvT)
    boundary_mask = jnp.asarray(
        add_shell.find_outer_boundary_mask(tets, knt.shape[0]), dtype=jnp.float64
    )

    geom = TetGeom(
        conn=jnp.asarray(conn32, dtype=jnp.int32),
        volume=jnp.asarray(volume, dtype=jnp.float64),
        mat_id=jnp.asarray(mat_id, dtype=jnp.int32),
        grad_phi=jnp.asarray(grad_phi, dtype=jnp.float64),
    )

    # Material Properties (Normalized NdFeB-like as in test_energy.cpp)
    Js_si = 1.6  # Tesla
    K1_si = 4.3e6  # J/m^3
    A_si = 7.7e-12  # J/m

    MU0_SI = 4e-7 * np.pi
    Kd_ref = (Js_si**2) / (2.0 * MU0_SI)

    # Normalized properties
    A_red = (A_si * 1e18) / Kd_ref
    K1_red_val = K1_si / Kd_ref
    Js_red_val = 1.0  # Js_si / Js_si

    # mat_id 1 = cube, mat_id 2 = air (shell)
    Js_lookup = np.array([Js_red_val, 0.0])
    K1_lookup = np.array([K1_red_val, 0.0])
    A_lookup_red = np.array([A_red, 0.0])
    k_easy_lookup = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    is_mag = mat_id == 1
    V_mag = np.sum(volume[is_mag])

    m0_vec = np.array([0.0, 0.0, 1.0])
    m0 = np.tile(m0_vec, (knt.shape[0], 1))

    params = LoopParams(
        h_dir=np.array([1.0, 0.0, 0.0]),
        B_start=0.0,
        B_end=2.0,
        dB=0.5,  # Reduced field range for a quick test
        loop=False,
        out_dir=f"hyst_{precond_type}",
        Js_ref=Js_si,
        max_iter=200,
        snapshot_every=0,
        verbose=True,
    )

    # Precompute M_nodal
    vol_Js = volume * np.array(Js_lookup[mat_id - 1])
    from dataclasses import replace

    geom_Js = replace(geom, volume=jnp.asarray(vol_Js))
    M_nodal = compute_node_volumes(geom_Js, chunk_elems=100_000)
    node_vols = compute_node_volumes(geom, chunk_elems=100_000)

    start_t = time.time()
    res = run_hysteresis_loop(
        points=knt,
        geom=geom,
        A_lookup=A_lookup_red,
        K1_lookup=K1_lookup,
        Js_lookup=Js_lookup,
        k_easy_lookup=k_easy_lookup,
        m0=m0,
        params=params,
        V_mag=float(V_mag),
        node_volumes=node_vols,
        M_nodal=M_nodal,
        grad_backend="stored_grad_phi",
        boundary_mask=boundary_mask,
        precond_type=precond_type,
        order=order,
        cg_tol=1e-9,
    )
    # Ensure all JAX operations are done
    jax.block_until_ready(res)
    end_t = time.time()

    duration = end_t - start_t
    print(f"\n{precond_type.upper()} Total Time: {duration:.3f} s")
    return duration


if __name__ == "__main__":
    t_amg = run_benchmark("amgcl", order=3, L_cube=20.0)
    print(f"\nSimulation finished. Total time: {t_amg:.3f} s")
