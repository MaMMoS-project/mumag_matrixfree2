"""test_hysteresis_x_axis.py

Test for magnetization curve along the hard axis (X).
Easy axis is Z. Field is applied along X.
Compares AMG vs Jacobi performance on a 60nm cube.
"""

from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pathlib import Path
import time

from fem_utils import TetGeom, compute_node_volumes
from loop import compute_volume_JinvT, compute_grad_phi_from_JinvT, load_materials_krn
from hysteresis_loop import LoopParams, run_hysteresis_loop
import add_shell
import mesh

def run_benchmark(precond_type, order, L_cube=60.0, h=2.0, layers=8):
    print(f"\n=== Benchmarking {precond_type.upper()} (order={order}) ===")
    
    print(f"Creating mesh: {L_cube}nm cube, h={h}nm, layers={layers}...")
    knt0, ijk0, _, _ = mesh.run_single_solid_mesher(
        geom='box', extent=f"{L_cube},{L_cube},{L_cube}", h=h, 
        backend='grid', no_vis=True, return_arrays=True
    )
    
    tmp_path = f"tmp_mesh_{precond_type}.npz"
    np.savez(tmp_path, knt=knt0, ijk=ijk0)
    knt, ijk = add_shell.run_add_shell_pipeline(in_npz=tmp_path, layers=layers, K=1.4, h0=h, verbose=False)
    if Path(tmp_path).exists(): Path(tmp_path).unlink()

    tets = ijk[:, :4].astype(np.int64)
    mat_id = ijk[:, 4].astype(np.int32)
    conn32, volume, JinvT = compute_volume_JinvT(knt, tets)
    grad_phi = compute_grad_phi_from_JinvT(JinvT)
    boundary_mask = jnp.asarray(add_shell.find_outer_boundary_mask(tets, knt.shape[0]), dtype=jnp.float64)

    geom = TetGeom(
        conn=jnp.asarray(conn32, dtype=jnp.int32),
        volume=jnp.asarray(volume, dtype=jnp.float64),
        mat_id=jnp.asarray(mat_id, dtype=jnp.int32),
        grad_phi=jnp.asarray(grad_phi, dtype=jnp.float64),
    )
    
    # Material Properties (NdFeB fallback)
    Js_lookup = np.array([1.6, 0.0])
    K1_lookup = np.array([4.3e6, 0.0])
    A_si = 7.7e-12
    k_easy_lookup = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    Js_ref = np.max(Js_lookup)
    MU0_SI = 4e-7 * np.pi
    Kd_ref = (Js_ref**2) / (2.0 * MU0_SI)
    
    A_red = (A_si * 1e18) / Kd_ref
    K1_red = K1_lookup / Kd_ref
    Js_red = Js_lookup / Js_ref
    A_lookup_red = np.array([A_red, 0.0])
    
    is_mag = np.isin(mat_id, np.where(Js_lookup > 0)[0] + 1)
    V_mag = np.sum(volume[is_mag])

    m0_vec = np.array([0.0, 0.0, 1.0])
    m0 = np.tile(m0_vec, (knt.shape[0], 1))
    
    params = LoopParams(
        h_dir=np.array([1.0, 0.0, 0.0]),
        B_start=0.0, B_end=8.0, dB=2.0, # Fewer steps for faster benchmark
        loop=False,
        out_dir=f'bench_{precond_type}',
        Js_ref=Js_ref,
        max_iter=100,
        snapshot_every=0,
        verbose=True
    )
    
    node_vols = compute_node_volumes(geom, chunk_elems=100000)
    
    start_t = time.time()
    res = run_hysteresis_loop(
        points=knt,
        geom=geom,
        A_lookup=A_lookup_red,
        K1_lookup=K1_red,
        Js_lookup=Js_red,
        k_easy_lookup=k_easy_lookup,
        m0=m0,
        params=params,
        V_mag=float(V_mag),
        node_volumes=node_vols,
        grad_backend='stored_grad_phi',
        boundary_mask=boundary_mask,
        precond_type=precond_type,
        order=order,
        cg_tol=1e-9
    )
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, res)
    end_t = time.time()
    
    duration = end_t - start_t
    print(f"\n{precond_type.upper()} Total Time: {duration:.3f} s")
    return duration

if __name__ == "__main__":
    t_amg = run_benchmark('amg', order=3)
    t_jac = run_benchmark('jacobi', order=0)
    
    print("\nSummary (60nm Cube, 8 Layers, 1e-10 tol):")
    print(f"AMG    : {t_amg:.3f} s")
    print(f"Jacobi : {t_jac:.3f} s")
    print(f"Speedup: {t_jac / t_amg:.2f}x")
