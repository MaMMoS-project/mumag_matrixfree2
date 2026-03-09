"""run_profiler.py
Captures a JAX execution trace for performance analysis on a larger mesh.
Compares 'scatter' vs 'segment_sum' for energy gradients.
"""
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pathlib import Path
import jax.profiler

from fem_utils import TetGeom, compute_node_volumes
from loop import compute_volume_JinvT, compute_grad_phi_from_JinvT, load_materials_krn
from hysteresis_loop import LoopParams, run_hysteresis_loop
import add_shell
import mesh

def run_profile():
    # Setup Geometry: 60nm cube, h=2.0
    # Expected nodes: ~30,000 for cube, much more with shell
    L_cube, h = 60.0, 2.0
    print(f"Creating mesh: {L_cube}nm cube, h={h}nm...")
    knt0, ijk0, _, _ = mesh.run_single_solid_mesher(
        geom='box', extent=f"{L_cube},{L_cube},{L_cube}", h=h, 
        backend='grid', no_vis=True, return_arrays=True
    )
    
    tmp_path = "tmp_prof_large.npz"
    np.savez(tmp_path, knt=knt0, ijk=ijk0)
    # Use fewer layers for shell to keep it manageable but still "large"
    knt, ijk = add_shell.run_add_shell_pipeline(in_npz=tmp_path, layers=2, K=1.4, h0=h, verbose=False)
    if Path(tmp_path).exists(): Path(tmp_path).unlink()

    tets, mat_id = ijk[:, :4].astype(np.int64), ijk[:, 4].astype(np.int32)
    print(f"Mesh created: {knt.shape[0]} nodes, {tets.shape[0]} elements.")
    
    conn32, volume, JinvT = compute_volume_JinvT(knt, tets)
    grad_phi = compute_grad_phi_from_JinvT(JinvT)
    boundary_mask = jnp.asarray(add_shell.find_outer_boundary_mask(tets, knt.shape[0]), dtype=jnp.float64)

    geom = TetGeom(
        conn=jnp.asarray(conn32, dtype=jnp.int32),
        volume=jnp.asarray(volume, dtype=jnp.float64),
        mat_id=jnp.asarray(mat_id, dtype=jnp.int32),
        grad_phi=jnp.asarray(grad_phi, dtype=jnp.float64),
    )
    
    # Materials (NdFeB fallback if krn missing)
    krn_path = "cube_20nm.krn"
    if Path(krn_path).exists():
        A_lookup, K1_lookup, Js_lookup, k_easy_lookup = load_materials_krn(krn_path, int(mat_id.max()))
    else:
        G = int(mat_id.max())
        A_lookup = np.ones(G) * 7.7e-12 * 1e18
        K1_lookup = np.ones(G) * 4.3e6
        Js_lookup = np.ones(G) * 1.6
        k_easy_lookup = np.zeros((G, 3))
        k_easy_lookup[:, 2] = 1.0

    Js_ref = np.max(Js_lookup)
    MU0_SI = 4e-7 * np.pi
    Kd_ref = (Js_ref**2) / (2.0 * MU0_SI)
    
    is_mag = np.isin(mat_id, np.where(Js_lookup > 0)[0] + 1)
    V_mag = np.sum(volume[is_mag])

    m0_vec = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    m0 = np.tile(m0_vec, (knt.shape[0], 1))
    
    # One field step - updated to 3 steps: 2 warmups, 1 profiled
    params = LoopParams(
        h_dir=np.array([1.0, 0.0, 0.0]),
        B_start=1.0 / Js_ref, B_end=1.2 / Js_ref, dB=0.1,
        loop=False, out_dir='prof_large_out', snapshot_every=0, verbose=True
    )
    
    node_vols = compute_node_volumes(geom, chunk_elems=100000)

    results = {}

    for mode in ['scatter', 'segment_sum']:
        print(f"\n--- Benchmarking Mode: {mode} ---")
        
        # We'll use a modified run_hysteresis_loop or manual steps to control profiling
        # For simplicity, let's modify run_hysteresis_loop locally or just run it once to warm up JIT
        print("Warming up JIT...")
        run_hysteresis_loop(points=knt, geom=geom, A_lookup=A_lookup/Kd_ref, K1_lookup=K1_lookup/Kd_ref, 
                            Js_lookup=Js_lookup/Js_ref, k_easy_lookup=k_easy_lookup, m0=m0, 
                            params=params, V_mag=float(V_mag), node_volumes=node_vols, 
                            boundary_mask=boundary_mask, energy_assembly=mode, precond_type='jacobi')
        
        # Profiled run (repeat exactly the same to ensure cache hits)
        log_dir = f"tensorboard_trace_{mode}"
        print(f"Starting trace for {mode} in {log_dir}...")
        
        # To TRULY avoid compilation in the trace, we want the trace to start AFTER 
        # the first call to run_hysteresis_loop finishes.
        with jax.profiler.trace(log_dir):
            start_t = time.time()
            res = run_hysteresis_loop(points=knt, geom=geom, A_lookup=A_lookup/Kd_ref, K1_lookup=K1_lookup/Kd_ref, 
                                Js_lookup=Js_lookup/Js_ref, k_easy_lookup=k_easy_lookup, m0=m0, 
                                params=params, V_mag=float(V_mag), node_volumes=node_vols, 
                                boundary_mask=boundary_mask, energy_assembly=mode, precond_type='jacobi')
            jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, res) 
            end_t = time.time()
        
        results[mode] = end_t - start_t
        print(f"Mode {mode} finished in {results[mode]:.3f} s.")
        
        results[mode] = end_t - start_t
        print(f"Mode {mode} finished in {results[mode]:.3f} s.")

    print("\nSummary (Large Mesh):")
    for mode, duration in results.items():
        print(f"{mode:<12}: {duration:.3f} s")

if __name__ == "__main__":
    run_profile()
