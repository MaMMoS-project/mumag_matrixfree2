"""profile_compilation.py

Profiling script to check for redundant JAX compilations.
"""

from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pathlib import Path

from fem_utils import TetGeom, compute_node_volumes
from loop import compute_volume_JinvT, compute_grad_phi_from_JinvT, load_materials_krn
from hysteresis_loop import LoopParams, run_hysteresis_loop
import add_shell
import mesh

# Monkey-patch jax.jit to track compilations
original_jit = jax.jit
compilation_counts = {}

def tracked_jit(fun=None, **kwargs):
    if fun is None:
        return lambda f: tracked_jit(f, **kwargs)
    
    name = getattr(fun, "__name__", str(fun))
    # Filter out internal/anonymous functions if needed, but here we want to see them
    
    def wrapping_fun(*args, **kwargs):
        if name not in compilation_counts:
            print(f"[COMPILATION] Compiling function: {name}")
            compilation_counts[name] = 1
        else:
            # This won't actually catch re-compilation of the SAME jit object 
            # because the wrapped function is only called once per compilation.
            # Wait, that's not right. The wrapped function is called on EVERY execution
            # IF it's not the JITed version.
            pass
        return fun(*args, **kwargs)
    
    # Actually, the best way to track JIT compilation is to put a print inside the function
    # that is being JITed.
    return original_jit(wrapping_fun, **kwargs)

# Instead of global monkey-patching which is tricky with JAX's decorators,
# let's just use the fact that code inside a JITed function (but not in lax loops)
# runs exactly once per compilation.

def test_compilation():
    # 1. Setup Geometry (small mesh for speed)
    L_cube = 20.0
    h = 4.0 # Coarser mesh
    
    print(f"Creating mesh...")
    knt0, ijk0, _, _ = mesh.run_single_solid_mesher(
        geom='box', extent=f"{L_cube},{L_cube},{L_cube}", h=h, 
        backend='grid', no_vis=True, return_arrays=True
    )
    
    tmp_path = "tmp_prof_mesh.npz"
    np.savez(tmp_path, knt=knt0, ijk=ijk0)
    knt, ijk = add_shell.run_add_shell_pipeline(in_npz=tmp_path, layers=2, K=1.4, h0=h, verbose=False)
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
    
    # 2. Materials
    Js = 1.6
    K1 = 4.3e6
    A_si = 7.7e-12
    k_easy = np.array([0.0, 0.0, 1.0])
    
    MU0_SI = 4e-7 * np.pi
    Kd_ref = (Js**2) / (2.0 * MU0_SI)
    
    A_red = (A_si * 1e18) / Kd_ref
    K1_red = K1 / Kd_ref
    Js_red = 1.0
    
    A_lookup = np.array([A_red, 0.0])
    K1_lookup = np.array([K1_red, 0.0])
    Js_lookup = np.array([Js_red, 0.0])
    k_easy_lookup = np.array([k_easy, k_easy])
    
    is_mag = (mat_id == 1)
    V_mag = np.sum(volume[is_mag])

    # 3. Instrument the code by re-importing and wrapping
    import energy_kernels
    import poisson_solve
    import curvilinear_bb_minimizer

    def wrap_with_print(name, original_make):
        def wrapper(*args, **kwargs):
            print(f"[TRACE] Calling {name}")
            return original_make(*args, **kwargs)
        return wrapper

    # We need to inject prints into the functions that get JITed.
    # Since they are JITed inside the make_... functions, we can't easily wrap them
    # without modifying the source files. 
    # Let's modify the source files temporarily or just use jax.profiler.

    # Actually, JAX 0.4.x has a better way:
    # jax.config.update("jax_log_compiles", True)
    
    jax.config.update("jax_log_compiles", True)

    m0_vec = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    m0 = np.tile(m0_vec, (knt.shape[0], 1))
    
    params = LoopParams(
        h_dir=np.array([1.0, 0.0, 0.0]),
        B_start=0.0,
        B_end=1.0, # Just 2 steps
        dB=1.0,
        loop=False,
        out_dir='prof_out',
        max_iter=5, # Keep it short
        verbose=False
    )
    
    node_vols = compute_node_volumes(geom, chunk_elems=100000)
    
    print("\n--- Starting Loop (Compilations should be logged) ---")
    run_hysteresis_loop(
        points=knt,
        geom=geom,
        A_lookup=A_lookup,
        K1_lookup=K1_lookup,
        Js_lookup=Js_lookup,
        k_easy_lookup=k_easy_lookup,
        m0=m0,
        params=params,
        V_mag=float(V_mag),
        node_volumes=node_vols,
        grad_backend='stored_grad_phi',
        boundary_mask=boundary_mask
    )
    print("--- Loop Finished ---\n")

if __name__ == "__main__":
    test_compilation()
