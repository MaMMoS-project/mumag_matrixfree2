"""profile_energy.py

Performance profiling for micromagnetic energy and gradient kernels.
Compares a full iteration (including Poisson solve) vs. energy/gradient kernels alone.
"""

from __future__ import annotations

import sys
from pathlib import Path
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from dataclasses import replace

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))
from fem_utils import TetGeom, compute_node_volumes
from loop import compute_volume_JinvT, compute_grad_phi_from_JinvT
from energy_kernels import make_energy_kernels
from poisson_solve import make_solve_U
import add_shell

def profile_energy():
    """Benchmark performance of energy kernels and the Poisson solver.

    Loads a 60nm cube mesh with an airbox, then measures and compares:
    1. Average time for a full iteration (Solve U + compute kernels).
    2. Average time for energy/gradient kernels alone (reusing U).
    3. Poisson solver overhead.
    """
    # 1. Load existing mesh
    mesh_path = "cube_60nm_shell.npz"
    print(f"Loading mesh from {mesh_path}...")
    data = np.load(mesh_path)
    knt, ijk = data['knt'], data['ijk']

    tets = ijk[:, :4].astype(np.int64); mat_id = ijk[:, 4].astype(np.int32)
    conn32, volume, JinvT = compute_volume_JinvT(knt, tets)
    grad_phi = compute_grad_phi_from_JinvT(JinvT)
    boundary_mask = jnp.asarray(add_shell.find_outer_boundary_mask(tets, knt.shape[0]), dtype=jnp.float64)

    geom = TetGeom(
        conn=jnp.asarray(conn32, dtype=jnp.int32),
        volume=jnp.asarray(volume, dtype=jnp.float64),
        mat_id=jnp.asarray(mat_id, dtype=jnp.int32),
        grad_phi=jnp.asarray(grad_phi, dtype=jnp.float64),
    )
    
    # 2. Material Properties
    Js = 1.0; A_red = 1.0; K1_red = 0.1
    A_lookup = jnp.array([A_red, 0.0]); K1_lookup = jnp.array([K1_red, 0.0])
    Js_lookup = jnp.array([Js, 0.0])
    k_easy = jnp.array([0.0, 0.0, 1.0]); k_easy_lookup = jnp.array([k_easy, k_easy])
    
    vol_Js = volume * np.array(Js_lookup[mat_id - 1])
    M_nodal = compute_node_volumes(replace(geom, volume=jnp.asarray(vol_Js)), chunk_elems=200_000)
    V_mag_nm = np.sum(volume[mat_id == 1])

    # 3. Kernels
    # Set tolerance to 1e-10 to match benchmark/cpp
    chunk_elems = 100_000
    solve_U = make_solve_U(geom, Js_lookup, cg_tol=1e-10, boundary_mask=boundary_mask, precond_type='amgcl', chunk_elems=chunk_elems)
    energy_and_grad, _, _ = make_energy_kernels(geom, A_lookup, K1_lookup, Js_lookup, k_easy_lookup, V_mag_nm, M_nodal, chunk_elems=chunk_elems)
    
    # Test state: Random magnetization
    key = jax.random.PRNGKey(42)
    m = jax.random.normal(key, (knt.shape[0], 3))
    m = m / jnp.linalg.norm(m, axis=1, keepdims=True)
    b_ext = jnp.array([0.01, 0.0, 0.0])

    print(f"Mesh Size: {knt.shape[0]} nodes, {tets.shape[0]} elements")
    print("Compiling kernels (warm-up)...")
    
    # Warm-up
    u_warm = solve_U(m, jnp.zeros(knt.shape[0]))
    e_warm, g_warm = energy_and_grad(m, u_warm, b_ext)
    jax.block_until_ready((u_warm, e_warm, g_warm))

    # 4. Profiling Loop 1: Full Iteration (Solve U + Kernels)
    n_repeats = 5
    print(f"\nLoop 1: Recomputing potential U every time ({n_repeats} iterations)...")
    
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        u = solve_U(m, jnp.zeros(knt.shape[0]))
        e, g = energy_and_grad(m, u, b_ext)
        jax.block_until_ready((e, g))
    t1 = time.perf_counter()
    
    total_full = t1 - t0
    avg_full = total_full / n_repeats

    # 5. Profiling Loop 2: Kernels Only (Reuse U)
    print(f"Loop 2: Reusing precomputed potential U ({n_repeats} iterations)...")
    u_fixed = solve_U(m, jnp.zeros(knt.shape[0]))
    jax.block_until_ready(u_fixed)
    
    t2 = time.perf_counter()
    for _ in range(n_repeats):
        e, g = energy_and_grad(m, u_fixed, b_ext)
        jax.block_until_ready((e, g))
    t3 = time.perf_counter()
    
    total_kernels = t3 - t2
    avg_kernels = total_kernels / n_repeats

    # 6. Report
    print("\n" + "="*40)
    print(f"{'Metric':<25} | {'Time (ms)':>10}")
    print("-" * 40)
    print(f"{'Full Iteration (Avg)':<25} | {avg_full*1000:>10.2f}")
    print(f"{'Kernels Only (Avg)':<25} | {avg_kernels*1000:>10.2f}")
    print(f"{'Poisson Solve Overhead':<25} | {(avg_full - avg_kernels)*1000:>10.2f}")
    print("="*40)

if __name__ == "__main__":
    profile_energy()
