"""test_poisson_convergence.py

Benchmark script for the Poisson solver.
Compares CG iterations for:
1. No Preconditioning (Identity)
2. Jacobi (Diagonal)
3. Chebyshev (Combined Jacobi + 4th-order polynomial)
"""

from __future__ import annotations

import sys
from pathlib import Path
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))
from fem_utils import TetGeom
from loop import compute_volume_JinvT, compute_grad_phi_from_JinvT
from poisson_solve import make_poisson_ops, estimate_spectral_radius, make_solve_U
import add_shell
import mesh

def benchmark_poisson():
    # 1. Setup Geometry (60 nm cube + 8 layer shell)
    L_cube = 60.0
    h = 2.0

    print(f"Creating mesh: cube={L_cube}nm, h={h}nm...")
    knt0, ijk0, _, _ = mesh.run_single_solid_mesher(
        geom='box', extent=f"{L_cube},{L_cube},{L_cube}", h=h, 
        backend='grid', no_vis=True, return_arrays=True
    )

    tmp_path = "tmp_benchmark.npz"
    np.savez(tmp_path, knt=knt0, ijk=ijk0)
    knt, ijk = add_shell.run_add_shell_pipeline(in_npz=tmp_path, layers=8, K=1.4, h0=h, verbose=False)
    if Path(tmp_path).exists(): Path(tmp_path).unlink()

    # Save final mesh for C++ comparison
    final_mesh_path = "cube_60nm_shell.npz"
    np.savez(final_mesh_path, knt=knt, ijk=ijk)
    print(f"Final mesh saved to {final_mesh_path}. Total elements: {ijk.shape[0]}")



    tets = ijk[:, :4].astype(np.int64)
    conn32, volume, JinvT = compute_volume_JinvT(knt, tets)
    grad_phi = compute_grad_phi_from_JinvT(JinvT)
    boundary_mask = jnp.asarray(add_shell.find_outer_boundary_mask(tets, knt.shape[0]), dtype=jnp.float64)

    geom = TetGeom(
        conn=jnp.asarray(conn32, dtype=jnp.int32),
        volume=jnp.asarray(volume, dtype=jnp.float64),
        mat_id=jnp.asarray(ijk[:, 4], dtype=jnp.int32),
        grad_phi=jnp.asarray(grad_phi, dtype=jnp.float64),
    )

    # 2. Material
    Js_lookup = jnp.array([1.0, 0.0])
    m_unif = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (knt.shape[0], 1))
    x0 = jnp.zeros(knt.shape[0])

    # 3. Benchmark Loop
    def solve_reporting(precond_type="jacobi"):
        solve_U = make_solve_U(
            geom, Js_lookup, 
            precond_type=precond_type, 
            cg_tol=1e-10, 
            cg_maxiter=2000, 
            boundary_mask=boundary_mask
        )

        def run_solve():
            U, it, rel_res = solve_U(m_unif, x0, return_info=True)
            U.block_until_ready()
            return int(it), float(rel_res)

        # Warm-up
        run_solve()

        # Timed run
        start_t = time.time()
        it, rel_res = run_solve()
        end_t = time.time()

        return it, end_t - start_t, rel_res

    print("\nStarting Poisson Benchmarks (Tolerance 1e-10):")
    print("-" * 75)

    # Test all preconditioning types supported by make_solve_U
    for pt in ["none", "jacobi", "chebyshev", "amg", "amgcl"]:
        try:
            it, duration, rel_res = solve_reporting(pt)
            name = pt.capitalize()
            print(f"{name:<12}: {it:4d} iterations, {duration:.3f} s, rel_res: {rel_res:.2e}")
        except Exception as e:
            print(f"{pt.capitalize():<12}: FAILED ({e})")

    print("-" * 75)



if __name__ == "__main__":
    benchmark_poisson()
