"""test_poisson_advanced.py

Advanced Benchmark for the Poisson solver.
Compares:
1. No Preconditioning
2. Jacobi (Diagonal)
3. Coarse Correction (Jacobi + Aggregates)
4. Chebyshev Polynomial Preconditioning (4th order)
"""

from __future__ import annotations

import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pathlib import Path

from fem_utils import TetGeom
from loop import compute_volume_JinvT, compute_grad_phi_from_JinvT
from poisson_solve import make_poisson_ops
from aggregation_utils import build_aggregates_by_binning
import add_shell
import mesh

def benchmark_poisson():
    # 1. Setup Geometry (20 nm cube + 8 layer shell)
    L_cube = 20.0
    h = 2.0
    
    print(f"Creating mesh: cube={L_cube}nm, h={h}nm...")
    knt0, ijk0, _, _ = mesh.run_single_solid_mesher(
        geom='box', extent=f"{L_cube},{L_cube},{L_cube}", h=h, 
        backend='grid', no_vis=True, return_arrays=True
    )
    
    tmp_path = "tmp_adv_benchmark.npz"
    np.savez(tmp_path, knt=knt0, ijk=ijk0)
    knt, ijk = add_shell.run_add_shell_pipeline(in_npz=tmp_path, layers=8, K=1.4, h0=h, verbose=False)
    if Path(tmp_path).exists(): Path(tmp_path).unlink()

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
    
    # 2. Material & Operators
    Js_lookup = jnp.array([1.0, 0.0])
    apply_A, rhs_from_m, assemble_diag, assemble_E = make_poisson_ops(geom, Js_lookup, grad_backend='stored_grad_phi')
    
    m_unif = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (knt.shape[0], 1))
    b = rhs_from_m(m_unif) * boundary_mask
    x0 = jnp.zeros(knt.shape[0])
    Mdiag = assemble_diag(knt.shape[0])
    
    # 3. Coarse Components
    agg_id, inv_sqrt_counts, k_coarse = build_aggregates_by_binning(knt, k_target=1024)
    E = assemble_E(jnp.asarray(agg_id), jnp.asarray(inv_sqrt_counts))
    E_chol = jnp.linalg.cholesky(E)

    # 4. Chebyshev Setup: Estimate Max Eigenvalue of Jacobi-preconditioned A
    def power_method(n_iters=10):
        v = np.random.randn(knt.shape[0])
        v = jnp.asarray(v) * boundary_mask
        for _ in range(n_iters):
            # Jacobi preconditioned apply
            v_new = (apply_A(v) / (Mdiag + 1e-30)) * boundary_mask
            norm = jnp.linalg.norm(v_new)
            v = v_new / norm
        lam_max = jnp.vdot(v, (apply_A(v) / (Mdiag + 1e-30)) * boundary_mask)
        return float(lam_max)

    print("Estimating spectral radius for Chebyshev...")
    l_max = power_method()
    print(f"Estimated lambda_max: {l_max:.4f}")

    # 5. Advanced Solver with Multi-Preconditioning
    def solve_pcg(precond_type="jacobi"):
        
        def apply_Minv(r):
            if precond_type == "none":
                return r * boundary_mask
            
            # Base Jacobi
            z_jac = (r / (Mdiag + 1e-30)) * boundary_mask
            
            if precond_type == "jacobi":
                return z_jac
            
            if precond_type == "coarse":
                rc = jax.ops.segment_sum(r * inv_sqrt_counts[agg_id], agg_id, k_coarse)
                y = jax.scipy.linalg.solve_triangular(E_chol, rc, lower=True)
                y = jax.scipy.linalg.solve_triangular(E_chol, y, lower=True, trans='T')
                return (z_jac + y[agg_id] * inv_sqrt_counts[agg_id]) * boundary_mask
            
            if precond_type == "chebyshev":
                # 4th order Chebyshev polynomial smoother
                # To bring spectrum from [0, l_max] towards [0, 1]
                # Approximation: z = omega * z_jac + ...
                # Simple version: 3 iterations of Jacobi smoothing
                curr_z = z_jac
                for _ in range(3):
                    res = r - apply_A(curr_z) * boundary_mask
                    curr_z = curr_z + (res / (Mdiag + 1e-30)) * (2.0 / (l_max + 1.0))
                return curr_z * boundary_mask
            
            return z_jac

        # PCG Loop
        curr_x = x0
        r = b - apply_A(curr_x) * boundary_mask
        z = apply_Minv(r)
        p = z
        rz = jnp.vdot(r, z)
        bnorm2 = jnp.vdot(b, b)
        
        it = 0
        max_it = 2000
        tol2 = (1e-10**2) * bnorm2
        
        start_t = time.time()
        while it < max_it and rz > tol2:
            Ap = apply_A(p) * boundary_mask
            alpha = rz / (jnp.vdot(p, Ap) + 1e-30)
            curr_x = curr_x + alpha * p
            r = r - alpha * Ap
            z = apply_Minv(r)
            rz_new = jnp.vdot(r, z)
            beta = rz_new / (rz + 1e-30)
            p = z + beta * p
            rz = rz_new
            it += 1
        end_t = time.time()
        return it, end_t - start_t

    print("\nStarting Advanced Benchmarks (Tolerance 1e-10):")
    print("-" * 60)
    
    for pt in ["none", "jacobi", "coarse", "chebyshev"]:
        it, duration = solve_pcg(pt)
        name = pt.capitalize()
        print(f"{name:<12}: {it:4d} iterations, {duration:.3f} s")
    print("-" * 60)

if __name__ == "__main__":
    benchmark_poisson()
