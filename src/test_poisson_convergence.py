"""test_poisson_convergence.py

Benchmark script for the Poisson solver.
Compares CG iterations for:
1. No Preconditioning (Identity)
2. Jacobi (Diagonal)
3. Chebyshev (Combined Jacobi + 4th-order polynomial)
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
from poisson_solve import make_poisson_ops, estimate_spectral_radius
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
    apply_A, rhs_from_m, assemble_diag = make_poisson_ops(geom, Js_lookup, grad_backend='stored_grad_phi')
    
    m_unif = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (knt.shape[0], 1))
    b = rhs_from_m(m_unif) * boundary_mask
    x0 = jnp.zeros(knt.shape[0])
    Mdiag = assemble_diag(knt.shape[0])
    
    # 3. Spectral Estimation
    print("Estimating spectral radius for Chebyshev...")
    l_max = 1.1 * estimate_spectral_radius(apply_A, Mdiag, boundary_mask, knt.shape[0])
    print(f"Estimated lambda_max (with buffer): {l_max:.4f}")

    # 4. PCG Benchmark Loop
    def solve_reporting(precond_type="jacobi"):
        order = 3
        apply_Minv_amg = None
        
        if precond_type == "amg":
            print("Setting up AMG hierarchy on CPU (PyAMG)...")
            from amg_utils import assemble_poisson_matrix_cpu, setup_amg_hierarchy, csr_to_jax_bCOO, make_jax_amg_vcycle
            A_cpu = assemble_poisson_matrix_cpu(
                np.array(geom.conn), 
                np.array(geom.volume), 
                np.array(geom.grad_phi), 
                boundary_mask=np.array(boundary_mask),
                reg=1e-12
            )
            hierarchy_cpu = setup_amg_hierarchy(A_cpu)
            hierarchy_jax = []
            for i, level in enumerate(hierarchy_cpu):
                level_dict = {
                    'P': csr_to_jax_bCOO(level['P']),
                    'R': csr_to_jax_bCOO(level['R']),
                    'A_sparse': csr_to_jax_bCOO(level['A']),
                    'Mdiag': jnp.asarray(level['A'].diagonal())
                }
                if i == len(hierarchy_cpu) - 1:
                    level_dict['A_dense'] = jnp.asarray(level['A'].todense())
                hierarchy_jax.append(level_dict)
            
            # Simple wrapper for matrix-free apply_A that handles boundary_mask
            def apply_A_masked(v):
                return apply_A(v) * boundary_mask
                
            vcycle = make_jax_amg_vcycle(apply_A_masked, Mdiag, hierarchy_jax)
            apply_Minv_amg = vcycle

        def apply_Minv(r):
            if precond_type == "none":
                return r * boundary_mask
            
            if precond_type == "amg":
                return apply_Minv_amg(r)
            
            # Base Jacobi
            z0 = (r / (Mdiag + 1e-30)) * boundary_mask
            
            if precond_type == "chebyshev":
                lam_max = l_max
                lam_min = lam_max / 10.0
                d = (lam_max + lam_min) / 2.0
                c = (lam_max - lam_min) / 2.0
                
                # k=0
                alpha = 1.0 / d
                y = alpha * z0
                y_prev = jnp.zeros_like(y)
                
                curr_alpha = alpha
                for _ in range(1, order):
                    res = r - apply_A(y) * boundary_mask
                    z_j = (res / (Mdiag + 1e-30)) * boundary_mask
                    beta = (c * curr_alpha / 2.0)**2
                    curr_alpha = 1.0 / (d - beta)
                    y_next = y + curr_alpha * z_j + curr_alpha * beta * (y - y_prev)
                    y_prev = y
                    y = y_next
                z = y
            else:
                z = z0
            
            return z * boundary_mask

        # PCG Implementation
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

    print("\nStarting Poisson Benchmarks (Tolerance 1e-10):")
    print("-" * 60)
    
    for pt in ["none", "jacobi", "chebyshev", "amg"]:
        it, duration = solve_reporting(pt)
        name = pt.capitalize()
        print(f"{name:<12}: {it:4d} iterations, {duration:.3f} s")
    print("-" * 60)

if __name__ == "__main__":
    benchmark_poisson()
