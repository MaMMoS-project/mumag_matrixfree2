"""compare_minimizers.py.

Benchmarking script to compare different energy minimizers:
- BB (Barzilai-Borwein)
- Cohen CG
- PCG (Preconditioned CG)
- P-Cohen CG (Preconditioned Cohen)
- L-BFGS

Usage: pixi run python benchmarking/compare_minimizers.py --mesh sphere_40nm.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import replace

import sys
sys.path.append('src')

from curvilinear_bb_minimizer import make_minimizer as make_bb_minimizer
from minimizers import make_minimizer as make_new_minimizer
from energy_kernels import make_energy_kernels
from poisson_solve import make_solve_U
from fem_utils import TetGeom, compute_node_volumes

def load_mesh(mesh_path):
    data = np.load(mesh_path)
    knt = np.asarray(data["knt"], dtype=np.float64)
    ijk = np.asarray(data["ijk"])
    conn = ijk[:, :4].astype(np.int64)
    mat_id = ijk[:, 4].astype(np.int32)
    return knt, conn, mat_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--krn", type=str, default=None)
    parser.add_argument("--max-iter", type=int, default=100)
    args = parser.parse_args()

    # 1. Setup Data
    knt, conn, mat_id = load_mesh(args.mesh)
    G = int(mat_id.max())
    
    # Default materials (Iron-like)
    Js_lookup = np.ones(G) * 1.0
    A_lookup = np.ones(G) * 1e-11
    K1_lookup = np.zeros(G)
    k_easy = np.zeros((G, 3))
    k_easy[:, 2] = 1.0
    
    # Scale A by mesh unit (nm)
    mesh_unit = 1e-9
    A_scale = (1.0 / mesh_unit) ** 2
    A_lookup *= A_scale
    
    # Normalization
    Js_ref = 1.0
    MU0 = 4e-7 * np.pi
    Kd_ref = Js_ref**2 / (2.0 * MU0)
    A_red = A_lookup / Kd_ref
    K1_red = K1_lookup / Kd_ref
    Js_red = Js_lookup / Js_ref

    # Geometry
    from loop import compute_volume_JinvT
    conn32, volume, JinvT = compute_volume_JinvT(knt, conn)
    geom = TetGeom(
        conn=jnp.asarray(conn32),
        volume=jnp.asarray(volume),
        mat_id=jnp.asarray(mat_id),
        JinvT=jnp.asarray(JinvT)
    )
    V_mag = float(np.sum(volume))

    # Preconditioning helpers
    node_vols = compute_node_volumes(geom, chunk_elems=5000)
    vol_Js = volume * Js_red[mat_id - 1]
    geom_Js = replace(geom, volume=jnp.asarray(vol_Js))
    M_nodal = compute_node_volumes(geom_Js, chunk_elems=5000)
    
    solve_U = make_solve_U(geom, jnp.asarray(Js_red), cg_tol=1e-8, grad_backend="stored_JinvT", chunk_elems=5000)

    # 2. Define Methods
    methods = ["bb", "cohen", "pcg", "pcohen", "pcohen_auto", "pcohen_auto_strict", "lbfgs"]
    
    params = {
        "max_iter": args.max_iter,
        "tau_f": 1e-8,
        "eps_a": 1e-10,
        "phi_tol": 1e-9,
        "L": 5,             # Restart for Cohen
        "pc_iters": 5,      # Inner CG iters for PCG
        "tau0": 1e-2,
        "ls_eta1": 0.1,
        "ls_eta2": 0.1,
        "ls_C": 2.0,
        "ls_c": 0.5,
        "grad_backend": "stored_JinvT",
        "memory": 5,        # L-BFGS / AA memory
        "tn_iters": 5,      # Inner Newton-CG iters
        "gamma": 5,         # Initial line search steps
        "lr": 0.1,          # Nesterov lr
        "mu": 0.9           # Nesterov momentum
    }

    # Applied field: tilted to ensure non-trivial path
    h = np.array([0.1, 0.1, 1.0])
    h /= np.linalg.norm(h)
    B_ext = 0.05 * h # 50 mT

    # Initial state: random perturbation from z-axis
    m0 = np.zeros((knt.shape[0], 3))
    m0[:, 2] = 1.0
    rng = np.random.default_rng(42)
    m0 += 0.1 * rng.standard_normal(m0.shape)
    m0 /= np.linalg.norm(m0, axis=1, keepdims=True)

    # Filter out non-numeric params for JAX kernels
    kernel_params = {k:v for k,v in params.items() if k != "grad_backend"}

    print(f"Comparing minimizers on {args.mesh} ({knt.shape[0]} nodes, {conn.shape[0]} elements)")
    print("-" * 60)
    print(f"{'Method':<10} | {'Iters':<6} | {'Time (s)':<10} | {'Final Energy':<15}")
    print("-" * 60)

    for method in methods:
        if method == "bb":
            # Original BB minimizer
            minimize_bb = make_bb_minimizer(
                geom, A_red, K1_red, Js_red, k_easy, V_mag, node_vols, M_nodal, solve_U, cg_tol=1e-8,
                grad_backend="stored_JinvT", chunk_elems=5000
            )
            # BB params
            bb_params = {k:v for k,v in kernel_params.items() if k in ["max_iter", "tau_f", "eps_a", "tau0", "ls_eta1", "ls_eta2", "ls_C", "ls_c"]}
            
            # Warmup
            minimize_bb(m0, B_ext, verbose=False, **{**bb_params, "max_iter": 1})
            # Benchmark
            m_final, U_final, info = minimize_bb(m0, B_ext, verbose=False, **bb_params)
            iters = info["iters"]
            time_val = info["total_time"]
            energy = info["E"]
        else:
            # New minimizers
            method_to_use = method
            local_params = {**kernel_params}
            if method == "pcohen_auto":
                method_to_use = "pcohen"
                local_params["pc_auto"] = True
            if method == "pcohen_auto_strict":
                method_to_use = "pcohen"
                local_params["pc_auto"] = True
                local_params["pc_force_eta"] = 0.1
                local_params["pc_force_alpha"] = 1.0
            
            minimize_new = make_new_minimizer(
                geom, A_red, K1_red, Js_red, k_easy, V_mag, node_vols, M_nodal, solve_U, cg_tol=1e-8,
                method=method_to_use, grad_backend="stored_JinvT", chunk_elems=5000
            )
            # Warmup
            minimize_new(m0, B_ext, **{**local_params, "max_iter": 1})
            # Benchmark
            m_final, U_final, info = minimize_new(m0, B_ext, **local_params)
            iters = info["iters"]
            time_val = info["time"]
            energy = info["E"]
            
        print(f"{method:<10} | {int(iters):<6} | {time_val:<10.4f} | {energy:<15.8e}")

    print("-" * 60)

if __name__ == "__main__":
    main()
