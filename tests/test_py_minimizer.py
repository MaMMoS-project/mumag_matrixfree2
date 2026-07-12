import os
import sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from loop import compute_volume_JinvT, compute_grad_phi_from_JinvT, load_materials
from amg_utils import (
    assemble_divergence_matrices_cpu,
    assemble_poisson_matrix_cpu,
    assemble_exchange_anisotropy_matrix_cpu,
    make_sparse_operator,
)
from fem_utils import TetGeom, compute_node_volumes
from minimizers import make_minimizer
from hysteresis_loop import LoopParams

def test():
    # 1. Load mesh
    mesh_path = os.path.join(os.path.dirname(__file__), "../single_solid.npz")
    data = np.load(mesh_path)
    knt = np.asarray(data["knt"], dtype=np.float64)
    conn = np.asarray(data["ijk"][:, :4], dtype=np.int64)
    mat_id = np.asarray(data["ijk"][:, 4], dtype=np.int32) if data["ijk"].shape[1] > 4 else np.ones(conn.shape[0], dtype=np.int32)
    if mat_id.min() == 0:
        mat_id = mat_id + 1
        
    G = int(mat_id.max())
    A_lookup, K1_lookup, Js_lookup, k_easy_lookup = load_materials(None, G)
    
    conn32, volume, JinvT = compute_volume_JinvT(knt, conn)
    is_mag = np.isin(mat_id, np.where(Js_lookup > 0)[0] + 1)
    V_mag = np.sum(volume[is_mag])
    if V_mag == 0:
        V_mag = 1.0
        
    Js_ref = np.max(Js_lookup) if np.max(Js_lookup) > 0 else 1.0
    MU0_SI = 4e-7 * np.pi
    Kd_ref = (Js_ref**2) / (2.0 * MU0_SI)
    
    A_red = A_lookup / Kd_ref
    K1_red = K1_lookup / Kd_ref
    Js_red = Js_lookup / Js_ref
    
    geom = TetGeom(
        conn=jnp.asarray(conn32, dtype=jnp.int32),
        volume=jnp.asarray(volume, dtype=jnp.float64),
        mat_id=jnp.asarray(mat_id, dtype=jnp.int32),
        JinvT=jnp.asarray(JinvT, dtype=jnp.float64),
        grad_phi=None,
        x_nodes=None,
    )
    
    m0_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    m = np.tile(m0_vec[None, :], (knt.shape[0], 1))
    
    # Introduce some random perturbation to m
    rng = np.random.default_rng(42)
    m = m + 0.1 * rng.standard_normal(m.shape)
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = jnp.asarray(m)
    
    node_vols = compute_node_volumes(geom, chunk_elems=200_000)
    vol_Js = volume * Js_red[mat_id - 1]
    from dataclasses import replace
    geom_Js = replace(geom, volume=jnp.asarray(vol_Js))
    M_nodal = compute_node_volumes(geom_Js, chunk_elems=200_000)
    
    l_grad_phi = compute_grad_phi_from_JinvT(JinvT)
    
    # Assembled matrices
    A_scipy = assemble_poisson_matrix_cpu(
        conn32, volume, l_grad_phi, boundary_mask=np.zeros(knt.shape[0], dtype=np.int32), reg=1e-12
    )
    A_diag = jnp.asarray(A_scipy.diagonal())
    A_sparse = make_sparse_operator(A_scipy, cpu_spmv_backend="mkl_ffi")
    
    Dx_scipy, Dy_scipy, Dz_scipy = assemble_divergence_matrices_cpu(conn32, volume, l_grad_phi, Js_red, mat_id)
    import scipy.sparse as sp
    D_scipy = sp.hstack([Dx_scipy, Dy_scipy, Dz_scipy]).tocsr()
    D_sparse = make_sparse_operator(D_scipy, cpu_spmv_backend="mkl_ffi")
    
    N = knt.shape[0]
    Gx_scipy = 2.0 * D_scipy[:, :N].transpose()
    Gy_scipy = 2.0 * D_scipy[:, N:2*N].transpose()
    Gz_scipy = 2.0 * D_scipy[:, 2*N:].transpose()
    G_scipy = sp.vstack([Gx_scipy, Gy_scipy, Gz_scipy]).tocsr()
    G_sparse = make_sparse_operator(G_scipy, cpu_spmv_backend="mkl_ffi")
    
    K_eff_scipy = assemble_exchange_anisotropy_matrix_cpu(
        conn32, volume, l_grad_phi, A_red, K1_red, k_easy_lookup, mat_id
    )
    K_eff_sparse = make_sparse_operator(K_eff_scipy, cpu_spmv_backend="mkl_ffi")
    
    # Setup Solve_U
    from poisson_solve import make_solve_U
    solve_U = make_solve_U(
        geom,
        jnp.asarray(Js_red, dtype=jnp.float64),
        precond_type="amgcl",
        order=1,
        chunk_elems=200_000,
        cg_maxiter=2000,
        cg_tol=1e-8,
        poisson_reg=1e-12,
        grad_backend="stored_JinvT",
        boundary_mask=None,
        mode="assembled",
        A_sparse=A_sparse,
        cpu_spmv_backend="mkl_ffi",
        poisson_solver="pardiso",
    )
    
    # Setup JAX minimizer
    minimize = make_minimizer(
        geom,
        A_lookup=jnp.asarray(A_red, dtype=jnp.float64),
        K1_lookup=jnp.asarray(K1_red, dtype=jnp.float64),
        Js_lookup=jnp.asarray(Js_red, dtype=jnp.float64),
        k_easy_lookup=jnp.asarray(k_easy_lookup, dtype=jnp.float64),
        V_mag=V_mag,
        node_volumes=node_vols,
        M_nodal=M_nodal,
        solve_U=solve_U,
        cg_tol=1e-8,
        method="pcohen_hs",
        grad_backend="stored_JinvT",
        mode="assembled",
    )
    
    U = jnp.zeros(N, dtype=jnp.float64)
    B_ext = jnp.array([0.0, 0.0, -0.5], dtype=jnp.float64)
    
    print("Running JAX minimize...")
    params = LoopParams(
        h_dir="0,0,1", B_start=0.0, B_end=0.0, dB=0.0,
        max_iter=500, tau_f=1e-8, eps_a=1e-6,
        method="pcohen_hs",
    )
    
    print("Running JAX minimize...")
    m_new, U_new, info = minimize(
        m, B_ext, U0=U,
        gamma=1.0, max_iter=params.max_iter, tau_f=params.tau_f, eps_a=params.eps_a,
        tau0=params.tau0, tau_min=params.tau_min, tau_max=params.tau_max,
        ls_eta1=params.ls_eta1, ls_eta2=params.ls_eta2, ls_C=params.ls_C,
        ls_c=params.ls_c, ls_s0=params.ls_s0, ls_max_evals=params.ls_max_evals,
        verbose=params.verbose, pc_iters=params.pc_iters, pc_auto=params.pc_auto,
        pc_force_eta=params.pc_force_eta, pc_force_alpha=params.pc_force_alpha,
        pc_stagnation_nu=params.pc_stagnation_nu, memory=params.memory,
        tn_iters=params.tn_iters, lr=params.lr, mu=params.mu, pc_reg=params.pc_reg,
        phi_extrapolate=params.phi_extrapolate, L=params.L,
        sparse_ops={
            "A_sparse": A_sparse, "A_diag": A_diag, "K_eff_sparse": K_eff_sparse,
            "D_sparse": D_sparse, "G_sparse": G_sparse,
        }
    )
    print("Finished JAX successfully!")
    print("Info:", info)
    assert info is not None
    print("U norm:", np.linalg.norm(U_new))
    
if __name__ == "__main__":
    test()
