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
import add_shell
from minimizers import make_minimizer, tangent_grad

def test_compare():
    # 1. Load mesh
    mesh_path = os.path.join(os.path.dirname(__file__), "single_solid.npz")
    data = np.load(mesh_path)
    knt = np.asarray(data["knt"], dtype=np.float64)
    ijk = np.asarray(data["ijk"])
    
    conn = ijk[:, :4].astype(np.int64)
    mat_id = ijk[:, 4].astype(np.int32) if ijk.shape[1] > 4 else np.ones(conn.shape[0], dtype=np.int32)
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
    
    # Introduce some random perturbation to m to make it non-trivial
    rng = np.random.default_rng(42)
    m = m + 0.1 * rng.standard_normal(m.shape)
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = jnp.asarray(m)
    
    # Dummy outer boundary mask
    mask_np = np.zeros(knt.shape[0], dtype=np.int32)
    boundary_mask = jnp.asarray(mask_np, dtype=jnp.float64)
    
    node_vols = compute_node_volumes(geom, chunk_elems=200_000)
    vol_Js = volume * Js_red[mat_id - 1]
    from dataclasses import replace
    geom_Js = replace(geom, volume=jnp.asarray(vol_Js))
    M_nodal = compute_node_volumes(geom_Js, chunk_elems=200_000)
    
    l_grad_phi = compute_grad_phi_from_JinvT(JinvT)
    
    # Assembled matrices
    A_scipy = assemble_poisson_matrix_cpu(
        conn32, volume, l_grad_phi, boundary_mask=mask_np, reg=1e-12
    )
    A_diag = jnp.asarray(A_scipy.diagonal())
    A_sparse = make_sparse_operator(A_scipy, cpu_spmv_backend="persistent_mkl")
    
    Dx_scipy, Dy_scipy, Dz_scipy = assemble_divergence_matrices_cpu(conn32, volume, l_grad_phi, Js_red, mat_id)
    import scipy.sparse as sp
    D_scipy = sp.hstack([Dx_scipy, Dy_scipy, Dz_scipy]).tocsr()
    D_sparse = make_sparse_operator(D_scipy, cpu_spmv_backend="persistent_mkl")
    
    N = knt.shape[0]
    Gx_scipy = 2.0 * D_scipy[:, :N].transpose()
    Gy_scipy = 2.0 * D_scipy[:, N:2*N].transpose()
    Gz_scipy = 2.0 * D_scipy[:, 2*N:].transpose()
    G_scipy = sp.vstack([Gx_scipy, Gy_scipy, Gz_scipy]).tocsr()
    G_sparse = make_sparse_operator(G_scipy, cpu_spmv_backend="persistent_mkl")
    
    K_eff_scipy = assemble_exchange_anisotropy_matrix_cpu(
        conn32, volume, l_grad_phi, A_red, K1_red, k_easy_lookup, mat_id
    )
    K_eff_sparse = make_sparse_operator(K_eff_scipy, cpu_spmv_backend="persistent_mkl")
    
    # Preconditioning setup in Python
    from energy_kernels import compute_exchange_diagonal
    d_diag = compute_exchange_diagonal(geom, jnp.asarray(A_red), V_mag, chunk_elems=200_000, assembly="segment_sum", grad_backend="stored_JinvT")
    inv_M_prec = jnp.where(d_diag > 1e-20, 1.0 / d_diag, 1.0)[:, None]
    
    # 2. Setup Solve_U
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
        boundary_mask=boundary_mask,
        mode="assembled",
        A_sparse=A_sparse,
        cpu_spmv_backend="persistent_mkl",
        poisson_solver="pardiso",
    )
    
    # Initial U
    U, _, _ = solve_U(m, jnp.zeros(knt.shape[0]), 1e-8, return_info=True, sparse_ops={
        "A_sparse": A_sparse,
        "Dx_sparse": None, "Dy_sparse": None, "Dz_sparse": None, "A_diag": A_diag,
        "K_eff_sparse": K_eff_sparse, "Gx_sparse": None, "Gy_sparse": None, "Gz_sparse": None,
        "D_sparse": D_sparse, "G_sparse": G_sparse,
    })
    
    # 4. Compute Energy and Gradient
    from energy_kernels import make_energy_kernels
    py_energy_and_grad, _, _, local_grad_only = make_energy_kernels(
        geom, jnp.asarray(A_red), jnp.asarray(K1_red), jnp.asarray(Js_red), jnp.asarray(k_easy_lookup), V_mag, M_nodal,
        mode="assembled", chunk_elems=200_000, grad_backend="stored_JinvT"
    )
    
    B_ext = jnp.array([0.0, 0.0, -0.5], dtype=jnp.float64) # Applied field
    
    # Python Energy & Grad
    sparse_ops_py = {
        "A_sparse": A_sparse, "A_diag": A_diag, "K_eff_sparse": K_eff_sparse,
        "D_sparse": D_sparse, "G_sparse": G_sparse,
    }
    py_E, py_g = py_energy_and_grad(m, U, B_ext, sparse_ops=sparse_ops_py)
    
    # C++ Energy & Grad
    import ctypes
    lib_path = os.path.join(os.path.dirname(__file__), "../lib/libcpp_mkl_minimizer.so")
    lib = ctypes.CDLL(lib_path)
    
    # Let's call evaluate_energy_and_grad using ctypes
    lib.evaluate_energy_and_grad.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_double
    ]
    lib.evaluate_energy_and_grad.restype = None
    
    # Recreate handles directly in our script to be safe
    mkl_lib = ctypes.CDLL("libmkl_rt.so")
    mkl_lib.mkl_sparse_d_create_csr.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)
    ]
    mkl_lib.mkl_sparse_d_create_csr.restype = ctypes.c_int
    
    K_csr = K_eff_scipy
    K_val = np.ascontiguousarray(K_csr.data, dtype=np.float64)
    K_col = np.ascontiguousarray(K_csr.indices, dtype=np.int32)
    K_ptr = np.ascontiguousarray(K_csr.indptr, dtype=np.int32)
    
    K_handle = ctypes.c_void_p()
    mkl_lib.mkl_sparse_d_create_csr(
        ctypes.byref(K_handle), 0, 3*N, 3*N,
        K_ptr[:-1].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        K_ptr[1:].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        K_col.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        K_val.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    
    G_csr = G_scipy
    G_val = np.ascontiguousarray(G_csr.data, dtype=np.float64)
    G_col = np.ascontiguousarray(G_csr.indices, dtype=np.int32)
    G_ptr = np.ascontiguousarray(G_csr.indptr, dtype=np.int32)
    
    G_handle = ctypes.c_void_p()
    mkl_lib.mkl_sparse_d_create_csr(
        ctypes.byref(G_handle), 0, 3*N, N,
        G_ptr[:-1].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        G_ptr[1:].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        G_col.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        G_val.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    
    # Call C++ energy and grad
    cpp_E = ctypes.c_double(0.0)
    cpp_g = np.zeros(3*N, dtype=np.float64)
    
    m_arr = np.ascontiguousarray(m.reshape(-1), dtype=np.float64)
    U_arr = np.ascontiguousarray(U, dtype=np.float64)
    B_ext_arr = np.ascontiguousarray(B_ext, dtype=np.float64)
    M_nodal_arr = np.ascontiguousarray(M_nodal, dtype=np.float64)
    
    lib.evaluate_energy_and_grad(
        N,
        m_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        U_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        B_ext_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        M_nodal_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        K_handle,
        G_handle,
        ctypes.byref(cpp_E),
        cpp_g.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        1.0 / V_mag
    )
    
    print("\n--- Energy Comparison ---")
    print(f"Python Energy: {py_E:.12f}")
    print(f"C++ Energy   : {cpp_E.value:.12f}")
    print(f"Difference   : {abs(py_E - cpp_E.value):.12e}")
    assert abs(py_E - cpp_E.value) < 1e-10
    
    print("\n--- Gradient Comparison ---")
    py_g_flat = np.array(py_g).reshape(-1)
    max_diff_g = np.max(np.abs(py_g_flat - cpp_g))
    print(f"Max Gradient Diff: {max_diff_g:.12e}")
    assert max_diff_g < 1e-10
    
    # 5. Compare Hessian Action (Ap) and first PCG step
    lib.solve_Py_g.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), # m, g_ext, g_tan_ext, inv_M_prec
        ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, # K_eff, max_iter, tol, reg, stagnation_nu, inv_Vmag
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int) # y, out_iters
    ]
    lib.solve_Py_g.restype = None
    
    # Compute g_tan_ext in python
    g_tan_ext = tangent_grad(m, py_g)
    g_tan_ext_arr = np.ascontiguousarray(g_tan_ext.reshape(-1), dtype=np.float64)
    
    # Prepare inv_M_prec for C++:
    inv_M_prec_arr = np.ascontiguousarray(inv_M_prec.flatten(), dtype=np.float64)
    
    # Run C++ solve_Py_g for exactly 1 iteration
    cpp_y = np.zeros(3*N, dtype=np.float64)
    out_iters = ctypes.c_int()
    lib.solve_Py_g(
        N,
        m_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        py_g_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        g_tan_ext_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        inv_M_prec_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        K_handle,
        1, # 1 iteration
        0.0, # tol
        0.0, # reg
        1e-3, # stagnation_nu
        1.0 / V_mag,
        cpp_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(out_iters)
    )
    
    # Compute Python PCG for exactly 1 iteration
    p_jax = g_tan_ext * inv_M_prec
    
    from minimizers import make_preconditioner_op
    inv_M_rel_arr = np.ascontiguousarray(1.0 / (M_nodal / np.max(M_nodal) + 1e-30), dtype=np.float64)
    apply_P_py, _ = make_preconditioner_op(local_grad_only, jnp.asarray(inv_M_rel_arr))
    
    Ap_py = apply_P_py(m, py_g, p_jax, reg=0.0, sparse_ops=sparse_ops_py)
    
    rho_py = jnp.vdot(g_tan_ext, p_jax)
    pAp_py = jnp.vdot(p_jax, Ap_py)
    alpha_py = rho_py / (pAp_py + 1e-30)
    
    y_py_1_iter = alpha_py * p_jax
    y_py_1_iter_arr = np.ascontiguousarray(y_py_1_iter.reshape(-1), dtype=np.float64)
    
    # Check max difference after 1 iteration of PCG
    max_diff_y = np.max(np.abs(y_py_1_iter_arr - cpp_y))
    print("\n--- PCG 1st Iteration Output (y) Comparison ---")
    print(f"Max Difference in y: {max_diff_y:.12e}")
    assert max_diff_y < 1e-10
    
    # Clean up MKL handles
    mkl_lib.mkl_sparse_destroy(K_handle)
    mkl_lib.mkl_sparse_destroy(G_handle)

if __name__ == "__main__":
    test_compare()
