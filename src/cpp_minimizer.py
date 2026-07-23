"""C++ minimizer bindings."""

import ctypes
import os

import numpy as np

lib_path = None
# 1. Try local compute node temporary directory first (Slurm Job ID)
slurm_job_id = os.environ.get("SLURM_JOB_ID")
if slurm_job_id:
    local_lib = f"/tmp/mumag_build_{slurm_job_id}/libcpp_mkl_minimizer.so"
    if os.path.exists(local_lib):
        lib_path = local_lib

# 2. Try MUMAG_LIB_OUT if set manually
if not lib_path and "MUMAG_LIB_OUT" in os.environ:
    env_lib = os.path.join(os.environ["MUMAG_LIB_OUT"], "libcpp_mkl_minimizer.so")
    if os.path.exists(env_lib):
        lib_path = env_lib

# 3. Fallback to shared Ceph drive
if not lib_path:
    lib_path = os.path.join(os.path.dirname(__file__), "../lib/libcpp_mkl_minimizer.so")

if not lib_path or not os.path.exists(lib_path):
    lib = None
else:
    lib = ctypes.CDLL(lib_path)

    lib.run_cpp_pcohen_hs_minimization.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),  # m, B_ext, U, M_nodal
        ctypes.POINTER(ctypes.c_double),  # boundary_mask
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),  # inv_M_rel, inv_M_prec
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,  # pc_auto, pc_force_eta, pc_force_alpha
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,  # cg_tol, pc_iters, pc_reg, pc_stagnation_nu, L, beta_type
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),  # out_iters, out_evals, out_demag, out_preco
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.run_cpp_pcohen_hs_minimization.restype = ctypes.c_int


def cpp_minimize(m, B_ext, U0, params, sparse_ops, solve_U=None, **kwargs):
    """Run minimization using C++ backend."""
    if lib is None:
        raise FileNotFoundError("Could not find libcpp_mkl_minimizer.so. Please compile the C++ minimizer.")
    N = U0.shape[0]

    m_arr = np.ascontiguousarray(m.reshape(-1), dtype=np.float64)
    B_ext_arr = np.ascontiguousarray(B_ext, dtype=np.float64)
    U_arr = np.ascontiguousarray(U0, dtype=np.float64)
    M_nodal = np.ascontiguousarray(params.M_nodal, dtype=np.float64)
    V_mag = params.V_mag if hasattr(params, "V_mag") else 1.0
    inv_M_rel_val = np.where(params.M_nodal > 1e-20, V_mag / params.M_nodal, 0.0)
    inv_M_rel = np.ascontiguousarray(inv_M_rel_val, dtype=np.float64)
    inv_M_prec = np.ascontiguousarray(params.inv_M_prec, dtype=np.float64)

    boundary_mask = kwargs.get("boundary_mask")
    if boundary_mask is not None:
        boundary_mask_np = np.ascontiguousarray(boundary_mask, dtype=np.float64)
        boundary_mask_ptr = boundary_mask_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        boundary_mask_np = None
        boundary_mask_ptr = None

    K_eff = sparse_ops["K_eff_sparse"]
    K_val = np.ascontiguousarray(K_eff.data, dtype=np.float64)
    K_col = np.ascontiguousarray(K_eff.indices, dtype=np.int32)
    K_ptr = np.ascontiguousarray(K_eff.indptr, dtype=np.int32)

    G_sparse = sparse_ops["G_sparse"]
    G_val = np.ascontiguousarray(G_sparse.data, dtype=np.float64)
    G_col = np.ascontiguousarray(G_sparse.indices, dtype=np.int32)
    G_ptr = np.ascontiguousarray(G_sparse.indptr, dtype=np.int32)

    D_sparse = sparse_ops["D_sparse"]
    D_val = np.ascontiguousarray(D_sparse.data, dtype=np.float64)
    D_col = np.ascontiguousarray(D_sparse.indices, dtype=np.int32)
    D_ptr = np.ascontiguousarray(D_sparse.indptr, dtype=np.int32)

    pardiso_handle = 0
    if solve_U is not None and hasattr(solve_U, "pardiso_obj"):
        pardiso_handle = int(solve_U.pardiso_obj.handle_id)

    out_iters = ctypes.c_int()
    out_evals = ctypes.c_int()
    out_demag = ctypes.c_int()
    out_preco = ctypes.c_int()
    out_E = ctypes.c_double()
    out_gnorm = ctypes.c_double()
    inv_Vmag = 1.0 / params.V_mag if hasattr(params, "V_mag") else 1.0

    pc_auto_val = 1 if getattr(params, "pc_auto", True) else 0
    pc_force_eta_val = getattr(params, "pc_force_eta", 0.5)
    pc_force_alpha_val = getattr(params, "pc_force_alpha", 0.5)

    method = getattr(params, "method", "pcohen_hs")
    beta_type_val = 1 if "hs" in method else 0

    lib.run_cpp_pcohen_hs_minimization(
        N,
        getattr(params, "max_iter", 2000),
        getattr(params, "tau_f", 1e-8),
        getattr(params, "eps_a", kwargs.get("eps_a", 1e-6)),
        getattr(params, "ls_eta1", 1e-4),
        getattr(params, "ls_eta2", 0.9),
        getattr(params, "ls_C", 2.0),
        getattr(params, "ls_c", 0.5),
        getattr(params, "ls_max_evals", 15),
        inv_Vmag,
        pardiso_handle,
        m_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        B_ext_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        U_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        M_nodal.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        boundary_mask_ptr,
        K_val.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        K_col.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        K_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        G_val.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        G_col.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        G_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        D_val.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        D_col.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        D_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        inv_M_rel.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        inv_M_prec.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pc_auto_val,
        pc_force_eta_val,
        pc_force_alpha_val,
        getattr(params, "pc_tol", 0.01),
        getattr(params, "pc_iters", 10),
        getattr(params, "pc_reg", 0.0),
        getattr(params, "pc_stagnation_nu", 1e-3),
        params.L if getattr(params, "L", None) is not None else N,
        beta_type_val,
        ctypes.byref(out_iters),
        ctypes.byref(out_evals),
        ctypes.byref(out_demag),
        ctypes.byref(out_preco),
        ctypes.byref(out_E),
        ctypes.byref(out_gnorm),
    )

    m_new = m_arr.reshape(N, 3)
    info = {
        "iters": out_iters.value,
        "evals": out_evals.value,
        "demag": out_demag.value,
        "demag_iters": out_demag.value,
        "preco_iters": out_preco.value,
        "E": out_E.value,
        "gnorm": out_gnorm.value,
    }

    import jax.numpy as jnp

    return jnp.array(m_new), jnp.array(U_arr), info
