import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
import pyamg
import scipy.sparse as sp

jax.config.update("jax_enable_x64", True)
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from amg_utils import make_jax_mkl_solve_linear, make_jax_amgcl_vcycle, compute_spai0_diagonal, make_sparse_operator
from poisson_solve import make_pcg_solve

def test():
    N = 100
    A = pyamg.gallery.poisson((N, N), format='csr')
    
    # Generate PyAMG hierarchy
    ml = pyamg.smoothed_aggregation_solver(A, strength="symmetric")
    
    # 1. JAX pure Python solve
    levels_jax = []
    for i in range(len(ml.levels)):
        lvl = ml.levels[i]
        csr_A = lvl.A.tocsr()
        level_dict = {
            "A_sparse": None if i == 0 else make_sparse_operator(csr_A, cpu_spmv_backend="scipy"),
            "Mdiag": jnp.asarray(csr_A.diagonal()),
            "Mdiag_spai0": jnp.asarray(compute_spai0_diagonal(csr_A))
        }
        if i < len(ml.levels) - 1:
            level_dict["P"] = make_sparse_operator(lvl.P.tocsr(), cpu_spmv_backend="scipy")
            level_dict["R"] = make_sparse_operator(lvl.R.tocsr(), cpu_spmv_backend="scipy")
        else:
            level_dict["A_dense"] = jnp.asarray(csr_A.todense())
        levels_jax.append(level_dict)
        
    def apply_A(sparse_ops, v):
        return A @ v
        
    apply_Minv_amg = make_jax_amgcl_vcycle(apply_A)
    solve_jax = make_pcg_solve(
        apply_A,
        jnp.asarray(A.diagonal()),
        precond_type="amgcl",
        apply_Minv_amg=apply_Minv_amg,
        order=1,
        maxiter=100,
        tol=1e-8,
        boundary_mask=None,
        l_max=2.0
    )
    
    from amg_utils import AMGHierarchy
    hierarchy_jax = AMGHierarchy(levels_jax)
    
    b = np.random.rand(A.shape[0])
    x0 = np.zeros_like(b)
    
    #x_jax_pure, it_jax, res_jax = solve_jax({}, jnp.array(b), jnp.array(x0), tol=1e-8, hierarchy=hierarchy_jax)
    
    #residual_jax = np.linalg.norm(b - A @ x_jax_pure)
    #print(f"Pure JAX Residual: {residual_jax}")
    
    # 2. JAX MKL solve
    solve_mkl = make_jax_mkl_solve_linear(ml, cg_maxiter=2000, cg_tol=1e-8)
    x_jax_mkl, _, _ = solve_mkl({}, jnp.array(b), jnp.array(x0), tol=1e-8)
    
    residual_mkl = np.linalg.norm(b - A @ x_jax_mkl)
    print(f"MKL FFI Residual: {residual_mkl}")
    assert residual_mkl < 1e-6

if __name__ == "__main__":
    test()
