"""amg_utils.py

Utilities for Algebraic Multigrid (AMG) setup using PyAMG.
Assembles the Poisson matrix on CPU and prepares the hierarchy for JAX.
"""

import numpy as np
import scipy.sparse as sp
import pyamg
import jax
import jax.numpy as jnp
from functools import partial
from typing import List, Tuple, Optional

def assemble_poisson_matrix_cpu(conn, volume, grad_phi, boundary_mask=None, reg=1e-12):
    """
    Assembles the Poisson matrix A in CSR format on the CPU.
    
    conn: (E, 4) node indices
    volume: (E,) element volumes
    grad_phi: (E, 4, 3) shape function gradients
    boundary_mask: (N,) mask where 1.0 is interior, 0.0 is Dirichlet boundary
    reg: regularization constant for the diagonal
    """
    E = conn.shape[0]
    N = np.max(conn) + 1
    
    # Each tet adds 4x4 = 16 entries to the global matrix
    # Local element stiffness matrix: Ke_ab = Ve * (grad_phi_a . grad_phi_b)
    Ke = volume[:, None, None] * np.einsum('eai,ebi->eab', grad_phi, grad_phi)
    
    # Indices for global assembly
    rows = np.repeat(conn, 4, axis=1).flatten()
    cols = np.tile(conn, (1, 4)).flatten()
    data = Ke.flatten()
    
    # Create sparse matrix
    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    
    if boundary_mask is not None:
        # For Dirichlet boundary nodes (mask == 0), we want A_ii = 1, A_ij = 0, A_ji = 0
        mask = np.array(boundary_mask)
        boundary_nodes = np.where(mask == 0)[0]
        
        # Zero out rows and columns to maintain symmetry
        # 1. Zero rows
        for i in boundary_nodes:
            r_start = A.indptr[i]
            r_end = A.indptr[i+1]
            A.data[r_start:r_end] = 0.0
            
        # 2. Zero columns (requires CSC format for efficiency or COO conversion)
        A = A.tocoo()
        mask_indices = (mask[A.row] > 0) & (mask[A.col] > 0)
        A.data = A.data[mask_indices]
        A.row = A.row[mask_indices]
        A.col = A.col[mask_indices]
        A = sp.csr_matrix((A.data, (A.row, A.col)), shape=(N, N))
        
    # Add regularization to diagonal
    A = A + reg * sp.eye(N, format='csr')
    
    if boundary_mask is not None:
        # Ensure diagonal is 1 for boundary nodes
        diag = A.diagonal()
        diag[boundary_nodes] = 1.0
        A.setdiag(diag)

    return A

def setup_amg_hierarchy(A_cpu, max_levels=5):
    """
    Uses PyAMG to compute the AMG hierarchy on CPU.
    Returns a list of restriction/prolongation matrices and coarse operators.
    """
    ml = pyamg.smoothed_aggregation_solver(A_cpu, max_levels=max_levels)
    
    hierarchy = []
    for i in range(len(ml.levels) - 1):
        level = ml.levels[i]
        # P is prolongation, R is restriction (usually P.T)
        P = level.P.tocsr()
        R = level.R.tocsr()
        # Coarse operator
        A_coarse = ml.levels[i+1].A.tocsr()
        
        hierarchy.append({
            'P': P,
            'R': R,
            'A': A_coarse
        })
        
    return hierarchy

def csr_to_jax_bCOO(mat):
    """Converts a SciPy CSR matrix to a JAX BCOO format."""
    from jax.experimental import sparse
    coo = mat.tocoo()
    indices = jnp.stack([coo.row, coo.col], axis=1)
    return sparse.BCOO((jnp.asarray(coo.data), jnp.asarray(indices)), shape=coo.shape)

@partial(jax.jit, static_argnums=(0,))
def jacobi_smooth(apply_A, b, x, Mdiag, iterations=1, omega=0.6667):
    """
    Standard Jacobi iteration: x_{k+1} = x_k + omega * D^-1 * (b - A x_k)
    This matches PyAMG's smoothing interface.
    """
    def body(i, x_curr):
        res = b - apply_A(x_curr)
        return x_curr + omega * (res / (Mdiag + 1e-30))
    return jax.lax.fori_loop(0, iterations, body, x)

def make_jax_amg_vcycle(apply_A_fine, Mdiag_fine, hierarchy_jax):
    """
    Returns a function that performs one AMG V-cycle in JAX.
    Matches PyAMG's MultilevelSolver._solve logic.
    """
    num_levels = len(hierarchy_jax) + 1

    def vcycle_recursive(level_idx, b_curr, x_curr):
        # Base case: Coarsest level
        if level_idx == num_levels - 1:
            # Solve exactly with dense solve if small
            A_dict = hierarchy_jax[-1]
            if 'A_dense' in A_dict:
                return jnp.linalg.solve(A_dict['A_dense'], b_curr)
            
            # Fallback for Jacobi iterations on coarsest
            A_coarse = A_dict['A_sparse']
            M_coarse = A_dict['Mdiag']
            def apply_A_coarse(v): return A_coarse @ v
            return jacobi_smooth(apply_A_coarse, b_curr, x_curr, M_coarse, iterations=10)

        # 1. Setup operator for CURRENT level
        if level_idx == 0:
            def apply_A_curr(v): return apply_A_fine(v)
            M_curr = Mdiag_fine
        else:
            # The operator for level i is stored in hierarchy_jax[i-1]
            A_sparse = hierarchy_jax[level_idx-1]['A_sparse']
            def apply_A_curr(v): return A_sparse @ v
            M_curr = hierarchy_jax[level_idx-1]['Mdiag']

        # 2. Pre-smooth (x = x + relax(A, x, b))
        x_curr = jacobi_smooth(apply_A_curr, b_curr, x_curr, M_curr, iterations=1)
        
        # 3. Residual calculation
        r_res = b_curr - apply_A_curr(x_curr)
        
        # 4. Restriction to level_idx + 1
        # b_coarse = R * r_res
        b_coarse = hierarchy_jax[level_idx]['R'] @ r_res
        
        # 5. Recurse (Initial guess for error is zero)
        x_coarse = jnp.zeros_like(b_coarse)
        e_coarse = vcycle_recursive(level_idx + 1, b_coarse, x_coarse)
        
        # 6. Prolongation and Correction (x = x + P * e_coarse)
        x_curr = x_curr + hierarchy_jax[level_idx]['P'] @ e_coarse
        
        # 7. Post-smooth
        x_curr = jacobi_smooth(apply_A_curr, b_curr, x_curr, M_curr, iterations=1)
        
        return x_curr

    return jax.jit(lambda r: vcycle_recursive(0, r, jnp.zeros_like(r)))
