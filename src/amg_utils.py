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
    A.sum_duplicates()

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

def compute_spai0_diagonal(A: sp.csr_matrix) -> np.ndarray:
    """
    Computes the SPAI0 diagonal preconditioner: M_ii = A_ii / sum_j (A_ij^2).
    """
    N = A.shape[0]
    m_diag = np.zeros(N)
    
    # Square of each element
    A_data_sq = A.data ** 2
    
    for i in range(N):
        row_start = A.indptr[i]
        row_end = A.indptr[i+1]
        
        row_sum_sq = np.sum(A_data_sq[row_start:row_end])
        
        # Find diagonal element A_ii
        a_ii = 0.0
        for j in range(row_start, row_end):
            if A.indices[j] == i:
                a_ii = A.data[j]
                break
        
        m_diag[i] = a_ii / (row_sum_sq + 1e-30)
        
    return m_diag


def setup_amg_hierarchy(A_cpu, max_levels=10):
    """
    Uses PyAMG to compute the AMG hierarchy on CPU.
    Returns a list of dictionaries, one per level of the hierarchy.
    """
    ml = pyamg.smoothed_aggregation_solver(A_cpu, max_levels=max_levels)
    
    hierarchy = []
    for i in range(len(ml.levels)):
        level = ml.levels[i]
        d = {}
        # Matrix A is present on all levels
        csr_A = level.A.tocsr()
        d['A'] = csr_A
        d['Mdiag'] = csr_A.diagonal()
        d['Mdiag_spai0'] = compute_spai0_diagonal(csr_A)
        
        # P and R are only present on levels that have a coarser level below them
        if i < len(ml.levels) - 1:
            d['P'] = level.P.tocsr()
            d['R'] = level.R.tocsr()
        
        # Store dense A for the coarsest level for exact solve
        if i == len(ml.levels) - 1:
            d['A_dense'] = csr_A.todense()
            
        hierarchy.append(d)
        
    return hierarchy

def csr_to_jax_bCOO(mat):
    """Converts a SciPy CSR matrix to a JAX BCOO format."""
    from jax.experimental import sparse
    coo = mat.tocoo()
    indices = jnp.stack([jnp.asarray(coo.row), jnp.asarray(coo.col)], axis=1)
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

@partial(jax.jit, static_argnums=(0,))
def spai0_smooth(apply_A, b, x, Mdiag_spai0, iterations=1):
    """
    SPAI0 smoothing: x = x + M (b - Ax)
    """
    def body(i, x_curr):
        res = b - apply_A(x_curr)
        return x_curr + Mdiag_spai0 * res
    return jax.lax.fori_loop(0, iterations, body, x)

@jax.tree_util.register_pytree_node_class
class AMGHierarchy:
    def __init__(self, levels):
        self.levels = levels
    def tree_flatten(self):
        return (tuple(self.levels), None)
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(list(children))
    def __len__(self):
        return len(self.levels)
    def __getitem__(self, i):
        return self.levels[i]

def make_jax_amg_vcycle(apply_A_fine):
    """
    Returns a function that performs one AMG V-cycle in JAX.
    Matches PyAMG's MultilevelSolver._solve logic.
    """
    def vcycle(r, hierarchy):
        num_levels = len(hierarchy)

        def vcycle_recursive(level_idx, b_curr, x_curr):
            lvl = hierarchy[level_idx]
            # Base case: Coarsest level
            if level_idx == num_levels - 1:
                if 'A_dense' in lvl:
                    return jnp.linalg.solve(lvl['A_dense'], b_curr)
                
                # Fallback
                def apply_A_coarse(v): return lvl['A_sparse'] @ v
                return jacobi_smooth(apply_A_coarse, b_curr, x_curr, lvl['Mdiag'], iterations=10)

            # 1. Setup operator for CURRENT level
            if level_idx == 0:
                def apply_A_curr(v): return apply_A_fine(v)
            else:
                def apply_A_curr(v): return lvl['A_sparse'] @ v
            
            M_curr = lvl['Mdiag']

            # 2. Pre-smooth
            x_curr = jacobi_smooth(apply_A_curr, b_curr, x_curr, M_curr, iterations=1)
            
            # 3. Residual calculation
            r_res = b_curr - apply_A_curr(x_curr)
            
            # 4. Restriction to level_idx + 1
            b_coarse = lvl['R'] @ r_res
            
            # 5. Recurse (Initial guess for error is zero)
            x_coarse = jnp.zeros_like(b_coarse)
            e_coarse = vcycle_recursive(level_idx + 1, b_coarse, x_coarse)
            
            # 6. Prolongation and Correction (x = x + P * e_coarse)
            x_curr = x_curr + lvl['P'] @ e_coarse
            
            # 7. Post-smooth
            x_curr = jacobi_smooth(apply_A_curr, b_curr, x_curr, M_curr, iterations=1)
            
            return x_curr

        return vcycle_recursive(0, r, jnp.zeros_like(r))

    return jax.jit(vcycle)



def make_jax_amgcl_vcycle(apply_A_fine):
    """
    Returns a function that performs one AMG V-cycle in JAX using SPAI0 smoothing.
    """
    def vcycle(r, hierarchy):
        num_levels = len(hierarchy)

        def vcycle_recursive(level_idx, b_curr, x_curr):
            lvl = hierarchy[level_idx]
            
            # Base case: Coarsest level
            if level_idx == num_levels - 1:
                if 'A_dense' in lvl:
                    return jnp.linalg.solve(lvl['A_dense'], b_curr)
                def apply_A_coarse(v): return lvl['A_sparse'] @ v
                return spai0_smooth(apply_A_coarse, b_curr, x_curr, lvl['Mdiag_spai0'], iterations=10)

            # Operator for current level
            if level_idx == 0:
                def apply_A_curr(v): return apply_A_fine(v)
            else:
                def apply_A_curr(v): return lvl['A_sparse'] @ v
            
            M_spai0 = lvl['Mdiag_spai0']

            # 1. Pre-smooth
            x_curr = spai0_smooth(apply_A_curr, b_curr, x_curr, M_spai0, iterations=1)
            
            # 2. Residual calculation
            r_res = b_curr - apply_A_curr(x_curr)
            
            # 3. Restriction
            b_coarse = lvl['R'] @ r_res
            
            # 4. Recurse
            x_coarse = jnp.zeros_like(b_coarse)
            e_coarse = vcycle_recursive(level_idx + 1, b_coarse, x_coarse)
            
            # 5. Prolongation and Correction
            x_curr = x_curr + lvl['P'] @ e_coarse
            
            # 6. Post-smooth
            x_curr = spai0_smooth(apply_A_curr, b_curr, x_curr, M_spai0, iterations=1)
            
            return x_curr

        return vcycle_recursive(0, r, jnp.zeros_like(r))

    return jax.jit(vcycle, static_argnums=(1,))


