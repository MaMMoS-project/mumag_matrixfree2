"""amg_utils.py.

Utilities for Algebraic Multigrid (AMG) setup using PyAMG.
Assembles the Poisson matrix on CPU and prepares the hierarchy for JAX.
"""

from collections.abc import Callable
from functools import partial
from typing import Any

import ctypes
import jax
import jax.numpy as jnp
import numpy as np
import pyamg
import scipy.sparse as sp


def assemble_poisson_matrix_cpu(
    conn: np.ndarray,
    volume: np.ndarray,
    grad_phi: np.ndarray,
    boundary_mask: np.ndarray | None = None,
    reg: float = 1e-12,
) -> sp.csr_matrix:
    """Assemble the Poisson stiffness matrix in CSR format on the CPU.

    Args:
        conn (np.ndarray): Tetrahedron connectivity (E, 4).
        volume (np.ndarray): Element volumes (E,).
        grad_phi (np.ndarray): Shape function gradients (E, 4, 3).
        boundary_mask (np.ndarray | None, optional): Mask where 1.0 is interior
            and 0.0 is Dirichlet boundary. Defaults to None.
        reg (float, optional): Regularization constant for the diagonal.
            Defaults to 1e-12.

    Returns:
        sp.csr_matrix: The assembled sparse stiffness matrix.
    """
    N = np.max(conn) + 1

    # Each tet adds 4x4 = 16 entries to the global matrix
    # Local element stiffness matrix: Ke_ab = Ve * (grad_phi_a . grad_phi_b)

    Ke = volume[:, None, None] * np.einsum("eai,ebi->eab", grad_phi, grad_phi)

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
        is_boundary = (mask == 0)

        # Zero out rows and columns to maintain symmetry
        # 1. Zero rows (vectorized)
        row_indices = np.repeat(np.arange(N), np.diff(A.indptr))
        A.data[is_boundary[row_indices]] = 0.0

        # 2. Zero columns (vectorized)
        A.data[is_boundary[A.indices]] = 0.0

        # Remove explicit zeros
        A.eliminate_zeros()

    # Add regularization to diagonal
    A = A + reg * sp.eye(N, format="csr")

    if boundary_mask is not None:
        # Ensure diagonal is 1 for boundary nodes
        diag = A.diagonal()
        diag[boundary_nodes] = 1.0
        A.setdiag(diag)

    return A


def compute_spai0_diagonal(A: sp.csr_matrix) -> np.ndarray:
    """Compute the SPAI0 (Sparse Approximate Inverse) diagonal preconditioner.

    M_ii = A_ii / sum_j (A_ij^2).

    Choice: SPAI0 is chosen for the AMGCL-style V-cycle because it only
    requires matrix-vector products for smoothing. Unlike Gauss-Seidel,
    this is highly parallelizable and efficient for JAX/GPU execution.

    Args:
        A (sp.csr_matrix): The sparse matrix.

    Returns:
        np.ndarray: The SPAI0 diagonal elements.
    """
    # Square of each element
    A_sq = sp.csr_matrix((A.data**2, A.indices, A.indptr), shape=A.shape)
    
    # Sum over rows
    row_sum_sq = np.array(A_sq.sum(axis=1)).ravel()
    
    # Diagonal elements A_ii
    a_ii = A.diagonal()

    return a_ii / (row_sum_sq + 1e-30)


def setup_amg_hierarchy(A_cpu: sp.csr_matrix, max_levels: int = 10) -> list[dict]:
    """Compute the AMG hierarchy on the CPU using PyAMG.

    Args:
        A_cpu (sp.csr_matrix): The fine-level stiffness matrix.
        max_levels (int, optional): Maximum number of levels. Defaults to 10.

    Returns:
        list[dict]: A list of dictionaries, one per level, containing matrices
            (A, P, R) and preconditioner diagonals.
    """
    ml = pyamg.smoothed_aggregation_solver(A_cpu, max_levels=max_levels)

    hierarchy = []
    for i in range(len(ml.levels)):
        level = ml.levels[i]
        d = {}
        # Matrix A is present on all levels
        csr_A = level.A.tocsr()
        d["A"] = csr_A
        d["Mdiag"] = csr_A.diagonal()
        d["Mdiag_spai0"] = compute_spai0_diagonal(csr_A)

        # P and R are only present on levels that have a coarser level below them
        if i < len(ml.levels) - 1:
            d["P"] = level.P.tocsr()
            d["R"] = level.R.tocsr()

        # Store dense A for the coarsest level for exact solve
        if i == len(ml.levels) - 1:
            d["A_dense"] = csr_A.todense()

        hierarchy.append(d)

    return hierarchy


def csr_to_jax_CSR(mat: sp.csr_matrix) -> Any:
    """Convert a SciPy CSR matrix to JAX CSR format.

    Args:
        mat (sp.csr_matrix): Input SciPy sparse matrix.

    Returns:
        jax.experimental.sparse.CSR: The JAX sparse matrix.
    """
    from jax.experimental import sparse

    return sparse.CSR(
        (jnp.asarray(mat.data), jnp.asarray(mat.indices), jnp.asarray(mat.indptr)),
        shape=mat.shape,
    )


@jax.tree_util.register_pytree_node_class
class SparseOperator:
    """A wrapper for sparse matrix operations that overrides the matmul (@) operator.
    This allows JAX to trace both CPU and GPU execution paths cleanly.
    """
    def __init__(self, apply_fn, pytree_parts=()):
        self.apply_fn = apply_fn
        self.pytree_parts = pytree_parts

    def __matmul__(self, other):
        # If there are dynamic JAX arrays (like the GPU CSR object), pass them to apply_fn
        if len(self.pytree_parts) > 0:
            return self.apply_fn(self.pytree_parts[0], other)
        return self.apply_fn(None, other)

    def tree_flatten(self):
        return (self.pytree_parts, (self.apply_fn,))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        apply_fn, = aux_data
        return cls(apply_fn, pytree_parts=children)


class MatrixDescr(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("mode", ctypes.c_int),
        ("diag", ctypes.c_int)
    ]


class OptimizedMKLMatrix:
    """Manages the lifecycle of an Intel MKL optimized sparse matrix handle (Inspector-Executor)."""
    def __init__(self, scipy_csr: sp.csr_matrix):
        try:
            from sparse_dot_mkl._mkl_interface import MKL
        except ImportError:
            raise ImportError("sparse_dot_mkl is required for MKL Inspector-Executor API.")

        # Bind prototype arguments dynamically if not already bound
        if not hasattr(MKL, "mkl_sparse_d_create_csr_bound"):
            MKL.mkl_sparse_d_create_csr.argtypes = [
                ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
            ]
            MKL.mkl_sparse_d_create_csr.restype = ctypes.c_int

            MKL.mkl_sparse_set_mv_hint.argtypes = [
                ctypes.c_void_p, ctypes.c_int, MatrixDescr, ctypes.c_int
            ]
            MKL.mkl_sparse_set_mv_hint.restype = ctypes.c_int

            MKL.mkl_sparse_optimize.argtypes = [ctypes.c_void_p]
            MKL.mkl_sparse_optimize.restype = ctypes.c_int

            MKL.mkl_sparse_d_mv.argtypes = [
                ctypes.c_int, ctypes.c_double, ctypes.c_void_p, MatrixDescr,
                ctypes.c_void_p, ctypes.c_double, ctypes.c_void_p
            ]
            MKL.mkl_sparse_d_mv.restype = ctypes.c_int

            MKL.mkl_sparse_destroy.argtypes = [ctypes.c_void_p]
            MKL.mkl_sparse_destroy.restype = ctypes.c_int
            MKL.mkl_sparse_d_create_csr_bound = True

        self.MKL = MKL
        self.shape = scipy_csr.shape

        # Enforce strict contiguous 32-bit integer array types for LP64 MKL compatibility
        self.data = np.ascontiguousarray(scipy_csr.data, dtype=np.float64)
        self.indices = np.ascontiguousarray(scipy_csr.indices, dtype=np.int32)
        self.indptr = np.ascontiguousarray(scipy_csr.indptr, dtype=np.int32)

        # Create MKL sparse matrix handle
        self.handle = ctypes.c_void_p()
        status = MKL.mkl_sparse_d_create_csr(
            ctypes.byref(self.handle),
            0, # 0-based indexing (C-style)
            self.shape[0],
            self.shape[1],
            self.indptr[:-1].ctypes.data,
            self.indptr[1:].ctypes.data,
            self.indices.ctypes.data,
            self.data.ctypes.data
        )
        if status != 0:
            raise RuntimeError(f"mkl_sparse_d_create_csr failed with status: {status}")

        # Set general matrix description
        self.descr = MatrixDescr(type=20, mode=0, diag=0) # SPARSE_MATRIX_TYPE_GENERAL

        # Set Hint: matrix-vector product (10), expected iterations 2000
        MKL.mkl_sparse_set_mv_hint(self.handle, 10, self.descr, 2000)

        # Optimize the sparse matrix layout
        status = MKL.mkl_sparse_optimize(self.handle)
        if status != 0:
            raise RuntimeError(f"mkl_sparse_optimize failed with status: {status}")

    def spmv(self, x: np.ndarray, y: np.ndarray) -> None:
        # y = 1.0 * A * x + 0.0 * y
        self.MKL.mkl_sparse_d_mv(
            10, # SPARSE_OPERATION_NON_TRANSPOSE
            1.0,
            self.handle,
            self.descr,
            x.ctypes.data,
            0.0,
            y.ctypes.data
        )

    def __del__(self):
        if hasattr(self, "handle") and self.handle:
            self.MKL.mkl_sparse_destroy(self.handle)


def make_cpu_csr_op(scipy_csr_mat: sp.csr_matrix):
    """Creates a fast, multicore CPU SpMV operator using optimized Intel MKL handles."""
    try:
        optimized_matrix = OptimizedMKLMatrix(scipy_csr_mat)
    except Exception as e:
        # Fall back to standard sparse_dot_mkl if optimization fails or isn't available
        from sparse_dot_mkl import dot_product_mkl
        def mkl_spmv_fallback(x_val, **kwargs):
            x_np_val = np.asarray(x_val, dtype=scipy_csr_mat.dtype)
            return dot_product_mkl(scipy_csr_mat, x_np_val)
        
        @jax.jit
        def fast_cpu_spmv_fallback(x_val):
            result_shape_dtype = jax.ShapeDtypeStruct((scipy_csr_mat.shape[0],), x_val.dtype)
            return jax.pure_callback(
                mkl_spmv_fallback,
                result_shape_dtype,
                x_val,
                vectorized=False
            )
        return fast_cpu_spmv_fallback

    # Local buffer callback for thread safety (no shared state)
    def mkl_spmv_callback(x_val, **kwargs):
        x_np_val = np.ascontiguousarray(x_val, dtype=np.float64)
        output_buffer = np.empty(scipy_csr_mat.shape[0], dtype=np.float64)
        optimized_matrix.spmv(x_np_val, output_buffer)
        return output_buffer

    @jax.jit
    def fast_cpu_spmv(x_val):
        result_shape_dtype = jax.ShapeDtypeStruct((scipy_csr_mat.shape[0],), x_val.dtype)
        return jax.pure_callback(
            mkl_spmv_callback,
            result_shape_dtype,
            x_val,
            vectorized=False
        )

    return fast_cpu_spmv


def make_sparse_operator(scipy_csr_mat: sp.csr_matrix) -> SparseOperator:
    """Dynamically creates the optimal sparse operator depending on the active platform."""
    device = jax.devices()[0]

    if device.platform == "cpu":
        cpu_op = make_cpu_csr_op(scipy_csr_mat)
        # On CPU: no dynamic JAX arrays, pass CPU op as static function
        return SparseOperator(lambda _, x: cpu_op(x), ())
    else:
        # On GPU: convert to JAX CSR and store it in pytree_parts
        jax_csr = csr_to_jax_CSR(scipy_csr_mat)
        return SparseOperator(lambda matrix, x: matrix @ x, (jax_csr,))


@partial(jax.jit, static_argnums=(0,))
def jacobi_smooth(
    apply_A: Callable,
    b: jnp.ndarray,
    x: jnp.ndarray,
    Mdiag: jnp.ndarray,
    iterations: int = 1,
    omega: float = 0.6667,
) -> jnp.ndarray:
    """Apply Jacobi smoothing: x_{k+1} = x_k + omega * D^-1 * (b - A x_k).

    Args:
        apply_A (Callable): Function that computes the matrix-vector product.
        b (Array): Right-hand side vector.
        x (Array): Initial guess.
        Mdiag (Array): Diagonal of the matrix A.
        iterations (int, optional): Number of iterations. Defaults to 1.
        omega (float, optional): Relaxation factor. Defaults to 0.6667.

    Returns:
        Array: The smoothed solution.
    """

    def body(i, x_curr):
        res = b - apply_A(x_curr)
        return x_curr + omega * (res / (Mdiag + 1e-30))

    return jax.lax.fori_loop(0, iterations, body, x)


@partial(jax.jit, static_argnums=(0,))
def spai0_smooth(
    apply_A: Callable,
    b: jnp.ndarray,
    x: jnp.ndarray,
    Mdiag_spai0: jnp.ndarray,
    iterations: int = 1,
) -> jnp.ndarray:
    """Apply SPAI0 smoothing: x = x + M (b - Ax).

    Args:
        apply_A (Callable): Function that computes the matrix-vector product.
        b (Array): Right-hand side vector.
        x (Array): Initial guess.
        Mdiag_spai0 (Array): SPAI0 diagonal preconditioner.
        iterations (int, optional): Number of iterations. Defaults to 1.

    Returns:
        Array: The smoothed solution.
    """

    def body(i, x_curr):
        res = b - apply_A(x_curr)
        return x_curr + Mdiag_spai0 * res

    return jax.lax.fori_loop(0, iterations, body, x)


@jax.tree_util.register_pytree_node_class
class AMGHierarchy:
    """JAX PyTree container for the AMG hierarchy levels.

    Allows the hierarchy to be passed through JAX-JITted functions.
    """

    def __init__(self, levels):
        """Initialize the hierarchy.

        Args:
            levels (list): List of dictionaries containing level data.
        """
        self.levels = levels

    def tree_flatten(self):
        """Flatten the hierarchy for JAX.

        Returns:
            tuple: (levels, aux_data).
        """
        return (tuple(self.levels), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten the hierarchy for JAX.

        Args:
            aux_data: Auxiliary data.
            children: The flattened components.

        Returns:
            AMGHierarchy: The unflattened hierarchy.
        """
        return cls(list(children))

    def __len__(self):
        """Return the number of levels.

        Returns:
            int: Number of levels.
        """
        return len(self.levels)

    def __getitem__(self, i):
        """Get a specific level.

        Args:
            i (int): Level index.

        Returns:
            dict: Level data.
        """
        return self.levels[i]


def make_jax_amg_vcycle(apply_A_fine: Callable) -> Callable:
    """Create a JAX function that performs one AMG V-cycle.

    Matches PyAMG's MultilevelSolver logic.

    Args:
        apply_A_fine (Callable): Matrix-vector product for the finest level.

    Returns:
        Callable: A JIT-compiled function vcycle(rhs, hierarchy).
    """

    def vcycle(sparse_ops, r, hierarchy):
        num_levels = len(hierarchy)

        def vcycle_recursive(level_idx, b_curr, x_curr):
            lvl = hierarchy[level_idx]
            # Base case: Coarsest level
            if level_idx == num_levels - 1:
                if "A_dense" in lvl:
                    return jnp.linalg.solve(lvl["A_dense"], b_curr)

                # Fallback
                def apply_A_coarse(v):
                    return lvl["A_sparse"] @ v

                return jacobi_smooth(apply_A_coarse, b_curr, x_curr, lvl["Mdiag"], iterations=10)

            # 1. Setup operator for CURRENT level
            if level_idx == 0:

                def apply_A_curr(v):
                    return apply_A_fine(sparse_ops, v)
            else:

                def apply_A_curr(v):
                    return lvl["A_sparse"] @ v

            M_curr = lvl["Mdiag"]

            # 2. Pre-smooth
            x_curr = jacobi_smooth(apply_A_curr, b_curr, x_curr, M_curr, iterations=1)

            # 3. Residual calculation
            r_res = b_curr - apply_A_curr(x_curr)

            # 4. Restriction to level_idx + 1
            b_coarse = lvl["R"] @ r_res

            # 5. Recurse (Initial guess for error is zero)
            x_coarse = jnp.zeros_like(b_coarse)
            e_coarse = vcycle_recursive(level_idx + 1, b_coarse, x_coarse)

            # 6. Prolongation and Correction (x = x + P * e_coarse)
            x_curr = x_curr + lvl["P"] @ e_coarse

            # 7. Post-smooth
            x_curr = jacobi_smooth(apply_A_curr, b_curr, x_curr, M_curr, iterations=1)

            return x_curr

        return vcycle_recursive(0, r, jnp.zeros_like(r))

    return jax.jit(vcycle)


def make_jax_amgcl_vcycle(apply_A_fine: Callable) -> Callable:
    """Create a JAX function for an AMG V-cycle using SPAI0 smoothing.

    Optimized for GPU execution through SPAI0 (matrix-vector products only).

    Args:
        apply_A_fine (Callable): Matrix-vector product for the finest level.

    Returns:
        Callable: A JIT-compiled function vcycle(rhs, hierarchy).
    """

    def vcycle(sparse_ops, r, hierarchy):
        num_levels = len(hierarchy)

        def vcycle_recursive(level_idx, b_curr, x_curr):
            lvl = hierarchy[level_idx]

            # Base case: Coarsest level
            if level_idx == num_levels - 1:
                if "A_dense" in lvl:
                    return jnp.linalg.solve(lvl["A_dense"], b_curr)

                def apply_A_coarse(v):
                    return lvl["A_sparse"] @ v

                return spai0_smooth(apply_A_coarse, b_curr, x_curr, lvl["Mdiag_spai0"], iterations=10)

            # Operator for current level
            if level_idx == 0:

                def apply_A_curr(v):
                    return apply_A_fine(sparse_ops, v)
            else:

                def apply_A_curr(v):
                    return lvl["A_sparse"] @ v

            M_spai0 = lvl["Mdiag_spai0"]

            # 1. Pre-smooth
            x_curr = spai0_smooth(apply_A_curr, b_curr, x_curr, M_spai0, iterations=1)

            # 2. Residual calculation
            r_res = b_curr - apply_A_curr(x_curr)

            # 3. Restriction
            b_coarse = lvl["R"] @ r_res

            # 4. Recurse
            x_coarse = jax.lax.cond(b_coarse[0] == 12345.6789, lambda: b_coarse, lambda: jnp.zeros_like(b_coarse))
            e_coarse = vcycle_recursive(level_idx + 1, b_coarse, x_coarse)

            # 5. Prolongation and Correction
            x_curr = x_curr + lvl["P"] @ e_coarse

            # 6. Post-smooth
            x_curr = spai0_smooth(apply_A_curr, b_curr, x_curr, M_spai0, iterations=1)

            return x_curr

        # Start with a dynamically-shielded zero vector to prevent XLA from 
        # treating `x_curr` as a static constant and unrolling/folding apply_A_fine.
        x_start = jax.lax.cond(r[0] == 12345.6789, lambda: r, lambda: jnp.zeros_like(r))
        return vcycle_recursive(0, r, x_start)

    return jax.jit(vcycle)


def assemble_exchange_matrix_cpu(
    conn: np.ndarray,
    volume: np.ndarray,
    grad_phi: np.ndarray,
    A_lookup: np.ndarray,
    mat_id: np.ndarray,
) -> sp.csr_matrix:
    """Assemble the Exchange stiffness matrix K_ex in CSR format on the CPU.

    Args:
        conn (np.ndarray): Tetrahedron connectivity (E, 4).
        volume (np.ndarray): Element volumes (E,).
        grad_phi (np.ndarray): Shape function gradients (E, 4, 3).
        A_lookup (np.ndarray): Material exchange constants (G,).
        mat_id (np.ndarray): Material IDs per element (E,).

    Returns:
        sp.csr_matrix: The assembled exchange matrix of shape (N, N).
    """
    N = np.max(conn) + 1
    A_elem = A_lookup[mat_id - 1]

    # Ke_ab = 2 * A_ex * Ve * (grad_phi_a . grad_phi_b)
    Ke = 2.0 * A_elem[:, None, None] * volume[:, None, None] * np.einsum("eai,ebi->eab", grad_phi, grad_phi)

    rows = np.repeat(conn, 4, axis=1).flatten()
    cols = np.tile(conn, (1, 4)).flatten()
    data = Ke.flatten()

    Kex = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    Kex.sum_duplicates()
    return Kex


def assemble_divergence_matrices_cpu(
    conn: np.ndarray,
    volume: np.ndarray,
    grad_phi: np.ndarray,
    Js_lookup: np.ndarray,
    mat_id: np.ndarray,
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Assemble the Divergence matrices Dx, Dy, Dz in CSR format on the CPU.

    These matrices map magnetization components to the Poisson charge density.
    (D_i)_ab^e = Js_e * (Ve / 4) * (grad_phi_a)_i for all column indices b.

    Args:
        conn (np.ndarray): Tetrahedron connectivity (E, 4).
        volume (np.ndarray): Element volumes (E,).
        grad_phi (np.ndarray): Shape function gradients (E, 4, 3).
        Js_lookup (np.ndarray): Material saturation polarizations (G,).
        mat_id (np.ndarray): Material IDs per element (E,).

    Returns:
        tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]: (Dx, Dy, Dz) CSR matrices.
    """
    N = np.max(conn) + 1
    Js_elem = Js_lookup[mat_id - 1]
    factor = Js_elem * volume / 4.0

    De_x = factor[:, None, None] * np.tile(grad_phi[:, :, 0][:, :, None], (1, 1, 4))
    De_y = factor[:, None, None] * np.tile(grad_phi[:, :, 1][:, :, None], (1, 1, 4))
    De_z = factor[:, None, None] * np.tile(grad_phi[:, :, 2][:, :, None], (1, 1, 4))

    rows = np.repeat(conn, 4, axis=1).flatten()
    cols = np.tile(conn, (1, 4)).flatten()

    Dx = sp.coo_matrix((De_x.flatten(), (rows, cols)), shape=(N, N)).tocsr()
    Dx.sum_duplicates()

    Dy = sp.coo_matrix((De_y.flatten(), (rows, cols)), shape=(N, N)).tocsr()
    Dy.sum_duplicates()

    Dz = sp.coo_matrix((De_z.flatten(), (rows, cols)), shape=(N, N)).tocsr()
    Dz.sum_duplicates()

    return Dx, Dy, Dz


def assemble_anisotropy_matrix_cpu(
    conn: np.ndarray,
    volume: np.ndarray,
    K1_lookup: np.ndarray,
    mat_id: np.ndarray,
) -> sp.csr_matrix:
    """Assemble the Uniaxial Anisotropy stiffness matrix K_an in CSR format on the CPU.

    Local element stiffness contribution:
    (K_an)_ab^e = - K1_e * (Ve / 10) * (1 + delta_ab)

    Args:
        conn (np.ndarray): Tetrahedron connectivity (E, 4).
        volume (np.ndarray): Element volumes (E,).
        K1_lookup (np.ndarray): Material anisotropy constants (G,).
        mat_id (np.ndarray): Material IDs per element (E,).

    Returns:
        sp.csr_matrix: The assembled anisotropy matrix of shape (N, N).
    """
    N = np.max(conn) + 1
    K1_elem = K1_lookup[mat_id - 1]
    val_elem = -K1_elem * volume / 10.0

    # Local 4x4 matrix: Ke_ab = val_elem * (1.0 + delta_ab)
    Ke = val_elem[:, None, None] * (np.ones((4, 4), dtype=np.float64) + np.eye(4, dtype=np.float64))

    rows = np.repeat(conn, 4, axis=1).flatten()
    cols = np.tile(conn, (1, 4)).flatten()
    data = Ke.flatten()

    Kan = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    Kan.sum_duplicates()
    return Kan



def assemble_exchange_anisotropy_matrix_cpu(
    conn: np.ndarray,
    volume: np.ndarray,
    grad_phi: np.ndarray,
    A_lookup: np.ndarray,
    K1_lookup: np.ndarray,
    k_easy_lookup: np.ndarray,
    mat_id: np.ndarray,
) -> sp.csr_matrix:
    """Assemble the combined Exchange and Anisotropy matrix in CSR format.
    The resulting matrix is of shape (3N, 3N) to handle cross-component anisotropy.
    """
    N = np.max(conn) + 1
    A_elem = A_lookup[mat_id - 1]
    K1_elem = K1_lookup[mat_id - 1]
    k_elem = k_easy_lookup[mat_id - 1]  # (E, 3)
    
    # Kex part (E, 4, 4)
    Kex_e = 2.0 * A_elem[:, None, None] * volume[:, None, None] * np.einsum("eai,ebi->eab", grad_phi, grad_phi)
    
    # Kan part (E, 4, 4)
    val_elem = -2.0 * K1_elem * volume / 20.0
    Kan_e = val_elem[:, None, None] * (np.ones((4, 4), dtype=np.float64) + np.eye(4, dtype=np.float64))
    
    # Kex_block: (E, 4, 4, 3, 3)
    I3 = np.eye(3, dtype=np.float64)
    Kex_block = Kex_e[:, :, :, None, None] * I3[None, None, None, :, :]
    
    # Kan_block: (E, 4, 4, 3, 3)
    kkT = np.einsum('eu,ev->euv', k_elem, k_elem)
    Kan_block = Kan_e[:, :, :, None, None] * kkT[:, None, None, :, :]
    
    K_block = Kex_block + Kan_block  # (E, 4, 4, 3, 3)
    
    # Global rows and cols
    row_nodes = np.repeat(conn, 4, axis=1).flatten()  # (E*16,)
    col_nodes = np.tile(conn, (1, 4)).flatten()       # (E*16,)
    
    # We expand to 3x3 components for each element in the 16 pairs
    r = 3 * row_nodes[:, None, None] + np.arange(3)[None, :, None]
    c = 3 * col_nodes[:, None, None] + np.arange(3)[None, None, :]
    rows, cols = np.broadcast_arrays(r, c)
    rows = rows.flatten()
    cols = cols.flatten()
    data = K_block.flatten()
    
    K_eff = sp.coo_matrix((data, (rows, cols)), shape=(3*N, 3*N)).tocsr()
    K_eff.sum_duplicates()
    return K_eff
