import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from amg_utils import make_sparse_operator, HAS_MKL_FFI

jax.config.update("jax_enable_x64", True)


def test_mkl_ffi_spmv():
    # Verify FFI module compiled and imported successfully
    assert HAS_MKL_FFI, "mkl_ffi_lib failed to import or register"

    # Create a small random sparse CSR matrix
    N = 1000
    density = 0.05
    np.random.seed(42)
    scipy_mat = sp.random(N, N, density=density, format="csr", dtype=np.float64)

    # Input vector
    x = np.random.rand(N)
    x_jax = jnp.asarray(x)

    # 1. Compute SpMV using persistent_mkl (baseline callback)
    op_mkl = make_sparse_operator(scipy_mat, cpu_spmv_backend="persistent_mkl")
    y_mkl_jax = op_mkl @ x_jax
    y_mkl = np.array(y_mkl_jax)

    # 2. Compute SpMV using mkl_ffi (zero-copy FFI custom call)
    op_ffi = make_sparse_operator(scipy_mat, cpu_spmv_backend="mkl_ffi")
    y_ffi_jax = op_ffi @ x_jax
    y_ffi = np.array(y_ffi_jax)

    # Verify they produce numerically identical outputs
    np.testing.assert_allclose(y_ffi, y_mkl, rtol=1e-12, atol=1e-12)

    # Test within a JIT-compiled function
    @jax.jit
    def jit_spmv(x_val):
        return op_ffi @ x_val

    y_jit_jax = jit_spmv(x_jax)
    y_jit = np.array(y_jit_jax)
    np.testing.assert_allclose(y_jit, y_mkl, rtol=1e-12, atol=1e-12)
