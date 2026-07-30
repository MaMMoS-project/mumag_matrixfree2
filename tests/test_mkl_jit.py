# ruff: noqa: E402
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="MKL tests are only supported on Linux")

from tommos._native_loader import probe_sparse_dot_mkl
from tommos.amg_utils import make_cpu_csr_op


def test_jit_mkl() -> None:
    """The probed wrapper must remain numerically correct through JAX JIT."""
    probe = probe_sparse_dot_mkl()
    if not probe.available:
        pytest.skip(f"sparse-dot-mkl unavailable: {probe.error!r}")
    N = 1000
    rng = np.random.default_rng(42)
    A = sp.random(N, N, density=0.01, format="csr", dtype=np.float64, random_state=rng)
    x = rng.random(N, dtype=np.float64)

    cpu_op = make_cpu_csr_op(A, cpu_spmv_backend="persistent_mkl")

    @jax.jit
    def test_func(x_jnp: jnp.ndarray) -> jnp.ndarray:
        """Apply the callback-backed operator inside JAX JIT.

        Args:
            x_jnp: Dense JAX input vector.

        Returns:
            Dense JAX result vector.
        """
        return cpu_op(x_jnp)

    print("Compiling and running JIT...")
    x_jnp = jnp.array(x)
    y_jax = test_func(x_jnp)

    # Block until ready
    y_jax.block_until_ready()
    print("JIT execution successful!")

    y_scipy = A @ x
    diff = np.linalg.norm(np.array(y_jax) - y_scipy)
    print(f"Diff: {diff}")
    assert diff < 1e-10
    cpu_op.close()
