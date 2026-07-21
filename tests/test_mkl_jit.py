import jax

jax.config.update("jax_enable_x64", True)
import os
import sys

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

# Append src to path to import amg_utils
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from amg_utils import make_cpu_csr_op


def test_jit_mkl():
    N = 1000
    A = sp.random(N, N, density=0.01, format="csr", dtype=np.float64)
    x = np.random.rand(N).astype(np.float64)

    cpu_op = make_cpu_csr_op(A)

    @jax.jit
    def test_func(x_jnp):
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


if __name__ == "__main__":
    test_jit_mkl()
