# ruff: noqa: E402
import sys

import numpy as np
import pytest
import scipy.sparse as sp

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="MKL tests are only supported on Linux")

from tommos._native_loader import probe_sparse_dot_mkl
from tommos.amg_utils import PersistentMKLOperator


def test_persistent_wrapper_matches_scipy() -> None:
    """The probed sparse wrapper must match a deterministic SciPy product."""
    probe = probe_sparse_dot_mkl()
    if not probe.available:
        pytest.skip(f"sparse-dot-mkl unavailable: {probe.error!r}")
    N = 1000
    rng = np.random.default_rng(42)
    A = sp.random(N, N, density=0.01, format="csr", dtype=np.float64, random_state=rng)
    x = rng.random(N, dtype=np.float64)
    operator = PersistentMKLOperator(A)

    output_arr = operator.apply(x)
    y_scipy = A @ x
    operator.close()

    np.testing.assert_allclose(output_arr, y_scipy, rtol=1e-12, atol=1e-12)
