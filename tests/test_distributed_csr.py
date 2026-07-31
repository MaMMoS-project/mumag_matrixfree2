import os
import sys
import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

jax.config.update("jax_enable_x64", True)

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from amg_utils import DistributedCSR

def test_distributed_csr_single_device():
    # Construct a random SciPy CSR matrix
    np.random.seed(42)
    N = 100
    density = 0.1
    mat_coo = sp.random(N, N, density=density, format='coo', dtype=np.float64)
    scipy_mat = mat_coo.tocsr()

    x_np = np.random.randn(N)
    x_jax = jnp.asarray(x_np)

    # Reference CPU matrix-vector product
    y_ref = scipy_mat @ x_np

    # DistributedCSR single-device
    dist_csr = DistributedCSR.from_scipy(scipy_mat, mesh=None)
    y_dist = dist_csr @ x_jax

    np.testing.assert_allclose(np.array(y_dist), y_ref, rtol=1e-12, atol=1e-12)


def test_distributed_csr_multi_device_halo_exchange():
    devices = jax.devices()
    num_devices = len(devices)
    if num_devices < 2:
        # If fewer than 2 devices are available, test using CPU mesh if possible
        cpu_devices = jax.devices("cpu")
        if len(cpu_devices) >= 2:
            devices = cpu_devices[:2]
            num_devices = 2
        else:
            print("Skipping multi-device halo exchange test (requires >= 2 devices).")
            return

    mesh = Mesh(np.array(devices), ("devices",))

    # Construct a matrix with cross-domain (ghost) dependencies
    N = 100
    np.random.seed(123)
    mat_coo = sp.random(N, N, density=0.15, format='coo', dtype=np.float64)
    scipy_mat = mat_coo.tocsr()

    x_np = np.random.randn(N)
    y_ref = scipy_mat @ x_np

    # Build DistributedCSR with Mesh
    dist_csr = DistributedCSR.from_scipy(scipy_mat, mesh=mesh)

    # Shard vector x across mesh
    P = jax.sharding.PartitionSpec("devices")
    sharding = jax.sharding.NamedSharding(mesh, P)
    x_sharded = jax.device_put(x_np, sharding)

    # Execute matmul via Halo Exchange
    y_dist = dist_csr @ x_sharded

    np.testing.assert_allclose(np.array(y_dist), y_ref, rtol=1e-12, atol=1e-12)
