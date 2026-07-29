import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from amg_utils import DistributedCSR, csr_to_jax_CSR

def test():
    print(f"JAX devices: {jax.devices()}")
    
    # Create a random SciPy CSR matrix (16x16)
    np.random.seed(0)
    dense = np.random.rand(16, 16)
    dense[dense < 0.5] = 0  # Make it sparse
    scipy_mat = sp.csr_matrix(dense)
    
    # Create a random vector (16,)
    x = jnp.array(np.random.rand(16))
    
    # Expected result
    expected_y = scipy_mat @ np.array(x)
    
    # 1. Test Single-GPU path
    print("Testing single-device path...")
    dist_csr_single = csr_to_jax_CSR(scipy_mat)
    y_single = dist_csr_single @ x
    np.testing.assert_allclose(expected_y, y_single, atol=1e-5)
    print("Single-device path OK.")
    
    # 2. Test Multi-GPU path
    print("Testing multi-device path (Mesh)...")
    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape((4,)), ('devices',))
    with jax.set_mesh(mesh):
        dist_csr_multi = csr_to_jax_CSR(scipy_mat, device=mesh)
        
        # We need to distribute x the same way
        P = jax.sharding.PartitionSpec('devices')
        x_sharded = jax.device_put(x, jax.sharding.NamedSharding(mesh, P))
        
        y_multi = dist_csr_multi @ x_sharded
        np.testing.assert_allclose(expected_y, y_multi, atol=1e-5)
    
    print("Multi-device path OK.")

if __name__ == "__main__":
    test()
