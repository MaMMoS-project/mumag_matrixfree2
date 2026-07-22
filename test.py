import os
os.environ["JAX_DISABLE_P2P"] = "1"
import jax
import jax.numpy as jnp
from src.poisson_solve import safe_device_put

def test():
    # test if safe_device_put works inside jit
    @jax.jit
    def f(x):
        return safe_device_put(x, jax.devices()[0])
    
    print(f(jnp.array(1.0)))

test()
