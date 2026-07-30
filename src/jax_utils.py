"""jax_utils.py.

Utility functions for JAX operations.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

def distribute_array(x: jnp.ndarray, mesh: jax.sharding.Mesh | None) -> jnp.ndarray:
    """Pad (if necessary) and distribute an array across a mesh.
    
    If mesh is None, returns the array natively.
    If the first dimension of x is not divisible by the number of devices in the mesh,
    it pads the array with zeros along the first dimension before distributing it.
    """
    if mesh is None:
        return jnp.asarray(x)
        
    num_devices = mesh.shape["devices"]
    n = x.shape[0]
    
    if n % num_devices != 0:
        pad_len = num_devices - (n % num_devices)
        pad_width = [(0, pad_len)] + [(0, 0)] * (x.ndim - 1)
        x = jnp.pad(x, pad_width)
        
    P = jax.sharding.PartitionSpec("devices")
    return jax.device_put(x, jax.sharding.NamedSharding(mesh, P))



