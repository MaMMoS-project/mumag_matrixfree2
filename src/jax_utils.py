"""jax_utils.py.

Utility functions for JAX operations.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

_DISABLE_P2P = os.environ.get("JAX_DISABLE_P2P", "0").strip() == "1"


def safe_device_put(x, target_device):
    """Safely transfer data to a device.
    
    If _DISABLE_P2P is True, routes through the CPU to bypass broken PCIe switches.
    Otherwise, uses native jax.device_put for optimal NVLink/PCIe P2P performance.
    """
    if hasattr(x, "devices") and target_device in x.devices():
        return x

    if _DISABLE_P2P:
        cpu_dev = jax.devices("cpu")[0]
        x_cpu = jax.device_put(x, cpu_dev)
        return jax.device_put(x_cpu, target_device)

    return jax.device_put(x, target_device)


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



