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



