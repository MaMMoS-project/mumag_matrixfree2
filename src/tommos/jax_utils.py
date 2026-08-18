"""Utility functions for JAX operations."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def distribute_array(x: jnp.ndarray, mesh: jax.sharding.Mesh | None) -> jnp.ndarray:
    """Pad and distribute an array across a device mesh.

    Args:
        x: Array to distribute.
        mesh: Device mesh. If omitted, return the array without sharding.

    Returns:
        The input array, padded along its first dimension when necessary and
        distributed across the mesh.
    """
    if mesh is None:
        return jnp.asarray(x)

    num_devices = mesh.shape["devices"]
    n = x.shape[0]

    if n % num_devices != 0:
        pad_len = num_devices - (n % num_devices)
        pad_width = [(0, pad_len)] + [(0, 0)] * (x.ndim - 1)
        x = jnp.pad(x, pad_width)

    partition_spec = jax.sharding.PartitionSpec("devices")
    return jax.device_put(x, jax.sharding.NamedSharding(mesh, partition_spec))
