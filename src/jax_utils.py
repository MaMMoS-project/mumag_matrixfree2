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


def safe_device_put_fanout(x, target_devices, source_device=None):
    """Safely transfer data to multiple devices concurrently.
    
    If _DISABLE_P2P is True, moves the data to the CPU exactly once, 
    then fans it out to all target devices from the CPU. This prevents 
    PCIe bottlenecks caused by a single GPU sending the same vector 
    multiple times.
    """
    # 1. Deduplicate target devices (saves sending twice to same remote GPU)
    unique_targets = list(dict.fromkeys(target_devices))
    
    # 2. Strip source_device to guarantee zero-copy skipping
    remote_targets = [tgt for tgt in unique_targets if tgt != source_device]
    
    # 3. Execute CPU bounce strictly for remote targets
    if _DISABLE_P2P and len(remote_targets) > 0:
        cpu_dev = jax.devices("cpu")[0]
        x_cpu = jax.device_put(x, cpu_dev)
        put_dict = {tgt: jax.device_put(x_cpu, tgt) for tgt in remote_targets}
    else:
        put_dict = {tgt: jax.device_put(x, tgt) for tgt in remote_targets}
        
    # 4. Reconstruct the output tuple safely
    if source_device is not None:
        put_dict[source_device] = x
        
    return tuple(put_dict[tgt] for tgt in target_devices)


def safe_device_put_fanin_concat(arrays, source_devices, target_device):
    """Safely transfer multiple arrays to a single device and concatenate them.
    
    If _DISABLE_P2P is True, moves the arrays to the CPU first, concatenates 
    them on the CPU, and sends a single contiguous array to the target device. 
    This minimizes PCIe packet overhead and prevents traffic jams at the destination.
    """
    if _DISABLE_P2P:
        cpu_dev = jax.devices("cpu")[0]
        
        # Pass 1: Identify contiguous chunks on the SAME remote GPU.
        # Concatenate them remotely BEFORE hitting the PCIe bus.
        gpu_chunks = []
        curr_chunk, curr_src = [], None
        for a, src in zip(arrays, source_devices):
            if src == curr_src:
                curr_chunk.append(a)
            else:
                if curr_chunk: 
                    gpu_chunks.append((curr_chunk, curr_src))
                curr_src = src
                curr_chunk = [a]
        if curr_chunk: 
            gpu_chunks.append((curr_chunk, curr_src))
            
        concat_gpu_chunks = [
            (jnp.concatenate(chunk), src) if len(chunk) > 1 else (chunk[0], src)
            for chunk, src in gpu_chunks
        ]

        # Pass 2: Transfer remote chunks to CPU, group them, and send to target_device once.
        # Arrays already on target_device skip the CPU entirely (Zero-Copy).
        final_arrays, cpu_group = [], []
        for a, src in concat_gpu_chunks:
            if src == target_device:
                if cpu_group:
                    final_arrays.append(jax.device_put(jnp.concatenate(cpu_group), target_device))
                    cpu_group = []
                final_arrays.append(a)
            else:
                cpu_group.append(jax.device_put(a, cpu_dev))
                
        if cpu_group:
            final_arrays.append(jax.device_put(jnp.concatenate(cpu_group), target_device))
            
        # Final merge on the target device
        return jnp.concatenate(final_arrays)
    else:
        arrays_tgt = [a if src == target_device else jax.device_put(a, target_device) 
                      for a, src in zip(arrays, source_devices)]
        return jnp.concatenate(arrays_tgt)
