# Arrays and Memory Placement in Matrix-Free MuMag

This document summarizes the primary arrays used during the multi-GPU minimization loops, mapping out their lifecycle, creation, and physical memory residency in the refactored dynamic `sparse_ops` architecture.

## Array Lifecycle and Memory Placement

| Array Name | Where it is created | Where it lives | Where it is used |
| :--- | :--- | :--- | :--- |
| **`m0`**, **`B_ext_vec`**, **`U_init`** | Setup scripts (e.g., `loop.py` / `hysteresis_loop.py`) as the initial state of the simulation. | **CPU Host Memory** (Previously pinned to GPU 0 in the `optimize` branch). | Used as the starting seed vectors for the minimization `while` loop in `solve_and_minimize_multigpu`. |
| **`m_local`** & **Trial steps** | Calculated on the fly inside the `while` loop in `minimizers.py`. | **CPU Host Memory** | Manually copied from CPU ➔ Remote GPUs (1, 2, 3) on every step to evaluate the sparse operators. |
| **`M_nodal`** | The initial geometry/mesh setup phase. | **CPU Host Memory** (Passed dynamically via `sparse_ops`). | Used in `energy_and_grad_multigpu` to compute the Zeeman energy (`g_z`). |
| **`d_diag`** | `hysteresis_loop.py` (by calling `compute_exchange_diagonal`). | **CPU Host Memory** | Used exclusively to compute the preconditioner mass matrix `inv_M_prec`. |
| **`inv_M_prec`** | `hysteresis_loop.py` (derived directly from `d_diag`). | **CPU Host Memory** (Passed dynamically via `sparse_ops`). | Used inside the inner preconditioner solvers (like `make_preconditioner_op`) to scale vectors during PCG steps. |
| **`inv_M_rel`**, **`M_rel`** | Initialization/Setup phase. | **CPU Host Memory** (Passed dynamically via `sparse_ops`). | Used heavily across all minimizer state updates to project gradients into the tangent space (`tangent_grad`). |
| **`A_diag`** | `loop.py` / `hysteresis_loop.py` (extracted from the assembled Poisson stiffness matrix `A_scipy`). | **CPU Host Memory** (Passed dynamically via `sparse_ops`). | Used inside `poisson_solve.py` (`assemble_diag`) to build AMG preconditioners for the scalar potential solver. |
| **`boundary_mask`** | Initialization/Setup phase (based on Dirichlet boundary conditions). | **CPU Host Memory** (Passed dynamically via `sparse_ops`). | Used inside `poisson_solve.py` to mask boundary nodes during the matrix-vector product `A @ U`. |
| **`B_bias`** | User input / initialization (Optional). | **CPU Host Memory** (Passed dynamically via `sparse_ops`). | Used in `energy_and_grad_multigpu` to offset the external field. |
| **`gx_gpu`**, **`gy_gpu`**, **`gz_gpu`**, **`g_dem_flat_gpu`** | Computed dynamically by the remote GPUs (GPUs 1, 2, 3) during the sparse matrix multiplications (`Kx`, `Ky`, `Kz`, `G`). | **Remote GPUs ➔ GPU 0** (Explicitly pinned back to `master_device` upon return). | Gathered and summed together on GPU 0 to form the total energy and raw gradient (`g_total`). |

## Historical Context: The Move to Dynamic Arrays

Before the multi-GPU optimizations, many of the arrays in the middle of the table (`inv_M_prec`, `inv_M_rel`, `M_rel`, `boundary_mask`) were **statically baked into the GPU execution graphs by the XLA compiler as closures**. 

While this kept data local to the GPUs, baking massive arrays into closures caused severe compilation overhead and high memory consumption. To solve this, these arrays were moved into the `sparse_ops` dictionary, converting them from static constants into **dynamic CPU runtime arguments**. 

Because JAX treats unpinned dynamic arguments as residing on the CPU, this shift forced JAX to silently execute Host-to-Device (CPU ➔ GPU) memory transfers every single time a step or matrix-vector product was evaluated in the `while` loop. Subsequent attempts to optimize this by explicitly pinning `m0` and `M_nodal` to GPU 0 successfully eliminated the CPU transfers but exposed a physical Peer-to-Peer (P2P) hardware deadlock on server nodes lacking NVLink (e.g., L40S nodes with PCIe ACS enabled), as remote GPUs were suddenly forced to fetch the arrays directly from GPU 0's memory over restricted PCIe routes. 

Reverting the GPU 0 pinning (or using `JAX_DISABLE_P2P=1`) restores the safe Host-to-Device staging behavior.
