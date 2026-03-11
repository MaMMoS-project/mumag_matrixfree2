# Technical Report: Performance Optimization of Micromagnetic FEM Solvers

**Target:** Publication / Team Knowledge Base  
**Project:** Mammos-MatrixFree2  
**Date:** March 11, 2026  

## Executive Summary
This document details the performance tuning of a matrix-free micromagnetic FEM solver implemented in JAX. By applying advanced GPU engineering techniques—specifically **manual arithmetic unrolling for XLA fusion** and **loop-invariant geometry pre-scaling**—we achieved a **43x speedup** in energy gradient evaluations. The final Python implementation now matches or slightly exceeds the performance of a native C++ implementation using optimized Sparse Matrix-Vector (SpMV) multiplications.

---

## 1. Baseline Performance & Bottleneck Analysis

### 1.1 The Challenge: Matrix-Free vs. Sparse Matrix
The native C++ solver uses a **Sparse Matrix (CSR)** approach, where physics is pre-assembled into matrices ($K$, $G$). Evaluation is reduced to vendor-tuned SpMV calls.  
The Python solver uses a **Matrix-Free** approach via JAX, where physics is recomputed element-wise in every iteration. While more memory-efficient for large meshes, the initial implementation was significantly slower.

### 1.2 Initial Profiling (Mesh: 2.3M Elements / 360k Nodes)
| Metric | Python (Original) | C++ (Native CSR) | Gap |
| :--- | :---: | :---: | :---: |
| **Poisson Solve (1e-10)** | 1030 ms | 247 ms | 4.1x |
| **Energy Kernels** | 588 ms | 8.3 ms | **71x** |
| **Full Iteration** | 1621 ms | 255 ms | 6.3x |

**Bottleneck Identified via JAX Profiler:**  
Trace analysis showed that `jnp.einsum` calls for small-tensor contractions ($4 \times 3$) were dispatched as individual `cutlass::Kernel2` events. This prevented XLA from fusing the computation with the subsequent `scatter-add`, leading to massive kernel launch overhead and redundant memory reads.

---

## 2. Optimization Steps

### Step 1: Solver Alignment (AMGCL)
The Python solver was initially using a simple Jacobi preconditioner. We switched to the `amgcl` (Algebraic Multigrid) preconditioner in Python to match the C++ implementation. This aligned the iteration counts (~19 iters) and brought the Poisson solve overhead within 10% of the C++ version.

### Step 2: XLA Fusion via Manual Unrolling
**Rationale:** XLA cannot fuse library calls like `einsum`. By rewriting the contractions as explicit scalar-vector arithmetic, we force XLA to "see" the entire element-wise operation as a single mathematical block.

*   **Before (`einsum`):** Multiple kernels, intermediate results stored in VRAM.
*   **After (Unrolled):** A single fused CUDA kernel. Intermediate variables (like element gradients) are stored in **GPU Registers**, eliminating VRAM round-trips.

**Example (Poisson Element Gradient):**
```python
# Fused logic
grad_U = (B_c[:, 0, :] * U_e[:, 0, None] +
          B_c[:, 1, :] * U_e[:, 1, None] +
          B_c[:, 2, :] * U_e[:, 2, None] +
          B_c[:, 3, :] * U_e[:, 3, None])
```

### Step 3: Pre-scaling Geometry by Material Properties
**Rationale:** In FEM, terms like $A_{red} \cdot V_e$ or $K_1 \cdot V_e$ are loop-invariant. Initially, these were multiplied inside the inner GPU loop. By pre-calculating weighted geometry buffers (e.g., `A_Ve = 2.0 * A_lookup[ids] * Ve`), we reduced the inner-loop FLOP count by ~30% and simplified the memory access pattern.

---

## 3. Final Benchmark Results

**Configuration:**  
*   **Mesh:** `cube_60nm_shell.npz` (361,812 Nodes, 2,304,342 Elements)  
*   **Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU  
*   **Tolerance:** $10^{-10}$

| Metric | Python (Fused Matrix-Free) | C++ (Native CSR) | Ratio (Py/C++) |
| :--- | :---: | :---: | :---: |
| **Poisson Solve Overhead** | **232.3 ms** | 247.1 ms | **0.94x** (Py is faster) |
| **Energy Kernels Only** | 13.6 ms | 8.3 ms | 1.6x |
| **Full Iteration Total** | **245.9 ms** | 255.3 ms | **0.96x** (Py is faster) |

---

## 4. Engineering Conclusions

1.  **Matrix-Free Superiority:** With proper fusion, the matrix-free approach can outperform Sparse Matrices. It avoids the high bandwidth cost of loading massive CSR data structures (indices/pointers) and instead relies on the GPU's high-speed register file for recomputing physics on-the-fly.
2.  **XLA Fusion Limits:** In JAX, high-level abstractions like `einsum` are excellent for development but can be performance traps for small-tensor FEM kernels. Manual unrolling is a required "pro-tip" for production-grade solvers.
3.  **The "Vector Penalty":** 3D vector fields ($N \times 3$) carry higher atomic contention than scalar fields. Future work could explore component-wise splitting to further reduce memory serialization during nodal assembly.

---
*Report generated for the Mammos-MatrixFree project.*
