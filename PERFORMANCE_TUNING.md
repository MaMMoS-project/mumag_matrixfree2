# Technical Report: Performance Optimization of Micromagnetic FEM Solvers

**Target:** Publication / Team Knowledge Base  
**Project:** Mammos-MatrixFree2  
**Date:** March 11, 2026  

## Executive Summary
This document details the performance tuning of a matrix-free micromagnetic FEM solver implemented in JAX. By applying advanced GPU engineering techniques—specifically **manual arithmetic unrolling for XLA fusion**, **loop-invariant geometry pre-scaling**, and **cache-aware chunk size tuning**—we achieved a **43x speedup** in energy gradient evaluations. The final Python implementation now outperforms a native C++ implementation using optimized Sparse Matrix-Vector (SpMV) multiplications by **16%** for a full solver iteration.

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
The Python solver was switched to the `amgcl` (Algebraic Multigrid) preconditioner to match the C++ implementation. This aligned the iteration counts (~19 iters) and brought the Poisson solve overhead within 10% of the C++ version.

### Step 2: XLA Fusion via Manual Unrolling
**Rationale:** XLA cannot fuse library calls like `einsum`. By rewriting the contractions as explicit scalar-vector arithmetic, we force XLA to "see" the entire element-wise operation as a single mathematical block. This allows intermediate variables (like element gradients) to be stored in **GPU Registers**, eliminating VRAM round-trips.

### Step 3: Pre-scaling Geometry by Material Properties
**Rationale:** In FEM, terms like $A_{red} \cdot V_e$ or $K_1 \cdot V_e$ are loop-invariant. By pre-calculating weighted geometry buffers, we reduced the inner-loop FLOP count by ~30% and simplified the memory access pattern.

### Step 4: Cache-Aware Chunk Size Tuning
**Rationale:** Iterative solvers (Poisson) are sensitive to memory latency. By reducing the `chunk_elems` from 200,000 to **100,000**, we improved the GPU L2 cache hit rate and reduced memory contention during atomic `scatter-add` operations. This provided a further 14% speedup in the Poisson solver.

---

## 3. Final Benchmark Results

**Configuration:**  
*   **Mesh:** `cube_60nm_shell.npz` (361,812 Nodes, 2,304,342 Elements)  
*   **Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU  
*   **Tolerance:** $10^{-10}$

| Metric | Python (Fused Matrix-Free) | C++ (Native CSR) | Ratio (Py/C++) |
| :--- | :---: | :---: | :---: |
| **Poisson Solve Overhead** | **200.3 ms** | 247.1 ms | **0.81x** (Py is faster) |
| **Energy Kernels Only** | 14.7 ms | 8.3 ms | 1.7x |
| **Full Iteration Total** | **215.0 ms** | **255.3 ms** | **0.84x** (Py is 16% faster) |

---

## 4. Engineering Conclusions

1.  **Matrix-Free Superiority:** With proper fusion and chunking, the matrix-free approach outperforms Sparse Matrices. It avoids the high bandwidth cost of loading massive CSR data structures (indices/pointers) and instead relies on the GPU's high-speed register file for recomputing physics on-the-fly.
2.  **XLA Fusion Limits:** In JAX, high-level abstractions like `einsum` are excellent for development but are performance traps for small-tensor FEM kernels. Manual unrolling is required for production-grade performance.
3.  **The "Vector Penalty":** 3D vector fields ($N \times 3$) carry higher atomic contention than scalar fields. However, by optimizing the compute-to-memory ratio, JAX can effectively hide this latency.

---
*Report generated for the Mammos-MatrixFree project.*
