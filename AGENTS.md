# MaMMoS-MuMag Project Context

This file provides persistent context for the Gemini CLI to ensure engineering consistency and technical accuracy within this repository.

## 1. Project Mission
A high-performance, matrix-free micromagnetics library built on **JAX**. It solves the demagnetization field using an element-wise FEM Poisson solver and minimizes energy using a curvilinear Barzilai-Borwein (BB) method.

## 2. Engineering Standards

### Environment & Tooling
- **Dependency Management**: Use **Pixi**. Never use `pip` or `conda` directly; always work through `pixi.toml` and defined tasks.
- **Hardware Targets**: Support both `cpu` and `cuda` environments. Prioritize GPU execution for large-scale simulations.

### Coding Style & Types
- **Documentation**: All functions must use **Google-style docstrings** without type hints in the docstrings. Each parameter must be documented separately.
- **Type Safety**: Use explicit **Python type hints** for all parameters and return types. Use `jnp.ndarray` (or `Array` alias) for JAX arrays and `np.ndarray` for CPU/IO data.
- **Floating Point**: Micromagnetic physical verification REQUIRES double precision. Always ensure `jax.config.update("jax_enable_x64", True)` is set in scripts and tests.
- **Static analysis**: All code must comply with ruff and pre-commit hooks. Adjusting ruff/pre-commit settings is not permitted.

### JAX Implementation Patterns
- **Matrix-Free**: NEVER assemble a global stiffness matrix. Operations must be computed element-wise.
- **XLA Fusion**: Avoid high-level abstractions like `jnp.einsum` for small-tensor contractions (e.g., $4 \times 3$ element gradients). Manually unroll these into explicit scalar-vector arithmetic to enable XLA kernel fusion and register utilization.
- **Pre-scaling**: Pre-calculate all loop-invariant weighted geometry terms (e.g., $A_{red} \cdot V_e$, $K_1 \cdot V_e$) outside the inner JAX loops to reduce FLOP counts.
- **Batching (Chunking)**: Large meshes must be processed in chunks to manage GPU memory. While `chunk_elems = 100,000` is a documented baseline for mid-range hardware (RTX 4060), this parameter must be **tuned to fit the GPU's L2 cache**. The goal is to maximize cache hit rates and minimize VRAM round-trips during atomic `scatter-add` assembly operations.
- **JIT & Pytrees**: Use `jax.jit` extensively. Complex states should be managed as registered `jax.tree_util` classes (see `MinimState` in `src/curvilinear_bb_minimizer.py`).

## 3. Physical & Numerical Methodology

### Minimization (Algorithm 2)
- **Algorithm**: Curvilinear Search for p-Harmonic flows (DOI: 10.1137/080726926).
- **Unit Length**: Maintain $|m|=1$ via the **Cayley Transform**, avoiding simple normalization.
- **Stopping Criteria**: Use the **Gill-Murray (1981)** U1-U4 criteria for convergence.

### Poisson Solver
- **Implementation**: Preconditioned Conjugate Gradient (PCG).
- **Stability**: Project RHS and initial guess to **zero-mean** for pure Neumann (open boundary) problems to ensure solvability.
- **Preconditioning**: Default to `amgcl`.

### Validation & Reference
- **C++ Alignment**: The JAX implementation is the performance benchmark for the native C++ (OpenCL/VexCL) port. Maintain strict mathematical alignment in iteration counts and physical scaling between JAX kernels and the C++ CSR-based reference.

## 4. Repository Structure
- `src/`: Core library modules (documented and typed).
- `samples/`: User-facing simulation examples and `.p2` config samples.
- `benchmarking/`: Native C++ (OpenCL) vs JAX performance comparisons.
- `tests/`: Physics and gradient verification suite.
- `develop/`: Magnetoelasticity and ongoing feature development.
