# Micromagnetics JAX-to-C++ Porting Progress

**Date:** March 9, 2026
**Status:** In Progress (Steps 4.1 & 4.2 Completed)

---

## 1. Project Goal
Port the JAX-based micromagnetic simulator to C++ using **VexCL** for GPU acceleration and **AMGCL** for the Poisson solver. The C++ version uses a pre-assembled sparse matrix approach instead of matrix-free kernels.

## 2. Completed Steps

### 4.1: Core Physics & Assembly
- **`src_cpp/fem_utils.hpp/cpp`**: Implements CPU-side FEM assembly for:
    - Scalar Stiffness Matrix ($L$) for Poisson.
    - $3N \times 3N$ Internal Field Matrix ($K_{int}$) combining Exchange and Uniaxial Anisotropy.
    - Divergence ($G_{div}$) and Gradient ($G_{grad}$) matrices for Demag field coupling.
- **`src_cpp/poisson_solve.hpp/cpp`**: Wraps AMGCL with a VexCL backend to solve $\nabla^2 U = \rho$.
- **`src_cpp/energy_kernels.hpp/cpp`**: GPU-side Energy and Effective Field calculations using VexCL `spmat` and vector kernels.

### 4.2: Poisson Convergence Test
- **`src_cpp/test_poisson_convergence.cpp`**: A benchmark tool that loads a `.npz` mesh, assembles matrices, and solves the Poisson equation on the GPU.

---

## 3. Current Architecture Decisions
- **Anisotropy:** Assembled into the $3N \times 3N$ internal matrix as node-wise $3 \times 3$ block contributions: $-2 K_1 V_i (\mathbf{k} \mathbf{k}^T)$.
- **Exchange:** Part of the $3N \times 3N$ internal matrix, based on the stiffness matrix $L$ weighted by $2A$.
- **Demag:** Solved using the potential $U$ via the Poisson equation. Coupling is handled by $G_{div}$ (source) and $G_{grad}$ (field).
- **GPU Backend:** OpenCL/VexCL.

---

## 4. Next Steps
1. **Verify Poisson Solve:** Run `./test_poisson_convergence` on the new GPU machine.
2. **Step 4.3: `test_energy.cpp`**: Create a C++ version of `test_energy.py` to validate $E_{ex}, E_{ani}, E_{zee}, E_{demag}$ against analytic results.
3. **Step 4.4: `test_minimizer_relaxation.cpp`**: Port the Barzilai-Borwein minimizer.

---

## 5. Instructions for the New Machine
1. Follow `install_cpp.md` to set up dependencies.
2. Build the project: `mkdir build && cd build && cmake .. && make`.
3. Generate a mesh using the Python scripts and run the test.
