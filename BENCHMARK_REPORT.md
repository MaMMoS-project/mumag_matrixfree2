# Poisson Solver Convergence Benchmark Report

## Hardware Information
- **OS**: Linux 6.8.0-101-generic
- **CPU**: 13th Gen Intel(R) Core(TM) i5-13500HX
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU

## Mesh Information
- **Nodes**: 361,812
- **Elements**: 2,304,342

## Performance Comparison (Tolerance 1e-10)

| Implementation | Iterations | Time (s) | Rel. Residual |
| :--- | :---: | :---: | :---: |
| Python (None) | 1446 | 5.037 | 9.60e-11 |
| Python (Jacobi) | 494 | 1.727 | 9.55e-11 |
| Python (Chebyshev) | 220 | 2.304 | 9.08e-11 |
| Python (Amg) | 22 | 0.341 | 6.25e-11 |
| Python (Amgcl) | 19 | 0.299 | 6.63e-11 |
| **C++ (Native)** | **22** | **0.292** | **4.75e-11** |

---
*Generated automatically by benchmark script.*
