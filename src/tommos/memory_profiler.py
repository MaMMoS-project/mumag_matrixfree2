"""Estimate peak CPU memory usage for a simulation."""

from __future__ import annotations

from argparse import Namespace


def estimate_cpu_memory(Nn: int, Ne: int, args: Namespace) -> float:
    """Estimate the peak CPU memory usage in megabytes.

    Args:
        Nn: Number of nodes.
        Ne: Number of elements.
        args: Parsed command-line arguments.

    Returns:
        Estimated peak CPU memory usage in megabytes.
    """
    # 1. Geometry and Nodal Arrays
    # knt, conn, volume, JinvT, l_grad_phi, m0, M_nodal, V_mag_nodal, boundary_mask
    base_memory = 72 * Nn + 192 * Ne

    # 2. Sparse Matrices
    # Average NNZ for standard 3D tetrahedral mesh is ~15 non-zeros per row.
    # COO building: 16 entries per element -> data(float64) + rows(int32) + cols(int32)
    coo_buffer = 256 * Ne
    # CSR storage: data(float64) + indices(int32) + indptr(int32)
    csr_matrix_size = 184 * Nn
    num_matrices = 6  # A, Dx, Dy, Dz, Kan, Kex

    # Stage A: Global Assembly Peak
    # 5 assembled CSR matrices + 1 currently being assembled (COO + partial CSR)
    assembly_peak = base_memory + (5 * csr_matrix_size) + coo_buffer

    # Stage B: Preconditioner / Solver Setup Peak
    if args.cpp_mkl and args.poisson_solver == "pardiso":
        # CPU C++ MKL: All 6 matrices kept. Pardiso factorization takes ~20x A_scipy size.
        pardiso_factorization = 20 * csr_matrix_size
        setup_peak = base_memory + (num_matrices * csr_matrix_size) + pardiso_factorization
    elif not args.cpp_mkl and args.cpu_spmv_backend == "scipy":
        # CPU Scipy: All 6 matrices kept inside SparseOperators. PyAMG hierarchy built.
        pyamg_hierarchy = 2.5 * csr_matrix_size
        setup_peak = base_memory + (num_matrices * csr_matrix_size) + pyamg_hierarchy
    else:
        # GPU / Multi-GPU: Only A_scipy kept. PyAMG hierarchy built.
        pyamg_hierarchy = 2.5 * csr_matrix_size
        setup_peak = base_memory + csr_matrix_size + pyamg_hierarchy

    # Return the maximum of the two stages, converted to Megabytes
    peak_bytes = max(assembly_peak, setup_peak)
    return peak_bytes / (1024 * 1024)
