#include "poisson_solve.hpp"

PoissonSolver::PoissonSolver(vex::Context& ctx, const SparseMatrixCSR& L, const std::vector<double>& mask)
    : mask_cpu(mask) {
    
    // Create a copy of L that enforces Dirichlet boundary conditions
    SparseMatrixCSR L_masked = L;
    
    for (int i = 0; i < L_masked.rows; ++i) {
        if (mask[i] == 0.0) { // Boundary node
            // Clear the row
            int start = L_masked.ptr[i];
            int end = L_masked.ptr[i+1];
            for (int j = start; j < end; ++j) {
                if (L_masked.indices[j] == i) {
                    L_masked.data[j] = 1.0;
                } else {
                    L_masked.data[j] = 0.0;
                }
            }
        }
    }

    // Parameters for AMGCL solver
    typename Solver::params prm;
    prm.solver.tol = 1e-8;
    prm.solver.maxiter = 500;
    
    // Adapt L_masked for VexCL/AMGCL
    auto A_gpu = amgcl::adapter::vexcl_sparse(
        ctx, L_masked.rows, L_masked.cols, 
        L_masked.ptr.data(), L_masked.indices.data(), L_masked.data.data()
    );

    // Build the solver
    solver = std::make_unique<Solver>(A_gpu, prm, amgcl::backend::vexcl_params(ctx));
}

void PoissonSolver::solve(const vex::vector<double>& b_gpu, vex::vector<double>& U_gpu) {
    // Before solve, make sure b_gpu satisfies b_i = 0 for boundary nodes.
    // This can be done with a VexCL element-wise product if we transfer the mask.
    // However, the caller should usually handle the RHS masking.
    
    (*solver)(b_gpu, U_gpu);
}
