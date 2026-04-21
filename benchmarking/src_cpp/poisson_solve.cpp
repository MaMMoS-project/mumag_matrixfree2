#include "poisson_solve.hpp"

PoissonSolver::PoissonSolver(vex::Context& ctx, const SparseMatrixCSR& L, const std::vector<double>& mask)
    : ctx(ctx), mask_cpu(mask) {

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
    prm.solver.tol = 1e-10; // Match Python benchmark tolerance
    prm.solver.maxiter = 2000;

    // Build the solver
    amgcl::backend::vexcl<double>::params bprm;
    bprm.q = ctx.queue();

    solver = std::make_unique<Solver>(
        std::tie(L_masked.rows, L_masked.ptr, L_masked.indices, L_masked.data),
        prm, 
        bprm
    );
}

std::pair<int, double> PoissonSolver::solve(const vex::vector<double>& b_gpu, vex::vector<double>& U_gpu) {
    ctx.finish();
    auto start = std::chrono::high_resolution_clock::now();

    int iters;
    double error;
    std::tie(iters, error) = (*solver)(b_gpu, U_gpu);

    ctx.finish();
    auto end = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double>(end - start).count();
    return {iters, duration};
}

