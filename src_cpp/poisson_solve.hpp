#ifndef POISSON_SOLVE_HPP
#define POISSON_SOLVE_HPP

#include <memory>
#include <vexcl/vexcl.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/adapter/vexcl.hpp>

#include "fem_utils.hpp"

/**
 * @brief Wrapper for the Poisson solver using AMGCL and VexCL.
 */
class PoissonSolver {
public:
    using Backend = amgcl::backend::vexcl<double>;
    using Solver = amgcl::make_solver<
        amgcl::amg<
            Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
        >,
        amgcl::solver::cg<Backend>
    >;

    PoissonSolver(vex::Context& ctx, const SparseMatrixCSR& L, const std::vector<double>& mask);

    /**
     * @brief Solves L * U = b.
     * @param b_gpu RHS vector on GPU.
     * @param U_gpu Initial guess/Output vector on GPU.
     */
    void solve(const vex::vector<double>& b_gpu, vex::vector<double>& U_gpu);

private:
    std::unique_ptr<Solver> solver;
    std::vector<double> mask_cpu;
};

#endif // POISSON_SOLVE_HPP
