#ifndef ENERGY_KERNELS_HPP
#define ENERGY_KERNELS_HPP

#include <vector>
#include <vexcl/vexcl.hpp>
#include "fem_utils.hpp"

/**
 * @brief Handles computation of micromagnetic energy and its gradient on GPU using VexCL.
 */
class EnergyKernels {
public:
    EnergyKernels(vex::Context& ctx, 
                 const SparseMatrixCSR& K_int,
                 const SparseMatrixCSR& G_div,
                 const SparseMatrixCSR& G_grad,
                 const std::vector<double>& Js_node_vols, 
                 double V_mag);

    double energy_and_grad(const vex::vector<double>& m_gpu,
                          const vex::vector<double>& U_gpu,
                          const Eigen::Vector3d& B_ext,
                          vex::vector<double>& g_gpu);

    double energy_only(const vex::vector<double>& m_gpu,
                      const vex::vector<double>& U_gpu,
                      const Eigen::Vector3d& B_ext);

    void compute_poisson_rhs(const vex::vector<double>& m_gpu, vex::vector<double>& b_gpu);

private:
    vex::Context& ctx;
    vex::sparse::matrix<double> mat_K_int;
    vex::sparse::matrix<double> mat_G_div;
    vex::sparse::matrix<double> mat_G_grad;
    
    vex::vector<double> Js_node_vols_gpu;
    vex::Reductor<double, vex::SUM> reduce_sum;
    
    double V_mag;

    double inv_Vmag;
};

#endif // ENERGY_KERNELS_HPP
