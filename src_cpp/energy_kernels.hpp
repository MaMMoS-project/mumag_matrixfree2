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
                 const std::vector<double>& Js_node_vols, // Node-wise lumped (Js * Vi)
                 double Kd_ref,
                 double V_mag);

    /**
     * @brief Computes total energy and the negative effective field (gradient).
     * @param m_gpu Current magnetization vector (3N).
     * @param U_gpu Current scalar potential (N).
     * @param B_ext External field (3).
     * @param g_gpu Output gradient vector (3N).
     * @return double Total Energy (normalized by Kd * Vmag).
     */
    double energy_and_grad(const vex::vector<double>& m_gpu,
                          const vex::vector<double>& U_gpu,
                          const Eigen::Vector3d& B_ext,
                          vex::vector<double>& g_gpu);

    /**
     * @brief Compute only the energy.
     */
    double energy_only(const vex::vector<double>& m_gpu,
                      const vex::vector<double>& U_gpu,
                      const Eigen::Vector3d& B_ext);

    /**
     * @brief Helper to compute the Poisson RHS: b = G_div * m.
     */
    void compute_poisson_rhs(const vex::vector<double>& m_gpu, vex::vector<double>& b_gpu);

private:
    vex::Context& ctx;
    vex::sparse::matrix<double> mat_K_int;
    vex::sparse::matrix<double> mat_G_div;
    vex::sparse::matrix<double> mat_G_grad;
    
    vex::vector<double> Js_node_vols_gpu;
    
    double Kd_ref;
    double V_mag;
    double inv_Kd_Vmag;
};

#endif // ENERGY_KERNELS_HPP
