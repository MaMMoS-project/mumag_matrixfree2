#include "energy_kernels.hpp"
#include <iostream>

EnergyKernels::EnergyKernels(vex::Context& ctx, 
                           const SparseMatrixCSR& K_int,
                           const SparseMatrixCSR& G_div,
                           const SparseMatrixCSR& G_grad,
                           const std::vector<double>& Js_node_vols,
                           double V_mag)
    : ctx(ctx), 
      mat_K_int(ctx, K_int.rows, K_int.cols, K_int.ptr, K_int.indices, K_int.data),
      mat_G_div(ctx, G_div.rows, G_div.cols, G_div.ptr, G_div.indices, G_div.data),
      mat_G_grad(ctx, G_grad.rows, G_grad.cols, G_grad.ptr, G_grad.indices, G_grad.data),
      Js_node_vols_gpu(ctx, Js_node_vols),
      reduce_sum(ctx),
      V_mag(V_mag),
      inv_Vmag(1.0 / (V_mag + 1e-30)) {}

void EnergyKernels::compute_poisson_rhs(const vex::vector<double>& m_gpu, vex::vector<double>& b_gpu) {
    b_gpu = mat_G_div * m_gpu;
}

// Function to get B component based on index
VEX_FUNCTION(double, get_b_comp, (int, i)(double, bx)(double, by)(double, bz),
    int c = i % 3;
    if (c == 0) return bx;
    if (c == 1) return by;
    return bz;
);

double EnergyKernels::energy_and_grad(const vex::vector<double>& m_gpu,
                                    const vex::vector<double>& U_gpu,
                                    const Eigen::Vector3d& B_ext,
                                    vex::vector<double>& g_gpu) {
    
    // Internal + Demag Effective Fields
    g_gpu = mat_K_int * m_gpu + 2.0 * (mat_G_grad * U_gpu);
    
    // Zeeman Field Contribution (Factor -2.0 from Python)
    // Js_v_gpu stretched to 3N: repeats V_i three times for x,y,z components
    auto js_v_stretched = vex::permutation(vex::element_index() / 3)(Js_node_vols_gpu);
    auto b_ext_stretched = get_b_comp(vex::element_index(), B_ext.x(), B_ext.y(), B_ext.z());
    
    g_gpu = g_gpu - 2.0 * js_v_stretched * b_ext_stretched;

    // Energy calculations (dimensionless)
    double E_int = 0.5 * reduce_sum(m_gpu * (mat_K_int * m_gpu));
    double E_demag = reduce_sum(m_gpu * (mat_G_grad * U_gpu));
    double E_zee = -2.0 * reduce_sum(m_gpu * js_v_stretched * b_ext_stretched);

    // Scale Energy & Gradient by 1/Vmag to match Python
    double E_total = (E_int + E_demag + E_zee) * inv_Vmag;
    g_gpu = g_gpu * inv_Vmag;
    
    return E_total;
}

double EnergyKernels::energy_only(const vex::vector<double>& m_gpu,
                                const vex::vector<double>& U_gpu,
                                const Eigen::Vector3d& B_ext) {
    
    auto js_v_stretched = vex::permutation(vex::element_index() / 3)(Js_node_vols_gpu);
    auto b_ext_stretched = get_b_comp(vex::element_index(), B_ext.x(), B_ext.y(), B_ext.z());

    double E_int = 0.5 * reduce_sum(m_gpu * (mat_K_int * m_gpu));
    double E_demag = reduce_sum(m_gpu * (mat_G_grad * U_gpu));
    double E_zee = -2.0 * reduce_sum(m_gpu * js_v_stretched * b_ext_stretched);

    return (E_int + E_demag + E_zee) * inv_Vmag;
}
