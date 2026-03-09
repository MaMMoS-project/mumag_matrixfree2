#include "energy_kernels.hpp"
#include <iostream>

EnergyKernels::EnergyKernels(vex::Context& ctx, 
                           const SparseMatrixCSR& K_int,
                           const SparseMatrixCSR& G_div,
                           const SparseMatrixCSR& G_grad,
                           const std::vector<double>& Js_node_vols,
                           double Kd_ref,
                           double V_mag)
    : ctx(ctx), 
      mat_K_int(ctx, K_int.rows, K_int.cols, K_int.ptr.data(), K_int.indices.data(), K_int.data.data()),
      mat_G_div(ctx, G_div.rows, G_div.cols, G_div.ptr.data(), G_div.indices.data(), G_div.data.data()),
      mat_G_grad(ctx, G_grad.rows, G_grad.cols, G_grad.ptr.data(), G_grad.indices.data(), G_grad.data.data()),
      Js_node_vols_gpu(ctx, Js_node_vols),
      Kd_ref(Kd_ref), 
      V_mag(V_mag),
      inv_Kd_Vmag(1.0 / (Kd_ref * V_mag + 1e-30)) {}

void EnergyKernels::compute_poisson_rhs(const vex::vector<double>& m_gpu, vex::vector<double>& b_gpu) {
    b_gpu = mat_G_div * m_gpu;
}

double EnergyKernels::energy_and_grad(const vex::vector<double>& m_gpu,
                                    const vex::vector<double>& U_gpu,
                                    const Eigen::Vector3d& B_ext,
                                    vex::vector<double>& g_gpu) {
    
    // Internal + Demag Effective Fields: g_gpu = K_int * m + G_grad * U
    g_gpu = mat_K_int * m_gpu + mat_G_grad * U_gpu;
    
    // Zeeman Field Contribution
    // B_ext is constant. H_zee_i = Js_node_vols_i * B_ext / Vmag?
    // Let's use a VexCL kernel to add Zeeman.
    VEX_FUNCTION(double, add_zeeman, (double, g_i)(double, js_v_i)(double, bx)(double, by)(double, bz)(int, i),
        int c = i % 3;
        double b = (c == 0) ? bx : ((c == 1) ? by : bz);
        return g_i - js_v_i * b; // Gradient of -Js * m * B is -Js * B
    );

    // Apply Zeeman field per component
    // Assuming m_gpu and g_gpu have shape (3N) where nodes are interleaved (x1, y1, z1, x2, ...)
    // Wait, Js_node_vols has size N. We need to broadcast it.
    VEX_FUNCTION(double, get_jsv, (int, i)(const double*, jsv), return jsv[i / 3];);
    
    g_gpu = g_gpu + vex::elementwise(add_zeeman)(g_gpu, vex::elementwise(get_jsv)(vex::elementwise(vex::tag<0>(vex::range(0, (int)m_gpu.size()))), Js_node_vols_gpu), B_ext.x(), B_ext.y(), B_ext.z(), vex::elementwise(vex::tag<0>(vex::range(0, (int)m_gpu.size()))));

    // Internal Energy: 1/2 m^T K_int m
    double E_int = 0.5 * vex::dot(m_gpu, mat_K_int * m_gpu);
    
    // Demag Energy: 1/2 m^T G_grad U
    double E_demag = 0.5 * vex::dot(m_gpu, mat_G_grad * U_gpu);
    
    // Zeeman Energy: - m^T G_zeeman? No, let's compute it node-wise.
    // E_zee = - sum_i Js_node_vols_i * (m_ix * Bx + m_iy * By + m_iz * Bz)
    VEX_FUNCTION(double, zee_node_contrib, (double, m_i)(double, js_v_i)(double, bx)(double, by)(double, bz)(int, i),
        int c = i % 3;
        double b = (c == 0) ? bx : ((c == 1) ? by : bz);
        return -m_i * js_v_i * b;
    );
    
    double E_zee = vex::sum(vex::elementwise(zee_node_contrib)(m_gpu, vex::elementwise(get_jsv)(vex::elementwise(vex::tag<0>(vex::range(0, (int)m_gpu.size()))), Js_node_vols_gpu), B_ext.x(), B_ext.y(), B_ext.z(), vex::elementwise(vex::tag<0>(vex::range(0, (int)m_gpu.size())))));

    // Scale Energy & Gradient
    double E_total = (E_int + E_demag + E_zee) * inv_Kd_Vmag;
    g_gpu = g_gpu * inv_Kd_Vmag;
    
    return E_total;
}

double EnergyKernels::energy_only(const vex::vector<double>& m_gpu,
                                const vex::vector<double>& U_gpu,
                                const Eigen::Vector3d& B_ext) {
    
    double E_int = 0.5 * vex::dot(m_gpu, mat_K_int * m_gpu);
    double E_demag = 0.5 * vex::dot(m_gpu, mat_G_grad * U_gpu);
    
    VEX_FUNCTION(double, get_jsv, (int, i)(const double*, jsv), return jsv[i / 3];);
    VEX_FUNCTION(double, zee_node_contrib, (double, m_i)(double, js_v_i)(double, bx)(double, by)(double, bz)(int, i),
        int c = i % 3;
        double b = (c == 0) ? bx : ((c == 1) ? by : bz);
        return -m_i * js_v_i * b;
    );
    double E_zee = vex::sum(vex::elementwise(zee_node_contrib)(m_gpu, vex::elementwise(get_jsv)(vex::elementwise(vex::tag<0>(vex::range(0, (int)m_gpu.size()))), Js_node_vols_gpu), B_ext.x(), B_ext.y(), B_ext.z(), vex::elementwise(vex::tag<0>(vex::range(0, (int)m_gpu.size())))));

    return (E_int + E_demag + E_zee) * inv_Kd_Vmag;
}
