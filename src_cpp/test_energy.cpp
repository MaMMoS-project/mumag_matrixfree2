#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

#include <vexcl/vexcl.hpp>
#include "fem_utils.hpp"
#include "poisson_solve.hpp"
#include "energy_kernels.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mesh.npz>" << std::endl;
        return 1;
    }

    // 1. Setup VexCL Context
    vex::Context ctx(vex::Filter::Type(CL_DEVICE_TYPE_GPU) && vex::Filter::Count(1));
    if (!ctx) {
        std::cerr << "No GPU found!" << std::endl;
        return 1;
    }
    std::cout << "Using device: " << ctx.device(0).getInfo<CL_DEVICE_NAME>() << std::endl;

    // 2. Load Mesh
    Mesh mesh = load_mesh_npz(argv[1]);
    std::cout << "Mesh: " << mesh.N << " nodes, " << mesh.E << " elements." << std::endl;

    // 3. Material Properties (Normalized NdFeB-like as in test_energy.py)
    double Js = 1.6; // Tesla
    double K1 = 4.3e6; // J/m^3
    double A_si = 7.7e-12; // J/m
    double MU0_SI = 4.0 * M_PI * 1e-7;
    double Kd = (Js * Js) / (2.0 * MU0_SI);

    // Reduced properties for nm mesh
    double A_red = (A_si * 1e18) / Kd;
    double K1_red = K1 / Kd;
    double Js_red = 1.0;

    MaterialProperties props;
    int num_mats = mesh.mat_id.maxCoeff();
    props.A.assign(num_mats, 0.0);
    props.K1.assign(num_mats, 0.0);
    props.Js.assign(num_mats, 0.0);
    props.k_easy.assign(num_mats, Eigen::Vector3d(0, 0, 1));

    // Assume mat_id 1 is the magnet, others are air
    if (num_mats >= 1) {
        props.A[0] = A_red;
        props.K1[0] = K1_red;
        props.Js[0] = Js_red;
    }

    // 4. Assemble Matrices
    SparseMatrixCSR L, K_int, G_div, G_grad;
    assemble_matrices(mesh, props, L, K_int, G_div, G_grad);

    double vmag = compute_vmag(mesh, props);
    std::vector<double> js_v = compute_js_node_volumes(mesh, props);
    std::cout << "Vmag: " << vmag << " nm^3" << std::endl;

    // 5. Setup Solver and Kernels
    std::vector<double> mask_cpu(mesh.N);
    for(int i=0; i<mesh.N; ++i) mask_cpu[i] = mesh.boundary_mask(i);
    PoissonSolver poisson(ctx, L, mask_cpu);
    EnergyKernels kernels(ctx, K_int, G_div, G_grad, js_v, vmag);

    // 6. Test States
    // --- HELICAL (Exchange) ---
    double L_cube = 20.0; // nm
    double k_wave = M_PI / L_cube;
    std::vector<double> m_hel_cpu(3 * mesh.N);
    for (int i = 0; i < mesh.N; ++i) {
        double x = mesh.points(i, 0);
        m_hel_cpu[3 * i + 0] = std::cos(k_wave * x);
        m_hel_cpu[3 * i + 1] = std::sin(k_wave * x);
        m_hel_cpu[3 * i + 2] = 0.0;
    }
    vex::vector<double> m_hel(ctx, m_hel_cpu);
    vex::vector<double> U_zero(ctx, mesh.N); U_zero = 0.0;
    vex::vector<double> g_gpu(ctx, 3 * mesh.N);

    double e_ex = kernels.energy_and_grad(m_hel, U_zero, Eigen::Vector3d::Zero(), g_gpu);
    double e_ex_an = A_red * k_wave * k_wave;
    std::cout << "\n--- EXCHANGE ---" << std::endl;
    std::cout << "Numerical: " << e_ex << " (Analytic: " << e_ex_an << ", Err: " << std::abs(e_ex - e_ex_an)/e_ex_an * 100 << "%)" << std::endl;

    // --- ZEEMAN ---
    double B_ext_si = 0.1; // Tesla
    double b_red = B_ext_si / Js;
    std::vector<double> m_unif_x_cpu(3 * mesh.N, 0.0);
    for (int i = 0; i < mesh.N; ++i) m_unif_x_cpu[3 * i + 0] = 1.0;
    vex::vector<double> m_unif_x(ctx, m_unif_x_cpu);
    
    double e_z = kernels.energy_and_grad(m_unif_x, U_zero, Eigen::Vector3d(b_red, 0, 0), g_gpu);
    double e_z_an = -2.0 * b_red;
    std::cout << "\n--- ZEEMAN ---" << std::endl;
    std::cout << "Numerical: " << e_z << " (Analytic: " << e_z_an << ", Err: " << std::abs(e_z - e_z_an)/std::abs(e_z_an) * 100 << "%)" << std::endl;

    // --- ANISOTROPY ---
    std::vector<double> m_aniso_45_cpu(3 * mesh.N, 0.0);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    for (int i = 0; i < mesh.N; ++i) {
        m_aniso_45_cpu[3 * i + 0] = inv_sqrt2;
        m_aniso_45_cpu[3 * i + 2] = inv_sqrt2;
    }
    vex::vector<double> m_aniso_45(ctx, m_aniso_45_cpu);
    double e_an = kernels.energy_and_grad(m_aniso_45, U_zero, Eigen::Vector3d::Zero(), g_gpu);
    double e_an_an = -K1_red * 0.5;
    std::cout << "\n--- ANISOTROPY ---" << std::endl;
    std::cout << "Numerical: " << e_an << " (Analytic: " << e_an_an << ", Err: " << std::abs(e_an - e_an_an)/std::abs(e_an_an) * 100 << "%)" << std::endl;

    // --- DEMAG ---
    vex::vector<double> b_poisson(ctx, mesh.N);
    vex::vector<double> U_demag(ctx, mesh.N); U_demag = 0.0;
    kernels.compute_poisson_rhs(m_unif_x, b_poisson);
    poisson.solve(b_poisson, U_demag);

    double e_dem = kernels.energy_and_grad(m_unif_x, U_demag, Eigen::Vector3d::Zero(), g_gpu);
    double e_dem_an = 1.0 / 3.0; // Approximation for cube
    std::cout << "\n--- DEMAG ---" << std::endl;
    std::cout << "Numerical: " << e_dem << " (Analytic ~ " << e_dem_an << ")" << std::endl;

    return 0;
}
