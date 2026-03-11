#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>

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
    int max_mat = mesh.mat_id.maxCoeff();
    props.A.assign(max_mat, 0.0);
    props.K1.assign(max_mat, 0.0);
    props.Js.assign(max_mat, 0.0);
    props.k_easy.assign(max_mat, Eigen::Vector3d(0, 0, 1));

    // mat_id 1 = cube, mat_id 2 = air (shell)
    if (max_mat >= 1) {
        props.A[0] = A_red;
        props.K1[0] = K1_red;
        props.Js[0] = Js_red;
    }
    // mat_id 2 remains 0 (air)

    // 4. Assemble Matrices
    SparseMatrixCSR L, K_int, G_div, G_grad;
    assemble_matrices(mesh, props, L, K_int, G_div, G_grad);

    double vmag = compute_vmag(mesh, props);
    std::vector<double> js_v = compute_js_node_volumes(mesh, props);
    
    double V_mag_si = vmag * 1e-27;
    double SI_FACTOR = Kd * V_mag_si;

    std::cout << std::scientific << std::setprecision(3);
    std::cout << "Cube Volume (SI): " << V_mag_si << " m^3" << std::endl;
    std::cout << "Normalization Kd: " << Kd << " J/m^3" << std::endl;
    std::cout << std::fixed << std::setprecision(6) << std::endl;

    // 5. Setup Solver and Kernels
    std::vector<double> mask_cpu(mesh.N);
    for(int i=0; i<mesh.N; ++i) mask_cpu[i] = mesh.boundary_mask(i);
    PoissonSolver poisson(ctx, L, mask_cpu);
    EnergyKernels kernels(ctx, K_int, G_div, G_grad, js_v, vmag);

    vex::vector<double> g_gpu(ctx, 3 * mesh.N);
    vex::vector<double> U_zero(ctx, mesh.N); U_zero = 0.0;

    // 6. Test States

    // --- HELICAL (Exchange) ---
    double L_cube = 20.0; // nm
    double k_wave = M_PI / L_cube;
    std::vector<double> m_hel_cpu(3 * mesh.N, 0.0);
    for (int i = 0; i < mesh.N; ++i) {
        double x = mesh.points(i, 0);
        m_hel_cpu[3 * i + 0] = std::cos(k_wave * x);
        m_hel_cpu[3 * i + 1] = std::sin(k_wave * x);
    }
    vex::vector<double> m_hel(ctx, m_hel_cpu);
    double e_ex = kernels.energy_and_grad(m_hel, U_zero, Eigen::Vector3d::Zero(), g_gpu);
    double e_ex_an = A_red * k_wave * k_wave;
    double E_ex_analytic_si = A_si * std::pow(k_wave * 1e9, 2) * V_mag_si;

    std::cout << "--- EXCHANGE ---" << std::endl;
    std::cout << "Internal:  " << e_ex << " (Analytic: " << e_ex_an << ", Err: " << std::abs(e_ex - e_ex_an)/e_ex_an * 100.0 << "%)" << std::endl;
    std::cout << "SI (J):    " << std::scientific << e_ex * SI_FACTOR << " (Analytic: " << E_ex_analytic_si << ")" << std::endl << std::fixed << std::endl;

    // --- ZEEMAN (Uniform X) ---
    double B_ext_si = 0.1; // Tesla
    double b_red = B_ext_si / Js;
    std::vector<double> m_unif_x_cpu(3 * mesh.N, 0.0);
    for (int i = 0; i < mesh.N; ++i) m_unif_x_cpu[3 * i + 0] = 1.0;
    vex::vector<double> m_unif_x(ctx, m_unif_x_cpu);
    
    double e_z = kernels.energy_and_grad(m_unif_x, U_zero, Eigen::Vector3d(b_red, 0, 0), g_gpu);
    double e_z_an = -2.0 * b_red;
    double E_z_analytic_si = -(1.0/MU0_SI) * Js * V_mag_si * B_ext_si;

    std::cout << "--- ZEEMAN ---" << std::endl;
    std::cout << "Internal:  " << e_z << " (Analytic: " << e_z_an << ", Err: " << std::abs(e_z - e_z_an)/std::abs(e_z_an) * 100.0 << "%)" << std::endl;
    std::cout << "SI (J):    " << std::scientific << e_z * SI_FACTOR << " (Analytic: " << E_z_analytic_si << ")" << std::endl << std::fixed << std::endl;

    // --- DEMAG (Uniform X) ---
    vex::vector<double> b_poisson(ctx, mesh.N);
    vex::vector<double> U_demag(ctx, mesh.N); U_demag = 0.0;
    kernels.compute_poisson_rhs(m_unif_x, b_poisson);
    poisson.solve(b_poisson, U_demag);

    double e_dem = kernels.energy_and_grad(m_unif_x, U_demag, Eigen::Vector3d::Zero(), g_gpu);
    double e_dem_an = 1.0 / 3.0; // Approximation for cube
    double E_d_analytic_si = (1.0/(6.0*MU0_SI)) * (Js*Js) * V_mag_si;

    std::cout << "--- DEMAG ---" << std::endl;
    std::cout << "Internal:  " << e_dem << " (Analytic ~ " << e_dem_an << ", Err: " << std::abs(e_dem - e_dem_an)/e_dem_an * 100.0 << "%)" << std::endl;
    std::cout << "SI (J):    " << std::scientific << e_dem * SI_FACTOR << " (Analytic ~ " << E_d_analytic_si << ")" << std::endl << std::fixed << std::endl;

    // --- ANISOTROPY (45 deg) ---
    std::vector<double> m_aniso_45_cpu(3 * mesh.N, 0.0);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    for (int i = 0; i < mesh.N; ++i) {
        m_aniso_45_cpu[3 * i + 0] = inv_sqrt2;
        m_aniso_45_cpu[3 * i + 2] = inv_sqrt2;
    }
    vex::vector<double> m_aniso_45(ctx, m_aniso_45_cpu);
    double e_an = kernels.energy_and_grad(m_aniso_45, U_zero, Eigen::Vector3d::Zero(), g_gpu);
    double e_an_an = -K1_red * 0.5;
    double E_an_expected_si = -K1 * V_mag_si * 0.5;

    std::cout << "--- ANISOTROPY ---" << std::endl;
    std::cout << "Internal:  " << e_an << " (Analytic: " << e_an_an << ", Err: " << std::abs(e_an - e_an_an)/std::abs(e_an_an) * 100.0 << "%)" << std::endl;
    std::cout << "SI (J):    " << std::scientific << e_an * SI_FACTOR << " (Analytic: " << E_an_expected_si << ")" << std::endl << std::fixed << std::endl;

    return 0;
}
