#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <random>

#include <vexcl/vexcl.hpp>
#include "fem_utils.hpp"
#include "poisson_solve.hpp"
#include "energy_kernels.hpp"

int main(int argc, char** argv) {
    std::string mesh_path = "cube_60nm_shell.npz";
    if (argc > 1) mesh_path = argv[1];

    // 1. Setup VexCL Context
    vex::Context ctx(vex::Filter::Type(CL_DEVICE_TYPE_GPU) && vex::Filter::Count(1));
    if (!ctx) {
        std::cerr << "No GPU found!" << std::endl;
        return 1;
    }
    std::cout << "Using device: " << ctx.device(0).getInfo<CL_DEVICE_NAME>() << std::endl;

    // 2. Load Mesh
    std::cout << "Loading mesh from " << mesh_path << "..." << std::endl;
    Mesh mesh = load_mesh_npz(mesh_path);
    std::cout << "Mesh Size: " << mesh.N << " nodes, " << mesh.E << " elements" << std::endl;

    // 3. Material Properties (Same as profile_energy.py)
    double Js_val = 1.0;
    double A_red = 1.0;
    double K1_red = 0.1;

    MaterialProperties props;
    int max_mat = mesh.mat_id.maxCoeff();
    props.A.assign(max_mat, 0.0);
    props.K1.assign(max_mat, 0.0);
    props.Js.assign(max_mat, 0.0);
    props.k_easy.assign(max_mat, Eigen::Vector3d(0, 0, 1));

    if (max_mat >= 1) {
        props.A[0] = A_red;
        props.K1[0] = K1_red;
        props.Js[0] = Js_val;
    }

    // 4. Assemble Matrices
    SparseMatrixCSR L, K_int, G_div, G_grad;
    assemble_matrices(mesh, props, L, K_int, G_div, G_grad);

    double vmag = compute_vmag(mesh, props);
    std::vector<double> js_v = compute_js_node_volumes(mesh, props);
    
    // 5. Setup Solver and Kernels
    std::vector<double> mask_cpu(mesh.N);
    for(int i=0; i<mesh.N; ++i) mask_cpu[i] = mesh.boundary_mask(i);
    
    PoissonSolver poisson(ctx, L, mask_cpu);
    EnergyKernels kernels(ctx, K_int, G_div, G_grad, js_v, vmag);

    // 6. Test State: Random magnetization
    std::vector<double> m_cpu(3 * mesh.N);
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < mesh.N; ++i) {
        double mx = dist(gen);
        double my = dist(gen);
        double mz = dist(gen);
        double norm = std::sqrt(mx*mx + my*my + mz*mz);
        m_cpu[3 * i + 0] = mx / norm;
        m_cpu[3 * i + 1] = my / norm;
        m_cpu[3 * i + 2] = mz / norm;
    }
    vex::vector<double> m_gpu(ctx, m_cpu);
    vex::vector<double> U_gpu(ctx, mesh.N); U_gpu = 0.0;
    vex::vector<double> b_poisson(ctx, mesh.N);
    vex::vector<double> g_gpu(ctx, 3 * mesh.N);
    Eigen::Vector3d B_ext(0.01, 0.0, 0.0);

    // Warm-up
    std::cout << "Compiling kernels (warm-up)..." << std::endl;
    kernels.compute_poisson_rhs(m_gpu, b_poisson);
    auto info_warm = poisson.solve(b_poisson, U_gpu);
    double energy_warm = kernels.energy_and_grad(m_gpu, U_gpu, B_ext, g_gpu);
    ctx.finish();
    std::cout << "Warm-up results: Energy = " << energy_warm << ", Iters = " << info_warm.first << std::endl;

    // 7. Profiling Loop 1: Full Iteration (Solve U + Kernels)
    int n_repeats = 5;
    std::cout << "\nLoop 1: Recomputing potential U every time (" << n_repeats << " iterations)..." << std::endl;
    
    double last_e = 0;
    int total_iters = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_repeats; ++i) {
        U_gpu = 0.0; // Reset initial guess to zero to match Python profile
        kernels.compute_poisson_rhs(m_gpu, b_poisson);
        auto info = poisson.solve(b_poisson, U_gpu);
        last_e = kernels.energy_and_grad(m_gpu, U_gpu, B_ext, g_gpu);
        total_iters += info.first;
    }
    ctx.finish();
    auto t1 = std::chrono::high_resolution_clock::now();
    
    std::cout << "Last Energy: " << last_e << ", Avg Poisson Iters: " << (double)total_iters/n_repeats << std::endl;
    double total_full = std::chrono::duration<double>(t1 - t0).count();
    double avg_full = total_full / n_repeats;

    // 8. Profiling Loop 2: Kernels Only (Reuse U)
    std::cout << "Loop 2: Reusing precomputed potential U (" << n_repeats << " iterations)..." << std::endl;
    
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_repeats; ++i) {
        kernels.energy_and_grad(m_gpu, U_gpu, B_ext, g_gpu);
    }
    ctx.finish();
    auto t3 = std::chrono::high_resolution_clock::now();
    
    double total_kernels = std::chrono::duration<double>(t3 - t2).count();
    double avg_kernels = total_kernels / n_repeats;

    // 9. Report
    std::cout << "\n" << std::string(40, '=') << std::endl;
    std::cout << std::left << std::setw(25) << "Metric" << " | " << std::right << std::setw(10) << "Time (ms)" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::left << std::setw(25) << "Full Iteration (Avg)" << " | " << std::right << std::setw(10) << avg_full * 1000.0 << std::endl;
    std::cout << std::left << std::setw(25) << "Kernels Only (Avg)" << " | " << std::right << std::setw(10) << avg_kernels * 1000.0 << std::endl;
    std::cout << std::left << std::setw(25) << "Poisson Solve Overhead" << " | " << std::right << std::setw(10) << (avg_full - avg_kernels) * 1000.0 << std::endl;
    std::cout << std::string(40, '=') << std::endl;

    return 0;
}
