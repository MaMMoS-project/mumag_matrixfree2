#include <iostream>
#include <vector>
#include <chrono>

#include <vexcl/vexcl.hpp>
#include "fem_utils.hpp"
#include "poisson_solve.hpp"

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
    std::string device_name = ctx.device(0).getInfo<CL_DEVICE_NAME>();
    std::cout << "Using device: " << device_name << std::endl;

    // 2. Load Mesh
    Mesh mesh;
    try {
        mesh = load_mesh_npz(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Error loading mesh: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Mesh: " << mesh.N << " nodes, " << mesh.E << " elements." << std::endl;

    // 3. Assemble Required Matrices (CPU)
    MaterialProperties props;
    int num_mats = mesh.mat_id.maxCoeff();
    props.Js.assign(num_mats, 0.0);
    if (num_mats >= 1) props.Js[0] = 1.0; // Benchmark use Js=1.0 for first material (magnet)

    SparseMatrixCSR L, G_div;
    std::cout << "Assembling matrices (L, G_div)..." << std::endl;
    auto start_asm = std::chrono::high_resolution_clock::now();
    assemble_poisson_matrices(mesh, props, L, G_div);
    auto end_asm = std::chrono::high_resolution_clock::now();
    std::cout << "Assembly took " << std::chrono::duration<double>(end_asm - start_asm).count() << " s." << std::endl;

    // 4. Boundary Mask
    std::vector<double> mask(mesh.N);
    for (int i = 0; i < mesh.N; ++i) mask[i] = mesh.boundary_mask(i);
    vex::vector<double> mask_gpu(ctx, mask);

    // 5. Setup Poisson Solver (AMG)
    std::cout << "Building AMG solver..." << std::endl;
    PoissonSolver solver(ctx, L, mask);

    // 6. Setup Source term b = G_div * m
    std::vector<double> m_cpu(3 * mesh.N, 0.0);
    for (int i = 0; i < mesh.N; ++i) {
        m_cpu[3 * i + 2] = 1.0; // m = (0,0,1)
    }
    vex::vector<double> m_gpu(ctx, m_cpu);
    vex::vector<double> b_gpu(ctx, mesh.N);
    
    vex::sparse::matrix<double> mat_G_div(ctx, G_div.rows, G_div.cols, G_div.ptr, G_div.indices, G_div.data);
    b_gpu = (mat_G_div * m_gpu) * mask_gpu;

    // 7. Solve
    vex::vector<double> U_gpu(ctx, mesh.N);
    U_gpu = 0.0;

    std::cout << "\nStarting Poisson Benchmarks (Tolerance 1e-10):" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    
    int iters;
    double duration;
    std::tie(iters, duration) = solver.solve(b_gpu, U_gpu);
    
    // Calculate final relative residual: ||b - L*U|| / ||b|| (interior nodes only)
    vex::sparse::matrix<double> L_gpu(ctx, L.rows, L.cols, L.ptr, L.indices, L.data);
    vex::vector<double> r = (b_gpu - L_gpu * U_gpu) * mask_gpu;
    vex::Reductor<double, vex::SUM> reduce_sum(ctx);
    double r2 = reduce_sum(r * r);
    double b2 = reduce_sum(b_gpu * b_gpu);
    double rel_res = std::sqrt(r2 / (b2 + 1e-30));

    std::cout << "Amg         : " << iters << " iterations, " << duration << " s, rel_res: " << rel_res << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;


    return 0;
}
