#include <iostream>
#include <vector>
#include <chrono>

#include <vexcl/vexcl.hpp>
#include "fem_utils.hpp"
#include "poisson_solve.hpp"
#include "energy_kernels.hpp"

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
    std::cout << "Loading mesh: " << argv[1] << std::endl;
    Mesh mesh;
    try {
        mesh = load_mesh_npz(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Error loading mesh: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Mesh: " << mesh.N << " nodes, " << mesh.E << " elements." << std::endl;

    // 3. Assemble Matrices (CPU)
    MaterialProperties props;
    int num_mats = mesh.mat_id.maxCoeff();
    props.A.assign(num_mats, 1e-11);
    props.K1.assign(num_mats, 0.0);
    props.Js.assign(num_mats, 1.0);
    props.k_easy.assign(num_mats, Eigen::Vector3d(0, 0, 1));

    SparseMatrixCSR L, K_int, G_div, G_grad;
    std::cout << "Assembling matrices..." << std::endl;
    auto start_asm = std::chrono::high_resolution_clock::now();
    assemble_matrices(mesh, props, L, K_int, G_div, G_grad);
    auto end_asm = std::chrono::high_resolution_clock::now();
    std::cout << "Assembly took " << std::chrono::duration<double>(end_asm - start_asm).count() << " s." << std::endl;

    // 4. Boundary Mask (Potential U=0 at outer shell boundary)
    // For this test, let's assume the user has correctly marked mesh.boundary_mask
    // If not, the Poisson solve might have a null space.
    std::vector<double> mask(mesh.N);
    for (int i = 0; i < mesh.N; ++i) mask[i] = mesh.boundary_mask(i);

    // 5. Setup Poisson Solver (AMG)
    std::cout << "Building AMG solver..." << std::endl;
    PoissonSolver solver(ctx, L, mask);

    // 6. Test RHS
    // Create a source term b = G_div * m where m = (0,0,1)
    std::vector<double> m_cpu(3 * mesh.N);
    for (int i = 0; i < mesh.N; ++i) {
        m_cpu[3 * i + 0] = 0.0;
        m_cpu[3 * i + 1] = 0.0;
        m_cpu[3 * i + 2] = 1.0;
    }
    vex::vector<double> m_gpu(ctx, m_cpu);
    vex::vector<double> b_gpu(ctx, mesh.N);
    
    // Use the divergence matrix (transferred to sparse matrix on GPU)
    vex::sparse::matrix<double> G_div_gpu(ctx, G_div.rows, G_div.cols, 
                                        G_div.ptr.data(), G_div.indices.data(), G_div.data.data());
    b_gpu = G_div_gpu * m_gpu;

    // 7. Solve Poisson
    vex::vector<double> U_gpu(ctx, mesh.N);
    U_gpu = 0.0; // Initial guess

    std::cout << "Solving Poisson equation..." << std::endl;
    auto start_solve = std::chrono::high_resolution_clock::now();
    solver.solve(b_gpu, U_gpu);
    auto end_solve = std::chrono::high_resolution_clock::now();
    std::cout << "Solve took " << std::chrono::duration<double>(end_solve - start_solve).count() << " s." << std::endl;

    // 8. Basic validation: Check residual
    vex::sparse::matrix<double> L_gpu(ctx, L.rows, L.cols, 
                                    L.ptr.data(), L.indices.data(), L.data.data());
    vex::vector<double> res = b_gpu - L_gpu * U_gpu;
    double res_norm = std::sqrt(vex::dot(res, res));
    std::cout << "L2 Residual Norm: " << res_norm << std::endl;

    return 0;
}
