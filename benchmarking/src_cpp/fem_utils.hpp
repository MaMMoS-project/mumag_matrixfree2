#ifndef FEM_UTILS_HPP
#define FEM_UTILS_HPP

#include <vector>
#include <string>
#include <Eigen/Core>
#include <Eigen/Sparse>

/**
 * @brief Simple CSR (Compressed Sparse Row) representation.
 */
struct SparseMatrixCSR {
    int rows, cols;
    std::vector<int> ptr;
    std::vector<int> indices;
    std::vector<double> data;

    SparseMatrixCSR() : rows(0), cols(0) {}
};

/**
 * @brief Node-based material and geometric data.
 */
struct Mesh {
    int N; // Number of nodes
    int E; // Number of elements
    Eigen::MatrixXd points;     // (N, 3)
    Eigen::MatrixXi conn;       // (E, 4)
    Eigen::VectorXi mat_id;     // (E,)
    
    // Lumped volumes at nodes
    Eigen::VectorXd node_volumes; 
    
    // Boundary mask: 1.0 for interior, 0.0 for Dirichlet boundary (potential U=0)
    Eigen::VectorXd boundary_mask; 
};

/**
 * @brief Material properties lookup (from KRN).
 */
struct MaterialProperties {
    std::vector<double> A;      // Exchange constant [J/m]
    std::vector<double> K1;     // Uniaxial anisotropy [J/m^3]
    std::vector<double> Js;     // Saturation polarization [Tesla]
    std::vector<Eigen::Vector3d> k_easy; // Easy axes (normalized)
};

/**
 * @brief Load a mesh from an NPZ file (requires cnpy or equivalent).
 * For this exercise, we will implement it using simple std::vectors for knt and ijk.
 */
Mesh load_mesh_npz(const std::string& path);

/**
 * @brief Assembles all matrices needed for the micromagnetic simulation.
 * 
 * Returns:
 * 1. Stiffness matrix L (N x N) for Poisson.
 * 2. Internal matrix K_int (3N x 3N) for Exchange + Anisotropy.
 * 3. Divergence matrix G_div (N x 3N) for Poisson RHS.
 * 4. Gradient matrix G_grad (3N x N) for Demag field.
 */
void assemble_matrices(
    const Mesh& mesh,
    const MaterialProperties& props,
    SparseMatrixCSR& L,
    SparseMatrixCSR& K_int,
    SparseMatrixCSR& G_div,
    SparseMatrixCSR& G_grad
);

/**
 * @brief Only assemble matrices required for Poisson solve: L and G_div.
 */
void assemble_poisson_matrices(
    const Mesh& mesh,
    const MaterialProperties& props,
    SparseMatrixCSR& L,
    SparseMatrixCSR& G_div
);


/**
 * @brief Compute lumped nodal volumes weighted by Js: Js_node_vols[i] = sum_{e in i} Js_e * Ve / 4
 */
std::vector<double> compute_js_node_volumes(const Mesh& mesh, const MaterialProperties& props);

/**
 * @brief Compute total magnetic volume: Vmag = sum_{e where Js > 0} Ve
 */
double compute_vmag(const Mesh& mesh, const MaterialProperties& props);

#endif // FEM_UTILS_HPP
