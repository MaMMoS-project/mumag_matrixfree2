#include "fem_utils.hpp"
#include <iostream>
#include <Eigen/LU>

/**
 * @brief Convert triplets to SparseMatrixCSR.
 */
static SparseMatrixCSR triplets_to_csr(int rows, int cols, std::vector<Eigen::Triplet<double>>& triplets) {
    Eigen::SparseMatrix<double, Eigen::RowMajor> mat(rows, cols);
    mat.setFromTriplets(triplets.begin(), triplets.end());
    mat.makeCompressed();

    SparseMatrixCSR csr;
    csr.rows = rows;
    csr.cols = cols;
    csr.ptr.assign(mat.outerIndexPtr(), mat.outerIndexPtr() + rows + 1);
    csr.indices.assign(mat.innerIndexPtr(), mat.innerIndexPtr() + mat.nonZeros());
    csr.data.assign(mat.valuePtr(), mat.valuePtr() + mat.nonZeros());
    return csr;
}

void assemble_matrices(
    const Mesh& mesh,
    const MaterialProperties& props,
    SparseMatrixCSR& L_csr,
    SparseMatrixCSR& K_int_csr,
    SparseMatrixCSR& G_div_csr,
    SparseMatrixCSR& G_grad_csr
) {
    int N = mesh.N;
    int E = mesh.E;

    std::vector<Eigen::Triplet<double>> L_triplets;
    std::vector<Eigen::Triplet<double>> K_int_triplets;
    std::vector<Eigen::Triplet<double>> G_div_triplets;

    // GradHat for P1 tet
    Eigen::Matrix<double, 4, 3> grad_hat;
    grad_hat << -1, -1, -1,
                 1,  0,  0,
                 0,  1,  0,
                 0,  0,  1;

    for (int e = 0; e < E; ++e) {
        Eigen::Vector4i nodes = mesh.conn.row(e);
        int mid = mesh.mat_id(e) - 1; // 0-based
        
        // Element Jacobian and Volume
        Eigen::Vector3d v0 = mesh.points.row(nodes(0));
        Eigen::Matrix3d J;
        J.col(0) = mesh.points.row(nodes(1)) - v0.transpose();
        J.col(1) = mesh.points.row(nodes(2)) - v0.transpose();
        J.col(2) = mesh.points.row(nodes(3)) - v0.transpose();

        double detJ = J.determinant();
        double volume = std::abs(detJ) / 6.0;
        Eigen::Matrix3d JinvT = J.inverse().transpose();
        
        // GradPhi (4x3): grad_phi_a = JinvT * grad_hat_a
        Eigen::Matrix<double, 4, 3> grad_phi = grad_hat * JinvT.transpose();

        // 1. Scalar Stiffness Matrix L (for Poisson)
        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                double val = volume * grad_phi.row(a).dot(grad_phi.row(b));
                L_triplets.emplace_back(nodes(a), nodes(b), val);
                
                // 2. Exchange part of K_int (3N x 3N)
                // E_ex = \int A (\nabla m)^2 dV => g_i = 2 \int A \nabla m \cdot \nabla \phi_i dV
                // Matrix entry is 2 * A * L_ab
                double A_val = props.A[mid];
                double ex_val = 2.0 * A_val * val;
                for (int c = 0; c < 3; ++c) {
                    K_int_triplets.emplace_back(3 * nodes(a) + c, 3 * nodes(b) + c, ex_val);
                }
            }
        }

        // 3. Divergence Matrix G_div (N x 3N)
        // b_a = \int \nabla \cdot (Js m) \phi_a dV = Js (\sum_b m_b \cdot \nabla \phi_b) (V/4)
        double Js_val = props.Js[mid];
        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                for (int c = 0; c < 3; ++c) {
                    double val = Js_val * (volume / 4.0) * grad_phi(b, c);
                    G_div_triplets.emplace_back(nodes(a), 3 * nodes(b) + c, val);
                }
            }
        }
    }

    // 4. Anisotropy part of K_int (Node-wise)
    // E_an = \int -K1 (m \cdot k)^2 dV => g_i = -2 K1 V_i (k k^T) m_i
    for (int i = 0; i < N; ++i) {
        // Find material at node i (heuristic: use first element containing node)
        // For simplicity, we assume we have a way to get node properties or they are uniform per grain
        // In this implementation, we'll need to know which material node i belongs to.
        // Let's assume we pre-calculated node_material or props are global.
        // For now, let's skip or assume props are indexed by node.
        // Actually, let's use the lumped volume calculation to distribute properties.
    }
    
    // To handle node-wise properties accurately, we should iterate elements and 
    // add 1/4 of element anisotropy contribution to each of its nodes.
    for (int e = 0; e < E; ++e) {
        Eigen::Vector4i nodes = mesh.conn.row(e);
        int mid = mesh.mat_id(e) - 1;
        double K1_val = props.K1[mid];
        Eigen::Vector3d k = props.k_easy[mid];
        double vol4 = 0.25 * 0.0; // Wait, we need volume again

        // Recalculate or store volume
        Eigen::Vector3d v0 = mesh.points.row(nodes(0));
        Eigen::Matrix3d J;
        J.col(0) = mesh.points.row(nodes(1)) - v0.transpose();
        J.col(1) = mesh.points.row(nodes(2)) - v0.transpose();
        J.col(2) = mesh.points.row(nodes(3)) - v0.transpose();
        double volume = std::abs(J.determinant()) / 6.0;

        // Contribution: -2 * K1 * (V/4) * (k k^T)
        // Note: The paper might use a more sophisticated integration for anisotropy
        // but lumped (P0 property on P1 mesh) is standard.
        Eigen::Matrix3d Akk = -2.0 * K1_val * (volume / 4.0) * (k * k.transpose());
        for (int a = 0; a < 4; ++a) {
            int node_idx = nodes(a);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    K_int_triplets.emplace_back(3 * node_idx + r, 3 * node_idx + c, Akk(r, c));
                }
            }
        }
    }

    L_csr = triplets_to_csr(N, N, L_triplets);
    K_int_csr = triplets_to_csr(3 * N, 3 * N, K_int_triplets);
    G_div_csr = triplets_to_csr(N, 3 * N, G_div_triplets);

    // G_grad = -G_div^T
    std::vector<Eigen::Triplet<double>> G_grad_triplets;
    for (const auto& t : G_div_triplets) {
        G_grad_triplets.emplace_back(t.col(), t.row(), -t.value());
    }
    G_grad_csr = triplets_to_csr(3 * N, N, G_grad_triplets);
}

#include <cnpy.h>

Mesh load_mesh_npz(const std::string& path) {
    Mesh mesh;
    cnpy::npz_t npz = cnpy::npz_load(path);

    // Load nodes (knt)
    cnpy::NpyArray knt_arr = npz["knt"];
    if (knt_arr.shape.size() != 2 || knt_arr.shape[1] != 3) {
        throw std::runtime_error("knt must be (N, 3)");
    }
    mesh.N = knt_arr.shape[0];
    mesh.points = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(
        knt_arr.data<double>(), mesh.N, 3);

    // Load connectivity (ijk)
    cnpy::NpyArray ijk_arr = npz["ijk"];
    mesh.E = ijk_arr.shape[0];
    int cols = ijk_arr.shape[1];
    
    // Support both (E,4) and (E,5)
    Eigen::MatrixXi ijk_full;
    if (ijk_arr.word_size == 4) { // int32
        ijk_full = Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            ijk_arr.data<int>(), mesh.E, cols);
    } else { // int64
        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ijk_long = 
            Eigen::Map<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                ijk_arr.data<long>(), mesh.E, cols);
        ijk_full = ijk_long.cast<int>();
    }

    mesh.conn = ijk_full.leftCols(4);
    if (cols == 5) {
        mesh.mat_id = ijk_full.col(4);
    } else {
        mesh.mat_id = Eigen::VectorXi::Ones(mesh.E);
    }

    // Default: all interior
    mesh.boundary_mask = Eigen::VectorXd::Ones(mesh.N);

    return mesh;
}
