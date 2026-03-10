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

void assemble_poisson_matrices(
    const Mesh& mesh,
    const MaterialProperties& props,
    SparseMatrixCSR& L_csr,
    SparseMatrixCSR& G_div_csr
) {
    int N = mesh.N;
    int E = mesh.E;

    std::vector<Eigen::Triplet<double>> L_triplets;
    std::vector<Eigen::Triplet<double>> G_div_triplets;

    Eigen::Matrix<double, 4, 3> grad_hat;
    grad_hat << -1, -1, -1,
                 1,  0,  0,
                 0,  1,  0,
                 0,  0,  1;

    for (int e = 0; e < E; ++e) {
        Eigen::Vector4i nodes = mesh.conn.row(e);
        int mid = mesh.mat_id(e) - 1;
        
        Eigen::Vector3d v0 = mesh.points.row(nodes(0));
        Eigen::Matrix3d J;
        J.col(0) = mesh.points.row(nodes(1)) - v0.transpose();
        J.col(1) = mesh.points.row(nodes(2)) - v0.transpose();
        J.col(2) = mesh.points.row(nodes(3)) - v0.transpose();

        double volume = std::abs(J.determinant()) / 6.0;
        Eigen::Matrix3d JinvT = J.inverse().transpose();
        Eigen::Matrix<double, 4, 3> grad_phi = grad_hat * JinvT.transpose();

        double Js_red = props.Js[mid];

        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                double val = volume * grad_phi.row(a).dot(grad_phi.row(b));
                L_triplets.emplace_back(nodes(a), nodes(b), val);
                
                for (int c = 0; c < 3; ++c) {
                    double g_val = Js_red * (volume / 4.0) * grad_phi(a, c);
                    G_div_triplets.emplace_back(nodes(a), 3 * nodes(b) + c, g_val);
                }
            }
        }
    }

    L_csr = triplets_to_csr(N, N, L_triplets);
    G_div_csr = triplets_to_csr(N, 3 * N, G_div_triplets);
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
        
        // Element Jacobian and Volume (mesh units nm)
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
                // g_i = 2 \int A_red \nabla m \cdot \nabla \phi_i dV
                double A_red = props.A[mid];
                double ex_val = 2.0 * A_red * val;
                for (int c = 0; c < 3; ++c) {
                    K_int_triplets.emplace_back(3 * nodes(a) + c, 3 * nodes(b) + c, ex_val);
                }
            }
        }

        // 3. Analytic Uniaxial Anisotropy part of K_int (3N x 3N)
        // E_an = -q_c * (V/20) * [(\sum v_i)^2 + \sum v_i^2] where v_i = m_i . k
        // g_i = -2 q_c (V/20) (\sum_j v_j + v_i) k
        // Contribution to K_ij (block i,j): -2 q_c (V/20) (k k^T) * (1 + delta_ij)
        double q_c = props.K1[mid];
        Eigen::Vector3d k = props.k_easy[mid];
        Eigen::Matrix3d K_base = -2.0 * q_c * (volume / 20.0) * (k * k.transpose());
        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                double factor = (a == b) ? 2.0 : 1.0;
                Eigen::Matrix3d Akk = factor * K_base;
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        K_int_triplets.emplace_back(3 * nodes(a) + r, 3 * nodes(b) + c, Akk(r, c));
                    }
                }
            }
        }

        // 4. Divergence Matrix G_div (N x 3N)
        // b_a = \int j_c m . \nabla \phi_a dV = j_c (V/4) (\sum m_b) . \nabla \phi_a
        // (G_div)_{a, 3b+c} = j_c (V/4) (\nabla \phi_a)_c
        double j_c = props.Js[mid];
        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                for (int c = 0; c < 3; ++c) {
                    double val = j_c * (volume / 4.0) * grad_phi(a, c);
                    G_div_triplets.emplace_back(nodes(a), 3 * nodes(b) + c, val);
                }
            }
        }
    }

    L_csr = triplets_to_csr(N, N, L_triplets);
    K_int_csr = triplets_to_csr(3 * N, 3 * N, K_int_triplets);
    G_div_csr = triplets_to_csr(N, 3 * N, G_div_triplets);

    // G_grad = G_div^T
    // Demag Energy E_dem = \int j_c m . \nabla U dV = m^T G_div^T U
    std::vector<Eigen::Triplet<double>> G_grad_triplets;
    for (const auto& t : G_div_triplets) {
        G_grad_triplets.emplace_back(t.col(), t.row(), t.value());
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

std::vector<double> compute_js_node_volumes(const Mesh& mesh, const MaterialProperties& props) {
    std::vector<double> js_v(mesh.N, 0.0);
    for (int e = 0; e < mesh.E; ++e) {
        Eigen::Vector4i nodes = mesh.conn.row(e);
        int mid = mesh.mat_id(e) - 1;
        double j_c = props.Js[mid];

        Eigen::Vector3d v0 = mesh.points.row(nodes(0));
        Eigen::Matrix3d J;
        J.col(0) = mesh.points.row(nodes(1)) - v0.transpose();
        J.col(1) = mesh.points.row(nodes(2)) - v0.transpose();
        J.col(2) = mesh.points.row(nodes(3)) - v0.transpose();
        double volume = std::abs(J.determinant()) / 6.0;

        for (int a = 0; a < 4; ++a) {
            js_v[nodes(a)] += j_c * (volume / 4.0);
        }
    }
    return js_v;
}

double compute_vmag(const Mesh& mesh, const MaterialProperties& props) {
    double vmag = 0.0;
    for (int e = 0; e < mesh.E; ++e) {
        int mid = mesh.mat_id(e) - 1;
        if (props.Js[mid] > 0) {
            Eigen::Vector4i nodes = mesh.conn.row(e);
            Eigen::Vector3d v0 = mesh.points.row(nodes(0));
            Eigen::Matrix3d J;
            J.col(0) = mesh.points.row(nodes(1)) - v0.transpose();
            J.col(1) = mesh.points.row(nodes(2)) - v0.transpose();
            J.col(2) = mesh.points.row(nodes(3)) - v0.transpose();
            vmag += std::abs(J.determinant()) / 6.0;
        }
    }
    return vmag;
}

