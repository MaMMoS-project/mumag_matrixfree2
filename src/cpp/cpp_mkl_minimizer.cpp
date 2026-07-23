#include <mkl.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <mutex>
#include <cstdint>

struct PardisoState {
    void* pt[64];
    int iparm[64];
    int mtype;
    int maxfct;
    int mnum;
    int n;
    int* ia;
    int* ja;
    double* a;
};

static std::unordered_map<int64_t, PardisoState*> pardiso_states;
static std::mutex pardiso_mutex;
static int64_t next_pardiso_id = 1;

extern "C" {
    int64_t init_pardiso(int n, double* a, int* ia, int* ja) {
        PardisoState* state = new PardisoState();
        state->n = n;
        state->a = a;
        state->ia = ia;
        state->ja = ja;
        state->mtype = 2; // Real symmetric positive definite
        state->maxfct = 1;
        state->mnum = 1;
        
        for (int i = 0; i < 64; i++) {
            state->pt[i] = 0;
            state->iparm[i] = 0;
        }
        state->iparm[0] = 1; // No default values
        state->iparm[1] = 2; // METIS nested dissection
        state->iparm[34] = 1; // 0-based indexing for CSR arrays
        
        int phase = 12; // Analysis and numerical factorization
        int perm = 0;
        int nrhs = 1;
        int msglvl = 0;
        int error = 0;
        double ddum;
        
        pardiso(state->pt, &state->maxfct, &state->mnum, &state->mtype, &phase,
                &state->n, state->a, state->ia, state->ja, &perm, &nrhs,
                state->iparm, &msglvl, &ddum, &ddum, &error);
                
        if (error != 0) {
            delete state;
            return -error;
        }
        
        std::lock_guard<std::mutex> lock(pardiso_mutex);
        int64_t id = next_pardiso_id++;
        pardiso_states[id] = state;
        return id;
    }

    void free_pardiso(int64_t id) {
        std::lock_guard<std::mutex> lock(pardiso_mutex);
        auto it = pardiso_states.find(id);
        if (it != pardiso_states.end()) {
            PardisoState* state = it->second;
            int phase = -1;
            int perm = 0;
            int nrhs = 1;
            int msglvl = 0;
            int error = 0;
            double ddum;
            pardiso(state->pt, &state->maxfct, &state->mnum, &state->mtype, &phase,
                    &state->n, state->a, state->ia, state->ja, &perm, &nrhs,
                    state->iparm, &msglvl, &ddum, &ddum, &error);
            delete state;
            pardiso_states.erase(it);
        }
    }

    int pardiso_solve_direct(int64_t handle_id, const double* b, double* x) {
        PardisoState* state = nullptr;
        {
            std::lock_guard<std::mutex> lock(pardiso_mutex);
            auto it = pardiso_states.find(handle_id);
            if (it == pardiso_states.end()) {
                return -1;
            }
            state = it->second;
        }
        
        int phase = 33; // Solve phase
        int perm = 0;
        int nrhs = 1;
        int msglvl = 0;
        int error = 0;
        
        pardiso(state->pt, &state->maxfct, &state->mnum, &state->mtype, &phase,
                &state->n, state->a, state->ia, state->ja, &perm, &nrhs,
                state->iparm, &msglvl, const_cast<double*>(b), x, &error);
                
        return error;
    }

    void cayley_update(int N, const double* m, const double* H, double tau, double* m_new) {
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            double k0 = 0.5 * tau * H[3 * i + 0];
            double k1 = 0.5 * tau * H[3 * i + 1];
            double k2 = 0.5 * tau * H[3 * i + 2];
            double k_sq = k0 * k0 + k1 * k1 + k2 * k2;
            double denom = 1.0 + k_sq;
            
            // km = k x m
            double km0 = k1 * m[3 * i + 2] - k2 * m[3 * i + 1];
            double km1 = k2 * m[3 * i + 0] - k0 * m[3 * i + 2];
            double km2 = k0 * m[3 * i + 1] - k1 * m[3 * i + 0];
            
            // kdotm
            double kdotm = k0 * m[3 * i + 0] + k1 * m[3 * i + 1] + k2 * m[3 * i + 2];
            
            double mn0 = ((1.0 - k_sq) * m[3 * i + 0] + 2.0 * km0 + 2.0 * kdotm * k0) / denom;
            double mn1 = ((1.0 - k_sq) * m[3 * i + 1] + 2.0 * km1 + 2.0 * kdotm * k1) / denom;
            double mn2 = ((1.0 - k_sq) * m[3 * i + 2] + 2.0 * km2 + 2.0 * kdotm * k2) / denom;
            
            double norm = std::sqrt(mn0 * mn0 + mn1 * mn1 + mn2 * mn2);
            if (norm > 0.0) {
                mn0 /= norm;
                mn1 /= norm;
                mn2 /= norm;
            }
            m_new[3 * i + 0] = mn0;
            m_new[3 * i + 1] = mn1;
            m_new[3 * i + 2] = mn2;
        }
    }

    void cayley_transport(int N, const double* v, const double* H, double tau, double* v_new) {
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            double k0 = 0.5 * tau * H[3 * i + 0];
            double k1 = 0.5 * tau * H[3 * i + 1];
            double k2 = 0.5 * tau * H[3 * i + 2];
            double k_sq = k0 * k0 + k1 * k1 + k2 * k2;
            double denom = 1.0 + k_sq;
            
            // kv = k x v
            double kv0 = k1 * v[3 * i + 2] - k2 * v[3 * i + 1];
            double kv1 = k2 * v[3 * i + 0] - k0 * v[3 * i + 2];
            double kv2 = k0 * v[3 * i + 1] - k1 * v[3 * i + 0];
            
            // kdotv
            double kdotv = k0 * v[3 * i + 0] + k1 * v[3 * i + 1] + k2 * v[3 * i + 2];
            
            v_new[3 * i + 0] = ((1.0 - k_sq) * v[3 * i + 0] + 2.0 * kv0 + 2.0 * kdotv * k0) / denom;
            v_new[3 * i + 1] = ((1.0 - k_sq) * v[3 * i + 1] + 2.0 * kv1 + 2.0 * kdotv * k1) / denom;
            v_new[3 * i + 2] = ((1.0 - k_sq) * v[3 * i + 2] + 2.0 * kv2 + 2.0 * kdotv * k2) / denom;
        }
    }

    void tangent_grad(int N, const double* m, const double* g_raw, double* g_tan) {
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            double dot = m[3 * i + 0] * g_raw[3 * i + 0] +
                         m[3 * i + 1] * g_raw[3 * i + 1] +
                         m[3 * i + 2] * g_raw[3 * i + 2];
            g_tan[3 * i + 0] = g_raw[3 * i + 0] - dot * m[3 * i + 0];
            g_tan[3 * i + 1] = g_raw[3 * i + 1] - dot * m[3 * i + 1];
            g_tan[3 * i + 2] = g_raw[3 * i + 2] - dot * m[3 * i + 2];
        }
    }

    void evaluate_energy_and_grad(
        int N, const double* m, const double* U, const double* B_ext, const double* M_nodal,
        sparse_matrix_t K_eff, sparse_matrix_t G_sparse, double* E, double* g_total, double inv_Vmag
    ) {
        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, K_eff, descr, m, 0.0, g_total);
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, G_sparse, descr, U, 1.0, g_total);
        
        double energy = 0.0;
        #pragma omp parallel for reduction(+:energy)
        for (int i = 0; i < N; ++i) {
            double gz0 = -2.0 * M_nodal[i] * B_ext[0];
            double gz1 = -2.0 * M_nodal[i] * B_ext[1];
            double gz2 = -2.0 * M_nodal[i] * B_ext[2];

            g_total[3 * i + 0] += gz0;
            g_total[3 * i + 1] += gz1;
            g_total[3 * i + 2] += gz2;

            energy += m[3 * i + 0] * (g_total[3 * i + 0] + gz0);
            energy += m[3 * i + 1] * (g_total[3 * i + 1] + gz1);
            energy += m[3 * i + 2] * (g_total[3 * i + 2] + gz2);
        }
        
        *E = 0.5 * energy * inv_Vmag;
        cblas_dscal(3 * N, inv_Vmag, g_total, 1);
    }

    void solve_poisson(int N, const double* m, int64_t pardiso_handle, sparse_matrix_t D_sparse, double* U, const double* boundary_mask) {
        if (pardiso_handle == 0) return;
        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        
        double* rhs = new double[N]();
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, D_sparse, descr, m, 0.0, rhs);
        
        if (boundary_mask != nullptr) {
            for (int i = 0; i < N; ++i) {
                rhs[i] *= boundary_mask[i];
            }
        }
        
        pardiso_solve_direct(pardiso_handle, rhs, U);
        
        if (boundary_mask != nullptr) {
            for (int i = 0; i < N; ++i) {
                U[i] *= boundary_mask[i];
            }
        }
        
        delete[] rhs;
    }

    void solve_Py_g(
        int N,
        const double* m,
        const double* g_ext,
        const double* g_tan_ext,
        const double* inv_M_prec,
        sparse_matrix_t K_eff,
        int max_iter,
        double tol,
        double reg,
        double stagnation_nu,
        double inv_Vmag,
        double* y,
        int* out_preco_it
    ) {
        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;

        double* r = new double[3 * N];
        double* z = new double[3 * N];
        double* p = new double[3 * N];
        double* Ap = new double[3 * N];
        double* Kp = new double[3 * N];

        std::fill(y, y + 3 * N, 0.0);
        std::copy(g_tan_ext, g_tan_ext + 3 * N, r);

        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            double factor = inv_M_prec[i];
            z[3 * i + 0] = r[3 * i + 0] * factor;
            z[3 * i + 1] = r[3 * i + 1] * factor;
            z[3 * i + 2] = r[3 * i + 2] * factor;
            p[3 * i + 0] = z[3 * i + 0];
            p[3 * i + 1] = z[3 * i + 1];
            p[3 * i + 2] = z[3 * i + 2];
        }

        double rho = cblas_ddot(3 * N, r, 1, z, 1);
        double target_rho = (tol * tol) * rho;
        double Q = 0.0;
        bool done = false;

        int it_loop = 0;
        for (it_loop = 0; it_loop < max_iter; ++it_loop) {
            if (rho <= target_rho || rho <= 1e-25 || rho >= 1e20 || done) {
                break;
            }

            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, K_eff, descr, p, 0.0, Kp);

            #pragma omp parallel for
            for (int i = 0; i < N; ++i) {
                double Cv0 = Kp[3 * i + 0] * inv_Vmag;
                double Cv1 = Kp[3 * i + 1] * inv_Vmag;
                double Cv2 = Kp[3 * i + 2] * inv_Vmag;

                double m_dot_Cv = m[3 * i + 0] * Cv0 + m[3 * i + 1] * Cv1 + m[3 * i + 2] * Cv2;
                double m_dot_g = m[3 * i + 0] * g_ext[3 * i + 0] + m[3 * i + 1] * g_ext[3 * i + 1] + m[3 * i + 2] * g_ext[3 * i + 2];

                Ap[3 * i + 0] = Cv0 - m_dot_Cv * m[3 * i + 0] - m_dot_g * p[3 * i + 0] + reg * p[3 * i + 0];
                Ap[3 * i + 1] = Cv1 - m_dot_Cv * m[3 * i + 1] - m_dot_g * p[3 * i + 1] + reg * p[3 * i + 1];
                Ap[3 * i + 2] = Cv2 - m_dot_Cv * m[3 * i + 2] - m_dot_g * p[3 * i + 2] + reg * p[3 * i + 2];
            }

            double pAp = cblas_ddot(3 * N, p, 1, Ap, 1);
            bool neg_curv = pAp <= 0.0;
            double alpha = rho / (pAp + 1e-30);

            double dq = 0.5 * alpha * rho;
            bool stagnated = (it_loop > 0) && (dq <= stagnation_nu * Q);

            done = neg_curv || stagnated;

            if (done) {
                break;
            }

            cblas_daxpy(3 * N, alpha, p, 1, y, 1);
            cblas_daxpy(3 * N, -alpha, Ap, 1, r, 1);

            #pragma omp parallel for
            for (int i = 0; i < N; ++i) {
                double factor = inv_M_prec[i];
                z[3 * i + 0] = r[3 * i + 0] * factor;
                z[3 * i + 1] = r[3 * i + 1] * factor;
                z[3 * i + 2] = r[3 * i + 2] * factor;
            }

            double rho_next = cblas_ddot(3 * N, r, 1, z, 1);
            double beta = rho_next / (rho + 1e-30);

            cblas_daxpby(3 * N, 1.0, z, 1, beta, p, 1);

            rho = rho_next;
            Q += dq;
        }

        *out_preco_it = it_loop;

        double y_norm = cblas_dnrm2(3 * N, y, 1);
        double* z_fallback = new double[3 * N];
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            double factor = inv_M_prec[i];
            z_fallback[3 * i + 0] = g_tan_ext[3 * i + 0] * factor;
            z_fallback[3 * i + 1] = g_tan_ext[3 * i + 1] * factor;
            z_fallback[3 * i + 2] = g_tan_ext[3 * i + 2] * factor;
        }
        double z_norm = cblas_dnrm2(3 * N, z_fallback, 1);

        if (y_norm > 10.0 * z_norm) {
            cblas_dscal(3 * N, (10.0 * z_norm) / (y_norm + 1e-30), y, 1);
        }

        double y_dot_g = cblas_ddot(3 * N, y, 1, g_tan_ext, 1);
        if (y_dot_g <= 1e-12) {
            std::copy(z_fallback, z_fallback + 3 * N, y);
        }

        delete[] r;
        delete[] z;
        delete[] p;
        delete[] Ap;
        delete[] Kp;
        delete[] z_fallback;
    }

    void armijo_ls(
        int N,
        const double* m,
        double pg,
        const double* H,
        double E0,
        const double* U_base,
        const double* g_raw_init,
        const double* B_ext,
        const double* M_nodal,
        sparse_matrix_t K_eff,
        sparse_matrix_t G_sparse,
        sparse_matrix_t D_sparse,
        int64_t pardiso_handle,
        const double* boundary_mask,
        double eta1,
        double eta2,
        double C,
        double c,
        double s0,
        int max_evals,
        double inv_Vmag,
        double* s_out,
        double* E_out,
        double* g_raw_out,
        double* U_out,
        double* m_out,
        int* evals_out,
        int* demag_out
    ) {
        if (pg >= 0.0) {
            *s_out = 0.0;
            *E_out = E0;
            std::copy(m, m + 3 * N, m_out);
            std::copy(U_base, U_base + N, U_out);
            std::copy(g_raw_init, g_raw_init + 3 * N, g_raw_out);
            *evals_out = 0;
            *demag_out = 0;
            return;
        }

        double s = s0;
        double s_min = 0.0;
        int it_exp = 0;
        bool done_exp = false;

        double* m_trial = new double[3 * N];
        double* U_trial = new double[N];
        double* g_trial = new double[3 * N];

        double E_exp = E0;
        double* g_raw_exp = new double[3 * N];
        double* U_exp = new double[N];
        double* m_exp = new double[3 * N];
        double d_exp = 0.0;
        int demag_accum = 0;

        std::copy(g_raw_init, g_raw_init + 3 * N, g_raw_exp);
        std::copy(U_base, U_base + N, U_exp);
        std::copy(m, m + 3 * N, m_exp);

        // Expansion loop
        while (it_exp < max_evals && !done_exp) {
            cayley_update(N, m, H, s, m_trial);
            solve_poisson(N, m_trial, pardiso_handle, D_sparse, U_trial, boundary_mask);
            demag_accum++;

            double E_trial = 0.0;
            evaluate_energy_and_grad(N, m_trial, U_trial, B_ext, M_nodal, K_eff, G_sparse, &E_trial, g_trial, inv_Vmag);

            if (!std::isfinite(E_trial)) {
                E_trial = 1e20;
            }

            double d_next = (E_trial - E0) / (s * pg + 1e-30);
            bool stop = (std::abs(1.0 - d_next) >= eta2) || (d_next < 0.0);

            double s_next = stop ? s : C * s;
            double s_min_next = stop ? s_min : s;

            E_exp = E_trial;
            std::copy(g_trial, g_trial + 3 * N, g_raw_exp);
            std::copy(U_trial, U_trial + N, U_exp);
            std::copy(m_trial, m_trial + 3 * N, m_exp);
            d_exp = d_next;

            s = s_next;
            s_min = s_min_next;
            it_exp++;
            done_exp = stop;
        }

        // Contraction loop
        bool con_done_init = (d_exp >= eta1) && (d_exp < 1e10);
        double s_final = s;
        double E_final = E_exp;
        double* g_raw_final = new double[3 * N];
        double* U_final = new double[N];
        double* m_final = new double[3 * N];
        double d_final = d_exp;
        int it_con = 0;

        std::copy(g_raw_exp, g_raw_exp + 3 * N, g_raw_final);
        std::copy(U_exp, U_exp + N, U_final);
        std::copy(m_exp, m_exp + 3 * N, m_final);

        bool done_con = con_done_init;
        while (it_con < max_evals && !done_con) {
            double s_next = s_min + c * (s_final - s_min);
            cayley_update(N, m, H, s_next, m_trial);
            solve_poisson(N, m_trial, pardiso_handle, D_sparse, U_trial, boundary_mask);
            demag_accum++;

            double E_trial = 0.0;
            evaluate_energy_and_grad(N, m_trial, U_trial, B_ext, M_nodal, K_eff, G_sparse, &E_trial, g_trial, inv_Vmag);

            if (!std::isfinite(E_trial)) {
                E_trial = 1e20;
            }

            double d_next = (E_trial - E0) / (s_next * pg + 1e-30);
            bool stop = (d_next >= eta1) && (d_next < 1e10);

            E_final = E_trial;
            std::copy(g_trial, g_trial + 3 * N, g_raw_final);
            std::copy(U_trial, U_trial + N, U_final);
            std::copy(m_trial, m_trial + 3 * N, m_final);
            d_final = d_next;
            s_final = s_next;

            it_con++;
            done_con = stop;
        }

        bool is_safe = d_final >= 0.0;
        if (is_safe) {
            *s_out = s_final;
            *E_out = E_final;
            std::copy(m_final, m_final + 3 * N, m_out);
            std::copy(U_final, U_final + N, U_out);
            std::copy(g_raw_final, g_raw_final + 3 * N, g_raw_out);
        } else {
            *s_out = 0.0;
            *E_out = E0;
            std::copy(m, m + 3 * N, m_out);
            std::copy(U_base, U_base + N, U_out);
            std::copy(g_raw_init, g_raw_init + 3 * N, g_raw_out);
        }

        *evals_out = it_exp + it_con;
        *demag_out = demag_accum;

        delete[] m_trial;
        delete[] U_trial;
        delete[] g_trial;
        delete[] g_raw_exp;
        delete[] U_exp;
        delete[] m_exp;
        delete[] g_raw_final;
        delete[] U_final;
        delete[] m_final;
    }

    bool check_convergence(
        int it, double E, double E_prev, int N, const double* m, const double* m_new,
        double gnorm_inf, double tau_f, double eps_a
    ) {
        double diff_m_norm_inf = 0.0;
        for (int i = 0; i < 3 * N; ++i) {
            diff_m_norm_inf = std::max(diff_m_norm_inf, std::abs(m_new[i] - m[i]));
        }
        double m_norm_inf = 1.0;

        bool u1 = (E_prev - E) < tau_f * (1.0 + std::abs(E));
        bool u2 = diff_m_norm_inf < std::sqrt(tau_f) * (1.0 + m_norm_inf);
        bool u3 = gnorm_inf <= (std::pow(tau_f, 1.0 / 3.0)) * (1.0 + std::abs(E));
        bool u4 = gnorm_inf < eps_a;

        return (it > 0) ? ((u1 && u2 && u3) || u4) : false;
    }

    int run_cpp_pcohen_hs_minimization(
        int N, int max_iter, double tau_f, double eps_a, double eta1, double eta2, 
        double C, double c, int max_ls_evals, double inv_Vmag,
        int64_t pardiso_handle,
        double* m, const double* B_ext, double* U, const double* M_nodal,
        const double* boundary_mask,
        double* K_val, int* K_col, int* K_ptr,
        double* G_val, int* G_col, int* G_ptr,
        double* D_val, int* D_col, int* D_ptr,
        const double* inv_M_rel, const double* inv_M_prec,
        int pc_auto, double pc_force_eta, double pc_force_alpha,
        double cg_tol, int pc_iters, double pc_reg, double pc_stagnation_nu,
        int L, int beta_type,
        int* out_iters, int* out_evals, int* out_demag, int* out_preco, double* out_E, double* out_gnorm
    ) {
        sparse_matrix_t K_eff, G_sparse, D_sparse;
        mkl_sparse_d_create_csr(&K_eff, SPARSE_INDEX_BASE_ZERO, 3*N, 3*N, K_ptr, K_ptr+1, K_col, K_val);
        mkl_sparse_d_create_csr(&G_sparse, SPARSE_INDEX_BASE_ZERO, 3*N, N, G_ptr, G_ptr+1, G_col, G_val);
        mkl_sparse_d_create_csr(&D_sparse, SPARSE_INDEX_BASE_ZERO, N, 3*N, D_ptr, D_ptr+1, D_col, D_val);
        
        mkl_sparse_optimize(K_eff); mkl_sparse_optimize(G_sparse); mkl_sparse_optimize(D_sparse);

        // Normalize m
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            double norm = std::sqrt(m[3*i]*m[3*i] + m[3*i+1]*m[3*i+1] + m[3*i+2]*m[3*i+2]);
            if (norm > 0.0) {
                m[3*i] /= norm; m[3*i+1] /= norm; m[3*i+2] /= norm;
            }
        }

        double* g_raw = new double[3 * N];
        double* g_prev = new double[3 * N];
        double* y_prev = new double[3 * N];
        double* d_prev = new double[3 * N];
        double* H = new double[3 * N];
        
        double E = 0.0;
        solve_poisson(N, m, pardiso_handle, D_sparse, U, boundary_mask);
        evaluate_energy_and_grad(N, m, U, B_ext, M_nodal, K_eff, G_sparse, &E, g_raw, inv_Vmag);
        
        // Print debug info
        double sum_g = 0.0;
        #pragma omp parallel for reduction(+:sum_g)
        for (int i = 0; i < 3 * N; ++i) sum_g += g_raw[i];
        // std::cout << "C++ initial E: " << E << std::endl;
        // std::cout << "C++ initial g_raw mean: " << sum_g / (3.0 * N) << std::endl;

        double* g_tan = new double[3 * N];
        double* g_tan_ext = new double[3 * N];
        
        double* g_raw_scaled = new double[3 * N];
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            g_raw_scaled[3*i] = g_raw[3*i] * inv_M_rel[i];
            g_raw_scaled[3*i+1] = g_raw[3*i+1] * inv_M_rel[i];
            g_raw_scaled[3*i+2] = g_raw[3*i+2] * inv_M_rel[i];
        }
        
        tangent_grad(N, m, g_raw_scaled, g_tan);
        tangent_grad(N, m, g_raw, g_tan_ext);
        
        double gnorm_inf = std::abs(g_tan[cblas_idamax(3 * N, g_tan, 1)]);
        
        double pc_tol = 0.0;
        if (pc_auto) {
            pc_tol = std::min(pc_force_eta, std::pow(gnorm_inf, pc_force_alpha));
        }
        int initial_preco_it = 0;
        solve_Py_g(N, m, g_raw, g_tan_ext, inv_M_prec, K_eff, pc_iters, pc_tol, pc_reg, pc_stagnation_nu, inv_Vmag, y_prev, &initial_preco_it);

        std::copy(g_tan_ext, g_tan_ext + 3 * N, g_prev);
        #pragma omp parallel for
        for (int i = 0; i < 3 * N; ++i) {
            d_prev[i] = -y_prev[i];
        }
        
        double tau = 1.0;
        double E_prev = E;
        int evals = 1;
        int demag = 1;
        int preco_iters = initial_preco_it;
        double final_gnorm_inf = gnorm_inf;
        double gnorm_inf_smooth = std::abs(y_prev[cblas_idamax(3 * N, y_prev, 1)]);
        bool converged = false;
        
        double* d = new double[3 * N];
        double* y = new double[3 * N];
        double* d_prev_proj = new double[3 * N];

        int it = 0;
        for (it = 0; it < max_iter; ++it) {
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) {
                g_raw_scaled[3*i] = g_raw[3*i] * inv_M_rel[i];
                g_raw_scaled[3*i+1] = g_raw[3*i+1] * inv_M_rel[i];
                g_raw_scaled[3*i+2] = g_raw[3*i+2] * inv_M_rel[i];
            }
            tangent_grad(N, m, g_raw_scaled, g_tan);
            tangent_grad(N, m, g_raw, g_tan_ext);
            
            gnorm_inf = std::abs(g_tan[cblas_idamax(3 * N, g_tan, 1)]);
            final_gnorm_inf = gnorm_inf;

            if (converged) {
                break;
            }

            double pc_tol = 0.0;
            if (pc_auto) {
                pc_tol = std::min(pc_force_eta, std::pow(gnorm_inf, pc_force_alpha));
            }
            
            int current_preco_it = 0;
            solve_Py_g(N, m, g_raw, g_tan_ext, inv_M_prec, K_eff, pc_iters, pc_tol, pc_reg, pc_stagnation_nu, inv_Vmag, y, &current_preco_it);
            preco_iters += current_preco_it;
            
            gnorm_inf_smooth = std::abs(y[cblas_idamax(3 * N, y, 1)]);
            
            double beta = 0.0;
            int L_val = (L > 0) ? L : N;
            if (it % L_val != 0) {
                double num = 0.0;
                double den = 0.0;
                if (beta_type == 0) { // PR
                    #pragma omp parallel for reduction(+:num, den)
                    for (int i = 0; i < 3 * N; ++i) {
                        num += y[i] * (g_tan_ext[i] - g_prev[i]);
                        den += y_prev[i] * g_prev[i];
                    }
                } else { // HS
                    #pragma omp parallel for reduction(+:num, den)
                    for (int i = 0; i < 3 * N; ++i) {
                        double diff_g = g_tan_ext[i] - g_prev[i];
                        num += y[i] * diff_g;
                        den += d_prev[i] * diff_g;
                    }
                }
                den += 1e-30;
                beta = std::max(0.0, num / den);
            }
            
            tangent_grad(N, m, d_prev, d_prev_proj);
            
            #pragma omp parallel for
            for (int i = 0; i < 3 * N; ++i) {
                d[i] = -y[i] + beta * d_prev_proj[i];
            }
            
            double d_dot_gtan = cblas_ddot(3 * N, d, 1, g_tan_ext, 1);
            if (d_dot_gtan > 0.0) {
                #pragma omp parallel for
                for (int i = 0; i < 3 * N; ++i) {
                    d[i] = -y[i];
                }
            }
            
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) {
                H[3 * i + 0] = m[3 * i + 1] * d[3 * i + 2] - m[3 * i + 2] * d[3 * i + 1];
                H[3 * i + 1] = m[3 * i + 2] * d[3 * i + 0] - m[3 * i + 0] * d[3 * i + 2];
                H[3 * i + 2] = m[3 * i + 0] * d[3 * i + 1] - m[3 * i + 1] * d[3 * i + 0];
            }
            
            double pg = cblas_ddot(3 * N, g_raw, 1, d, 1);
            
            double* m_new = new double[3 * N];
            double* U_new = new double[N];
            double* g_raw_new = new double[3 * N];
            double E_new = E;
            int ls_evals = 0;
            int ls_demag = 0;
            
            armijo_ls(N, m, pg, H, E, U, g_raw, B_ext, M_nodal, K_eff, G_sparse, D_sparse, pardiso_handle,
                      boundary_mask,
                      eta1, eta2, C, c, 1.0, max_ls_evals, inv_Vmag,
                      &tau, &E_new, g_raw_new, U_new, m_new, &ls_evals, &ls_demag);
            
            evals += ls_evals;
            demag += ls_demag;
            
            converged = check_convergence(it, E_new, E_prev, N, m, m_new, gnorm_inf_smooth, tau_f, eps_a);
            
            E_prev = E;
            E = E_new;
            std::copy(g_tan_ext, g_tan_ext + 3 * N, g_prev);
            std::copy(y, y + 3 * N, y_prev);
            std::copy(d, d + 3 * N, d_prev);
            std::copy(m_new, m_new + 3 * N, m);
            std::copy(U_new, U_new + N, U);
            std::copy(g_raw_new, g_raw_new + 3 * N, g_raw);
            
            delete[] m_new;
            delete[] U_new;
            delete[] g_raw_new;
        }
        
        *out_iters = it;
        *out_evals = evals;
        *out_demag = demag;
        if (out_preco) *out_preco = preco_iters;
        if (out_E) *out_E = E;
        if (out_gnorm) *out_gnorm = final_gnorm_inf;
        
        delete[] g_raw;
        delete[] g_prev;
        delete[] y_prev;
        delete[] d_prev;
        delete[] H;
        delete[] g_tan;
        delete[] g_tan_ext;
        delete[] g_raw_scaled;
        delete[] d;
        delete[] y;
        delete[] d_prev_proj;
        
        mkl_sparse_destroy(K_eff);
        mkl_sparse_destroy(G_sparse);
        mkl_sparse_destroy(D_sparse);
        
        return 0;
    }
}
