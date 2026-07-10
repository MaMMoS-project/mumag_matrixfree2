
#include "xla/ffi/api/ffi.h"
#include "mkl_spblas.h"
#include <mkl.h>
#include <Python.h>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <vector>

// --- SpMV Implementation ---

xla::ffi::Error MklSpmvFFI(
    xla::ffi::Buffer<xla::ffi::DataType::F64> x,
    xla::ffi::Buffer<xla::ffi::DataType::S64> handle_buf,
    xla::ffi::ResultBuffer<xla::ffi::DataType::F64> out
) {
    int64_t handle_addr = *handle_buf.typed_data();
    sparse_matrix_t mkl_a = reinterpret_cast<sparse_matrix_t>(handle_addr);
    const double* x_ptr = x.typed_data();
    double* y_ptr = out->typed_data();

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    sparse_status_t status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, mkl_a, descr, x_ptr, 0.0, y_ptr);
    if (status != SPARSE_STATUS_SUCCESS) {
        return xla::ffi::Error::InvalidArgument("MKL SpMV call returned non-success status");
    }
    return xla::ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mkl_spmv_handler, MklSpmvFFI,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::Buffer<xla::ffi::DataType::F64>>()
        .Arg<xla::ffi::Buffer<xla::ffi::DataType::S64>>()
        .Ret<xla::ffi::Buffer<xla::ffi::DataType::F64>>()
);

// --- PARDISO Implementation ---

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
} // extern "C"

xla::ffi::Error PardisoSolveFFI(
    xla::ffi::Buffer<xla::ffi::DataType::F64> b,
    xla::ffi::Buffer<xla::ffi::DataType::S64> handle_buf,
    xla::ffi::ResultBuffer<xla::ffi::DataType::F64> out
) {
    int64_t handle_id = *handle_buf.typed_data();
    int error = pardiso_solve_direct(handle_id, b.typed_data(), out->typed_data());
    if (error != 0) {
        return xla::ffi::Error::Internal("PARDISO solve failed");
    }
    return xla::ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pardiso_solve_handler, PardisoSolveFFI,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::Buffer<xla::ffi::DataType::F64>>()
        .Arg<xla::ffi::Buffer<xla::ffi::DataType::S64>>()
        .Ret<xla::ffi::Buffer<xla::ffi::DataType::F64>>()
);

// --- JAX MKL PCG/AMG Implementation ---

struct AmgLevel {
    int n;
    int n_c; // coarse size (only valid if not the last level)
    sparse_matrix_t A;
    sparse_matrix_t P;
    sparse_matrix_t R;
    double* Mdiag_spai0;
    
    // Temporaries
    double* r;
    double* b_c;
    double* x_c;
    
    // PARDISO handle for the last level
    int64_t pardiso_id;
};

struct AmgState {
    std::vector<AmgLevel> levels;
    int maxiter;
    double tol;
    
    // PCG temporaries (size 4*n for MKL RCI)
    double* tmp; 
    
    ~AmgState() {
        if (tmp) {
            delete[] tmp;
        }
        for (auto& lvl : levels) {
            if (lvl.A) mkl_sparse_destroy(lvl.A);
            if (lvl.P) mkl_sparse_destroy(lvl.P);
            if (lvl.R) mkl_sparse_destroy(lvl.R);
            if (lvl.r) delete[] lvl.r;
            if (lvl.b_c) delete[] lvl.b_c;
            if (lvl.x_c) delete[] lvl.x_c;
            if (lvl.pardiso_id > 0) free_pardiso(lvl.pardiso_id);
        }
    }
};

static std::unordered_map<int64_t, AmgState*> amg_states;
static std::mutex amg_mutex;
static int64_t next_amg_id = 1;

extern "C" {
int64_t init_amg_state(int maxiter, double tol) {
    AmgState* state = new AmgState();
    state->maxiter = maxiter;
    state->tol = tol;
    state->tmp = nullptr;
    
    std::lock_guard<std::mutex> lock(amg_mutex);
    int64_t id = next_amg_id++;
    amg_states[id] = state;
    return id;
}

void free_amg_state(int64_t id) {
    std::lock_guard<std::mutex> lock(amg_mutex);
    auto it = amg_states.find(id);
    if (it != amg_states.end()) {
        delete it->second;
        amg_states.erase(it);
    }
}

int add_amg_level(int64_t id, int n, int n_c,
                  double* a_val, int* a_col, int* a_ptr,
                  double* p_val, int* p_col, int* p_ptr,
                  double* r_val, int* r_col, int* r_ptr,
                  double* mdiag_spai0, int64_t pardiso_id) {
    std::lock_guard<std::mutex> lock(amg_mutex);
    auto it = amg_states.find(id);
    if (it == amg_states.end()) return -1;
    AmgState* state = it->second;
    
    AmgLevel lvl;
    lvl.n = n;
    lvl.n_c = n_c;
    lvl.A = nullptr;
    lvl.P = nullptr;
    lvl.R = nullptr;
    lvl.r = nullptr;
    lvl.b_c = nullptr;
    lvl.x_c = nullptr;
    lvl.Mdiag_spai0 = mdiag_spai0;
    lvl.pardiso_id = pardiso_id;
    
    if (a_val) {
        mkl_sparse_d_create_csr(&lvl.A, SPARSE_INDEX_BASE_ZERO, n, n, a_ptr, a_ptr + 1, a_col, a_val);
        mkl_sparse_optimize(lvl.A);
    }
    if (p_val) {
        mkl_sparse_d_create_csr(&lvl.P, SPARSE_INDEX_BASE_ZERO, n, n_c, p_ptr, p_ptr + 1, p_col, p_val);
        mkl_sparse_optimize(lvl.P);
    }
    if (r_val) {
        mkl_sparse_d_create_csr(&lvl.R, SPARSE_INDEX_BASE_ZERO, n_c, n, r_ptr, r_ptr + 1, r_col, r_val);
        mkl_sparse_optimize(lvl.R);
    }
    
    if (n_c > 0 && pardiso_id == 0) { // Not the last level
        lvl.r = new double[n];
        lvl.b_c = new double[n_c];
        lvl.x_c = new double[n_c];
    }
    
    state->levels.push_back(lvl);
    return 0;
}

int finalize_amg_state(int64_t id) {
    std::lock_guard<std::mutex> lock(amg_mutex);
    auto it = amg_states.find(id);
    if (it == amg_states.end()) return -1;
    AmgState* state = it->second;
    
    if (!state->levels.empty()) {
        int n_finest = state->levels[0].n;
        state->tmp = new double[4 * n_finest];
    }
    return 0;
}
} // extern "C"

// Forward declaration of recursive v-cycle
void amg_vcycle_recursive(AmgState* state, int level_idx, const double* b, double* x);

void amg_vcycle_recursive(AmgState* state, int level_idx, const double* b, double* x) {
    AmgLevel& lvl = state->levels[level_idx];
    
    if (level_idx == state->levels.size() - 1) {
        int err = pardiso_solve_direct(lvl.pardiso_id, b, x);
        if (err != 0) {
            std::cerr << "PARDISO solve failed with error " << err << std::endl;
        }
        return;
    }
    
    int n = lvl.n;
    int n_c = lvl.n_c;
    
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    
    // Initialize x = 0 (assuming zero initial guess for the v-cycle correction)
    // Wait, in make_jax_amgcl_vcycle, x is usually 0, but spai0_smooth does: x = x + M * (b - Ax). 
    // If x is 0, then x = M * b.
    // So pre-smoothing:
    for (int i = 0; i < n; i++) {
        x[i] = lvl.Mdiag_spai0[i] * b[i];
    }
    
    // Restriction: r = b - A*x
    // First, r = b
    cblas_dcopy(n, b, 1, lvl.r, 1);
    // r = r - A*x
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, lvl.A, descr, x, 1.0, lvl.r);
    
    // b_c = R * r
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, lvl.R, descr, lvl.r, 0.0, lvl.b_c);
    
    // Coarse Grid Solve
    // initialize x_c to 0
    for(int i=0; i<n_c; i++) lvl.x_c[i] = 0.0;
    amg_vcycle_recursive(state, level_idx + 1, lvl.b_c, lvl.x_c);
    
    // Prolongation: x = x + P * x_c
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, lvl.P, descr, lvl.x_c, 1.0, x);
    
    // Post-smoothing: r = b - A*x
    cblas_dcopy(n, b, 1, lvl.r, 1);
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1.0, lvl.A, descr, x, 1.0, lvl.r);
    
    // x = x + M * r
    for (int i = 0; i < n; i++) {
        x[i] += lvl.Mdiag_spai0[i] * lvl.r[i];
    }
}


xla::ffi::Error JaxMklSolveFFI(
    xla::ffi::Buffer<xla::ffi::DataType::F64> b,
    xla::ffi::Buffer<xla::ffi::DataType::S64> handle_buf,
    xla::ffi::ResultBuffer<xla::ffi::DataType::F64> out
) {
    int64_t handle_id = *handle_buf.typed_data();
    
    AmgState* state = nullptr;
    {
        std::lock_guard<std::mutex> lock(amg_mutex);
        auto it = amg_states.find(handle_id);
        if (it == amg_states.end()) {
            return xla::ffi::Error::InvalidArgument("Invalid AMG handle ID");
        }
        state = it->second;
    }
    
    if (state->levels.empty()) {
        return xla::ffi::Error::InvalidArgument("AMG state has no levels");
    }
    
    const double* b_ptr = b.typed_data();
    double* x_ptr = out->typed_data();
    
    MKL_INT n = state->levels[0].n;
    
    // Initialize x_ptr to 0
    for(int i=0; i<n; i++) x_ptr[i] = 0.0;
    
    MKL_INT rci_request = 0;
    MKL_INT ipar[128] = {0};
    double dpar[128] = {0.0};
    double* tmp = state->tmp;
    
    dcg_init(&n, x_ptr, b_ptr, &rci_request, ipar, dpar, tmp);
    
    ipar[4] = state->maxiter;
    ipar[7] = 1; // Do maximum iterations check
    ipar[8] = 0; // Do not do residual check, we do it ourselves
    ipar[9] = 1; // User defined stopping test
    ipar[10] = 1; // Preconditioned CG
    dpar[0] = state->tol;
    
    dcg_check(&n, x_ptr, b_ptr, &rci_request, ipar, dpar, tmp);
    
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    
    while (true) {
        dcg(&n, x_ptr, b_ptr, &rci_request, ipar, dpar, tmp);
        
        if (rci_request == 0) {
            break; // Convergence reached or max iter
        } else if (rci_request == 1) {
            // SpMV: A * tmp[0:n] -> tmp[n:2n]
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, state->levels[0].A, descr, &tmp[0], 0.0, &tmp[n]);
        } else if (rci_request == 3) {
            // Preconditioner: M^-1 * tmp[2n:3n] -> tmp[3n:4n]
            for(int i=0; i<n; i++) tmp[3*n + i] = 0.0;
            amg_vcycle_recursive(state, 0, &tmp[2 * n], &tmp[3 * n]);
        } else if (rci_request == 2) {
            // User defined stopping test
            double r_norm = cblas_dnrm2(n, &tmp[2*n], 1);
            double b_norm = cblas_dnrm2(n, b_ptr, 1);
            if (b_norm == 0.0) b_norm = 1.0; // avoid division by zero if b=0
            if (r_norm < state->tol * b_norm) {
                break;
            } else {
                continue;
            }
        } else {
            std::cerr << "MKL Error rci_request = " << rci_request << std::endl;
            break; // Unexpected or error
        }
    }
    
    return xla::ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    jax_mkl_solve_handler, JaxMklSolveFFI,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::Buffer<xla::ffi::DataType::F64>>()
        .Arg<xla::ffi::Buffer<xla::ffi::DataType::S64>>()
        .Ret<xla::ffi::Buffer<xla::ffi::DataType::F64>>()
);

// --- Registration ---

static PyObject* get_registrations(PyObject* self, PyObject* args) {
    PyObject* dict = PyDict_New();
    
    PyObject* cap_spmv = PyCapsule_New(reinterpret_cast<void*>(mkl_spmv_handler), "xla_ffi", nullptr);
    PyDict_SetItemString(dict, "mkl_spmv_ffi", cap_spmv);
    Py_DECREF(cap_spmv);
    
    PyObject* cap_pardiso = PyCapsule_New(reinterpret_cast<void*>(pardiso_solve_handler), "xla_ffi", nullptr);
    PyDict_SetItemString(dict, "pardiso_solve_ffi", cap_pardiso);
    Py_DECREF(cap_pardiso);
    
    PyObject* cap_jax_mkl = PyCapsule_New(reinterpret_cast<void*>(jax_mkl_solve_handler), "xla_ffi", nullptr);
    PyDict_SetItemString(dict, "jax_mkl_solve_ffi", cap_jax_mkl);
    Py_DECREF(cap_jax_mkl);
    
    return dict;
}

static PyMethodDef MklFfiMethods[] = {
    {"get_registrations", get_registrations, METH_NOARGS, "Get JAX FFI registrations"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mkl_ffi_module = {
    PyModuleDef_HEAD_INIT,
    "mkl_ffi_lib",
    "MKL JAX FFI Library",
    -1,
    MklFfiMethods
};

PyMODINIT_FUNC PyInit_mkl_ffi_lib(void) {
    return PyModule_Create(&mkl_ffi_module);
}
