#include <mkl.h>
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

} // extern "C"
