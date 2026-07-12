import numpy as np
import scipy.sparse as sp
import ctypes
import ctypes.util
import ctypes
from sparse_dot_mkl._mkl_interface import _create_mkl_sparse, _destroy_mkl_handle, matrix_descr, _output_dtypes, _mkl_scalar, _out_matrix, MKL as MKL_wrapper

mkl_lib_path = ctypes.util.find_library("mkl_rt")
if not mkl_lib_path:
    # Try direct load if in LD_LIBRARY_PATH or conda env
    mkl_lib_path = "libmkl_rt.so"
libmkl = ctypes.cdll.LoadLibrary(mkl_lib_path)

# Bind mkl_sparse_optimize
libmkl.mkl_sparse_optimize.argtypes = [ctypes.c_void_p]
libmkl.mkl_sparse_optimize.restype = ctypes.c_int

libmkl.mkl_sparse_set_mv_hint.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
libmkl.mkl_sparse_set_mv_hint.restype = ctypes.c_int

def test_inspector_executor():
    # Create test matrix and vector
    N = 1000
    A = sp.random(N, N, density=0.01, format='csr', dtype=np.float64)
    x = np.random.rand(N).astype(np.float64)
    
    # Create MKL handle
    mkl_a, dbl, cplx = _create_mkl_sparse(A)
    
    print("MKL Handle created successfully.")
    
    # Hint: SPARSE_OPERATION_NON_TRANSPOSE = 10
    hint_res = libmkl.mkl_sparse_set_mv_hint(mkl_a, 10, ctypes.c_void_p(0), 1000)
    print(f"Hint result: {hint_res}")
    
    # Optimize
    opt_res = libmkl.mkl_sparse_optimize(mkl_a)
    print(f"Optimize result: {opt_res}")
    
    # Execute
    output_dtype = _output_dtypes[(dbl, cplx)]
    output_arr = _out_matrix((N,), output_dtype)
    func = MKL_wrapper._mkl_sparse_d_mv
    
    scalar = _mkl_scalar(1.0, cplx, dbl)
    out_scalar = _mkl_scalar(0.0, cplx, dbl)
    
    # Call executor
    exec_res = func(10, scalar, mkl_a, matrix_descr(), x, out_scalar, output_arr)
    print(f"Execute result: {exec_res}")
    
    # Verify correctness
    y_scipy = A @ x
    diff = np.linalg.norm(y_scipy - output_arr)
    print(f"Difference from Scipy: {diff}")
    assert diff < 1e-10
    
    _destroy_mkl_handle(mkl_a)
    print("MKL Handle destroyed.")

if __name__ == "__main__":
    test_inspector_executor()
