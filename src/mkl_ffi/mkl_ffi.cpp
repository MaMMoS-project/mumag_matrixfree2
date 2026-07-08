#include "xla/ffi/api/ffi.h"
#include "mkl_spblas.h"
#include <Python.h>
#include <iostream>

// 1. Core SpMV FFI function
xla::ffi::Error MklSpmvFFI(
    xla::ffi::Buffer<xla::ffi::DataType::F64> x,
    xla::ffi::Buffer<xla::ffi::DataType::S64> handle_buf,
    xla::ffi::ResultBuffer<xla::ffi::DataType::F64> out
) {
    // Extract the raw pointer address of the MKL sparse matrix handle
    int64_t handle_addr = *handle_buf.typed_data();
    sparse_matrix_t mkl_a = reinterpret_cast<sparse_matrix_t>(handle_addr);

    // Get input and output raw data pointers (directly from JAX-managed memory)
    const double* x_ptr = x.typed_data();
    double* y_ptr = out->typed_data();

    // Configure General sparse matrix type descriptor
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    // Execute zero-copy MKL sparse matrix-vector product
    sparse_status_t status = mkl_sparse_d_mv(
        SPARSE_OPERATION_NON_TRANSPOSE,
        1.0,
        mkl_a,
        descr,
        x_ptr,
        0.0,
        y_ptr
    );

    if (status != SPARSE_STATUS_SUCCESS) {
        return xla::ffi::Error::InvalidArgument("MKL SpMV call returned non-success status");
    }

    return xla::ffi::Error::Success();
}

// 2. Bind the XLA FFI Handler
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    mkl_spmv_handler, MklSpmvFFI,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::Buffer<xla::ffi::DataType::F64>>()   // input x
        .Arg<xla::ffi::Buffer<xla::ffi::DataType::S64>>()   // handle address scalar
        .Ret<xla::ffi::Buffer<xla::ffi::DataType::F64>>()   // output out
);

// 3. Expose the handler via a standard Python C Extension Module (Capsule)
// This avoids dependencies on pybind11
static PyObject* get_registrations(PyObject* self, PyObject* args) {
    PyObject* dict = PyDict_New();
    PyObject* capsule = PyCapsule_New(reinterpret_cast<void*>(mkl_spmv_handler), "xla_ffi", nullptr);
    PyDict_SetItemString(dict, "mkl_spmv_ffi", capsule);
    Py_DECREF(capsule);
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
