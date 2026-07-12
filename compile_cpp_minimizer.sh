#!/bin/bash
gcc -shared -fPIC -O3 -march=native -ffast-math -funroll-loops -fopenmp -std=c++17 \
    src/cpp_mkl_minimizer.cpp src/mkl_ffi_lib.so \
    -o src/libcpp_mkl_minimizer.so \
    -I/ceph/home/schrefl/jax_dev/mumag_matrixfree2/.pixi/envs/default/include \
    -L/ceph/home/schrefl/jax_dev/mumag_matrixfree2/.pixi/envs/default/lib \
    -I/ceph/home/schrefl/jax_dev/mumag_matrixfree2/.pixi/envs/cpu/include/python3.12 \
    -L/ceph/home/schrefl/jax_dev/mumag_matrixfree2/.pixi/envs/cpu/lib \
    -lmkl_rt -lstdc++ -lpython3.12

