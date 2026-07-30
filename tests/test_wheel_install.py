"""Artifact and clean-installed-wheel verification for Task 4."""

from __future__ import annotations

import ctypes
import os
import sys
import tarfile
import zipfile
from importlib.metadata import distribution
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Task 4 native wheel verification requires Linux",
)


def _required_path(variable: str) -> Path:
    """Return a required verification artifact path.

    Args:
        variable: Environment variable naming the artifact.

    Returns:
        Existing absolute artifact path.
    """
    value = os.environ.get(variable)
    if value is None:
        pytest.skip(f"{variable} is not configured")
    path = Path(value).resolve(strict=True)
    return path


@linux_only
def test_linux_wheel_contains_only_tommos_native_library() -> None:
    """Require a Linux-tagged wheel with the C++ library but no bundled oneMKL."""
    wheel = _required_path("TOMMOS_WHEEL_PATH")
    assert wheel.name.endswith("-linux_x86_64.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    assert "tommos/_native/libcpp_mkl_minimizer.so" in names
    assert not [name for name in names if Path(name).name.startswith("libmkl")]


def test_sdist_contains_all_native_build_sources() -> None:
    """Retain every native CMake/build source in the source distribution."""
    sdist = _required_path("TOMMOS_SDIST_PATH")
    with tarfile.open(sdist, "r:gz") as archive:
        names = set(archive.getnames())
    root = sdist.name.removesuffix(".tar.gz")
    required = {
        f"{root}/src/cpp/CMakeLists.txt",
        f"{root}/src/cpp/cpp_mkl_minimizer.cpp",
        f"{root}/src/cpp/find_mkl.py",
    }
    assert required <= names


@linux_only
def test_clean_install_exposes_independent_native_capabilities() -> None:
    """Require each installed native capability from its own probe."""
    if os.environ.get("TOMMOS_VERIFY_CLEAN_INSTALL") != "1":
        pytest.skip("clean-installed-wheel verification is not enabled")
    from tommos._native_loader import (
        probe_cpp_minimizer,
        probe_pardiso,
        probe_sparse_dot_mkl,
    )

    probes = {
        probe.name: probe
        for probe in (
            probe_cpp_minimizer(),
            probe_pardiso(),
            probe_sparse_dot_mkl(),
        )
    }
    assert all(probe.available for probe in probes.values()), {
        name: repr(probe.error) for name, probe in probes.items()
    }
    assert probes["cpp_minimizer"].source == "tommos._native"
    assert probes["sparse_dot_mkl"].source == "sparse_dot_mkl._mkl_interface"

    installed_files = distribution("tommos").files or ()
    installed_names = {str(path) for path in installed_files}
    assert "tommos/_native/libcpp_mkl_minimizer.so" in installed_names
    assert not [name for name in installed_names if Path(name).name.startswith("libmkl")]


@linux_only
def test_clean_install_pardiso_and_sparse_computations() -> None:
    """Match deterministic PARDISO and Inspector-Executor results to literals."""
    if os.environ.get("TOMMOS_VERIFY_CLEAN_INSTALL") != "1":
        pytest.skip("clean-installed-wheel verification is not enabled")
    from tommos._native_loader import create_pardiso_handle
    from tommos.amg_utils import PersistentMKLOperator

    matrix = sp.csr_matrix(np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64))
    upper_triangle = sp.triu(matrix, format="csr")
    expected = np.array([1.0, 2.0], dtype=np.float64)
    rhs = np.array([6.0, 7.0], dtype=np.float64)
    pardiso = create_pardiso_handle(
        2,
        upper_triangle.data,
        upper_triangle.indptr,
        upper_triangle.indices,
    )
    operator = PersistentMKLOperator(matrix)
    try:
        np.testing.assert_allclose(pardiso.solve(rhs), expected, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(operator.apply(expected), rhs, rtol=1e-13, atol=1e-13)
    finally:
        operator.close()
        pardiso.close()


@linux_only
def test_clean_install_propagates_pardiso_init_and_nested_minimizer_errors() -> None:
    """Exercise failure IDs and nested PARDISO status through the built C++ library."""
    if os.environ.get("TOMMOS_VERIFY_CLEAN_INSTALL") != "1":
        pytest.skip("clean-installed-wheel verification is not enabled")
    from tommos._native_loader import PardisoHandle, require_pardiso
    from tommos.cpp_minimizer import cpp_minimize

    bindings = require_pardiso()
    values = np.array([1.0], dtype=np.float64)
    indptr = np.array([0, 1], dtype=np.int32)
    indices = np.array([0], dtype=np.int32)
    failed_id = bindings.init_pardiso(
        0,
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )
    assert failed_id < 0

    invalid_owner = PardisoHandle(2**62, values, indptr, indices, bindings)
    identity = sp.eye(3, format="csr", dtype=np.float64)
    divergence = sp.csr_matrix(np.array([[1.0, 0.0, 0.0]], dtype=np.float64))
    params = SimpleNamespace(
        M_nodal=np.array([1.0], dtype=np.float64),
        V_mag=1.0,
        inv_M_prec=np.array([1.0], dtype=np.float64),
        max_iter=0,
        L=None,
    )
    try:
        with pytest.raises(RuntimeError, match="C\\+\\+ minimizer failed with native status -1"):
            cpp_minimize(
                np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.zeros(1, dtype=np.float64),
                params,
                {
                    "K_eff_sparse": identity,
                    "D_sparse": divergence,
                    "G_sparse": divergence.transpose().tocsr(),
                },
                solve_U=SimpleNamespace(pardiso_obj=invalid_owner),
            )
    finally:
        invalid_owner.close()


def test_macos_wheel_is_portable_and_runs_outside_checkout() -> None:
    """Validate the built macOS wheel and execute its portable SciPy/JAX path."""
    if os.environ.get("TOMMOS_VERIFY_PORTABLE_INSTALL") != "1":
        pytest.skip("portable installed-wheel verification is not enabled")
    assert sys.platform == "darwin"
    wheel = _required_path("TOMMOS_WHEEL_PATH")
    assert wheel.name.endswith("-py3-none-any.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    assert not [name for name in names if name.startswith("tommos/_native/")]

    from packaging.requirements import Requirement

    import tommos
    from tommos._native_loader import NativeSelections, resolve_native_selections
    from tommos.amg_utils import make_cpu_csr_op

    checkout = Path(os.environ["TOMMOS_CHECKOUT_PATH"]).resolve(strict=True)
    package_path = Path(tommos.__file__).resolve(strict=True)
    assert not package_path.is_relative_to(checkout)
    active_requirements = [
        Requirement(value)
        for value in distribution("tommos").requires or ()
        if Requirement(value).marker is None or Requirement(value).marker.evaluate()
    ]
    assert not [requirement for requirement in active_requirements if requirement.name in {"mkl", "sparse-dot-mkl"}]

    selections = resolve_native_selections(None, "auto", "auto", has_gpu=False)
    assert selections == NativeSelections("python", "jax", "scipy")
    operation = make_cpu_csr_op(
        sp.csr_matrix(np.array([[2.0, 0.0], [1.0, 3.0]], dtype=np.float64)),
        cpu_spmv_backend=selections.cpu_spmv_backend,
    )
    result = operation(np.array([4.0, 5.0], dtype=np.float64))
    np.testing.assert_allclose(np.asarray(result), np.array([8.0, 19.0]), rtol=0.0, atol=0.0)
