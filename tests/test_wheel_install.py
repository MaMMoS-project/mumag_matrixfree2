"""Artifact and clean-installed-wheel verification."""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from importlib.metadata import distribution
from importlib.resources import files
from pathlib import Path

import pytest

linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="native wheel verification requires Linux",
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


def _verification_enabled(variable: str, reason: str) -> None:
    """Skip unless an opt-in artifact verification is enabled.

    Args:
        variable: Environment variable controlling verification.
        reason: Skip reason used when verification is disabled.
    """
    if os.environ.get(variable) != "1":
        pytest.skip(reason)


def _sdist_path() -> Path:
    """Return the configured or single locally built source distribution.

    Returns:
        Existing absolute source-distribution path.
    """
    if os.environ.get("TOMMOS_SDIST_PATH") is not None:
        return _required_path("TOMMOS_SDIST_PATH")
    candidates = sorted(Path("dist").glob("tommos-*.tar.gz"))
    if not candidates:
        pytest.skip("no locally built source distribution is available")
    assert len(candidates) == 1, f"expected one source distribution, found {candidates}"
    return candidates[0].resolve(strict=True)


@linux_only
def test_linux_wheel_contains_native_library_without_onemkl() -> None:
    """Require the Linux native library without bundled oneMKL libraries."""
    wheel = _required_path("TOMMOS_WHEEL_PATH")
    assert wheel.name.startswith("tommos-")
    assert wheel.name.endswith(("-linux_x86_64.whl", "-manylinux_2_28_x86_64.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    assert "tommos/_native/libcpp_mkl_minimizer.so" in names
    assert not [name for name in names if Path(name).name.startswith("libmkl")]


def test_sdist_contains_all_native_build_sources() -> None:
    """Retain the CMake project and C++ source in the source distribution."""
    sdist = _sdist_path()
    assert sdist.name.startswith("tommos-")
    assert sdist.name.endswith(".tar.gz")
    with tarfile.open(sdist, "r:gz") as archive:
        names = set(archive.getnames())
    root = sdist.name.removesuffix(".tar.gz")
    required = {
        f"{root}/src/cpp/CMakeLists.txt",
        f"{root}/src/cpp/cpp_mkl_minimizer.cpp",
    }
    assert required <= names


@linux_only
def test_clean_install_is_outside_checkout_and_contains_native_library() -> None:
    """Require a clean installed package outside the source checkout."""
    _verification_enabled(
        "TOMMOS_VERIFY_CLEAN_INSTALL",
        "clean-installed-wheel verification is not enabled",
    )
    import tommos

    checkout = _required_path("TOMMOS_CHECKOUT_PATH")
    package_path = Path(tommos.__file__).resolve(strict=True)
    assert not package_path.is_relative_to(checkout)
    installed_files = distribution("tommos").files or ()
    installed_names = {str(path) for path in installed_files}
    assert "tommos/_native/libcpp_mkl_minimizer.so" in installed_names
    assert not [name for name in installed_names if Path(name).name.startswith("libmkl")]


@linux_only
def test_clean_install_loads_native_library_with_ctypes() -> None:
    """Load the installed native library directly with ctypes."""
    _verification_enabled(
        "TOMMOS_VERIFY_CLEAN_INSTALL",
        "clean-installed-wheel verification is not enabled",
    )
    native_library = files("tommos").joinpath("_native", "libcpp_mkl_minimizer.so")
    assert native_library.is_file()
    ctypes.CDLL(str(native_library))


@linux_only
def test_linux_wheel_has_repaired_needed_entries_and_runpath() -> None:
    """Require renamed GNU OpenMP and both repaired RUNPATH components."""
    _verification_enabled(
        "TOMMOS_VERIFY_REPAIRED_WHEEL",
        "repaired-wheel verification is not enabled",
    )
    wheel = _required_path("TOMMOS_WHEEL_PATH")
    with tempfile.TemporaryDirectory() as temporary_directory:
        extension = Path(temporary_directory) / "libcpp_mkl_minimizer.so"
        with zipfile.ZipFile(wheel) as archive:
            extension.write_bytes(archive.read("tommos/_native/libcpp_mkl_minimizer.so"))
        dynamic = subprocess.run(
            ["readelf", "-d", extension],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    needed = set(re.findall(r"Shared library: \[([^]]+)\]", dynamic))
    assert "libmkl_rt.so.3" in needed
    assert any(re.fullmatch(r"libgomp-[^.]+\.so(?:\.[0-9]+)*", name) for name in needed)
    assert "(RPATH)" not in dynamic
    runpaths = re.findall(r"\(RUNPATH\).*Library runpath: \[([^]]*)\]", dynamic)
    assert len(runpaths) == 1
    components = set(runpaths[0].split(":"))
    assert "$ORIGIN/../../tommos.libs" in components
    assert "$ORIGIN/../../../.." in components


def test_macos_wheel_is_portable_and_runs_outside_checkout() -> None:
    """Validate the built macOS wheel and execute its portable SciPy/JAX path."""
    _verification_enabled(
        "TOMMOS_VERIFY_PORTABLE_INSTALL",
        "portable installed-wheel verification is not enabled",
    )
    assert sys.platform == "darwin"
    wheel = _required_path("TOMMOS_WHEEL_PATH")
    assert wheel.name.startswith("tommos-")
    assert wheel.name.endswith("-py3-none-any.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    assert not [name for name in names if name.startswith("tommos/_native/")]
    assert not [name for name in names if Path(name).name.startswith("libmkl")]

    import numpy as np
    import scipy.sparse as sp
    from packaging.requirements import Requirement

    import tommos
    from tommos.amg_utils import make_cpu_csr_op

    checkout = _required_path("TOMMOS_CHECKOUT_PATH")
    package_path = Path(tommos.__file__).resolve(strict=True)
    assert not package_path.is_relative_to(checkout)
    active_requirements = [
        Requirement(value)
        for value in distribution("tommos").requires or ()
        if Requirement(value).marker is None or Requirement(value).marker.evaluate()
    ]
    assert not [requirement for requirement in active_requirements if requirement.name in {"mkl", "sparse-dot-mkl"}]

    operation = make_cpu_csr_op(
        sp.csr_matrix(np.array([[2.0, 0.0], [1.0, 3.0]], dtype=np.float64)),
        cpu_spmv_backend="scipy",
    )
    result = operation(np.array([4.0, 5.0], dtype=np.float64))
    np.testing.assert_allclose(np.asarray(result), np.array([8.0, 19.0]), rtol=0.0, atol=0.0)
