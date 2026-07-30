"""Lazy discovery and capability-specific bindings for optional native support."""

from __future__ import annotations

import atexit
import ctypes
import importlib
import os
import re
import threading
import weakref
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from importlib import metadata, resources
from pathlib import Path
from typing import Any

import numpy as np

NATIVE_ABI_VERSION: int = 1
_LIBRARY_NAME = "libcpp_mkl_minimizer.so"
_MKL_RUNTIME_PATTERN = re.compile(r"libmkl_rt\.so\.\d+(?:\.\d+)*")
_SPARSE_OPERATION_NON_TRANSPOSE = 10
_SPARSE_MATRIX_TYPE_GENERAL = 20
_resource_stack = ExitStack()
atexit.register(_resource_stack.close)


class MKLRuntimeError(RuntimeError):
    """Report invalid or unavailable distribution-owned oneMKL runtime state."""


class MKLMatrixDescription(ctypes.Structure):
    """C ABI descriptor for oneMKL Inspector-Executor sparse matrices."""

    _fields_ = [
        ("matrix_type", ctypes.c_int),
        ("fill_mode", ctypes.c_int),
        ("diag_type", ctypes.c_int),
    ]


@dataclass(frozen=True, slots=True)
class CapabilityProbe:
    """Result of checking one optional native capability.

    Attributes:
        name: Stable capability identifier.
        available: Whether the capability can be used.
        path: Native library path when the capability uses the C++ library.
        source: Discovery source or optional-package module name.
        error: Original error that made the capability unavailable.
    """

    name: str
    available: bool
    path: Path | None
    source: str | None
    error: BaseException | None


@dataclass(frozen=True, slots=True)
class PardisoBindings:
    """ABI-validated PARDISO functions from the native C++ library.

    Attributes:
        init_pardiso: Factorization setup function.
        pardiso_solve_direct: Direct-solve function.
        free_pardiso: Factorization-release function.
        library: Loaded library retained for the functions' lifetime.
    """

    init_pardiso: Any
    pardiso_solve_direct: Any
    free_pardiso: Any
    library: Any


@dataclass(frozen=True, slots=True)
class NativeSelections:
    """Resolved native and portable backend names.

    Attributes:
        cpp_minimizer: ``cpp_mkl`` or ``python``.
        poisson_solver: ``pardiso`` or ``jax``.
        cpu_spmv_backend: Selected sparse matrix-vector backend.
    """

    cpp_minimizer: str
    poisson_solver: str
    cpu_spmv_backend: str


def _finalize_pardiso_handle(
    lock: Any,
    bindings: PardisoBindings,
    library: Any,
    handle_id: int,
    values: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
) -> None:
    """Release a PARDISO handle while retaining all native dependencies.

    Args:
        lock: Per-handle synchronization lock.
        bindings: Functions used to release the factorization.
        library: Native library retained through release.
        handle_id: Native factorization identifier.
        values: CSR values retained through release.
        indptr: CSR row offsets retained through release.
        indices: CSR column indices retained through release.
    """
    _ = (library, values, indptr, indices)
    with lock:
        bindings.free_pardiso(handle_id)


class PardisoHandle:
    """Own one native PARDISO factorization and its backing CSR storage."""

    def __init__(
        self,
        handle_id: int,
        values: np.ndarray,
        indptr: np.ndarray,
        indices: np.ndarray,
        bindings: PardisoBindings,
    ) -> None:
        """Initialize a PARDISO resource owner.

        Args:
            handle_id: Native factorization identifier.
            values: Contiguous float64 CSR values.
            indptr: Contiguous int32 CSR row offsets.
            indices: Contiguous int32 CSR column indices.
            bindings: Functions and library that created the factorization.
        """
        self.handle_id = handle_id
        self.values = values
        self.indptr = indptr
        self.indices = indices
        self.bindings = bindings
        self.library = bindings.library
        self._lock = threading.Lock()
        self._finalizer = weakref.finalize(
            self,
            _finalize_pardiso_handle,
            self._lock,
            bindings,
            bindings.library,
            handle_id,
            values,
            indptr,
            indices,
        )

    @property
    def closed(self) -> bool:
        """Return whether the native factorization has been released.

        Returns:
            True after explicit close or backup finalization.
        """
        return not self._finalizer.alive

    def close(self) -> None:
        """Release the native factorization exactly once."""
        self._finalizer()

    @contextmanager
    def borrow_handle_id(self) -> Iterator[int]:
        """Borrow the live native identifier under the owner lock.

        Yields:
            Strictly positive native factorization identifier.

        Raises:
            RuntimeError: If the native factorization has been released.
        """
        with self._lock:
            if self.closed:
                raise RuntimeError("PARDISO handle is closed")
            yield self.handle_id

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve one right-hand side with the retained factorization.

        Args:
            rhs: Dense right-hand side values.

        Returns:
            Contiguous float64 solution values.

        Raises:
            RuntimeError: If the handle is closed or the native solve fails.
        """
        with self._lock:
            if self.closed:
                raise RuntimeError("PARDISO handle is closed")
            rhs_array = np.ascontiguousarray(rhs, dtype=np.float64)
            solution = np.zeros_like(rhs_array)
            error = self.bindings.pardiso_solve_direct(
                self.handle_id,
                rhs_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                solution.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            if error != 0:
                raise RuntimeError(f"PARDISO solve failed with error {error}")
            return solution


_library: Any | None = None
_library_path: Path | None = None
_library_source: str | None = None
_library_error: BaseException | None = None
_library_attempted = False
_minimizer_function: Any | None = None
_minimizer_error: BaseException | None = None
_minimizer_attempted = False
_pardiso_bindings: PardisoBindings | None = None
_pardiso_error: BaseException | None = None
_pardiso_attempted = False
_sparse_dot_mkl_probe: CapabilityProbe | None = None
_mkl_runtime: Any | None = None
_mkl_runtime_path: Path | None = None
_mkl_runtime_error: BaseException | None = None
_mkl_runtime_attempted = False
_mkl_sparse_set_mv_hint_function: Any | None = None
_mkl_sparse_optimize_function: Any | None = None
_mkl_sparse_inspector_error: BaseException | None = None
_mkl_sparse_inspector_attempted = False


def load_mkl_runtime() -> Any:
    """Load the unique versioned oneMKL runtime owned by the ``mkl`` distribution.

    Returns:
        Cached runtime handle loaded with global symbol visibility.

    Raises:
        BaseException: The original cached metadata, path, or loader error.
    """
    global _mkl_runtime, _mkl_runtime_attempted, _mkl_runtime_error, _mkl_runtime_path
    if _mkl_runtime_attempted:
        if _mkl_runtime is not None:
            return _mkl_runtime
        assert _mkl_runtime_error is not None
        raise _mkl_runtime_error

    try:
        try:
            distribution = metadata.distribution("mkl")
        except metadata.PackageNotFoundError as error:
            raise MKLRuntimeError("mkl distribution is not installed; install mkl") from error

        matches: set[Path] = set()
        for entry in distribution.files or ():
            if _MKL_RUNTIME_PATTERN.fullmatch(entry.name) is None:
                continue
            located_path = Path(distribution.locate_file(entry))
            try:
                matches.add(located_path.resolve(strict=True))
            except FileNotFoundError as error:
                missing_path = located_path.resolve()
                raise MKLRuntimeError(
                    f"mkl distribution-owned versioned libmkl_rt.so.* does not exist: {missing_path}; reinstall mkl"
                ) from error

        ordered_matches = sorted(matches)
        if not ordered_matches:
            raise MKLRuntimeError("mkl distribution does not own a versioned libmkl_rt.so.*; reinstall mkl")
        if len(ordered_matches) > 1:
            candidates = ", ".join(str(path) for path in ordered_matches)
            raise MKLRuntimeError(
                f"mkl distribution owns multiple versioned libmkl_rt.so.* files: {candidates}; expected exactly one"
            )

        runtime_path = ordered_matches[0]
        runtime = ctypes.CDLL(str(runtime_path), mode=ctypes.RTLD_GLOBAL)
        _mkl_runtime = runtime
        _mkl_runtime_path = runtime_path
        _mkl_runtime_attempted = True
        return runtime
    except Exception as error:
        _mkl_runtime_attempted = True
        _mkl_runtime_error = error
        raise


def _bind_mkl_sparse_inspector() -> tuple[Any, Any]:
    """Bind oneMKL sparse hint and optimization symbols against the cached runtime.

    Returns:
        Configured hint and optimization functions.

    Raises:
        BaseException: The original cached runtime or symbol-binding error.
    """
    global _mkl_sparse_inspector_attempted
    global _mkl_sparse_inspector_error
    global _mkl_sparse_optimize_function
    global _mkl_sparse_set_mv_hint_function
    if _mkl_sparse_inspector_attempted:
        if _mkl_sparse_set_mv_hint_function is not None and _mkl_sparse_optimize_function is not None:
            return _mkl_sparse_set_mv_hint_function, _mkl_sparse_optimize_function
        assert _mkl_sparse_inspector_error is not None
        raise _mkl_sparse_inspector_error

    try:
        runtime = load_mkl_runtime()
        set_mv_hint = runtime.mkl_sparse_set_mv_hint
        set_mv_hint.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            MKLMatrixDescription,
            ctypes.c_int,
        ]
        set_mv_hint.restype = ctypes.c_int
        optimize = runtime.mkl_sparse_optimize
        optimize.argtypes = [ctypes.c_void_p]
        optimize.restype = ctypes.c_int
        _mkl_sparse_set_mv_hint_function = set_mv_hint
        _mkl_sparse_optimize_function = optimize
        _mkl_sparse_inspector_attempted = True
        return set_mv_hint, optimize
    except Exception as error:
        _mkl_sparse_inspector_attempted = True
        _mkl_sparse_inspector_error = error
        raise


def mkl_sparse_set_mv_hint(handle: Any, expected_calls: int) -> int:
    """Set a non-transpose, general-matrix Inspector-Executor SpMV hint.

    Args:
        handle: Wrapper-created opaque ``sparse_matrix_t`` handle.
        expected_calls: Expected number of subsequent SpMV executions.

    Returns:
        oneMKL ``sparse_status_t`` integer.
    """
    set_mv_hint, _ = _bind_mkl_sparse_inspector()
    description = MKLMatrixDescription(_SPARSE_MATRIX_TYPE_GENERAL, 0, 0)
    return int(
        set_mv_hint(
            handle,
            _SPARSE_OPERATION_NON_TRANSPOSE,
            description,
            expected_calls,
        )
    )


def mkl_sparse_optimize(handle: Any) -> int:
    """Optimize a wrapper-created sparse handle through the cached runtime.

    Args:
        handle: Wrapper-created opaque ``sparse_matrix_t`` handle.

    Returns:
        oneMKL ``sparse_status_t`` integer.
    """
    _, optimize = _bind_mkl_sparse_inspector()
    return int(optimize(handle))


def _repository_library_path() -> Path:
    """Return the source-tree fallback native-library path.

    Returns:
        Absolute path of the repository fallback candidate.
    """
    return Path(__file__).resolve().parents[2] / "lib" / _LIBRARY_NAME


def _package_library_path() -> Path | None:
    """Resolve the packaged native library while keeping extracted resources alive.

    Returns:
        Absolute extracted resource path, or None when the package has no library.
    """
    package_resource = resources.files("tommos").joinpath("_native", _LIBRARY_NAME)
    if not package_resource.is_file():
        return None
    return _resource_stack.enter_context(resources.as_file(package_resource)).resolve()


def _automatic_candidates() -> tuple[list[tuple[Path, str]], Exception | None]:
    """Return automatic candidates and the first ordinary discovery error.

    Returns:
        Existing candidate paths paired with their discovery-source labels and
        the first ordinary discovery error, if any.
    """
    candidates: list[tuple[Path, str]] = []
    first_error: Exception | None = None
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        try:
            slurm_path = Path("/tmp") / f"mumag_build_{slurm_job_id}" / _LIBRARY_NAME
            if slurm_path.is_file():
                candidates.append((slurm_path.resolve(), "SLURM_JOB_ID"))
        except Exception as error:
            first_error = error
    environment_path = os.environ.get("MUMAG_LIB_OUT")
    if environment_path:
        try:
            mumag_path = Path(environment_path) / _LIBRARY_NAME
            if mumag_path.is_file():
                candidates.append((mumag_path.resolve(), "MUMAG_LIB_OUT"))
        except Exception as error:
            if first_error is None:
                first_error = error
    try:
        package_path = _package_library_path()
        if package_path is not None:
            candidates.append((package_path, "tommos._native"))
    except Exception as error:
        if first_error is None:
            first_error = error
    try:
        repository_path = _repository_library_path()
        if repository_path.is_file():
            candidates.append((repository_path.resolve(), "repository"))
    except Exception as error:
        if first_error is None:
            first_error = error
    return candidates, first_error


def _configured_candidate() -> tuple[Path, str] | None:
    """Validate and return the explicitly configured native-library candidate.

    Returns:
        Configured absolute path and its source label, or None when unset.

    Raises:
        ValueError: If the configured path is relative.
        FileNotFoundError: If the configured absolute path is not a file.
    """
    configured_path = os.environ.get("TOMMOS_NATIVE_LIBRARY")
    if configured_path is None:
        return None
    path = Path(configured_path)
    if not path.is_absolute():
        raise ValueError("TOMMOS_NATIVE_LIBRARY must be an absolute path")
    if not path.is_file():
        raise FileNotFoundError(f"TOMMOS_NATIVE_LIBRARY does not exist: {path}")
    return path.resolve(), "TOMMOS_NATIVE_LIBRARY"


def _validate_abi(library: Any) -> None:
    """Validate the library ABI before any operation symbol is accessed.

    Args:
        library: Loaded shared-library handle.

    Raises:
        RuntimeError: If the library ABI differs from the supported version.
    """
    abi_function = library.tommos_native_abi_version
    abi_function.argtypes = []
    abi_function.restype = ctypes.c_int
    abi_version = abi_function()
    if abi_version != NATIVE_ABI_VERSION:
        raise RuntimeError(f"Unsupported native ABI version {abi_version}; expected {NATIVE_ABI_VERSION}")


def _load_library() -> tuple[Any, Path, str]:
    """Load and ABI-validate the first usable native-library candidate.

    Returns:
        Loaded library handle, absolute path, and discovery-source label.

    Raises:
        BaseException: The original explicit or first automatic discovery/load error.
    """
    global _library, _library_attempted, _library_error, _library_path, _library_source
    if _library_attempted:
        if _library is not None and _library_path is not None and _library_source is not None:
            return _library, _library_path, _library_source
        assert _library_error is not None
        raise _library_error

    try:
        configured = _configured_candidate()
        if configured is not None:
            candidate_path, source = configured
            load_mkl_runtime()
            library = ctypes.CDLL(str(candidate_path))
            _validate_abi(library)
            _library = library
            _library_path = candidate_path
            _library_source = source
            _library_attempted = True
            return library, candidate_path, source

        candidates, first_error = _automatic_candidates()
        for candidate_path, source in candidates:
            try:
                load_mkl_runtime()
                library = ctypes.CDLL(str(candidate_path))
                _validate_abi(library)
            except Exception as error:
                if first_error is None:
                    first_error = error
                continue
            _library = library
            _library_path = candidate_path
            _library_source = source
            _library_attempted = True
            return library, candidate_path, source
        if first_error is not None:
            raise first_error
        raise FileNotFoundError(f"Could not find {_LIBRARY_NAME} in any supported location")
    except Exception as error:
        _library_attempted = True
        _library_error = error
        raise


def _bind_cpp_minimizer() -> Any:
    """Bind the C++ minimizer operation without touching PARDISO symbols.

    Returns:
        Configured minimizer function.
    """
    global _minimizer_attempted, _minimizer_error, _minimizer_function
    if _minimizer_attempted:
        if _minimizer_function is not None:
            return _minimizer_function
        assert _minimizer_error is not None
        raise _minimizer_error
    try:
        library, _, _ = _load_library()
        function = library.run_cpp_pcohen_hs_minimization
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        function.restype = ctypes.c_int
        _minimizer_function = function
        _minimizer_attempted = True
        return function
    except Exception as error:
        _minimizer_attempted = True
        _minimizer_error = error
        raise


def _bind_pardiso() -> PardisoBindings:
    """Bind the PARDISO operations without touching minimizer symbols.

    Returns:
        Configured PARDISO operation bindings.
    """
    global _pardiso_attempted, _pardiso_bindings, _pardiso_error
    if _pardiso_attempted:
        if _pardiso_bindings is not None:
            return _pardiso_bindings
        assert _pardiso_error is not None
        raise _pardiso_error
    try:
        library, _, _ = _load_library()
        init_pardiso = library.init_pardiso
        init_pardiso.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        init_pardiso.restype = ctypes.c_int64
        pardiso_solve_direct = library.pardiso_solve_direct
        pardiso_solve_direct.argtypes = [
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        pardiso_solve_direct.restype = ctypes.c_int
        free_pardiso = library.free_pardiso
        free_pardiso.argtypes = [ctypes.c_int64]
        free_pardiso.restype = None
        _pardiso_bindings = PardisoBindings(
            init_pardiso=init_pardiso,
            pardiso_solve_direct=pardiso_solve_direct,
            free_pardiso=free_pardiso,
            library=library,
        )
        _pardiso_attempted = True
        return _pardiso_bindings
    except Exception as error:
        _pardiso_attempted = True
        _pardiso_error = error
        raise


def _native_probe(name: str, binding: Callable[[], Any]) -> CapabilityProbe:
    """Return a probe result for one native-library operation binding.

    Args:
        name: Stable capability identifier.
        binding: Binding operation that verifies the requested capability.

    Returns:
        Available or unavailable capability result with the original failure.
    """
    try:
        binding()
    except Exception as error:
        return CapabilityProbe(name, False, _library_path, _library_source, error)
    return CapabilityProbe(name, True, _library_path, _library_source, None)


def probe_cpp_minimizer() -> CapabilityProbe:
    """Probe whether the ABI-compatible C++ minimizer is available.

    Returns:
        Capability result for the C++ minimizer operation.
    """
    return _native_probe("cpp_minimizer", _bind_cpp_minimizer)


def require_cpp_minimizer() -> Any:
    """Return the C++ minimizer function or raise its original loading error.

    Returns:
        ABI-validated, ctypes-configured minimizer function.

    Raises:
        BaseException: The original discovery, ABI, or symbol-binding error.
    """
    probe = probe_cpp_minimizer()
    if not probe.available:
        assert probe.error is not None
        raise probe.error
    return _bind_cpp_minimizer()


def probe_pardiso() -> CapabilityProbe:
    """Probe whether all ABI-compatible PARDISO functions are available.

    Returns:
        Capability result for the PARDISO operation group.
    """
    return _native_probe("pardiso", _bind_pardiso)


def require_pardiso() -> PardisoBindings:
    """Return PARDISO bindings or raise their original loading error.

    Returns:
        ABI-validated, ctypes-configured PARDISO bindings.

    Raises:
        BaseException: The original discovery, ABI, or symbol-binding error.
    """
    probe = probe_pardiso()
    if not probe.available:
        assert probe.error is not None
        raise probe.error
    return _bind_pardiso()


def create_pardiso_handle(
    size: int,
    values: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
) -> PardisoHandle:
    """Create an owning PARDISO factorization from CSR arrays.

    Args:
        size: Matrix row count.
        values: CSR nonzero values.
        indptr: CSR row offsets.
        indices: CSR column indices.

    Returns:
        Owner retaining normalized arrays, bindings, and native library.

    Raises:
        RuntimeError: If native factorization initialization fails.
    """
    bindings = require_pardiso()
    values_array = np.ascontiguousarray(values, dtype=np.float64)
    indptr_array = np.ascontiguousarray(indptr, dtype=np.int32)
    indices_array = np.ascontiguousarray(indices, dtype=np.int32)
    handle_id = bindings.init_pardiso(
        size,
        values_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        indptr_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        indices_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )
    if handle_id <= 0:
        raise RuntimeError(
            f"PARDISO initialization failed with native status {handle_id}; expected a strictly positive handle ID"
        )
    return PardisoHandle(
        int(handle_id),
        values_array,
        indptr_array,
        indices_array,
        bindings,
    )


def probe_sparse_dot_mkl() -> CapabilityProbe:
    """Probe the oneMKL runtime and sparse-dot-mkl wrapper independently of C++.

    Returns:
        Capability result requiring both runtime and wrapper availability.
    """
    global _sparse_dot_mkl_probe
    if _sparse_dot_mkl_probe is not None:
        return _sparse_dot_mkl_probe
    source = "sparse_dot_mkl._mkl_interface"
    try:
        load_mkl_runtime()
        module = importlib.import_module(source)
        _ = module.MKL
        module_file = getattr(module, "__file__", None)
        path = Path(module_file).resolve() if module_file is not None else None
        _sparse_dot_mkl_probe = CapabilityProbe("sparse_dot_mkl", True, path, source, None)
    except Exception as error:
        _sparse_dot_mkl_probe = CapabilityProbe("sparse_dot_mkl", False, None, source, error)
    return _sparse_dot_mkl_probe


def require_sparse_dot_mkl() -> Any:
    """Return the independently probed sparse-dot-mkl interface module.

    Returns:
        Imported ``sparse_dot_mkl._mkl_interface`` module.

    Raises:
        BaseException: The original wrapper import or access error.
    """
    probe = probe_sparse_dot_mkl()
    if not probe.available:
        assert probe.error is not None
        raise probe.error
    assert probe.source is not None
    return importlib.import_module(probe.source)


def resolve_native_selections(
    cpp_mkl: bool | None,
    poisson_solver: str,
    cpu_spmv_backend: str,
    has_gpu: bool,
) -> NativeSelections:
    """Resolve optional native backends independently.

    Args:
        cpp_mkl: True for strict C++, False for Python, or None for automatic.
        poisson_solver: ``auto``, ``jax``, or strict ``pardiso``.
        cpu_spmv_backend: ``auto`` or an explicit sparse backend.
        has_gpu: Whether GPU execution is active.

    Returns:
        Fully resolved backend names.

    Raises:
        BaseException: Original capability error for an explicit native choice.
        ValueError: If a requested backend name is unsupported.
    """
    if cpp_mkl is True and poisson_solver == "jax":
        raise ValueError("C++ minimizer requires the PARDISO Poisson solver")

    if poisson_solver == "jax" or (poisson_solver == "auto" and has_gpu and cpp_mkl is not True):
        poisson_selection = "jax"
    elif poisson_solver in {"auto", "pardiso"}:
        pardiso_probe = probe_pardiso()
        if (poisson_solver == "pardiso" or cpp_mkl is True) and not pardiso_probe.available:
            assert pardiso_probe.error is not None
            raise pardiso_probe.error
        poisson_selection = "pardiso" if pardiso_probe.available else "jax"
    else:
        raise ValueError(f"Unknown Poisson solver: {poisson_solver}")

    if cpp_mkl is False or (cpp_mkl is None and (has_gpu or poisson_selection != "pardiso")):
        cpp_selection = "python"
    else:
        cpp_probe = probe_cpp_minimizer()
        if cpp_mkl is True and not cpp_probe.available:
            assert cpp_probe.error is not None
            raise cpp_probe.error
        cpp_selection = "cpp_mkl" if cpp_probe.available else "python"

    if cpu_spmv_backend in {"scipy", "jax_default", "custom_jax"}:
        spmv_selection = cpu_spmv_backend
    elif cpu_spmv_backend == "auto" and has_gpu:
        spmv_selection = "scipy"
    elif cpu_spmv_backend in {"auto", "persistent_mkl", "dot_product_mkl"}:
        sparse_probe = probe_sparse_dot_mkl()
        if cpu_spmv_backend != "auto" and not sparse_probe.available:
            assert sparse_probe.error is not None
            raise sparse_probe.error
        spmv_selection = "persistent_mkl" if sparse_probe.available else "scipy"
        if cpu_spmv_backend == "dot_product_mkl" and sparse_probe.available:
            spmv_selection = "dot_product_mkl"
    else:
        raise ValueError(f"Unknown CPU SpMV backend: {cpu_spmv_backend}")

    return NativeSelections(cpp_selection, poisson_selection, spmv_selection)


def native_diagnostics() -> dict[str, CapabilityProbe]:
    """Return independently evaluated native capability diagnostics.

    Returns:
        Capability results keyed by their stable identifiers.
    """
    return {
        "cpp_minimizer": probe_cpp_minimizer(),
        "pardiso": probe_pardiso(),
        "sparse_dot_mkl": probe_sparse_dot_mkl(),
    }
