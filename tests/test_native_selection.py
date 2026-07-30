"""Tests for native backend selection and consumer integration."""

from __future__ import annotations

import gc
import shutil
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp

from tommos import _native_loader


def _probe(name: str, available: bool, error: BaseException | None = None) -> _native_loader.CapabilityProbe:
    """Create a capability result for selection tests.

    Args:
        name: Capability identifier.
        available: Whether the capability is available.
        error: Original failure for an unavailable capability.

    Returns:
        Capability result with no filesystem provenance.
    """
    return _native_loader.CapabilityProbe(name, available, None, "test", error)


def test_explicit_portable_choices_do_not_probe_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """Portable explicit choices must not access any optional native capability."""

    def unexpected_probe() -> Any:
        """Fail if a native probe is attempted.

        Returns:
            This function never returns.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError("native capability was probed")

    monkeypatch.setattr(_native_loader, "probe_cpp_minimizer", unexpected_probe)
    monkeypatch.setattr(_native_loader, "probe_pardiso", unexpected_probe)
    monkeypatch.setattr(_native_loader, "probe_sparse_dot_mkl", unexpected_probe)

    selections = _native_loader.resolve_native_selections(False, "jax", "scipy", has_gpu=False)

    assert selections == _native_loader.NativeSelections("python", "jax", "scipy")


def test_gpu_auto_choices_do_not_probe_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """Automatic GPU selection must retain JAX paths without native probes."""

    def unexpected_probe() -> Any:
        """Fail if a native probe is attempted.

        Returns:
            This function never returns.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError("native capability was probed")

    monkeypatch.setattr(_native_loader, "probe_cpp_minimizer", unexpected_probe)
    monkeypatch.setattr(_native_loader, "probe_pardiso", unexpected_probe)
    monkeypatch.setattr(_native_loader, "probe_sparse_dot_mkl", unexpected_probe)

    selections = _native_loader.resolve_native_selections(None, "auto", "auto", has_gpu=True)

    assert selections == _native_loader.NativeSelections("python", "jax", "scipy")


@pytest.mark.parametrize(
    ("cpp_available", "pardiso_available", "sparse_available", "expected"),
    [
        (True, True, True, ("cpp_mkl", "pardiso", "persistent_mkl")),
        (False, True, False, ("python", "pardiso", "scipy")),
        (True, False, True, ("python", "jax", "persistent_mkl")),
        (False, False, False, ("python", "jax", "scipy")),
    ],
)
def test_cpu_auto_resolves_capabilities_independently(
    monkeypatch: pytest.MonkeyPatch,
    cpp_available: bool,
    pardiso_available: bool,
    sparse_available: bool,
    expected: tuple[str, str, str],
) -> None:
    """One unavailable CPU capability must not disable another available one."""
    monkeypatch.setattr(
        _native_loader,
        "probe_cpp_minimizer",
        lambda: _probe("cpp_minimizer", cpp_available, RuntimeError("cpp")),
    )
    monkeypatch.setattr(
        _native_loader,
        "probe_pardiso",
        lambda: _probe("pardiso", pardiso_available, RuntimeError("pardiso")),
    )
    monkeypatch.setattr(
        _native_loader,
        "probe_sparse_dot_mkl",
        lambda: _probe("sparse_dot_mkl", sparse_available, RuntimeError("sparse")),
    )

    selections = _native_loader.resolve_native_selections(None, "auto", "auto", has_gpu=False)

    assert (
        selections.cpp_minimizer,
        selections.poisson_solver,
        selections.cpu_spmv_backend,
    ) == expected


@pytest.mark.parametrize(
    ("cpp_mkl", "poisson_solver", "cpu_spmv_backend", "probe_name"),
    [
        (True, "pardiso", "scipy", "cpp_minimizer"),
        (False, "pardiso", "scipy", "pardiso"),
        (False, "jax", "persistent_mkl", "sparse_dot_mkl"),
        (False, "jax", "dot_product_mkl", "sparse_dot_mkl"),
    ],
)
def test_explicit_native_choice_reraises_original_error(
    monkeypatch: pytest.MonkeyPatch,
    cpp_mkl: bool,
    poisson_solver: str,
    cpu_spmv_backend: str,
    probe_name: str,
) -> None:
    """An explicit unavailable native choice must raise its original failure."""
    original_error = RuntimeError(f"{probe_name} unavailable")
    monkeypatch.setattr(
        _native_loader,
        "probe_cpp_minimizer",
        lambda: _probe("cpp_minimizer", True),
    )
    monkeypatch.setattr(
        _native_loader,
        "probe_pardiso",
        lambda: _probe("pardiso", True),
    )
    monkeypatch.setattr(
        _native_loader,
        "probe_sparse_dot_mkl",
        lambda: _probe("sparse_dot_mkl", True),
    )
    monkeypatch.setattr(
        _native_loader,
        f"probe_{probe_name}",
        lambda: _probe(probe_name, False, original_error),
    )

    with pytest.raises(RuntimeError) as caught:
        _native_loader.resolve_native_selections(
            cpp_mkl,
            poisson_solver,
            cpu_spmv_backend,
            has_gpu=False,
        )

    assert caught.value is original_error


def test_explicit_cpp_rejects_jax_poisson_without_probing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit C++ minimization must reject the incompatible JAX Poisson path."""
    monkeypatch.setattr(
        _native_loader,
        "probe_cpp_minimizer",
        lambda: pytest.fail("incompatibility must be rejected before probing C++"),
    )
    monkeypatch.setattr(
        _native_loader,
        "probe_pardiso",
        lambda: pytest.fail("incompatibility must be rejected before probing PARDISO"),
    )

    with pytest.raises(ValueError, match="C\\+\\+ minimizer requires the PARDISO Poisson solver"):
        _native_loader.resolve_native_selections(True, "jax", "scipy", has_gpu=False)


@pytest.mark.parametrize("poisson_solver", ["auto", "pardiso"])
def test_explicit_cpp_reraises_original_pardiso_error(
    monkeypatch: pytest.MonkeyPatch,
    poisson_solver: str,
) -> None:
    """Explicit C++ selection must preserve an unavailable PARDISO cause."""
    original_error = RuntimeError("PARDISO unavailable")
    monkeypatch.setattr(
        _native_loader,
        "probe_pardiso",
        lambda: _probe("pardiso", False, original_error),
    )
    monkeypatch.setattr(
        _native_loader,
        "probe_cpp_minimizer",
        lambda: _probe("cpp_minimizer", True),
    )

    with pytest.raises(RuntimeError) as caught:
        _native_loader.resolve_native_selections(
            True,
            poisson_solver,
            "scipy",
            has_gpu=False,
        )

    assert caught.value is original_error


def test_auto_cpp_with_explicit_jax_selects_python_without_cpp_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic C++ selection must stay portable with explicit JAX Poisson."""
    monkeypatch.setattr(
        _native_loader,
        "probe_cpp_minimizer",
        lambda: pytest.fail("C++ must not be probed when JAX Poisson is explicit"),
    )

    selections = _native_loader.resolve_native_selections(None, "jax", "scipy", has_gpu=False)

    assert selections.cpp_minimizer == "python"


@pytest.mark.parametrize("backend", ["jax_default", "custom_jax"])
def test_legacy_jax_cpu_backend_passes_through_without_native_probe(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """Explicit JAX CPU backends must remain pass-through selections."""
    monkeypatch.setattr(
        _native_loader,
        "probe_sparse_dot_mkl",
        lambda: pytest.fail("sparse wrapper must not be probed"),
    )

    selections = _native_loader.resolve_native_selections(False, "jax", backend, has_gpu=False)

    assert selections.cpu_spmv_backend == backend


def test_cli_resolves_explicit_native_choice_before_mesh_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable explicit native CLI choice must fail before mesh validation."""
    from tommos import loop

    original_error = RuntimeError("cpp unavailable")
    monkeypatch.setattr(sys, "argv", ["tommos.loop", "--cpp-mkl"])
    monkeypatch.setattr(loop.jax, "devices", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        loop,
        "resolve_native_selections",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(original_error),
        raising=False,
    )

    with pytest.raises(RuntimeError) as caught:
        loop.main()

    assert caught.value is original_error


def test_p2_native_choices_participate_in_the_single_final_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Native `.p2` values must reach one final resolver call before mesh loading."""
    from tommos import loop

    model = tmp_path / "portable-model"
    model.with_suffix(".p2").write_text(
        "[minimizer]\ncpp_mkl = false\n[poisson]\npoisson_solver = jax\ncpu_spmv_backend = scipy\n",
        encoding="utf-8",
    )
    resolution_calls: list[tuple[bool | None, str, str, bool]] = []
    stop_after_resolution = RuntimeError("stop after final native resolution")

    def resolve(
        cpp_mkl: bool | None,
        poisson_solver: str,
        cpu_spmv_backend: str,
        *,
        has_gpu: bool,
    ) -> _native_loader.NativeSelections:
        """Record requested native values and stop after final resolution.

        Args:
            cpp_mkl: Requested minimizer choice.
            poisson_solver: Requested Poisson choice.
            cpu_spmv_backend: Requested CPU sparse backend.
            has_gpu: Whether GPU execution is available.

        Returns:
            This function never returns.
        """
        resolution_calls.append((cpp_mkl, poisson_solver, cpu_spmv_backend, has_gpu))
        raise stop_after_resolution

    monkeypatch.setattr(sys, "argv", ["tommos.loop", str(model)])
    monkeypatch.setattr(loop.jax, "devices", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(loop, "resolve_native_selections", resolve)

    with pytest.raises(RuntimeError) as caught:
        loop.main()

    assert caught.value is stop_after_resolution
    assert resolution_calls == [(False, "jax", "scipy", False)]


def test_explicit_native_cli_overrides_p2_after_early_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Explicit native flags must validate early and remain final over `.p2`."""
    from tommos import loop

    model = tmp_path / "native-model"
    model.with_suffix(".p2").write_text(
        "[minimizer]\ncpp_mkl = true\n[poisson]\npoisson_solver = pardiso\ncpu_spmv_backend = persistent_mkl\n",
        encoding="utf-8",
    )
    resolution_calls: list[tuple[bool | None, str, str, bool]] = []
    selections = _native_loader.NativeSelections("python", "jax", "scipy")
    stop_after_final = RuntimeError("stop after final native resolution")

    def resolve(
        cpp_mkl: bool | None,
        poisson_solver: str,
        cpu_spmv_backend: str,
        *,
        has_gpu: bool,
    ) -> _native_loader.NativeSelections:
        """Return early validation and stop on the one final resolution.

        Args:
            cpp_mkl: Requested minimizer choice.
            poisson_solver: Requested Poisson choice.
            cpu_spmv_backend: Requested CPU sparse backend.
            has_gpu: Whether GPU execution is available.

        Returns:
            Portable authoritative selections for early validation.
        """
        resolution_calls.append((cpp_mkl, poisson_solver, cpu_spmv_backend, has_gpu))
        if len(resolution_calls) == 1:
            return selections
        raise stop_after_final

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tommos.loop",
            str(model),
            "--no-cpp-mkl",
            "--poisson-solver",
            "jax",
            "--cpu-spmv-backend",
            "scipy",
        ],
    )
    monkeypatch.setattr(loop.jax, "devices", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(loop, "resolve_native_selections", resolve)

    with pytest.raises(RuntimeError) as caught:
        loop.main()

    assert caught.value is stop_after_final
    assert resolution_calls == [
        (False, "jax", "scipy", False),
        (False, "jax", "scipy", False),
    ]


def test_p2_native_log_and_effective_selections_agree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Resolved `.p2` provenance must agree in params, logs, and loop inputs."""
    from tommos import io_utils, loop

    model = tmp_path / "portable-model"
    shutil.copy(Path(__file__).with_name("single_solid.npz"), model.with_suffix(".npz"))
    model.with_suffix(".p2").write_text(
        "[minimizer]\ncpp_mkl = false\n[poisson]\npoisson_solver = jax\ncpu_spmv_backend = scipy\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    selections = _native_loader.NativeSelections("python", "jax", "scipy")
    resolution_calls: list[tuple[bool | None, str, str]] = []
    loop_calls: list[dict[str, Any]] = []

    def resolve(
        cpp_mkl: bool | None,
        poisson_solver: str,
        cpu_spmv_backend: str,
        *,
        has_gpu: bool,
    ) -> _native_loader.NativeSelections:
        """Resolve only the expected portable `.p2` request.

        Args:
            cpp_mkl: Requested minimizer choice.
            poisson_solver: Requested Poisson choice.
            cpu_spmv_backend: Requested CPU sparse backend.
            has_gpu: Whether GPU execution is available.

        Returns:
            Expected portable authoritative selections.
        """
        assert not has_gpu
        resolution_calls.append((cpp_mkl, poisson_solver, cpu_spmv_backend))
        return selections

    def run_loop(**kwargs: Any) -> dict[str, Any]:
        """Record authoritative loop inputs and return an empty history.

        Args:
            **kwargs: Hysteresis-loop keyword arguments.

        Returns:
            Minimal result consumed by the CLI.
        """
        loop_calls.append(kwargs)
        return {"history": []}

    monkeypatch.setattr(sys, "argv", ["tommos.loop", str(model), "--out-dir", str(output_dir)])
    monkeypatch.setattr(loop.jax, "devices", lambda *_args, **_kwargs: [SimpleNamespace(platform="cpu")])
    monkeypatch.setattr(loop, "resolve_native_selections", resolve)
    monkeypatch.setattr(loop, "run_hysteresis_loop", run_loop)
    monkeypatch.setattr(loop, "write_mh", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(io_utils, "convert_sim_csv_to_mammos", lambda *_args, **_kwargs: None)

    loop.main()

    assert resolution_calls == [(False, "jax", "scipy")]
    assert len(loop_calls) == 1
    assert loop_calls[0]["native_selections"] == selections
    assert loop_calls[0]["cpu_spmv_backend"] == "scipy"
    assert loop_calls[0]["params"].cpp_mkl is False
    assert loop_calls[0]["params"].poisson_solver == "jax"
    parameter_log = (output_dir / "params.log").read_text(encoding="utf-8")
    assert "| cpp_mkl | False | .p2 |" in parameter_log
    assert "| poisson_solver | jax | .p2 |" in parameter_log
    assert "| cpu_spmv_backend | scipy | .p2 |" in parameter_log


def test_pardiso_owner_retains_csr_storage_and_closes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The PARDISO owner must retain normalized CSR arrays and free one time."""
    freed: list[int] = []
    library = object()
    bindings = _native_loader.PardisoBindings(
        init_pardiso=lambda *_args: 17,
        pardiso_solve_direct=lambda *_args: 0,
        free_pardiso=lambda handle_id: freed.append(handle_id),
        library=library,
    )
    monkeypatch.setattr(_native_loader, "require_pardiso", lambda: bindings)
    values = np.arange(8, dtype=np.float32)[::2]
    indptr = np.arange(6, dtype=np.int64)[::2]
    indices = np.arange(8, dtype=np.int64)[::2]

    owner = _native_loader.create_pardiso_handle(2, values, indptr, indices)

    assert owner.handle_id == 17
    assert owner.values.dtype == np.float64 and owner.values.flags.c_contiguous
    assert owner.indptr.dtype == np.int32 and owner.indptr.flags.c_contiguous
    assert owner.indices.dtype == np.int32 and owner.indices.flags.c_contiguous
    assert owner.bindings is bindings
    assert owner.library is library
    owner.close()
    owner.close()
    assert freed == [17]


def test_pardiso_close_waits_for_active_solve_and_frees_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PARDISO close must serialize behind an active solve on the same handle."""
    solve_started = threading.Event()
    release_solve = threading.Event()
    close_started = threading.Event()
    close_done = threading.Event()
    freed: list[int] = []

    def solve_direct(*_args: Any) -> int:
        """Block a fake native solve until the test releases it.

        Args:
            *_args: Ignored native solve arguments.

        Returns:
            Zero success status.
        """
        solve_started.set()
        if not release_solve.wait(timeout=2.0):
            raise AssertionError("test did not release the native solve")
        return 0

    bindings = _native_loader.PardisoBindings(
        init_pardiso=lambda *_args: 23,
        pardiso_solve_direct=solve_direct,
        free_pardiso=lambda handle_id: freed.append(handle_id),
        library=object(),
    )
    monkeypatch.setattr(_native_loader, "require_pardiso", lambda: bindings)
    owner = _native_loader.create_pardiso_handle(
        1,
        np.array([1.0]),
        np.array([0, 1]),
        np.array([0]),
    )

    solve_thread = threading.Thread(target=owner.solve, args=(np.array([1.0]),))

    def close_owner() -> None:
        """Close the owner and signal completion."""
        close_started.set()
        owner.close()
        close_done.set()

    close_thread = threading.Thread(target=close_owner)
    solve_thread.start()
    assert solve_started.wait(timeout=1.0)
    close_thread.start()
    assert close_started.wait(timeout=1.0)
    try:
        assert not close_done.wait(timeout=0.05)
        assert freed == []
    finally:
        release_solve.set()
        solve_thread.join(timeout=1.0)
        close_thread.join(timeout=1.0)
    owner.close()

    assert not solve_thread.is_alive()
    assert not close_thread.is_alive()
    assert freed == [23]


def test_persistent_mkl_operator_uses_loader_inspector_and_wrapper_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspect once via the loader while create, execute, and destroy stay wrapper-owned."""
    from tommos import amg_utils

    calls: list[str] = []
    wrapper = ModuleType("sparse_dot_mkl._mkl_interface")
    matrix = sp.csr_matrix(np.array([[2.0, 0.0], [1.0, 3.0]]))

    class FakeMKL:
        """Sparse executor stand-in."""

        @staticmethod
        def _mkl_sparse_d_mv(
            _operation: int,
            _scalar: float,
            handle: sp.csr_matrix,
            _description: object,
            vector: np.ndarray,
            _out_scalar: float,
            output: np.ndarray,
        ) -> int:
            """Apply the retained SciPy matrix.

            Args:
                _operation: Ignored operation identifier.
                _scalar: Ignored input scalar.
                handle: Retained sparse matrix.
                _description: Ignored matrix description.
                vector: Dense input vector.
                _out_scalar: Ignored output scalar.
                output: Dense output buffer.

            Returns:
                Zero success status.
            """
            calls.append("execute")
            output[:] = handle @ vector
            return 0

    wrapper.MKL = FakeMKL
    wrapper._create_mkl_sparse = lambda value: (calls.append("create") or value, True, False)
    wrapper._destroy_mkl_handle = lambda _handle: calls.append("destroy")
    wrapper._mkl_scalar = lambda value, _cplx, _dbl: value
    wrapper._out_matrix = lambda shape, dtype: np.empty(shape, dtype=dtype)
    wrapper._output_dtypes = {(True, False): np.float64}
    wrapper.matrix_descr = object
    active_loader = sys.modules["tommos._native_loader"]
    monkeypatch.setattr(active_loader, "require_sparse_dot_mkl", lambda: wrapper, raising=False)
    monkeypatch.setattr(
        active_loader,
        "mkl_sparse_set_mv_hint",
        lambda _handle, expected_calls: calls.append(f"hint:{expected_calls}") or 0,
        raising=False,
    )
    monkeypatch.setattr(
        active_loader,
        "mkl_sparse_optimize",
        lambda _handle: calls.append("optimize") or 0,
        raising=False,
    )

    operator = amg_utils.PersistentMKLOperator(matrix)
    result = operator.apply(np.array([4.0, 5.0]))
    operator.close()
    operator.close()

    np.testing.assert_allclose(result, np.array([8.0, 19.0]))
    assert calls == ["create", "hint:1000", "optimize", "execute", "destroy"]


@pytest.mark.parametrize(
    ("hint_status", "optimize_status", "execute_status", "operation", "status"),
    [
        (3, 0, 0, "mkl_sparse_set_mv_hint", 3),
        (0, 4, 0, "mkl_sparse_optimize", 4),
        (0, 0, 5, "mkl_sparse_d_mv", 5),
    ],
)
def test_persistent_mkl_operator_reports_sparse_status_failures(
    monkeypatch: pytest.MonkeyPatch,
    hint_status: int,
    optimize_status: int,
    execute_status: int,
    operation: str,
    status: int,
) -> None:
    """Raise a clear error and destroy the wrapper handle for nonzero MKL status."""
    from tommos import amg_utils

    destroyed: list[object] = []
    wrapper = ModuleType("sparse_dot_mkl._mkl_interface")
    handle = object()

    class FakeMKL:
        """Sparse executor returning a controlled status."""

        @staticmethod
        def _mkl_sparse_d_mv(*_args: Any) -> int:
            """Return the controlled execution status.

            Args:
                *_args: Ignored wrapper execution arguments.

            Returns:
                Controlled sparse status.
            """
            return execute_status

    wrapper.MKL = FakeMKL
    wrapper._create_mkl_sparse = lambda _value: (handle, True, False)
    wrapper._destroy_mkl_handle = lambda value: destroyed.append(value)
    wrapper._mkl_scalar = lambda value, _cplx, _dbl: value
    wrapper._out_matrix = lambda shape, dtype: np.empty(shape, dtype=dtype)
    wrapper._output_dtypes = {(True, False): np.float64}
    wrapper.matrix_descr = object
    active_loader = sys.modules["tommos._native_loader"]
    monkeypatch.setattr(active_loader, "require_sparse_dot_mkl", lambda: wrapper)
    monkeypatch.setattr(
        active_loader,
        "mkl_sparse_set_mv_hint",
        lambda _handle, expected_calls: hint_status,
        raising=False,
    )
    monkeypatch.setattr(active_loader, "mkl_sparse_optimize", lambda _handle: optimize_status, raising=False)

    if hint_status != 0 or optimize_status != 0:
        with pytest.raises(RuntimeError, match=rf"{operation} failed with sparse_status_t {status}"):
            amg_utils.PersistentMKLOperator(sp.eye(1, format="csr"))
    else:
        operator = amg_utils.PersistentMKLOperator(sp.eye(1, format="csr"))
        with pytest.raises(RuntimeError, match=rf"{operation} failed with sparse_status_t {status}"):
            operator.apply(np.array([1.0]))
        operator.close()

    assert destroyed == [handle]


def test_persistent_mkl_operator_cleans_up_late_setup_failure_without_masking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Destroy once after late setup failure and preserve its original exception."""
    from tommos import amg_utils

    setup_error = RuntimeError("output dtype setup failed")
    cleanup_error = RuntimeError("destroy failed")
    destroyed: list[object] = []
    finalized_closed_states: list[bool] = []
    wrapper = ModuleType("sparse_dot_mkl._mkl_interface")
    handle = object()

    class FailingOutputDtypes:
        """Mapping stand-in that fails during late operator initialization."""

        def __getitem__(self, _key: tuple[bool, bool]) -> Any:
            """Raise the controlled setup error.

            Args:
                _key: Precision and complexity flags.

            Raises:
                RuntimeError: Always, to emulate late setup failure.
            """
            raise setup_error

    def destroy_handle(value: object) -> None:
        """Record one synchronous cleanup attempt and then fail.

        Args:
            value: Wrapper-created sparse handle.

        Raises:
            RuntimeError: Always, to verify setup-error preservation.
        """
        destroyed.append(value)
        raise cleanup_error

    wrapper.MKL = object()
    wrapper._create_mkl_sparse = lambda _value: (handle, True, False)
    wrapper._destroy_mkl_handle = destroy_handle
    wrapper._output_dtypes = FailingOutputDtypes()
    active_loader = sys.modules["tommos._native_loader"]
    monkeypatch.setattr(active_loader, "require_sparse_dot_mkl", lambda: wrapper)
    monkeypatch.setattr(
        active_loader,
        "mkl_sparse_set_mv_hint",
        lambda _handle, expected_calls: 0,
    )
    monkeypatch.setattr(active_loader, "mkl_sparse_optimize", lambda _handle: 0)
    original_finalizer = amg_utils.PersistentMKLOperator.__del__

    def record_finalizer(operator: Any) -> None:
        """Record closed state and run the real destructor fallback.

        Args:
            operator: Partially initialized persistent operator.
        """
        finalized_closed_states.append(operator._closed)
        original_finalizer(operator)

    monkeypatch.setattr(amg_utils.PersistentMKLOperator, "__del__", record_finalizer)

    with pytest.raises(RuntimeError) as raised:
        amg_utils.PersistentMKLOperator(sp.eye(1, format="csr"))

    assert raised.value is setup_error
    assert destroyed == [handle]
    raised.value.__traceback__ = None
    cleanup_error.__traceback__ = None
    cleanup_error.__context__ = None
    del raised
    gc.collect()
    assert finalized_closed_states == [True]


def test_persistent_mkl_close_waits_for_active_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sparse wrapper destruction must serialize behind an active execution."""
    from tommos import amg_utils

    apply_started = threading.Event()
    release_apply = threading.Event()
    close_done = threading.Event()
    calls: list[str] = []
    wrapper = ModuleType("sparse_dot_mkl._mkl_interface")
    matrix = sp.eye(1, format="csr")

    class FakeMKL:
        """Blocking sparse executor stand-in."""

        @staticmethod
        def _mkl_sparse_d_mv(*_args: Any) -> int:
            """Block execution until the test releases it.

            Args:
                *_args: Ignored wrapper execution arguments.

            Returns:
                Zero success status.
            """
            apply_started.set()
            if not release_apply.wait(timeout=2.0):
                raise AssertionError("test did not release sparse execution")
            return 0

    wrapper.MKL = FakeMKL
    wrapper._create_mkl_sparse = lambda value: (value, True, False)
    wrapper._destroy_mkl_handle = lambda _handle: calls.append("destroy")
    wrapper._mkl_scalar = lambda value, _cplx, _dbl: value
    wrapper._out_matrix = lambda shape, dtype: np.empty(shape, dtype=dtype)
    wrapper._output_dtypes = {(True, False): np.float64}
    wrapper.matrix_descr = object
    active_loader = sys.modules["tommos._native_loader"]
    monkeypatch.setattr(active_loader, "require_sparse_dot_mkl", lambda: wrapper)
    monkeypatch.setattr(
        active_loader,
        "mkl_sparse_set_mv_hint",
        lambda _handle, expected_calls: 0,
        raising=False,
    )
    monkeypatch.setattr(active_loader, "mkl_sparse_optimize", lambda _handle: 0, raising=False)
    operator = amg_utils.PersistentMKLOperator(matrix)

    apply_thread = threading.Thread(target=operator.apply, args=(np.array([1.0]),))

    def close_operator() -> None:
        """Close the sparse operator and signal completion."""
        operator.close()
        close_done.set()

    close_thread = threading.Thread(target=close_operator)
    apply_thread.start()
    assert apply_started.wait(timeout=1.0)
    close_thread.start()
    try:
        assert not close_done.wait(timeout=0.05)
        assert calls == []
    finally:
        release_apply.set()
        apply_thread.join(timeout=1.0)
        close_thread.join(timeout=1.0)
    operator.close()

    assert not apply_thread.is_alive()
    assert not close_thread.is_alive()
    assert calls == ["destroy"]


def test_concrete_cpu_factory_does_not_reresolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concrete CPU backend must execute without re-entering global selection."""
    from tommos import amg_utils

    active_loader = sys.modules["tommos._native_loader"]
    monkeypatch.setattr(
        active_loader,
        "resolve_native_selections",
        lambda *_args, **_kwargs: pytest.fail("concrete backend was re-resolved"),
    )

    operation = amg_utils.make_cpu_csr_op(sp.eye(1, format="csr"), cpu_spmv_backend="scipy")

    assert operation is not None


def test_concrete_poisson_factory_does_not_reresolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authoritative Poisson and SpMV choices must bypass global selection."""
    from tommos import poisson_solve

    active_loader = sys.modules["tommos._native_loader"]
    monkeypatch.setattr(
        active_loader,
        "resolve_native_selections",
        lambda *_args, **_kwargs: pytest.fail("concrete choices were re-resolved"),
    )
    monkeypatch.setattr(
        poisson_solve,
        "make_poisson_ops",
        lambda *_args, **_kwargs: (
            lambda _ops, value: value,
            lambda _ops, value: value[:, 0],
            lambda _ops, size: np.ones(size),
        ),
    )
    monkeypatch.setattr(poisson_solve, "make_pcg_solve", lambda *_args, **_kwargs: object())
    geom = SimpleNamespace(x_nodes=np.zeros((1, 3)), conn=np.zeros((1, 4), dtype=np.int32))

    solve = poisson_solve.make_solve_U(
        geom,
        np.array([1.0]),
        poisson_solver="jax",
        cpu_spmv_backend="scipy",
        native_selections=_native_loader.NativeSelections("python", "jax", "scipy"),
    )

    assert solve is not None


def test_direct_explicit_pardiso_fails_before_operator_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A direct strict PARDISO request must fail before Poisson setup."""
    from tommos import poisson_solve

    original_error = RuntimeError("PARDISO unavailable")
    active_loader = sys.modules["tommos._native_loader"]
    monkeypatch.setattr(
        active_loader,
        "resolve_native_selections",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(original_error),
    )
    monkeypatch.setattr(
        poisson_solve,
        "make_poisson_ops",
        lambda *_args, **_kwargs: pytest.fail("Poisson operators were assembled before strict validation"),
    )
    geom = SimpleNamespace(x_nodes=np.zeros((1, 3)), conn=np.zeros((1, 4), dtype=np.int32))

    with pytest.raises(RuntimeError) as caught:
        poisson_solve.make_solve_U(
            geom,
            np.array([1.0]),
            poisson_solver="pardiso",
            cpu_spmv_backend="scipy",
        )

    assert caught.value is original_error


def test_sparse_operator_forwards_idempotent_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The principal sparse factory must preserve callback resource ownership."""
    from tommos import amg_utils

    closed: list[str] = []

    def operation(value: Any) -> Any:
        """Return an input value for the factory stand-in.

        Args:
            value: Input value.

        Returns:
            The unchanged input value.
        """
        return value

    operation.close = lambda: closed.append("close")
    monkeypatch.setattr(amg_utils.jax, "devices", lambda *_args, **_kwargs: [SimpleNamespace(platform="cpu")])
    monkeypatch.setattr(amg_utils, "make_cpu_csr_op", lambda *_args, **_kwargs: operation)

    operator = amg_utils.make_sparse_operator(sp.eye(1, format="csr"), cpu_spmv_backend="scipy")
    operator.close()
    operator.close()

    assert closed == ["close"]


def test_authoritative_loop_selection_bypasses_resolution_without_mutating_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller-provided selection must be authoritative and preserve LoopParams."""
    from tommos import hysteresis_loop

    params = hysteresis_loop.LoopParams(
        h_dir=np.array([0.0, 0.0, 1.0]),
        B_start=0.0,
        B_end=0.0,
        dB=1.0,
    )
    selections = _native_loader.NativeSelections("python", "jax", "scipy")
    original_error = RuntimeError("stop after selection")
    active_loader = sys.modules["tommos._native_loader"]

    def fail_resolution(*_args: Any, **_kwargs: Any) -> Any:
        """Fail if authoritative selection re-enters resolution.

        Args:
            *_args: Unexpected positional resolver arguments.
            **_kwargs: Unexpected keyword resolver arguments.

        Returns:
            This function never returns.
        """
        pytest.fail("authoritative selection was ignored")

    monkeypatch.setattr(
        active_loader,
        "resolve_native_selections",
        fail_resolution,
    )
    monkeypatch.setattr(hysteresis_loop, "resolve_native_selections", fail_resolution, raising=False)
    monkeypatch.setattr(
        hysteresis_loop,
        "ensure_dir",
        lambda _path: (_ for _ in ()).throw(original_error),
    )

    with pytest.raises(RuntimeError) as caught:
        hysteresis_loop.run_hysteresis_loop(
            points=np.empty((0, 3)),
            geom=None,
            A_lookup=np.empty(0),
            K1_lookup=np.empty(0),
            Js_lookup=np.empty(0),
            k_easy_lookup=np.empty((0, 3)),
            m0=np.empty((0, 3)),
            params=params,
            V_mag=1.0,
            node_volumes=np.empty(0),
            M_nodal=np.empty(0),
            native_selections=selections,
        )

    assert caught.value is original_error
    assert params.cpp_mkl is None
    assert params.poisson_solver == "auto"


def test_successful_loop_output_finalization_closes_pardiso_after_sync() -> None:
    """Successful loop finalization must synchronize outputs before explicit close."""
    from tommos import hysteresis_loop

    events: list[str] = []

    class Output:
        """Synchronizable output stand-in."""

        def __init__(self, name: str) -> None:
            """Initialize an output stand-in.

            Args:
                name: Event label for synchronization and conversion.
            """
            self.name = name

        def block_until_ready(self) -> None:
            """Record output synchronization."""
            events.append(f"sync-{self.name}")

        def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
            """Convert to a NumPy array after synchronization.

            Args:
                dtype: Optional requested NumPy dtype.
                copy: Optional NumPy copy request.

            Returns:
                One-element output array.
            """
            events.append(f"array-{self.name}")
            return np.asarray([1.0], dtype=dtype)

    owner = SimpleNamespace(close=lambda: events.append("close-pardiso"))
    solver = SimpleNamespace(pardiso_obj=owner)
    sparse_operator = SimpleNamespace(close=lambda: events.append("close-sparse"))

    last_m, last_U = hysteresis_loop._finalize_loop_outputs(
        Output("m"),
        Output("U"),
        solver,
        (sparse_operator, sparse_operator),
    )

    np.testing.assert_array_equal(last_m, np.array([1.0]))
    np.testing.assert_array_equal(last_U, np.array([1.0]))
    assert events == [
        "sync-m",
        "sync-U",
        "array-m",
        "array-U",
        "close-pardiso",
        "close-sparse",
    ]
