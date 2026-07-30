"""Focused lifetime and status-boundary tests for the C++ minimizer wrapper."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp

from tommos import _native_loader
from tommos.cpp_minimizer import cpp_minimize


def _minimizer_arguments(owner: _native_loader.PardisoHandle | None) -> dict[str, Any]:
    """Create the smallest valid Python-side minimizer argument set.

    Args:
        owner: Optional PARDISO owner exposed through the Poisson solver.

    Returns:
        Keyword arguments accepted by :func:`cpp_minimize`.
    """
    identity = sp.eye(3, format="csr", dtype=np.float64)
    divergence = sp.csr_matrix(np.array([[1.0, 0.0, 0.0]], dtype=np.float64))
    params = SimpleNamespace(
        M_nodal=np.array([1.0], dtype=np.float64),
        V_mag=1.0,
        inv_M_prec=np.array([1.0], dtype=np.float64),
        max_iter=0,
        L=None,
    )
    solve_u = None if owner is None else SimpleNamespace(pardiso_obj=owner)
    return {
        "m": np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        "B_ext": np.zeros(3, dtype=np.float64),
        "U0": np.zeros(1, dtype=np.float64),
        "params": params,
        "sparse_ops": {
            "K_eff_sparse": identity,
            "D_sparse": divergence,
            "G_sparse": divergence.transpose().tocsr(),
        },
        "solve_U": solve_u,
    }


def _pardiso_owner(handle_id: int, freed: list[int]) -> _native_loader.PardisoHandle:
    """Create a PARDISO owner backed by deterministic fake bindings.

    Args:
        handle_id: Strictly positive native identifier.
        freed: List that records native release calls.

    Returns:
        Resource owner suitable for minimizer lifetime tests.
    """
    bindings = _native_loader.PardisoBindings(
        init_pardiso=lambda *_args: handle_id,
        pardiso_solve_direct=lambda *_args: 0,
        free_pardiso=lambda released_id: freed.append(released_id),
        library=object(),
    )
    return _native_loader.PardisoHandle(
        handle_id,
        np.array([1.0], dtype=np.float64),
        np.array([0, 1], dtype=np.int32),
        np.array([0], dtype=np.int32),
        bindings,
    )


@pytest.mark.parametrize("native_id", [-1, 0])
def test_pardiso_factory_rejects_every_nonpositive_native_id(
    monkeypatch: pytest.MonkeyPatch,
    native_id: int,
) -> None:
    """A failed or invalid native initialization must never become an owner."""
    bindings = _native_loader.PardisoBindings(
        init_pardiso=lambda *_args: native_id,
        pardiso_solve_direct=lambda *_args: 0,
        free_pardiso=lambda _handle_id: pytest.fail("invalid native ID became a live owner"),
        library=object(),
    )
    monkeypatch.setattr(_native_loader, "require_pardiso", lambda: bindings)

    with pytest.raises(
        RuntimeError,
        match=rf"native status {native_id}; expected a strictly positive handle ID",
    ):
        _native_loader.create_pardiso_handle(
            1,
            np.array([1.0]),
            np.array([0, 1]),
            np.array([0]),
        )


def test_pardiso_close_waits_for_complete_cpp_minimizer_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing a borrowed owner must wait until the native minimizer returns."""
    native_started = threading.Event()
    release_native = threading.Event()
    close_started = threading.Event()
    close_done = threading.Event()
    freed: list[int] = []
    failures: list[BaseException] = []
    owner = _pardiso_owner(31, freed)

    def minimizer(*_args: Any) -> int:
        """Block the fake native minimizer until the test permits return.

        Args:
            *_args: Native minimizer arguments.

        Returns:
            Zero success status.
        """
        native_started.set()
        if not release_native.wait(timeout=2.0):
            raise AssertionError("test did not release the native minimizer")
        return 0

    def run_minimizer() -> None:
        """Run the wrapper and retain any thread exception."""
        try:
            cpp_minimize(**_minimizer_arguments(owner))
        except BaseException as error:
            failures.append(error)

    def close_owner() -> None:
        """Close the owner and signal when native release completes."""
        close_started.set()
        owner.close()
        close_done.set()

    monkeypatch.setattr(_native_loader, "require_cpp_minimizer", lambda: minimizer)
    minimizer_thread = threading.Thread(target=run_minimizer)
    close_thread = threading.Thread(target=close_owner)
    minimizer_thread.start()
    assert native_started.wait(timeout=1.0)
    close_thread.start()
    assert close_started.wait(timeout=1.0)
    try:
        assert not close_done.wait(timeout=0.05)
        assert freed == []
    finally:
        release_native.set()
        minimizer_thread.join(timeout=3.0)
        close_thread.join(timeout=3.0)

    assert not minimizer_thread.is_alive()
    assert not close_thread.is_alive()
    assert failures == []
    assert freed == [31]


def test_closed_pardiso_owner_fails_before_native_minimizer_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A closed owner must be rejected without entering the C++ minimizer."""
    freed: list[int] = []
    owner = _pardiso_owner(37, freed)
    native_calls: list[str] = []
    owner.close()
    monkeypatch.setattr(
        _native_loader,
        "require_cpp_minimizer",
        lambda: lambda *_args: native_calls.append("called"),
    )

    with pytest.raises(RuntimeError, match="PARDISO handle is closed"):
        cpp_minimize(**_minimizer_arguments(owner))

    assert native_calls == []
    assert freed == [37]


@pytest.mark.parametrize(
    ("operation", "native_status"),
    [
        ("mkl_sparse_d_create_csr", 11),
        ("mkl_sparse_optimize", 12),
        ("mkl_sparse_d_mv", 13),
        ("pardiso_solve_direct", -14),
        ("mkl_sparse_destroy", 15),
    ],
)
def test_cpp_minimizer_raises_for_every_nonzero_native_status_without_owner(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    native_status: int,
) -> None:
    """The Python boundary must reject every propagated native operation status."""
    received_calls: list[tuple[str, int]] = []

    def minimizer(*args: Any) -> int:
        """Record the null PARDISO ID and return a controlled failure.

        Args:
            *args: Native minimizer arguments.

        Returns:
            Controlled nonzero status.
        """
        received_calls.append((operation, int(args[10])))
        return native_status

    monkeypatch.setattr(_native_loader, "require_cpp_minimizer", lambda: minimizer)

    with pytest.raises(
        RuntimeError,
        match=rf"C\+\+ minimizer failed with native status {native_status}",
    ):
        cpp_minimize(**_minimizer_arguments(None))

    assert received_calls == [(operation, 0)]
