"""Tests for lazy discovery and capability-specific native bindings."""

import ctypes
import importlib
import shutil
import sys
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import pytest


class FakeFunction:
    """Callable native symbol stand-in with assignable ctypes metadata.

    Args:
        result: Value returned when the stand-in is called.
    """

    def __init__(self, result: int = 0) -> None:
        """Initialize the stand-in.

        Args:
            result: Value returned when the stand-in is called.
        """
        self.argtypes: list[Any] | None = None
        self.restype: Any = None
        self.result = result

    def __call__(self, *_args: Any) -> int:
        """Return the configured result.

        Args:
            *_args: Ignored call arguments.

        Returns:
            The configured integer result.
        """
        return self.result


class FakeLibrary:
    """Attribute-tracking native library stand-in.

    Args:
        abi_version: ABI version returned by the ABI symbol, if present.
        symbols: Mapping of operation-symbol names to fake functions.
        include_abi: Whether the ABI symbol is exported.
    """

    def __init__(self, abi_version: int, symbols: dict[str, FakeFunction], include_abi: bool = True) -> None:
        """Initialize the fake library.

        Args:
            abi_version: ABI version returned by the ABI symbol, if present.
            symbols: Mapping of operation-symbol names to fake functions.
            include_abi: Whether the ABI symbol is exported.
        """
        self.accessed: list[str] = []
        self._symbols = symbols
        if include_abi:
            self._symbols = {"tommos_native_abi_version": FakeFunction(abi_version), **symbols}

    def __getattr__(self, name: str) -> FakeFunction:
        """Return an exported fake symbol or record a missing lookup.

        Args:
            name: Native symbol name to retrieve.

        Returns:
            The matching fake function.

        Raises:
            AttributeError: If the symbol is not exported.
        """
        self.accessed.append(name)
        try:
            return self._symbols[name]
        except KeyError as error:
            raise AttributeError(name) from error


class FakeDistribution:
    """Minimal distribution metadata fake with owned-file locations.

    Args:
        root: Directory relative to which distribution files are located.
        files: Distribution-owned file entries.
    """

    def __init__(self, root: Path, files: list[str]) -> None:
        """Initialize the fake distribution.

        Args:
            root: Directory relative to which distribution files are located.
            files: Distribution-owned file entries.
        """
        self._root = root
        self.files = [PurePosixPath(entry) for entry in files]

    def locate_file(self, path: PurePosixPath) -> Path:
        """Locate one distribution-owned file.

        Args:
            path: Distribution file entry to locate.

        Returns:
            Path represented by the entry.
        """
        return self._root / path


def _import_loader() -> ModuleType:
    """Import a fresh copy of the loader without using importlib.import_module.

    Returns:
        The freshly imported native-loader module.
    """
    sys.modules.pop("tommos._native_loader", None)
    __import__("tommos._native_loader", fromlist=["*"])
    return sys.modules["tommos._native_loader"]


def _materialize_owned_files(distribution: FakeDistribution) -> None:
    """Create every file represented by fake distribution metadata.

    Args:
        distribution: Fake whose owned files should exist on disk.
    """
    for entry in distribution.files:
        path = distribution.locate_file(entry).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _native_symbols() -> dict[str, FakeFunction]:
    """Create all operation symbols exported by a complete fake library.

    Returns:
        Mapping of every Task 1 operation symbol to a fake function.
    """
    return {
        "run_cpp_pcohen_hs_minimization": FakeFunction(),
        "init_pardiso": FakeFunction(),
        "pardiso_solve_direct": FakeFunction(),
        "free_pardiso": FakeFunction(),
    }


def _configure_automatic_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> tuple[Path, Path, Path, Path]:
    """Create controllable Slurm, environment, package, and repository candidates.

    Args:
        monkeypatch: Pytest environment and attribute patch helper.
        tmp_path: Per-test temporary directory.
        request: Pytest request used to remove the temporary Slurm directory.

    Returns:
        Slurm, environment, package-resource, and repository library paths.
    """
    slurm_id = f"tommos-loader-{tmp_path.name}"
    slurm_dir = Path("/tmp") / f"mumag_build_{slurm_id}"
    slurm_dir.mkdir(parents=True)

    def remove_slurm_directory() -> None:
        """Remove the test-owned Slurm candidate directory after the test."""
        shutil.rmtree(slurm_dir)

    request.addfinalizer(remove_slurm_directory)
    slurm_path = slurm_dir / "libcpp_mkl_minimizer.so"
    slurm_path.touch()
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    env_path = env_dir / "libcpp_mkl_minimizer.so"
    env_path.touch()
    resource_path = tmp_path / "resources" / "_native" / "libcpp_mkl_minimizer.so"
    resource_path.parent.mkdir(parents=True)
    resource_path.touch()
    repository_path = tmp_path / "repository" / "libcpp_mkl_minimizer.so"
    repository_path.parent.mkdir()
    repository_path.touch()
    monkeypatch.setenv("SLURM_JOB_ID", slurm_id)
    monkeypatch.setenv("MUMAG_LIB_OUT", str(env_dir))
    return slurm_path, env_path, resource_path, repository_path


def test_import_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the loader must not load libraries or optional packages."""
    cdll_calls: list[str] = []
    import_calls: list[str] = []

    def record_cdll(path: str) -> Any:
        """Record an unexpected library load.

        Args:
            path: Requested shared-library path.

        Returns:
            This function always raises before returning.
        """
        cdll_calls.append(path)
        raise AssertionError("native loading is not allowed during module import")

    def record_import(name: str, *args: Any, **kwargs: Any) -> ModuleType:
        """Record an unexpected optional-package import.

        Args:
            name: Requested module name.
            *args: Additional import arguments.
            **kwargs: Additional import keyword arguments.

        Returns:
            This function always raises before returning.
        """
        import_calls.append(name)
        raise AssertionError("importlib.import_module is not allowed during module import")

    monkeypatch.setattr(ctypes, "CDLL", record_cdll)
    monkeypatch.setattr(importlib, "import_module", record_import)

    loader = _import_loader()

    assert loader.NATIVE_ABI_VERSION == 1
    assert cdll_calls == []
    assert import_calls == []


def test_automatic_candidate_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """Discovery must select the first valid automatic candidate in documented order."""
    slurm_path, env_path, resource_path, repository_path = _configure_automatic_candidates(
        monkeypatch, tmp_path, request
    )
    resource_root = resource_path.parents[1]
    fake_library = FakeLibrary(1, _native_symbols())
    loaded_paths: list[Path] = []

    def load(path: str) -> FakeLibrary:
        """Record the selected path and return an ABI-compatible library.

        Args:
            path: Candidate path selected by the loader.

        Returns:
            ABI-compatible fake library.
        """
        loaded_paths.append(Path(path))
        return fake_library

    def configured_loader() -> ModuleType:
        """Import and configure an isolated loader instance.

        Returns:
            Loader instance with all automatic-candidate dependencies patched.
        """
        result = _import_loader()
        monkeypatch.setattr(result.ctypes, "CDLL", load)
        monkeypatch.setattr(result, "load_mkl_runtime", lambda: object())
        monkeypatch.setattr(result.resources, "files", lambda _package: resource_root)
        monkeypatch.setattr(result, "_repository_library_path", lambda: repository_path)
        return result

    loader = configured_loader()

    assert loader.probe_cpp_minimizer().path == slurm_path
    assert loaded_paths == [slurm_path]

    monkeypatch.delenv("SLURM_JOB_ID")
    loader = configured_loader()
    assert loader.probe_cpp_minimizer().path == env_path

    monkeypatch.delenv("MUMAG_LIB_OUT")
    loader = configured_loader()
    assert loader.probe_cpp_minimizer().path == resource_path

    resource_path.unlink()
    loader = configured_loader()
    assert loader.probe_cpp_minimizer().path == repository_path


def test_explicit_candidate_requires_absolute_existing_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit invalid path must fail without automatic fallback loading."""
    loader = _import_loader()
    cdll_calls: list[str] = []
    monkeypatch.setenv("TOMMOS_NATIVE_LIBRARY", "relative/library.so")
    monkeypatch.setattr(loader.ctypes, "CDLL", lambda path: cdll_calls.append(path))

    probe = loader.probe_cpp_minimizer()

    assert not probe.available
    assert isinstance(probe.error, ValueError)
    assert cdll_calls == []
    with pytest.raises(ValueError):
        loader.require_cpp_minimizer()


def test_explicit_candidate_requires_existing_absolute_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An absolute explicit path that does not exist must fail without fallback."""
    loader = _import_loader()
    missing_path = tmp_path / "missing" / "library.so"
    cdll_calls: list[str] = []
    monkeypatch.setenv("TOMMOS_NATIVE_LIBRARY", str(missing_path))
    monkeypatch.setattr(loader.ctypes, "CDLL", lambda path: cdll_calls.append(path))

    probe = loader.probe_cpp_minimizer()

    assert not probe.available
    assert isinstance(probe.error, FileNotFoundError)
    assert cdll_calls == []
    with pytest.raises(FileNotFoundError):
        loader.require_cpp_minimizer()


def test_explicit_candidate_precedes_every_automatic_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """A valid absolute explicit candidate must outrank all automatic candidates."""
    loader = _import_loader()
    slurm_path, _env_path, resource_path, repository_path = _configure_automatic_candidates(
        monkeypatch, tmp_path, request
    )
    explicit_path = tmp_path / "explicit" / "library.so"
    explicit_path.parent.mkdir()
    explicit_path.touch()
    loaded_paths: list[Path] = []
    fake_library = FakeLibrary(1, _native_symbols())

    def load(path: str) -> FakeLibrary:
        """Record the selected candidate and return an ABI-compatible library.

        Args:
            path: Candidate path selected by the loader.

        Returns:
            ABI-compatible fake library.
        """
        loaded_paths.append(Path(path))
        return fake_library

    monkeypatch.setenv("TOMMOS_NATIVE_LIBRARY", str(explicit_path))
    monkeypatch.setattr(loader.ctypes, "CDLL", load)
    monkeypatch.setattr(loader, "load_mkl_runtime", lambda: object())
    monkeypatch.setattr(loader.resources, "files", lambda _package: resource_path.parents[1])
    monkeypatch.setattr(loader, "_repository_library_path", lambda: repository_path)

    probe = loader.probe_cpp_minimizer()

    assert probe.available
    assert probe.path == explicit_path
    assert probe.source == "TOMMOS_NATIVE_LIBRARY"
    assert loaded_paths == [explicit_path]
    assert slurm_path != explicit_path


def test_automatic_candidate_continues_after_load_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """An automatic load error must not prevent selecting a later candidate."""
    loader = _import_loader()
    slurm_path, env_path, _resource_path, repository_path = _configure_automatic_candidates(
        monkeypatch, tmp_path, request
    )
    fake_library = FakeLibrary(1, _native_symbols())

    def load(path: str) -> FakeLibrary:
        """Reject the Slurm candidate and accept the environment candidate.

        Args:
            path: Candidate path selected by the loader.

        Returns:
            ABI-compatible fake library for the environment candidate.

        Raises:
            OSError: If the Slurm candidate is loaded.
        """
        if Path(path) == slurm_path:
            raise OSError("bad Slurm library")
        return fake_library

    monkeypatch.setattr(loader.ctypes, "CDLL", load)
    monkeypatch.setattr(loader, "load_mkl_runtime", lambda: object())
    monkeypatch.setattr(loader.resources, "files", lambda _package: tmp_path / "no-resource")
    monkeypatch.setattr(loader, "_repository_library_path", lambda: repository_path)

    probe = loader.probe_cpp_minimizer()

    assert probe.available
    assert probe.path == env_path
    assert probe.source == "MUMAG_LIB_OUT"


def test_package_resource_error_preserves_earlier_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """A package-resource error must not discard an earlier valid candidate."""
    loader = _import_loader()
    slurm_path, _env_path, _resource_path, repository_path = _configure_automatic_candidates(
        monkeypatch, tmp_path, request
    )
    fake_library = FakeLibrary(1, _native_symbols())
    resource_requests: list[str] = []

    def fail_package_resource(package: str) -> Any:
        """Record the package lookup and raise an ordinary discovery error.

        Args:
            package: Requested package name.

        Raises:
            OSError: Always, to emulate an unavailable package resource.
        """
        resource_requests.append(package)
        raise OSError("package resources unavailable")

    monkeypatch.setattr(loader.ctypes, "CDLL", lambda _path: fake_library)
    monkeypatch.setattr(loader, "load_mkl_runtime", lambda: object())
    monkeypatch.setattr(loader.resources, "files", fail_package_resource)
    monkeypatch.setattr(loader, "_repository_library_path", lambda: repository_path)

    probe = loader.probe_cpp_minimizer()

    assert probe.available
    assert probe.path == slurm_path
    assert resource_requests == ["tommos"]


def test_package_resource_error_preserves_repository_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """A package-resource error must not discard the repository fallback."""
    loader = _import_loader()
    _slurm_path, _env_path, _resource_path, repository_path = _configure_automatic_candidates(
        monkeypatch, tmp_path, request
    )
    fake_library = FakeLibrary(1, _native_symbols())

    def fail_package_resource(_package: str) -> Any:
        """Raise an ordinary discovery error for the package resource.

        Args:
            _package: Requested package name.

        Raises:
            OSError: Always, to emulate an unavailable package resource.
        """
        raise OSError("package resources unavailable")

    monkeypatch.delenv("SLURM_JOB_ID")
    monkeypatch.delenv("MUMAG_LIB_OUT")
    monkeypatch.setattr(loader.ctypes, "CDLL", lambda _path: fake_library)
    monkeypatch.setattr(loader, "load_mkl_runtime", lambda: object())
    monkeypatch.setattr(loader.resources, "files", fail_package_resource)
    monkeypatch.setattr(loader, "_repository_library_path", lambda: repository_path)

    probe = loader.probe_cpp_minimizer()

    assert probe.available
    assert probe.path == repository_path
    assert probe.source == "repository"


def test_abi_validation_precedes_operation_symbol_lookup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A rejected ABI must be detected before any operation symbol is read."""
    loader = _import_loader()
    library_path = tmp_path / "library.so"
    library_path.touch()
    fake_library = FakeLibrary(2, _native_symbols())
    monkeypatch.setenv("TOMMOS_NATIVE_LIBRARY", str(library_path))
    monkeypatch.setattr(loader.ctypes, "CDLL", lambda _path: fake_library)
    monkeypatch.setattr(loader, "load_mkl_runtime", lambda: object())

    probe = loader.probe_cpp_minimizer()

    assert not probe.available
    assert isinstance(probe.error, RuntimeError)
    assert fake_library.accessed == ["tommos_native_abi_version"]


def test_minimizer_binding_uses_existing_42_argument_signature(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The minimizer binding must exactly preserve the existing 42-argument ABI."""
    loader = _import_loader()
    library_path = tmp_path / "library.so"
    library_path.touch()
    fake_library = FakeLibrary(1, _native_symbols())
    monkeypatch.setenv("TOMMOS_NATIVE_LIBRARY", str(library_path))
    monkeypatch.setattr(loader.ctypes, "CDLL", lambda _path: fake_library)
    monkeypatch.setattr(loader, "load_mkl_runtime", lambda: object())

    function = loader.require_cpp_minimizer()

    assert function.argtypes == [
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
    assert len(function.argtypes) == 42
    assert function.restype is ctypes.c_int


def test_missing_abi_symbol_blocks_operation_symbol_lookup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An absent ABI symbol must block every operation-symbol lookup."""
    loader = _import_loader()
    library_path = tmp_path / "library.so"
    library_path.touch()
    fake_library = FakeLibrary(1, _native_symbols(), include_abi=False)
    monkeypatch.setenv("TOMMOS_NATIVE_LIBRARY", str(library_path))
    monkeypatch.setattr(loader.ctypes, "CDLL", lambda _path: fake_library)
    monkeypatch.setattr(loader, "load_mkl_runtime", lambda: object())

    probe = loader.probe_pardiso()

    assert not probe.available
    assert isinstance(probe.error, AttributeError)
    assert fake_library.accessed == ["tommos_native_abi_version"]


def test_keyboard_interrupt_propagates_from_native_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """A KeyboardInterrupt during discovery must not become an unavailable probe."""
    loader = _import_loader()

    def interrupt_package_resource(_package: str) -> Any:
        """Raise KeyboardInterrupt while resolving the package resource.

        Args:
            _package: Requested package name.

        Raises:
            KeyboardInterrupt: Always, to emulate user interruption.
        """
        raise KeyboardInterrupt()

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("MUMAG_LIB_OUT", raising=False)
    monkeypatch.setattr(loader.resources, "files", interrupt_package_resource)

    with pytest.raises(KeyboardInterrupt):
        loader.probe_cpp_minimizer()


def test_missing_minimizer_symbol_does_not_invalidate_pardiso(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A missing minimizer export must leave an independent PARDISO probe usable."""
    loader = _import_loader()
    library_path = tmp_path / "library.so"
    library_path.touch()
    symbols = _native_symbols()
    del symbols["run_cpp_pcohen_hs_minimization"]
    fake_library = FakeLibrary(1, symbols)
    monkeypatch.setenv("TOMMOS_NATIVE_LIBRARY", str(library_path))
    monkeypatch.setattr(loader.ctypes, "CDLL", lambda _path: fake_library)
    monkeypatch.setattr(loader, "load_mkl_runtime", lambda: object())

    minimizer_probe = loader.probe_cpp_minimizer()
    pardiso_probe = loader.probe_pardiso()

    assert not minimizer_probe.available
    assert pardiso_probe.available
    assert fake_library.accessed == [
        "tommos_native_abi_version",
        "run_cpp_pcohen_hs_minimization",
        "init_pardiso",
        "pardiso_solve_direct",
        "free_pardiso",
    ]


def test_missing_pardiso_symbol_does_not_invalidate_minimizer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A missing PARDISO export must leave an independent minimizer probe usable."""
    loader = _import_loader()
    library_path = tmp_path / "library.so"
    library_path.touch()
    symbols = _native_symbols()
    del symbols["free_pardiso"]
    fake_library = FakeLibrary(1, symbols)
    monkeypatch.setenv("TOMMOS_NATIVE_LIBRARY", str(library_path))
    monkeypatch.setattr(loader.ctypes, "CDLL", lambda _path: fake_library)
    monkeypatch.setattr(loader, "load_mkl_runtime", lambda: object())

    pardiso_probe = loader.probe_pardiso()
    minimizer_probe = loader.probe_cpp_minimizer()

    assert not pardiso_probe.available
    assert minimizer_probe.available


def test_mkl_runtime_loads_unique_distribution_owned_version_globally_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Load and cache only the unique real versioned runtime owned by ``mkl``."""
    loader = _import_loader()
    distribution = FakeDistribution(
        tmp_path,
        [
            "lib/libmkl_rt.so",
            "lib/libmkl_rt.so.backup",
            "lib/libmkl_rt.so.3",
            "lib/../lib/libmkl_rt.so.3",
        ],
    )
    _materialize_owned_files(distribution)
    runtime = object()
    distribution_calls: list[str] = []
    load_calls: list[tuple[str, int]] = []

    def find_distribution(name: str) -> FakeDistribution:
        """Return the controlled distribution.

        Args:
            name: Requested distribution name.

        Returns:
            Controlled ``mkl`` metadata.
        """
        distribution_calls.append(name)
        return distribution

    def load_library(path: str, mode: int) -> object:
        """Record the exact runtime load operation.

        Args:
            path: Shared-library path.
            mode: Dynamic-loader flags.

        Returns:
            Stable fake runtime handle.
        """
        load_calls.append((path, mode))
        return runtime

    monkeypatch.setattr(loader.metadata, "distribution", find_distribution)
    monkeypatch.setattr(loader.ctypes, "CDLL", load_library)

    assert loader.load_mkl_runtime() is runtime
    assert loader.load_mkl_runtime() is runtime
    assert distribution_calls == ["mkl"]
    assert load_calls == [(str((tmp_path / "lib/libmkl_rt.so.3").resolve(strict=True)), ctypes.RTLD_GLOBAL)]


@pytest.mark.parametrize(
    ("files", "error_fragment"),
    [
        (
            ["lib/libmkl_rt.so", "lib/libmkl_rt.so.backup"],
            "does not own a versioned libmkl_rt.so.*",
        ),
        (
            ["lib/libmkl_rt.so.3", "alternative/libmkl_rt.so.4"],
            "owns multiple versioned libmkl_rt.so.* files",
        ),
    ],
)
def test_mkl_runtime_rejects_absent_or_ambiguous_distribution_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    files: list[str],
    error_fragment: str,
) -> None:
    """Reject missing or ambiguous runtime ownership without trying a fallback."""
    loader = _import_loader()
    distribution = FakeDistribution(tmp_path, files)
    _materialize_owned_files(distribution)
    load_calls: list[str] = []
    monkeypatch.setattr(loader.metadata, "distribution", lambda name: distribution)
    monkeypatch.setattr(loader.ctypes, "CDLL", lambda path, mode: load_calls.append(path))

    with pytest.raises(loader.MKLRuntimeError, match=error_fragment):
        loader.load_mkl_runtime()

    assert load_calls == []


def test_mkl_runtime_caches_original_stale_metadata_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Retain the first strict-resolution error rather than retrying or replacing it."""
    loader = _import_loader()
    distribution = FakeDistribution(tmp_path, ["lib/libmkl_rt.so.3"])
    distribution_calls: list[str] = []

    def find_distribution(name: str) -> FakeDistribution:
        """Return stale metadata and record the lookup.

        Args:
            name: Requested distribution name.

        Returns:
            Controlled stale metadata.
        """
        distribution_calls.append(name)
        return distribution

    monkeypatch.setattr(loader.metadata, "distribution", find_distribution)

    with pytest.raises(loader.MKLRuntimeError) as first:
        loader.load_mkl_runtime()
    with pytest.raises(loader.MKLRuntimeError) as second:
        loader.load_mkl_runtime()

    assert second.value is first.value
    assert isinstance(first.value.__cause__, FileNotFoundError)
    assert distribution_calls == ["mkl"]


def test_packaged_cpp_library_preloads_mkl_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Preload the distribution-owned runtime before opening the packaged C++ library."""
    loader = _import_loader()
    runtime_path = tmp_path / "libmkl_rt.so.3"
    native_path = tmp_path / "libcpp_mkl_minimizer.so"
    runtime_path.touch()
    native_path.touch()
    distribution = FakeDistribution(tmp_path, [runtime_path.name])
    fake_native = FakeLibrary(1, _native_symbols())
    load_order: list[str] = []

    def load_library(path: str, mode: int | None = None) -> Any:
        """Return controlled runtime/native handles while recording order.

        Args:
            path: Shared-library path.
            mode: Optional dynamic-loader flags.

        Returns:
            Fake handle for the requested path.
        """
        load_order.append(Path(path).name)
        if Path(path) == runtime_path:
            assert mode == ctypes.RTLD_GLOBAL
            return object()
        assert Path(path) == native_path
        assert mode is None
        return fake_native

    monkeypatch.setattr(loader.metadata, "distribution", lambda name: distribution)
    monkeypatch.setattr(loader, "_configured_candidate", lambda: None)
    monkeypatch.setattr(loader, "_automatic_candidates", lambda: ([(native_path, "tommos._native")], None))
    monkeypatch.setattr(loader.ctypes, "CDLL", load_library)

    assert loader.probe_cpp_minimizer().available
    assert load_order == ["libmkl_rt.so.3", "libcpp_mkl_minimizer.so"]


@pytest.mark.parametrize(
    ("source", "configured"),
    [
        ("TOMMOS_NATIVE_LIBRARY", True),
        ("SLURM_JOB_ID", False),
        ("MUMAG_LIB_OUT", False),
        ("tommos._native", False),
        ("repository", False),
    ],
)
def test_every_native_candidate_requires_strict_cached_mkl_preload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: str,
    configured: bool,
) -> None:
    """A strict PyPI runtime failure must prevent opening every candidate provenance."""
    loader = _import_loader()
    native_path = tmp_path / source / "libcpp_mkl_minimizer.so"
    native_path.parent.mkdir()
    native_path.touch()
    runtime_error = loader.MKLRuntimeError(f"runtime unavailable for {source}")
    preload_calls: list[str] = []
    cdll_calls: list[str] = []

    def fail_runtime() -> None:
        """Raise the controlled strict runtime discovery failure."""
        preload_calls.append(source)
        raise runtime_error

    monkeypatch.setattr(loader, "load_mkl_runtime", fail_runtime)
    monkeypatch.setattr(loader.ctypes, "CDLL", lambda path: cdll_calls.append(path))
    if configured:
        monkeypatch.setattr(loader, "_configured_candidate", lambda: (native_path, source))
        monkeypatch.setattr(
            loader,
            "_automatic_candidates",
            lambda: pytest.fail("explicit candidate fell through to automatic discovery"),
        )
    else:
        monkeypatch.setattr(loader, "_configured_candidate", lambda: None)
        monkeypatch.setattr(loader, "_automatic_candidates", lambda: ([(native_path, source)], None))

    unavailable = loader.probe_cpp_minimizer()
    with pytest.raises(loader.MKLRuntimeError) as cached:
        loader.require_cpp_minimizer()

    assert unavailable.error is runtime_error
    assert cached.value is runtime_error
    assert preload_calls == [source]
    assert cdll_calls == []


def test_loader_owned_sparse_inspector_bindings_use_lp64_abi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind hint and optimize against the cached runtime with the verified LP64 ABI."""
    loader = _import_loader()
    hint = FakeFunction()
    optimize = FakeFunction()
    runtime = FakeLibrary(
        1,
        {
            "mkl_sparse_set_mv_hint": hint,
            "mkl_sparse_optimize": optimize,
        },
        include_abi=False,
    )
    monkeypatch.setattr(loader, "load_mkl_runtime", lambda: runtime)
    handle = ctypes.c_void_p(123)

    assert loader.mkl_sparse_set_mv_hint(handle, expected_calls=1000) == 0
    assert loader.mkl_sparse_optimize(handle) == 0
    assert hint.argtypes == [
        ctypes.c_void_p,
        ctypes.c_int,
        loader.MKLMatrixDescription,
        ctypes.c_int,
    ]
    assert hint.restype is ctypes.c_int
    assert optimize.argtypes == [ctypes.c_void_p]
    assert optimize.restype is ctypes.c_int


def test_sparse_dot_mkl_probe_requires_wrapper_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sparse probe must require the runtime before the wrapper's MKL object."""
    loader = _import_loader()
    wrapper = ModuleType("sparse_dot_mkl._mkl_interface")
    imported: list[str] = []

    def import_wrapper(name: str) -> ModuleType:
        """Return a wrapper module that lacks its required MKL attribute.

        Args:
            name: Requested module name.

        Returns:
            The incomplete wrapper module.
        """
        imported.append(name)
        return wrapper

    monkeypatch.setattr(loader, "load_mkl_runtime", lambda: object())
    monkeypatch.setattr(loader.importlib, "import_module", import_wrapper)

    unavailable_probe = loader.probe_sparse_dot_mkl()

    assert not unavailable_probe.available
    assert isinstance(unavailable_probe.error, AttributeError)
    assert imported == ["sparse_dot_mkl._mkl_interface"]

    wrapper.MKL = object()
    loader = _import_loader()
    monkeypatch.setattr(loader, "load_mkl_runtime", lambda: object())
    monkeypatch.setattr(loader.importlib, "import_module", import_wrapper)
    available_probe = loader.probe_sparse_dot_mkl()

    assert available_probe.available
    assert available_probe.source == "sparse_dot_mkl._mkl_interface"


def test_sparse_probe_preserves_runtime_error_without_cpp_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the original runtime failure before wrapper import or C++ probing."""
    loader = _import_loader()
    runtime_error = loader.MKLRuntimeError("runtime unavailable")
    imported: list[str] = []

    def fail_runtime() -> None:
        """Raise the controlled runtime failure."""
        raise runtime_error

    monkeypatch.setattr(loader, "load_mkl_runtime", fail_runtime)
    monkeypatch.setattr(loader, "_load_library", lambda: pytest.fail("C++ library must remain independent"))
    monkeypatch.setattr(loader.importlib, "import_module", lambda name: imported.append(name))

    probe = loader.probe_sparse_dot_mkl()

    assert not probe.available
    assert probe.error is runtime_error
    assert imported == []


def test_native_diagnostics_reports_each_capability(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Diagnostics must provide the independently probed capability results."""
    loader = _import_loader()
    library_path = tmp_path / "library.so"
    library_path.touch()
    fake_library = FakeLibrary(1, _native_symbols())
    wrapper = ModuleType("sparse_dot_mkl._mkl_interface")
    wrapper.MKL = object()
    monkeypatch.setenv("TOMMOS_NATIVE_LIBRARY", str(library_path))
    monkeypatch.setattr(loader.ctypes, "CDLL", lambda _path: fake_library)
    monkeypatch.setattr(loader, "load_mkl_runtime", lambda: object())
    monkeypatch.setattr(loader.importlib, "import_module", lambda _name: wrapper)

    diagnostics = loader.native_diagnostics()

    assert set(diagnostics) == {"cpp_minimizer", "pardiso", "sparse_dot_mkl"}
    assert all(probe.available for probe in diagnostics.values())
