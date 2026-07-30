"""Tests for deterministic oneMKL discovery from distribution metadata."""

from __future__ import annotations

import importlib.util
from pathlib import Path, PurePosixPath
from types import ModuleType

import pytest


class FakeDistribution:
    """Minimal distribution metadata fake with owned-file locations."""

    def __init__(self, root: Path, files: list[str]) -> None:
        """Initialize the fake distribution.

        Args:
            root: Directory relative to which distribution files are located.
            files: Distribution-owned file entries.
        """
        self._root = root
        self.files = [PurePosixPath(entry) for entry in files]

    def locate_file(self, path: PurePosixPath) -> Path:
        """Locate a distribution-owned file.

        Args:
            path: Distribution file entry to locate.

        Returns:
            Absolute path represented by the entry.
        """
        return self._root / path


def materialize_owned_files(distribution: FakeDistribution) -> None:
    """Create every file represented by fake distribution metadata.

    Args:
        distribution: Fake whose owned files should exist on disk.
    """
    for entry in distribution.files:
        path = distribution.locate_file(entry).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


@pytest.fixture
def find_mkl_module() -> ModuleType:
    """Load the standalone oneMKL discovery helper."""
    script = Path(__file__).parents[1] / "src" / "cpp" / "find_mkl.py"
    spec = importlib.util.spec_from_file_location("find_mkl", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cmake_dir_uses_unique_mkl_devel_owned_config(
    find_mkl_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Return the absolute parent of the unique owned MKL config."""
    site_packages = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    distribution = FakeDistribution(
        site_packages,
        [
            "mkl_devel-2026.0.0.dist-info/METADATA",
            "../../cmake/mkl/MKLConfig.cmake",
        ],
    )
    materialize_owned_files(distribution)
    monkeypatch.setattr(find_mkl_module.metadata, "distribution", lambda name: distribution)

    assert find_mkl_module.main(["--cmake-dir"]) == 0

    expected = (tmp_path / "venv" / "lib" / "cmake" / "mkl").resolve()
    assert capsys.readouterr() == (f"{expected}\n", "")


def test_runtime_uses_unique_mkl_owned_versioned_library(
    find_mkl_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Return the absolute unique versioned runtime library."""
    site_packages = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    distribution = FakeDistribution(
        site_packages,
        [
            "mkl-2026.0.0.dist-info/METADATA",
            "../../libmkl_rt.so.3",
        ],
    )
    materialize_owned_files(distribution)
    monkeypatch.setattr(find_mkl_module.metadata, "distribution", lambda name: distribution)

    assert find_mkl_module.main(["--runtime"]) == 0

    expected = (tmp_path / "venv" / "lib" / "libmkl_rt.so.3").resolve()
    assert capsys.readouterr() == (f"{expected}\n", "")


@pytest.mark.parametrize(
    ("argument", "distribution_name", "expected_error"),
    [
        (
            "--cmake-dir",
            "mkl-devel",
            "mkl-devel distribution does not own MKLConfig.cmake; reinstall mkl-devel",
        ),
        (
            "--runtime",
            "mkl",
            "mkl distribution does not own a versioned libmkl_rt.so.*; reinstall mkl",
        ),
    ],
)
def test_no_owned_match_is_actionable(
    find_mkl_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    argument: str,
    distribution_name: str,
    expected_error: str,
) -> None:
    """Reject a distribution that does not own the required file."""
    distribution = FakeDistribution(tmp_path, ["package.dist-info/METADATA"])

    def fake_distribution(name: str) -> FakeDistribution:
        assert name == distribution_name
        return distribution

    monkeypatch.setattr(find_mkl_module.metadata, "distribution", fake_distribution)

    assert find_mkl_module.main([argument]) == 1
    assert capsys.readouterr() == ("", f"find_mkl.py: error: {expected_error}\n")


@pytest.mark.parametrize(
    ("argument", "distribution_name", "files", "owned_description"),
    [
        (
            "--cmake-dir",
            "mkl-devel",
            ["../../first/MKLConfig.cmake", "../../second/MKLConfig.cmake"],
            "MKLConfig.cmake",
        ),
        (
            "--runtime",
            "mkl",
            ["../../libmkl_rt.so.3", "../../alternative/libmkl_rt.so.4"],
            "versioned libmkl_rt.so.*",
        ),
    ],
)
def test_ambiguous_owned_matches_list_every_candidate(
    find_mkl_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    argument: str,
    distribution_name: str,
    files: list[str],
    owned_description: str,
) -> None:
    """Reject ambiguous metadata instead of selecting an arbitrary path."""
    distribution = FakeDistribution(tmp_path, files)
    materialize_owned_files(distribution)
    monkeypatch.setattr(find_mkl_module.metadata, "distribution", lambda name: distribution)
    expected_paths = sorted(str((tmp_path / entry).resolve()) for entry in files)
    expected_error = (
        f"{distribution_name} distribution owns multiple {owned_description} files: "
        f"{', '.join(expected_paths)}; expected exactly one"
    )

    assert find_mkl_module.main([argument]) == 1
    assert capsys.readouterr() == ("", f"find_mkl.py: error: {expected_error}\n")


def test_duplicate_metadata_entries_resolving_to_one_runtime_are_unique(
    find_mkl_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Deduplicate metadata aliases that locate the same runtime file."""
    distribution = FakeDistribution(
        tmp_path,
        ["lib/libmkl_rt.so.3", "lib/../lib/libmkl_rt.so.3"],
    )
    materialize_owned_files(distribution)
    monkeypatch.setattr(find_mkl_module.metadata, "distribution", lambda name: distribution)

    assert find_mkl_module.main(["--runtime"]) == 0
    assert capsys.readouterr() == (f"{(tmp_path / 'lib/libmkl_rt.so.3').resolve()}\n", "")


def test_runtime_rejects_non_numeric_version_suffix(
    find_mkl_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ignore backup or other non-SONAME files with a runtime-like prefix."""
    distribution = FakeDistribution(tmp_path, ["lib/libmkl_rt.so.backup"])
    materialize_owned_files(distribution)
    monkeypatch.setattr(find_mkl_module.metadata, "distribution", lambda name: distribution)

    assert find_mkl_module.main(["--runtime"]) == 1
    expected_error = "mkl distribution does not own a versioned libmkl_rt.so.*; reinstall mkl"
    assert capsys.readouterr() == ("", f"find_mkl.py: error: {expected_error}\n")


def test_missing_distribution_owned_runtime_is_actionable(
    find_mkl_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject stale metadata whose matching runtime does not exist."""
    distribution = FakeDistribution(tmp_path, ["lib/libmkl_rt.so.3"])
    monkeypatch.setattr(find_mkl_module.metadata, "distribution", lambda name: distribution)
    missing_path = (tmp_path / "lib/libmkl_rt.so.3").resolve()
    expected_error = f"mkl distribution-owned versioned libmkl_rt.so.* does not exist: {missing_path}; reinstall mkl"

    assert find_mkl_module.main(["--runtime"]) == 1
    assert capsys.readouterr() == ("", f"find_mkl.py: error: {expected_error}\n")
