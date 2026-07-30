"""Locate oneMKL files owned by their Python distributions."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Sequence
from importlib import metadata
from pathlib import Path, PurePath


class MKLDiscoveryError(RuntimeError):
    """Report invalid oneMKL distribution file metadata."""


def _find_unique_owned_file(
    distribution_name: str,
    predicate: Callable[[PurePath], bool],
    description: str,
    missing_description: str | None = None,
) -> Path:
    """Find exactly one matching file owned by a distribution.

    Args:
        distribution_name: Installed Python distribution to inspect.
        predicate: Predicate selecting the required owned file.
        description: Human-readable file description for diagnostics.
        missing_description: Optional singular description for missing-file
            diagnostics.

    Returns:
        Absolute path to the unique matching file.

    Raises:
        MKLDiscoveryError: If the distribution is absent or does not own
            exactly one matching file.
    """
    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError as error:
        raise MKLDiscoveryError(
            f"{distribution_name} distribution is not installed; install {distribution_name}"
        ) from error

    matches: set[Path] = set()
    for entry in distribution.files or ():
        if not predicate(entry):
            continue
        located_path = Path(distribution.locate_file(entry))
        try:
            matches.add(located_path.resolve(strict=True))
        except FileNotFoundError as error:
            missing_path = located_path.resolve()
            raise MKLDiscoveryError(
                f"{distribution_name} distribution-owned {description} does not exist: "
                f"{missing_path}; reinstall {distribution_name}"
            ) from error

    ordered_matches = sorted(matches)
    if not matches:
        absent_file = missing_description or description
        raise MKLDiscoveryError(
            f"{distribution_name} distribution does not own {absent_file}; reinstall {distribution_name}"
        )
    if len(matches) > 1:
        candidates = ", ".join(str(path) for path in ordered_matches)
        raise MKLDiscoveryError(
            f"{distribution_name} distribution owns multiple {description} files: {candidates}; expected exactly one"
        )
    return ordered_matches[0]


def find_mkl_cmake_dir() -> Path:
    """Find the distribution-owned oneMKL CMake configuration directory.

    Returns:
        Absolute directory containing ``MKLConfig.cmake``.
    """
    config = _find_unique_owned_file(
        "mkl-devel",
        lambda entry: entry.name == "MKLConfig.cmake",
        "MKLConfig.cmake",
    )
    return config.parent


def find_mkl_runtime() -> Path:
    """Find the distribution-owned versioned oneMKL runtime library.

    Returns:
        Absolute path to ``libmkl_rt.so.*``.
    """
    return _find_unique_owned_file(
        "mkl",
        lambda entry: re.fullmatch(r"libmkl_rt\.so\.\d+(?:\.\d+)*", entry.name) is not None,
        "versioned libmkl_rt.so.*",
        "a versioned libmkl_rt.so.*",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Print the requested distribution-owned oneMKL path.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        Zero on success and one when discovery fails.
    """
    parser = argparse.ArgumentParser()
    requested = parser.add_mutually_exclusive_group(required=True)
    requested.add_argument("--cmake-dir", action="store_true")
    requested.add_argument("--runtime", action="store_true")
    arguments = parser.parse_args(argv)

    try:
        path = find_mkl_cmake_dir() if arguments.cmake_dir else find_mkl_runtime()
    except MKLDiscoveryError as error:
        print(f"find_mkl.py: error: {error}", file=sys.stderr)
        return 1

    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
