"""Tests for installed ``tommos`` distribution metadata."""

import tomllib
from importlib.metadata import metadata, requires, version
from pathlib import Path

from packaging.requirements import Requirement


def test_distribution_metadata_matches_import_package() -> None:
    """Verify installed metadata agrees with the import package."""
    project_metadata = metadata("tommos")
    assert project_metadata["Name"] == "tommos"
    assert version("tommos") == "0.1.0"
    assert project_metadata["Requires-Python"] == ">=3.11"


def test_linux_x86_64_runtime_dependencies_are_platform_guarded() -> None:
    """Publish oneMKL runtime dependencies only for supported Linux wheels."""
    raw_requirements = requires("tommos") or ()
    assert 'mkl<2027,>=2026.0; sys_platform == "linux" and platform_machine == "x86_64"' in raw_requirements
    assert 'sparse-dot-mkl>=0.9.10; sys_platform == "linux" and platform_machine == "x86_64"' in raw_requirements
    requirements = [Requirement(value) for value in raw_requirements]
    runtime_requirements = {
        requirement.name: requirement for requirement in requirements if requirement.name in {"mkl", "sparse-dot-mkl"}
    }

    assert set(runtime_requirements) == {"mkl", "sparse-dot-mkl"}
    assert str(runtime_requirements["mkl"].specifier) == "<2027,>=2026.0"
    assert str(runtime_requirements["sparse-dot-mkl"].specifier) == ">=0.9.10"

    supported = {"sys_platform": "linux", "platform_machine": "x86_64"}
    macos = {"sys_platform": "darwin", "platform_machine": "x86_64"}
    unsupported_linux = {"sys_platform": "linux", "platform_machine": "aarch64"}
    for requirement in runtime_requirements.values():
        assert requirement.marker is not None
        assert requirement.marker.evaluate(supported)
        assert not requirement.marker.evaluate(macos)
        assert not requirement.marker.evaluate(unsupported_linux)


def test_darwin_build_override_disables_native_platlib() -> None:
    """Configure Darwin distributions as portable Python-only wheels."""
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    configuration = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    scikit_build = configuration["tool"]["scikit-build"]
    darwin_overrides = [
        override for override in scikit_build["overrides"] if override["if"]["platform-system"] == "^darwin"
    ]

    assert scikit_build["wheel"]["py-api"] == "py3"
    assert darwin_overrides == [
        {
            "if": {"platform-system": "^darwin"},
            "wheel": {"cmake": False, "platlib": False},
        }
    ]
