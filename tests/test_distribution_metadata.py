"""Tests for installed ``tommos`` distribution metadata."""

import tomllib
from importlib.metadata import metadata, requires, version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


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


def test_project_manifest_defines_the_packaging_and_pixi_contract() -> None:
    """Keep packaging metadata and Pixi environments in one manifest."""
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    configuration = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    assert configuration["project"]["name"] == "tommos"
    assert configuration["project"]["requires-python"] == ">=3.11"

    requirements = [Requirement(value) for value in configuration["project"]["dependencies"]]
    native_requirements = {
        requirement.name: requirement for requirement in requirements if requirement.name in {"mkl", "sparse-dot-mkl"}
    }
    assert set(native_requirements) == {"mkl", "sparse-dot-mkl"}
    for requirement in native_requirements.values():
        assert requirement.marker is not None
        assert requirement.marker.evaluate({"sys_platform": "linux", "platform_machine": "x86_64"})
        assert not requirement.marker.evaluate({"sys_platform": "win32", "platform_machine": "AMD64"})
    assert "src/cpp/find_mkl.py" not in configuration["tool"]["scikit-build"]["sdist"]["include"]

    pixi = configuration["tool"]["pixi"]
    assert pixi["workspace"]["platforms"] == ["linux-64", "osx-arm64", "osx-64"]
    assert set(pixi["environments"]) == {"default", "cpu", "test", "lint", "build", "sample", "cuda"}
    assert "dependency-groups" not in configuration
    assert "dependencies" not in pixi
    assert pixi["pypi-dependencies"]["tommos"] == {"path": ".", "editable": True}
    assert pixi["feature"]["test"]["pypi-dependencies"]["packaging"] == "*"
    assert set(pixi["feature"]["test"]["tasks"]) == {"test"}
    assert set(pixi["feature"]["lint"]["tasks"]) == {"lint"}
    assert set(pixi["feature"]["build"]["tasks"]) == {"build-package"}
    assert set(pixi["feature"]["sample"]["tasks"]) == {"sample", "clean-samples"}
    assert set(pixi["feature"]["cuda"]["tasks"]) == {"sample-gpu"}
    assert pixi["environments"]["default"] == {"solve-group": "tommos"}
    assert pixi["environments"]["cpu"] == {"solve-group": "tommos"}
    assert pixi["environments"]["sample"]["features"] == ["sample", "io"]
    assert pixi["environments"]["cuda"]["features"] == ["cuda", "sample", "io"]
    assert {environment["solve-group"] for environment in pixi["environments"].values()} == {"tommos"}

    project_owned_pypi_names = {
        canonicalize_name(Requirement(requirement).name)
        for dependency_group in [
            configuration["project"]["dependencies"],
            *configuration["project"]["optional-dependencies"].values(),
        ]
        for requirement in dependency_group
    }
    ordinary_feature_pypi_names = {
        canonicalize_name(name)
        for feature_name in {"test", "build", "sample"}
        for name in pixi["feature"][feature_name].get("pypi-dependencies", {})
    }
    assert project_owned_pypi_names.isdisjoint(ordinary_feature_pypi_names)
