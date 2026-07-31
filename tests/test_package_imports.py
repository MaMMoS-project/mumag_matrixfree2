import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

IMPORTABLE_MODULES = (
    "add_shell",
    "amg_utils",
    "cpp_minimizer",
    "energy_kernels",
    "extract_nucleation",
    "fem_utils",
    "hysteresis_loop",
    "io_utils",
    "loop",
    "make_krn",
    "mesh",
    "mesh_convert",
    "minimizers",
    "plot_hysteresis",
    "poisson_solve",
    "reorder_mesh",
    "salomeMeshToNpz",
)

HELP_MODULES = (
    "tommos.add_shell",
    "tommos.loop",
    "tommos.mesh",
)


def test_tommos_namespace_is_importable() -> None:
    """Import the tommos namespace."""
    module = importlib.import_module("tommos")
    assert module.__name__ == "tommos"


@pytest.mark.parametrize("module_name", IMPORTABLE_MODULES)
def test_modules_import_through_tommos_namespace(module_name: str) -> None:
    """Import a module through the tommos namespace.

    Args:
        module_name: Module name under the tommos namespace.
    """
    module = importlib.import_module(f"tommos.{module_name}")
    assert module.__package__ == "tommos"


def test_package_modules_use_no_absolute_internal_imports() -> None:
    """Reject internal imports that bypass the package namespace."""
    package_dir = Path(__file__).parent.parent / "src" / "tommos"
    absolute_internal_imports: list[str] = []

    for module_name in IMPORTABLE_MODULES:
        module_path = package_dir / f"{module_name}.py"
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level_name = alias.name.split(".", maxsplit=1)[0]
                    if top_level_name in IMPORTABLE_MODULES:
                        absolute_internal_imports.append(f"{module_path}:{node.lineno}: {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                top_level_name = node.module.split(".", maxsplit=1)[0]
                if top_level_name in IMPORTABLE_MODULES:
                    absolute_internal_imports.append(f"{module_path}:{node.lineno}: {node.module}")

    assert not absolute_internal_imports, "\n".join(absolute_internal_imports)


@pytest.mark.parametrize("module_name", HELP_MODULES)
def test_primary_cli_modules_expose_help(module_name: str) -> None:
    """Expose CLI help for a primary module.

    Args:
        module_name: Fully qualified primary CLI module name.
    """
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
