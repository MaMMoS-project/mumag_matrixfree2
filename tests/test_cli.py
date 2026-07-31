from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from tommos import cli


def _run_tommos(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the installed Tommos command.

    Args:
        arguments: Arguments passed to the command.

    Returns:
        Completed command result.
    """
    executable = shutil.which("tommos")
    assert executable is not None
    return subprocess.run(
        [executable, *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_top_level_help_lists_primary_commands() -> None:
    result = _run_tommos("--help")

    assert result.returncode == 0, result.stderr
    assert "loop" in result.stdout
    assert "mesh" in result.stdout
    assert "add-shell" in result.stdout
    assert "mesh-convert" not in result.stdout


@pytest.mark.parametrize("arguments", ((), ("--",)))
def test_missing_command_is_a_usage_error(arguments: tuple[str, ...]) -> None:
    result = _run_tommos(*arguments)

    assert result.returncode == 2
    assert "usage:" in result.stderr.lower()
    assert "command is required" in result.stderr.lower()


def test_unknown_command_is_a_usage_error() -> None:
    result = _run_tommos("unknown")

    assert result.returncode == 2
    assert "invalid choice" in result.stderr.lower()
    for command in ("loop", "mesh", "add-shell"):
        assert command in result.stderr


def test_dispatch_forwards_arguments_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    received_module: list[str] = []
    received_arguments: list[str] = []

    def fake_main(arguments: Sequence[str] | None = None) -> None:
        assert arguments is not None
        received_arguments.extend(arguments)

    def fake_import_module(module_name: str) -> SimpleNamespace:
        received_module.append(module_name)
        return SimpleNamespace(main=fake_main)

    monkeypatch.setattr(cli.importlib, "import_module", fake_import_module)
    forwarded = ["model", "--mesh=mesh.npz", "--verbose"]

    cli.main(["loop", *forwarded])

    assert received_module == ["tommos.loop"]
    assert received_arguments == forwarded


@pytest.mark.parametrize(
    ("command", "expected_usage"),
    (
        ("loop", "usage: tommos loop"),
        ("mesh", "usage: tommos mesh"),
        ("add-shell", "usage: tommos add-shell"),
    ),
)
def test_primary_subcommand_exposes_help(command: str, expected_usage: str) -> None:
    result = _run_tommos(command, "--help")

    assert result.returncode == 0, result.stderr
    assert expected_usage in result.stdout
