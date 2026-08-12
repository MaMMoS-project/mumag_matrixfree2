"""Top-level command-line interface for Tommos."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class _Command:
    module_name: str
    description: str


_COMMANDS = {
    "loop": _Command("tommos.loop", "Run a micromagnetic hysteresis simulation."),
    "mesh": _Command("tommos.mesh", "Generate a tetrahedral simulation mesh."),
    "add-shell": _Command("tommos.add_shell", "Add exterior shell layers to a mesh."),
}


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level Tommos argument parser.

    Returns:
        Parser configured with the registered commands.
    """
    command_help = "\n".join(f"  {name:<12} {command.description}" for name, command in _COMMANDS.items())
    parser = argparse.ArgumentParser(
        prog="tommos",
        description="Matrix-free micromagnetic simulation tools.",
        epilog=f"commands:\n{command_help}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=tuple(_COMMANDS),
        metavar="COMMAND",
        help="Command to run.",
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return parser


def _run_command(command_name: str, arguments: Sequence[str]) -> None:
    """Import and invoke a registered command.

    Args:
        command_name: Registered command name to invoke.
        arguments: Arguments to forward to the command.
    """
    command = _COMMANDS[command_name]
    module = importlib.import_module(command.module_name)
    module.main(arguments)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the Tommos command-line interface.

    Args:
        argv: Arguments after the executable name. Uses `sys.argv` when omitted.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    parsed = parser.parse_args(arguments)
    if parsed.command is None:
        parser.print_help(sys.stderr)
        parser.exit(2, f"\n{parser.prog}: error: a command is required\n")

    _run_command(parsed.command, parsed.arguments)
