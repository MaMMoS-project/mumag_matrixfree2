"""Test configuration."""

import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def loop_bin():
    """Define `loop` bin."""
    return f"{sys.executable} {Path(__file__).resolve().parent.parent / 'src' / 'loop.py'}"


@pytest.fixture(scope="session")
def mesh_bin():
    """Define `mesh` bin."""
    return f"{sys.executable} {Path(__file__).resolve().parent.parent / 'src' / 'mesh.py'}"
