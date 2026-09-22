"""Smoke: ``@numpy_fn`` converts and restores types in a subprocess (PS-211).

Subprocess-driven (``sys.executable -c ...``) so this proves the
installed package resolves its hard dependencies — an in-process call
would not. Hermetic: numpy is a hard dependency, no network, no
credentials, no writes outside tmp dirs.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.smoke

CODE = "\n".join(
    [
        "import scitex_decorators as dec",
        "",
        "@dec.numpy_fn",
        "def double(x):",
        "    return x * 2",
        "",
        "print(double([1, 2]))",
    ]
)


def test_numpy_fn_roundtrip_in_subprocess() -> None:
    # Arrange
    argv = [sys.executable, "-c", CODE]

    # Act
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=30)

    # Assert
    assert (completed.returncode, completed.stdout.strip()) == (0, "[2, 4]")
