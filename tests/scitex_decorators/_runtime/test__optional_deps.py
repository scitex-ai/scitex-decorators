#!/usr/bin/env python3
"""Tests for scitex_decorators._runtime._optional_deps.

The absent-library path runs in a child interpreter in which the library is
blocked via ``sys.modules[name] = None`` (the documented way to make an
import fail), so the real guard is exercised without patching this process.
"""

import subprocess
import sys

import pytest

from scitex_decorators._runtime._optional_deps import INSTALL_HINT, import_joblib_memory

_BLOCKED_JOBLIB = (
    "import sys\n"
    "sys.modules['joblib'] = None\n"
    "from scitex_decorators._runtime._optional_deps import import_joblib_memory\n"
    "try:\n"
    "    import_joblib_memory('cache_disk')\n"
    "except ModuleNotFoundError as err:\n"
    "    print(err)\n"
)


@pytest.fixture(scope="module")
def blocked_joblib_message():
    # Act
    completed = subprocess.run(
        [sys.executable, "-c", _BLOCKED_JOBLIB],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return completed.stdout


def test_missing_library_message_names_the_all_extra(blocked_joblib_message):
    # Arrange
    expected = INSTALL_HINT
    # Act
    message = blocked_joblib_message
    # Assert
    assert expected in message


def test_missing_library_message_names_the_feature(blocked_joblib_message):
    # Arrange
    expected = "cache_disk requires `joblib`"
    # Act
    message = blocked_joblib_message
    # Assert
    assert expected in message


def test_install_hint_is_the_all_extra():
    # Arrange
    expected = 'pip install "scitex-decorators[all]"'
    # Act
    hint = INSTALL_HINT
    # Assert
    assert hint == expected


def test_import_joblib_memory_returns_joblib_memory_class():
    # Arrange
    joblib = pytest.importorskip("joblib")
    # Act
    memory_cls = import_joblib_memory("cache_disk")
    # Assert
    assert memory_cls is joblib.Memory
