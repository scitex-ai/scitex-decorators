#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 15:45:34 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/tests/scitex/decorators/test__timeout.py
# ----------------------------------------
import os

__FILE__ = "./tests/scitex/decorators/test__timeout.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import time

import pytest

# Required for scitex_decorators module
pytest.importorskip("tqdm")

from scitex_decorators import timeout


def test_timeout_decorator_returns_success_value_when_within_limit():
    """Timeout decorator should return the wrapped function's value when it
    finishes before the timeout window elapses."""

    # Arrange
    # Generous budget: the decorator spawns a multiprocessing.Process whose
    # startup cost (re-import + coverage tracing under CI) can be seconds.
    # A tight 2 s window made this test flaky under coverage; the success
    # path it exercises only needs the function to finish before the budget.
    @timeout(seconds=30, error_message="Test timed out")
    def quick_function():
        return "Success"

    # Act
    result = quick_function()
    # Assert
    assert result == "Success"


def test_timeout_decorator_raises_timeout_error_when_exceeding_limit():
    """Timeout decorator should raise TimeoutError when the wrapped function
    takes longer than the allowed time budget."""

    # Arrange
    @timeout(seconds=0.5, error_message="Custom timeout message")
    def slow_function():
        time.sleep(1)
        return "This should not be returned"

    ctx = pytest.raises(TimeoutError)
    # Act
    # Assert
    with ctx:
        slow_function()


def test_timeout_decorator_includes_custom_error_message_on_timeout():
    """When the timeout fires, the raised TimeoutError's message should
    contain the user-supplied ``error_message`` string."""

    # Arrange
    @timeout(seconds=0.5, error_message="Custom timeout message")
    def slow_function():
        time.sleep(1)
        return "This should not be returned"

    ctx = pytest.raises(TimeoutError, match="Custom timeout message")
    # Act
    # Assert
    with ctx:
        slow_function()


def test_timeout_decorator_forwards_positional_arguments_correctly():
    """Timeout decorator should forward positional arguments to the wrapped
    function so its computed result is returned unchanged."""

    # Arrange
    # Generous budget — multiprocessing.Process spawn + coverage tracing
    # under CI can take seconds; this test only checks arg forwarding.
    @timeout(seconds=30)
    def function_with_args(xx, yy):
        return xx + yy

    # Act
    result = function_with_args(2, 3)
    # Assert
    assert result == 5


def test_timeout_decorator_forwards_keyword_arguments_correctly():
    """Timeout decorator should forward keyword arguments to the wrapped
    function so its computed result is returned unchanged."""

    # Arrange
    # Generous budget — see test_timeout_decorator_forwards_positional_*.
    @timeout(seconds=30)
    def function_with_kwargs(xx=0, yy=0):
        return xx * yy

    # Act
    result = function_with_kwargs(xx=5, yy=4)
    # Assert
    assert result == 20


def test_timeout_decorator_reraises_exception_from_wrapped_function():
    """An exception raised inside the child is re-raised in the caller,
    promptly, instead of the caller blocking on a result that never comes."""

    # Arrange
    @timeout(seconds=30)
    def failing_function():
        raise ValueError("boom from child")

    ctx = pytest.raises(ValueError, match="boom from child")
    # Act
    # Assert
    with ctx:
        failing_function()


def test_timeout_decorator_raises_runtime_error_when_child_dies_without_result():
    """A child that exits without reporting a result raises RuntimeError
    naming its exit code, rather than hanging the caller forever."""

    # Arrange
    @timeout(seconds=30)
    def dying_function():
        os._exit(3)

    ctx = pytest.raises(RuntimeError, match="code 3")
    # Act
    # Assert
    with ctx:
        dying_function()


def test_timeout_decorator_returns_result_larger_than_pipe_buffer():
    """A result bigger than the OS pipe buffer is returned, not misreported
    as a timeout (the child cannot exit until its result is drained)."""

    # Arrange
    payload_size = 4 * 1024 * 1024

    @timeout(seconds=30)
    def big_result_function():
        return b"x" * payload_size

    # Act
    result = big_result_function()
    # Assert
    assert len(result) == payload_size


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# EOF
