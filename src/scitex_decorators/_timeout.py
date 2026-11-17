#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-07 05:58:41 (ywatanabe)"
# File: ./scitex_repo/src/scitex/decorators/_timeout.py

"""Run a function in a child process and give up after a wall-clock budget."""

import queue as _queue
import time
from multiprocessing import Process, Queue

# How often the parent re-checks whether the child is still alive while it
# waits for the result. Small enough that a child that dies early is noticed
# promptly, large enough not to spin.
_POLL_INTERVAL_S = 0.1


def _run_and_report(func, result_queue, args, kwargs):
    """Child-process body: run ``func`` and report its outcome on the queue."""
    try:
        result_queue.put(("ok", func(*args, **kwargs)))
    except BaseException as exc:  # noqa: BLE001 - re-raised in the parent
        result_queue.put(("error", exc))


def timeout(seconds=10, error_message="Timeout"):
    """Raise ``TimeoutError(error_message)`` if the call exceeds ``seconds``.

    The wrapped function runs in a child process. Its return value is
    returned; an exception it raises is re-raised in the caller. A child that
    dies without reporting (killed, crashed, unpicklable exception) raises
    ``RuntimeError`` naming its exit code instead of blocking forever.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            result_queue = Queue()
            process = Process(
                target=_run_and_report, args=(func, result_queue, args, kwargs)
            )
            process.start()

            # Read the result BEFORE joining: a child cannot exit until its
            # queued result is drained, so join-then-get stalls (and used to
            # misreport a large result as a timeout).
            deadline = time.monotonic() + seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    process.terminate()
                    process.join()
                    raise TimeoutError(error_message)
                try:
                    status, payload = result_queue.get(
                        timeout=min(_POLL_INTERVAL_S, remaining)
                    )
                    break
                except _queue.Empty:
                    if process.is_alive():
                        continue
                    # The child exited; anything it queued has been flushed.
                    try:
                        status, payload = result_queue.get(timeout=_POLL_INTERVAL_S)
                        break
                    except _queue.Empty:
                        process.join()
                        raise RuntimeError(
                            f"{getattr(func, '__name__', func)!s} exited with "
                            f"code {process.exitcode} without returning a result"
                        ) from None

            process.join()
            if status == "error":
                raise payload
            return payload

        return wrapper

    return decorator


# EOF
