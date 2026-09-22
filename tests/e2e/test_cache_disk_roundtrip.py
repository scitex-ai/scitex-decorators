"""E2E: ``@cache_disk`` serves repeat calls from real disk cache (PS-212).

Drives the real joblib-backed disk cache against a tmp ``SCITEX_DIR``
(see conftest) — no network, no mocks. The body must execute exactly
once for two identical calls; the second result comes from disk.
"""

from __future__ import annotations

import pytest

pytest.importorskip("joblib", reason="caching extra absent (pip install scitex-decorators[caching])")

from scitex_decorators import cache_disk

pytestmark = pytest.mark.e2e


def test_cache_disk_serves_repeat_call_from_disk() -> None:
    # Arrange
    calls: list[int] = []

    @cache_disk
    def expensive(x: int) -> int:
        calls.append(x)
        return x * 2

    # Act
    first = expensive(21)
    second = expensive(21)

    # Assert
    assert (first, second, len(calls)) == (42, 42, 1)
