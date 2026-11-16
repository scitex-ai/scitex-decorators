#!/usr/bin/env python3
"""Guarded imports for the optional capabilities of scitex-decorators.

torch, pandas, xarray and joblib are optional: the core decorators
(numpy_fn, batch_fn, cache_mem, timeout, ...) work without them, so they are
declared only in the ``[all]`` extra. Every use site imports them through
these helpers. A missing library raises ``ModuleNotFoundError`` (so existing
``except ImportError`` / ``except ModuleNotFoundError`` callers keep working)
whose message names the decorator and the one install command, instead of a
bare "No module named ..." from deep inside a decorator.
"""

from __future__ import annotations

from types import ModuleType

INSTALL_HINT = 'pip install "scitex-decorators[all]"'


def _missing(library: str, feature: str) -> str:
    return (
        f"{feature} requires `{library}`, which is not installed. "
        f"Install the optional capabilities with: {INSTALL_HINT}"
    )


def import_torch(feature: str) -> ModuleType:
    """Return the ``torch`` module, or raise ModuleNotFoundError naming *feature*."""
    try:
        import torch
    except ImportError as err:
        raise ModuleNotFoundError(_missing("torch", feature), name="torch") from err
    return torch


def import_pandas(feature: str) -> ModuleType:
    """Return the ``pandas`` module, or raise ModuleNotFoundError naming *feature*."""
    try:
        import pandas
    except ImportError as err:
        raise ModuleNotFoundError(_missing("pandas", feature), name="pandas") from err
    return pandas


def import_xarray(feature: str) -> ModuleType:
    """Return the ``xarray`` module, or raise ModuleNotFoundError naming *feature*."""
    try:
        import xarray
    except ImportError as err:
        raise ModuleNotFoundError(_missing("xarray", feature), name="xarray") from err
    return xarray


def import_joblib_memory(feature: str) -> type:
    """Return ``joblib.Memory``, or raise ModuleNotFoundError naming *feature*."""
    try:
        from joblib import Memory
    except ImportError as err:
        raise ModuleNotFoundError(_missing("joblib", feature), name="joblib") from err
    return Memory


__all__ = [
    "INSTALL_HINT",
    "import_joblib_memory",
    "import_pandas",
    "import_torch",
    "import_xarray",
]

# EOF
