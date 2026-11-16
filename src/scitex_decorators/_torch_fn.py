#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-30 15:40:43 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/decorators/_torch_fn.py
# ----------------------------------------
import os

__FILE__ = "./src/scitex/decorators/_torch_fn.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from functools import wraps
from typing import Any as _Any
from typing import Callable

import numpy as np

from ._converters import _return_always, is_nested_decorator, to_torch
from ._runtime._optional_deps import import_pandas, import_torch, import_xarray


def torch_fn(func: Callable) -> Callable:
    """Decorator for PyTorch function compatibility.

    Automatically converts inputs to PyTorch tensors and handles various data types
    gracefully. Preserves the original input type in the output.

    Features
    --------
    - Converts inputs to PyTorch tensors
    - Preserves scalar parameters (int, float, bool, str)
    - Preserves dimension tuples like dim=(0, 1)
    - Handles nested lists/tuples gracefully
    - Automatically converts axis to dim for torch functions
    - Places tensors on CUDA when the installed torch build can run on the
      present GPU, else on the CPU (see ``_runtime._device.default_device``)
    - Returns output in same type as input (numpy->numpy, pandas->pandas, etc.)

    Parameters
    ----------
    func : Callable
        The function to decorate

    Returns
    -------
    Callable
        The decorated function

    Examples
    --------
    >>> @torch_fn
    ... def mean_squared(x, dim=None):
    ...     return (x ** 2).mean(dim=dim)
    >>>
    >>> # Works with numpy arrays
    >>> result = mean_squared(np.array([1, 2, 3]))
    >>>
    >>> # Works with nested lists
    >>> result = mean_squared([[1, 2], [3, 4]])
    >>>
    >>> # Preserves dimension tuples
    >>> result = mean_squared(data, dim=(0, 1))

    Notes
    -----
    For optimal performance with batch processing, apply torch_fn before batch_fn:
    @batch_fn
    @torch_fn
    def my_function(x): ...

    Or use auto-ordering to handle this automatically.
    """

    @wraps(func)
    def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
        # Skip conversion if already in a nested decorator context
        if is_nested_decorator():
            results = func(*args, **kwargs)
            return results

        # Set the current decorator context
        wrapper._current_decorator = "torch_fn"

        # Store original object for type preservation
        original_object = args[0] if args else None

        converted_args, converted_kwargs = to_torch(
            *args, return_fn=_return_always, **kwargs
        )

        # Skip strict assertion for certain types that may not convert to tensors
        # Instead, convert what we can and pass through what we can't
        torch = import_torch("torch_fn")

        validated_args = []
        for arg_index, arg in enumerate(converted_args):
            if isinstance(arg, torch.Tensor):
                validated_args.append(arg)
            elif isinstance(arg, (int, float, str, type(None))):
                # Pass through scalars and strings unchanged
                validated_args.append(arg)
            elif isinstance(arg, list) and all(
                isinstance(item, torch.Tensor) for item in arg
            ):
                # List of tensors - pass through as is
                validated_args.append(arg)
            else:
                # Try one more conversion attempt
                try:
                    validated_args.append(torch.tensor(arg).float())
                except:
                    # If all else fails, pass through unchanged
                    validated_args.append(arg)

        results = func(*validated_args, **converted_kwargs)

        # Convert results back to original input types
        torch = import_torch("torch_fn")

        if isinstance(results, torch.Tensor):
            if original_object is not None:
                if isinstance(original_object, list):
                    return results.detach().cpu().numpy().tolist()
                elif isinstance(original_object, np.ndarray):
                    return results.detach().cpu().numpy()
                elif (
                    hasattr(original_object, "__class__")
                    and original_object.__class__.__name__ == "DataFrame"
                ):
                    pd = import_pandas("torch_fn")

                    return pd.DataFrame(results.detach().cpu().numpy())
                elif (
                    hasattr(original_object, "__class__")
                    and original_object.__class__.__name__ == "Series"
                ):
                    pd = import_pandas("torch_fn")

                    return pd.Series(results.detach().cpu().numpy().flatten())
                elif (
                    hasattr(original_object, "__class__")
                    and original_object.__class__.__name__ == "DataArray"
                ):
                    xr = import_xarray("torch_fn")

                    return xr.DataArray(results.detach().cpu().numpy())
            return results

        return results

    # Mark as a wrapper for detection
    wrapper._is_wrapper = True
    wrapper._decorator_type = "torch_fn"
    return wrapper


# EOF
