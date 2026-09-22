#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-30 15:41:19 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/decorators/_xarray_fn.py
# ----------------------------------------
import os

__FILE__ = "./src/scitex/decorators/_xarray_fn.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
from functools import wraps
from typing import Any as _Any
from typing import Callable

import numpy as np

from ._converters import is_nested_decorator


def xarray_fn(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
        # Skip conversion if already in a nested decorator context
        if is_nested_decorator():
            results = func(*args, **kwargs)
            return results

        # Set the current decorator context
        wrapper._current_decorator = "xarray_fn"

        # Store original object for type preservation
        original_object = args[0] if args else None

        # Convert args to xarray DataArrays
        def to_xarray(data):
            try:
                import xarray as xr
            except ImportError as exc:
                raise ImportError(
                    "xarray_fn requires xarray: pip install scitex-decorators[xarray]"
                ) from exc

            if isinstance(data, xr.DataArray):
                return data
            elif isinstance(data, np.ndarray):
                return xr.DataArray(data)
            elif isinstance(data, list):
                return xr.DataArray(data)
            elif hasattr(data, "__class__") and data.__class__.__name__ == "Tensor":
                return xr.DataArray(data.detach().cpu().numpy())
            elif hasattr(data, "__class__") and data.__class__.__name__ == "DataFrame":
                return xr.DataArray(data.values)
            elif hasattr(data, "__class__") and data.__class__.__name__ == "Series":
                return xr.DataArray(data.values)
            else:
                return xr.DataArray([data])

        converted_args = [to_xarray(arg) for arg in args]
        converted_kwargs = {k: to_xarray(v) for k, v in kwargs.items()}

        # Assertion to ensure all args are converted to xarray DataArrays
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError(
                "xarray_fn requires xarray: pip install scitex-decorators[xarray]"
            ) from exc

        for arg_index, arg in enumerate(converted_args):
            assert isinstance(
                arg, xr.DataArray
            ), f"Argument {arg_index} not converted to DataArray: {type(arg)}"

        results = func(*converted_args, **converted_kwargs)

        # Convert results back to original input types
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError(
                "xarray_fn requires xarray: pip install scitex-decorators[xarray]"
            ) from exc

        if isinstance(results, xr.DataArray):
            if original_object is not None:
                if isinstance(original_object, list):
                    return results.values.tolist()
                elif isinstance(original_object, np.ndarray):
                    return results.values
                elif (
                    hasattr(original_object, "__class__")
                    and original_object.__class__.__name__ == "Tensor"
                ):
                    try:
                        import torch
                    except ImportError as exc:
                        raise ImportError(
                            "xarray_fn requires torch for Tensor inputs: pip install scitex-decorators[torch]"
                        ) from exc

                    return torch.tensor(results.values)
                elif (
                    hasattr(original_object, "__class__")
                    and original_object.__class__.__name__ == "DataFrame"
                ):
                    try:
                        import pandas as pd
                    except ImportError as exc:
                        raise ImportError(
                            "xarray_fn requires pandas for DataFrame inputs: pip install scitex-decorators[pandas]"
                        ) from exc

                    return pd.DataFrame(results.values)
                elif (
                    hasattr(original_object, "__class__")
                    and original_object.__class__.__name__ == "Series"
                ):
                    try:
                        import pandas as pd
                    except ImportError as exc:
                        raise ImportError(
                            "xarray_fn requires pandas for Series inputs: pip install scitex-decorators[pandas]"
                        ) from exc

                    return pd.Series(results.values.flatten())
            return results

        return results

    # Mark as a wrapper for detection
    wrapper._is_wrapper = True
    wrapper._decorator_type = "xarray_fn"
    return wrapper


# EOF
