#!/usr/bin/env python3
"""Default torch device selection for the type-conversion decorators.

``torch.cuda.is_available()`` only says that a driver and a device exist. It
does NOT say that the installed torch build carries kernels for that device:
a GPU whose compute capability was dropped from the wheel (e.g. a Pascal
sm_61 card under a torch built for sm_75+) reports ``is_available() == True``
and then fails the first kernel launch with ``CUDA error: no kernel image is
available for execution on the device``. Selecting "cuda" on that signal
alone makes every decorated call crash, so the default device is "cuda" only
when the device's architecture is actually covered by the build.
"""

from __future__ import annotations

import functools
import re
import warnings
from typing import Iterable, Tuple

from ._optional_deps import import_torch

_ARCH_RE = re.compile(r"^(sm|compute)_(\d+)(\d)[a-z]?$")


def cuda_arch_supported(capability: Tuple[int, int], arch_list: Iterable[str]) -> bool:
    """Return True when a torch build for *arch_list* can run on *capability*.

    A cubin built for ``sm_XY`` runs on devices of the same major version
    with minor >= Y; PTX built for ``compute_XY`` is JIT-compiled for any
    device with capability >= X.Y.
    """
    major, minor = capability
    for arch in arch_list:
        match = _ARCH_RE.match(arch)
        if match is None:
            continue
        kind, arch_major, arch_minor = (
            match.group(1),
            int(match.group(2)),
            int(match.group(3)),
        )
        if kind == "sm" and arch_major == major and arch_minor <= minor:
            return True
        if kind == "compute" and (arch_major, arch_minor) <= (major, minor):
            return True
    return False


@functools.lru_cache(maxsize=None)
def default_device() -> str:
    """Return "cuda" when a CUDA device usable by this torch build exists, else "cpu".

    When a device exists but the build has no kernels for it, a warning names
    the device, its capability and the build's architectures, once per process.
    """
    torch = import_torch("torch device selection")
    if not torch.cuda.is_available():
        return "cpu"
    capability = torch.cuda.get_device_capability()
    arch_list = torch.cuda.get_arch_list()
    if cuda_arch_supported(capability, arch_list):
        return "cuda"
    warnings.warn(
        f"CUDA device {torch.cuda.get_device_name()!r} (compute capability "
        f"{capability[0]}.{capability[1]}) is not supported by the installed "
        f"torch {torch.__version__} (built for {', '.join(arch_list)}); "
        "scitex-decorators uses the CPU. Install a torch build that supports "
        "this GPU to run on it.",
        UserWarning,
        stacklevel=2,
    )
    return "cpu"


__all__ = ["cuda_arch_supported", "default_device"]

# EOF
