#!/usr/bin/env python3
"""Tests for scitex_decorators._runtime._device.

``cuda_arch_supported`` is pure (capability + arch list in, bool out), so
the GPU-generation cases are exercised with explicit inputs. The regression
it guards: a GTX 1070 (sm_61) under a torch build for sm_75+ reports
``torch.cuda.is_available() == True`` but has no kernels, and every
decorated call crashed with "no kernel image is available".
"""

import pytest

from scitex_decorators._runtime._device import cuda_arch_supported, default_device

torch = pytest.importorskip("torch")

_BUILD_SM75_PLUS = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]


def test_cuda_arch_supported_rejects_pascal_on_sm75_plus_build():
    # Arrange
    capability = (6, 1)
    # Act
    supported = cuda_arch_supported(capability, _BUILD_SM75_PLUS)
    # Assert
    assert supported is False


def test_cuda_arch_supported_accepts_exact_sm_match():
    # Arrange
    capability = (8, 0)
    # Act
    supported = cuda_arch_supported(capability, _BUILD_SM75_PLUS)
    # Assert
    assert supported is True


def test_cuda_arch_supported_accepts_same_major_newer_minor():
    # Arrange
    capability = (8, 9)
    # Act
    supported = cuda_arch_supported(capability, ["sm_80"])
    # Assert
    assert supported is True


def test_cuda_arch_supported_rejects_same_major_older_minor():
    # Arrange
    capability = (8, 0)
    # Act
    supported = cuda_arch_supported(capability, ["sm_86"])
    # Assert
    assert supported is False


def test_cuda_arch_supported_parses_three_digit_arch():
    # Arrange
    capability = (12, 0)
    # Act
    supported = cuda_arch_supported(capability, ["sm_120"])
    # Assert
    assert supported is True


def test_cuda_arch_supported_accepts_older_ptx_via_jit():
    # Arrange
    capability = (8, 9)
    # Act
    supported = cuda_arch_supported(capability, ["compute_75"])
    # Assert
    assert supported is True


def test_cuda_arch_supported_rejects_newer_ptx():
    # Arrange
    capability = (8, 6)
    # Act
    supported = cuda_arch_supported(capability, ["compute_90"])
    # Assert
    assert supported is False


def test_cuda_arch_supported_rejects_empty_arch_list():
    # Arrange
    capability = (8, 6)
    # Act
    supported = cuda_arch_supported(capability, [])
    # Assert
    assert supported is False


def test_default_device_can_execute_a_kernel():
    # Arrange
    device = default_device()
    # Act
    total = torch.ones(4, device=device).sum().item()
    # Assert
    assert total == 4.0
