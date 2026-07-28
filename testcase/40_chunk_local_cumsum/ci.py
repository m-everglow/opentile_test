# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# flash-linear-attention project LICENSE.
#
# Standalone extraction of tests/ops/utils/test_cumsum.py. It validates the
# true-NPU-tested dense FP32 contract without importing flash-linear-attention.

import os

os.environ.setdefault('OPENTILE_KERNEL_MODE', 'aiv')
os.environ.setdefault('OPENTILE_ENABLE_APPROX', '0')
os.environ.setdefault('OPENTILE_ENABLE_FTZ', '0')

import pytest
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None
else:
    torch.opentile_ascend = torch.npu

import triton.runtime.driver as driver

from ci_chunk_local_cumsum import chunk_local_cumsum


CASES = [
    pytest.param(1, 63, 1, 16, 30, id='B1-T63-H1-C16-D30-fp32'),
    pytest.param(2, 500, 4, 32, 60, id='B2-T500-H4-C32-D60-fp32'),
    pytest.param(2, 1000, 5, 64, 128, id='B2-T1000-H5-C64-D128-fp32'),
    pytest.param(3, 1024, 6, 64, 500, id='B3-T1024-H6-C64-D500-fp32'),
    pytest.param(4, 2048, 8, 128, 1024, id='B4-T2048-H8-C128-D1024-fp32'),
]


def _compile_only_enabled() -> bool:
    return os.environ.get('TRITON_COMPILE_ONLY', '0') == '1' or os.environ.get('OPENTILE_COMPILE_ONLY', '0') == '1'


@pytest.fixture(scope='session')
def device() -> torch.device:
    if torch_npu is None:
        pytest.fail('torch_npu is required for physical-NPU execution; no CPU/CUDA fallback is allowed')
    if not hasattr(torch, 'npu') or not torch.npu.is_available():
        pytest.fail('torch_npu is installed, but torch.npu reports no available physical NPU')

    target = driver.active.get_current_target()
    backend = str(target.backend)
    if backend != 'opentile' and not backend.startswith('opentile_'):
        pytest.fail(f'expected an OpenTile Triton backend, got {backend!r}')
    if os.environ.get('OPENTILE_KERNEL_MODE', '').lower() != 'aiv':
        pytest.fail('OPENTILE_KERNEL_MODE=aiv is required for the cumsum vector kernels')

    selected = torch.device('npu', torch.npu.current_device())
    print(
        f'OpenTile route: backend={backend}, arch={target.arch}, '
        f'device={selected}, mode={os.environ["OPENTILE_KERNEL_MODE"]}',
    )
    return selected


def _chunkwise_reference(source: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return torch.cat(
        [source[:, start:start + chunk_size].float().cumsum(1) for start in range(0, source.shape[1], chunk_size)],
        dim=1,
    )


def _assert_close(prefix: str, reference: torch.Tensor, actual: torch.Tensor, ratio: float) -> None:
    assert reference.shape == actual.shape, f'{prefix}: shape mismatch: {reference.shape} != {actual.shape}'
    assert not torch.isnan(reference).any(), f'{prefix}: NaN detected in reference'
    assert not torch.isnan(actual).any(), f'{prefix}: NaN detected in actual (possible unwritten output)'
    absolute_error = (reference - actual).flatten().abs().max().item()
    rms_error = (reference - actual).flatten().square().mean().sqrt().item()
    rms_base = reference.flatten().square().mean().sqrt().item()
    error_ratio = rms_error / (rms_base + 1e-8)
    message = f'{prefix:>24} diff: {absolute_error:.6f} ratio: {error_ratio:.6f}'
    print(message)
    if absolute_error > 1e-6:
        assert error_ratio < ratio, message


@pytest.mark.parametrize(('B', 'T', 'H', 'C', 'D'), CASES)
def test_local_cumsum(
    B: int,
    T: int,
    H: int,
    C: int,
    D: int,
    device: torch.device,
):
    generator = torch.Generator(device='cpu').manual_seed(42)

    scalar_cpu = torch.randn(B, T, H, dtype=torch.float32, generator=generator)
    scalar_reference = _chunkwise_reference(scalar_cpu, C)
    scalar_actual = chunk_local_cumsum(scalar_cpu.to(device), chunk_size=C)

    vector_cpu = torch.randn(B, T, H, D, dtype=torch.float32, generator=generator)
    vector_reference = _chunkwise_reference(vector_cpu, C)
    vector_actual = chunk_local_cumsum(vector_cpu.to(device), chunk_size=C)

    if _compile_only_enabled():
        print('compile-only: scalar/vector object generation requested; numerical comparison skipped')
        return

    torch.npu.synchronize()
    _assert_close('local_cumsum_scalar', scalar_reference, scalar_actual.cpu(), 1e-3)
    _assert_close('local_cumsum_vector', vector_reference, vector_actual.cpu(), 1e-3)
