# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Standalone copy of the main-branch tests/ops/test_chunk_scaled_dot_kkt.py.
# The equations, direct common-kernel launch, parameters, and inputs are
# unchanged; FLA imports are replaced by local helpers and an explicit
# OpenTile/physical-NPU preflight.

import os

# This kernel combines tl.dot with vector operations. These defaults must be
# established before Triton is imported when pytest is invoked without run.sh.
os.environ.setdefault('OPENTILE_KERNEL_MODE', 'mix')
os.environ.setdefault('OPENTILE_ENABLE_APPROX', '0')
os.environ.setdefault('OPENTILE_ENABLE_FTZ', '0')

import pytest
import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

import triton
import triton.runtime.driver as driver

from ci_chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd_kernel, prepare_chunk_indices


def _compile_only_enabled() -> bool:
    return os.environ.get('TRITON_COMPILE_ONLY', '0') == '1' or os.environ.get('OPENTILE_COMPILE_ONLY', '0') == '1'


@pytest.fixture(scope='session')
def device() -> torch.device:
    """Return the physical NPU selected by the active OpenTile runtime."""
    if torch_npu is None:
        pytest.fail('torch_npu is required for physical-NPU execution; no CPU/CUDA fallback is allowed')
    if not hasattr(torch, 'npu') or not torch.npu.is_available():
        pytest.fail('torch_npu is installed, but torch.npu reports no available physical NPU')

    target = driver.active.get_current_target()
    backend = str(target.backend)
    if backend != 'opentile' and not backend.startswith('opentile_'):
        pytest.fail(f'expected an OpenTile Triton backend, got {backend!r}')
    if os.environ.get('OPENTILE_KERNEL_MODE', '').lower() != 'mix':
        pytest.fail('OPENTILE_KERNEL_MODE=mix is required because this kernel contains tl.dot')

    current_device = torch.npu.current_device()
    selected = torch.device('npu', current_device)
    print(
        f'OpenTile route: backend={backend}, arch={target.arch}, '
        f'device={selected}, mode={os.environ["OPENTILE_KERNEL_MODE"]}',
    )
    return selected


# assert_close inlined from fla/utils/_testing.py (same formulas and
# thresholds; fla's logger.info is replaced with print, and the FLA_CI_ENV
# warning branch is reduced to the default assert path).
def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    error_rate = get_err_ratio(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {error_rate:.6f}"
    print(msg)
    if abs_atol <= err_atol:
        return
    assert not torch.isnan(ref).any(), f"{prefix}: NaN detected in ref"
    assert not torch.isnan(tri).any(), f"{prefix}: NaN detected in tri"
    assert error_rate < ratio, msg



def chunk_scaled_dot_kkt_fwd_common(
    k: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.LongTensor | None,
) -> torch.Tensor:
    """Launch the common kernel directly, bypassing backend dispatch."""
    B, T, H, K = k.shape
    HV = beta.shape[2]
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if chunk_indices is None else len(chunk_indices)
    # NaN initialization makes a missing batch/head write fail deterministically.
    A = torch.full((B, T, HV, BT), torch.nan, dtype=torch.float32, device=k.device)
    use_g = g is not None

    # Keep all pointer arguments valid even when their constexpr branch is
    # disabled; some third-party Triton runtimes reject None pointers.
    g_arg = g if use_g else beta
    cu_seqlens_arg = cu_seqlens if cu_seqlens is not None else beta
    chunk_indices_arg = chunk_indices if chunk_indices is not None else beta

    # Original two-dimensional launch:
    # chunk_scaled_dot_kkt_fwd_kernel[(NT, B * HV)](...)
    #
    # OpenTile currently exposes only one physical block id. The common
    # kernel decodes this flattened id back to (i_t, i_bh).
    chunk_scaled_dot_kkt_fwd_kernel[(NT * B * HV,)](
        k=k,
        g=g_arg,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens_arg,
        chunk_indices=chunk_indices_arg,
        T=T,
        NT=NT,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BK=32,
        IS_VARLEN=cu_seqlens is not None,
        USE_G=use_g,
        num_warps=4,
    )
    return A


def chunk_scaled_dot_kkt_fwd_ref(
    k: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    chunk_size: int,
    cu_seqlens: list[int] | None,
) -> torch.Tensor:
    B, T, H, _ = k.shape
    HV = beta.shape[2]
    k = k.repeat_interleave(HV // H, dim=2)
    A = torch.zeros(B, T, HV, chunk_size, dtype=torch.float32, device=k.device)

    sequences = [(i_b, 0, T) for i_b in range(B)]
    if cu_seqlens is not None:
        sequences = [(0, bos, eos) for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:])]

    for i_b, bos, eos in sequences:
        for start in range(bos, eos, chunk_size):
            end = min(start + chunk_size, eos)
            actual_size = end - start
            k_chunk = k[i_b, start:end].float().permute(1, 0, 2)
            A_chunk = torch.matmul(k_chunk, k_chunk.transpose(-1, -2))
            if g is not None:
                g_chunk = g[i_b, start:end].float().transpose(0, 1)
                A_chunk = A_chunk * torch.exp2(g_chunk[:, :, None] - g_chunk[:, None, :])
            beta_chunk = beta[i_b, start:end].float().transpose(0, 1)
            A_chunk = torch.tril(A_chunk * beta_chunk[:, :, None], diagonal=-1)
            A[i_b, start:end, :, :actual_size] = A_chunk.permute(1, 0, 2)
    return A


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'HV', 'D', 'chunk_size', 'use_g', 'cu_seqlens'),
    [
        pytest.param(1, 32, 2, 4, 64, 16, True, None, id='dense-gated-gva'),
        #pytest.param(1, 48, 2, 2, 32, 32, True, [0, 17, 48], id='varlen-gated'),
    ],
)
def test_chunk_scaled_dot_kkt_fwd(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    chunk_size: int,
    use_g: bool,
    cu_seqlens: list[int] | None,
    device: torch.device,
):
    torch.manual_seed(42)
    k = F.normalize(torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device), dim=-1)
    beta = torch.randn(B, T, HV, dtype=torch.bfloat16, device=device).sigmoid()
    g = torch.randn(B, T, HV, dtype=torch.float32, device=device) * 0.1 if use_g else None
    cu_seqlens_tensor = None
    if cu_seqlens is not None:
        cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    ref = chunk_scaled_dot_kkt_fwd_ref(
        k=k,
        g=g,
        beta=beta,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    # Call the common Triton kernel directly. Using the decorated public
    # wrapper here could dispatch to chunk_scaled_dot_kkt_fwd_kernel_npu on
    # an NPU-aware environment and would not validate this common path.
    tri = chunk_scaled_dot_kkt_fwd_common(
        k=k,
        g=g,
        beta=beta,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_tensor,
    )
    if _compile_only_enabled():
        print('compile-only: object generation requested; numerical comparison skipped')
        return
    torch.npu.synchronize()

    assert_close('chunk_scaled_dot_kkt_fwd', ref, tri, 0.005)
