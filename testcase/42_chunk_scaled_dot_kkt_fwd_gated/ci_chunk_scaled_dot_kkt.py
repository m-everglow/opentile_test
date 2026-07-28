# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Standalone copy of the main-branch fla/ops/common/chunk_scaled_dot_kkt.py
# (master @ 3c4967f). Kernel, launch logic, and wrapper are identical to the
# main branch; only the fla package imports are replaced by the inlined
# equivalents below (exp2, prepare_chunk_indices), and the dispatch decorator
# is dropped because the fla backend registry is not available standalone.

import torch
import triton
import triton.language as tl


@triton.jit
def exp2(x):
    # fla.ops.utils.op.exp2 default path (FLA_USE_FAST_OPS unset).
    return tl.math.exp2(x.to(tl.float32))


def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    # Dependency-free equivalent of fla.ops.utils.index.prepare_chunk_indices.
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    counts = torch.div(lens + (chunk_size - 1), chunk_size, rounding_mode='floor')
    seg_id = torch.repeat_interleave(
        torch.arange(len(counts), device=counts.device), counts)
    starts = torch.cumsum(counts, 0) - counts
    intra = torch.arange(int(counts.sum()), device=counts.device) \
        - torch.repeat_interleave(starts, counts)
    return torch.stack([seg_id, intra], 1).to(cu_seqlens)


# Fixed launch parameters for backends that do not implement Triton autotune.
# BK=32 covers the current test shapes (K=32/64); larger K values are handled
# by the kernel's existing tiled loop.
_DEFAULT_BK = 32
_DEFAULT_NUM_WARPS = 4

@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    NT: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    # Original two-dimensional Triton grid mapping:
    # i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64)
    # Previous single-BH workaround while OpenTile did not pass grid-y:
    # i_t, i_bh = tl.program_id(0), 0
    #
    # OpenTile currently exposes one physical block id for all grid axes.
    # Launch a one-dimensional grid and decode it with grid-x (time chunk)
    # as the fastest-changing logical axis. This is equivalent to the
    # original (NT, B * HV) grid for dense and variable-length inputs.
    i_pid = tl.program_id(0)
    i_t = i_pid % NT
    i_bh = (i_pid // NT).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_b = tl.make_block_ptr(beta + bos*HV + i_h, (T,), (HV,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos*H + i_h // (HV // H)) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, tl.trans(b_k))

    if USE_G:
        p_g = tl.make_block_ptr(g + bos*HV + i_h, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A *= exp2(b_g_diff)
    b_A *= b_b[:, None]

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = tl.make_block_ptr(A + (bos*HV + i_h) * BT, (T, BT), (BT*HV, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


# @dispatch('common') from fla.ops.backends is intentionally omitted: without
# the fla package there is no backend registry, and the wrapper below is the
# main-branch common implementation itself.
def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]` where `H` is the number of query/key heads.
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, HV]`. Default: `None`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, HV]` where `HV` is the number of value/output heads.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`
        chunk_indices (torch.LongTensor):
            The chunk indices of the input tensor. Default: None.
    Returns:
        beta * K * K^T of shape `[B, T, HV, BT]` where `BT` is the chunk size.
        For GVA, H < HV and HV % H == 0. For standard attention, H == HV.
    """
    B, T, H, K, HV = *k.shape, beta.shape[2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = torch.empty(B, T, HV, BT, device=k.device, dtype=output_dtype)
    use_g = g is not None
    # Some third-party Triton runtimes do not accept None as a pointer
    # argument even when the corresponding constexpr branch is disabled.
    g_arg = g if use_g else beta
    # Original two-dimensional launch:
    # chunk_scaled_dot_kkt_fwd_kernel[(NT, B * HV)](...)
    #
    # OpenTile currently provides only one physical block id. Flatten the
    # logical (time-chunk, batch-head) grid and decode it inside the common
    # kernel instead of depending on program_id(1).
    chunk_scaled_dot_kkt_fwd_kernel[(NT * B * HV,)](
        k=k,
        g=g_arg,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        NT=NT,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BK=_DEFAULT_BK,
        IS_VARLEN=cu_seqlens is not None,
        USE_G=use_g,
        num_warps=_DEFAULT_NUM_WARPS,
    )
    return A
