# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# flash-linear-attention project LICENSE.
#
# Dense FP32 standalone extraction of fla/ops/utils/cumsum.py. It deliberately
# has no FLA imports and does not expose the unverified varlen path.

import torch
import triton
import triton.language as tl


LOCAL_CUMSUM_VECTOR_BS = 32
LOCAL_CUMSUM_NUM_WARPS = 4


@triton.jit
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    p_s = tl.make_block_ptr(
        s + bos * H + i_h,
        (T,),
        (H,),
        (i_t * BT,),
        (BT,),
        (0,),
    )
    p_o = tl.make_block_ptr(
        o + bos * H + i_h,
        (T,),
        (H,),
        (i_t * BT,),
        (BT,),
        (0,),
    )
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.jit
def chunk_local_cumsum_vector_kernel(
    s,
    o,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    p_s = tl.make_block_ptr(
        s + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def _validate_input(g: torch.Tensor, chunk_size: int) -> None:
    if g.ndim not in (3, 4):
        raise ValueError(f'expected [B, T, H] or [B, T, H, D], got shape {tuple(g.shape)}')
    if g.dtype != torch.float32:
        raise TypeError(f'standalone validated dtype is torch.float32, got {g.dtype}')
    if g.device.type != 'npu':
        raise ValueError(f'physical NPU tensor required, got device {g.device}')
    if not g.is_contiguous():
        raise ValueError('contiguous input required')
    if chunk_size <= 0 or chunk_size & (chunk_size - 1):
        raise ValueError(f'chunk_size must be a positive power of two, got {chunk_size}')


def chunk_local_cumsum(g: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Run dense FP32 chunk-local cumsum on a contiguous NPU tensor."""
    _validate_input(g, chunk_size)
    B, T, H = g.shape[:3]
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    output = torch.full_like(g, torch.nan, dtype=torch.float32)

    if g.ndim == 3:
        chunk_local_cumsum_scalar_kernel[(NT, B * H)](
            s=g,
            o=output,
            T=T,
            H=H,
            BT=BT,
            num_warps=LOCAL_CUMSUM_NUM_WARPS,
        )
        return output

    S = g.shape[3]

    def grid(meta):
        return (triton.cdiv(meta['S'], meta['BS']), NT, B * H)

    chunk_local_cumsum_vector_kernel[grid](
        s=g,
        o=output,
        T=T,
        H=H,
        S=S,
        BT=BT,
        BS=LOCAL_CUMSUM_VECTOR_BS,
        num_warps=LOCAL_CUMSUM_NUM_WARPS,
    )
    return output
