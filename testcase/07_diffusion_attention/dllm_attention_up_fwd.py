"""Standalone OpenTile port of mojo_opset DLLM upper forward attention."""

from __future__ import annotations

from functools import cache
from typing import Any

import torch
import triton
import triton.language as tl
from triton.runtime import driver


@triton.jit
def micro_kernel_fwd(
    block_q,
    k,
    v,
    block_o,
    block_m,
    block_l,
    scale,
    offset_c,
    offset_c_ed,
    block_mask,
    idx_n,
    offs_h,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    boundary_mask=None,
):
    tl.static_assert(STRIDE_K_H == 1)
    tl.static_assert(STRIDE_V_H == 1)
    ptr_k = (
        k
        + (idx_n // GROUP_SIZE) * STRIDE_K_N
        + (offset_c + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
        + offs_h[None, :] * STRIDE_K_H
    )
    ptr_v = (
        v
        + (idx_n // GROUP_SIZE) * STRIDE_V_N
        + (offset_c + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
        + offs_h[None, :] * STRIDE_V_H
    )

    mask_kv = (offset_c + tl.arange(0, BLOCK_C))[:, None] < offset_c_ed
    block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
    block_k = tl.trans(block_k)
    block_s = tl.dot(block_q, block_k) * scale
    block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)
    if boundary_mask is not None:
        block_s += (boundary_mask.to(tl.float32) - 1.0) * 1e6
    if block_mask is not None:
        block_s = tl.where(block_mask, block_s, -1.0e6)
    block_m_1 = tl.maximum(block_m, tl.max(block_s, axis=1))
    block_s = tl.exp(block_s - block_m_1[:, None])
    block_l_1 = tl.exp(block_m - block_m_1) * block_l + tl.sum(block_s, axis=1)
    block_o = tl.exp(block_m - block_m_1)[:, None] * block_o
    block_o = block_o + tl.dot(block_s.to(tl.bfloat16), block_v)
    return block_o, block_m_1, block_l_1


@triton.jit(do_not_specialize=["cu_seqlens", "num_seqs", "S", "STRIDE_D_N"])
def kernel_da_fwd_u(
    q,
    k,
    v,
    o,
    lse,
    cu_seqlens,
    num_seqs,
    scale,
    mask_ul,
    mask_ur,
    GROUP_SIZE: tl.constexpr,
    S,
    N: tl.constexpr,
    H: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    STRIDE_D_N,
    STRIDE_MASK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pnum = tl.num_programs(axis=0)

    seq_st = 0
    offset_block_r_st = 0

    offset_r_local = tl.arange(0, BLOCK_R)[:, None]
    offset_c_local = tl.arange(0, BLOCK_R)[None, :]
    block_mask_ul = tl.load(mask_ul + offset_r_local * STRIDE_MASK + offset_c_local)
    block_mask_ur = tl.load(mask_ur + offset_r_local * STRIDE_MASK + offset_c_local)

    for idx_seq in range(num_seqs):
        seq_ed = tl.load(cu_seqlens + idx_seq)
        offset_block_r_ed = offset_block_r_st + tl.cdiv(seq_ed - seq_st, BLOCK_R)
        for task_id in range(
            offset_block_r_st * N
            + ((pid % pnum - offset_block_r_st * N % pnum + pnum) % pnum),
            offset_block_r_ed * N,
            pnum,
        ):
            idx_r = task_id // N - offset_block_r_st
            idx_n = task_id % N
            offs_h = tl.arange(0, H)

            ptr_q = (
                q
                + idx_n * STRIDE_Q_N
                + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None]
                * STRIDE_Q_S
                + offs_h[None, :] * STRIDE_Q_H
            )
            ptr_o = (
                o
                + idx_n * STRIDE_Q_N
                + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:, None]
                * STRIDE_Q_S
                + offs_h[None, :] * STRIDE_Q_H
            )
            ptr_lse = (
                lse
                + idx_n * STRIDE_D_N
                + (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] * STRIDE_D_S
            )

            mask_q = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[
                :, None
            ] < seq_ed
            mask_lse = (seq_st + idx_r * BLOCK_R + tl.arange(0, BLOCK_R))[:] < seq_ed

            block_q = tl.load(ptr_q, mask=mask_q, other=0.0)
            block_o = tl.full([BLOCK_R, H], 0.0, dtype=tl.float32)
            block_l = tl.full([BLOCK_R], 0.0, dtype=tl.float32)
            block_m = tl.full([BLOCK_R], -1e6, dtype=tl.float32)

            boundary_mask = (seq_st + idx_r * BLOCK_R + offset_r_local < seq_ed) & (
                seq_st + idx_r * BLOCK_R + offset_c_local < seq_ed
            )

            block_o, block_m, block_l = micro_kernel_fwd(
                block_q,
                k,
                v,
                block_o,
                block_m,
                block_l,
                scale,
                seq_st + idx_r * BLOCK_R,
                seq_ed,
                block_mask_ul,
                idx_n,
                offs_h,
                STRIDE_K_S,
                STRIDE_K_N,
                STRIDE_K_H,
                STRIDE_V_S,
                STRIDE_V_N,
                STRIDE_V_H,
                GROUP_SIZE,
                BLOCK_R,
                boundary_mask=boundary_mask,
            )

            block_o, block_m, block_l = micro_kernel_fwd(
                block_q,
                k,
                v,
                block_o,
                block_m,
                block_l,
                scale,
                S + seq_st + idx_r * BLOCK_R,
                S + seq_ed,
                block_mask_ur,
                idx_n,
                offs_h,
                STRIDE_K_S,
                STRIDE_K_N,
                STRIDE_K_H,
                STRIDE_V_S,
                STRIDE_V_N,
                STRIDE_V_H,
                GROUP_SIZE,
                BLOCK_R,
                boundary_mask=boundary_mask,
            )

            for idx_tile_r in range(
                idx_r * BLOCK_R // BLOCK_C * BLOCK_C // BLOCK_R,
                idx_r,
            ):
                block_o, block_m, block_l = micro_kernel_fwd(
                    block_q,
                    k,
                    v,
                    block_o,
                    block_m,
                    block_l,
                    scale,
                    S + seq_st + idx_tile_r * BLOCK_R,
                    S + seq_ed,
                    None,
                    idx_n,
                    offs_h,
                    STRIDE_K_S,
                    STRIDE_K_N,
                    STRIDE_K_H,
                    STRIDE_V_S,
                    STRIDE_V_N,
                    STRIDE_V_H,
                    GROUP_SIZE,
                    BLOCK_R,
                )

            for idx_c in range(idx_r * BLOCK_R // BLOCK_C):
                block_o, block_m, block_l = micro_kernel_fwd(
                    block_q,
                    k,
                    v,
                    block_o,
                    block_m,
                    block_l,
                    scale,
                    S + seq_st + idx_c * BLOCK_C,
                    S + seq_ed,
                    None,
                    idx_n,
                    offs_h,
                    STRIDE_K_S,
                    STRIDE_K_N,
                    STRIDE_K_H,
                    STRIDE_V_S,
                    STRIDE_V_N,
                    STRIDE_V_H,
                    GROUP_SIZE,
                    BLOCK_C,
                )

            block_o = block_o / block_l[:, None]
            block_lse = tl.log(block_l) + block_m
            tl.store(ptr_o, block_o, mask=mask_q)
            tl.store(ptr_lse, block_lse, mask=mask_lse)

        seq_st = seq_ed
        offset_block_r_st = offset_block_r_ed


@cache
def _device_properties() -> tuple[torch.device, int, int, str]:
    target = driver.active.get_current_target()
    backend = str(target.backend)
    if backend != "opentile" and not backend.startswith("opentile_"):
        raise RuntimeError(f"expected OpenTile target, got {backend!r}")
    active_device = driver.active.get_active_torch_device()
    if active_device.type != "npu":
        raise RuntimeError(f"expected NPU device, got {active_device}")
    logical_device = torch.npu.current_device()
    properties: dict[str, Any] = driver.active.utils.get_device_properties(
        logical_device
    )
    num_aicore = int(properties["num_aicore"])
    num_vectorcore = int(properties["num_vectorcore"])
    if num_aicore <= 0 or num_vectorcore <= 0:
        raise RuntimeError(f"invalid NPU properties: {properties!r}")
    return active_device, num_aicore, num_vectorcore, backend


def apply_dllm_attention_up_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlen: torch.Tensor,
    scale: float,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.dtype != torch.bfloat16:
        raise ValueError("DLLM source contract requires BF16")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, and v must share the BF16 dtype")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must be rank-3 tensors")
    if q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0]:
        raise ValueError("q, k, and v must share TOTAL_SEQ")
    if q.shape[0] % 2:
        raise ValueError("TOTAL_SEQ must contain equal upper/lower halves")
    if k.shape[1] != v.shape[1] or q.shape[1] % k.shape[1]:
        raise ValueError("grouped heads require Hq % Hkv == 0")
    if q.shape[2] != k.shape[2] or k.shape[2] != v.shape[2]:
        raise ValueError("q, k, and v must share HEAD_DIM")
    if q.shape[2] not in (64, 128):
        raise ValueError("HEAD_DIM must be 64 or 128")
    if cu_seqlen.ndim != 1 or cu_seqlen.dtype != torch.int32:
        raise ValueError("cu_seqlen must be a rank-1 int32 tensor")

    active_device, num_aicore, num_vectorcore, backend = _device_properties()
    sequence_half = q.shape[0] // 2
    out = torch.full_like(q, float("nan"), dtype=torch.float32)
    out[sequence_half:].zero_()
    lse = torch.full(
        (q.shape[0], q.shape[1]),
        float("nan"),
        device=q.device,
        dtype=torch.float32,
    )
    lse[sequence_half:].zero_()

    block_mask = 64
    offset_r = torch.arange(block_mask, device=q.device)[:, None]
    offset_c = torch.arange(block_mask, device=q.device)[None, :]
    chunk_r = offset_r // block_size
    chunk_c = offset_c // block_size
    mask_ul = (chunk_r == chunk_c).contiguous()
    mask_ur = (chunk_r > chunk_c).contiguous()

    # OpenTile submits the kernel directly to the ACL stream, while the tensor
    # producers above may still be pending in torch_npu's software task queue.
    # Flush them before launch so output initialization and mask construction
    # cannot execute after the kernel.
    torch.npu.synchronize()

    kernel_da_fwd_u[(num_aicore,)](
        q,
        k,
        v,
        out,
        lse,
        cu_seqlen,
        cu_seqlen.shape[0],
        scale,
        mask_ul,
        mask_ur,
        q.shape[1] // k.shape[1],
        sequence_half,
        q.shape[1],
        q.shape[2],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        lse.stride(0),
        lse.stride(1),
        mask_ul.stride(0),
        BLOCK_R=32,
        BLOCK_C=128,
    )
    print(
        "[OPENTILE_E2E] op=dllm_attention_up_fwd "
        f"route_ok backend={backend} active_device={active_device} "
        f"logical_device={torch.npu.current_device()} "
        f"num_aicore={num_aicore} num_vectorcore={num_vectorcore} "
        "kernel_mode=mix BLOCK_R=32 BLOCK_C=128"
    )
    return out, lse
