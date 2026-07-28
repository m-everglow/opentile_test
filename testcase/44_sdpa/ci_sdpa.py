"""Focused masked GQA SDPA kernel for Triton/OpenTile."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.runtime import driver


@triton.jit
def sdpa_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
    out_ptr,
    scale,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_os: tl.constexpr,
    stride_od: tl.constexpr,
    BATCH: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    SEQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    programs = tl.num_programs(axis=0)
    blocks_m = (SEQ + BLOCK_M - 1) // BLOCK_M
    tasks = BATCH * Q_HEADS * blocks_m

    for task in range(pid, tasks, programs):
        block_m = task % blocks_m
        q_head = (task // blocks_m) % Q_HEADS
        batch = task // (blocks_m * Q_HEADS)
        kv_head = q_head // (Q_HEADS // KV_HEADS)
        offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        q_mask = offs_m[:, None] < SEQ
        q_offsets = (
            batch * stride_qb
            + q_head * stride_qh
            + offs_m[:, None] * stride_qs
            + offs_d[None, :] * stride_qd
        )
        q = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)

        row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        row_sum = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

        for start_n in range(0, SEQ, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_mask = offs_n[:, None] < SEQ
            k_offsets = (
                batch * stride_kb
                + kv_head * stride_kh
                + offs_n[:, None] * stride_ks
                + offs_d[None, :] * stride_kd
            )
            v_offsets = (
                batch * stride_vb
                + kv_head * stride_vh
                + offs_n[:, None] * stride_vs
                + offs_d[None, :] * stride_vd
            )
            k = tl.load(k_ptr + k_offsets, mask=kv_mask, other=0.0)
            v = tl.load(v_ptr + v_offsets, mask=kv_mask, other=0.0)
            scores = tl.dot(q, tl.trans(k)) * scale
            valid = (offs_m[:, None] < SEQ) & (offs_n[None, :] < SEQ)
            allowed = tl.load(
                mask_ptr + offs_m[:, None] * SEQ + offs_n[None, :],
                mask=valid,
                other=False,
            )
            scores = tl.where(valid & allowed, scores, -1.0e6)

            next_max = tl.maximum(row_max, tl.max(scores, axis=1))
            alpha = tl.exp(row_max - next_max)
            probabilities = tl.exp(scores - next_max[:, None])
            next_sum = row_sum * alpha + tl.sum(probabilities, axis=1)
            acc = acc * alpha[:, None] + tl.dot(probabilities.to(v.dtype), v)
            row_max = next_max
            row_sum = next_sum

        output = acc / row_sum[:, None]
        out_offsets = (
            batch * stride_ob
            + q_head * stride_oh
            + offs_m[:, None] * stride_os
            + offs_d[None, :] * stride_od
        )
        tl.store(out_ptr + out_offsets, output, mask=q_mask)


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, scale: float):
    target = driver.active.get_current_target()
    backend = str(target.backend)
    if (backend != "opentile" and not backend.startswith("opentile_")) or driver.active.get_active_torch_device().type != "npu":
        raise RuntimeError("expected the OpenTile NPU route")
    batch, q_heads, seq_len, head_dim = q.shape
    kv_heads = k.shape[1]
    if q.dtype != torch.bfloat16 or k.shape != v.shape or q_heads % kv_heads:
        raise ValueError("focused scope is BF16 GQA with Q_HEADS divisible by KV_HEADS")
    if head_dim != 128 or mask.shape != (seq_len, seq_len) or mask.dtype != torch.bool:
        raise ValueError("focused scope requires D=128 and a square boolean mask")
    props = driver.active.utils.get_device_properties(torch.npu.current_device())
    output = torch.full_like(q, float("nan"))
    sdpa_kernel[(int(props["num_aicore"]),)](
        q,
        k,
        v,
        mask,
        output,
        scale,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *output.stride(),
        BATCH=batch,
        Q_HEADS=q_heads,
        KV_HEADS=kv_heads,
        SEQ=seq_len,
        HEAD_DIM=head_dim,
        BLOCK_M=64,
        BLOCK_N=64,
    )
    return output

