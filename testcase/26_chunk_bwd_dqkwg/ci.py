"""Standalone NPU chunk_bwd_dqkwg kernel + CPU golden.

[realhw_run_20260727] 与 testcase/26 ci.py 同 kernel 同流程; kernel 为 3D grid
(NK, NT, B*HV), 真机运行需要 launcher grid 补丁 (10_patch_runtime.py).

Run directly:
  python3 test_chunk_bwd_dqkwg_standalone.py

Run with pytest:
  pytest -s --assert=plain test_chunk_bwd_dqkwg_standalone.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_BACKEND", "opentile")
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/opentile_mr330_rework_dq_cache")

import torch
import triton
import triton.language as tl
import torch.nn.functional as F


# ============================================================
# NPU kernels (standard FLA, no _npu variant)
# ============================================================

@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))



def prepare_lens(cu_seqlens):
    return cu_seqlens[1:] - cu_seqlens[:-1]


def _segmented_arange(chunk_counts):
    seg_id = torch.repeat_interleave(torch.arange(len(chunk_counts)), chunk_counts)
    intra_chunk_idx = torch.cat([torch.arange(c) for c in chunk_counts])
    return seg_id, intra_chunk_idx


def prepare_chunk_indices(cu_seqlens, chunk_size, cu_seqlens_cpu=None):
    src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    chunk_counts = (prepare_lens(src) + (chunk_size - 1)).div(chunk_size, rounding_mode='floor')
    seg_id, intra_chunk_idx = _segmented_arange(chunk_counts)
    return torch.stack([seg_id, intra_chunk_idx], 1).to(cu_seqlens)


def prepare_chunk_offsets(cu_seqlens, chunk_size):
    return F.pad(triton.cdiv(prepare_lens(cu_seqlens), chunk_size), (1, 0), value=0).cumsum(-1)



@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dv(
    q,
    k,
    g,
    g_gamma,
    do,
    dv,
    dh,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    b_dv = tl.zeros([BT, BV], dtype=tl.float32)

    # offset calculation
    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    do += (bos * HV + i_h) * V
    dv += (bos * HV + i_h) * V
    dh += (i_tg * HV + i_h).to(tl.int64) * K*V

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, b_q)
        if STATE_V_FIRST:
            p_dh = tl.make_block_ptr(dh, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            b_dh = tl.trans(tl.load(p_dh, boundary_check=(0, 1)))
        else:
            p_dh = tl.make_block_ptr(dh, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dv += tl.dot(b_k, b_dh.to(b_k.dtype))

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    if USE_G:
        g += bos * HV + i_h
        p_g = tl.make_block_ptr(g, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * HV)
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_g_last = b_gamma * min(BT, T - i_t * BT)

    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    if USE_G or USE_G_GAMMA:
        b_A = tl.where(m_A, b_A * exp2(b_g[None, :] - b_g[:, None]) * scale, 0).to(do.dtype.element_ty)
        b_dv *= tl.where(m_t, exp2(-b_g + b_g_last), 0)[:, None]
    else:
        b_A = tl.where(m_A, b_A * scale, 0).to(do.dtype.element_ty)
    p_do = tl.make_block_ptr(do, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dv = tl.make_block_ptr(dv, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dv += tl.dot(b_A.to(b_do.dtype), b_do)
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dv_local(
    q,
    k,
    g,
    g_gamma,
    A,
    do,
    dv,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_A: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    do += (bos * HV + i_h) * V
    dv += (bos * HV + i_h) * V

    if USE_A:
        p_A = tl.make_block_ptr(A + (bos * HV + i_h) * BT, (BT, T), (1, HV*BT), (0, i_t * BT), (BT, BT), (0, 1))
        b_A = tl.load(p_A, boundary_check=(0, 1))
    else:
        if USE_G:
            g += bos * HV + i_h
            p_g = tl.make_block_ptr(g, (T,), (HV,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
        if USE_G_GAMMA:
            b_gamma = tl.load(g_gamma + i_h)
            b_g = b_gamma * (tl.arange(0, BT) + 1)

        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))

            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_A += tl.dot(b_k, b_q) * scale
        if USE_G or USE_G_GAMMA:
            b_A *= exp2(b_g[None, :] - b_g[:, None])
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0).to(do.dtype.element_ty)

    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(do, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dqkwg(
    q,
    k,
    v,
    g,
    g_gamma,
    h,
    do,
    dh,
    dq,
    dk,
    dw,
    dv,
    dg,
    cu_seqlens,
    chunk_indices,
    scale,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_DW: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV

    all = B * T
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    v += (bos * HV + i_h) * V
    do += (bos * HV + i_h) * V
    h += (i_tg * HV + i_h).to(tl.int64) * K*V
    dh += (i_tg * HV + i_h).to(tl.int64) * K*V
    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    dq += (bos * HV + i_h) * K
    dk += (bos * HV + i_h) * K

    # for delta rule only
    if USE_DW:
        dw += (bos * HV + i_h) * K
        dv += (bos * HV + i_h) * V

    if USE_G:
        dg += i_k * all * HV
        b_dg_last = tl.zeros([1], dtype=tl.float32) if USE_G else None
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_g_last = b_gamma * min(BT, T - i_t * BT)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32) if USE_DW else None

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        if STATE_V_FIRST:
            p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            p_dh = tl.make_block_ptr(dh, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
        else:
            p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        if USE_G:
            b_dg_last += (tl.sum(b_h * b_dh))
        # [BT, BV] @ [BV, BT] -> [BT, BT]
        b_ds += tl.dot(b_do, tl.trans(b_v))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))
        if USE_DW:
            p_dv = tl.make_block_ptr(dv, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_dw += tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype))

    if USE_DW:
        p_dw = tl.make_block_ptr(dw, (T, K), (HV*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()
    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))

    p_dq = tl.make_block_ptr(dq, (T, K), (HV*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk, (T, K), (HV*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    if USE_G:
        g += bos * HV + i_h
        dg += bos * HV + i_h
        p_g = tl.make_block_ptr(g, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * HV)
        b_dg_last *= exp2(b_g_last)
        b_dq = b_dq * exp2(b_g)[:, None] * scale
        b_dk = b_dk * tl.where(m_t, exp2(-b_g + b_g_last), 0)[:, None]
        b_dg_last += tl.sum(b_dk * b_k)

        b_ds = tl.where(m_A, b_ds * exp2(b_g[:, None] - b_g[None, :]), 0) * scale
        b_ds = b_ds.to(b_k.dtype)
        # [BT, BK]
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q)

        b_dg = tl.sum(b_dq * b_q, axis=1) - tl.sum(b_dk * b_k, axis=1)

        p_dg = tl.make_block_ptr(dg, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        # (SY 09/21) revcumsum in a separate kernel due to strange triton compiler issue
        # b_dg = tl.dot(tl.where(o_t[:, None] <= o_t[None, :], 1., 0.), b_dg, allow_tf32=False) + b_dg_last)
        b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last)
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))

    elif USE_G_GAMMA:
        b_dq = b_dq * exp2(b_g)[:, None] * scale
        b_dk = b_dk * tl.where(m_t, exp2(-b_g + b_g_last), 0)[:, None]
        b_ds = tl.where(m_A, b_ds * exp2(b_g[:, None] - b_g[None, :]), 0) * scale
        b_ds = b_ds.to(b_k.dtype)
        # [BT, BK]
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q)
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    else:
        b_ds = tl.where(m_A, b_ds, 0)
        b_ds = b_ds.to(b_k.dtype)
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q) * scale
        b_dq *= scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


def chunk_bwd_dqkwg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    w: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    dv: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    if False:
        CONST_TILING = 128
    elif False:
        CONST_TILING = 64
    else:
        CONST_TILING = 64
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NK = triton.cdiv(K, BK)
    dq = q.new_empty(B, T, HV, K)
    dk = k.new_empty(B, T, HV, K)
    dg = torch.empty(NK, *g.shape, dtype=torch.float32, device=g.device) if g is not None else None
    dw = torch.empty_like(w) if w is not None else None

    grid = (NK, NT, B * HV)
    chunk_bwd_kernel_dqkwg[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        h=h,
        do=do,
        dh=dh,
        dw=dw,
        dq=dq,
        dk=dk,
        dv=dv,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        B=B,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        STATE_V_FIRST=state_v_first,
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        USE_DW=w is not None,
        IS_VARLEN=cu_seqlens is not None,
    )

    if H != HV:
        dq = dq.view(B, T, H, HV // H, K).sum(3)
        dk = dk.view(B, T, H, HV // H, K).sum(3)
    if dg is not None:
        dg = dg.sum(0)
    return dq, dk, dw, dg


def _num_vector_cores():
    return triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]


def _setup_npu() -> torch.device:
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("torch_npu is required on the real-hardware host") from exc
    device_id = int(os.environ.get("OPENTILE_TEST_DEVICE", "0"))
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is false")
    torch.npu.set_device(device_id)
    return torch.device(f"npu:{device_id}")


def _compare(name: str, actual: torch.Tensor, golden: torch.Tensor, atol: float, rtol: float) -> None:
    actual_f32 = actual.detach().cpu().float()
    golden_f32 = golden.detach().cpu().float()
    absolute = (actual_f32 - golden_f32).abs()
    tolerance = atol + rtol * golden_f32.abs()
    passed = absolute <= tolerance
    mismatch = int((~passed).sum().item())
    total = passed.numel()
    rms = float(torch.sqrt(((actual_f32 - golden_f32) ** 2).mean())) if total else 0.0
    max_abs = float(absolute.max().item()) if total else 0.0
    print(
        f"{name} shape={tuple(actual.shape)} "
        f"pass={total - mismatch}/{total} bad={mismatch} "
        f"max_abs={max_abs:.9g} rms={rms:.9g} "
        f"atol={atol} rtol={rtol}"
    )
    return mismatch



SEED = 450045
B, T, H, HV, D, BT = 2, 128, 2, 2, 64, 64
NT = T // BT
SCALE = D ** -0.5
TOLERANCES = {
    "f32": (1e-4, 1e-4),
    "f16": (1e-3, 1e-3),
    "bf16": (5e-3, 5e-3),
}
DTYPES = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
}


def _make_gate(gen, b, t, hv):
    raw = torch.rand((b, t, hv), generator=gen) * 0.01
    g = torch.empty_like(raw)
    for s in range(0, t, BT):
        g[:, s:s+BT] = -torch.cumsum(raw[:, s:s+BT], dim=1)
    return g


def _chunk_fwd_o(q, k, v, h, g):
    o = torch.empty((B, T, HV, D), dtype=q.dtype)
    m = torch.tril(torch.ones(BT, BT))
    for it in range(NT):
        s, e = it * BT, (it + 1) * BT
        qc = q[:, s:e].float()
        kc = k[:, s:e].float()
        vc = v[:, s:e].float()
        hc = h[:, it].float()
        gc = g[:, s:e]
        oi = torch.einsum('bthk,bhkv->bthv', qc, hc) * torch.exp2(gc)[:, :, :, None]
        scores = torch.matmul(qc.permute(0, 2, 1, 3), kc.permute(0, 2, 1, 3).transpose(-1, -2))
        gd = gc.permute(0, 2, 1)[:, :, :, None] - gc.permute(0, 2, 1)[:, :, None, :]
        scores = scores * torch.exp2(gd.masked_fill(~m.bool(), 0.0)) * m
        intra = torch.matmul(scores, vc.permute(0, 2, 1, 3))
        o[:, s:e] = ((oi.permute(0, 2, 1, 3) + intra) * SCALE).permute(0, 2, 1, 3).to(q.dtype)
    return o


def _cpu_golden_dqkwg(q, k, v, do, h, dh, dv, g, w, scale):
    """CPU golden for chunk_bwd_dqkwg using autograd."""
    qr = q.float().requires_grad_()
    kr = k.float().requires_grad_()
    gr = g.clone().requires_grad_()
    o = _chunk_fwd_o(qr, kr, v, h, gr)
    hend = torch.empty((B, NT, HV, D, D), dtype=torch.float32)
    for it in range(NT):
        s, e = it * BT, (it + 1) * BT
        gc = gr[:, s:e]
        gl = gc[:, -1]
        bv = v[:, s:e].float() * torch.exp2(gl[:, None, :, None] - gc[:, :, :, None])
        hend[:, it] = h[:, it].float() * torch.exp2(gl)[:, :, None, None] + torch.einsum('bthk,bthv->bhkv', kr[:, s:e], bv)
    loss = (do.float() * o.float()).sum() + (dh.float() * hend).sum()
    dq, dk, dg = torch.autograd.grad(loss, [qr, kr, gr])
    dg = dg * (1.0 / 0.6931471805599453)
    dw = torch.empty((B, T, HV, D), dtype=torch.float32)
    for it in range(NT):
        s, e = it * BT, (it + 1) * BT
        dw[:, s:e] = -torch.einsum('bthv,bhkv->bthk', dv[:, s:e].float(), h[:, it].float())
    return dq, dk, dg, dw


def run_one(dtype_name: str, device: torch.device) -> None:
    torch_dtype = DTYPES[dtype_name]
    atol, rtol = TOLERANCES[dtype_name]
    gen = torch.Generator().manual_seed(SEED)

    def rand_bf16(shape, scale=0.03):
        return (torch.randn(shape, generator=gen) * scale).to(torch_dtype)

    q = rand_bf16((B, T, H, D))
    k = rand_bf16((B, T, H, D))
    v = rand_bf16((B, T, HV, D))
    do = rand_bf16((B, T, HV, D))
    h = rand_bf16((B, NT, HV, D, D))
    dh = rand_bf16((B, NT, HV, D, D))
    dv = rand_bf16((B, T, HV, D))
    g = (torch.randn((B, T, HV), generator=gen) * 0.03)
    w = rand_bf16((B, T, HV, D))

    ref_dq, ref_dk, ref_dg, ref_dw = _cpu_golden_dqkwg(q, k, v, do, h, dh, dv, g, w, SCALE)

    q_d = q.to(device)
    k_d = k.to(device)
    v_d = v.to(device)
    do_d = do.to(device)
    h_d = h.to(device)
    dh_d = dh.to(device)
    dv_d = dv.to(device)
    g_d = g.to(device)
    w_d = w.to(device)

    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q_d, k=k_d, v=v_d, do=do_d,
        h=h_d, dh=dh_d, w=w_d, g=g_d,
        dv=dv_d, scale=SCALE,
    )
    torch.npu.synchronize()

    dq_bad = _compare(f"dq_{dtype_name}", dq, ref_dq.to(dq.dtype), atol, rtol)
    dk_bad = _compare(f"dk_{dtype_name}", dk, ref_dk.to(dk.dtype), atol, rtol)
    dw_bad = _compare(f"dw_{dtype_name}", dw, ref_dw.to(dw.dtype), atol, rtol)
    dg_bad = _compare(f"dg_{dtype_name}", dg, ref_dg, atol, rtol) if dg is not None else 0

    del q_d, k_d, v_d, do_d, h_d, dh_d, dv_d, g_d, w_d, dq, dk, dw, dg
    torch.npu.empty_cache()
    return dq_bad, dk_bad, dw_bad, dg_bad


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _print_identity_and_artifacts() -> None:
    print("DIAGNOSTIC_IDENTITY")
    for tool in ("tile-opt", "tile-translate", "ccec"):
        found = shutil.which(tool)
        resolved = str(Path(found).resolve()) if found else "NOT_FOUND"
        sha = _sha256(Path(resolved)) if found else "N/A"
        print(f"  {tool}: path={resolved} sha256={sha}")
    print(f"  TRITON_CACHE_DIR={os.environ['TRITON_CACHE_DIR']}")
    cache = Path(os.environ["TRITON_CACHE_DIR"])
    if cache.exists():
        for path in sorted(cache.rglob("*")):
            if path.is_file() and path.suffix in {".o", ".ll", ".mlir", ".log"}:
                print(f"  ARTIFACT path={path} size={path.stat().st_size} sha256={_sha256(path)}")


def test_chunk_bwd_dqkwg_standalone() -> None:
    device = _setup_npu()
    print("DIAGNOSTIC_CASE kernel=chunk_bwd_dqkwg dtype=bf16 B=2 T=128 H=2 HV=2 K=64 V=64 grid=2x2x4 repeats=5")
    results = []
    for repeat in range(5):
        print(f"DIAGNOSTIC_REPEAT={repeat}")
        results.append(run_one("bf16", device))
    _print_identity_and_artifacts()
    print(f"DQKWG_DIAGNOSTIC_RESULTS={results}")
    assert all(all(bad == 0 for bad in result) for result in results), results


def main() -> None:
    test_chunk_bwd_dqkwg_standalone()
    print("RESULT: PASS")


if __name__ == "__main__":
    main()

