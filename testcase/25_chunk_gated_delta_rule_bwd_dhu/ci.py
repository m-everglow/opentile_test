"""Standalone NPU chunk_gated_delta_rule_bwd_dhu kernel + CPU golden.

[realhw_run_20260727] _npu 变体 (V_OFFSET/NH_OFFSET 分块 launch) + dh0 输出修复.

Run directly:
  python3 test_chunk_gated_delta_rule_bwd_dhu_standalone.py

Run with pytest:
  pytest -s --assert=plain test_chunk_gated_delta_rule_bwd_dhu_standalone.py
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
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/opentile_mr330_rework_dhu_cache")

import torch
import triton
import triton.language as tl
import torch.nn.functional as F


# ============================================================
# NPU kernel (standard FLA, no _npu variant)
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
def chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64_npu(
    q,
    k,
    w,
    g,
    gk,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    V_OFFSET: tl.constexpr,
    NH_OFFSET: tl.constexpr,
):
    i_v = tl.program_id(0) + V_OFFSET
    i_nh = tl.program_id(1) + NH_OFFSET
    i_n, i_h = i_nh // HV, i_nh % HV
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    if STATE_V_FIRST:
        b_dh1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    q += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    do += (bos * HV + i_h).to(tl.int64) * V
    dv += (bos * HV + i_h).to(tl.int64) * V
    dv2 += (bos * HV + i_h).to(tl.int64) * V
    dh += (boh * HV + i_h).to(tl.int64) * K*V
    if USE_GK:
        gk += (bos * HV + i_h).to(tl.int64) * K

    if USE_INITIAL_STATE:
        dh0 += i_nh * K*V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K*V

    if USE_FINAL_STATE_GRADIENT:
        if STATE_V_FIRST:
            p_dht1 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_dht1 = tl.make_block_ptr(dht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_dh1 += tl.load(p_dht1, boundary_check=(0, 1))
        if K > 64:
            if STATE_V_FIRST:
                p_dht2 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_dht2 = tl.make_block_ptr(dht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
        if K > 128:
            if STATE_V_FIRST:
                p_dht3 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_dht3 = tl.make_block_ptr(dht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
        if K > 192:
            if STATE_V_FIRST:
                p_dht4 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_dht4 = tl.make_block_ptr(dht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))

    for i_t in range(NT - 1, -1, -1):
        i_t_int64 = i_t.to(tl.int64)
        if STATE_V_FIRST:
            p_dh1 = tl.make_block_ptr(dh + i_t_int64*HV*K*V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_dh1 = tl.make_block_ptr(dh + i_t_int64*HV*K*V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if STATE_V_FIRST:
                p_dh2 = tl.make_block_ptr(dh + i_t_int64*HV*K*V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_dh2 = tl.make_block_ptr(dh + i_t_int64*HV*K*V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if STATE_V_FIRST:
                p_dh3 = tl.make_block_ptr(dh + i_t_int64*HV*K*V, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_dh3 = tl.make_block_ptr(dh + i_t_int64*HV*K*V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if STATE_V_FIRST:
                p_dh4 = tl.make_block_ptr(dh + i_t_int64*HV*K*V, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_dh4 = tl.make_block_ptr(dh + i_t_int64*HV*K*V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            bg_last = tl.load(g + (bos + last_idx) * HV + i_h).to(tl.float32)
            p_g = tl.make_block_ptr(g + bos * HV + i_h, (T,), (HV,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            bg_last_exp = exp2(bg_last)
            b_g_exp = exp2(b_g)
        p_dv = tl.make_block_ptr(dv, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv2 = tl.make_block_ptr(dv2, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        b_do = tl.load(p_do, boundary_check=(0, 1))

        # Update dv
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + last_idx * HV*K + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
        if STATE_V_FIRST:
            b_dv = tl.dot(b_k, tl.trans(b_dh1).to(b_k.dtype))
        else:
            b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

        if K > 64:
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + last_idx * HV*K + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
            if STATE_V_FIRST:
                b_dv += tl.dot(b_k, tl.trans(b_dh2).to(b_k.dtype))
            else:
                b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

        if K > 128:
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + last_idx * HV*K + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
            if STATE_V_FIRST:
                b_dv += tl.dot(b_k, tl.trans(b_dh3).to(b_k.dtype))
            else:
                b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

        if K > 192:
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + last_idx * HV*K + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
            if STATE_V_FIRST:
                b_dv += tl.dot(b_k, tl.trans(b_dh4).to(b_k.dtype))
            else:
                b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
        b_dv += tl.load(p_dv, boundary_check=(0, 1))

        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
        # Update dh
        p_w = tl.make_block_ptr(w, (K, T), (1, HV*K), (0, i_t * BT), (64, BT), (0, 1))
        p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (0, i_t * BT), (64, BT), (0, 1))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        if USE_G:
            b_dh1 *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]
        if USE_GK:
            if STATE_V_FIRST:
                b_dh1 *= exp2(b_gk_last1)[None, :]
            else:
                b_dh1 *= exp2(b_gk_last1[:, None])
        if STATE_V_FIRST:
            b_dh1 += tl.trans(tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype)))
        else:
            b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 64:
            p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (64, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, HV*K), (64, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh2 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if STATE_V_FIRST:
                    b_dh2 *= exp2(b_gk_last2)[None, :]
                else:
                    b_dh2 *= exp2(b_gk_last2[:, None])
            if STATE_V_FIRST:
                b_dh2 += tl.trans(tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype)))
            else:
                b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 128:
            p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (128, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, HV*K), (128, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh3 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if STATE_V_FIRST:
                    b_dh3 *= exp2(b_gk_last3)[None, :]
                else:
                    b_dh3 *= exp2(b_gk_last3[:, None])
            if STATE_V_FIRST:
                b_dh3 += tl.trans(tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype)))
            else:
                b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 192:
            p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (192, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, HV*K), (192, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh4 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if STATE_V_FIRST:
                    b_dh4 *= exp2(b_gk_last4)[None, :]
                else:
                    b_dh4 *= exp2(b_gk_last4[:, None])
            if STATE_V_FIRST:
                b_dh4 += tl.trans(tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype)))
            else:
                b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            p_dh0 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if STATE_V_FIRST:
                p_dh1 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_dh1 = tl.make_block_ptr(dh0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if STATE_V_FIRST:
                p_dh2 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_dh2 = tl.make_block_ptr(dh0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if STATE_V_FIRST:
                p_dh3 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_dh3 = tl.make_block_ptr(dh0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    if state_v_first:
        h = k.new_empty(B, NT, HV, V, K)
        final_state = k.new_zeros(N, HV, V, K, dtype=torch.float32) if output_final_state else None
    else:
        h = k.new_empty(B, NT, HV, K, V)
        final_state = k.new_zeros(N, HV, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u) if save_new_value else None
    def grid(meta): return (triton.cdiv(V, meta['BV']), N*HV)
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
    )
    return h, v_new, final_state


ASCEND_MAX_GRID_DIM = 65535


def max_grid_axis_chunks(
    axis_size: int,
    other_grid_product: int,
    *,
    max_grid: int = ASCEND_MAX_GRID_DIM,
) -> int:
    """Max launch chunks along one grid axis while keeping the product <= max_grid."""
    return max(1, max_grid // max(other_grid_product, 1))


def _launch_fwd_h_kernel(kernel, *, nv_chunks: int, nh_total: int, kernel_kwargs: dict) -> None:
    max_nv = max_grid_axis_chunks(nv_chunks, nh_total, max_grid=ASCEND_MAX_GRID_DIM)
    for v_off in range(0, nv_chunks, max_nv):
        v_len = min(max_nv, nv_chunks - v_off)
        kernel_kwargs['V_OFFSET'] = v_off
        max_nh = max_grid_axis_chunks(nh_total, v_len, max_grid=ASCEND_MAX_GRID_DIM)
        for nh_off in range(0, nh_total, max_nh):
            nh_len = min(max_nh, nh_total - nh_off)
            kernel_kwargs['NH_OFFSET'] = nh_off
            kernel[(v_len, nh_len)](**kernel_kwargs)


def chunk_gated_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HV = *q.shape, do.shape[-1], do.shape[2]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    BT = chunk_size
    assert K <= 256, "current kernel does not support head dimension being larger than 256."

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    if state_v_first:
        dh = q.new_empty(B, NT, HV, V, K)
    else:
        dh = q.new_empty(B, NT, HV, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)

    BV = min(64, triton.next_power_of_2(V))
    nv_chunks = triton.cdiv(V, BV)
    _launch_fwd_h_kernel(
        chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64_npu,
        nv_chunks=nv_chunks,
        nh_total=N * HV,
        kernel_kwargs={
            'q': q, 'k': k, 'w': w, 'g': g, 'gk': gk,
            'dht': dht, 'dh0': dh0, 'do': do, 'dh': dh, 'dv': dv, 'dv2': dv2,
            'cu_seqlens': cu_seqlens, 'chunk_offsets': chunk_offsets,
            'scale': scale, 'T': T,
            'H': H, 'HV': HV, 'K': K, 'V': V, 'BT': BT, 'BV': BV,
            'USE_G': g is not None,
            'USE_GK': gk is not None,
            'USE_INITIAL_STATE': h0 is not None,
            'USE_FINAL_STATE_GRADIENT': dht is not None,
            'STATE_V_FIRST': state_v_first,
            'IS_VARLEN': cu_seqlens is not None,
            'V_OFFSET': 0,
            'NH_OFFSET': 0,
        },
    )
    return dh, dh0, dv2


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



SEED = 440045
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


def _cpu_golden_dhu(q, k, w, do, dv, g, dht, scale):
    """CPU golden for chunk_gated_delta_rule_bwd_dhu."""
    b, t, h, d = q.shape
    hv = do.shape[2]
    nt = t // BT
    dh = torch.zeros((b, nt, hv, d, d), dtype=torch.float32)
    dv2 = torch.zeros((b, t, hv, d), dtype=torch.float32)
    b_dh = dht.clone()
    for it in range(nt - 1, -1, -1):
        s, e = it * BT, (it + 1) * BT
        dh[:, it] = b_dh
        qc = q[:, s:e].float()
        kc = k[:, s:e].float()
        wc = w[:, s:e].float()
        doc = do[:, s:e].float()
        dvc = dv[:, s:e].float()
        gc = g[:, s:e]
        gl = gc[:, -1]
        kdh = torch.einsum('bthk,bhkv->bthv', kc, b_dh)
        bdv = kdh * torch.exp2(gl[:, None, :, None] - gc[:, :, :, None]) + dvc
        dv2[:, s:e] = bdv
        qg = qc * torch.exp2(gc)[:, :, :, None]
        b_dh = b_dh * torch.exp2(gl)[:, :, None, None]
        b_dh = b_dh + torch.einsum('bthk,bthv->bhkv', qg, doc) * scale
        b_dh = b_dh - torch.einsum('bthk,bthv->bhkv', wc, bdv)
    return dh, b_dh, dv2


def run_one(dtype_name: str, device: torch.device) -> None:
    torch_dtype = DTYPES[dtype_name]
    atol, rtol = TOLERANCES[dtype_name]
    gen = torch.Generator().manual_seed(SEED)

    def rand_bf16(shape, scale=0.03):
        return (torch.randn(shape, generator=gen) * scale).to(torch_dtype)

    q = rand_bf16((B, T, H, D))
    k = rand_bf16((B, T, H, D))
    w = rand_bf16((B, T, HV, D))
    do = rand_bf16((B, T, HV, D))
    dv = rand_bf16((B, T, HV, D))
    g = _make_gate(gen, B, T, HV)
    dht = (torch.randn((B, HV, D, D), generator=gen) * 0.03).to(torch.float32)

    ref_dh, ref_dh0, ref_dv2 = _cpu_golden_dhu(q, k, w, do, dv, g, dht, SCALE)

    q_d = q.to(device)
    k_d = k.to(device)
    w_d = w.to(device)
    do_d = do.to(device)
    dv_d = dv.to(device)
    g_d = g.to(device)
    dht_d = dht.to(device)

    # dh0 输出需要 USE_INITIAL_STATE=True 才分配/写回; kernel 只写 dh0 不读 h0,
    # 传任意 h0 张量即可获得 dh0 输出 (数值与 h0 内容无关, 见 golden 注释).
    h0_d = torch.empty((B, HV, D, D), dtype=torch.float32, device=device)
    dh, dh0, dv2 = chunk_gated_delta_rule_bwd_dhu(
        q=q_d, k=k_d, w=w_d, do=do_d, dv=dv_d,
        g=g_d, dht=dht_d, scale=SCALE, h0=h0_d,
    )
    torch.npu.synchronize()

    dh_bad = _compare(f"dh_{dtype_name}", dh, ref_dh.to(dh.dtype), atol, rtol)
    dh0_bad = _compare(f"dh0_{dtype_name}", dh0, ref_dh0, atol, rtol)
    dv2_bad = _compare(f"dv2_{dtype_name}", dv2, ref_dv2.to(dv2.dtype), atol, rtol)

    del q_d, k_d, w_d, do_d, dv_d, g_d, dht_d, dh, dh0, dv2
    torch.npu.empty_cache()
    return dh_bad, dh0_bad, dv2_bad


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


def test_chunk_gated_delta_rule_bwd_dhu_standalone() -> None:
    device = _setup_npu()
    print("DIAGNOSTIC_CASE kernel=chunk_gated_delta_rule_bwd_dhu dtype=bf16 B=2 T=128 H=2 HV=2 K=64 V=64 grid=2x4x1 repeats=5")
    results = []
    for repeat in range(5):
        print(f"DIAGNOSTIC_REPEAT={repeat}")
        results.append(run_one("bf16", device))
    _print_identity_and_artifacts()
    print(f"DHU_DIAGNOSTIC_RESULTS={results}")
    assert all(all(bad == 0 for bad in result) for result in results), results


def main() -> None:
    test_chunk_gated_delta_rule_bwd_dhu_standalone()
    print("RESULT: PASS")


if __name__ == "__main__":
    main()

