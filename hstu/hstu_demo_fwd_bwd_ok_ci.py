# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# !/usr/bin/env python3


# commit 97a671d18e8dfd20b99ceeea9cab29e3ea3304ac (HEAD -> feature_bishengir, origin/feature_bishengir)
# # Merge: 737d7f8c7449 86ce82aa93cc
# # Author: chenxiangting 00624253 <chenxiangting@huawei.com>
# # Date:   Thu May 8 22:58:02 2025 -0400


from typing import List, Optional, Tuple

import pytest
import torch
import torch_npu
import triton
import triton.language as tl

import numpy as np
import torch.nn.functional as F


import os

import shutil
import unittest

import random
import sysconfig
import os
import sys
import torch.nn.functional as F
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
# torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libhstu_dense_ops.so")
ENABLE_DATACACHE = True


DEVICE = "npu"


BLOCK_M = 16
BLOCK_N = 16

autotune_max_seq_len = None
prev_power_of_2 = None
switch_to_contiguous_if_needed = None
triton_autotune = None


@triton.jit
def _hstu_attn_fwd_one_block( 
        start_n,
        seq_len,
        offs_m,
        offs_n,
        q,
        K_block_ptr,
        V_block_ptr,
        n_targets,
        alpha,
        MAX_SEQ_LEN,
        contextual_seq_len,
        max_attn_len,
        CAUSAL: tl.constexpr,
        HAS_MULTIPLE_TARGETS: tl.constexpr,
        HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
        HAS_MAX_ATTN_LEN: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    # -- compute qk ----

    k = tl.load(K_block_ptr)

    qk = tl.dot(q, tl.trans(k)) * alpha


    invalid_mask = offs_m[:, None] == offs_n[None, :]
    max_ids = seq_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        offs_m = offs_m - contextual_seq_len + 1
        offs_m = tl.where(
            offs_m > 0,
            offs_m,
            0,
        )
        offs_n = offs_n - contextual_seq_len + 1
        offs_n = tl.where(
            offs_n > 0,
            offs_n,
            0,
        )
        max_ids = max_ids - contextual_seq_len + 1
    if HAS_MULTIPLE_TARGETS:
        max_ids = max_ids - n_targets
        offs_m = tl.where(
            offs_m < max_ids,
            offs_m,
            max_ids,
        )
        offs_n = tl.where(
            offs_n < max_ids,
            offs_n,
            max_ids,
        )
    offs_m_minus_n = offs_m[:, None] - offs_n[None, :]
    silu = qk / (1.0 + tl.exp(-qk)) * (1.0 / MAX_SEQ_LEN)

    v = tl.load(V_block_ptr)

    silu = silu.to(v.dtype)

    return tl.dot(silu, v)


@triton.jit
def _hstu_attn_fwd_compute(  # noqa C901
        Q,
        K,
        V,
        seq_offsets,
        num_targets,
        Out,
        stride_qm: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_kn: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_vn: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        alpha,
        MAX_SEQ_LEN,
        DeltaSize,
        contextual_seq_len,
        max_attn_len,
        off_z,
        off_h,
        pid,
        CAUSAL: tl.constexpr,
        HAS_MULTIPLE_TARGETS: tl.constexpr,
        IS_DELTA_Q: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_D_Q: tl.constexpr,
        BLOCK_D_V: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
        HAS_MAX_ATTN_LEN: tl.constexpr,
):
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    off_h = off_h.to(tl.int64)
    off_z = off_z.to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1).to(tl.int64)
    seq_len = (seq_end - seq_start).to(tl.int32)
    if IS_DELTA_Q:
        start_m_delta = pid * BLOCK_M
        start_m = (start_m_delta + seq_len - DeltaSize).to(tl.int32)
    else:
        start_m_delta = 0
        start_m = pid * BLOCK_M

    if start_m < seq_len:
        if HAS_MULTIPLE_TARGETS:
            n_targets = tl.load(num_targets + off_z).to(tl.int32)
        else:
            n_targets = None

        # initialize offsets
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        if IS_DELTA_Q:
            Q_block_ptr = tl.make_block_ptr(
                base=Q + off_h * stride_qh + off_z * DeltaSize * stride_qm,
                shape=(DeltaSize, BLOCK_D_Q),
                strides=(stride_qm, 1),
                offsets=(start_m_delta, 0),
                block_shape=(BLOCK_M, BLOCK_D_Q),
                order=(1, 0),
            )
        else:
            Q_block_ptr = tl.make_block_ptr(
                base=Q + off_h * stride_qh + seq_start * stride_qm,
                shape=(seq_len, BLOCK_D_Q),
                strides=(stride_qm, 1),
                offsets=(start_m, 0),
                block_shape=(BLOCK_M, BLOCK_D_Q),
                order=(1, 0),
            )

        K_block_ptr = tl.make_block_ptr(
            base=K + off_h * stride_kh + seq_start * stride_kn,
            shape=(seq_len, BLOCK_D_Q),
            strides=(stride_kn, 1),
            offsets=(0, 0),
            block_shape=( BLOCK_N, BLOCK_D_Q),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + off_h * stride_vh + seq_start * stride_vn,
            shape=(seq_len, BLOCK_D_V),
            strides=(stride_vn, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_D_V),
            order=(1, 0),
        )
        q = tl.load(Q_block_ptr)
        acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)
        if CAUSAL:
            if HAS_MULTIPLE_TARGETS:
                uih_end = seq_len - n_targets
            else:
                uih_end = seq_len
            if HAS_CONTEXTUAL_SEQ_LEN is True and start_m < contextual_seq_len:
                # uih_end must be larger than start_m
                low = 0
                high = seq_len
            else:
                low = 0
                high = seq_len
                if HAS_MAX_ATTN_LEN:
                    low = start_m - max_attn_len
                    low = low if low > 0 else 0
                else:
                    low = 0

                    if HAS_CONTEXTUAL_SEQ_LEN:

                        low = low if low > contextual_seq_len else 0
                    else:

                        low = low if low > 0 else 0
                if HAS_MULTIPLE_TARGETS:
                    uih_end = (uih_end + BLOCK_N - 1) // BLOCK_N * BLOCK_N
                    if uih_end < start_m:
                        high = seq_len - n_targets
        else:
            low = 0
            high = seq_len

        if low > 0:

            K_block_ptr = tl.advance(K_block_ptr, (low, 0))
            V_block_ptr = tl.advance(V_block_ptr, (low, 0))

        end_n = low

        for start_n in range(low, high, BLOCK_N):

            acc += _hstu_attn_fwd_one_block(
                start_n=start_n,
                seq_len=seq_len,
                offs_m=offs_m,
                offs_n=offs_n + start_n,
                q=q,
                K_block_ptr=K_block_ptr,
                V_block_ptr=V_block_ptr,
                n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                alpha=alpha,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                contextual_seq_len=contextual_seq_len,
                max_attn_len=max_attn_len,
                CAUSAL=CAUSAL,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_N=BLOCK_N,
            )
            K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
            end_n += BLOCK_N

        if HAS_MULTIPLE_TARGETS and CAUSAL:
            # pyre-ignore[61]
            if uih_end < start_m:
                low_delta = start_m
                high_delta = start_m + BLOCK_M
                offset = (low_delta - end_n).to(tl.int32)
                K_block_ptr = tl.advance(K_block_ptr, (offset, 0))
                V_block_ptr = tl.advance(V_block_ptr, (offset, 0))
                for start_delta in tl.range(
                        low_delta, high_delta, BLOCK_N, num_stages=0
                ):
                    acc += _hstu_attn_fwd_one_block(
                        start_n=start_delta,
                        seq_len=seq_len,
                        offs_m=offs_m,
                        offs_n=offs_n + start_delta,
                        q=q,
                        K_block_ptr=K_block_ptr,
                        V_block_ptr=V_block_ptr,
                        n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                        alpha=alpha,
                        MAX_SEQ_LEN=MAX_SEQ_LEN,
                        contextual_seq_len=contextual_seq_len,
                        max_attn_len=max_attn_len,
                        CAUSAL=CAUSAL,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                        ALLOW_TF32=ALLOW_TF32,
                        BLOCK_N=BLOCK_N,
                    )
                    K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
                    V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

        if IS_DELTA_Q:
            start_m_delta = pid * BLOCK_M
            offs_m_delta = start_m_delta + tl.arange(0, BLOCK_M)
            offs_v_d = tl.arange(0, BLOCK_D_V)
            off_o = Out + off_z * DeltaSize * stride_om + off_h * stride_oh
            out_ptrs = off_o + offs_m_delta[:, None] * stride_om + offs_v_d[None, :]
            tl.store(out_ptrs, acc, mask=(offs_m_delta < DeltaSize)[:, None])
        else:
            # rematerialize offsets to save registers
            start_m = pid * BLOCK_M
            offs_m = start_m + tl.arange(0, BLOCK_M)
            offs_v_d = tl.arange(0, BLOCK_D_V)
            off_o = Out + seq_start * stride_om + off_h * stride_oh
            out_ptrs = off_o + offs_m[:, None] * stride_om + offs_v_d[None, :]
            tl.store(out_ptrs, acc, mask=(offs_m < seq_len)[:, None])




@triton.jit
def _hstu_attn_fwd(  # noqa C901
        Q,
        K,
        V,
        sort_by_length_indices,
        seq_offsets,
        num_targets,
        Out,
        stride_qm,
        stride_qh,
        stride_kn,
        stride_kh,
        stride_vn,
        stride_vh,
        stride_om,
        stride_oh,
        alpha,
        Z,
        AUTOTUNE_Z,
        H,
        MAX_SEQ_LEN,
        AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
        DimQ,
        DimV,
        DeltaSize,
        contextual_seq_len,
        max_attn_len,
        CAUSAL: tl.constexpr,
        HAS_MULTIPLE_TARGETS: tl.constexpr,
        IS_DELTA_Q: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_D_Q: tl.constexpr,
        BLOCK_D_V: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
        HAS_MAX_ATTN_LEN: tl.constexpr,
        HAS_SORT_BY_LENGTH_INDICES: tl.constexpr,
):
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)
    off_h = off_hz % H
    pid = tl.program_id(0)
    _hstu_attn_fwd_compute(
        Q=Q,
        K=K,
        V=V,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        Out=Out,
        stride_qm=stride_qm,
        stride_qh=stride_qh,
        stride_kn=stride_kn,
        stride_kh=stride_kh,
        stride_vn=stride_vn,
        stride_vh=stride_vh,
        stride_om=stride_om,
        stride_oh=stride_oh,
        alpha=alpha,
        MAX_SEQ_LEN=MAX_SEQ_LEN,
        DeltaSize=DeltaSize,
        contextual_seq_len=contextual_seq_len,
        max_attn_len=max_attn_len,
        off_z=off_z,
        off_h=off_h,
        pid=pid,
        CAUSAL=CAUSAL,
        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
        IS_DELTA_Q=IS_DELTA_Q,
        ALLOW_TF32=ALLOW_TF32,
        BLOCK_D_Q=BLOCK_D_Q,
        BLOCK_D_V=BLOCK_D_V,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
    )

@triton.jit
def _hstu_attn_bwd_one_block(  # noqa C901
        start_m,
        offs_n,
        offs_m,
        q_ptrs_trans,
        dq_ptrs_trans,
        mask_n,
        do_ptrs,
        dk,
        dv,
        k,
        v,
        pos_offs_n,
        seq_len,
        n_targets,
        max_ids,
        contextual_seq_len,
        max_attn_len,
        LOCK,
        stride_qm,
        stride_dom,
        stride_dqm,
        alpha,
        MAX_SEQ_LEN,
        CAUSAL: tl.constexpr,
        HAS_MULTIPLE_TARGETS: tl.constexpr,
        HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
        HAS_MAX_ATTN_LEN: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        ATOMIC_ADD: tl.constexpr,
):
    pos_offs_m = offs_m + start_m
    mask_m = pos_offs_m < seq_len
    invalid_mask_trans = pos_offs_m[None, :] == offs_n[:, None]
    # recompute qk and silu
    if HAS_CONTEXTUAL_SEQ_LEN:
        pos_offs_m = pos_offs_m - contextual_seq_len + 1
        pos_offs_m = tl.where(
            pos_offs_m > 0,
            pos_offs_m,
            0,
        )
    if HAS_MULTIPLE_TARGETS:
        pos_offs_m = tl.where(
            pos_offs_m < max_ids,
            pos_offs_m,
            max_ids,
        )
    q_ = tl.load(
        q_ptrs_trans + start_m * stride_qm,
    )
    q_trans = tl.trans(q_)
    qk_trans = tl.dot(k, q_trans) * alpha
    sig_trans = 1.0 / (1.0 + tl.exp(-qk_trans))
    silu_trans = qk_trans * sig_trans * (1.0 / MAX_SEQ_LEN)
    pos_offs_m_minus_n = pos_offs_m[None, :] - pos_offs_n[:, None]

    silu_trans = silu_trans.to(k.dtype)
    # compute dv
    do = tl.load(
        do_ptrs + start_m * stride_dom,
        mask=mask_m[:, None],
        other=0.0,
    )
    dv += tl.dot(silu_trans, do)
    dqk_trans = tl.dot(v, tl.trans(do))
    dqk_trans = (
            dqk_trans * sig_trans * (1 + qk_trans * (1 - sig_trans)) * (1.0 / MAX_SEQ_LEN)
    )
    # dqk_trans = dqk_trans.to(k.dtype)
    q_trans = q_trans.to(dqk_trans.dtype)

    # Note: the factor `alpha` is delayed until the end of the function to reduce the cost
    acc_dk = tl.dot(dqk_trans, tl.trans(q_trans))
    dqk_trans = dqk_trans.to(k.dtype)
    dk += acc_dk
    if ATOMIC_ADD:
        lock_id = start_m // BLOCK_M
        stride_lock = tl.cdiv(MAX_SEQ_LEN, BLOCK_M)
        lock = LOCK + tl.program_id(0) * stride_lock + lock_id
        tl.debug_barrier()  # add a barrier to force sync
        while tl.atomic_cas(lock, 0, 1) == 1:
            pass

    dq_trans = tl.load(
        dq_ptrs_trans + start_m * stride_dqm,
        mask=mask_m[:, None],
        other=0.0,
        eviction_policy="evict_first",
    )

    dq_acc = tl.dot(tl.trans(dqk_trans), k)

    dq_trans += dq_acc * alpha  ## todo:

    dq_trans = dq_trans.to(k.dtype)

    tl.store(
        dq_ptrs_trans + start_m * stride_dqm,
        dq_trans,
        mask=mask_m[:, None],
        eviction_policy="evict_first",
    )
    if ATOMIC_ADD:
        tl.atomic_xchg(lock, 0)  # pyre-ignore [61]
    return dk, dv


@triton.jit
def _hstu_attn_bwd_one_col_block(  # noqa C901
        start_n,
        seq_len,
        n_targets,
        contextual_seq_len,
        max_attn_len,
        Q,
        K,
        V,
        DOut,
        DQ,
        DK,
        DV,
        LOCK,
        stride_qm,
        stride_kn,
        stride_vn,
        stride_dom,
        stride_dqm,
        stride_dkn,
        stride_dvn,
        alpha,
        MAX_SEQ_LEN,
        CAUSAL: tl.constexpr,
        HAS_MULTIPLE_TARGETS: tl.constexpr,
        HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
        HAS_MAX_ATTN_LEN: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_D_Q: tl.constexpr,
        BLOCK_D_V: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        UNROLL: tl.constexpr,
        ATOMIC_ADD: tl.constexpr,
):
    # Work on the subsequence dv[start_n, start_n + BLOCK_N, :]
    if CAUSAL:
        if HAS_MULTIPLE_TARGETS:
            low = start_n
            if HAS_MAX_ATTN_LEN:
                high = start_n + max_attn_len + BLOCK_N
                high = high if high + n_targets < seq_len else seq_len
            else:
                high = seq_len
        else:
            low = start_n
            if HAS_MAX_ATTN_LEN:
                high = start_n + max_attn_len + BLOCK_N
                high = high if high < seq_len else seq_len
            else:
                high = seq_len
        if HAS_CONTEXTUAL_SEQ_LEN:
            contextual_block_end = tl.cdiv(contextual_seq_len, BLOCK_M) * BLOCK_M
            if low < contextual_block_end:
                low = contextual_block_end
    else:
        low = 0
        high = start_n + BLOCK_N
    low = 0
    high = seq_len

    # initialize row/col offsets
    offs_m = tl.arange(0, BLOCK_M)
    offs_qk_d = tl.arange(0, BLOCK_D_Q)
    offs_v_d = tl.arange(0, BLOCK_D_V)
    offs_n = start_n + tl.arange(0, BLOCK_N)

    # initialize pointers to value-like data

    q_ptrs_trans = Q + (offs_m[:, None] * stride_qm + offs_qk_d[None, :])
    dq_ptrs_trans = DQ + (offs_m[:, None] * stride_dqm + offs_qk_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_qk_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_v_d[None, :])
    mask_n = offs_n < seq_len

    do_ptrs = DOut + (offs_m[:, None] * stride_dom + offs_v_d[None, :])
    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
    # k and v stay in SRAM throughout
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    max_ids = seq_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        pos_offs_n = offs_n - contextual_seq_len + 1
        pos_offs_n = tl.where(
            pos_offs_n > 0,
            pos_offs_n,
            0,
        )
        max_ids = max_ids - contextual_seq_len + 1
    else:
        pos_offs_n = offs_n
    if HAS_MULTIPLE_TARGETS:
        max_ids = max_ids - n_targets
        pos_offs_n = tl.where(
            pos_offs_n < max_ids,
            pos_offs_n,
            max_ids,
        )
    # loop over rows
    if HAS_CONTEXTUAL_SEQ_LEN and CAUSAL:
        for start_m in range(0, contextual_seq_len, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            dk, dv = _hstu_attn_bwd_one_block(
                start_m=start_m,
                offs_n=offs_n,
                offs_m=offs_m,
                q_ptrs_trans=q_ptrs_trans,
                dq_ptrs_trans=dq_ptrs_trans,
                mask_n=mask_n,
                do_ptrs=do_ptrs,
                dk=dk,
                dv=dv,
                k=k,
                v=v,
                pos_offs_n=pos_offs_n,
                seq_len=seq_len,
                n_targets=n_targets,
                max_ids=max_ids,
                contextual_seq_len=contextual_seq_len,
                max_attn_len=max_attn_len,
                LOCK=LOCK,
                stride_qm=stride_qm,
                stride_dom=stride_dom,
                stride_dqm=stride_dqm,
                alpha=alpha,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                CAUSAL=CAUSAL,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                ATOMIC_ADD=ATOMIC_ADD,
            )
    for start_m in tl.range(low, high, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        dk, dv = _hstu_attn_bwd_one_block(
            start_m=start_m,
            offs_n=offs_n,
            offs_m=offs_m,
            q_ptrs_trans=q_ptrs_trans,
            dq_ptrs_trans=dq_ptrs_trans,
            mask_n=mask_n,
            do_ptrs=do_ptrs,
            dk=dk,
            dv=dv,
            k=k,
            v=v,
            pos_offs_n=pos_offs_n,
            seq_len=seq_len,
            n_targets=n_targets,
            max_ids=max_ids,
            contextual_seq_len=contextual_seq_len,
            max_attn_len=max_attn_len,
            LOCK=LOCK,
            stride_qm=stride_qm,
            stride_dom=stride_dom,
            stride_dqm=stride_dqm,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            CAUSAL=CAUSAL,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
            ALLOW_TF32=ALLOW_TF32,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            ATOMIC_ADD=ATOMIC_ADD,
        )
    # write-back
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_qk_d[None, :])
    dk = dk * alpha
    tl.store(dv_ptrs, dv.to(k.dtype), mask=mask_n[:, None])
    tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])


@triton.jit
def _hstu_attn_bwd(  
        Q,
        K,
        V,
        sort_by_length_indices,
        seq_offsets,
        num_targets,
        DOut,
        DQ,
        DK,
        DV,
        LOCK,
        stride_qm,
        stride_qh,
        stride_kn,
        stride_kh,
        stride_vn,
        stride_vh,
        stride_dom,
        stride_doh,
        stride_dqm,
        stride_dqh,
        stride_dkn,
        stride_dkh,
        stride_dvn,
        stride_dvh,
        alpha,
        contextual_seq_len,
        max_attn_len,
        Z,
        AUTOTUNE_Z,
        H,
        MAX_SEQ_LEN,
        AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
        DimQ,
        DimV,
        CAUSAL: tl.constexpr,
        HAS_MULTIPLE_TARGETS: tl.constexpr,
        HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
        HAS_MAX_ATTN_LEN: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_D_Q: tl.constexpr,
        BLOCK_D_V: tl.constexpr,
        SEQUENCE_PARALLEL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        UNROLL: tl.constexpr,
        HAS_SORT_BY_LENGTH_INDICES: tl.constexpr,
):
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)

    off_h = off_h.to(tl.int64)
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    if HAS_MULTIPLE_TARGETS:
        n_targets = tl.load(num_targets + off_z).to(tl.int32)
    else:
        n_targets = None
    # offset pointers for batch/head
    Q = Q + seq_start * stride_qm + off_h * stride_qh
    K = K + seq_start * stride_kn + off_h * stride_kh
    V = V + seq_start * stride_vn + off_h * stride_vh
    DOut = DOut + seq_start * stride_dom + off_h * stride_doh
    DQ = DQ + seq_start * stride_dqm + off_h * stride_dqh
    DK = DK + seq_start * stride_dkn + off_h * stride_dkh
    DV = DV + seq_start * stride_dvn + off_h * stride_dvh
    ## zhangfeng  tmp  start_n = pid * BLOCK_N
    if SEQUENCE_PARALLEL: ## zhangfeng tmp
        start_n = tl.program_id(1) * BLOCK_N
        if start_n >= seq_len:
            return
        _hstu_attn_bwd_one_col_block(
            start_n=start_n,
            seq_len=seq_len,
            n_targets=n_targets,
            contextual_seq_len=contextual_seq_len,
            max_attn_len=max_attn_len,
            Q=Q,
            K=K,
            V=V,
            DOut=DOut,
            DQ=DQ,
            DK=DK,
            DV=DV,
            LOCK=LOCK,
            stride_qm=stride_qm,
            stride_kn=stride_kn,
            stride_vn=stride_vn,
            stride_dom=stride_dom,
            stride_dqm=stride_dqm,
            stride_dkn=stride_dkn,
            stride_dvn=stride_dvn,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            CAUSAL=CAUSAL,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
            ALLOW_TF32=ALLOW_TF32,
            BLOCK_D_Q=BLOCK_D_Q,
            BLOCK_D_V=BLOCK_D_V,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            UNROLL=UNROLL,
            ATOMIC_ADD=False,
        )
    else:
        for start_n in range(0, seq_len, BLOCK_N):
            _hstu_attn_bwd_one_col_block(
                start_n=start_n,
                seq_len=seq_len,
                n_targets=n_targets,
                contextual_seq_len=contextual_seq_len,
                max_attn_len=max_attn_len,
                Q=Q,
                K=K,
                V=V,
                DOut=DOut,
                DQ=DQ,
                DK=DK,
                DV=DV,
                LOCK=LOCK,
                stride_qm=stride_qm,
                stride_kn=stride_kn,
                stride_vn=stride_vn,
                stride_dom=stride_dom,
                stride_dqm=stride_dqm,
                stride_dkn=stride_dkn,
                stride_dvn=stride_dvn,
                alpha=alpha,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                CAUSAL=CAUSAL,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_D_Q=BLOCK_D_Q,
                BLOCK_D_V=BLOCK_D_V,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                UNROLL=UNROLL,
                ATOMIC_ADD=False,
            )


def triton_hstu_attention_fwd(
        N: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets: torch.Tensor,
        causal: bool,
        num_targets: Optional[torch.Tensor],
        max_attn_len: int,
        contextual_seq_len: int,
        sort_by_length_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    Z = seq_offsets.numel() - 1
    AUTOTUNE_Z = 7
    L, H, DimQ = q.shape
    _, _, DimV = v.shape
    out = torch.empty_like(v)
    has_multiple_targets = num_targets is not None
    has_contextual_seq_len = contextual_seq_len > 0
    has_max_attn_len = max_attn_len > 0
    has_sort_by_length_indices = sort_by_length_indices is not None
    if L == 0:
        return out

    grid = lambda meta: (  # noqa E731
        triton.cdiv(N, meta["BLOCK_M"]),
        Z * H,
    )

    _hstu_attn_fwd[grid](
        Q=q,
        K=k,
        V=v,
        sort_by_length_indices=sort_by_length_indices,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        Out=out,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_om=out.stride(0),
        stride_oh=out.stride(1),
        alpha=alpha,
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        MAX_SEQ_LEN=N,
        AUTOTUNE_MAX_SEQ_LEN=N,
        DimQ=DimQ,
        DimV=DimV,
        DeltaSize=0,
        contextual_seq_len=contextual_seq_len,
        max_attn_len=max_attn_len,
        CAUSAL=causal,
        HAS_MULTIPLE_TARGETS=has_multiple_targets,
        IS_DELTA_Q=False,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HAS_CONTEXTUAL_SEQ_LEN=has_contextual_seq_len,
        HAS_MAX_ATTN_LEN=has_max_attn_len,
        HAS_SORT_BY_LENGTH_INDICES=has_sort_by_length_indices,
        enable_mixed_cv=True,
        enable_auto_bind_sub_block=True,
    )
    return out


def triton_hstu_attention_bwd(
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dq: torch.Tensor,
        dk: torch.Tensor,
        dv: torch.Tensor,
        seq_offsets: torch.Tensor,
        num_targets: Optional[torch.Tensor],
        N: int,
        alpha: float,
        max_attn_len: int,
        causal: float,
        contextual_seq_len: int,
        sort_by_length_indices: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dout.shape[0] == 0:
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    Z = seq_offsets.numel() - 1
    L, H, DimQ = q.shape
    _, _, DimV = v.shape
   
    # The minimum size of BLOCK_M used in `_get_bw_configs`.
    # TODO (linjianma): avoid hardcoding the value.
    MIN_BLOCK_M = 16
    lock = torch.empty(
        (Z * H, triton.cdiv(N, MIN_BLOCK_M)),
        dtype=torch.int32,
        device=q.device,
    )

    has_multiple_targets = num_targets is not None
    has_contextual_seq_len = contextual_seq_len > 0
    has_max_attn_len = max_attn_len > 0
    has_sort_by_length_indices = sort_by_length_indices is not None
    if L == 0:
        return out

    grid = (
        1,
        Z * H,
    )
    AUTOTUNE_Z = 7

    _hstu_attn_bwd[grid](
        Q=q,
        K=k,
        V=v,
        sort_by_length_indices=sort_by_length_indices,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        DOut=dout,
        DQ=dq,
        DK=dk,
        DV=dv,
        LOCK=lock,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_dom=dout.stride(0),
        stride_doh=dout.stride(1),
        stride_dqm=dq.stride(0),
        stride_dqh=dq.stride(1),
        stride_dkn=dk.stride(0),
        stride_dkh=dk.stride(1),
        stride_dvn=dv.stride(0),
        stride_dvh=dv.stride(1),
        alpha=alpha,
        contextual_seq_len=contextual_seq_len,
        max_attn_len=max_attn_len,
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        MAX_SEQ_LEN=N,
        AUTOTUNE_MAX_SEQ_LEN=N,  # autotune_max_seq_len(N),
        DimQ=DimQ,
        DimV=DimV,
        CAUSAL=causal,
        HAS_MULTIPLE_TARGETS=num_targets is not None,
        HAS_CONTEXTUAL_SEQ_LEN=has_contextual_seq_len,
        HAS_MAX_ATTN_LEN=has_max_attn_len,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        SEQUENCE_PARALLEL=0,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        UNROLL=1,
        HAS_SORT_BY_LENGTH_INDICES=has_sort_by_length_indices,
        enable_mixed_cv=True,
        enable_auto_bind_sub_block=True,
    )
    return dq, dk, dv


class _AttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx,
            N: int,
            alpha: float,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            seq_offsets: torch.Tensor,
            causal: bool,
            num_targets: Optional[torch.Tensor],
            max_attn_len: int,
            contextual_seq_len: int,
            sort_by_length: bool,
    ) -> torch.Tensor:
        sort_by_length_indices = None
        if sort_by_length:
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            _, sort_by_length_indices = torch.sort(
                seq_lengths, descending=True, stable=False
            )
        saved_tensors = [q, k, v, seq_offsets]
        if num_targets is not None:
            saved_tensors.append(num_targets)
        if sort_by_length_indices is not None:
            saved_tensors.append(sort_by_length_indices)
        ctx.save_for_backward(*saved_tensors)
        ctx.alpha = alpha
        ctx.causal = causal
        ctx.has_multiple_targets = num_targets is not None
        ctx.max_attn_len = max_attn_len
        ctx.N = N
        ctx.contextual_seq_len = contextual_seq_len
        ctx.sort_by_length = sort_by_length
        return triton_hstu_attention_fwd(
            N=N,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=causal,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            sort_by_length_indices=sort_by_length_indices,
        )

    @staticmethod
    def backward(
            ctx, dout: torch.Tensor
    ) -> Tuple[
        None,
        None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        with torch.inference_mode():
            q, k, v, seq_offsets = ctx.saved_tensors[:4]
            idx = 4
            if ctx.has_multiple_targets:
                num_targets = ctx.saved_tensors[idx]
                idx += 1
            else:
                num_targets = None
            if ctx.sort_by_length:
                sort_by_length_indices = ctx.saved_tensors[idx]
            else:
                sort_by_length_indices = None

            dq = torch.zeros_like(q).to(torch.float32).requires_grad_()
            # dq = torch.zeros(q.shape, device="npu", dtype=torch.float32).requires_grad_()
            dk = torch.zeros_like(k).requires_grad_()
            dv = torch.zeros_like(v).requires_grad_()
            dq, dk, dv = triton_hstu_attention_bwd(
                dout=dout,
                q=q,
                k=k,
                v=v,
                dq=dq,
                dk=dk,
                dv=dv,
                seq_offsets=seq_offsets,
                num_targets=num_targets,
                N=ctx.N,
                alpha=ctx.alpha,
                max_attn_len=ctx.max_attn_len,
                causal=ctx.causal,
                contextual_seq_len=ctx.contextual_seq_len,
                sort_by_length_indices=sort_by_length_indices,
            )
            return None, None, dq, dk, dv,  None, None, None, None, None, None

def triton_hstu_mha(
        N: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets: torch.Tensor,
        causal: bool,
        num_targets: Optional[torch.Tensor] = None,
        max_attn_len: int = 0,
        contextual_seq_len: int = 0,
        sort_by_length: bool = False,
) -> torch.Tensor:
    return _AttentionFunction.apply(
        N,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        causal,
        num_targets,
        max_attn_len,
        contextual_seq_len,
        False,
    )


def triton_hstu_bwd(
        N: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dout: torch.Tensor,
        seq_offsets: torch.Tensor,
        causal: bool,
        num_targets: Optional[torch.Tensor] = None,
        max_attn_len: int = 0,
        contextual_seq_len: int = 0,
        sort_by_length: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
]:
    out = _AttentionFunction.apply(
        N,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        causal,
        num_targets,
        max_attn_len,
        contextual_seq_len,
        sort_by_length,
    )


    assert dout.is_contiguous()
    out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    return out , tri_dq.reshape(-1), tri_dk.reshape(-1), tri_dv.reshape(-1)

import torch.nn.functional as F

def jagged_data_gen(batch_size, max_seq_len, num_heads, attention_dim, dataType):
    ## todo: add maskType later
    seq_lens = np.random.randint(max_seq_len, max_seq_len + 1, (batch_size))

    seq_offset = torch.concat((torch.zeros((1,), dtype=torch.int64), \
                               torch.cumsum(torch.as_tensor(seq_lens), axis=0))).to(torch.int64).numpy()

    max_seq_len = np.max(seq_lens)
    total_seqs = np.sum(seq_lens)

    q = torch.rand((int(total_seqs), num_heads, attention_dim), dtype=dataType, device=DEVICE).requires_grad_()
    k = torch.rand((int(total_seqs), num_heads, attention_dim), dtype=dataType, device=DEVICE).requires_grad_()
    v = torch.rand((int(total_seqs), num_heads, attention_dim), dtype=dataType, device=DEVICE).requires_grad_()

    # q = torch.ones((int(total_seqs), num_heads, attention_dim), dtype=dataType, device=DEVICE).requires_grad_()
    # k = torch.ones((int(total_seqs), num_heads, attention_dim), dtype=dataType, device=DEVICE).requires_grad_()
    # v = torch.ones((int(total_seqs), num_heads, attention_dim), dtype=dataType, device=DEVICE).requires_grad_()


    rel_attn_bias = torch.zeros(batch_size, num_heads, int(max_seq_len),int(max_seq_len)).to(dataType)
    for batch_id in range(batch_size):
       seq_len = seq_lens[batch_id]
       rel_attn_bias[batch_id, :, 0:seq_len, 0:seq_len] = torch.rand(seq_len, seq_len).to(dataType)

    return q, k, v, seq_offset, rel_attn_bias,  max_seq_len

def dense_to_jagged(q, dense_tensor, seq_lens):
  tensor = torch.zeros_like(q)
  offset = 0
  for batch_id, seq_len in enumerate(seq_lens):
      tensor[offset : offset + seq_len, :, :] = dense_tensor[batch_id, 0: seq_len, :, :]
      offset = offset + seq_len
  return tensor


def jagged_to_dense(jagged_tensor, seq_lens, head_nums, atten_dim):
    need_pad_seq = []
    offset = 0
    for batch_id, seq_len in enumerate(seq_lens):
        src_tensor = jagged_tensor[offset: offset + seq_len, :, :].reshape(seq_len, head_nums, atten_dim)
        need_pad_seq.append(src_tensor)
        offset = offset + seq_len

    dense_tensor = torch.nn.utils.rnn.pad_sequence(need_pad_seq, batch_first=True)
    return dense_tensor

def gloden_op_exec(q, k, v, alpha, seq_offset, bias, max_seq_len, enableBias, maskType, siluScale, dataType):
    head_nums = q.shape[1]
    head_dim = q.shape[2]
    batch_size = bias.shape[0]
    seq_lens = np.zeros((batch_size, )).astype(np.int64)
    for batch_id in range(batch_size):
        seq_lens[batch_id] = seq_offset[batch_id + 1] - seq_offset[batch_id]
    siluScale = 1 / max_seq_len if siluScale == 0 else siluScale
    q_dens = jagged_to_dense(q, seq_lens, head_nums, head_dim).to(dataType)
    k_dens = jagged_to_dense(k, seq_lens, head_nums, head_dim).to(dataType)
    v_dens = jagged_to_dense(v, seq_lens, head_nums, head_dim).to(dataType)
    q_dens = q_dens.permute(0, 2, 1, 3)
    k_dens = k_dens.permute(0, 2, 3, 1)
    qk_attn = torch.matmul(q_dens, k_dens) * alpha
    qk_attn = qk_attn.to(torch.float32)

    silu = F.silu(qk_attn) * siluScale

    v_dens = v_dens.permute(0, 2, 1, 3)
    silu = silu.to(dataType)
    atten_output = torch.matmul(silu, v_dens)
    atten_output = atten_output.permute(0, 2, 1, 3).cpu()
    atten_output = dense_to_jagged(q, atten_output, seq_lens)
    torch.npu.synchronize()
    return atten_output.to(dataType).reshape(-1)


def golden_op_exec_bwd(grad, q, k, v, bias, mask, max_seq_len, seq_offset, mask_type, silu_scale, enable_bias, data_type):
    def jagged_to_dense(jagged_tensor, seq_lens, max_seq_len, head_num, head_dim):
        batch_size = len(seq_lens)
        dense_tensor = torch.zeros(batch_size, max_seq_len, head_num, head_dim, dtype=jagged_tensor.dtype)

        offset = 0
        for batch_id, seq_len in enumerate(seq_lens):
            dense_tensor[batch_id, :seq_len, :, :] = jagged_tensor[offset: offset + seq_len, :, :]
            offset = offset + seq_len

        return dense_tensor

    def dense_to_jagged(jagged_tensor, dense_tensor, seq_lens):
        tensor = torch.zeros_like(jagged_tensor)

        offset = 0
        for batch_id, seq_len in enumerate(seq_lens):
            tensor[offset: offset + seq_len, :, :] = dense_tensor[batch_id, 0: seq_len, :, :]
            offset = offset + seq_len

        return tensor

    head_nums = grad.shape[1]
    head_dim = grad.shape[2]
    batch_size = bias.shape[0]
    seq_lens = np.zeros((batch_size,)).astype(np.int64)
    for batch_id in range(batch_size):
        seq_lens[batch_id] = seq_offset[batch_id + 1] - seq_offset[batch_id]
    grad_dens = jagged_to_dense(grad, seq_lens, max_seq_len, head_nums, head_dim).to(data_type)
    q_dens = jagged_to_dense(q, seq_lens, max_seq_len, head_nums, head_dim).to(data_type)
    k_dens = jagged_to_dense(k, seq_lens, max_seq_len, head_nums, head_dim).to(data_type)
    v_dens = jagged_to_dense(v, seq_lens, max_seq_len, head_nums, head_dim).to(data_type)
    actual_seq_lens = torch.from_numpy(seq_lens).reshape(batch_size, 1, 1, 1).to(data_type)
    actual_seq_lens = torch.broadcast_to(actual_seq_lens, bias.shape)
    qk = torch.matmul(q_dens.permute(0, 2, 1, 3), k_dens.permute(0, 2, 3, 1))
    gv = torch.matmul(grad_dens.permute(0, 2, 1, 3), v_dens.permute(0, 2, 3, 1))
    qk = qk.float()
    gv = gv.float()
    bias = bias.float()
    if mask_type == 0 or mask_type == 3:
        mask = mask.to(dataType)
        mask = mask.float()
    if enable_bias:
        bias = bias.to(dataType)
        bias = bias.float()
        qkb = qk + bias
    else:
        qkb = qk
    real_silu_scale = 1 / max_seq_len if silu_scale == 0.0 else silu_scale

    if mask_type == 0 or mask_type == 3:
        score = F.silu(qkb) * real_silu_scale * mask
    else:
        score = F.silu(qkb) * real_silu_scale
    score = score.to(data_type)
    v_grad_dens = torch.matmul(score.permute(0, 1, 3, 2), grad_dens.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    if mask_type == 0 or mask_type == 3:
        bias_grad = gv * real_silu_scale * mask * F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))
    else:
        bias_grad = gv * real_silu_scale * F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))

    bias_grad = bias_grad.to(data_type)
    k_grad_dens = torch.matmul(bias_grad.permute(0, 1, 3, 2), q_dens.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    q_grad_dens = torch.matmul(bias_grad, k_dens.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    bias_grad = bias_grad.cpu()
    q_grad_dens = q_grad_dens.cpu()
    q_grad = dense_to_jagged(q, q_grad_dens, seq_lens)
    k_grad_dens = k_grad_dens.cpu()
    k_grad = dense_to_jagged(k, k_grad_dens, seq_lens)
    v_grad_dens = v_grad_dens.cpu()
    v_grad = dense_to_jagged(v, v_grad_dens, seq_lens)
    torch.npu.synchronize()
    return q_grad, k_grad, v_grad, bias_grad


def run_test_op(attention_dim, num_heads, batch_size, max_len, dtype_str):
    type_mapper = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
    dtype = type_mapper.get(dtype_str)
    alpha = 1.0 / (attention_dim ** 0.5)
    q, k, v, seq_offset, bias, max_seq_len = jagged_data_gen(batch_size = batch_size, max_seq_len = max_len, num_heads = num_heads, attention_dim = attention_dim, dataType = dtype)


    seq_offsets = torch.tensor(seq_offset, dtype=torch.int64, device=DEVICE)

    atten_output = gloden_op_exec(q, k, v, alpha, seq_offset, bias, max_seq_len, enableBias=False, maskType=0, siluScale=0, dataType=dtype)
    grad = torch.ones_like(atten_output.reshape(-1, num_heads, attention_dim)).to(
        dtype)  ## max_seq_len, num_heads, attention_dim
    # q_grad_golden, k_grad_golden, v_grad_golden, attn_bias_grad_golden = golden_op_exec_bwd(grad , q, k, v, bias, None, max_seq_len, seq_offset, mask_type=2, silu_scale=0, enable_bias=False, data_type=dtype)
    q_grad_golden, k_grad_golden, v_grad_golden, attn_bias_grad_golden = torch.ops.mxrec.hstu_jagged_backward(
            grad,
            q,
            k,
            v,
            bias,
            None,
            mask_type=2,
            max_seq_len=max_seq_len,
            max_seq_len_k=max_seq_len,
            silu_scale=1.0 / max_seq_len,
            seq_offset=seq_offsets,
            seq_offset_k=seq_offsets,
            num_context=None,
            num_target=None,
            target_group_size=None,
            alpha=alpha,
        )

    print("pytorch 前向输出", flush=True)
    print(atten_output, flush=True)
    print("pytorch 后向输出: ", flush=True)
    print("q_grad_golden.shape ", q_grad_golden.shape, flush=True)
    print(q_grad_golden, flush=True)

    out, tri_dq, tri_dk, tri_dv = triton_hstu_bwd(
        N=int(max_seq_len),
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        dout=grad,
        seq_offsets=seq_offsets,
        causal=False
    )
    print("triton 前向输出: ", flush=True)
    print(out, flush=True)
    print("triton 后向输出: ", flush=True)
    print(tri_dq, flush=True)

    ATOL, RTOL = 1e-3, 1e-3
    if dtype == torch.bfloat16:
        ATOL, RTOL = 1e-2, 1e-2
    assert torch.allclose(out.reshape(-1), atten_output, ATOL, RTOL)
    assert torch.allclose(tri_dv, v_grad_golden.reshape(-1), ATOL, RTOL)
    assert torch.allclose(tri_dk, k_grad_golden.reshape(-1), ATOL, RTOL)
    assert torch.allclose(tri_dq, q_grad_golden.reshape(-1), ATOL, RTOL)

@pytest.mark.parametrize('list_4', [
                  [16, 1, 1, 16],
                  [16, 1, 1, 32],
                  [128, 4, 32, 64],
                  [128, 8, 8, 8000],
                  [256, 8, 8, 8000],
                  [256, 2, 2048, 32],
                  ])
@pytest.mark.parametrize("dtype_str", ["fp16", "bf16", "fp8"]) # TODO: fp8暂不支持
def test_op(list_4, dtype_str):
    attention_dim, num_heads, batch_size, max_len = list_4
    run_test_op(attention_dim, num_heads, batch_size, max_len, dtype_str)

# CI看护用例入口（QK等长）
@pytest.mark.parametrize('list_ci', [
    # ============== torch.float16 ================
    ["fp16", 16, 1, 1, 16],
    ["fp16", 16, 1, 1, 32],
    ["fp16", 128, 4, 32, 64],
    ["fp16", 128, 8, 8, 8000],
    ["fp16", 256, 8, 8, 8000],
    ["fp16", 256, 2, 2048, 32],
    # ============== torch.bfloat16 ================
    ["bf16", 16, 1, 1, 16],
    ["bf16", 16, 1, 1, 32],
    ["bf16", 128, 4, 32, 64],
    ["bf16", 128, 8, 8, 8000],
    ["bf16", 256, 8, 8, 8000],
    ["bf16", 256, 2, 2048, 32],
])
def test_op_ci(list_ci):
    dtype_str, attention_dim, num_heads, batch_size, max_len = list_ci
    run_test_op(attention_dim, num_heads, batch_size, max_len, dtype_str)
