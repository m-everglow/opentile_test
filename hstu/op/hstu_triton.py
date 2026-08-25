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
# import torch_npu
import triton
import triton.language as tl

import numpy as np
import torch.nn.functional as F


# DEVICE = "npu"


BLOCK_M = 16
BLOCK_N = 16

autotune_max_seq_len = None
prev_power_of_2 = None
switch_to_contiguous_if_needed = None
triton_autotune = None


@triton.jit
def _hstu_attn_fwd_one_block( 
        start_n,
        seq_len_q,
        seq_len_k,
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
    start_n = tl.multiple_of(start_n, BLOCK_N)  # hint:让编译器知道start_n是BLOCK_N的倍数
    # -- compute qk ----

    k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
    qk = tl.dot(q, tl.trans(k)) * alpha

    invalid_mask = offs_m[:, None] == offs_n[None, :] 

    max_ids = seq_len_q
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

    v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

    silu = silu.to(v.dtype)

    return tl.dot(silu, v)


@triton.jit
def _hstu_attn_fwd_compute(  # noqa C901
        Q,
        K,
        V,
        seq_offsets_q,
        seq_offsets_k,
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
        off_h1,
        off_h2,
        pid,
        CAUSAL: tl.constexpr,
        HAS_MULTIPLE_TARGETS: tl.constexpr,
        IS_DELTA_Q: tl.constexpr,  # false
        ALLOW_TF32: tl.constexpr,
        BLOCK_D_Q: tl.constexpr,
        BLOCK_D_V: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
        HAS_MAX_ATTN_LEN: tl.constexpr,
):

    seq_start_q = tl.load(seq_offsets_q + off_z).to(tl.int64)
    off_h1 = off_h1.to(tl.int64)
    off_h2 = off_h2.to(tl.int64)
    off_z = off_z.to(tl.int64)
    seq_end_q = tl.load(seq_offsets_q + off_z + 1).to(tl.int64)
    seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)  # 计算当前batch的seq_len

    seq_start_k = tl.load(seq_offsets_k + off_z).to(tl.int64)
    seq_end_k = tl.load(seq_offsets_k + off_z + 1).to(tl.int64)
    seq_len_k = (seq_end_k - seq_start_k).to(tl.int32)  # 计算当前batch的seq_len
    if IS_DELTA_Q:
        start_m_delta = pid * BLOCK_M
        start_m = (start_m_delta + seq_len_q - DeltaSize).to(tl.int32)
    else:
        start_m_delta = 0
        start_m = pid * BLOCK_M

    if start_m < seq_len_q:
        if HAS_MULTIPLE_TARGETS:
            n_targets = tl.load(num_targets + off_z).to(tl.int32)
        else:
            n_targets = None

        # initialize offsets
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        if IS_DELTA_Q:
            Q_block_ptr = tl.make_block_ptr(
                base=Q + off_h1 * stride_qh + off_z * DeltaSize * stride_qm,
                shape=(DeltaSize, BLOCK_D_Q),
                strides=(stride_qm, 1),
                offsets=(start_m_delta, 0),
                block_shape=(BLOCK_M, BLOCK_D_Q),
                order=(1, 0),
            )
        else:
            Q_block_ptr = tl.make_block_ptr(
                base=Q + off_h1 * stride_qh + seq_start_q * stride_qm,
                shape=(seq_len_q, BLOCK_D_Q),
                strides=(stride_qm, 1),
                offsets=(start_m, 0),
                block_shape=(BLOCK_M, BLOCK_D_Q),
                order=(1, 0),
            )

        K_block_ptr = tl.make_block_ptr(
            base=K + off_h2 * stride_kh + seq_start_k * stride_kn,
            shape=(seq_len_k, BLOCK_D_Q),
            strides=(stride_kn, 1),
            offsets=(0, 0),
            block_shape=( BLOCK_N, BLOCK_D_Q),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + off_h2 * stride_vh + seq_start_k * stride_vn,
            shape=(seq_len_k, BLOCK_D_V),
            strides=(stride_vn, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_D_V),
            order=(1, 0),
        )

        q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
        acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)
        if CAUSAL:  # seq_len替换成seq_len_q还是seq_len_k存疑
            if HAS_MULTIPLE_TARGETS:
                uih_end = seq_len_k - n_targets
            else:
                uih_end = seq_len_k
            if HAS_CONTEXTUAL_SEQ_LEN is True and start_m < contextual_seq_len:
                # uih_end must be larger than start_m
                low = 0
                high = seq_len_k
            else:
                low = 0
                high = seq_len_k
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
                        high = seq_len_k - n_targets
        else:
            low = 0
            high = seq_len_k

        if low > 0:

            K_block_ptr = tl.advance(K_block_ptr, (low, 0))
            V_block_ptr = tl.advance(V_block_ptr, (low, 0))

        end_n = low
        for start_n in range(low, high, BLOCK_N):

            acc += _hstu_attn_fwd_one_block(
                start_n=start_n,
                seq_len_q=seq_len_q,
                seq_len_k=seq_len_k,
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
                        seq_len_q=seq_len_q,
                        seq_len_k=seq_len_k,
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
            off_o = Out + off_z * DeltaSize * stride_om + off_h1 * stride_oh
            out_ptrs = off_o + offs_m_delta[:, None] * stride_om + offs_v_d[None, :]
            tl.store(out_ptrs, acc, mask=(offs_m_delta < DeltaSize)[:, None])
        else:
            # rematerialize offsets to save registers
            start_m = pid * BLOCK_M
            offs_m = start_m + tl.arange(0, BLOCK_M)
            offs_v_d = tl.arange(0, BLOCK_D_V)
            off_o = Out + seq_start_q * stride_om + off_h1 * stride_oh
            out_ptrs = off_o + offs_m[:, None] * stride_om + offs_v_d[None, :]
            tl.store(out_ptrs, acc, mask=(offs_m < seq_len_q)[:, None])

@triton.jit
def _hstu_attn_fwd(  # noqa C901
        Q,
        K,
        V,
        sort_by_length_indices,
        seq_offsets_q,
        seq_offsets_k,
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
        Z1,
        Z2,
        AUTOTUNE_Z,
        H1,
        GroupSize,
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
    off_z = off_hz // H1  # 除H1还是H2
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)
    off_h1 = off_hz % H1
    off_h2 = off_h1 // GroupSize
    pid = tl.program_id(0)
    _hstu_attn_fwd_compute(
        Q=Q,
        K=K,
        V=V,
        seq_offsets_q=seq_offsets_q,
        seq_offsets_k=seq_offsets_k,
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
        off_h1=off_h1,
        off_h2=off_h2,
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
        seq_len_q,
        seq_len_k,
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
    mask_m = pos_offs_m < seq_len_q
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
        mask=mask_m[:, None],
        other=0.0,
    )
    # qk_dim * M
    # q_trans = tl.trans(q_).to(tl.float32)
    # k = k.to(tl.float32)
    # v = v.to(tl.float32)
    q_trans = tl.trans(q_)
    # N * qk_dim * qk_dim * M
    qk_trans = tl.dot(k, q_trans) * alpha
    sig_trans = 1.0 / (1.0 + tl.exp(-qk_trans))
    silu_trans = qk_trans * sig_trans * (1.0 / MAX_SEQ_LEN)
    silu_trans = silu_trans.to(k.dtype)
    pos_offs_m_minus_n = pos_offs_m[None, :] - pos_offs_n[:, None]

    # compute dv
    do = tl.load(
        do_ptrs + start_m * stride_dom,
        mask=mask_m[:, None],
        other=0.0,
    ) #.to(tl.float32)
    # N * M * M * v_dim
    acc_dv = tl.dot(silu_trans, do)
    # tl.device_print("acc_dv: ", acc_dv)
    dv += acc_dv
    # N * v_dim * v_dim * M
    dqk_trans = tl.dot(v, tl.trans(do))
    dqk_trans = (
            dqk_trans * sig_trans * (1 + qk_trans * (1 - sig_trans)) * (1.0 / MAX_SEQ_LEN)
    )
    # dqk_trans = dqk_trans.to(tl.float16)
    q_trans = q_trans.to(dqk_trans.dtype)

    # Note: the factor `alpha` is delayed until the end of the function to reduce the cost
    acc_dk = tl.dot(dqk_trans, tl.trans(q_trans))
    # tl.device_print("acc_dk: ", acc_dk)
    dqk_trans = dqk_trans.to(k.dtype)
    dk += acc_dk
    if ATOMIC_ADD:
        lock_id = start_m // BLOCK_M
        stride_lock = tl.cdiv(MAX_SEQ_LEN, BLOCK_M)
        lock = LOCK + tl.program_id(0) * stride_lock + lock_id
        tl.debug_barrier()  # add a barrier to force sync
        while tl.atomic_cas(lock, 0, 1) == 1:
            pass

    # dq_trans = tl.load(
    #    dq_ptrs_trans + start_m * stride_dqm,
    #    mask=mask_m[:, None],
    #    other=0.0,
    # )
    dq_acc = tl.dot(tl.trans(dqk_trans), k)
    # tl.device_print("dq_trans:", dq_trans)
    # tl.device_print("dqk_trans:", dqk_trans)
    dq_acc = dq_acc * alpha

    # dq_trans = dq_trans.to(k.dtype)
    # tl.static_print("dq_trans", dq_trans.shape)
    # tl.static_print("dq_acc", dq_acc.shape)
    tl.atomic_add(
        dq_ptrs_trans + start_m * stride_dqm,
        dq_acc,
        mask=mask_m[:, None],
    )
    if ATOMIC_ADD:
        tl.atomic_xchg(lock, 0)  # pyre-ignore [61]
    return dk, dv


@triton.jit
def _hstu_attn_bwd_one_col_block(  # noqa C901
        start_n,
        seq_len_q,
        seq_len_k,
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
                high = high if high + n_targets < seq_len_q else seq_len_q
            else:
                high = seq_len_q
        else:
            low = start_n
            if HAS_MAX_ATTN_LEN:
                high = start_n + max_attn_len + BLOCK_N
                high = high if high < seq_len_q else seq_len_q
            else:
                high = seq_len_q
        if HAS_CONTEXTUAL_SEQ_LEN:
            contextual_block_end = tl.cdiv(contextual_seq_len, BLOCK_M) * BLOCK_M
            if low < contextual_block_end:
                low = contextual_block_end
    else:
        low = 0
        high = start_n + BLOCK_N
    low = 0
    high = seq_len_q

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
    mask_n = offs_n < seq_len_k

    do_ptrs = DOut + (offs_m[:, None] * stride_dom + offs_v_d[None, :])
    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
    # k and v stay in SRAM throughout
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    max_ids = seq_len_q
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
                seq_len_q=seq_len_q,
                seq_len_k=seq_len_k,
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
            seq_len_q=seq_len_q,
            seq_len_k=seq_len_k,
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
    # dk_acc = tl.load(dk_ptrs, mask=mask_n[:, None])
    # dv_acc = tl.load(dv_ptrs, mask=mask_n[:, None])
    # dk_acc += dk
    # dv_acc += dv
    # tl.store(dv_ptrs, dv_acc.to(k.dtype), mask=mask_n[:, None])
    # tl.store(dk_ptrs, dk_acc.to(k.dtype), mask=mask_n[:, None])
    tl.atomic_add(dv_ptrs, dv.to(k.dtype), mask=mask_n[:, None])
    tl.atomic_add(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])


@triton.jit
def _hstu_attn_bwd(  
        Q,
        K,
        V,
        sort_by_length_indices,
        seq_offsets_q,
        seq_offsets_k,
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
        Z,  # 貌似没用
        AUTOTUNE_Z,
        H1,
        GroupSize,
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
    off_z = off_hz // H1
    off_h1 = off_hz % H1
    off_h2 = off_h1 // GroupSize
    tl.device_print("off_hz: ", off_hz)
    tl.device_print("off_z: ", off_z)
    tl.device_print("off_h1: ", off_h1)
    tl.device_print("off_h2: ", off_h2)
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)

    off_h1 = off_h1.to(tl.int64)
    off_h2 = off_h2.to(tl.int64)
    seq_start_q = tl.load(seq_offsets_q + off_z).to(tl.int64)
    seq_end_q = tl.load(seq_offsets_q + off_z + 1).to(tl.int64)
    seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)

    seq_start_k = tl.load(seq_offsets_k + off_z).to(tl.int64)
    seq_end_k = tl.load(seq_offsets_k + off_z + 1).to(tl.int64)
    seq_len_k = (seq_end_k - seq_start_k).to(tl.int32)
    if HAS_MULTIPLE_TARGETS:
        n_targets = tl.load(num_targets + off_z).to(tl.int32)
    else:
        n_targets = None
    # offset pointers for batch/head
    Q = Q + seq_start_q * stride_qm + off_h1 * stride_qh
    K = K + seq_start_k * stride_kn + off_h2 * stride_kh
    V = V + seq_start_k * stride_vn + off_h2 * stride_vh
    DOut = DOut + seq_start_q * stride_dom + off_h1 * stride_doh
    DQ = DQ + seq_start_q * stride_dqm + off_h1 * stride_dqh
    DK = DK + seq_start_k * stride_dkn + off_h2 * stride_dkh
    DV = DV + seq_start_k * stride_dvn + off_h2 * stride_dvh
    ## zhangfeng  tmp  start_n = pid * BLOCK_N
    if SEQUENCE_PARALLEL: ## zhangfeng tmp
        start_n = tl.program_id(1) * BLOCK_N
        if start_n >= seq_len_k:
            return
        _hstu_attn_bwd_one_col_block(
            start_n=start_n,
            seq_len_q=seq_len_q,
            seq_len_k=seq_len_k,
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
        for start_n in range(0, seq_len_k, BLOCK_N):
            _hstu_attn_bwd_one_col_block(
                start_n=start_n,
                seq_len_q=seq_len_q,
                seq_len_k=seq_len_k,
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
        seq_offsets_q: torch.Tensor,
        seq_offsets_k: torch.Tensor,
        causal: bool,
        num_targets: Optional[torch.Tensor],
        max_attn_len: int,
        contextual_seq_len: int,
        sort_by_length_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    Z1 = seq_offsets_q.numel() - 1  # S1
    Z2 = seq_offsets_k.numel() - 1  # S2
    AUTOTUNE_Z = 7
    L, H1, DimQ = q.shape  # N1
    _, H2, DimV = v.shape
    GroupSize = H1 // H2  # assert H1 % H2 == 0
    out = torch.empty((L, H1, DimV), dtype=q.dtype, layout=q.layout, device=q.device)
    has_multiple_targets = num_targets is not None
    has_contextual_seq_len = contextual_seq_len > 0
    has_max_attn_len = max_attn_len > 0
    has_sort_by_length_indices = sort_by_length_indices is not None
    if L == 0:
        return out

    grid = lambda meta: (  # noqa E731
        triton.cdiv(N, meta["BLOCK_M"]),
        Z1 * H1,
    )

    _hstu_attn_fwd[grid](
        Q=q,
        K=k,
        V=v,
        sort_by_length_indices=sort_by_length_indices,
        seq_offsets_q=seq_offsets_q,
        seq_offsets_k=seq_offsets_k,
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
        Z1=Z1,
        Z2=Z2,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H1=H1,
        GroupSize=GroupSize,
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
        seq_offsets_q: torch.Tensor,
        seq_offsets_k: torch.Tensor,
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
    Z = seq_offsets_q.numel() - 1  # B
    L, H1, DimQ = q.shape
    _, H2, DimV = v.shape
    GroupSize = H1 // H2  # assert H1 % H2 == 0
    # The minimum size of BLOCK_M used in `_get_bw_configs`.
    # TODO (linjianma): avoid hardcoding the value.
    MIN_BLOCK_M = 16
    lock = torch.empty(
        (Z * H1, triton.cdiv(N, MIN_BLOCK_M)),
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
        Z * H1,
        # 64
    )
    AUTOTUNE_Z = 7
    print(Z * H1)

    _hstu_attn_bwd[grid](
        Q=q,
        K=k,
        V=v,
        sort_by_length_indices=sort_by_length_indices,
        seq_offsets_q=seq_offsets_q,
        seq_offsets_k=seq_offsets_k,
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
        H1=H1,
        GroupSize=GroupSize,
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
            seq_offsets_q: torch.Tensor,
            seq_offsets_k: torch.Tensor,
            causal: bool,
            num_targets: Optional[torch.Tensor],
            max_attn_len: int,
            contextual_seq_len: int,
            sort_by_length: bool,
    ) -> torch.Tensor:
        sort_by_length_indices = None
        if sort_by_length:
            seq_offsets_q = seq_offsets_q[1:] - seq_offsets_q[:-1]
            _, sort_by_length_indices = torch.sort(
                seq_offsets_q, descending=True, stable=False
            )
        saved_tensors = [q, k, v, seq_offsets_q, seq_offsets_k]
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
        # print("sort_by_length_indices: ", sort_by_length_indices)
        return triton_hstu_attention_fwd(
            N=N,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets_q=seq_offsets_q,
            seq_offsets_k=seq_offsets_k,
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
        None,
    ]:
        with torch.inference_mode():
            q, k, v, seq_offsets_q, seq_offsets_k= ctx.saved_tensors[:5]
            idx = 5
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
            # dq = torch.zeros_like(q).requires_grad_()
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
                seq_offsets_q=seq_offsets_q,
                seq_offsets_k=seq_offsets_k,
                num_targets=num_targets,
                N=ctx.N,
                alpha=ctx.alpha,
                max_attn_len=ctx.max_attn_len,
                causal=ctx.causal,
                contextual_seq_len=ctx.contextual_seq_len,
                sort_by_length_indices=sort_by_length_indices,
            )
            return None, None, dq, dk, dv, None, None, None, None, None, None, None

def triton_hstu_mha(
        N: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets_q: torch.Tensor,
        seq_offsets_k: torch.Tensor,
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
        seq_offsets_q,
        seq_offsets_k,
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
        seq_offsets_q: torch.Tensor,
        seq_offsets_k: torch.Tensor,
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
        seq_offsets_q,
        seq_offsets_k,
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
    return out , tri_dq, tri_dk, tri_dv





