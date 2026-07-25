import os

import torch
import torch.utils.checkpoint
from torch.nn.attention.flex_attention import BlockMask, _create_sparse_block_from_block_mask
from typing import NamedTuple, Optional
import triton
import triton.language as tl


Q_BUILD_CHUNK = int(os.environ.get("Q_BUILD_CHUNK", "512"))
APPLY_Q_CHUNK = int(os.environ.get("APPLY_Q_CHUNK", "2048"))
USE_TRITON_BLOCK_SUMMARY = os.environ.get("Z_USE_TRITON_BLOCK_SUMMARY", "0") == "1"

TILE_BLOCK_SIZE = 128

_BITS = 32


def _get_num_aicore():
    npu_mod = getattr(torch, "npu", None)
    if npu_mod is None or not hasattr(npu_mod, "current_device"):
        return 1
    device = npu_mod.current_device()
    props = triton.runtime.driver.active.utils.get_device_properties(device)
    return max(int(props.get("num_aicore", 1)), 1)


def _persistent_launch_config(num_tasks):
    num_tasks = max(int(num_tasks), 1)
    return (min(_get_num_aicore(), num_tasks),), num_tasks


@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_lse_z", "stride_lse_h", "stride_kv_idx_m",
        "Q_LEN", "KV_LEN", "NUM_TASKS", "NUM_Q_BLOCKS",
        "stride_partial_p", "stride_partial_m",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_out_z", "stride_out_h",
    ]
)
def flex_attention_kernel(
    Q,
    K,
    V,
    KV_NUM_BLKS,
    KV_IDX,
    FULL_KV_NUM_BLKS,
    FULL_KV_IDX,
    DENSE_MASK,
    stride_mask_m,
    stride_mask_n,
    PARTIAL_MASK_PACKED,
    PARTIAL_MASK_OFFSETS,
    stride_partial_p,
    stride_partial_m,
    stride_partial_n,
    stride_partial_offset_m,
    OUT,
    LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_out_z, stride_out_h, stride_out_m, stride_out_k,
    stride_lse_z, stride_lse_h, stride_lse_m,
    stride_kv_idx_m,
    SM_SCALE,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_TASKS,
    NUM_Q_BLOCKS,
    Q_HEAD,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    USE_PACKED_PARTIAL_MASK: tl.constexpr = False,
):
    """
    Triton kernel for FlexAttention forward pass.

    Args:
        Q: Query tensor [Z, Hq, M, D]
        K: Key tensor [Z, Hkv, N, D]
        V: Value tensor [Z, Hkv, N, Dv]
        KV_NUM_BLKS: Number of KV blocks per Q block
        KV_IDX: Indices of KV blocks per Q block
        FULL_KV_NUM_BLKS: Number of fully unmasked KV blocks per Q block
        FULL_KV_IDX: Indices of fully unmasked KV blocks per Q block
        OUT: Output tensor [Z, Hq, M, Dv]
        LSE: Log-sum-exp output tensor [Z, Hq, M]
        SM_SCALE: Scale factor for softmax
    """
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    for task_id in range(pid, NUM_TASKS, num_core):
        q_start = task_id % NUM_Q_BLOCKS
        off_z = (task_id // NUM_Q_BLOCKS) // Q_HEAD
        off_hq = (task_id // NUM_Q_BLOCKS) % Q_HEAD
        off_hkv = off_hq // GQA_SHARED_HEADS

        off_z = off_z.to(tl.int64)
        off_hq = off_hq.to(tl.int64)
        off_hkv = off_hkv.to(tl.int64)

        q_offset = off_z * stride_qz + off_hq * stride_qh
        k_offset = off_z * stride_kz + off_hkv * stride_kh
        v_offset = off_z * stride_vz + off_hkv * stride_vh
        out_offset = off_z * stride_out_z + off_hq * stride_out_h
        lse_offset = off_z * stride_lse_z + off_hq * stride_lse_h

        Q_ptr = Q + q_offset
        K_ptr = K + k_offset
        V_ptr = V + v_offset
        OUT_ptr = OUT + out_offset
        LSE_ptr = LSE + lse_offset

        # m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, V_HEAD_DIM], dtype=tl.float32)

        offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, QK_HEAD_DIM)
        offs_v = tl.arange(0, V_HEAD_DIM)

        q = tl.load(
            Q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
            mask=(offs_m[:, None] < Q_LEN),     #   & (offs_k[None, :] < QK_HEAD_DIM),
            other=0.0
        )

        SPARSE_Q_MULTIPLE = SPARSE_Q_BLOCK_SIZE // BLOCK_M
        SPARSE_KV_MULTIPLE = SPARSE_KV_BLOCK_SIZE // BLOCK_N

        q_sparse_idx = q_start // SPARSE_Q_MULTIPLE
        sparse_kv_num_blks_offset = q_sparse_idx
        sparse_kv_idx_offset = q_sparse_idx * stride_kv_idx_m
        partial_mask_offset = tl.load(PARTIAL_MASK_OFFSETS + q_sparse_idx * stride_partial_offset_m)
        q_sparse_base = q_sparse_idx * SPARSE_Q_BLOCK_SIZE

        kv_indices = KV_IDX + sparse_kv_idx_offset
        kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)
        block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
        for start_n in range(0, block_n_end):
            blk_idx_in_list = start_n // SPARSE_KV_MULTIPLE
            kv_block = tl.load(kv_indices + blk_idx_in_list)
            kv_start = kv_block * SPARSE_KV_BLOCK_SIZE + (start_n % SPARSE_KV_MULTIPLE) * BLOCK_N

            offs_n_load = kv_start + tl.arange(0, BLOCK_N)
            if USE_PACKED_PARTIAL_MASK:
                partial_block_idx = partial_mask_offset + blk_idx_in_list
                offs_m_in_block = offs_m - q_sparse_base
                offs_n_in_block = (start_n % SPARSE_KV_MULTIPLE) * BLOCK_N + tl.arange(0, BLOCK_N)
                mask = load_packed_partial_mask(
                    PARTIAL_MASK_PACKED,
                    stride_partial_p,
                    stride_partial_m,
                    stride_partial_n,
                    partial_block_idx,
                    offs_m_in_block,
                    offs_n_in_block,
                    SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                    SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                )
            else:
                mask = load_dense_mask(
                    DENSE_MASK,
                    stride_mask_m,
                    stride_mask_n,
                    offs_m,
                    offs_n_load,
                    Q_LEN=Q_LEN,
                    KV_LEN=KV_LEN,
                )

            k = tl.load(
                K_ptr + offs_n_load[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                mask=(offs_n_load[:, None] < KV_LEN),   # & (offs_k[None, :] < QK_HEAD_DIM),
                other=0.0
            )
            v = tl.load(
                V_ptr + offs_n_load[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                mask=(offs_n_load[:, None] < KV_LEN),   # & (offs_v[None, :] < V_HEAD_DIM),
                other=0.0
            )
            k = tl.trans(k)

            qk = tl.dot(q, k, input_precision="ieee")
            qk *= SM_SCALE

            # & (offs_n_load[None, :] < KV_LEN)
            qk = tl.where(mask, qk, float("-inf"))
            # qk = tl.where(mask & (offs_n_load[None, :] < KV_LEN), qk, -1e6)

            m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
            masked_out_rows = (m_ij == float("-inf"))
            m_ij_masked = tl.where(masked_out_rows, 0, m_ij)

            alpha = tl.math.exp(m_i - m_ij_masked)
            p = tl.math.exp(qk - m_ij_masked[:, None])

            pv = tl.dot(p.to(Q.dtype.element_ty), v, input_precision="ieee")
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + pv
            m_i = m_ij

        if HAS_FULL_BLOCKS:
            kv_indices = FULL_KV_IDX + sparse_kv_idx_offset
            kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_kv_num_blks_offset)
            block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)

            for start_n in range(0, block_n_end):
                blk_idx_in_list = start_n // SPARSE_KV_MULTIPLE
                kv_block = tl.load(kv_indices + blk_idx_in_list)
                kv_start = kv_block * SPARSE_KV_BLOCK_SIZE + (start_n % SPARSE_KV_MULTIPLE) * BLOCK_N

                offs_n_load = kv_start + tl.arange(0, BLOCK_N)
                k = tl.load(
                    K_ptr + offs_n_load[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                    mask=(offs_n_load[:, None] < KV_LEN), # & (offs_k[None, :] < QK_HEAD_DIM),
                    other=0.0
                )
                v = tl.load(
                    V_ptr + offs_n_load[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                    mask=(offs_n_load[:, None] < KV_LEN), # & (offs_v[None, :] < V_HEAD_DIM),
                    other=0.0
                )
                k = tl.trans(k)

                qk = tl.dot(q, k, input_precision="ieee")
                qk *= SM_SCALE

                # qk = tl.where(offs_n_load[None, :] < KV_LEN, qk, float("-inf"))

                m_ij = tl.maximum(m_i, tl.max(qk, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
                # masked_out_rows = (m_ij == float("-inf"))
                # m_ij_masked = tl.where(masked_out_rows, 0, m_ij)

                # alpha = tl.math.exp(m_i - m_ij_masked)
                # p = tl.math.exp(qk - m_ij_masked[:, None])
                alpha = tl.math.exp(m_i - m_ij)
                p = tl.math.exp(qk - m_ij[:, None])

                pv = tl.dot(p.to(Q.dtype.element_ty), v, input_precision="ieee")
                l_i = l_i * alpha + tl.sum(p, 1)
                acc = acc * alpha[:, None] + pv
                m_i = m_ij
        l_i = tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_i[:, None]

        out_mask = (offs_m[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM)
        tl.store(
            OUT_ptr + offs_m[:, None] * stride_out_m + offs_v[None, :] * stride_out_k,
            acc,
            mask=out_mask
        )

        lse = m_i + tl.math.log(l_i)
        tl.store(LSE_ptr + offs_m * stride_lse_m, lse, mask=offs_m < Q_LEN)


@triton.jit
def load_dense_mask(
    DENSE_MASK,
    stride_mask_m,
    stride_mask_n,
    offs_m,
    offs_n,
    Q_LEN,
    KV_LEN,
):
    stride_mask_m = stride_mask_m.to(tl.int64)
    # stride_mask_n = stride_mask_n.to(tl.int64)
    ptrs = DENSE_MASK + offs_m[:, None] * stride_mask_m + offs_n[None, :] * stride_mask_n
    valid = (offs_m[:, None] < Q_LEN) & (offs_n[None, :] < KV_LEN)
    return tl.load(ptrs, mask=valid, other=0)


@triton.jit
def load_packed_partial_mask(
    PARTIAL_MASK_PACKED,
    stride_partial_p,
    stride_partial_m,
    stride_partial_n,
    partial_block_idx,
    offs_m_in_block,
    offs_n_in_block,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
):
    ptrs = (
        PARTIAL_MASK_PACKED
        + partial_block_idx * stride_partial_p
        + offs_m_in_block[:, None] * stride_partial_m
        + offs_n_in_block[None, :] * stride_partial_n
    )
    valid = (
        (offs_m_in_block[:, None] < SPARSE_Q_BLOCK_SIZE)
        & (offs_n_in_block[None, :] < SPARSE_KV_BLOCK_SIZE)
    )
    return tl.load(ptrs, mask=valid, other=0)


@triton.jit
def bwd_dq_block_mn(
    q, do, lse, delta,
    K_ptr, V_ptr,
    DENSE_MASK, stride_mask_m, stride_mask_n,
    PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
    PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
    Q_LEN, KV_LEN,
    offs_m, offs_n, offs_k, offs_v,
    q_sparse_idx, kv_block, kv_sub, q_sparse_base,
    stride_kn, stride_kk, stride_vn, stride_vk,
    MATMUL_PRECISION,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    SM_SCALE: tl.constexpr,
    IS_FULL_BLOCKS: tl.constexpr,
    USE_PACKED_PARTIAL_MASK: tl.constexpr,
):
    k = tl.load(
        K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
        mask=(offs_n[:, None] < KV_LEN),    # & (offs_k[None, :] < QK_HEAD_DIM)
        other=0.0,
    )
    v = tl.load(
        V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
        mask=(offs_n[:, None] < KV_LEN),    # & (offs_v[None, :] < V_HEAD_DIM)
        other=0.0,
    )

    qk = tl.dot(q, tl.trans(k), input_precision="ieee")
    qk *= SM_SCALE

    mask = True
    if not IS_FULL_BLOCKS:
        if USE_PACKED_PARTIAL_MASK:
            partial_block_idx = tl.load(
                PARTIAL_BLOCK_TABLE
                + q_sparse_idx * stride_partial_table_m
                + kv_block * stride_partial_table_n
            )
            safe_partial_block_idx = tl.maximum(partial_block_idx, 0)
            offs_m_in_block = offs_m - q_sparse_base
            offs_n_in_block = kv_sub * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = load_packed_partial_mask(
                PARTIAL_MASK_PACKED,
                stride_partial_p,
                stride_partial_m,
                stride_partial_n,
                safe_partial_block_idx,
                offs_m_in_block,
                offs_n_in_block,
                SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
            )
            mask = mask & (partial_block_idx >= 0)
        else:
            mask = load_dense_mask(
                DENSE_MASK,
                stride_mask_m,
                stride_mask_n,
                offs_m,
                offs_n,
                Q_LEN=Q_LEN,
                KV_LEN=KV_LEN,
            )
        qk = tl.where(mask & (offs_n[None, :] < KV_LEN), qk, float("-inf"))
    else:
        qk = tl.where(offs_n[None, :] < KV_LEN, qk, float("-inf"))

    p = tl.math.exp(qk - lse[:, None])
    dp = tl.dot(do, tl.trans(v), input_precision="ieee")
    ds = p * (dp - delta[:, None])

    # if not IS_FULL_BLOCKS:
    #     ds = tl.where(mask, ds, 0.0)

    dq = tl.dot(ds.to(MATMUL_PRECISION), k, input_precision="ieee")
    return dq


@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_partial_p", "stride_partial_m",
        "stride_partial_table_m",
        "stride_lse_z", "stride_lse_h", "stride_kv_idx_m",
        "Q_LEN", "KV_LEN", "NUM_TASKS", "NUM_Q_BLOCKS",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_doz", "stride_doh",
        "stride_delta_z", "stride_delta_h",
        "stride_dqz", "stride_dqh",
    ]
)
def flex_attention_backward_dq_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    KV_NUM_BLKS,
    KV_IDX,
    FULL_KV_NUM_BLKS,
    FULL_KV_IDX,
    DENSE_MASK,
    stride_mask_m,
    stride_mask_n,
    PARTIAL_MASK_PACKED,
    PARTIAL_MASK_OFFSETS,
    PARTIAL_BLOCK_TABLE,
    stride_partial_p,
    stride_partial_m,
    stride_partial_n,
    stride_partial_offset_m,
    stride_partial_table_m,
    stride_partial_table_n,
    DQ,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_lse_z, stride_lse_h, stride_lse_m,
    stride_delta_z, stride_delta_h, stride_delta_m,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_kv_idx_m,
    SM_SCALE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SUB_BLOCKS: tl.constexpr,
    NUM_TASKS,
    NUM_Q_BLOCKS,
    Q_HEAD,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS: tl.constexpr,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    USE_PACKED_PARTIAL_MASK: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)
    sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
    KV_BLOCK_SIZE: tl.constexpr = BLOCK_N * NUM_KV_SUB_BLOCKS
    sparse_kv_multiple = SPARSE_KV_BLOCK_SIZE // KV_BLOCK_SIZE
    MATMUL_PRECISION = Q.dtype.element_ty

    for task_id in range(pid, NUM_TASKS, num_core):
        q_start = task_id % NUM_Q_BLOCKS
        off_z = (task_id // NUM_Q_BLOCKS) // Q_HEAD
        off_hq = (task_id // NUM_Q_BLOCKS) % Q_HEAD
        off_hkv = off_hq // GQA_SHARED_HEADS

        off_z = off_z.to(tl.int64)
        off_hq = off_hq.to(tl.int64)
        off_hkv = off_hkv.to(tl.int64)

        q_offset = off_z * stride_qz + off_hq * stride_qh
        k_offset = off_z * stride_kz + off_hkv * stride_kh
        v_offset = off_z * stride_vz + off_hkv * stride_vh
        do_offset = off_z * stride_doz + off_hq * stride_doh
        lse_offset = off_z * stride_lse_z + off_hq * stride_lse_h
        delta_offset = off_z * stride_delta_z + off_hq * stride_delta_h
        dq_offset = off_z * stride_dqz + off_hq * stride_dqh

        Q_ptr = Q + q_offset
        K_ptr = K + k_offset
        V_ptr = V + v_offset
        DO_ptr = DO + do_offset
        LSE_ptr = LSE + lse_offset
        DELTA_ptr = DELTA + delta_offset
        DQ_ptr = DQ + dq_offset

        offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, QK_HEAD_DIM)
        offs_v = tl.arange(0, V_HEAD_DIM)

        q = tl.load(
            Q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
            mask=(offs_m[:, None] < Q_LEN),     # & (offs_k[None, :] < QK_HEAD_DIM)
            other=0.0,
        )
        do = tl.load(
            DO_ptr + offs_m[:, None] * stride_dom + offs_v[None, :] * stride_dok,
            mask=(offs_m[:, None] < Q_LEN),     # & (offs_v[None, :] < V_HEAD_DIM)
            other=0.0,
        )

        lse = tl.load(LSE_ptr + offs_m * stride_lse_m, mask=offs_m < Q_LEN, other=float("-inf"))
        delta = tl.load(DELTA_ptr + offs_m * stride_delta_m, mask=offs_m < Q_LEN, other=0.0)
        lse = tl.where(lse == float("-inf"), 0.0, lse)

        dq = tl.zeros([BLOCK_M, QK_HEAD_DIM], dtype=tl.float32)

        q_sparse_idx = q_start // sparse_q_multiple
        sparse_kv_num_blks_offset = q_sparse_idx
        sparse_kv_idx_offset = q_sparse_idx * stride_kv_idx_m
        q_sparse_base = q_sparse_idx * SPARSE_Q_BLOCK_SIZE

        kv_indices = KV_IDX + sparse_kv_idx_offset
        kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)

        for blk_idx_in_list in range(0, kv_num_blocks):
            kv_block = tl.load(kv_indices + blk_idx_in_list)
            kv_start_full = kv_block * SPARSE_KV_BLOCK_SIZE

            for kv_sub in range(NUM_KV_SUB_BLOCKS):
                start_n = kv_start_full + kv_sub * BLOCK_N
                offs_n = start_n + tl.arange(0, BLOCK_N)

                k = tl.load(
                    K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                    mask=(offs_n[:, None] < KV_LEN),    # & (offs_k[None, :] < QK_HEAD_DIM)
                    other=0.0,
                )
                v = tl.load(
                    V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                    mask=(offs_n[:, None] < KV_LEN),    # & (offs_v[None, :] < V_HEAD_DIM)
                    other=0.0,
                )

                qk = tl.dot(q, tl.trans(k), input_precision="ieee")
                qk *= SM_SCALE

                if USE_PACKED_PARTIAL_MASK:
                    partial_block_idx = tl.load(
                        PARTIAL_BLOCK_TABLE
                        + q_sparse_idx * stride_partial_table_m
                        + kv_block * stride_partial_table_n
                    )
                    safe_partial_block_idx = tl.maximum(partial_block_idx, 0, propagate_nan=True)
                    offs_m_in_block = offs_m - q_sparse_base
                    offs_n_in_block = kv_sub * BLOCK_N + tl.arange(0, BLOCK_N)
                    mask = load_packed_partial_mask(
                        PARTIAL_MASK_PACKED,
                        stride_partial_p,
                        stride_partial_m,
                        stride_partial_n,
                        safe_partial_block_idx,
                        offs_m_in_block,
                        offs_n_in_block,
                        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                    )
                    mask = mask & (partial_block_idx >= 0)
                else:
                    mask = load_dense_mask(
                        DENSE_MASK,
                        stride_mask_m,
                        stride_mask_n,
                        offs_m,
                        offs_n,
                        Q_LEN=Q_LEN,
                        KV_LEN=KV_LEN,
                    )
                qk = tl.where(mask, qk, float("-inf"))  # & (offs_n[None, :] < KV_LEN)

                p = tl.math.exp(qk - lse[:, None])
                dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                ds = p * (dp - delta[:, None])
                ds *= SM_SCALE
                # ds = tl.where(mask, ds, 0.0)
                dq += tl.dot(ds.to(MATMUL_PRECISION), k, input_precision="ieee")

        if HAS_FULL_BLOCKS:
            kv_indices_f = FULL_KV_IDX + sparse_kv_idx_offset
            kv_num_blocks_f = tl.load(FULL_KV_NUM_BLKS + sparse_kv_num_blks_offset)
            for blk_idx_in_list in range(0, kv_num_blocks_f):
                kv_block = tl.load(kv_indices_f + blk_idx_in_list)
                kv_start_full = kv_block * SPARSE_KV_BLOCK_SIZE

                for kv_sub in range(NUM_KV_SUB_BLOCKS):
                    start_n = kv_start_full + kv_sub * BLOCK_N
                    offs_n = start_n + tl.arange(0, BLOCK_N)

                    k = tl.load(
                        K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                        mask=(offs_n[:, None] < KV_LEN),    # & (offs_k[None, :] < QK_HEAD_DIM)
                        other=0.0,
                    )
                    v = tl.load(
                        V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                        mask=(offs_n[:, None] < KV_LEN),    #  & (offs_v[None, :] < V_HEAD_DIM)
                        other=0.0,
                    )

                    qk = tl.dot(q, tl.trans(k), input_precision="ieee")
                    qk *= SM_SCALE
                    # qk = tl.where(offs_n[None, :] < KV_LEN, qk, float("-inf"))

                    p = tl.math.exp(qk - lse[:, None])
                    dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                    ds = p * (dp - delta[:, None])
                    ds *= SM_SCALE
                    dq += tl.dot(ds.to(MATMUL_PRECISION), k, input_precision="ieee")

        # dq *= SM_SCALE
        tl.store(
            DQ_ptr + offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk,
            dq,
            mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
        )



@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_partial_p", "stride_partial_m",
        "stride_partial_table_m",
        "stride_lse_z", "stride_lse_h", "stride_q_idx_m",
        "Q_LEN", "KV_LEN", "NUM_TASKS", "NUM_KV_BLOCKS",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_doz", "stride_doh",
        "stride_delta_z", "stride_delta_h",
        "stride_dkz", "stride_dkh",
        "stride_dvz", "stride_dvh",
    ]
)
def flex_attention_backward_dkdv_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    Q_NUM_BLKS,
    Q_IDX,
    FULL_Q_NUM_BLKS,
    FULL_Q_IDX,
    DENSE_MASK,
    stride_mask_m,
    stride_mask_n,
    PARTIAL_MASK_PACKED,
    PARTIAL_MASK_OFFSETS,
    PARTIAL_BLOCK_TABLE,
    stride_partial_p,
    stride_partial_m,
    stride_partial_n,
    stride_partial_offset_m,
    stride_partial_table_m,
    stride_partial_table_n,
    DQ,
    DK,
    DV,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_lse_z, stride_lse_h, stride_lse_m,
    stride_delta_z, stride_delta_h, stride_delta_m,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    stride_q_idx_m,
    SM_SCALE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SUB_BLOCKS: tl.constexpr,
    NUM_TASKS,
    NUM_KV_BLOCKS,
    KV_HEAD,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    USE_PACKED_PARTIAL_MASK: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    MATMUL_PRECISION = Q.dtype.element_ty
    KV_BLOCK_SIZE: tl.constexpr = BLOCK_N * NUM_KV_SUB_BLOCKS

    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

    for task_id in range(pid, NUM_TASKS, num_core):
        kv_start_block = task_id % NUM_KV_BLOCKS
        off_z = (task_id // NUM_KV_BLOCKS) // KV_HEAD
        off_hkv = (task_id // NUM_KV_BLOCKS) % KV_HEAD

        off_z = off_z.to(tl.int64)
        off_hkv = off_hkv.to(tl.int64)

        k_offset = off_z * stride_kz + off_hkv * stride_kh
        v_offset = off_z * stride_vz + off_hkv * stride_vh
        dk_offset = off_z * stride_dkz + off_hkv * stride_dkh
        dv_offset = off_z * stride_dvz + off_hkv * stride_dvh

        K_ptr = K + k_offset
        V_ptr = V + v_offset
        DK_ptr = DK + dk_offset
        DV_ptr = DV + dv_offset

        start_n_full = kv_start_block * KV_BLOCK_SIZE

        sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
        sparse_kv_multiple = SPARSE_KV_BLOCK_SIZE // KV_BLOCK_SIZE

        kv_sparse_idx = kv_start_block // sparse_kv_multiple
        sparse_q_num_blks_offset = kv_sparse_idx
        sparse_q_idx_offset = kv_sparse_idx * stride_q_idx_m

        for kv_sub in range(NUM_KV_SUB_BLOCKS):
            sub_offset = kv_sub * BLOCK_N
            start_n = start_n_full + sub_offset
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(
                K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                other=0.0,
            )
            v = tl.load(
                V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                mask=(offs_n[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                other=0.0,
            )

            for off_g in range(0, GQA_SHARED_HEADS):
                off_hq = off_hkv * GQA_SHARED_HEADS + off_g
                off_hq = off_hq.to(tl.int64)

                q_offset = off_z * stride_qz + off_hq * stride_qh
                do_offset = off_z * stride_doz + off_hq * stride_doh
                dq_offset = off_z * stride_qz + off_hq * stride_qh
                lse_offset = off_z * stride_lse_z + off_hq * stride_lse_h
                delta_offset = off_z * stride_delta_z + off_hq * stride_delta_h

                Q_h = Q + q_offset
                DQ_h = DQ + dq_offset
                DO_h = DO + do_offset
                LSE_h = LSE + lse_offset
                DELTA_h = DELTA + delta_offset

                q_indices = Q_IDX + sparse_q_idx_offset
                q_num_blocks = tl.load(Q_NUM_BLKS + sparse_q_num_blks_offset)
                block_m_end = tl.minimum(
                    q_num_blocks * sparse_q_multiple,
                    tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL
                )
                for start_m in range(0, block_m_end):
                    blk_idx_in_list = start_m // sparse_q_multiple
                    q_block = tl.load(q_indices + blk_idx_in_list)
                    q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                    offs_m = q_start + tl.arange(0, BLOCK_M)
                    q_sparse_idx = q_block

                    bwd_dkdv_block_mn(
                        Q_h, DO_h, DQ_h, DK_ptr, DELTA_h, LSE_h, DV_ptr,
                        DENSE_MASK, stride_mask_m, stride_mask_n,
                        PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                        PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                        k, v, Q_LEN, KV_LEN,
                        off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_sparse_idx, kv_sparse_idx, kv_sub, offs_k, offs_v,
                        stride_qm, stride_qk, stride_dom, stride_dok, stride_qm, stride_qk,
                        stride_dvn, stride_dvk, stride_dkn, stride_dkk,
                        MATMUL_PRECISION,
                        SM_SCALE,
                        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                        QK_HEAD_DIM=QK_HEAD_DIM,
                        V_HEAD_DIM=V_HEAD_DIM,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        IS_FULL_BLOCKS=False,
                        USE_PACKED_PARTIAL_MASK=USE_PACKED_PARTIAL_MASK,
                        COMPUTE_DQ=False,
                    )

                if HAS_FULL_BLOCKS:
                    q_indices = FULL_Q_IDX + sparse_q_idx_offset
                    q_num_blocks = tl.load(FULL_Q_NUM_BLKS + sparse_q_num_blks_offset)
                    block_m_end = tl.minimum(
                        q_num_blocks * sparse_q_multiple,
                        tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL
                    )

                    for start_m in range(0, block_m_end):
                        blk_idx_in_list = start_m // sparse_q_multiple
                        q_block = tl.load(q_indices + blk_idx_in_list)
                        q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                        offs_m = q_start + tl.arange(0, BLOCK_M)

                        bwd_dkdv_block_mn(
                            Q_h, DO_h, DQ_h, DK_ptr, DELTA_h, LSE_h, DV_ptr,
                            DENSE_MASK, stride_mask_m, stride_mask_n,
                            PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                            PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                            k, v, Q_LEN, KV_LEN,
                            off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_block, kv_sparse_idx, kv_sub, offs_k, offs_v,
                            stride_qm, stride_qk, stride_dom, stride_dok, stride_qm, stride_qk,
                            stride_dvn, stride_dvk, stride_dkn, stride_dkk,
                            MATMUL_PRECISION,
                            SM_SCALE,
                            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                            QK_HEAD_DIM=QK_HEAD_DIM,
                            V_HEAD_DIM=V_HEAD_DIM,
                            BLOCK_M=BLOCK_M,
                            BLOCK_N=BLOCK_N,
                            IS_FULL_BLOCKS=True,
                            USE_PACKED_PARTIAL_MASK=USE_PACKED_PARTIAL_MASK,
                            COMPUTE_DQ=False,
                        )


@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_partial_p", "stride_partial_m",
        "stride_partial_table_m",
        "stride_lse_z", "stride_lse_h", "stride_q_idx_m",
        "Q_LEN", "KV_LEN", "NUM_TASKS", "NUM_KV_BLOCKS",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_doz", "stride_doh",
        "stride_delta_z", "stride_delta_h",
        "stride_dkz", "stride_dkh",
        "stride_dvz", "stride_dvh",
    ]
)
def flex_attention_backward_dqdkdv_kernel_b128(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    Q_NUM_BLKS,
    Q_IDX,
    FULL_Q_NUM_BLKS,
    FULL_Q_IDX,
    DENSE_MASK,
    stride_mask_m,
    stride_mask_n,
    PARTIAL_MASK_PACKED,
    PARTIAL_MASK_OFFSETS,
    PARTIAL_BLOCK_TABLE,
    stride_partial_p,
    stride_partial_m,
    stride_partial_n,
    stride_partial_offset_m,
    stride_partial_table_m,
    stride_partial_table_n,
    DQ,
    DK,
    DV,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_lse_z, stride_lse_h, stride_lse_m,
    stride_delta_z, stride_delta_h, stride_delta_m,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    stride_q_idx_m,
    SM_SCALE,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_TASKS,
    NUM_KV_BLOCKS,
    KV_HEAD,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    USE_PACKED_PARTIAL_MASK: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    for task_id in range(pid, NUM_TASKS, num_core):
        kv_start_block = task_id % NUM_KV_BLOCKS
        off_z = (task_id // NUM_KV_BLOCKS) // KV_HEAD
        off_hkv = (task_id // NUM_KV_BLOCKS) % KV_HEAD

        off_z = off_z.to(tl.int64)
        off_hkv = off_hkv.to(tl.int64)

        k_offset = off_z * stride_kz + off_hkv * stride_kh
        v_offset = off_z * stride_vz + off_hkv * stride_vh
        dk_offset = off_z * stride_dkz + off_hkv * stride_dkh
        dv_offset = off_z * stride_dvz + off_hkv * stride_dvh

        K_ptr = K + k_offset
        V_ptr = V + v_offset
        DK_ptr = DK + dk_offset
        DV_ptr = DV + dv_offset

        offs_n = kv_start_block * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, QK_HEAD_DIM)
        offs_v = tl.arange(0, V_HEAD_DIM)

        k = tl.load(
            K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
            mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
            other=0.0,
        )
        v = tl.load(
            V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
            mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < V_HEAD_DIM),
            other=0.0,
        )

        sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
        sparse_kv_multiple = SPARSE_KV_BLOCK_SIZE // BLOCK_N

        kv_sparse_idx = kv_start_block // sparse_kv_multiple
        sparse_q_num_blks_offset = kv_sparse_idx
        sparse_q_idx_offset = kv_sparse_idx * stride_q_idx_m

        for off_g in range(0, GQA_SHARED_HEADS):
            off_hq = off_hkv * GQA_SHARED_HEADS + off_g
            off_hq = off_hq.to(tl.int64)

            q_offset = off_z * stride_qz + off_hq * stride_qh
            do_offset = off_z * stride_doz + off_hq * stride_doh
            lse_offset = off_z * stride_lse_z + off_hq * stride_lse_h
            delta_offset = off_z * stride_delta_z + off_hq * stride_delta_h

            Q_h = Q + q_offset
            DQ_h = DQ + q_offset
            DO_h = DO + do_offset
            LSE_h = LSE + lse_offset
            DELTA_h = DELTA + delta_offset

            q_indices = Q_IDX + sparse_q_idx_offset
            q_num_blocks = tl.load(Q_NUM_BLKS + sparse_q_num_blks_offset)
            block_m_end = tl.minimum(
                q_num_blocks * sparse_q_multiple,
                tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL
            )
            for start_m in range(0, block_m_end):
                blk_idx_in_list = start_m // sparse_q_multiple
                q_block = tl.load(q_indices + blk_idx_in_list)
                q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                offs_m = q_start + tl.arange(0, BLOCK_M)
                q_sparse_idx = q_block

                q = tl.load(
                    Q_h + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                    mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                    other=0.0,
                )
                do = tl.load(
                    DO_h + offs_m[:, None] * stride_dom + offs_v[None, :] * stride_dok,
                    mask=(offs_m[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                    other=0.0,
                )
                lse = tl.load(LSE_h + offs_m * stride_lse_m, mask=offs_m < Q_LEN, other=float("-inf"))
                delta = tl.load(DELTA_h + offs_m * stride_delta_m, mask=offs_m < Q_LEN, other=0.0).to(
                    V.dtype.element_ty)
                lse = tl.where(lse == float("-inf"), 0.0, lse)

                qk = tl.dot(q, tl.trans(k), input_precision="ieee")
                qk *= SM_SCALE

                if USE_PACKED_PARTIAL_MASK:
                    partial_block_idx = tl.load(
                        PARTIAL_BLOCK_TABLE
                        + q_sparse_idx * stride_partial_table_m
                        + kv_sparse_idx * stride_partial_table_n
                    )
                    safe_partial_block_idx = tl.maximum(partial_block_idx, 0)
                    offs_m_in_block = (start_m % sparse_q_multiple) * BLOCK_M + tl.arange(0, BLOCK_M)
                    offs_n_in_block = offs_n - kv_sparse_idx * SPARSE_KV_BLOCK_SIZE
                    mask = load_packed_partial_mask(
                        PARTIAL_MASK_PACKED,
                        stride_partial_p,
                        stride_partial_m,
                        stride_partial_n,
                        safe_partial_block_idx,
                        offs_m_in_block,
                        offs_n_in_block,
                        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                    )
                    mask = mask & (partial_block_idx >= 0)
                else:
                    mask = load_dense_mask(
                        DENSE_MASK,
                        stride_mask_m,
                        stride_mask_n,
                        offs_m,
                        offs_n,
                        Q_LEN=Q_LEN,
                        KV_LEN=KV_LEN,
                    )
                qk = tl.where(mask, qk, float("-inf"))
                qk = tl.where(offs_n[:, None] < KV_LEN, qk, float("-inf"))

                p = tl.math.exp(qk - lse[:, None]).to(V.dtype.element_ty)
                dp = tl.dot(do, tl.trans(v), input_precision="ieee").to(V.dtype.element_ty)
                ds = (p * (dp - delta[:, None])).to(V.dtype.element_ty)
                ds = tl.where(mask, ds, 0.0)

                dq = tl.dot(ds.to(Q.dtype.element_ty), k, input_precision="ieee")
                tl.atomic_add(
                    DQ_h + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                    dq,
                    mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                )

                dk = tl.dot(tl.trans(ds).to(Q.dtype.element_ty), q, input_precision="ieee")
                tl.atomic_add(
                    DK_ptr + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk,
                    dk,
                    mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                )

                dv = tl.dot(tl.trans(p).to(V.dtype.element_ty), do, input_precision="ieee")
                tl.atomic_add(
                    DV_ptr + offs_n[:, None] * stride_dvn + offs_v[None, :] * stride_dvk,
                    dv,
                    mask=(offs_n[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                )

            if HAS_FULL_BLOCKS:
                q_indices = FULL_Q_IDX + sparse_q_idx_offset
                q_num_blocks = tl.load(FULL_Q_NUM_BLKS + sparse_q_num_blks_offset)
                block_m_end = tl.minimum(
                    q_num_blocks * sparse_q_multiple,
                    tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL
                )

                for start_m in range(0, block_m_end):
                    blk_idx_in_list = start_m // sparse_q_multiple
                    q_block = tl.load(q_indices + blk_idx_in_list)
                    q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                    offs_m = q_start + tl.arange(0, BLOCK_M)

                    q = tl.load(
                        Q_h + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                        mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                        other=0.0,
                    )
                    do = tl.load(
                        DO_h + offs_m[:, None] * stride_dom + offs_v[None, :] * stride_dok,
                        mask=(offs_m[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                        other=0.0,
                    )
                    lse = tl.load(LSE_h + offs_m * stride_lse_m, mask=offs_m < Q_LEN, other=float("-inf"))
                    delta = tl.load(DELTA_h + offs_m * stride_delta_m, mask=offs_m < Q_LEN, other=0.0)
                    lse = tl.where(lse == float("-inf"), 0.0, lse)

                    qk = tl.dot(q, tl.trans(k), input_precision="ieee")
                    qk *= SM_SCALE
                    qk = tl.where(offs_n[:, None] < KV_LEN, qk, float("-inf"))

                    p = tl.math.exp(qk - lse[:, None])
                    dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                    ds = p * (dp - delta[:, None])

                    dq = tl.dot(ds.to(Q.dtype.element_ty), k, input_precision="ieee")
                    tl.atomic_add(
                        DQ_h + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                        dq,
                        mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                    )

                    dk = tl.dot(tl.trans(ds).to(Q.dtype.element_ty), q, input_precision="ieee")
                    tl.atomic_add(
                        DK_ptr + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk,
                        dk,
                        mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                    )

                    dv = tl.dot(tl.trans(p).to(V.dtype.element_ty), do, input_precision="ieee")
                    tl.atomic_add(
                        DV_ptr + offs_n[:, None] * stride_dvn + offs_v[None, :] * stride_dvk,
                        dv,
                        mask=(offs_n[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                    )


@triton.jit
def bwd_dkdv_block_mn(
    Q, DO, DQ, DK_ptr, DELTA, LSE, DV_ptr,
    DENSE_MASK, stride_mask_m, stride_mask_n,
    PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
    PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
    k, v, Q_LEN, KV_LEN,
    off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_sparse_idx, kv_sparse_idx, kv_sub, offs_k, offs_v,
    stride_qm, stride_qk, stride_dom, stride_dok, stride_dqm, stride_dqd,
    stride_dvn, stride_dvk, stride_dkn, stride_dkk,
    MATMUL_PRECISION,
    SM_SCALE: tl.constexpr,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_FULL_BLOCKS: tl.constexpr,
    USE_PACKED_PARTIAL_MASK: tl.constexpr,
    COMPUTE_DQ: tl.constexpr = True,
):
    q = tl.load(
        Q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
        mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
        other=0.0,
    )
    do = tl.load(
        DO + offs_m[:, None] * stride_dom + offs_v[None, :] * stride_dok,
        mask=(offs_m[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM),
        other=0.0,
    )
    lse = tl.load(LSE + offs_m, mask=offs_m < Q_LEN, other=float("-inf"))
    lse = tl.where(lse == float("-inf"), 0.0, lse)

    qk = tl.dot(q, tl.trans(k), input_precision="ieee")
    qk *= SM_SCALE

    mask = True
    if not IS_FULL_BLOCKS:
        if USE_PACKED_PARTIAL_MASK:
            partial_block_idx = tl.load(
                PARTIAL_BLOCK_TABLE
                + q_sparse_idx * stride_partial_table_m
                + kv_sparse_idx * stride_partial_table_n
            )
            safe_partial_block_idx = tl.maximum(partial_block_idx, 0)
            sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
            offs_m_in_block = (start_m % sparse_q_multiple) * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n_in_block = kv_sub * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = load_packed_partial_mask(
                PARTIAL_MASK_PACKED,
                stride_partial_p,
                stride_partial_m,
                stride_partial_n,
                safe_partial_block_idx,
                offs_m_in_block,
                offs_n_in_block,
                SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
            )
            mask = mask & (partial_block_idx >= 0)
        else:
            mask = load_dense_mask(
                DENSE_MASK,
                stride_mask_m,
                stride_mask_n,
                offs_m,
                offs_n,
                Q_LEN=Q_LEN,
                KV_LEN=KV_LEN,
            )
        qk = tl.where(mask, qk, float("-inf"))      # & (offs_n[None, :] < KV_LEN)
    # else:
    #     qk = tl.where(offs_n[None, :] < KV_LEN, qk, float("-inf"))

    p = tl.math.exp(qk - lse[:, None])

    dv = tl.dot(tl.trans(p.to(MATMUL_PRECISION)), do, input_precision="ieee")
    tl.atomic_add(
        DV_ptr + offs_n[:, None] * stride_dvn + offs_v[None, :] * stride_dvk,
        dv,
        mask=(offs_n[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
    )

    Di = tl.load(DELTA + offs_m, mask=offs_m < Q_LEN, other=0.0)
    dp = tl.dot(do, tl.trans(v), input_precision="ieee")
    ds = (p * (dp - Di[:, None]))
    ds *= SM_SCALE

    # if not IS_FULL_BLOCKS:
    #     ds = tl.where(mask, ds, 0.0)

    if COMPUTE_DQ:
        dq = tl.dot(ds.to(MATMUL_PRECISION), k, input_precision="ieee")
        # dq *= SM_SCALE
        tl.atomic_add(
            DQ + offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqd,
            dq,
            mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
        )

    dk = tl.dot(tl.trans(ds.to(MATMUL_PRECISION)), q, input_precision="ieee")
    # dk *= SM_SCALE
    tl.atomic_add(
        DK_ptr + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk,
        dk,
        mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
    )


@triton.jit(
    do_not_specialize=[
        "stride_mask_m",
        "stride_partial_p", "stride_partial_m",
        "stride_partial_table_m",
        "stride_lse_z", "stride_lse_h", "stride_q_idx_m",
        "Q_LEN", "KV_LEN", "NUM_TASKS", "NUM_KV_BLOCKS",
        "stride_qz", "stride_qh",
        "stride_kz", "stride_kh",
        "stride_vz", "stride_vh",
        "stride_doz", "stride_doh",
        "stride_delta_z", "stride_delta_h",
        "stride_dkz", "stride_dkh",
        "stride_dvz", "stride_dvh",
    ]
)
def flex_attention_backward_dqdkdv_kernel_b64(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    Q_NUM_BLKS,
    Q_IDX,
    FULL_Q_NUM_BLKS,
    FULL_Q_IDX,
    DENSE_MASK,
    stride_mask_m,
    stride_mask_n,
    PARTIAL_MASK_PACKED,
    PARTIAL_MASK_OFFSETS,
    PARTIAL_BLOCK_TABLE,
    stride_partial_p,
    stride_partial_m,
    stride_partial_n,
    stride_partial_offset_m,
    stride_partial_table_m,
    stride_partial_table_n,
    DQ,
    DK,
    DV,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_lse_z, stride_lse_h, stride_lse_m,
    stride_delta_z, stride_delta_h, stride_delta_m,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    stride_q_idx_m,
    SM_SCALE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SUB_BLOCKS: tl.constexpr,
    NUM_TASKS,
    NUM_KV_BLOCKS,
    KV_HEAD,
    SPARSE_Q_BLOCK_SIZE: tl.constexpr,
    SPARSE_KV_BLOCK_SIZE: tl.constexpr,
    Q_LEN,
    KV_LEN,
    GQA_SHARED_HEADS,
    HAS_FULL_BLOCKS: tl.constexpr = True,
    USE_PACKED_PARTIAL_MASK: tl.constexpr = False,
    IS_DIVISIBLE: tl.constexpr = False,
    PERSISTENT_MODE: tl.constexpr = True,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    MATMUL_PRECISION = Q.dtype.element_ty
    KV_BLOCK_SIZE: tl.constexpr = BLOCK_N * NUM_KV_SUB_BLOCKS

    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

    for task_id in range(pid, NUM_TASKS, num_core):
        kv_start_block = task_id % NUM_KV_BLOCKS
        off_z = (task_id // NUM_KV_BLOCKS) // KV_HEAD
        off_hkv = (task_id // NUM_KV_BLOCKS) % KV_HEAD

        off_z = off_z.to(tl.int64)
        off_hkv = off_hkv.to(tl.int64)

        k_offset = off_z * stride_kz + off_hkv * stride_kh
        v_offset = off_z * stride_vz + off_hkv * stride_vh
        dk_offset = off_z * stride_dkz + off_hkv * stride_dkh
        dv_offset = off_z * stride_dvz + off_hkv * stride_dvh

        K_ptr = K + k_offset
        V_ptr = V + v_offset
        DK_ptr = DK + dk_offset
        DV_ptr = DV + dv_offset

        start_n_full = kv_start_block * KV_BLOCK_SIZE

        sparse_q_multiple = SPARSE_Q_BLOCK_SIZE // BLOCK_M
        sparse_kv_multiple = SPARSE_KV_BLOCK_SIZE // KV_BLOCK_SIZE

        kv_sparse_idx = kv_start_block // sparse_kv_multiple
        sparse_q_num_blks_offset = kv_sparse_idx
        sparse_q_idx_offset = kv_sparse_idx * stride_q_idx_m

        for kv_sub in range(NUM_KV_SUB_BLOCKS):
            sub_offset = kv_sub * BLOCK_N
            start_n = start_n_full + sub_offset
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(
                K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                other=0.0,
            )
            v = tl.load(
                V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                mask=(offs_n[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                other=0.0,
            )

            for off_g in range(0, GQA_SHARED_HEADS):
                off_hq = off_hkv * GQA_SHARED_HEADS + off_g
                off_hq = off_hq.to(tl.int64)

                q_offset = off_z * stride_qz + off_hq * stride_qh
                do_offset = off_z * stride_doz + off_hq * stride_doh
                dq_offset = off_z * stride_qz + off_hq * stride_qh
                lse_offset = off_z * stride_lse_z + off_hq * stride_lse_h
                delta_offset = off_z * stride_delta_z + off_hq * stride_delta_h

                Q_h = Q + q_offset
                DQ_h = DQ + dq_offset
                DO_h = DO + do_offset
                LSE_h = LSE + lse_offset
                DELTA_h = DELTA + delta_offset

                q_indices = Q_IDX + sparse_q_idx_offset
                q_num_blocks = tl.load(Q_NUM_BLKS + sparse_q_num_blks_offset)
                block_m_end = tl.minimum(
                    q_num_blocks * sparse_q_multiple,
                    tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL
                )
                for start_m in range(0, block_m_end):
                    blk_idx_in_list = start_m // sparse_q_multiple
                    q_block = tl.load(q_indices + blk_idx_in_list)
                    q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                    offs_m = q_start + tl.arange(0, BLOCK_M)
                    q_sparse_idx = q_block

                    bwd_dkdv_block_mn(
                        Q_h, DO_h, DQ_h, DK_ptr, DELTA_h, LSE_h, DV_ptr,
                        DENSE_MASK, stride_mask_m, stride_mask_n,
                        PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                        PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                        k, v, Q_LEN, KV_LEN,
                        off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_sparse_idx, kv_sparse_idx, kv_sub, offs_k, offs_v,
                        stride_qm, stride_qk, stride_dom, stride_dok, stride_qm, stride_qk,
                        stride_dvn, stride_dvk, stride_dkn, stride_dkk,
                        MATMUL_PRECISION,
                        SM_SCALE,
                        SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                        SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                        QK_HEAD_DIM=QK_HEAD_DIM,
                        V_HEAD_DIM=V_HEAD_DIM,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                        IS_FULL_BLOCKS=False,
                        USE_PACKED_PARTIAL_MASK=USE_PACKED_PARTIAL_MASK,
                    )

                if HAS_FULL_BLOCKS:
                    q_indices = FULL_Q_IDX + sparse_q_idx_offset
                    q_num_blocks = tl.load(FULL_Q_NUM_BLKS + sparse_q_num_blks_offset)
                    block_m_end = tl.minimum(
                        q_num_blocks * sparse_q_multiple,
                        tl.maximum(tl.cdiv(Q_LEN, BLOCK_M), 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL
                    )

                    for start_m in range(0, block_m_end):
                        blk_idx_in_list = start_m // sparse_q_multiple
                        q_block = tl.load(q_indices + blk_idx_in_list)
                        q_start = q_block * SPARSE_Q_BLOCK_SIZE + (start_m % sparse_q_multiple) * BLOCK_M
                        offs_m = q_start + tl.arange(0, BLOCK_M)

                        bwd_dkdv_block_mn(
                            Q_h, DO_h, DQ_h, DK_ptr, DELTA_h, LSE_h, DV_ptr,
                            DENSE_MASK, stride_mask_m, stride_mask_n,
                            PARTIAL_MASK_PACKED, stride_partial_p, stride_partial_m, stride_partial_n,
                            PARTIAL_BLOCK_TABLE, stride_partial_table_m, stride_partial_table_n,
                            k, v, Q_LEN, KV_LEN,
                            off_z, off_hq, off_hkv, offs_n, offs_m, start_m, q_block, kv_sparse_idx, kv_sub, offs_k, offs_v,
                            stride_qm, stride_qk, stride_dom, stride_dok, stride_qm, stride_qk,
                            stride_dvn, stride_dvk, stride_dkn, stride_dkk,
                            MATMUL_PRECISION,
                            SM_SCALE,
                            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
                            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
                            QK_HEAD_DIM=QK_HEAD_DIM,
                            V_HEAD_DIM=V_HEAD_DIM,
                            BLOCK_M=BLOCK_M,
                            BLOCK_N=BLOCK_N,
                            IS_FULL_BLOCKS=True,
                            USE_PACKED_PARTIAL_MASK=USE_PACKED_PARTIAL_MASK,
                        )


class FlexAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        block_mask=None,
        score_mod=None,
        mask_type="full",
        doc_start=None,
        sliding_window=None,
        global_window=None
    ):
        assert q.dim() == 4, "Q must be 4D tensor"
        assert k.dim() == 4, "K must be 4D tensor"
        assert v.dim() == 4, "V must be 4D tensor"
        del score_mod

        Z, Hq, M, D = q.shape
        _, Hkv, N, Dv = k.shape

        GQA_SHARED_HEADS = Hq // Hkv if Hq >= Hkv else 1
        assert k.shape == v.shape, "K and V must have same shape"

        SM_SCALE = 1.0 / (D ** 0.5)

        BLOCK_M = TILE_BLOCK_SIZE
        BLOCK_N = TILE_BLOCK_SIZE
        SPARSE_Q_BLOCK_SIZE = BLOCK_M
        SPARSE_KV_BLOCK_SIZE = BLOCK_N

        num_q_blocks = (M + SPARSE_Q_BLOCK_SIZE - 1) // SPARSE_Q_BLOCK_SIZE

        output = torch.empty_like(q)
        lse = torch.empty((Z, Hq, M), dtype=torch.float32, device=q.device)

        kv_num_blks = block_mask.kv_num_blocks
        kv_idx = block_mask.kv_indices
        full_kv_num_blks = getattr(block_mask, "full_kv_num_blocks", torch.zeros_like(kv_num_blks))
        full_kv_idx = getattr(block_mask, "full_kv_indices", torch.zeros_like(kv_idx))

        q_num_blks = getattr(block_mask, "q_num_blocks", None)
        q_idx = getattr(block_mask, "q_indices", None)
        assert q_num_blks is not None, "q_num_blocks and q_indices must be provided"
        assert q_idx is not None, "q_indices must be provided"
        full_q_num_blks = getattr(block_mask, "full_q_num_blocks", torch.zeros_like(q_num_blks))
        full_q_idx = getattr(block_mask, "full_q_indices", torch.zeros_like(q_idx))

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        kv_num_blks = kv_num_blks.contiguous()
        kv_idx = kv_idx.contiguous()
        full_kv_num_blks = full_kv_num_blks.contiguous()
        full_kv_idx = full_kv_idx.contiguous()

        q_num_blks = q_num_blks.contiguous()
        q_idx = q_idx.contiguous()
        full_q_num_blks = full_q_num_blks.contiguous()
        full_q_idx = full_q_idx.contiguous()

        dense_mask = getattr(block_mask, "dense_mask", None)
        packed_partial_mask = getattr(block_mask, "packed_partial_mask", None)
        partial_mask_offsets = getattr(block_mask, "partial_mask_offsets", None)
        partial_block_table = getattr(block_mask, "partial_block_table", None)
        use_packed_partial_mask = (
            packed_partial_mask is not None and
            partial_mask_offsets is not None and
            partial_block_table is not None
        )

        num_tasks = num_q_blocks * Z * Hq
        grid, num_tasks = _persistent_launch_config(num_tasks)

        kv_num_blks = kv_num_blks.to(torch.int32)
        kv_idx = kv_idx.to(torch.int32)
        full_kv_num_blks = full_kv_num_blks.to(torch.int32)
        full_kv_idx = full_kv_idx.to(torch.int32)
        q_num_blks = q_num_blks.to(torch.int32)
        q_idx = q_idx.to(torch.int32)
        full_q_num_blks = full_q_num_blks.to(torch.int32)
        full_q_idx = full_q_idx.to(torch.int32)

        if dense_mask is None:
            dense_mask = torch.zeros((1, 1, 1, 1), dtype=torch.bool, device=q.device)
        dense_mask = dense_mask.contiguous()

        if use_packed_partial_mask:
            packed_partial_mask = packed_partial_mask.contiguous()
            partial_mask_offsets = partial_mask_offsets.to(torch.int32).contiguous()
            partial_block_table = partial_block_table.to(torch.int32).contiguous()
        else:
            packed_partial_mask = torch.zeros(
                (1, SPARSE_Q_BLOCK_SIZE, SPARSE_KV_BLOCK_SIZE),
                dtype=torch.bool,
                device=q.device,
            )
            partial_mask_offsets = torch.zeros(
                (1, 1, max(num_q_blocks, 1)),
                dtype=torch.int32,
                device=q.device,
            )
            partial_block_table = torch.full(
                (max(num_q_blocks, 1), max((N + SPARSE_KV_BLOCK_SIZE - 1) // SPARSE_KV_BLOCK_SIZE, 1)),
                -1,
                dtype=torch.int32,
                device=q.device,
            )

        flex_attention_kernel[grid](
            q, k, v,
            kv_num_blks, kv_idx, full_kv_num_blks, full_kv_idx,
            dense_mask, dense_mask.stride(2), dense_mask.stride(3),
            packed_partial_mask, partial_mask_offsets,
            packed_partial_mask.stride(0), packed_partial_mask.stride(1), packed_partial_mask.stride(2),
            partial_mask_offsets.stride(2),
            output, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            kv_idx.stride(2),
            SM_SCALE = SM_SCALE,
            QK_HEAD_DIM = D,
            V_HEAD_DIM = Dv,
            BLOCK_M = BLOCK_M,
            BLOCK_N = BLOCK_N,
            NUM_TASKS = num_tasks,
            NUM_Q_BLOCKS = num_q_blocks,
            Q_HEAD = Hq,
            SPARSE_Q_BLOCK_SIZE = SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE = SPARSE_KV_BLOCK_SIZE,
            Q_LEN = M,
            KV_LEN = N,
            GQA_SHARED_HEADS = GQA_SHARED_HEADS,
            HAS_FULL_BLOCKS = True,
            USE_PACKED_PARTIAL_MASK = use_packed_partial_mask,
            limit_auto_multi_buffer_buffer="no-limit",
            hfusion_enable_multiple_consumer_fusion=True,
            intra_cache_num=3,
            inter_cache_num=2,
            enable_cross_if_fusion=True,
            enable_buffer_insert_optimization=True,
            enable_ub_refine_opt = True,
        )

        if use_packed_partial_mask:
            dense_mask_for_save = torch.zeros((1, 1, 1, 1), dtype=torch.bool, device=q.device)
        else:
            dense_mask_for_save = dense_mask

        ctx.save_for_backward(
            q, k, v, output, lse,
            dense_mask_for_save, packed_partial_mask, partial_mask_offsets, partial_block_table,
            kv_num_blks, kv_idx, full_kv_num_blks, full_kv_idx,
            q_num_blks, q_idx, full_q_num_blks, full_q_idx
        )
        ctx.mask_type = mask_type
        ctx.sliding_window = sliding_window
        ctx.global_window = global_window
        ctx.gqa_shared_heads = GQA_SHARED_HEADS
        ctx.sm_scale = SM_SCALE
        ctx.sparse_q_block_size = SPARSE_Q_BLOCK_SIZE
        ctx.sparse_kv_block_size = SPARSE_KV_BLOCK_SIZE
        ctx.has_full_blocks = True
        ctx.use_packed_partial_mask = use_packed_partial_mask

        return output, lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse=None):
        (
            q, k, v, output, lse,
            dense_mask, packed_partial_mask, partial_mask_offsets, partial_block_table,
            kv_num_blks, kv_idx, full_kv_num_blks, full_kv_idx,
            q_num_blks, q_idx, full_q_num_blks, full_q_idx
        ) = ctx.saved_tensors

        grad_output = grad_output.contiguous()
        delta = (output * grad_output).sum(dim=-1).to(torch.float32).contiguous()

        Z, Hq, M, D = q.shape
        _, Hkv, N, Dv = k.shape
        GQA_SHARED_HEADS = ctx.gqa_shared_heads
        persistent_kernel = True
        if persistent_kernel:
            dq = torch.empty_like(q)
            dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
            dv = torch.zeros(v.shape, dtype=torch.float32, device=v.device)

            BLOCK_M_DQ = TILE_BLOCK_SIZE
            BLOCK_N_DQ = TILE_BLOCK_SIZE
            NUM_KV_SUB_BLOCKS_VAL = ctx.sparse_kv_block_size // BLOCK_N_DQ
            num_q_blocks = triton.cdiv(M, BLOCK_M_DQ)
            grid_dq, num_tasks_dq = _persistent_launch_config(num_q_blocks * Z * Hq)
            flex_attention_backward_dq_kernel[grid_dq](
                q, k, v, grad_output, lse, delta,
                kv_num_blks, kv_idx, full_kv_num_blks, full_kv_idx,
                dense_mask, dense_mask.stride(2), dense_mask.stride(3),
                packed_partial_mask, partial_mask_offsets, partial_block_table,
                packed_partial_mask.stride(0), packed_partial_mask.stride(1), packed_partial_mask.stride(2),
                partial_mask_offsets.stride(2),
                partial_block_table.stride(0), partial_block_table.stride(1),
                dq,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                delta.stride(0), delta.stride(1), delta.stride(2),
                dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                kv_idx.stride(2),
                SM_SCALE=ctx.sm_scale,
                QK_HEAD_DIM=D,
                V_HEAD_DIM=Dv,
                BLOCK_M=BLOCK_M_DQ,
                BLOCK_N=BLOCK_N_DQ,
                NUM_KV_SUB_BLOCKS=NUM_KV_SUB_BLOCKS_VAL,
                NUM_TASKS=num_tasks_dq,
                NUM_Q_BLOCKS=num_q_blocks,
                Q_HEAD=Hq,
                SPARSE_Q_BLOCK_SIZE=ctx.sparse_q_block_size,
                SPARSE_KV_BLOCK_SIZE=ctx.sparse_kv_block_size,
                Q_LEN=M,
                KV_LEN=N,
                GQA_SHARED_HEADS=GQA_SHARED_HEADS,
                HAS_FULL_BLOCKS=ctx.has_full_blocks,
                USE_PACKED_PARTIAL_MASK=ctx.use_packed_partial_mask,
                limit_auto_multi_buffer_buffer="no-limit",
                hfusion_enable_multiple_consumer_fusion=True,
                enable_select_analysis=False,
                limit_auto_multi_buffer_of_local_buffer = "no-l0c",
                intra_cache_num=3,
                inter_cache_num=2,
            )

            BLOCK_M_DKDV = TILE_BLOCK_SIZE
            BLOCK_N_DKDV = TILE_BLOCK_SIZE
            NUM_KV_SUB_BLOCKS_VAL = ctx.sparse_kv_block_size // BLOCK_N_DKDV
            num_kv_blocks = triton.cdiv(N, ctx.sparse_kv_block_size)
            grid_dkv, num_tasks_dkv = _persistent_launch_config(num_kv_blocks * Z * Hkv)
            flex_attention_backward_dkdv_kernel[grid_dkv](
                q, k, v, grad_output, lse, delta,
                q_num_blks, q_idx, full_q_num_blks, full_q_idx,
                dense_mask, dense_mask.stride(2), dense_mask.stride(3),
                packed_partial_mask, partial_mask_offsets, partial_block_table,
                packed_partial_mask.stride(0), packed_partial_mask.stride(1), packed_partial_mask.stride(2),
                partial_mask_offsets.stride(2),
                partial_block_table.stride(0), partial_block_table.stride(1),
                dq, dk, dv,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                delta.stride(0), delta.stride(1), delta.stride(2),
                dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                q_idx.stride(2),
                SM_SCALE=ctx.sm_scale,
                QK_HEAD_DIM=D,
                V_HEAD_DIM=Dv,
                BLOCK_M=BLOCK_M_DKDV,
                BLOCK_N=BLOCK_N_DKDV,
                NUM_KV_SUB_BLOCKS=NUM_KV_SUB_BLOCKS_VAL,
                NUM_TASKS=num_tasks_dkv,
                NUM_KV_BLOCKS=num_kv_blocks,
                KV_HEAD=Hkv,
                SPARSE_Q_BLOCK_SIZE=ctx.sparse_q_block_size,
                SPARSE_KV_BLOCK_SIZE=ctx.sparse_kv_block_size,
                Q_LEN=M,
                KV_LEN=N,
                GQA_SHARED_HEADS=GQA_SHARED_HEADS,
                HAS_FULL_BLOCKS=ctx.has_full_blocks,
                USE_PACKED_PARTIAL_MASK=ctx.use_packed_partial_mask,
                limit_auto_multi_buffer_buffer="no-limit",
                hfusion_enable_multiple_consumer_fusion=True,
                unit_flag=True,
                limit_auto_multi_buffer_of_local_buffer="no-l0c",
                intra_cache_num=2,
                inter_cache_num=1,
            )
        else:
            dq = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
            dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
            dv = torch.zeros(v.shape, dtype=torch.float32, device=k.device)

            if True:
                BLOCK_M_DKDV = 128
                BLOCK_N_DKDV = 64
                NUM_KV_SUB_BLOCKS_VAL = ctx.sparse_kv_block_size // BLOCK_N_DKDV
                num_kv_blocks = triton.cdiv(N, ctx.sparse_kv_block_size)
                grid_dkdv, num_tasks_dkv = _persistent_launch_config(num_kv_blocks * Z * Hkv)
                flex_attention_backward_dqdkdv_kernel_b64[grid_dkdv](
                    q, k, v, grad_output, lse, delta,
                    q_num_blks, q_idx, full_q_num_blks, full_q_idx,
                    dense_mask, dense_mask.stride(2), dense_mask.stride(3),
                    packed_partial_mask, partial_mask_offsets, partial_block_table,
                    packed_partial_mask.stride(0), packed_partial_mask.stride(1), packed_partial_mask.stride(2),
                    partial_mask_offsets.stride(2),
                    partial_block_table.stride(0), partial_block_table.stride(1),
                    dq, dk, dv,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
                    lse.stride(0), lse.stride(1), lse.stride(2),
                    delta.stride(0), delta.stride(1), delta.stride(2),
                    dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                    dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                    q_idx.stride(2),
                    SM_SCALE=ctx.sm_scale,
                    QK_HEAD_DIM=D,
                    V_HEAD_DIM=Dv,
                    BLOCK_M=BLOCK_M_DKDV,
                    BLOCK_N=BLOCK_N_DKDV,
                    NUM_KV_SUB_BLOCKS=NUM_KV_SUB_BLOCKS_VAL,
                    NUM_TASKS=num_tasks_dkv,
                    NUM_KV_BLOCKS=num_kv_blocks,
                    KV_HEAD=Hkv,
                    SPARSE_Q_BLOCK_SIZE=ctx.sparse_q_block_size,
                    SPARSE_KV_BLOCK_SIZE=ctx.sparse_kv_block_size,
                    Q_LEN=M,
                    KV_LEN=N,
                    GQA_SHARED_HEADS=GQA_SHARED_HEADS,
                    HAS_FULL_BLOCKS=ctx.has_full_blocks,
                    USE_PACKED_PARTIAL_MASK=ctx.use_packed_partial_mask,
                )
            else:
                BLOCK_M_DKDV = TILE_BLOCK_SIZE
                BLOCK_N_DKDV = TILE_BLOCK_SIZE
                num_kv_blocks = triton.cdiv(N, BLOCK_N_DKDV)
                grid_dkdv, num_tasks_dkv = _persistent_launch_config(num_kv_blocks * Z * Hkv)
                flex_attention_backward_dqdkdv_kernel_b128[grid_dkdv](
                    q, k, v, grad_output, lse, delta,
                    q_num_blks, q_idx, full_q_num_blks, full_q_idx,
                    dense_mask, dense_mask.stride(2), dense_mask.stride(3),
                    packed_partial_mask, partial_mask_offsets, partial_block_table,
                    packed_partial_mask.stride(0), packed_partial_mask.stride(1), packed_partial_mask.stride(2),
                    partial_mask_offsets.stride(2),
                    partial_block_table.stride(0), partial_block_table.stride(1),
                    dq, dk, dv,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
                    lse.stride(0), lse.stride(1), lse.stride(2),
                    delta.stride(0), delta.stride(1), delta.stride(2),
                    dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                    dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                    q_idx.stride(2),
                    SM_SCALE=ctx.sm_scale,
                    QK_HEAD_DIM=D,
                    V_HEAD_DIM=Dv,
                    BLOCK_M=BLOCK_M_DKDV,
                    BLOCK_N=BLOCK_N_DKDV,
                    NUM_TASKS=num_tasks_dkv,
                    NUM_KV_BLOCKS=num_kv_blocks,
                    KV_HEAD=Hkv,
                    SPARSE_Q_BLOCK_SIZE=ctx.sparse_q_block_size,
                    SPARSE_KV_BLOCK_SIZE=ctx.sparse_kv_block_size,
                    Q_LEN=M,
                    KV_LEN=N,
                    GQA_SHARED_HEADS=GQA_SHARED_HEADS,
                    HAS_FULL_BLOCKS=ctx.has_full_blocks,
                    USE_PACKED_PARTIAL_MASK=ctx.use_packed_partial_mask,
                )
                dq *= ctx.sm_scale
                dk *= ctx.sm_scale
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, None, None


def flex_attention(
    q,
    k,
    v,
    block_mask=None,
    score_mod=None,
    return_lse=None,
    mask_type="full",
    doc_start=None,
    sliding_window=None,
    global_window=None,
):
    """
    Args:
        q: Query tensor [Z, Hq, M, D]
        k: Key tensor [Z, Hkv, N, D]
        v: Value tensor [Z, Hkv, N, Dv]
        block_mask: Block mask from torch.nn.attention.flex_attention
        score_mod: Optional score modification function

    Returns:
        Output tensor [Z, Hq, N, Dv]
    """
    output, lse = FlexAttentionFunc.apply(
        q,
        k,
        v,
        block_mask,
        score_mod,
        mask_type,
        doc_start,
        sliding_window,
        global_window,
    )
    if return_lse:
        return output, lse
    return output
