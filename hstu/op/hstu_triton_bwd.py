# Copyright 2025 Huawei Technologies Co., Ltd
from typing import Tuple

import functools
import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver

DEVICE = "npu"

TILE_CONFIGS = [
    {
        "condition": lambda q, k, d: q <= 32 and k <= 32 and d == 256,
        "BLOCK_M": 32, "BLOCK_N": 32,
        "seq_type": 3,
    },
    {
        "condition": lambda q, k, d: q <= 64 and k <= 1024 and d == 64,
        "BLOCK_M": 64, "BLOCK_N": 256,
        "seq_type": 3,  # seq_len_q < BLOCK_M
    },
    {
        "condition": lambda q, k, d: q <= 32 and k <= 512 and d == 32,
        "BLOCK_M": 32, "BLOCK_N": 256,
        "seq_type": 3,
    },
    {
        "condition": lambda q, k, d: d >= 512,
        "BLOCK_M": 32, "BLOCK_N": 32,
        "seq_type": 1,
    },
    {
        "condition": lambda q, k, d: d >= 256,
        "BLOCK_M": 64, "BLOCK_N": 72,
        "seq_type": 1,
    },
    {
        "condition": lambda q, k, d: True,
        "BLOCK_M": 64, "BLOCK_N": 144,
        "seq_type": 1,
    },
]


@triton.jit
def _init_batch_state(
        task_start,
        seq_offsets_q,
        BATCH_SIZE,
        BLOCK_M,
        total_q_blocks,
        seq_type,
):
    # 计算起始 task 对应的全局 block index
    start_global_q_block_idx = task_start % total_q_blocks
    if seq_type == 0:  # SeqType.BASIC_QKV
        state_batch_id = tl.cast(BATCH_SIZE - 1, tl.int64)
        state_cum_blocks = state_batch_id
        state_batch_num_blocks = tl.cast(1, tl.int64)
    else:
        # 执行一次全量扫描，找到该 core 起始状态
        # 使用 tl.where 规避 break 报错
        cur_batch_id = tl.cast(0, tl.int64)
        cur_cum_blocks = tl.cast(0, tl.int64)
        start_batch_cum_blocks = tl.cast(0, tl.int64)

        for b in range(BATCH_SIZE):
            seq_len = tl.load(seq_offsets_q + b + 1) - tl.load(seq_offsets_q + b)
            nb = tl.cdiv(seq_len, BLOCK_M)

            # 找到最后一个满足 start_global >= cum_blocks 的位置
            cond = start_global_q_block_idx >= cur_cum_blocks
            cur_batch_id = tl.where(cond, b, cur_batch_id)
            start_batch_cum_blocks = tl.where(cond, cur_cum_blocks, start_batch_cum_blocks)
            cur_cum_blocks += nb

        state_batch_id = cur_batch_id
        state_cum_blocks = start_batch_cum_blocks

        # 预加载当前 batch 的 block 数量
        cur_seq_len = tl.load(seq_offsets_q + state_batch_id + 1) - tl.load(seq_offsets_q + state_batch_id)
        state_batch_num_blocks = tl.cdiv(cur_seq_len, BLOCK_M)

    return state_batch_id, state_cum_blocks, state_batch_num_blocks


@triton.jit
def _get_task_info(
        task_idx,
        state_batch_id,
        state_cum_blocks,
        state_batch_num_blocks,
        seq_offsets_q,
        BLOCK_M,
        total_q_blocks,
        seq_type,
):
    head_id = task_idx // total_q_blocks
    global_q_block_idx = task_idx % total_q_blocks
    if seq_type == 0:
        state_batch_id = global_q_block_idx
        state_cum_blocks = global_q_block_idx
        state_batch_num_blocks = tl.cast(1, tl.int64)
        q_block_id = tl.cast(0, tl.int64)
    else:
        # 如果 global_q_block_idx 小于当前的累积块数，说明进入了下一个 Head 的处理序列，
        # 需要将 Batch 状态重置为初始状态（Batch 0）。
        if global_q_block_idx < state_cum_blocks:
            state_batch_id = tl.cast(0, tl.int64)
            state_cum_blocks = tl.cast(0, tl.int64)
            # 重新加载 Batch 0 的长度信息
            cur_seq_len = tl.load(seq_offsets_q + 1) - tl.load(seq_offsets_q)
            state_batch_num_blocks = tl.cdiv(cur_seq_len, BLOCK_M)
        # 增量更新逻辑：如果当前全局块索引超出了当前 Batch 的范围，移动到下一个 Batch
        while global_q_block_idx >= state_cum_blocks + state_batch_num_blocks:
            state_cum_blocks += state_batch_num_blocks
            state_batch_id += 1
            # 仅在切换 Batch 时加载新的长度信息
            cur_seq_len = tl.load(seq_offsets_q + state_batch_id + 1) - tl.load(seq_offsets_q + state_batch_id)
            state_batch_num_blocks = tl.cdiv(cur_seq_len, BLOCK_M)

        # 计算局部的 q_block_id
        q_block_id = global_q_block_idx - state_cum_blocks

    # 返回更新后的状态变量和计算结果
    return state_batch_id, state_cum_blocks, state_batch_num_blocks, q_block_id, head_id

# SYNC_QK_READY = 0  # cube -> vector: QK 结果已经写到 qk_ub
# SYNC_S_READY  = 1  # vector -> cube: S 已经写到 s_l1，可以做 SV

@triton.jit
def _inner_qk_gv_matmul(
        q_reuse, dout_reuse,
        q_base, k_base,
        dout_base, v_base,
        seq_len_q, seq_len_k,
        q_block_id, n_offset,
        stride_qm, stride_kn, stride_om, stride_vn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr, V_HEAD_DIM: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_vd = tl.arange(0, V_HEAD_DIM)

    k_row_idx = n_offset + offs_n
    k_row_mask = k_row_idx < seq_len_k
    k_ptrs = k_base + k_row_idx[:, None] * stride_kn + offs_d[None, :]
    k_sub = tl.load(k_ptrs, mask=k_row_mask[:, None], other=0.0)

    s_tile = tl.dot(q_reuse, tl.trans(k_sub))

    v_ptrs = v_base + k_row_idx[:, None] * stride_vn + offs_vd[None, :]
    v_sub = tl.load(v_ptrs, mask=k_row_mask[:, None], other=0.0)

    gv_tile = tl.dot(dout_reuse, tl.trans(v_sub))
    return s_tile, gv_tile


@triton.jit
def _inner_vec_score(
        qk_ub,
        gv_ub,
        alpha, silu_scale,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        q_block_id, n_offset,
        seq_len_q, seq_len_k,
        dtype,
):
    qk = qk_ub
    qk_scaled = qk * alpha
    sig = tl.sigmoid(qk_scaled)
    s_silu = qk_scaled * sig
    score = s_silu * silu_scale
    score = score.to(dtype)

    gv = gv_ub
    ds_silu = gv * silu_scale
    silu_grad = sig * (1.0 + qk_scaled * (1.0 - sig))
    ds_scaled = ds_silu * silu_grad
    ds_raw = ds_scaled * alpha
    attn_bias_grad = ds_raw.to(dtype)

    return score, attn_bias_grad


@triton.jit
def _do_dq_cast_store(
    dq_fp32_base, dq_base,
    seq_len_q, q_block_id,
    stride_qm, stride_qh,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
    DATA_TYPE,
):
    offs_m = tl.arange(0, BLOCK_M)
    q_row_idx = q_block_id * BLOCK_M + offs_m

    offs_d = tl.arange(0, HEAD_DIM)
    mask = q_row_idx < seq_len_q

    # # 从 GM 读取 FP32 累加结果
    acc_ptrs = dq_fp32_base + q_row_idx[:, None] * stride_qm + offs_d[None, :]
    acc_val = tl.load(acc_ptrs, mask=mask[:, None], other=0.0)

    out_val = acc_val.to(DATA_TYPE)

    # 写入 Output
    out_ptrs = dq_base + q_row_idx[:, None] * stride_qm + offs_d[None, :]
    tl.store(out_ptrs, out_val, mask=mask[:, None])


@triton.jit
def _do_dtype_cast(
        input_ptr, output_ptr,
        n_elements,
        INNER_BLOCK_SIZE: tl.constexpr,
        DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    block_size = (n_elements + num_pids - 1) // num_pids
    program_start = pid * block_size
    program_limit = program_start + block_size

    for inner_offset in range(0, block_size, INNER_BLOCK_SIZE):
        offsets = program_start + inner_offset + tl.arange(0, INNER_BLOCK_SIZE)
        # mask = (offsets < n_elements) & (offsets < program_limit)
        real_limit = tl.minimum(n_elements, program_limit)
        mask = offsets < real_limit

        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        y = x.to(DTYPE)
        tl.store(output_ptr + offsets, y, mask=mask)

@triton.jit
def _inner_grad_matmul(
        q_reuse, dout_reuse,
        q_base, k_base, v_base, dout_base,
        dq_fp32_base, dk_fp32_base, dv_fp32_base,
        score, attn_bias_grad,
        seq_len_q, seq_len_k,
        q_block_id, n_offset,
        seq_type, DATA_TYPE,
        stride_vn, stride_om, stride_qm, stride_kn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr, V_HEAD_DIM: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_vd = tl.arange(0, V_HEAD_DIM)

    q_row_idx = q_block_id * BLOCK_M + offs_m
    q_row_mask = q_row_idx < seq_len_q

    k_row_idx = n_offset + offs_n
    k_row_mask = k_row_idx < seq_len_k

    # dQ += dQ_part
    k_ptrs = k_base + k_row_idx[:, None] * stride_kn + offs_d[None, :]
    k = tl.load(k_ptrs, mask=k_row_mask[:, None], other=0.0)
    dq_part = tl.dot(attn_bias_grad, k)
    dq_ptrs = dq_fp32_base + q_row_idx[:, None] * stride_qm + offs_d[None, :]
    tl.atomic_add(dq_ptrs, dq_part, mask=q_row_mask[:, None])

    # dK = dS_raw^T @ Q: [BLOCK_N, HEAD_DIM]
    dk = tl.dot(tl.trans(attn_bias_grad), q_reuse)
    dk_ptrs = dk_fp32_base + (k_row_idx[:, None] * stride_kn + offs_d[None, :])
    if seq_type == 3:
        tl.store(dk_ptrs, dk.to(DATA_TYPE), mask=k_row_mask[:, None])
    else:
        tl.atomic_add(dk_ptrs, dk, mask=k_row_mask[:, None])

    # dV = S_final^T @ dOut: [BLOCK_N, HEAD_DIM]
    dv = tl.dot(tl.trans(score), dout_reuse)
    dv_ptrs = dv_fp32_base + (k_row_idx[:, None] * stride_vn + offs_vd[None, :])
    if seq_type == 3:
        tl.store(dv_ptrs, dv.to(DATA_TYPE), mask=k_row_mask[:, None])
    else:
        tl.atomic_add(dv_ptrs, dv, mask=k_row_mask[:, None])


@triton.jit
def _parallel_hstu_attn_bwd(
        DOut,
        Q,
        K,
        V,
        DQ_fp32,
        DK_fp32,
        DV_fp32,
        DQ,
        DK,
        DV,
        seq_offsets_q,
        seq_offsets_k,
        stride_qm: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_kn: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_vn: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_om: tl.constexpr,
        stride_oh: tl.constexpr,
        alpha,
        silu_scale,
        DeltaSize,
        HEAD_DIM: tl.constexpr,
        V_HEAD_DIM: tl.constexpr,
        HEAD_NUM_Q: tl.constexpr,
        BATCH_SIZE: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        IS_DELTA_Q: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        dtype: tl.constexpr,
        seq_type: tl.constexpr,
        numel_k: tl.constexpr,
        numel_v: tl.constexpr,
        inner_cast_num_k: tl.constexpr,
        inner_cast_num_v: tl.constexpr,
):
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    # kernel内部计算总的q_blocks数量
    if seq_type == 0:
        total_q_blocks = tl.cast(BATCH_SIZE, tl.int64)
    else:
        total_q_blocks = tl.cast(0, tl.int64)
        for b in range(BATCH_SIZE):
            seq_len = tl.load(seq_offsets_q + b + 1) - tl.load(seq_offsets_q + b)
            num_blocks = tl.cdiv(seq_len, BLOCK_M)
            total_q_blocks += num_blocks
    total_tasks = HEAD_NUM_Q * total_q_blocks

    # kernel内部平均分配：每个core处理的任务范围
    tasks_per_core = tl.cdiv(total_tasks, num_cores)
    task_start = pid * tasks_per_core
    task_end = tl.minimum((pid + 1) * tasks_per_core, total_tasks)

    DATA_TYPE: tl.constexpr = tl.float16 if dtype == 0 else tl.bfloat16 if dtype == 1 else tl.float8e4nv
    state_batch_id, state_cum_blocks, state_batch_num_blocks = _init_batch_state(
        task_start, seq_offsets_q, BATCH_SIZE, BLOCK_M, total_q_blocks, seq_type,
    )
    for task_idx in tl.range(task_start, task_end):
        state_batch_id, state_cum_blocks, state_batch_num_blocks, q_block_id, head_id = _get_task_info(
            task_idx, state_batch_id, state_cum_blocks, state_batch_num_blocks,
            seq_offsets_q, BLOCK_M, total_q_blocks, seq_type,
        )
        batch_id = state_batch_id
        kv_head_id = head_id // GROUP_SIZE

        seq_start_q = tl.load(seq_offsets_q + batch_id)
        seq_end_q = tl.load(seq_offsets_q + batch_id + 1)
        seq_start_k = tl.load(seq_offsets_k + batch_id)
        seq_end_k = tl.load(seq_offsets_k + batch_id + 1)
        seq_len_q = seq_end_q - seq_start_q
        seq_len_k = seq_end_k - seq_start_k

        q_base = Q + head_id * stride_qh + seq_start_q * stride_qm
        k_base = K + kv_head_id * stride_kh + seq_start_k * stride_kn
        v_base = V + kv_head_id * stride_vh + seq_start_k * stride_vn
        dout_base = DOut + head_id * stride_oh + seq_start_q * stride_om

        dq_fp32_base = DQ_fp32 + head_id * stride_qh + seq_start_q * stride_qm
        dq_base = DQ + head_id * stride_qh + seq_start_q * stride_qm
        if seq_type == 3:
            dk_fp32_base = DK + kv_head_id * stride_kh + seq_start_k * stride_kn
            dv_fp32_base = DV + kv_head_id * stride_vh + seq_start_k * stride_vn
        else:
            dk_fp32_base = DK_fp32 + kv_head_id * stride_kh + seq_start_k * stride_kn
            dv_fp32_base = DV_fp32 + kv_head_id * stride_vh + seq_start_k * stride_vn

        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        offs_vd = tl.arange(0, V_HEAD_DIM)
        q_row_idx = q_block_id * BLOCK_M + offs_m
        q_row_mask = q_row_idx < seq_len_q
        dout_ptrs = dout_base + q_row_idx[:, None] * stride_om + offs_vd[None, :]
        dout_reuse = tl.load(dout_ptrs, mask=q_row_mask[:, None], other=0.0)
        q_ptrs = q_base + q_row_idx[:, None] * stride_qm + offs_d[None, :]
        q_reuse = tl.load(q_ptrs, mask=q_row_mask[:, None], other=0.0)

        num_tiles = tl.cdiv(seq_len_k, BLOCK_N)
        for p in range(0, num_tiles):
            #############tile_qk#############
            tile_qk = p
            n_offset_qk = tile_qk * BLOCK_N
            qk_ub, gv_ub = _inner_qk_gv_matmul(
                q_reuse, dout_reuse,
                q_base, k_base,
                dout_base, v_base,
                seq_len_q, seq_len_k, q_block_id, n_offset_qk,
                stride_qm, stride_kn, stride_om, stride_vn,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                HEAD_DIM=HEAD_DIM, V_HEAD_DIM=V_HEAD_DIM,
            )

            #############tile_vec#############
            tile_vec = p
            n_offset_vec = tile_vec * BLOCK_N
            score, attn_bias_grad = _inner_vec_score(
                qk_ub,
                gv_ub,
                alpha, silu_scale,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                q_block_id=q_block_id, n_offset=n_offset_vec,
                seq_len_q=seq_len_q, seq_len_k=seq_len_k,
                dtype=DATA_TYPE,
            )
            #############tile_grad#############
            tile_grad = p
            n_offset_grad = tile_grad * BLOCK_N
            _inner_grad_matmul(
                q_reuse, dout_reuse,
                q_base, k_base, v_base, dout_base,
                dq_fp32_base, dk_fp32_base, dv_fp32_base,
                score, attn_bias_grad,
                seq_len_q, seq_len_k, q_block_id, n_offset_grad,
                seq_type, DATA_TYPE,
                stride_vn, stride_om, stride_qm, stride_kn,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                HEAD_DIM=HEAD_DIM, V_HEAD_DIM=V_HEAD_DIM,
            )


        _do_dq_cast_store(
            dq_fp32_base, dq_base,
            seq_len_q, q_block_id,
            stride_qm, stride_qh,
            BLOCK_M, HEAD_DIM,
            DATA_TYPE,
        )

@functools.cache
def get_core_num():
    return driver.active.utils.get_device_properties("npu")["num_aicore"]

def triton_hstu_attention_backward(
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        max_seq_len_q: int,
        max_seq_len_k: int,
        seq_offsets_q: torch.Tensor,
        seq_offsets_k: torch.Tensor,
        num_context: torch.Tensor,
        num_target: torch.Tensor,
        alpha: float,
        silu_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = q.device
    batch_size = len(seq_offsets_q) - 1
    total_tokens, head_num_q, head_dim = q.shape
    head_num_k = k.shape[1]
    total_tokens_kv, _, v_head_dim = v.shape
    numel_k = k.numel()
    numel_v = v.numel()
    UBThreshold = 32 * 1024
    core_num = get_core_num()
    block_size_k = (numel_k + core_num - 1) // core_num
    block_size_v = (numel_v + core_num - 1) // core_num
    assert head_num_q % head_num_k == 0, f"Query heads ({head_num_q}) must be divisible by KV heads ({head_num_k})"
    group_size = head_num_q // head_num_k
    # todo autotune
    for config in TILE_CONFIGS:
        if config["condition"](max_seq_len_q, max_seq_len_k, v_head_dim):
            BLOCK_M = config["BLOCK_M"]
            BLOCK_N = config["BLOCK_N"]
            seq_type = config["seq_type"]
            break
    # fp8 BLOCK_N必须是64的倍数
    if q.dtype == torch.float8_e4m3fn:
        BLOCK_M = BLOCK_N = 64
    print(f"[Debug] BLOCK_M {BLOCK_M} BLOCK_N {BLOCK_N} seq_type {seq_type}")

    dq_fp32 = torch.zeros(q.shape, dtype=torch.float32).to(device)
    dk_fp32 = torch.zeros(k.shape, dtype=torch.float32).to(device)
    dv_fp32 = torch.zeros(v.shape, dtype=torch.float32).to(device)
    out_dtype = torch.float16 if q.dtype == torch.float8_e4m3fn else q.dtype
    dq = torch.zeros(q.shape, dtype=out_dtype).to(device)
    dk = torch.zeros(k.shape, dtype=out_dtype).to(device)
    dv = torch.zeros(v.shape, dtype=out_dtype).to(device)

    # 0:float16 1:bfloat16 2:fp8
    type_mapper = {torch.float16: 0, torch.bfloat16: 1, torch.float8_e4m3fn: 2}
    dtype = type_mapper.get(q.dtype, 0)

    grid = (core_num,)

    _parallel_hstu_attn_bwd[grid](
        DOut=dout,
        Q=q,
        K=k,
        V=v,
        DQ_fp32=dq_fp32,
        DK_fp32=dk_fp32,
        DV_fp32=dv_fp32,
        DQ=dq,
        DK=dk,
        DV=dv,
        seq_offsets_q=seq_offsets_q,
        seq_offsets_k=seq_offsets_k,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_om=dout.stride(0),
        stride_oh=dout.stride(1),
        alpha=alpha,
        silu_scale=silu_scale,
        DeltaSize=0,
        HEAD_DIM=head_dim,
        V_HEAD_DIM=v_head_dim,
        GROUP_SIZE=group_size,
        HEAD_NUM_Q=head_num_q,
        BATCH_SIZE=batch_size,
        IS_DELTA_Q=False,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        dtype=dtype,
        seq_type=seq_type,
        numel_k=numel_k,
        numel_v=numel_v,
        inner_cast_num_k=block_size_k//2 if block_size_k//2 < UBThreshold else UBThreshold,
        inner_cast_num_v=block_size_v//2 if block_size_v//2 < UBThreshold else UBThreshold,
        enable_mixed_cv=True,
        enable_auto_bind_sub_block=True,
        enable_flatten=False,
        # sync_solver=True,
    )
    if seq_type == 3:
        return dq, dk, dv
    else:
        return dq, dk_fp32.to(q.dtype), dv_fp32.to(q.dtype)
