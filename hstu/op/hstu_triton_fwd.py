import math
import functools
import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

from enum import IntEnum

class SeqType(IntEnum):
    BASIC_QKV = 0 # 基本块模式，sv可以直接store
    FEW_KV = 1    # K块数量<=3，sv在fp16上累加精度可接受
    LONG_QKV = 2  # 长序列模式，需在f32上累加，增加cast流水

# 长度为len搬运n次的最优BLOCK（使多余搬运最少，BLOCK被32整除）
# BLOCK = 32*ceil(len/(32*n)) n=1,2,...
TILE_CONFIGS = [
    {# (2048, 2, 2, 32, 32, 256, 256),
        "condition": lambda q, k, d: q <= 32 and k <= 32 and d == 256,
        "BLOCK_M": 32, "BLOCK_N": 32,
    },
    {# (96, 2, 2, 512, 3072, 256, 256), # delta_q
        "condition": lambda q, k, d: q <= 512 and k <= 4096 and d == 256,
        "BLOCK_M": 32, "BLOCK_N": 384,
    },
    {# (5, 8, 4, 128, 256, 96, 128), # (v_d != qk_d)
        "condition": lambda q, k, d: q <= 128 and k <= 256 and d == 128,
        "BLOCK_M": 128, "BLOCK_N": 256,
    },
    {# (2, 4, 1, 1001, 901, 128, 48), # (v_d != qk_d)
        "condition": lambda q, k, d: q <= 1024 and k <= 1024 and d == 128,
        "BLOCK_M": 64, "BLOCK_N": 512,
    },
    {# (16, 4, 4, 32, 499, 64, 32), # (v_d != qk_d)
        "condition": lambda q, k, d: q <= 32 and k <= 512 and d == 64,
        "BLOCK_M": 32, "BLOCK_N": 512,
    },
    {# (2048, 4, 4, 52, 1000, 64, 64), # delta_q
        "condition": lambda q, k, d: q <= 64 and k <= 1024 and d == 64,
        "BLOCK_M": 64, "BLOCK_N": 512,
    },
    {# (2, 8, 4, 256, 64, 64, 64), # GQA
        "condition": lambda q, k, d: q <= 256 and k <= 64 and d == 64,
        "BLOCK_M": 256, "BLOCK_N": 64,
    },
    {
        "condition": lambda q, k, d: q <= 256 and k <= 256 and d == 64,
        "BLOCK_M": 128, "BLOCK_N": 256,
    },
    {# (16, 4, 4, 501, 1000, 64, 64), # delta_q
        "condition": lambda q, k, d: q <= 512 and k <= 1024 and d == 64,
        "BLOCK_M": 128, "BLOCK_N": 256,
    },
    {
        "condition": lambda q, k, d: q <= 32 and k <= 512 and d == 32,
        "BLOCK_M": 32, "BLOCK_N": 512,
    },
    {
        "condition": lambda q, k, d: d <= 64,
        "BLOCK_M": 128, "BLOCK_N": 256,
    },
    {# (4, 4, 4, 512, 1024, 72, 64), # (v_d != qk_d)
        "condition": lambda q, k, d: d <= 80,
        "BLOCK_M": 64, "BLOCK_N": 512,
    },
    {# (8, 8, 8, 8000, 8000, 128, 128),
        "condition": lambda q, k, d: d <= 128,
        "BLOCK_M": 128, "BLOCK_N": 256,
    },
    {# (8, 8, 8, 8000, 8000, 256, 256),
        "condition": lambda q, k, d: d <= 256,
        "BLOCK_M": 64, "BLOCK_N": 384,
    },
    {# (96, 2, 2, 512, 3072, 512, 512), # delta_q
        "condition": lambda q, k, d: d <= 512,
        "BLOCK_M": 32, "BLOCK_N": 192,
    },
    {
        "condition": lambda q, k, d: True,
        "BLOCK_M": 128, "BLOCK_N": 256,
    },
]

@triton.jit
def _upper_bound_prefix(prefix_ptr, n: tl.constexpr, log_n: tl.constexpr, x):
    # return first idx such that prefix[idx] > x
    l = 0
    r = n
    for _ in range(log_n):
        m = (l + r) // 2
        v = tl.load(prefix_ptr + m)
        if v <= x:
            l = m + 1
        else:
            r = m
    return l

@triton.jit
def _map_qtask_to_batch_qblk(
    q_task,
    total_q_blocks,
    q_blk_offsets,
    BATCH_SIZE: tl.constexpr,
    LOG_BATCH_SIZE_P1: tl.constexpr,
    MAX_SEQ_LEN_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    head_id = q_task // total_q_blocks
    gq = q_task % total_q_blocks
    if MAX_SEQ_LEN_Q <= BLOCK_M:
        batch_id = gq               # 0..B-1
        q_block_id = tl.cast(0, tl.int64)
    else:
        ub = _upper_bound_prefix(q_blk_offsets, BATCH_SIZE + 1, LOG_BATCH_SIZE_P1, gq)  # 1..B
        batch_id = tl.cast(ub - 1, tl.int64)
        base = tl.load(q_blk_offsets + batch_id)  # prefix[batch]
        q_block_id = gq - base

    return head_id, batch_id, q_block_id

@triton.jit
def _split_core_range(total_tasks, pid, num_cores):
    used = tl.minimum(num_cores, total_tasks)

    split_next = total_tasks // used
    split_prev = split_next + 1
    rem = total_tasks % used

    in_used = pid < used
    in_prev = pid < rem

    start_prev = pid * split_prev
    start_next = rem * split_prev + (pid - rem) * split_next

    task_start = tl.where(in_used, tl.where(in_prev, start_prev, start_next), 0)
    task_cnt   = tl.where(in_used, tl.where(in_prev, split_prev, split_next), 0)
    task_end   = task_start + task_cnt
    return task_start, task_end, used

@triton.jit
def _do_qk_matmul(
    q_base, k_base,
    seq_len_q, seq_len_k,
    q_block_id, n_offset,
    stride_qm, stride_kn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_row_idx = q_block_id * BLOCK_M + offs_m
    q_row_mask = q_row_idx < seq_len_q

    q_ptrs = q_base + q_row_idx[:, None] * stride_qm + offs_d[None, :]

    q_sub = tl.load(q_ptrs, mask=q_row_mask[:, None], other=0.0)

    k_row_idx = n_offset + offs_n
    k_row_mask = k_row_idx < seq_len_k

    k_ptrs = k_base + k_row_idx[:, None] * stride_kn + offs_d[None, :]
    k_sub = tl.load(k_ptrs, mask=k_row_mask[:, None])
    s_tile = tl.dot(q_sub, tl.trans(k_sub))

    return s_tile

@triton.jit
def _do_vec_score(
    qk_ub,
    alpha, silu_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    seq_len_q, seq_len_k,
    q_block_id, n_offset,
    DATA_TYPE: tl.constexpr,
):
    s = qk_ub
    S = s * alpha
    s_act = S * tl.sigmoid(S)
    s_out = s_act * silu_scale
    s_out = s_out.to(DATA_TYPE)

    return s_out

@triton.jit
def _do_sv_matmul(
    s_l1, v_base, acc_base,
    out_base,
    seq_len_q, seq_len_k,
    q_block_id, n_offset,
    stride_vn, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    seq_type: tl.constexpr,
    OUT_DATA_TYPE: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, V_HEAD_DIM)

    q_row_idx = q_block_id * BLOCK_M + offs_m
    q_row_mask = q_row_idx < seq_len_q

    v_row_idx = n_offset + offs_n
    v_row_mask = v_row_idx < seq_len_k

    s = s_l1

    v_ptrs = v_base + v_row_idx[:, None] * stride_vn + offs_d[None, :]
    v = tl.load(v_ptrs, mask=v_row_mask[:, None], other=0.0)
    acc = tl.dot(s, v)

    if seq_type == 0:
        out_ptr = out_base + q_row_idx[:, None] * stride_om + offs_d[None, :]
        tl.store(out_ptr, acc.to(OUT_DATA_TYPE), mask=q_row_mask[:, None])
    elif seq_type == 1:
        out_ptr = out_base + q_row_idx[:, None] * stride_om + offs_d[None, :]
        tl.atomic_add(out_ptr, acc.to(OUT_DATA_TYPE), mask=q_row_mask[:, None])
    else:
        acc_ptr = acc_base + q_row_idx[:, None] * stride_om + offs_d[None, :]
        tl.atomic_add(acc_ptr, acc, mask=q_row_mask[:, None])

@triton.jit
def _do_cast_store(
    acc_base, out_base,
    seq_len_q, q_block_id,
    stride_om, stride_oh,
    BLOCK_M: tl.constexpr, V_HEAD_DIM: tl.constexpr,
    OUT_DATA_TYPE: tl.constexpr,
    MAX_ELEMS_PER_STEP: tl.constexpr,
):
    ROWS_PER_STEP: tl.constexpr = MAX_ELEMS_PER_STEP // V_HEAD_DIM

    base_row = q_block_id * BLOCK_M

    offs_d = tl.arange(0, V_HEAD_DIM)
    offs_step = tl.arange(0, ROWS_PER_STEP)
    for row_start in tl.range(0, BLOCK_M, ROWS_PER_STEP):
        offs_m = row_start + offs_step
        q_row_idx = base_row + offs_m
        row_mask = q_row_idx < seq_len_q

        acc_ptrs = acc_base + q_row_idx[:, None] * stride_om + offs_d[None, :]
        acc_val = tl.load(acc_ptrs, mask=row_mask[:, None])

        out_val = acc_val.to(OUT_DATA_TYPE)
        out_ptrs = out_base + q_row_idx[:, None] * stride_om + offs_d[None, :]
        tl.store(out_ptrs, out_val, mask=row_mask[:, None])

@triton.jit
def _hstu_attn_fwd(
    Q, K, V,
    seq_offsets_q, seq_offsets_k,
    q_blk_offsets,
    Acc,
    Out,
    stride_qm: tl.constexpr, stride_qh: tl.constexpr,
    stride_kn: tl.constexpr, stride_kh: tl.constexpr,
    stride_vn: tl.constexpr, stride_vh: tl.constexpr,
    stride_om: tl.constexpr, stride_oh: tl.constexpr,
    alpha, silu_scale,
    MAX_SEQ_LEN_Q, MAX_SEQ_LEN_K,
    DeltaSize,
    HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_NUM_Q: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    dtype: tl.constexpr,
    seq_type: tl.constexpr,
):
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    if MAX_SEQ_LEN_Q <= BLOCK_M:
        total_q_blocks = tl.cast(BATCH_SIZE, tl.int64)
    else:
        total_q_blocks = tl.load(q_blk_offsets + BATCH_SIZE).to(tl.int64)
    total_tasks = HEAD_NUM_Q * total_q_blocks
    task_start, task_end, used = _split_core_range(total_tasks, pid, num_cores)
    if pid >= used:
        return
    q_task_cnt = task_end - task_start
    if q_task_cnt <= 0:
        return
    LOG_BATCH_SIZE_P1 = int(math.log2(BATCH_SIZE + 1)) + 1

    # KV tile 上界（用 MAX_SEQ_LEN_K 做扁平化长度）
    kv_blk_num = tl.cdiv(MAX_SEQ_LEN_K, BLOCK_N)
    blk_cnt = q_task_cnt * kv_blk_num

    DATA_TYPE: tl.constexpr = tl.float16 if dtype == 0 else tl.bfloat16 if dtype == 1 else tl.float8e4nv
    OUT_DATA_TYPE: tl.constexpr = tl.bfloat16 if dtype == 1 else tl.float16

    for p in range(0, blk_cnt):
        q_task = task_start + (p // kv_blk_num)
        tile = p % kv_blk_num

        head_id, batch_id, q_block_id = _map_qtask_to_batch_qblk(
            q_task, total_q_blocks, q_blk_offsets, BATCH_SIZE, LOG_BATCH_SIZE_P1, MAX_SEQ_LEN_Q, BLOCK_M
        )
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
        acc_base = Acc + head_id * stride_oh + seq_start_q * stride_om
        out_base = Out + head_id * stride_oh + seq_start_q * stride_om

        num_tiles = tl.cdiv(seq_len_k, BLOCK_N)
        n_offset = tile * BLOCK_N

        qk_ub = _do_qk_matmul(
            q_base, k_base,
            seq_len_q, seq_len_k,
            q_block_id, n_offset,
            stride_qm, stride_kn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            HEAD_DIM=HEAD_DIM,
        )

        s_ub = _do_vec_score(
            qk_ub,
            alpha, silu_scale,
            BLOCK_M, BLOCK_N,
            seq_len_q, seq_len_k,
            q_block_id, n_offset,
            DATA_TYPE,
        )

        _do_sv_matmul(
            s_ub,
            v_base, acc_base,
            out_base,
            seq_len_q, seq_len_k,
            q_block_id, n_offset,
            stride_vn, stride_om,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            V_HEAD_DIM=V_HEAD_DIM,
            seq_type=seq_type,
            OUT_DATA_TYPE=OUT_DATA_TYPE,
        )

        # 长序列模式 最后一个k块后做cast
        if seq_type == 2 and tile == num_tiles - 1:
            acc_base = Acc + head_id * stride_oh + seq_start_q * stride_om
            out_base = Out + head_id * stride_oh + seq_start_q * stride_om
            CAST_MAX_ELEMS_PER_STEP: tl.constexpr = 72 * 128
            _do_cast_store(
                acc_base, out_base,
                seq_len_q, q_block_id,
                stride_om, stride_oh,
                BLOCK_M, V_HEAD_DIM,
                OUT_DATA_TYPE,
                CAST_MAX_ELEMS_PER_STEP,
            )


@functools.cache
def get_core_num():
    return driver.active.utils.get_device_properties("npu")["num_aicore"]


def triton_hstu_attention_fwd(
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
) -> torch.Tensor:
    BLOCK_M = 128
    BLOCK_N = 128
    device = q.device
    batch_size = len(seq_offsets_q) - 1
    total_tokens, head_num_q, head_dim = q.shape
    head_num_k = k.shape[1]
    _, _, v_head_dim = v.shape

    for config in TILE_CONFIGS:
        if config["condition"](max_seq_len_q, max_seq_len_k, max(head_dim, v_head_dim)):
            BLOCK_M = config["BLOCK_M"]
            BLOCK_N = config["BLOCK_N"]
            print("BLOCK_M:",BLOCK_M," BLOCK_N:",BLOCK_N)
            break

    k_block_count = (max_seq_len_k + BLOCK_N - 1) // BLOCK_N
    if max_seq_len_q <= BLOCK_M and k_block_count <= 1:
        seq_type = SeqType.BASIC_QKV
    # elif k_block_count <= 3: # 此模式不满足L2精度
    #     seq_type = SeqType.FEW_KV
    else:
        seq_type = SeqType.LONG_QKV

    assert head_num_q % head_num_k == 0, f"Query heads ({head_num_q}) must be divisible by KV heads ({head_num_k})"
    group_size = head_num_q // head_num_k
    out_dtype = torch.float16 if q.dtype == torch.float8_e4m3fn else q.dtype
    acc = torch.zeros((total_tokens, head_num_q, v_head_dim), dtype=torch.float32, device=device)
    out = torch.zeros((total_tokens, head_num_q, v_head_dim), dtype=out_dtype, device=device)
    type_mapper = {torch.float16: 0, torch.bfloat16: 1, torch.float8_e4m3fn: 2}
    dtype = type_mapper.get(q.dtype, 0)
    grid = (get_core_num(),)
    if seq_type == 0:
        q_blk_offsets = torch.arange(len(seq_offsets_q), dtype=torch.int32)
    else:
        # 计算每个 batch 的块数并做前缀和
        nb = (seq_offsets_q.diff() + BLOCK_M - 1) // BLOCK_M
        q_blk_offsets = nb.new_zeros(nb.numel() + 1)
        q_blk_offsets[1:] = nb.cumsum(0)
    q_blk_offsets = q_blk_offsets.to(device)
    _hstu_attn_fwd[grid](
        Q=q, K=k, V=v,
        Acc=acc,
        Out=out,
        seq_offsets_q=seq_offsets_q,
        seq_offsets_k=seq_offsets_k,
        q_blk_offsets=q_blk_offsets,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_om=out.stride(0),
        stride_oh=out.stride(1),
        alpha=alpha,
        silu_scale=silu_scale,
        MAX_SEQ_LEN_Q=max_seq_len_q,
        MAX_SEQ_LEN_K=max_seq_len_k,
        DeltaSize=0,
        HEAD_DIM=head_dim,
        V_HEAD_DIM=v_head_dim,
        GROUP_SIZE=group_size,
        HEAD_NUM_Q=head_num_q,
        BATCH_SIZE=batch_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        dtype=dtype,
        seq_type=seq_type,
        enable_mixed_cv=True,
        enable_auto_bind_sub_block=True,
        enable_flatten=False,
        #sync_solver=True,
        #set_workspace_multibuffer=2,
    )
    return out
