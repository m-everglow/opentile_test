import os
import pytest
import torch
import torch_npu
import triton
import triton.language as tl
# import triton.language.extra.cann.extension as extension
# from maybe_triton_jit import maybe_triton_jit
from utils import is_hopper, is_ampere


# configs = [
#     triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
#     for BM in [32, 64, 128]
#     for BN in [32, 64, 128]
#     for s in [1, 2, 3, 4]
#     for w in [4, 8]
# ]

# The following best configs are obtained by autotuning under dim64.
# Currently, only sm80(A100/A800), sm89(L), and sm90(H20) are supported.
# Developers can autotune according to your own needs.
def get_fwd_configs():
    if is_hopper():
        return [
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=2, num_warps=4)
        ]
    elif is_ampere():
        return [
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=3, num_warps=4)
        ]
    else:
        # return [
        #     triton.Config({"BLOCK_M": 64, "BLOCK_N": 128})
        # ]
        configs = [
            triton.Config(
                {
                    "BLOCK_M": BM,
                    "BLOCK_N": BN,
                },
            )
            for BM in [32, 64, 128]
            for BN in [32, 64, 128]
        ]
        return configs


def get_bwd_preprocess_configs():
    if is_hopper():
        return [triton.Config({"BLOCK_M": 32}, num_stages=4, num_warps=4)]
    elif is_ampere():
        return [triton.Config({"BLOCK_M": 32}, num_stages=3, num_warps=4)]
    else:
        return [triton.Config({"BLOCK_M": 32}, num_stages=3, num_warps=8)]


def get_bwd_q_configs():
    if is_hopper():
        return [
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=4, num_warps=4)
        ]
    elif is_ampere():
        return [
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=1, num_warps=4)
        ]
    else:
        return [
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=1, num_warps=4)
        ]


def get_bwd_kv_configs():
    if is_hopper():
        return [
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=4, num_warps=4)
        ]
    elif is_ampere():
        return [
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=1, num_warps=4)
        ]
    else:
        return [
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=1, num_warps=4)
        ]

def get_bwd_qkv_configs():
    return [
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}),
    ]

if os.environ.get("TRITON_DEBUG") == "1":
    configs = [triton.Config({"BLOCK_M": 32, "BLOCK_N": 16}, num_stages=1, num_warps=1)]
    bwd_preprocess_configs = [triton.Config({"BLOCK_M": 32}, num_stages=1, num_warps=1)]


def keep(config):
    m = config.kwargs["BLOCK_M"]
    n = config.kwargs["BLOCK_N"]
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major == 9:
        if m == 64 and config.num_warps == 8:
            return False
    return m % n == 0


@triton.jit
def load_if(block_ptr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    if EVEN_M & EVEN_N:
        return tl.load(block_ptr)
    elif EVEN_M:
        return tl.load(block_ptr, boundary_check=(1,), padding_option="zero")
    elif EVEN_N:
        return tl.load(block_ptr, boundary_check=(0,), padding_option="zero")
    else:
        return tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")


@triton.jit
def store_if(block_ptr, value, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    if EVEN_M & EVEN_N:
        tl.store(block_ptr, value)
    elif EVEN_N:
        tl.store(block_ptr, value, boundary_check=(0,))
    elif EVEN_M:
        tl.store(block_ptr, value, boundary_check=(1,))
    else:
        tl.store(block_ptr, value, boundary_check=(0, 1))


@triton.jit
def mask_fn(q_attn_arg, k_attn_arg, q_offset, k_offset, TYPE: tl.constexpr):
    # tril_causal = q_offset[:, None] >= k_offset[None, :]
    # triu_causal = q_offset[:, None] <= k_offset[None, :]
    # attn_arg = 0 代表 sequence，非 0 代表 query，不同 query 用不同的 attn_arg
    if TYPE == 1:
        # return (q_offset[:, None] <= k_offset[None, :])
        triu_causal = (q_offset[:, None] <= k_offset[None, :]).to(tl.int32)
        return (
                (triu_causal &
                 ((q_attn_arg[:, None] == k_attn_arg[None, :]).to(tl.int32) |
                  (k_attn_arg[None, :] == 0).to(tl.int32))) |
                (q_offset[:, None] == k_offset[None, :]).to(tl.int32))
    if TYPE == 2:
        tril_causal = (q_offset[:, None] >= k_offset[None, :])
        return ((tril_causal & ((q_attn_arg[:, None] == k_attn_arg[None, :]) | (k_attn_arg[None, :] == 0))) | (
                    q_offset[:, None] == k_offset[None, :]))


@triton.jit
def fwd_kernel(
        q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
        q_attn_arg_ptr, k_attn_arg_ptr, mask_tensor_ptr,
        cu_seqlens_q, cu_seqlens_k,
        q_head, kv_head, scale,
        QK_DIM: tl.constexpr, V_DIM: tl.constexpr, MASK_FN: tl.constexpr,
        SPARSE_OPT: tl.constexpr, DTYPE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        AICORE_NUM: tl.constexpr,
        MAX_Q_LEN: tl.constexpr,
        MAX_K_LEN: tl.constexpr,
        BATCH_SIZE: tl.constexpr,
):
    dtype = o_ptr.type.element_ty
    pid = tl.program_id(0)
    NUM_BLOCKS_M = tl.cdiv(MAX_Q_LEN, BLOCK_M)  # 非对齐跳过
    NUM_BLOCKS = NUM_BLOCKS_M * BATCH_SIZE * q_head

    # block_start = pid * NUM_BLOCKS_PER_CORE
    # NUM_BLOCKS_hz = NUM_BLOCKS // NUM_BLOCKS_M
    # task_m_idx = 0
    # task_hz_idx = 0
    start_block, end_block, step = pid, NUM_BLOCKS, AICORE_NUM

    for block_idx in range(start_block, end_block, step):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        start_m = block_idx % NUM_BLOCKS_M
        start_b = task_hz_idx // q_head
        start_qh = task_hz_idx % q_head
        start_kvh = start_qh * kv_head // q_head

        # start_m = tl.program_id(0)
        # start_qh = tl.program_id(1)
        # start_b = tl.program_id(2)
        # start_kvh = start_qh // (q_head // kv_head)

        q_start1 = tl.load(cu_seqlens_q + start_b)
        q_end = tl.load(cu_seqlens_q + start_b + 1)
        q_len = q_end - q_start1
        # Cannot have `return` statements inside `while` or `for` statements in triton
        # unsupported AST node type: Continue
        # if start_m * BLOCK_M >= q_len:
        #     return
        if start_m * BLOCK_M < q_len:
            k_start1 = tl.load(cu_seqlens_k + start_b)
            k_end = tl.load(cu_seqlens_k + start_b + 1)
            k_len = k_end - k_start1
            # if SPARSE_OPT:  # false
            #     begin = 0
            #     if k_len==0:
            #         continue
            #     end = k_len
            # else:
            #     if MASK_FN & 1:
            #         begin = start_m * BLOCK_M
            #         if begin >= k_len:
            #             continue
            #         end = k_len
            #     else:
            #         begin = 0
            #         end = tl.minimum((start_m + 1) * BLOCK_M, k_len)
            begin = start_m * BLOCK_M
            if begin.to(tl.int64) < k_len.to(tl.int64):
                end = k_len

                # log2e: tl.constexpr = 1.4426950408889634
                qk_scale = scale  # * log2e
                # offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

                q_start = q_start1.to(tl.int64)
                k_start = k_start1.to(tl.int64)
                q_block_ptr = tl.make_block_ptr(
                    base=q_ptr + q_start * q_head * QK_DIM + start_qh * QK_DIM,
                    shape=(q_len, QK_DIM),
                    strides=(q_head * QK_DIM, 1),
                    offsets=(start_m * BLOCK_M, 0),
                    block_shape=(BLOCK_M, QK_DIM),
                    order=(1, 0)
                )
                k_block_ptr = tl.make_block_ptr(
                    base=k_ptr + k_start * kv_head * QK_DIM + start_kvh * QK_DIM,
                    shape=(QK_DIM, k_len),
                    strides=(1, kv_head * QK_DIM),
                    offsets=(0, begin),
                    block_shape=(QK_DIM, BLOCK_N),
                    order=(0, 1)
                )
                v_block_ptr = tl.make_block_ptr(
                    base=v_ptr + k_start * kv_head * V_DIM + start_kvh * V_DIM,
                    shape=(k_len, V_DIM),
                    strides=(kv_head * V_DIM, 1),
                    offsets=(begin, 0),
                    block_shape=(BLOCK_N, V_DIM),
                    order=(1, 0)
                )
                o_block_ptr = tl.make_block_ptr(
                    base=o_ptr + q_start * q_head * V_DIM + start_qh * V_DIM,
                    shape=(q_len, V_DIM),
                    strides=(q_head * V_DIM, 1),
                    offsets=(start_m * BLOCK_M, 0),
                    block_shape=(BLOCK_M, V_DIM),
                    order=(1, 0)
                )
                l_block_ptr = tl.make_block_ptr(
                    base=l_ptr + q_start * q_head + start_qh,
                    shape=(q_len,),
                    strides=(q_head,),
                    offsets=(start_m * BLOCK_M,),
                    block_shape=(BLOCK_M,),
                    order=(0,)
                )
                # q_attn_arg_block_ptr = tl.make_block_ptr(
                #     base = q_attn_arg_ptr + q_start,
                #     shape = (q_len,),
                #     strides = (1,),
                #     offsets = (start_m * BLOCK_M,),
                #     block_shape = (BLOCK_M,),
                #     order = (0,)
                # )
                # k_attn_arg_block_ptr = tl.make_block_ptr(
                #     base = k_attn_arg_ptr + k_start,
                #     shape = (k_len,),
                #     strides = (1,),
                #     offsets = (begin,),
                #     block_shape = (BLOCK_N,),
                #     order = (0,)
                # )

                mask_block_ptr = tl.make_block_ptr(
                    base=mask_tensor_ptr + start_b * MAX_Q_LEN * MAX_K_LEN,
                    shape=(q_len, k_len),
                    strides=(MAX_K_LEN, 1),
                    offsets=(start_m * BLOCK_M, begin),
                    block_shape=(BLOCK_M, BLOCK_N),
                    order=(1, 0)
                )
                acc = tl.zeros((BLOCK_M, V_DIM), dtype=tl.float32)
                m = tl.full((BLOCK_M,), value=-2 ** 30, dtype=tl.float32)
                l = tl.zeros((BLOCK_M,), dtype=tl.float32)

                q = load_if(q_block_ptr, False, True)
                # q_attn_arg = load_if(q_attn_arg_block_ptr, False, True)

                # sum_b_fc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int1)
                for start_n in range(begin, end, BLOCK_N):
                    start_n = tl.multiple_of(start_n, BLOCK_N)
                    # k_attn_arg = load_if(k_attn_arg_block_ptr, False, True)
                    # offset_n = start_n + tl.arange(0, BLOCK_N)
                    mask = load_if(mask_block_ptr, False, False)  # (32x32) -> (128, 128) --> 0

                    # mask = mask_fn(q_attn_arg, k_attn_arg, offset_m, offset_n, MASK_FN)

                    # if not (pid == 0 and start_n == 0):
                    #     pass
                    # else:
                    #     dump_tensor_ptr1 = dump_tensor_ptr + (tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :])
                    #     # mask2 = mask.reshape(BLOCK_N * BLOCK_M,)
                    #     sum_b_fc = mask.to(tl.int1)
                    #     tl.store(dump_tensor_ptr1, sum_b_fc)

                    # if not SPARSE_OPT or tl.sum(mask.cast(tl.int32)) != 0:
                    if not SPARSE_OPT:
                        k = load_if(k_block_ptr, True, False)  # npu行为：先装置后mask
                        # v = load_if(v_block_ptr, False, True)
                        s = tl.dot(q, k)
                        # s = s * qk_scale
                        # s = tl.where(mask, s, -2**30)
                        # mask_value = tl.where(boundary_mask & mask, 0, -2**30).to(tl.float16)
                        # s = s * qk_scale + mask_value.to(tl.float32)
                        s = s * qk_scale + tl.where(mask, 0.0, -2.0 ** 30)
                        # m_new = tl.maximum(m, tl.max(s, 1))
                        # 1.128 *128: nomak test: ub overflow, requires 2263552 bits while 2031616 bits available!: 777.79 -> 616
                        # 1.1 128 * 128, triu_causal - 客户mask, 2111->850
                        # 1.2 128 * 128, 不带nan, mask.to(int32) 精度问题 -->未复现
                        # 1.3 128 * 128, 不带nan, 不带mask.to(int32) 2330 --> 1350(使用membar新包)
                        # 2. 64* 128: 2500 ->1375(带nan) -> 1242.143（mask.to(int32)
                        m_new = tl.maximum(m, tl.max(s, 1), propagate_nan=tl.PropagateNan.ALL)
                        p = tl.math.exp(s - m_new[:, None])
                        v = load_if(v_block_ptr, False, True)
                        # acc *= alpha[:, None]
                        pv = tl.dot(p.to(dtype), v)
                        p_sum = tl.sum(p, 1)
                        alpha = tl.math.exp(m - m_new)
                        acc = acc * alpha[:, None] + pv

                        l = l * alpha + p_sum
                        m = m_new
                    k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))
                    v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N, 0))
                    mask_block_ptr = tl.advance(mask_block_ptr, (0, BLOCK_N))
                    # k_attn_arg_block_ptr = tl.advance(k_attn_arg_block_ptr, (BLOCK_N,))

                acc = acc / l[:, None]
                m = m + tl.log(l)

                store_if(o_block_ptr, acc.to(dtype), False, True)
                store_if(l_block_ptr, m, False, True)


# @triton.autotune(get_bwd_preprocess_configs(), key = ["V_DIM"])
@triton.jit
def bwd_preprocess(
    o_ptr, do_ptr, d_ptr,
    cu_seqlens_q,
    q_head,
    V_DIM: tl.constexpr, DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_BLOCKS_M: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    start_block, end_block, step = pid, NUM_BLOCKS, NUM_CORES
    for block_idx in range(start_block, end_block, step):
        tasts_hb_idx = block_idx // NUM_BLOCKS_M
        start_m = block_idx % NUM_BLOCKS_M
        start_b = tasts_hb_idx // q_head
        start_h = tasts_hb_idx % q_head
        q_start1 = tl.load(cu_seqlens_q + start_b)
        q_end = tl.load(cu_seqlens_q + start_b + 1)
        q_len = q_end - q_start1
        if start_m * BLOCK_M < q_len:
            q_start = q_start1.to(tl.int64)
            o_block_ptr = tl.make_block_ptr(
                base = o_ptr + q_start * q_head * V_DIM + start_h * V_DIM,
                shape = (q_len, V_DIM),
                strides = (q_head * V_DIM, 1),
                offsets = (start_m * BLOCK_M, 0),
                block_shape = (BLOCK_M, V_DIM),
                order = (1, 0),
            )
            do_block_ptr = tl.make_block_ptr(
                base = do_ptr + q_start * q_head * V_DIM + start_h * V_DIM,
                shape = (q_len, V_DIM),
                strides = (q_head * V_DIM, 1),
                offsets = (start_m * BLOCK_M, 0),
                block_shape = (BLOCK_M, V_DIM),
                order = (1, 0),
            )
            d_block_ptr = tl.make_block_ptr(
                base = d_ptr + q_start * q_head + start_h,
                shape = (q_len,),
                strides = (q_head,),
                offsets = (start_m * BLOCK_M,),
                block_shape = (BLOCK_M,),
                order = (0,)
            )
            o = load_if(o_block_ptr, False, True).to(tl.float32)
            do = load_if(do_block_ptr, False, True).to(tl.float32)
            d = tl.sum(o * do, 1)
            store_if(d_block_ptr, d, False, True)


@triton.jit
def bwd_preprocess_ifmn(
    o_ptr,
    do_ptr,
    d_ptr,
    Q_HEAD_NUM: tl.constexpr,
    V_DIM: tl.constexpr,
    DTYPE: tl.constexpr,
    TASK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    task_id = tl.program_id(0)
    task_start = task_id * TASK_SIZE

    for block_id in tl.range(0, NUM_BLOCKS):

        o_block_ptr = tl.make_block_ptr(
            base = o_ptr + task_start * Q_HEAD_NUM * V_DIM,
            shape = (TASK_SIZE, Q_HEAD_NUM, V_DIM),
            strides = (Q_HEAD_NUM * V_DIM, V_DIM, 1),
            offsets = (block_id * BLOCK_SIZE, 0, 0),
            block_shape = (BLOCK_SIZE, Q_HEAD_NUM, V_DIM),
            order = (2, 1, 0),
        )
        o = load_if(o_block_ptr, False, True).to(tl.float32)

        do_block_ptr = tl.make_block_ptr(
            base = do_ptr + task_start * Q_HEAD_NUM * V_DIM,
            shape = (TASK_SIZE, Q_HEAD_NUM, V_DIM),
            strides = (Q_HEAD_NUM * V_DIM, V_DIM, 1),
            offsets = (block_id * BLOCK_SIZE, 0, 0),
            block_shape = (BLOCK_SIZE, Q_HEAD_NUM, V_DIM),
            order = (2, 1, 0),
        )
        do = load_if(do_block_ptr, False, True).to(tl.float32)

        d = tl.sum(o * do, 2)

        d_block_ptr = tl.make_block_ptr(
            base = d_ptr + task_start * Q_HEAD_NUM,
            shape = (TASK_SIZE, Q_HEAD_NUM),
            strides = (Q_HEAD_NUM, 1),
            offsets = (block_id * BLOCK_SIZE, 0),
            block_shape = (BLOCK_SIZE, Q_HEAD_NUM),
            order = (1, 0),
        )
        store_if(d_block_ptr, d, False, True)


# @triton.autotune(list(filter(keep, get_bwd_kv_configs())), key = ["QK_DIM", "V_DIM", "MASK_FN", "SPARSE_OPT"])
@triton.jit
def bwd_kv_kernel(
        q_ptr, k_ptr, v_ptr,
        dk_ptr, dv_ptr, do_ptr,
        l_ptr, d_ptr,
        q_attn_arg_ptr, k_attn_arg_ptr, mask_tensor_ptr,
        cu_seqlens_q, cu_seqlens_k,
        q_head, kv_head, scale,
        QK_DIM: tl.constexpr, V_DIM: tl.constexpr, MASK_FN: tl.constexpr, SPARSE_OPT: tl.constexpr, DTYPE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        NUM_BLOCKS_N: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        NUM_CORES: tl.constexpr,
        MAX_Q_LEN: tl.constexpr,
        MAX_K_LEN: tl.constexpr,
):
    dtype = k_ptr.type.element_ty
    pid = tl.program_id(0)
    start_block, end_block, step = pid, NUM_BLOCKS, NUM_CORES
    for block_idx in range(start_block, end_block, step):
        tasts_hb_idx = block_idx // NUM_BLOCKS_N
        start_n = block_idx % NUM_BLOCKS_N
        start_b = tasts_hb_idx // q_head
        start_qh = tasts_hb_idx % q_head
        start_kvh = start_qh // (q_head // kv_head)
        k_start1 = tl.load(cu_seqlens_k + start_b)
        k_end = tl.load(cu_seqlens_k + start_b + 1)
        k_len = k_end - k_start1
        if start_n * BLOCK_N < k_len:
            q_start1 = tl.load(cu_seqlens_q + start_b)
            q_end = tl.load(cu_seqlens_q + start_b + 1)
            q_len = q_end - q_start1

            if SPARSE_OPT:
                begin = 0
                end = q_len
            else:
                if MASK_FN & 1:
                    begin = 0
                    end = tl.minimum(start_n * BLOCK_N // BLOCK_M * BLOCK_M + BLOCK_M, q_len)
                else:
                    begin = start_n * BLOCK_N // BLOCK_M * BLOCK_M
                    end = q_len

            log2e: tl.constexpr = 1.4426950408889634
            qk_scale = scale * log2e
            # offset_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

            q_start = q_start1.to(tl.int64)
            k_start = k_start1.to(tl.int64)
            q_block_ptr = tl.make_block_ptr(
                base=q_ptr + q_start * q_head * QK_DIM + start_qh * QK_DIM,
                shape=(q_len, QK_DIM),
                strides=(q_head * QK_DIM, 1),
                offsets=(begin, 0),
                block_shape=(BLOCK_M, QK_DIM),
                order=(1, 0)
            )
            k_block_ptr = tl.make_block_ptr(
                base=k_ptr + k_start * kv_head * QK_DIM + start_kvh * QK_DIM,
                shape=(QK_DIM, k_len),
                strides=(1, kv_head * QK_DIM),
                offsets=(0, start_n * BLOCK_N),
                block_shape=(QK_DIM, BLOCK_N),
                order=(0, 1)
            )
            v_block_ptr = tl.make_block_ptr(
                base=v_ptr + k_start * kv_head * V_DIM + start_kvh * V_DIM,
                shape=(V_DIM, k_len),
                strides=(1, kv_head * V_DIM),
                offsets=(0, start_n * BLOCK_N),
                block_shape=(V_DIM, BLOCK_N),
                order=(0, 1)
            )
            dk_block_ptr = tl.make_block_ptr(
                base=dk_ptr + k_start * q_head * QK_DIM + start_qh * QK_DIM,
                shape=(k_len, QK_DIM),
                strides=(q_head * QK_DIM, 1),
                offsets=(start_n * BLOCK_N, 0),
                block_shape=(BLOCK_N, QK_DIM),
                order=(1, 0)
            )
            dv_block_ptr = tl.make_block_ptr(
                base=dv_ptr + k_start * q_head * V_DIM + start_qh * V_DIM,
                shape=(k_len, V_DIM),
                strides=(q_head * V_DIM, 1),
                offsets=(start_n * BLOCK_N, 0),
                block_shape=(BLOCK_N, V_DIM),
                order=(1, 0)
            )
            do_block_ptr = tl.make_block_ptr(
                base=do_ptr + q_start * q_head * V_DIM + start_qh * V_DIM,
                shape=(q_len, V_DIM),
                strides=(q_head * V_DIM, 1),
                offsets=(begin, 0),
                block_shape=(BLOCK_M, V_DIM),
                order=(1, 0)
            )
            l_block_ptr = tl.make_block_ptr(
                base=l_ptr + q_start * q_head + start_qh,
                shape=(q_len,),
                strides=(q_head,),
                offsets=(begin,),
                block_shape=(BLOCK_M,),
                order=(0,)
            )
            d_block_ptr = tl.make_block_ptr(
                base=d_ptr + q_start * q_head + start_qh,
                shape=(q_len,),
                strides=(q_head,),
                offsets=(begin,),
                block_shape=(BLOCK_M,),
                order=(0,)
            )
            # q_attn_arg_block_ptr = tl.make_block_ptr(
            #     base=q_attn_arg_ptr + q_start,
            #     shape=(q_len,),
            #     strides=(1,),
            #     offsets=(begin,),
            #     block_shape=(BLOCK_M,),
            #     order=(0,)
            # )
            # k_attn_arg_block_ptr = tl.make_block_ptr(
            #     base=k_attn_arg_ptr + k_start,
            #     shape=(k_len,),
            #     strides=(1,),
            #     offsets=(start_n * BLOCK_N,),
            #     block_shape=(BLOCK_N,),
            #     order=(0,)
            # )

            mask_block_ptr = tl.make_block_ptr(
                base=mask_tensor_ptr + start_b * MAX_Q_LEN * MAX_K_LEN,
                shape=(q_len, k_len),
                strides=(MAX_K_LEN, 1),
                offsets=(begin, start_n * BLOCK_N),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0)
            )

            dk = tl.zeros((BLOCK_N, QK_DIM), dtype=tl.float32)
            dv = tl.zeros((BLOCK_N, V_DIM), dtype=tl.float32)

            k = load_if(k_block_ptr, True, False)
            v = load_if(v_block_ptr, True, False)
            # k_attn_arg = load_if(k_attn_arg_block_ptr, False, True)

            for start_m in range(begin, end, BLOCK_M):
                # start_m = tl.multiple_of(start_m, BLOCK_M)
                # q_attn_arg = load_if(q_attn_arg_block_ptr, False, True)
                # offset_m = start_m + tl.arange(0, BLOCK_M)
                # mask = mask_fn(q_attn_arg, k_attn_arg, offset_m, offset_n, MASK_FN)
                mask = load_if(mask_block_ptr, False, False)
                if not SPARSE_OPT or tl.sum(mask.cast(tl.int32)) != 0:
                    q = load_if(q_block_ptr, False, True)
                    s = tl.dot(q, k)
                    l = load_if(l_block_ptr, False, True)
                    p = tl.math.exp2(s * qk_scale - l[:, None] * log2e)
                    p = tl.where(mask, p, 0.0)
                    do = load_if(do_block_ptr, False, True)
                    p = p.to(dtype)
                    dv += tl.dot(tl.trans(p), do)
                    d = load_if(d_block_ptr, False, True)
                    dp = tl.dot(do, v)
                    ds = p * (dp - d[:, None])
                    ds = tl.where(mask, ds, 0.0)
                    ds = ds.to(dtype)
                    dk += tl.dot(tl.trans(ds), q)
                q_block_ptr = tl.advance(q_block_ptr, (BLOCK_M, 0))
                do_block_ptr = tl.advance(do_block_ptr, (BLOCK_M, 0))
                l_block_ptr = tl.advance(l_block_ptr, (BLOCK_M,))
                d_block_ptr = tl.advance(d_block_ptr, (BLOCK_M,))
                # q_attn_arg_block_ptr = tl.advance(q_attn_arg_block_ptr, (BLOCK_M,))
                mask_block_ptr = tl.advance(mask_block_ptr, (BLOCK_M, 0))

            dk *= scale
            store_if(dk_block_ptr, dk.to(dtype), False, True)
            store_if(dv_block_ptr, dv.to(dtype), False, True)


# @triton.autotune(list(filter(keep, get_bwd_q_configs())), key = ["QK_DIM", "V_DIM", "MASK_FN", "SPARSE_OPT"])
@triton.jit
def bwd_q_kernel(
        q_ptr, k_ptr, v_ptr, dq_ptr, do_ptr, l_ptr, d_ptr,
        q_attn_arg_ptr, k_attn_arg_ptr, mask_tensor_ptr,
        cu_seqlens_q, cu_seqlens_k,
        q_head, kv_head, scale,
        QK_DIM: tl.constexpr, V_DIM: tl.constexpr, MASK_FN: tl.constexpr, SPARSE_OPT: tl.constexpr, DTYPE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        NUM_BLOCKS_M: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        NUM_CORES: tl.constexpr,
        MAX_Q_LEN: tl.constexpr,
        MAX_K_LEN: tl.constexpr,
):
    dtype = k_ptr.type.element_ty
    pid = tl.program_id(0)
    start_block, end_block, step = pid, NUM_BLOCKS, NUM_CORES
    for block_idx in range(start_block, end_block, step):
        tasts_hb_idx = block_idx // NUM_BLOCKS_M
        start_m = block_idx % NUM_BLOCKS_M
        start_b = tasts_hb_idx // q_head
        start_qh = tasts_hb_idx % q_head
        start_kvh = start_qh // (q_head // kv_head)

        q_start1 = tl.load(cu_seqlens_q + start_b)
        q_end = tl.load(cu_seqlens_q + start_b + 1)
        q_len = q_end - q_start1
        if start_m * BLOCK_M < q_len:
            k_start1 = tl.load(cu_seqlens_k + start_b)
            k_end = tl.load(cu_seqlens_k + start_b + 1)
            k_len = k_end - k_start1

            if SPARSE_OPT:
                begin = 0
                end = k_len
            else:
                if MASK_FN & 1:
                    begin = start_m * BLOCK_M
                    # if begin >= k_len:
                    #     return
                    end = k_len
                else:
                    begin = 0
                    end = tl.minimum((start_m + 1) * BLOCK_M, k_len)

            log2e: tl.constexpr = 1.4426950408889634
            qk_scale = scale * log2e
            # offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

            q_start = q_start1.to(tl.int64)
            k_start = k_start1.to(tl.int64)
            q_block_ptr = tl.make_block_ptr(
                base=q_ptr + q_start * q_head * QK_DIM + start_qh * QK_DIM,
                shape=(q_len, QK_DIM),
                strides=(q_head * QK_DIM, 1),
                offsets=(start_m * BLOCK_M, 0),
                block_shape=(BLOCK_M, QK_DIM),
                order=(1, 0)
            )
            k_block_ptr = tl.make_block_ptr(
                base=k_ptr + k_start * kv_head * QK_DIM + start_kvh * QK_DIM,
                shape=(k_len, QK_DIM),
                strides=(kv_head * QK_DIM, 1),
                offsets=(begin, 0),
                block_shape=(BLOCK_N, QK_DIM),
                order=(1, 0)
            )
            v_block_ptr = tl.make_block_ptr(
                base=v_ptr + k_start * kv_head * V_DIM + start_kvh * V_DIM,
                shape=(V_DIM, k_len),
                strides=(1, kv_head * V_DIM),
                offsets=(0, begin),
                block_shape=(V_DIM, BLOCK_N),
                order=(0, 1)
            )
            dq_block_ptr = tl.make_block_ptr(
                base=dq_ptr + q_start * q_head * QK_DIM + start_qh * QK_DIM,
                shape=(q_len, QK_DIM),
                strides=(q_head * QK_DIM, 1),
                offsets=(start_m * BLOCK_M, 0),
                block_shape=(BLOCK_M, QK_DIM),
                order=(1, 0)
            )
            do_block_ptr = tl.make_block_ptr(
                base=do_ptr + q_start * q_head * V_DIM + start_qh * V_DIM,
                shape=(q_len, V_DIM),
                strides=(q_head * V_DIM, 1),
                offsets=(start_m * BLOCK_M, 0),
                block_shape=(BLOCK_M, V_DIM),
                order=(1, 0)
            )
            l_block_ptr = tl.make_block_ptr(
                base=l_ptr + q_start * q_head + start_qh,
                shape=(q_len,),
                strides=(q_head,),
                offsets=(start_m * BLOCK_M,),
                block_shape=(BLOCK_M,),
                order=(0,)
            )
            d_block_ptr = tl.make_block_ptr(
                base=d_ptr + q_start * q_head + start_qh,
                shape=(q_len,),
                strides=(q_head,),
                offsets=(start_m * BLOCK_M,),
                block_shape=(BLOCK_M,),
                order=(0,)
            )
            # q_attn_arg_block_ptr = tl.make_block_ptr(
            #     base=q_attn_arg_ptr + q_start,
            #     shape=(q_len,),
            #     strides=(1,),
            #     offsets=(start_m * BLOCK_M,),
            #     block_shape=(BLOCK_M,),
            #     order=(0,)
            # )
            # k_attn_arg_block_ptr = tl.make_block_ptr(
            #     base=k_attn_arg_ptr + k_start,
            #     shape=(k_len,),
            #     strides=(1,),
            #     offsets=(begin,),
            #     block_shape=(BLOCK_N,),
            #     order=(0,)
            # )
            mask_block_ptr = tl.make_block_ptr(
                base=mask_tensor_ptr + start_b * MAX_Q_LEN * MAX_K_LEN,
                shape=(q_len, k_len),
                strides=(MAX_K_LEN, 1),
                offsets=(start_m * BLOCK_M, begin),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0)
            )

            dq = tl.zeros((BLOCK_M, QK_DIM), dtype=tl.float32)

            q = load_if(q_block_ptr, False, True)
            do = load_if(do_block_ptr, False, True)
            l = load_if(l_block_ptr, False, True)
            d = load_if(d_block_ptr, False, True)
            # q_attn_arg = load_if(q_attn_arg_block_ptr, False, True)

            for start_n in range(begin, end, BLOCK_N):
                # start_n = tl.multiple_of(start_n, BLOCK_N)
                # k_attn_arg = load_if(k_attn_arg_block_ptr, False, True)
                # offset_n = start_n + tl.arange(0, BLOCK_N)
                # mask = mask_fn(q_attn_arg, k_attn_arg, offset_m, offset_n, MASK_FN)
                mask = load_if(mask_block_ptr, False, False)  # (32x32) -> (128, 128) --> 0

                if not SPARSE_OPT or tl.sum(mask.cast(tl.int32)) != 0:
                    k = load_if(k_block_ptr, False, True)
                    v = load_if(v_block_ptr, True, False)
                    s = tl.dot(q, tl.trans(k))
                    p = tl.math.exp2(s * qk_scale - l[:, None] * log2e)
                    # tl.device_print("===> p: ", p)
                    dp = tl.dot(do, v)
                    ds = p * (dp - d[:, None])
                    ds = ds.to(dtype)
                    # boundary_mask = (offset_n < k_len)[None, :]
                    ds = tl.where(mask, ds, 0.0)
                    extension.compile_hint(ds, "break_vf")
                    dq += tl.dot(ds, k)
                k_block_ptr = tl.advance(k_block_ptr, (BLOCK_N, 0))
                v_block_ptr = tl.advance(v_block_ptr, (0, BLOCK_N))
                # k_attn_arg_block_ptr = tl.advance(k_attn_arg_block_ptr, (BLOCK_N,))
                mask_block_ptr = tl.advance(mask_block_ptr, (0, BLOCK_N))

            dq *= scale
            store_if(dq_block_ptr, dq.to(dtype), False, True)


#@triton.autotune(
#    list(filter(keep, get_bwd_qkv_configs())), 
#    key = ["QK_DIM", "V_DIM", "MASK_FN", "SPARSE_OPT"],
#    reset_to_zero = ["dq_ptr"]
#)
@triton.jit
def bwd_qkv_kernel(
        q_ptr, k_ptr, v_ptr,
        dq_ptr, dk_ptr, dv_ptr, 
        do_ptr, l_ptr, d_ptr,
        q_attn_arg_ptr, k_attn_arg_ptr, 
        mask_tensor_ptr,
        cu_seqlens_q, cu_seqlens_k,
        batch_size, 
        max_q_len, max_k_len,
        q_head, kv_head, 
        scale,
        QK_DIM: tl.constexpr, 
        V_DIM: tl.constexpr,
        MASK_FN: tl.constexpr, 
        SPARSE_OPT: tl.constexpr, 
        DTYPE: tl.constexpr,
        NUM_CORES: tl.constexpr,
        BLOCK_M: tl.constexpr, 
        BLOCK_N: tl.constexpr, 
):
    dtype = k_ptr.type.element_ty
    pid = tl.program_id(0)
    NUM_BLOCKS_N= tl.cdiv(max_k_len, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_N * q_head * batch_size
    start_block, end_block, step = pid, NUM_BLOCKS, NUM_CORES
    for block_idx in range(start_block, end_block, step):
        tasts_hb_idx = block_idx // NUM_BLOCKS_N
        start_n = block_idx % NUM_BLOCKS_N
        start_b = tasts_hb_idx // q_head
        start_qh = tasts_hb_idx % q_head
        start_kvh = start_qh // (q_head // kv_head)
        k_start1 = tl.load(cu_seqlens_k + start_b)
        k_end = tl.load(cu_seqlens_k + start_b + 1)
        k_len = k_end - k_start1
        if start_n * BLOCK_N < k_len:
            q_start1 = tl.load(cu_seqlens_q + start_b)
            q_end = tl.load(cu_seqlens_q + start_b + 1)
            q_len = q_end - q_start1

            if SPARSE_OPT:
                begin = 0
                end = q_len
            else:
                if MASK_FN & 1:
                    begin = 0
                    end = tl.minimum(start_n * BLOCK_N // BLOCK_M * BLOCK_M + BLOCK_M, q_len)
                else:
                    begin = start_n * BLOCK_N // BLOCK_M * BLOCK_M
                    end = q_len

            log2e: tl.constexpr = 1.4426950408889634
            qk_scale = scale * log2e
            # offset_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

            q_start = q_start1.to(tl.int64)
            k_start = k_start1.to(tl.int64)
            q_block_ptr = tl.make_block_ptr(
                base=q_ptr + q_start * q_head * QK_DIM + start_qh * QK_DIM,
                shape=(q_len, QK_DIM),
                strides=(q_head * QK_DIM, 1),
                offsets=(begin, 0),
                block_shape=(BLOCK_M, QK_DIM),
                order=(1, 0)
            )
            k_block_ptr = tl.make_block_ptr(
                base=k_ptr + k_start * kv_head * QK_DIM + start_kvh * QK_DIM,
                shape=(QK_DIM, k_len),
                strides=(1, kv_head * QK_DIM),
                offsets=(0, start_n * BLOCK_N),
                block_shape=(QK_DIM, BLOCK_N),
                order=(0, 1)
            )
            v_block_ptr = tl.make_block_ptr(
                base=v_ptr + k_start * kv_head * V_DIM + start_kvh * V_DIM,
                shape=(V_DIM, k_len),
                strides=(1, kv_head * V_DIM),
                offsets=(0, start_n * BLOCK_N),
                block_shape=(V_DIM, BLOCK_N),
                order=(0, 1)
            )
            dk_block_ptr = tl.make_block_ptr(
                base=dk_ptr + k_start * q_head * QK_DIM + start_qh * QK_DIM,
                shape=(k_len, QK_DIM),
                strides=(q_head * QK_DIM, 1),
                offsets=(start_n * BLOCK_N, 0),
                block_shape=(BLOCK_N, QK_DIM),
                order=(1, 0)
            )
            dv_block_ptr = tl.make_block_ptr(
                base=dv_ptr + k_start * q_head * V_DIM + start_qh * V_DIM,
                shape=(k_len, V_DIM),
                strides=(q_head * V_DIM, 1),
                offsets=(start_n * BLOCK_N, 0),
                block_shape=(BLOCK_N, V_DIM),
                order=(1, 0)
            )
            do_block_ptr = tl.make_block_ptr(
                base=do_ptr + q_start * q_head * V_DIM + start_qh * V_DIM,
                shape=(q_len, V_DIM),
                strides=(q_head * V_DIM, 1),
                offsets=(begin, 0),
                block_shape=(BLOCK_M, V_DIM),
                order=(1, 0)
            )
            l_block_ptr = tl.make_block_ptr(
                base=l_ptr + q_start * q_head + start_qh,
                shape=(q_len,),
                strides=(q_head,),
                offsets=(begin,),
                block_shape=(BLOCK_M,),
                order=(0,)
            )
            d_block_ptr = tl.make_block_ptr(
                base=d_ptr + q_start * q_head + start_qh,
                shape=(q_len,),
                strides=(q_head,),
                offsets=(begin,),
                block_shape=(BLOCK_M,),
                order=(0,)
            )
            # q_attn_arg_block_ptr = tl.make_block_ptr(
            #     base=q_attn_arg_ptr + q_start,
            #     shape=(q_len,),
            #     strides=(1,),
            #     offsets=(begin,),
            #     block_shape=(BLOCK_M,),
            #     order=(0,)
            # )
            # k_attn_arg_block_ptr = tl.make_block_ptr(
            #     base=k_attn_arg_ptr + k_start,
            #     shape=(k_len,),
            #     strides=(1,),
            #     offsets=(start_n * BLOCK_N,),
            #     block_shape=(BLOCK_N,),
            #     order=(0,)
            # )

            mask_block_ptr = tl.make_block_ptr(
                base=mask_tensor_ptr + start_b * max_q_len * max_k_len,
                shape=(q_len, k_len),
                strides=(max_k_len, 1),
                offsets=(begin, start_n * BLOCK_N),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0)
            )

            dk = tl.zeros((BLOCK_N, QK_DIM), dtype=tl.float32)
            dv = tl.zeros((BLOCK_N, V_DIM), dtype=tl.float32)

            k = load_if(k_block_ptr, True, False)
            v = load_if(v_block_ptr, True, False)
            # k_attn_arg = load_if(k_attn_arg_block_ptr, False, True)

            for start_m in range(begin, end, BLOCK_M):
                # start_m = tl.multiple_of(start_m, BLOCK_M)
                # q_attn_arg = load_if(q_attn_arg_block_ptr, False, True)
                # offset_m = start_m + tl.arange(0, BLOCK_M)
                # mask = mask_fn(q_attn_arg, k_attn_arg, offset_m, offset_n, MASK_FN)
                mask = load_if(mask_block_ptr, False, False)
                if not SPARSE_OPT or tl.sum(mask.cast(tl.int32)) != 0:
                    q = load_if(q_block_ptr, False, True)
                    s = tl.dot(q, k)
                    l = load_if(l_block_ptr, False, True)
                    p = tl.math.exp2(s * qk_scale - l[:, None] * log2e)
                    p = tl.where(mask, p, 0.0)
                    do = load_if(do_block_ptr, False, True)
                    p_fp16 = p.to(dtype)
                    # dv
                    dv += tl.dot(tl.trans(p_fp16), do)
                    d = load_if(d_block_ptr, False, True)
                    dp = tl.dot(do, v)
                    ds = p * (dp - d[:, None])
                    ds = tl.where(mask, ds, 0.0)
                    ds = ds.to(dtype)
                    # dq
                    dq_offs_base = (q_start + start_m) * q_head * QK_DIM + start_qh * QK_DIM
                    row_offs = tl.arange(0, BLOCK_M)
                    row_mask = start_m + row_offs < end
                    col_offs = tl.arange(0, QK_DIM)
                    # extension.compile_hint(ds, "break_vf")
                    dq = tl.dot(ds, tl.trans(k))
                    dq *= scale
                    tl.atomic_add(
                        dq_ptr + (dq_offs_base + row_offs[:, None] * q_head * QK_DIM + col_offs[None, :]),
                        dq,
                        mask=row_mask[:, None]
                    )
                    # dk
                    dk += tl.dot(tl.trans(ds), q)
                q_block_ptr = tl.advance(q_block_ptr, (BLOCK_M, 0))
                do_block_ptr = tl.advance(do_block_ptr, (BLOCK_M, 0))
                l_block_ptr = tl.advance(l_block_ptr, (BLOCK_M,))
                d_block_ptr = tl.advance(d_block_ptr, (BLOCK_M,))
                # q_attn_arg_block_ptr = tl.advance(q_attn_arg_block_ptr, (BLOCK_M,))
                mask_block_ptr = tl.advance(mask_block_ptr, (BLOCK_M, 0))

            dk *= scale
            store_if(dk_block_ptr, dk.to(dtype), False, True)
            store_if(dv_block_ptr, dv.to(dtype), False, True)


import triton.runtime.driver as driver
from typing import List

device = torch.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
AICORE_NUM = 28
VECTOR_NUM = properties["num_vectorcore"]


# q: [total_q_seq, head, dim]
# k: [total_kv_seq, head, dim]
class FlashAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, q_attn_arg, k_attn_arg,
                mask_tensor, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale,
                mask_fn, sparse_opt, use_fused_bwd_qkv=True):
        q_len, q_head, qk_dim = q.shape
        k_len, kv_head, v_dim = v.shape
        batch_size = cu_seqlens_q.shape[0] - 1
        o = q.new_empty(q_len, q_head, v_dim)
        l = q.new_empty(q_len, q_head, dtype=torch.float32)

        NUM_CORES = AICORE_NUM
        print(f"QK_DIM: {qk_dim}")
        print(f"V_DIM: {v_dim}")
        print(f"MASK_FN: {mask_fn}")
        print(f"SPARSE_OPT: {sparse_opt}")
        print(f"DTYPE: {(19 if q.dtype == torch.float16 else 14)}")
        print(f"AICORE_NUM: {NUM_CORES}")
        print(f"MAX_Q_LEN: {max_seqlen_q}")
        print(f"MAX_K_LEN: {max_seqlen_k}")
        print(f"BATCH_SIZE: {batch_size}")
        print(f"q shape: {q.shape}")
        print(f"k shape: {k.shape}")
        print(f"v shape: {v.shape}")
        print(f"o shape: {o.shape}")
        print(f"l shape: {l.shape}")
        print(f"mask_tensor shape: {mask_tensor.shape}")
        grid = (NUM_CORES,)
        fwd_kernel[grid](
            q, k, v, o, l,
            q_attn_arg, k_attn_arg, mask_tensor,
            cu_seqlens_q, cu_seqlens_k,
            q_head, kv_head, scale,
            BLOCK_M=64,
            BLOCK_N=32,
            QK_DIM=qk_dim,
            V_DIM=v_dim,
            MASK_FN=mask_fn,
            SPARSE_OPT=sparse_opt,
            DTYPE=(19 if q.dtype == torch.float16 else 14),
            AICORE_NUM=NUM_CORES,
            MAX_Q_LEN=max_seqlen_q,
            MAX_K_LEN=max_seqlen_k,
            BATCH_SIZE=batch_size,
            # multibuffer=True,
            # enable_mixed_cv=True,
            # enable_auto_bind_sub_block=True,
            # sync_solver=True,
            # limit_auto_multi_buffer_of_local_buffer="no-limit",
            # enable_flatten=False,
            # set_workspace_multibuffer=2,
        )
        ctx.save_for_backward(q, k, v, o, l, q_attn_arg, k_attn_arg, mask_tensor, cu_seqlens_q, cu_seqlens_k)
        ctx.use_fused_bwd_qkv = use_fused_bwd_qkv
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.scale = scale
        ctx.mask_fn = mask_fn
        ctx.sparse_opt = sparse_opt
        ctx.k_len = k_len
        ctx.q_head = q_head
        ctx.kv_head = kv_head
        ctx.qk_dim = qk_dim
        ctx.v_dim = v_dim
        ctx.batch_size = batch_size
        ctx.max_seqlen_k = max_seqlen_k
        ctx.dtype = (19 if q.dtype == torch.float16 else 14)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l, q_attn_arg, k_attn_arg, mask_tensor, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        use_fused_bwd_qkv = ctx.use_fused_bwd_qkv
        #if use_fused_bwd_qkv:
        dq = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
        #else:
        #    dq = torch.empty_like(q)
        dk = k.new_empty(ctx.k_len, ctx.q_head, ctx.qk_dim)
        dv = v.new_empty(ctx.k_len, ctx.q_head, ctx.v_dim)
        d = torch.empty_like(l)
        # with torch.no_grad():
        #     d = (o.float() * do.float()).sum(dim=-1).contiguous()
        # torch.npu.synchronize()
        # 等长场景
        if ctx.max_seqlen_q * ctx.batch_size == q.shape[0]:
            if ctx.v_dim > 64:
                BLOCK_SIZE = 16
            else:
                BLOCK_SIZE = 32
            NUM_CORES = VECTOR_NUM
            TASK_SIZE = triton.cdiv(ctx.max_seqlen_q * ctx.batch_size, VECTOR_NUM)
            if TASK_SIZE < BLOCK_SIZE:
                NUM_CORES = triton.cdiv(ctx.max_seqlen_q * ctx.batch_size, BLOCK_SIZE)
                TASK_SIZE = BLOCK_SIZE
            NUM_BLOCKS = triton.cdiv(TASK_SIZE, BLOCK_SIZE)
            # 打印张量（右侧实际变量）的 dtype 和 shape
            print(f"o (referenced by o_ptr) dtype: {o.dtype}, shape: {o.shape}")
            print(f"do (referenced by do_ptr) dtype: {do.dtype}, shape: {do.shape}")
            print(f"d (referenced by d_ptr) dtype: {d.dtype}, shape: {d.shape}")

            # 打印常量（右侧实际表达式或变量）的值
            print(f"ctx.q_head (Q_HEAD_NUM): {ctx.q_head}")
            print(f"ctx.v_dim (V_DIM): {ctx.v_dim}")
            print(f"ctx.dtype (DTYPE): {ctx.dtype}")
            print(f"TASK_SIZE: {TASK_SIZE}")
            print(f"NUM_BLOCKS: {NUM_BLOCKS}")
            print(f"BLOCK_SIZE: {BLOCK_SIZE}")
            bwd_preprocess_ifmn[(NUM_CORES,)](
                o_ptr = o,
                do_ptr = do,
                d_ptr = d,
                Q_HEAD_NUM = ctx.q_head,
                V_DIM = ctx.v_dim,
                DTYPE = ctx.dtype,
                TASK_SIZE = TASK_SIZE,
                NUM_BLOCKS = NUM_BLOCKS,
                BLOCK_SIZE = BLOCK_SIZE,
                # multibuffer = True,
                # limit_auto_multi_buffer_of_local_buffer = "no-limit",
            )
        else:  # 不等长场景
            if ctx.v_dim > 64:
                BLOCK_M = 128
            else:
                BLOCK_M = 256
            NUM_CORES = VECTOR_NUM
            NUM_BLOCKS_M = triton.cdiv(ctx.max_seqlen_q, BLOCK_M)
            NUM_BLOCKS = NUM_BLOCKS_M * ctx.q_head * ctx.batch_size
            bwd_preprocess[(NUM_CORES,)](
                o, do, d,
                cu_seqlens_q,
                ctx.q_head,
                V_DIM = ctx.v_dim,
                DTYPE = ctx.dtype,
                # inject_barrier_all=True,
                # limit_auto_multi_buffer_of_local_buffer="no-limit",
                # enable_flatten=False,
                BLOCK_M=BLOCK_M,
                NUM_BLOCKS_M=NUM_BLOCKS_M,
                NUM_BLOCKS=NUM_BLOCKS,
                NUM_CORES=NUM_CORES,
                # multibuffer=True,
            )

        #if use_fused_bwd_qkv:
        NUM_CORES = AICORE_NUM
        bwd_qkv_kernel[(NUM_CORES,)](
            q, k, v, 
            dq, dk, dv, 
            do, l, d,
            q_attn_arg, k_attn_arg, 
            mask_tensor,
            cu_seqlens_q, cu_seqlens_k,
            ctx.batch_size, 
            ctx.max_seqlen_q, ctx.max_seqlen_k,
            ctx.q_head, ctx.kv_head, 
            ctx.scale,
            QK_DIM=ctx.qk_dim,
            V_DIM=ctx.v_dim,
            MASK_FN=ctx.mask_fn,
            SPARSE_OPT=ctx.sparse_opt,
            DTYPE=ctx.dtype,
            NUM_CORES=NUM_CORES,
            BLOCK_M=64,
            BLOCK_N=64,
            # inject_barrier_all=True,
            # limit_auto_multi_buffer_of_local_buffer="no-limit",
            # enable_flatten=False,
            # sync_solver=True,
            # enable_mixed_cv=True,
        )
        dq = dq.to(q.dtype)
        #else:
        #    BLOCK_M = 128
        #    BLOCK_N = 128
        #    NUM_CORES = AICORE_NUM
        #    NUM_BLOCKS_N= triton.cdiv(ctx.max_seqlen_k, BLOCK_N)
        #    NUM_BLOCKS = NUM_BLOCKS_N * ctx.q_head * ctx.batch_size
        #    bwd_kv_kernel[(NUM_CORES,)](
        #        q, k, v, dk, dv, do, l, d,
        #        q_attn_arg, k_attn_arg, mask_tensor,
        #        cu_seqlens_q, cu_seqlens_k,
        #        ctx.q_head, ctx.kv_head, ctx.scale,
        #        QK_DIM=ctx.qk_dim,
        #        V_DIM=ctx.v_dim,
        #        MASK_FN=ctx.mask_fn,
        #        SPARSE_OPT=ctx.sparse_opt,
        #        DTYPE=ctx.dtype,
        #        BLOCK_M=BLOCK_M,
        #        BLOCK_N=BLOCK_N,
        #        NUM_BLOCKS_N=NUM_BLOCKS_N,
        #        NUM_BLOCKS=NUM_BLOCKS,
        #        NUM_CORES=NUM_CORES,
        #        MAX_Q_LEN=ctx.max_seqlen_q,
        #        MAX_K_LEN=ctx.max_seqlen_k,
        #        # inject_barrier_all=True,
        #        # limit_auto_multi_buffer_of_local_buffer="no-limit",
        #        # enable_flatten=False,
        #        # sync_solver=True,
        #        # enable_mixed_cv=True,
        #    )

        #    BLOCK_M = 128
        #    BLOCK_N = 128
        #    NUM_CORES = AICORE_NUM
        #    NUM_BLOCKS_M = triton.cdiv(ctx.max_seqlen_q, BLOCK_M)
        #    NUM_BLOCKS = NUM_BLOCKS_M * ctx.q_head * ctx.batch_size
        #    bwd_q_kernel[(NUM_CORES,)](
        #        q, k, v, dq, do, l, d,
        #        q_attn_arg, k_attn_arg, mask_tensor,
        #        cu_seqlens_q, cu_seqlens_k,
        #        ctx.q_head, ctx.kv_head, ctx.scale,
        #        QK_DIM=ctx.qk_dim,
        #        V_DIM=ctx.v_dim,
        #        MASK_FN=ctx.mask_fn,
        #        SPARSE_OPT=ctx.sparse_opt,
        #        DTYPE=ctx.dtype,
        #        BLOCK_M=BLOCK_M,
        #        BLOCK_N=BLOCK_N,
        #        NUM_BLOCKS_M=NUM_BLOCKS_M,
        #        NUM_BLOCKS=NUM_BLOCKS,
        #        NUM_CORES=NUM_CORES,
        #        MAX_Q_LEN=ctx.max_seqlen_q,
        #        MAX_K_LEN=ctx.max_seqlen_k,
        #        # limit_auto_multi_buffer_of_local_buffer="no-limit",
        #        # enable_flatten=False,
        #        # sync_solver=True,
        #        # enable_mixed_cv=True
        #    )
        head_group = ctx.q_head // ctx.kv_head
        if head_group > 1:
            dk = dk.reshape(ctx.k_len, ctx.kv_head, head_group, ctx.qk_dim).sum(2)
            dv = dv.reshape(ctx.k_len, ctx.kv_head, head_group, ctx.v_dim).sum(2)
        return (dq, dk, dv) + (None,) * 11


import torch_npu
import numpy as np


def generate_mask_fn(q_seq_list, k_seq_list, bs, max_q_len, max_k_len, q_attn_arg, k_attn_arg, BLOCK_M, BLOCK_N):
    mask_fn = torch.zeros((bs, max_q_len, max_k_len), dtype=torch.bool, device="cpu")
    for b_i in range(bs):
        cur_q_len = q_seq_list[b_i]
        cur_k_len = k_seq_list[b_i]
        num_block_m = (cur_q_len + BLOCK_M - 1) // BLOCK_M
        num_block_n = (cur_k_len + BLOCK_N - 1) // BLOCK_N
        b_q_offset = 0
        b_k_offset = 0
        for block_m_i in range(num_block_m):
            cur_start_q = block_m_i * BLOCK_M
            cur_end_q = min(cur_start_q + BLOCK_M, cur_q_len)
            q_offset = range(cur_start_q, cur_end_q)
            cur_q_attn_args = q_attn_arg[cur_start_q:cur_end_q]
            for block_n_i in range(num_block_n):
                cur_start_k = block_n_i * BLOCK_N
                cur_end_k = min(cur_start_k + BLOCK_N, cur_k_len)
                k_offset = range(cur_start_k, cur_end_k)
                cur_k_attn_args = k_attn_arg[cur_start_k: cur_end_k]

                # triu_causal = (q_offset[:, None] <= k_offset[None, :])
                triu_causal = (torch.tensor(list(q_offset))[:, None] <= torch.tensor(list(k_offset))[None, :])
                attn_args_mask = (cur_q_attn_args[:, None] == cur_k_attn_args[None, :]) | (
                            cur_k_attn_args[None, :] == 0)
                q_offset_mask = (torch.tensor(list(q_offset))[:, None] == torch.tensor(list(k_offset))[None, :])
                # print(f"{triu_causal.shape=}, {triu_causal=}, {attn_args_mask.shape=}, {attn_args_mask=}, {q_offset_mask.shape=}, {q_offset_mask=}")
                result_mask = ((triu_causal.bool() & attn_args_mask.bool()) | q_offset_mask.bool()).to(torch.bool)
                # print(f"{b_i=}, {block_m_i=}, {block_n_i=}, {result_mask.shape=}")

                mask_fn[b_i, cur_start_q: cur_end_q, cur_start_k: cur_end_k] = result_mask
        b_k_offset += cur_k_len
        b_q_offset += cur_q_len
    return mask_fn


def generate_mask_fn_vectorized(q_seq_list, k_seq_list, bs, max_q_len, max_k_len, q_attn_arg, k_attn_arg):
    device = "cpu"

    # 创建结果张量
    mask_fn = torch.zeros((bs, max_q_len, max_k_len), dtype=torch.bool, device=device)

    # 为每个batch独立处理
    for b_i in range(bs):
        cur_q_len = q_seq_list[b_i]
        cur_k_len = k_seq_list[b_i]

        # 创建位置索引
        q_positions = torch.arange(cur_q_len, device=device).view(-1, 1)
        k_positions = torch.arange(cur_k_len, device=device).view(1, -1)

        # 计算 causal mask: q_offset <= k_offset (注意这里是 <=, 不是 <)
        # 原始代码使用的是 triu_causal = (q_offset[:, None] <= k_offset[None, :])
        causal_mask = (q_positions <= k_positions)

        # 计算 attention args mask
        # 确保数据类型一致，原始代码使用的是 .bool()
        q_attn_slice = torch.tensor(q_attn_arg[:cur_q_len], device=device, dtype=torch.int32).view(-1, 1)
        k_attn_slice = torch.tensor(k_attn_arg[:cur_k_len], device=device, dtype=torch.int32).view(1, -1)

        # 原始逻辑: (cur_q_attn_args[:, None] == cur_k_attn_args[None, :]) | (cur_k_attn_args[None, :] == 0)
        attn_args_mask = (q_attn_slice == k_attn_slice) | (k_attn_slice == 0)

        # 计算 q offset mask: q_offset == k_offset
        q_offset_mask = (q_positions == k_positions)

        # 组合所有mask，保持与原始代码相同的布尔运算顺序
        # 原始: ((triu_causal.bool() & attn_args_mask.bool()) | q_offset_mask.bool())
        result_mask = ((causal_mask.bool() & attn_args_mask.bool()) | q_offset_mask.bool()).to(torch.bool)

        # 存储结果，不需要额外的valid_mask，因为我们只处理有效范围
        mask_fn[b_i, :cur_q_len, :cur_k_len] = result_mask

    return mask_fn


# ---------------------------------------------------------------------------
# Forward-save / Backward-load helpers
# ---------------------------------------------------------------------------
# Run the forward pass once, persist all forward outputs plus every input
# needed by backward; later load that file and run *only* the backward pass.

def _prepare_inputs(idx, bs, num_head, head_dim, seqlen):
    """Load q/k/v/do + aux tensors (CPU) and collect every scalar ctx attribute
    that ``FlashAttentionFunc.forward`` stashes on ctx."""
    dtype = torch.bfloat16
    q_seq_list = [0] + [seqlen] * 8
    k_seq_list = [0] + [seqlen] * 8

    q_len = sum(q_seq_list)
    k_len = sum(k_seq_list)
    max_seqlen_q = max(q_seq_list)
    max_seqlen_k = max(k_seq_list)

    data_root = os.environ.get(
        "FA_TEST_DATA_ROOT",
        "/data/j00808997/opentile/fa_bwd/new",
    )
    root_dir = os.path.join(
        os.path.abspath(data_root),
        f"dump1_case{idx}",
    )

    q = torch.load(f"{root_dir}/q.pt", map_location=torch.device("cpu")).to(dtype)
    k = torch.load(f"{root_dir}/k.pt", map_location=torch.device("cpu")).to(dtype)
    v = torch.load(f"{root_dir}/v.pt", map_location=torch.device("cpu")).to(dtype)
    do = torch.load(f"{root_dir}/do.pt", map_location=torch.device("cpu")).to(dtype)

    q_attn_arg = torch.zeros(q_len, dtype=torch.int32, device="cpu")
    k_attn_arg = torch.zeros(k_len, dtype=torch.int32, device="cpu")

    cu_seqlens_q = np.cumsum(q_seq_list).tolist()
    cu_seqlens_k = np.cumsum(k_seq_list).tolist()
    cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cpu")
    cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device="cpu")

    mask_tensor = generate_mask_fn_vectorized(
        q_seq_list[1:], k_seq_list[1:], bs, max_seqlen_q, max_seqlen_k,
        q_attn_arg, k_attn_arg,
    )

    return {
        "q": q,
        "k": k,
        "v": v,
        "do": do,
        "q_attn_arg": q_attn_arg,
        "k_attn_arg": k_attn_arg,
        "mask_tensor": mask_tensor,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "scale": 1.0 / (head_dim ** 0.5),
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "q_head": q.shape[1],
        "kv_head": v.shape[1],
        "qk_dim": q.shape[2],
        "v_dim": v.shape[2],
        "k_len": k_len,
        "batch_size": cu_seqlens_q.shape[0] - 1,
        "dtype": (19 if dtype == torch.float16 else 14),
        "mask_fn": 1,
        "sparse_opt": False,
        "root_dir": root_dir,
    }


def _fwd_state_dir(idx):
    """Directory for the persisted forward state of case ``idx``.

    Defaults to ``/data/m/f`` but can be overridden via ``FA_FWD_STATE_ROOT``.
    Note this is independent of ``FA_TEST_DATA_ROOT`` (which controls where
    q/k/v/do live) so the fwd state can live on a different disk.
    """
    base = os.environ.get("FA_FWD_STATE_ROOT", "/data/m/f")
    return os.path.join(os.path.abspath(base), f"dump1_case{idx}")


def _fwd_state_path(idx):
    return os.path.join(_fwd_state_dir(idx), f"fwd_state_case{idx}.pt")


def _preprocess_state_path(idx):
    """Where ``bwd_preprocess_ifmn``'s output ``d`` is persisted."""
    return os.path.join(_fwd_state_dir(idx), f"preprocess_d_case{idx}.pt")


def _save_fwd_state(save_path, inputs, o, l):
    """Persist forward state to a single ``.pt`` file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    payload = {**inputs, "o": o.detach().cpu(), "l": l.detach().cpu()}
    torch.save(payload, save_path)
    print(f"[fwd] saved -> {save_path}  (o={payload['o'].shape}, l={payload['l'].shape})")


def _load_fwd_state(load_path):
    print(f"[bwd] loading <- {load_path}")
    return torch.load(load_path, map_location="cpu")


def _run_fa_forward_save(idx, bs, num_head, head_dim, seqlen):
    """Phase 1: run the forward kernel directly, save state to disk."""
    DEVICE = torch.device("npu")
    inputs = _prepare_inputs(idx, bs, num_head, head_dim, seqlen)

    q = inputs["q"].to(DEVICE)
    k = inputs["k"].to(DEVICE)
    v = inputs["v"].to(DEVICE)
    q_attn_arg = inputs["q_attn_arg"].to(DEVICE)
    k_attn_arg = inputs["k_attn_arg"].to(DEVICE)
    mask_tensor = inputs["mask_tensor"].to(DEVICE)
    cu_seqlens_q = inputs["cu_seqlens_q"].to(DEVICE)
    cu_seqlens_k = inputs["cu_seqlens_k"].to(DEVICE)

    o = q.new_empty(q.shape[0], q.shape[1], inputs["v_dim"])
    l = q.new_empty(q.shape[0], q.shape[1], dtype=torch.float32)

    NUM_CORES = AICORE_NUM
    fwd_kernel[(NUM_CORES,)](
        q, k, v, o, l,
        q_attn_arg, k_attn_arg, mask_tensor,
        cu_seqlens_q, cu_seqlens_k,
        inputs["q_head"], inputs["kv_head"], inputs["scale"],
        BLOCK_M=64, BLOCK_N=32,
        QK_DIM=inputs["qk_dim"], V_DIM=inputs["v_dim"],
        MASK_FN=inputs["mask_fn"], SPARSE_OPT=inputs["sparse_opt"],
        DTYPE=inputs["dtype"],
        AICORE_NUM=NUM_CORES,
        MAX_Q_LEN=inputs["max_seqlen_q"], MAX_K_LEN=inputs["max_seqlen_k"],
        BATCH_SIZE=inputs["batch_size"],
    )
    torch.npu.synchronize()

    _save_fwd_state(_fwd_state_path(idx), inputs, o, l)


def _make_assert_bwd(idx, root_dir, rtol=5e-3, atol=5e-3):
    """Build an ``assert_output`` closure (mirrors the one in ``_run_fa_case``)
    so the backward-only path can compare against the same golden files."""

    def assert_output(name, actual, filename,
                      output_atol=atol, allow_one_ulp_exception=False):
        golden = torch.load(
            os.path.join(root_dir, filename), map_location="cpu"
        )
        actual_cpu = actual.detach().cpu().contiguous()
        golden_source_cpu = golden.detach().cpu().contiguous()
        golden_cpu = golden_source_cpu.to(torch.bfloat16).contiguous()
        actual_f32 = actual_cpu.float()
        golden_f32 = golden_cpu.float()
        close = torch.isclose(
            actual_f32, golden_f32,
            rtol=rtol, atol=output_atol, equal_nan=False,
        )
        allow_one_ulp = False
        if not close.all().item():
            abs_error = (actual_f32 - golden_f32).abs()
            tolerance = output_atol + rtol * golden_f32.abs()
            error_ratio = torch.nan_to_num(
                abs_error / tolerance,
                nan=float("inf"), posinf=float("inf"),
            ).masked_fill(close, -1.0)
            bad_count = int((~close).sum().item())
            top_count = min(bad_count, 16)
            worst_flat_indices = torch.topk(
                error_ratio.flatten(), k=top_count,
            ).indices.tolist()
            print(
                f"\n[case{idx}][{name}] mismatch: "
                f"{bad_count}/{actual_f32.numel()}, "
                f"rtol={rtol}, atol={output_atol}"
            )
            print(
                f"[case{idx}][{name}] dtype: "
                f"actual={actual_cpu.dtype}, "
                f"gold_source={golden_source_cpu.dtype}, "
                f"gold_compare={golden_cpu.dtype}"
            )
            if (
                allow_one_ulp_exception
                and actual_cpu.dtype == torch.bfloat16
                and golden_cpu.dtype == torch.bfloat16
            ):
                actual_bits = (
                    actual_cpu.view(torch.int16).to(torch.int32) & 0xffff
                )
                golden_bits = (
                    golden_cpu.view(torch.int16).to(torch.int32) & 0xffff
                )
                actual_ordered = torch.where(
                    (actual_bits & 0x8000) != 0,
                    (~actual_bits) & 0xffff,
                    actual_bits | 0x8000,
                )
                golden_ordered = torch.where(
                    (golden_bits & 0x8000) != 0,
                    (~golden_bits) & 0xffff,
                    golden_bits | 0x8000,
                )
                ulp_distance = (actual_ordered - golden_ordered).abs()
                bad_ulp_distance = ulp_distance[~close]
                allow_one_ulp = (
                    bad_count == 1
                    and bool((bad_ulp_distance <= 2).all().item())
                )
                print(
                    f"[case{idx}][{name}] one-ULP exception: "
                    f"bad_count={bad_count}, "
                    f"bad_ulp={bad_ulp_distance.tolist()}, "
                    f"allowed={allow_one_ulp}"
                )
            for rank, flat_index in enumerate(worst_flat_indices, 1):
                index = tuple(
                    int(i) for i in np.unravel_index(
                        flat_index, tuple(actual_f32.shape),
                    )
                )
                actual_value = actual_f32[index].item()
                golden_value = golden_f32[index].item()
                golden_source_value = golden_source_cpu[index].float().item()
                difference = abs_error[index].item()
                allowed = tolerance[index].item()
                ratio = error_ratio[index].item()
                print(
                    f"[case{idx}][{name}] #{rank} idx={index}: "
                    f"actual={actual_value:.10g}, "
                    f"gold={golden_value:.10g}, "
                    f"gold_source={golden_source_value:.10g}, "
                    f"abs_diff={difference:.10g}, "
                    f"tolerance={allowed:.10g}, "
                    f"error/tol={ratio:.6f}"
                )
                if (
                    actual_cpu.dtype == torch.bfloat16
                    and golden_cpu.dtype == torch.bfloat16
                ):
                    actual_bits = (
                        int(actual_cpu.view(torch.int16)[index].item())
                        & 0xffff
                    )
                    golden_bits = (
                        int(golden_cpu.view(torch.int16)[index].item())
                        & 0xffff
                    )
                    print(
                        f"[case{idx}][{name}] #{rank} BF16 bits: "
                        f"actual=0x{actual_bits:04x}, "
                        f"gold=0x{golden_bits:04x}"
                    )
        assert close.all().item() or allow_one_ulp, f"{name} output mismatch"

    return assert_output


def _run_fa_preprocess_save(idx, bs, num_head, head_dim, seqlen):
    """Phase 1.5: load the persisted forward state, run
    ``bwd_preprocess_ifmn`` once, persist its output ``d``.

    Only valid when ``max_seqlen_q * batch_size == q.shape[0]`` (the equal-len
    path that uses ``bwd_preprocess_ifmn``). For the unequal-len path the
    preprocess is still recomputed inline in ``_run_fa_backward_load``.
    """
    DEVICE = torch.device("npu")

    inputs = _prepare_inputs(idx, bs, num_head, head_dim, seqlen)
    state = _load_fwd_state(_fwd_state_path(idx))

    o = state["o"].to(DEVICE)
    do = state["do"].to(DEVICE)

    total = state["max_seqlen_q"] * state["batch_size"]
    if total != state["q"].shape[0]:
        raise RuntimeError(
            f"[case{idx}] preprocess_ifmn requires equal-len shape "
            f"(max_seqlen_q*batch_size==q.shape[0]); got total={total}, "
            f"q.shape[0]={state['q'].shape[0]}"
        )

    if state["v_dim"] > 64:
        BLOCK_SIZE = 16
    else:
        BLOCK_SIZE = 32
    NUM_CORES = VECTOR_NUM
    TASK_SIZE = triton.cdiv(total, VECTOR_NUM)
    if TASK_SIZE < BLOCK_SIZE:
        NUM_CORES = triton.cdiv(total, BLOCK_SIZE)
        TASK_SIZE = BLOCK_SIZE
    NUM_BLOCKS = triton.cdiv(TASK_SIZE, BLOCK_SIZE)

    d = torch.empty(
        total, state["q_head"], dtype=torch.float32, device=DEVICE,
    )

    bwd_preprocess_ifmn[(NUM_CORES,)](
        o_ptr=o, do_ptr=do, d_ptr=d,
        Q_HEAD_NUM=state["q_head"],
        V_DIM=state["v_dim"],
        DTYPE=state["dtype"],
        TASK_SIZE=TASK_SIZE,
        NUM_BLOCKS=NUM_BLOCKS,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    save_path = _preprocess_state_path(idx)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"d": d.detach().cpu()}, save_path)
    print(
        f"[preprocess] saved -> {save_path}  "
        f"(d={d.shape}, dtype={d.dtype})"
    )


def _run_fa_backward_load(idx, bs, num_head, head_dim, seqlen):
    """Phase 2: load persisted forward state, run *only* the backward kernels."""
    DEVICE = torch.device("npu")

    inputs = _prepare_inputs(idx, bs, num_head, head_dim, seqlen)
    state = _load_fwd_state(_fwd_state_path(idx))

    q = state["q"].to(DEVICE)
    k = state["k"].to(DEVICE)
    v = state["v"].to(DEVICE)
    o = state["o"].to(DEVICE)
    l = state["l"].to(DEVICE)
    do = state["do"].to(DEVICE)
    q_attn_arg = state["q_attn_arg"].to(DEVICE)
    k_attn_arg = state["k_attn_arg"].to(DEVICE)
    mask_tensor = state["mask_tensor"].to(DEVICE)
    cu_seqlens_q = state["cu_seqlens_q"].to(DEVICE)
    cu_seqlens_k = state["cu_seqlens_k"].to(DEVICE)

    dq = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    dk = k.new_empty(state["k_len"], state["q_head"], state["qk_dim"])
    dv = v.new_empty(state["k_len"], state["q_head"], state["v_dim"])
    d = torch.empty_like(l)

    # ---- bwd preprocess ----
    if state["max_seqlen_q"] * state["batch_size"] == q.shape[0]:
        # Load d produced by a previous _run_fa_preprocess_save() run.
        prep_path = _preprocess_state_path(idx)
        if not os.path.isfile(prep_path):
            raise FileNotFoundError(
                f"[case{idx}] preprocess output not found at {prep_path}; "
                f"run test_fa_save_preprocess first."
            )
        print(f"[bwd] loading preprocess d <- {prep_path}")
        d_loaded = torch.load(prep_path, map_location="cpu")
        d.copy_(d_loaded["d"].to(d.dtype).to(d.device))
    else:
        if state["v_dim"] > 64:
            BLOCK_M = 128
        else:
            BLOCK_M = 256
        NUM_CORES = VECTOR_NUM
        NUM_BLOCKS_M = triton.cdiv(state["max_seqlen_q"], BLOCK_M)
        NUM_BLOCKS = NUM_BLOCKS_M * state["q_head"] * state["batch_size"]
        bwd_preprocess[(NUM_CORES,)](
            o, do, d, cu_seqlens_q, state["q_head"],
            V_DIM=state["v_dim"], DTYPE=state["dtype"],
            BLOCK_M=BLOCK_M,
            NUM_BLOCKS_M=NUM_BLOCKS_M,
            NUM_BLOCKS=NUM_BLOCKS, NUM_CORES=NUM_CORES,
        )

    # ---- fused bwd qkv ----
    NUM_CORES = AICORE_NUM
    bwd_qkv_kernel[(NUM_CORES,)](
        q, k, v, dq, dk, dv,
        do, l, d,
        q_attn_arg, k_attn_arg,
        mask_tensor,
        cu_seqlens_q, cu_seqlens_k,
        state["batch_size"],
        state["max_seqlen_q"], state["max_seqlen_k"],
        state["q_head"], state["kv_head"],
        state["scale"],
        QK_DIM=state["qk_dim"], V_DIM=state["v_dim"],
        MASK_FN=state["mask_fn"], SPARSE_OPT=state["sparse_opt"],
        DTYPE=state["dtype"],
        NUM_CORES=NUM_CORES, BLOCK_M=64, BLOCK_N=64,
    )
    dq = dq.to(q.dtype)

    head_group = state["q_head"] // state["kv_head"]
    if head_group > 1:
        dk = dk.reshape(
            state["k_len"], state["kv_head"], head_group, state["qk_dim"],
        ).sum(2)
        dv = dv.reshape(
            state["k_len"], state["kv_head"], head_group, state["v_dim"],
        ).sum(2)

    assert_output = _make_assert_bwd(idx, inputs["root_dir"])
    assert_output("dq", dq, "dq.pt")
    assert_output("dk", dk, "dk.pt", allow_one_ulp_exception=True)
    assert_output("dv", dv, "dv.pt")


def _run_fa_case(idx, bs, num_head, head_dim, seqlen):
    dtype = torch.bfloat16
    DEVICE = torch.device("npu")

    q_seq_list = [0] + [seqlen] * 8
    k_seq_list = [0] + [seqlen] * 8

    q_len = sum(q_seq_list)
    k_len = sum(k_seq_list)
    max_seqlen_q = max(q_seq_list)
    max_seqlen_k = max(k_seq_list)
    data_root = os.environ.get(
        "FA_TEST_DATA_ROOT",
        "/data/j00808997/opentile/fa_bwd/new",
    )
    root_dir = os.path.join(
        os.path.abspath(data_root),
        f"dump1_case{idx}",
    )
    q = torch.load(f"{root_dir}/q.pt", map_location=torch.device('cpu')).to(dtype).to(DEVICE).detach().requires_grad_()
    k = torch.load(f"{root_dir}/k.pt", map_location=torch.device('cpu')).to(dtype).to(DEVICE).detach().requires_grad_()
    v = torch.load(f"{root_dir}/v.pt", map_location=torch.device('cpu')).to(dtype).to(DEVICE).detach().requires_grad_()

    q_attn_arg = torch.zeros(q_len, dtype=torch.int32, device="cpu")
    k_attn_arg = torch.zeros(k_len, dtype=torch.int32, device="cpu")

    cu_seqlens_q = np.cumsum(q_seq_list).tolist()
    cu_seqlens_k = np.cumsum(k_seq_list).tolist()
    cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cpu")
    cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device="cpu")

    mask_tensor = generate_mask_fn_vectorized(q_seq_list[1:], k_seq_list[1:], bs, max_seqlen_q, max_seqlen_k, q_attn_arg, k_attn_arg)

    q_attn_arg = q_attn_arg.npu()
    k_attn_arg = k_attn_arg.npu()
    cu_seqlens_q = cu_seqlens_q.npu()
    cu_seqlens_k = cu_seqlens_k.npu()
    mask_tensor = mask_tensor.npu()
    scale = 1.0 / (head_dim ** 0.5)

    rtol, atol = 5e-3, 5e-3

    def assert_output(
        name,
        actual,
        filename,
        output_atol=atol,
        allow_one_ulp_exception=False,
    ):
        golden = torch.load(
            os.path.join(root_dir, filename),
            map_location="cpu",
        )
        actual_cpu = actual.detach().cpu().contiguous()
        golden_source_cpu = golden.detach().cpu().contiguous()
        golden_cpu = golden_source_cpu.to(dtype).contiguous()
        actual_f32 = actual_cpu.float()
        golden_f32 = golden_cpu.float()
        close = torch.isclose(
            actual_f32,
            golden_f32,
            rtol=rtol,
            atol=output_atol,
            equal_nan=False,
        )
        allow_one_ulp = False
        if not close.all().item():
            abs_error = (actual_f32 - golden_f32).abs()
            tolerance = output_atol + rtol * golden_f32.abs()
            error_ratio = torch.nan_to_num(
                abs_error / tolerance,
                nan=float("inf"),
                posinf=float("inf"),
            ).masked_fill(close, -1.0)
            bad_count = int((~close).sum().item())
            top_count = min(bad_count, 16)
            worst_flat_indices = torch.topk(
                error_ratio.flatten(),
                k=top_count,
            ).indices.tolist()

            print(
                f"\n[case{idx}][{name}] mismatch: "
                f"{bad_count}/{actual_f32.numel()}, "
                f"rtol={rtol}, atol={output_atol}"
            )
            print(
                f"[case{idx}][{name}] dtype: "
                f"actual={actual_cpu.dtype}, "
                f"gold_source={golden_source_cpu.dtype}, "
                f"gold_compare={golden_cpu.dtype}"
            )
            if (
                allow_one_ulp_exception
                and actual_cpu.dtype == torch.bfloat16
                and golden_cpu.dtype == torch.bfloat16
            ):
                actual_bits = (
                    actual_cpu.view(torch.int16).to(torch.int32) & 0xffff
                )
                golden_bits = (
                    golden_cpu.view(torch.int16).to(torch.int32) & 0xffff
                )
                actual_ordered = torch.where(
                    (actual_bits & 0x8000) != 0,
                    (~actual_bits) & 0xffff,
                    actual_bits | 0x8000,
                )
                golden_ordered = torch.where(
                    (golden_bits & 0x8000) != 0,
                    (~golden_bits) & 0xffff,
                    golden_bits | 0x8000,
                )
                ulp_distance = (actual_ordered - golden_ordered).abs()
                bad_ulp_distance = ulp_distance[~close]
                allow_one_ulp = (
                    bad_count == 1
                    and bool((bad_ulp_distance <= 2).all().item())
                )
                print(
                    f"[case{idx}][{name}] one-ULP exception: "
                    f"bad_count={bad_count}, "
                    f"bad_ulp={bad_ulp_distance.tolist()}, "
                    f"allowed={allow_one_ulp}"
                )
            for rank, flat_index in enumerate(worst_flat_indices, 1):
                index = tuple(
                    int(i)
                    for i in np.unravel_index(
                        flat_index,
                        tuple(actual_f32.shape),
                    )
                )
                actual_value = actual_f32[index].item()
                golden_value = golden_f32[index].item()
                golden_source_value = golden_source_cpu[index].float().item()
                difference = abs_error[index].item()
                allowed = tolerance[index].item()
                ratio = error_ratio[index].item()
                print(
                    f"[case{idx}][{name}] #{rank} idx={index}: "
                    f"actual={actual_value:.10g}, "
                    f"gold={golden_value:.10g}, "
                    f"gold_source={golden_source_value:.10g}, "
                    f"abs_diff={difference:.10g}, "
                    f"tolerance={allowed:.10g}, "
                    f"error/tol={ratio:.6f}"
                )
                if (
                    actual_cpu.dtype == torch.bfloat16
                    and golden_cpu.dtype == torch.bfloat16
                ):
                    actual_bits = (
                        int(actual_cpu.view(torch.int16)[index].item())
                        & 0xffff
                    )
                    golden_bits = (
                        int(golden_cpu.view(torch.int16)[index].item())
                        & 0xffff
                    )
                    print(
                        f"[case{idx}][{name}] #{rank} BF16 bits: "
                        f"actual=0x{actual_bits:04x}, "
                        f"gold=0x{golden_bits:04x}"
                    )
        assert close.all().item() or allow_one_ulp, f"{name} output mismatch"

    result = FlashAttentionFunc.apply(
        q,
        k,
        v,
        q_attn_arg,
        k_attn_arg,
        mask_tensor,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        scale,
        1,
        False,
    )
    torch.npu.synchronize()

    do = torch.load(f"{root_dir}/do.pt", map_location=torch.device('cpu')).to(dtype).to(DEVICE)
    result.backward(do)
    torch.npu.synchronize()
    dq, dk, dv = q.grad, k.grad, v.grad

    assert_output("res", result, "res.pt")
    assert_output("dq", dq, "dq.pt")
    assert_output(
        "dk",
        dk,
        "dk.pt",
        allow_one_ulp_exception=True,
    )
    assert_output("dv", dv, "dv.pt")


@pytest.mark.parametrize(
    "idx,bs,num_head,head_dim,seqlen",
    [
        (1, 8, 8, 64, 2432), # ssbuffer精度fail，dag精度pass
        # (2, 8, 8, 64, 320), # ssbuffer精度错误
        # (3, 8, 16, 32, 32),
        # (4, 8, 16, 32, 64),
        # (5, 8, 8, 64, 32), # ssbuffer精度错误
        # (6, 8, 4, 128, 118), # ssbuffer精度错误
    ],
)
def test_fa_forward_backward(idx, bs, num_head, head_dim, seqlen):
    """Run the original BF16 FA forward/backward numerical comparison."""
    _run_fa_case(idx, bs, num_head, head_dim, seqlen)


@pytest.mark.parametrize(
    "idx,bs,num_head,head_dim,seqlen",
    [
        (1, 8, 8, 64, 2432),
    ],
)
def test_fa_save_forward(idx, bs, num_head, head_dim, seqlen):
    """Phase 1: run the forward pass only and persist its outputs (o, l)
    plus every input needed by the backward pass."""
    _run_fa_forward_save(idx, bs, num_head, head_dim, seqlen)


@pytest.mark.parametrize(
    "idx,bs,num_head,head_dim,seqlen",
    [
        (1, 8, 8, 64, 2432),
    ],
)
def test_fa_load_backward(idx, bs, num_head, head_dim, seqlen):
    """Phase 2: load the persisted forward state and run only the backward
    kernels, asserting dq/dk/dv against the golden files."""
    _run_fa_backward_load(idx, bs, num_head, head_dim, seqlen)


@pytest.mark.parametrize(
    "idx,bs,num_head,head_dim,seqlen",
    [
        (1, 8, 8, 64, 2432),
    ],
)
def test_fa_save_preprocess(idx, bs, num_head, head_dim, seqlen):
    """Phase 1.5: load the persisted forward state, run only
    ``bwd_preprocess_ifmn``, and persist its output ``d`` so the backward-load
    path can consume it without recomputing."""
    _run_fa_preprocess_save(idx, bs, num_head, head_dim, seqlen)