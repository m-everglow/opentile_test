import os
import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension
# from maybe_triton_jit import maybe_triton_jit
try:
    from .utils import is_hopper, is_ampere
except ImportError:
    from utils import is_hopper, is_ampere


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
        return [
             triton.Config({"BLOCK_M": 64, "BLOCK_N": 128})
        ]
        configs = [
            triton.Config(
                {
                    "BLOCK_M": BM,
                    "BLOCK_N": BN,
                },
            )
            for BM in [32, 64, 128]
            for BN in [32, 64, 128, 256]
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


#@triton.autotune(
#   configs=get_fwd_configs(),
#   key=["QK_DIM", "V_DIM", "MASK_FN", "SPARSE_OPT"],
#)
@triton.jit
def fwd_kernel(
        q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
        q_attn_arg_ptr, k_attn_arg_ptr,
        cu_seqlens_q, cu_seqlens_k,
        q_head, kv_head, scale,
        QK_DIM: tl.constexpr, V_DIM: tl.constexpr, MASK_FN: tl.constexpr,
        SPARSE_OPT: tl.constexpr, DTYPE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        AICORE_NUM, # tl.constexpr,
        MAX_Q_LEN, #: tl.constexpr,
        MAX_K_LEN, #: tl.constexpr,
        BATCH_SIZE: tl.constexpr,
        q_len_total,
        k_len_total,
):
    dtype = o_ptr.type.element_ty
    pid = tl.program_id(0)
    task_nums = tl.cdiv(MAX_Q_LEN, BLOCK_M)
    task_nums = task_nums * q_head
    task_nums = task_nums % AICORE_NUM
    MAX_Q_LEN = tl.where(task_nums == 0, MAX_Q_LEN + BLOCK_M, MAX_Q_LEN)

    NUM_BLOCKS_M = tl.cdiv(MAX_Q_LEN, BLOCK_M)  # 非对齐跳过
    NUM_BLOCKS = NUM_BLOCKS_M * BATCH_SIZE * q_head

    start_block, end_block, step = pid, NUM_BLOCKS, AICORE_NUM

    for block_idx in range(start_block, end_block, step):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        start_m = (block_idx % NUM_BLOCKS_M).to(tl.int32)
        start_b = task_hz_idx // q_head
        #if start_b % 2 == 1:
        #    start_m = NUM_BLOCKS_M - 1 - start_m
        start_qh = task_hz_idx % q_head
        start_kvh = start_qh * kv_head // q_head

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
                offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

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
                q_attn_arg_block_ptr = tl.make_block_ptr(
                    base = q_attn_arg_ptr + q_start,
                    shape = (q_len,),
                    strides = (1,),
                    offsets = (start_m * BLOCK_M,),
                    block_shape = (BLOCK_M,),
                    order = (0,)
                )
                k_attn_arg_block_ptr = tl.make_block_ptr(
                    base = k_attn_arg_ptr + k_start,
                    shape = (k_len,),
                    strides = (1,),
                    offsets = (begin,),
                    block_shape = (BLOCK_N,),
                    order = (0,)
                )

                acc = tl.zeros((BLOCK_M, V_DIM), dtype=tl.float32)
                m = tl.full((BLOCK_M,), value=-2 ** 30, dtype=tl.float32)
                l = tl.zeros((BLOCK_M,), dtype=tl.float32)

                q = load_if(q_block_ptr, False, True)
                q_attn_arg = load_if(q_attn_arg_block_ptr, False, True)

                for start_n in range(begin, end, BLOCK_N):
                    start_n = tl.multiple_of(start_n, BLOCK_N)
                    k_attn_arg = load_if(k_attn_arg_block_ptr, False, True)
                    offset_n = start_n + tl.arange(0, BLOCK_N)
                    # mask = load_if(mask_block_ptr, False, False)  # (32x32) -> (128, 128) --> 0

                    mask = mask_fn(q_attn_arg, k_attn_arg, offset_m, offset_n, MASK_FN)

                    # if not SPARSE_OPT or tl.sum(mask.cast(tl.int32)) != 0:
                    if not SPARSE_OPT:
                        k = load_if(k_block_ptr, True, False)  # npu行为：先装置后mask
                        # v = load_if(v_block_ptr, False, True)
                        s = tl.dot(q, k)
                        s = s * qk_scale
                        boundary_mask = (offset_n < k_len)[None, :]
                        s = tl.where(mask & boundary_mask, s, -2**30)
                        # mask_value = tl.where(boundary_mask & mask, 0, -2**30)
                        # s = s * qk_scale + mask_value
                        # s = s * qk_scale + tl.where(mask, 0.0, -2.0 ** 30)
                        # m_new = tl.maximum(m, tl.max(s, 1))
                        # 1.128 *128: nomak test: ub overflow, requires 2263552 bits while 2031616 bits available!: 777.79 -> 616
                        # 1.1 128 * 128, triu_causal - 客户mask, 2111->850
                        # 1.2 128 * 128, 不带nan, mask.to(int32) 精度问题 -->未复现
                        # 1.3 128 * 128, 不带nan, 不带mask.to(int32) 2330 --> 1350(使用membar新包)
                        # 2. 64* 128: 2500 ->1375(带nan) -> 1242.143（mask.to(int32)
                        m_new = tl.maximum(m, tl.max(s, 1, propagate_nan=True), propagate_nan=tl.PropagateNan.ALL)
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
                    k_attn_arg_block_ptr = tl.advance(k_attn_arg_block_ptr, (BLOCK_N,))

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

            k = load_if(k_block_ptr, False, True)
            v = load_if(v_block_ptr, False, True)
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
                    v = load_if(v_block_ptr, False, True)
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


@triton.autotune(
    list(filter(keep, get_bwd_qkv_configs())), 
    key = ["QK_DIM", "V_DIM", "MASK_FN", "SPARSE_OPT"],
    reset_to_zero = ["dq_ptr"]
)
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

            k = load_if(k_block_ptr, False, True)
            v = load_if(v_block_ptr, False, True)
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
                    # dv
                    dv += tl.dot(tl.trans(p), do)
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
                    extension.compile_hint(ds, "break_vf")
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
AICORE_NUM = properties["num_aicore"]
VECTOR_NUM = properties["num_vectorcore"]


# q: [total_q_seq, head, dim]
# k: [total_kv_seq, head, dim]
class FlashAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, q_attn_arg, k_attn_arg,
                cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale,
                mask_fn, sparse_opt, use_fused_bwd_qkv=True):
        q_len, q_head, qk_dim = q.shape
        k_len, kv_head, v_dim = v.shape
        batch_size = cu_seqlens_q.shape[0] - 1
        o = q.new_empty(q_len, q_head, v_dim)
        l = q.new_empty(q_len, q_head, dtype=torch.float32)
        #if ((max_seqlen_q + 63) // 64 * q_head) % AICORE_NUM == 0:
        #    max_seqlen_q = max_seqlen_q + 64
        print(f"===>max_seqlen_q: {max_seqlen_q}, max_seqlen_k: {max_seqlen_k}")
        NUM_CORES = AICORE_NUM
        grid = (NUM_CORES,)
        # breakpoint()
        fwd_kernel[grid](
            q, k, v, o, l,
            q_attn_arg.to(torch.int32), k_attn_arg.to(torch.int32),
            cu_seqlens_q.to(torch.int32), cu_seqlens_k.to(torch.int32),
            q_head, kv_head, scale,
            QK_DIM=qk_dim,
            V_DIM=v_dim,
            MASK_FN=mask_fn,
            SPARSE_OPT=sparse_opt,
            DTYPE=(19 if q.dtype == torch.float16 else 14),
            BLOCK_M=64,
            BLOCK_N=128,
            AICORE_NUM=NUM_CORES,
            MAX_Q_LEN=max_seqlen_q,
            MAX_K_LEN=max_seqlen_k,
            BATCH_SIZE=batch_size,
            q_len_total = q_len,
            k_len_total = k_len,
            multibuffer=True,
            # num_stages=2,
            enable_mixed_cv=True,
            enable_auto_bind_sub_block=True,
            sync_solver=True,
            limit_auto_multi_buffer_of_local_buffer="no-limit",
            # limit_auto_multi_buffer_buffer="no-limit",
            enable_flatten=False,
            set_workspace_multibuffer=2,
        )
        ctx.save_for_backward(q, k, v, o, l, q_attn_arg, k_attn_arg, cu_seqlens_q, cu_seqlens_k)
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
        if use_fused_bwd_qkv:
            dq = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
        else:
            dq = torch.empty_like(q)
        dk = k.new_empty(ctx.k_len, ctx.q_head, ctx.qk_dim)
        dv = v.new_empty(ctx.k_len, ctx.q_head, ctx.v_dim)
        d = torch.empty_like(l)

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
            multibuffer = True,
            limit_auto_multi_buffer_of_local_buffer = "no-limit",
        )
        # print(f"{d.sum()=}", flush=True)

        # if ctx.v_dim > 64:
        #     BLOCK_M = 128
        # else:
        #     BLOCK_M = 256
        # NUM_CORES = VECTOR_NUM
        # NUM_BLOCKS_M = triton.cdiv(ctx.max_seqlen_q, BLOCK_M)
        # NUM_BLOCKS = NUM_BLOCKS_M * ctx.q_head * ctx.batch_size
        # bwd_preprocess[(NUM_CORES,)](
        #     o, do, d,
        #     cu_seqlens_q,
        #     ctx.q_head,
        #     V_DIM = ctx.v_dim,
        #     DTYPE = ctx.dtype,
        #     # inject_barrier_all=True,
        #     limit_auto_multi_buffer_of_local_buffer="no-limit",
        #     enable_flatten=False,
        #     BLOCK_M=BLOCK_M,
        #     NUM_BLOCKS_M=NUM_BLOCKS_M,
        #     NUM_BLOCKS=NUM_BLOCKS,
        #     NUM_CORES=NUM_CORES,
        #     multibuffer=True,
        # )
        # print(f"{d.sum()=}", flush=True)

        if use_fused_bwd_qkv:
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
                # inject_barrier_all=True,
                limit_auto_multi_buffer_of_local_buffer="no-limit",
                enable_flatten=False,
                sync_solver=True,
                enable_mixed_cv=True,
            )
            dq = dq.to(q.dtype)
        else:
            BLOCK_M = 128
            BLOCK_N = 128
            NUM_CORES = AICORE_NUM
            NUM_BLOCKS_N= triton.cdiv(ctx.max_seqlen_k, BLOCK_N)
            NUM_BLOCKS = NUM_BLOCKS_N * ctx.q_head * ctx.batch_size
            bwd_kv_kernel[(NUM_CORES,)](
                q, k, v, dk, dv, do, l, d,
                q_attn_arg, k_attn_arg, mask_tensor,
                cu_seqlens_q, cu_seqlens_k,
                ctx.q_head, ctx.kv_head, ctx.scale,
                QK_DIM=ctx.qk_dim,
                V_DIM=ctx.v_dim,
                MASK_FN=ctx.mask_fn,
                SPARSE_OPT=ctx.sparse_opt,
                DTYPE=ctx.dtype,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                NUM_BLOCKS_N=NUM_BLOCKS_N,
                NUM_BLOCKS=NUM_BLOCKS,
                NUM_CORES=NUM_CORES,
                MAX_Q_LEN=ctx.max_seqlen_q,
                MAX_K_LEN=ctx.max_seqlen_k,
                # inject_barrier_all=True,
                limit_auto_multi_buffer_of_local_buffer="no-limit",
                enable_flatten=False,
                sync_solver=True,
                enable_mixed_cv=True,
            )

            BLOCK_M = 128
            BLOCK_N = 128
            NUM_CORES = AICORE_NUM
            NUM_BLOCKS_M = triton.cdiv(ctx.max_seqlen_q, BLOCK_M)
            NUM_BLOCKS = NUM_BLOCKS_M * ctx.q_head * ctx.batch_size
            bwd_q_kernel[(NUM_CORES,)](
                q, k, v, dq, do, l, d,
                q_attn_arg, k_attn_arg, mask_tensor,
                cu_seqlens_q, cu_seqlens_k,
                ctx.q_head, ctx.kv_head, ctx.scale,
                QK_DIM=ctx.qk_dim,
                V_DIM=ctx.v_dim,
                MASK_FN=ctx.mask_fn,
                SPARSE_OPT=ctx.sparse_opt,
                DTYPE=ctx.dtype,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                NUM_BLOCKS_M=NUM_BLOCKS_M,
                NUM_BLOCKS=NUM_BLOCKS,
                NUM_CORES=NUM_CORES,
                MAX_Q_LEN=ctx.max_seqlen_q,
                MAX_K_LEN=ctx.max_seqlen_k,
                limit_auto_multi_buffer_of_local_buffer="no-limit",
                enable_flatten=False,
                sync_solver=True,
                enable_mixed_cv=True
            )
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
    device = "npu"

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

def test_case():
    # /home/yangj/serving/qianchuan_huawei/fa_input.pt
    # /home/huawei0424/qianchuan_huawei_yixuan/fa_input_lm.pt
    # /home/huawei0424/qianchuan_huawei_rjw/triton_ops/kernels/fa_simulator.pt
    # data = torch.load("/home/huawei0424/qianchuan_huawei_rjw/triton_ops/kernels/fa_simulator_task56.pt", weights_only=False)
    data = torch.load("/home/huawei0424/qianchuan_huawei_yixuan/fa_input_lm.pt")
    q = data["q"].to(torch.float16)
    k = data["k"].to(torch.float16)
    v = data["v"].to(torch.float16)
    q_attn_arg = data["q_attn_arg"].to(torch.int32)
    k_attn_arg = data["k_attn_arg"].to(torch.int32)
    cu_seqlens_q = data["q_cumsum"].to(torch.int32)
    cu_seqlens_k = data["k_cumsum"].to(torch.int32)
    max_seqlen_q = 2689 #data["q_max"]
    max_seqlen_k = 2689 #data["k_max"]
    scale = data["scale"]
    # mask_fn = data["mask_fn"]
    seqlen = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    max_len = seqlen.cpu().max().item()
    print(f"{q.shape=}, {q_attn_arg.shape=}, {cu_seqlens_q.shape=}, {max_seqlen_q=}, {mask_fn=}, {max_len=}, {seqlen=}, {cu_seqlens_q=}")
    result = FlashAttentionFunc.apply(
                q,
                k,
                v,
                q_attn_arg,
                k_attn_arg,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                scale,
                1,
                False,
            )
    torch.npu.synchronize()
    print(f"{result.shape=}")
    print(f"\n======================== prof begin ====================")
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
    )

    with torch_npu.profiler.profile(
            activities=[  # torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU],
            with_stack=False,  # 采集torch 算子的函数调用栈的开关，该参数选填，默认关闭
            record_shapes=False,  # 采集torch 算子的input shape和input type的开关，该参数选填，默认关闭
            profile_memory=False,  # 采集memory相关数据的开关，该参数选填，默认关闭
            schedule=torch_npu.profiler.schedule(wait=1,
                                                 warmup=1,
                                                 active=30,
                                                 repeat=1,
                                                 skip_first=1),
            # warmup默认为0，老版本torch_npu包该参数为必填项
            experimental_config=experimental_config,  # 该参数选填，默认为Level0
            # 产生的profling文件的位置
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./prof_dir")
    ) as prof:
        # prof.start()
        for i in range(10):
            #mask_tensor_new = generate_mask_fn_vectorized(cu_seqlens_q, cu_seqlens_k, bs, max_seqlen_q, max_seqlen_k, q_attn_arg, k_attn_arg)
            #mask_tensor_new = mask_tensor_new.npu()
            #print(f"{mask_tensor_new.shape=}")
            torch.npu.synchronize()
            result = FlashAttentionFunc.apply(
                q,
                k,
                v,
                q_attn_arg,
                k_attn_arg,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                scale,
                1,
                False,
            )
            # result.backward(do)
            torch.npu.synchronize()  # 确保 kernel 真正执行完
            prof.step()
    print(f"======================== prof end ====================")


def test_main():
    # input params
    dtype = torch.bfloat16
    DEVICE = torch.device("npu")
    _dir = 'fa_v6'
    # 291526, 8, 64
    num_heads = [1, 8,8,16,16,8,4]
    head_dims = [64, 64,64,32,32,64,128]
    seqs = [2048, 2432,320,32,64,32,118]
    for i in range(1):
        num_head = num_heads[i]
        head_dim = head_dims[i]
        q_seq_list = [0] + [seqs[i]] * 1  # 0 for cumsum
        k_seq_list = [0] + [seqs[i]] * 1  # 0 for cumsum

        bs = len(q_seq_list) - 1
        q_len = sum(q_seq_list)
        k_len = sum(k_seq_list)
        max_seqlen_q = max(q_seq_list)
        max_seqlen_k = max(k_seq_list)
        q = torch.randn((q_len, num_head, head_dim), dtype=dtype, device=DEVICE).requires_grad_()
        k = torch.randn((k_len, num_head, head_dim), dtype=dtype, device=DEVICE).requires_grad_()
        v = torch.randn((k_len, num_head, head_dim), dtype=dtype, device=DEVICE).requires_grad_()

        q_attn_arg = torch.zeros(q_len, dtype=torch.int32, device="cpu")
        k_attn_arg = torch.zeros(k_len, dtype=torch.int32, device="cpu")

        cu_seqlens_q = np.cumsum(q_seq_list).tolist()
        cu_seqlens_k = np.cumsum(k_seq_list).tolist()
        print(f"{cu_seqlens_q=}")
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cpu")
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device="cpu")

        # BLOCK_M, BLOCK_N = 128, 128
        mask_tensor = generate_mask_fn_vectorized(q_seq_list[1:], k_seq_list[1:], bs, max_seqlen_q, max_seqlen_k, q_attn_arg, k_attn_arg)
        print(f"{mask_tensor.shape=}")

        q_attn_arg = q_attn_arg.npu()
        k_attn_arg = k_attn_arg.npu()
        cu_seqlens_q = cu_seqlens_q.npu()
        cu_seqlens_k = cu_seqlens_k.npu()
        mask_tensor = mask_tensor.npu()
        scale = 1.0 / (head_dim ** 0.5)
        #torch.save({"q": q.data, "k": k.data, "v": v.data, "atten_mask": mask_tensor.data \
        #        , "actual_seq_lengths": cu_seqlens_q[1:], "actual_seq_lengths_kv": cu_seqlens_k[1:] \
        #        , "num_heads": q.shape[1], "scale": scale}, "fia_input_seq2048.pt")
        torch.save({"q": q.data, "k": k.data, "v": v.data, "q_attn_arg": q_attn_arg, "k_attn_arg": k_attn_arg \
                , "cu_seqlen_q": cu_seqlens_q, "cu_seqlen_k": cu_seqlens_k, "max_seqlen_q": max_seqlen_q, "max_seqlen_k": max_seqlen_k \
                , "scale": scale}, "fa_input_seq2048.pt")
        rtol, atol = 5e-3, 5e-3
        print(f"\n======================== fwd acc begin ====================")
        result = FlashAttentionFunc.apply(
            q,
            k,
            v,
            q_attn_arg,
            k_attn_arg,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            1,
            False,
        )
        torch.npu.synchronize()


def test_case_ascendc():
    # /home/yangj/serving/qianchuan_huawei/fa_input.pt
    # /home/huawei0424/qianchuan_huawei_yixuan/fa_input_lm.pt
    # /home/huawei0424/qianchuan_huawei_rjw/triton_ops/kernels/fa_simulator.pt
    # data = torch.load("/home/huawei0424/qianchuan_huawei_rjw/triton_ops/kernels/fa_simulator_task56.pt", weights_only=False)
    """
    data = torch.load("/home/huawei0424/qianchuan_huawei_yixuan/fa_input_lm.pt")
    q = data["q"].to(torch.float16)
    k = data["k"].to(torch.float16)
    v = data["v"].to(torch.float16)
    q_attn_arg = data["q_attn_arg"].to(torch.int32)
    k_attn_arg = data["k_attn_arg"].to(torch.int32)
    cu_seqlens_q = data["q_cumsum"].to(torch.int32)
    cu_seqlens_k = data["k_cumsum"].to(torch.int32)
    max_seqlen_q = data["q_max"]
    max_seqlen_k = data["k_max"]
    scale = data["scale"]
    bs = cu_seqlens_q.shape[0] - 1
    # mask_fn = data["mask_fn"]
    seqlen = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    max_len = seqlen.cpu().max().item()
    print(f"{q.shape=}, {q_attn_arg.shape=}, {cu_seqlens_q.shape=}, {max_seqlen_q=}, {mask_fn=}, {max_len=}, {seqlen=}, {cu_seqlens_q=}")

    # sparse_mode = 3
    mask_tensor = generate_mask_fn_vectorized(seqlen, seqlen, bs, max_seqlen_q, max_seqlen_k, q_attn_arg, k_attn_arg)
    sparse_mode = 1

    ref_out = torch_npu.npu_fused_infer_attention_score(q, k, v,
                                        atten_mask=mask_tensor,
                                        scale=scale,
                                        sparse_mode=sparse_mode,
                                        input_layout="TND",
                                        num_heads=q.shape[1],
                                        next_tokens=0,
                                        actual_seq_lengths=cu_seqlens_q[1:],
                                        actual_seq_lengths_kv=cu_seqlens_k[1:])[0]
    """

    attn_mask = torch.triu(torch.ones(2048, 2048, device="cpu"), diagonal=1).bool()
    data = torch.load("/home/huawei0424/rjw/qianchuan_huawei_main/triton_ops/kernels/fia_input_seq2048.pt", weights_only=False)
    q = data["q"].to(torch.float16)
    k = data["k"].to(torch.float16)
    v = data["v"].to(torch.float16)
    mask_tensor = data["atten_mask"].to(torch.bool)
    actual_seq_lengths = data["actual_seq_lengths"].to(torch.int32)
    actual_seq_lengths_kv = data["actual_seq_lengths_kv"].to(torch.int32)
    scale = data["scale"]
    ref_out = torch_npu.npu_fused_infer_attention_score(q, k, v,
                                            atten_mask=attn_mask.npu(),
                                            scale=scale,
                                            sparse_mode=3,
                                            input_layout="TND",
                                            num_heads=q.shape[1],
                                            next_tokens=0,
                                            actual_seq_lengths=actual_seq_lengths,
                                            actual_seq_lengths_kv=actual_seq_lengths_kv)[0]
    #"""
    #torch.save({"q": q.data, "k": k.data, "v": v.data, "atten_mask": mask_tensor.data \
    #        , "actual_seq_lengths": cu_seqlens_q[1:], "actual_seq_lengths_kv": cu_seqlens_k[1:] \
    #        , "num_heads": q.shape[1], "scale": scale}, "fia_input.pt")
    torch.npu.synchronize()
    print(f"{ref_out.shape=}")

def test_varlen_case():  # 10288.268555 (2689) -> 自适应2399：10.6ms  -> (cann：11.1ms) -> mask load/vf: 10774.582031
    # input params
    dtype = torch.bfloat16
    DEVICE = torch.device("npu")
    
    q_cumsum = np.array([     0,    856,    879,    900,   1093,   2912,   3016,   4569,   5999,
          6384,   7961,   9233,  10667,  11006,  11972,  12000,  13436,  14675,
         14790,  16597,  18588,  19847,  21766,  24162,  25817,  27342,  28678,
         29266,  30446,  31398,  33586,  34596,  35653,  35887,  38168,  39758,
         40029,  40311,  42155,  42202,  43291,  43853,  44179,  45726,  46452,
         48794,  49105,  50795,  51637,  53378,  54993,  55394,  55587,  57620,
         59536,  61283,  63182,  63512,  65452,  65893,  67187,  67825,  68011,
         68336,  70046,  71568,  72851,  74855,  75578,  76079,  77199,  77692,
         78582,  79742,  81670,  83082,  83927,  85238,  87365,  87871,  89364,
         89885,  90358,  92610,  94055,  94362,  95971,  96161,  97401,  99217,
         99268, 100330, 101601, 103571, 103625, 105593, 107450, 109650, 110659,
        112381, 114410, 114446, 115427, 117345, 118386, 119475, 120617, 121652,
        123996, 124751, 125013, 126082, 127651, 130004, 132352, 132767, 132847,
        133609, 134267, 134488, 135708, 138031, 139296, 140622, 141561, 142308,
        142713, 143886, 143931, 144960, 146407, 146809, 147175, 148152, 149172,
        150348, 152693, 153556, 154122, 155658, 157245, 157955, 159130, 159534,
        159816, 161075, 162644, 162990, 165221, 165845, 166845, 168731, 170287,
        170351, 171934, 172121, 173460, 174002, 175866, 176838, 178065, 178722,
        180749, 181912, 183399, 183594, 183722, 184132, 184825, 185276, 187100,
        187645, 190004, 190676, 191492, 193630, 194512, 195604, 196546, 196651,
        198266, 198990, 200550, 202261, 204337, 206736, 207433, 209172, 211224,
        212827, 213752, 213866, 215875, 216534, 218151, 220069, 220527, 222176,
        223997, 225973, 227021, 227928, 230264, 231197, 231988, 232091, 233374,
        234415, 234883, 236873, 239190, 241027, 242356, 243989, 245793, 246326,
        246661, 246898, 248464, 248872, 250293, 250587, 252758, 253477, 254503,
        255231, 257250, 259242, 260156, 260542, 261212, 262637, 263555, 263772,
        265998, 266051, 266234, 266966, 267718, 268964, 271004, 273031, 273235,
        274588, 275102, 275680, 277695, 278585, 278978, 280224, 282023, 283694,
        285586, 287605, 288770, 289261, 291526])
    '''
    q_cumsum = np.array([     0,    463,   2330,   3521,   4960,   5918,   6973,   9215,  10091,
         10529,  10591,  12058,  13186,  14492,  15484,  15875,  17326,  19330,
         20832,  21249,  21467,  23749,  25115,  27081,  28221,  28419,  29676,
         31842,  32608,  33117,  34406,  35596,  37527,  38160,  39257,  40427,
         40595,  42973,  43952,  46268,  46563,  48286,  48420,  48449,  49116,
         51453,  53598,  53704,  54189,  55904,  56680,  58662,  59044,  59484,
         59925,  60020,  60084,  61369,  62624,  63784,  65843,  67178,  68680,
         70634,  71083,  72637,  72999,  73838,  74906,  75554,  77535,  79793,
         80965,  82341,  83166,  83315,  83337,  85564,  86374,  86389,  86568,
         88143,  88876,  89035,  89742,  91843,  93824,  94282,  95991,  97067,
         97253,  99064, 101312, 103402, 103986, 104500, 105580, 107597, 109583,
        111512, 111809, 114173, 115579, 116445, 118602, 120831, 122151, 122776,
        123673, 124803, 125001, 125133, 125544, 126841, 128670, 129087, 130045,
        130812, 131235, 133636, 135457, 135737, 136830, 138444, 138994, 139160,
        141542, 142724, 144386, 145051, 146032, 147311, 147653, 149451, 149772,
        150505, 150648, 153029, 154995, 155380, 157645, 159785, 160403, 162297,
        163216, 164985, 166990, 167340, 168905, 169671, 171720, 172251, 173404,
        175525, 175670, 176255, 176733, 178871, 180524, 181613, 181749, 182301,
        182902, 183092, 184914, 185380, 187647, 188084, 189903, 192252, 193518,
        194474, 194565, 196420, 197195, 198007, 198029, 199099, 200951, 202150,
        202506, 204479, 204759, 206946, 208447, 208690, 209367, 210233, 212356,
        213698, 214041, 214505, 215606, 217607, 219068, 221204, 221702, 222240,
        222539, 222947, 225062, 226600, 227443, 227505, 227927, 228475, 228490,
        230634, 232442, 232856, 232914, 234978, 236459, 238584, 240928, 243036,
        245280, 247259, 247844, 248147, 248581, 249515, 250101, 252259, 253760,
        254190, 255907, 256717, 256922, 258810, 259963, 260523, 262468, 263924,
        265354, 267747, 268063, 269916, 271456, 273288, 274241, 276418, 277536,
        278313, 280244, 282231, 283291, 283735, 284067, 284194, 286084, 287745,
        289456, 291706, 292411, 293000, 293184])
    '''
    num_head = 8
    head_dim = 64
    q_seq_list = q_cumsum[1:] - q_cumsum[:-1] # [0] + [320] * 8  # 0 for cumsum
    k_seq_list = q_seq_list # [0] + [320] * 8  # 0 for cumsum

    # q_seq_list = sorted(q_seq_list)
    # k_seq_list = sorted(k_seq_list)
    # q_cumsum = np.cumsum([0] + q_seq_list)

    bs = len(q_seq_list) - 1
    q_len = sum(q_seq_list)
    k_len = sum(k_seq_list)
    max_seqlen_q = int(np.max(q_seq_list))
    max_seqlen_k = int(np.max(k_seq_list))

    print(f"==>{bs=}, {max_seqlen_q=}, {q_seq_list=}, {len(q_cumsum)=}, {q_cumsum=}")
    max_seqlen_q = 2656 # 2656/2900
    max_seqlen_k = 2656 #2689
    # qkv
    q = torch.randn((q_len, num_head, head_dim), dtype=torch.float16, device=DEVICE)
    k = torch.randn((q_len, num_head, head_dim), dtype=torch.float16, device=DEVICE)
    v = torch.randn((q_len, num_head, head_dim), dtype=torch.float16, device=DEVICE)
    # data = torch.load("/home/huawei0424/qianchuan_huawei_yixuan/fa_input_lm.pt")
    # q = data["q"]
    # k = data["k"]
    # v = data["v"]

    q_attn_arg = torch.zeros(q_len, dtype=torch.int32, device="cpu")
    q_attn_arg[0] = 1
    k_attn_arg = torch.zeros(k_len, dtype=torch.int32, device="cpu")
    k_attn_arg[0] = 1
    cu_seqlens_q = q_cumsum.tolist()
    cu_seqlens_k = q_cumsum.tolist()
    cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cpu")
    cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device="cpu")

    # BLOCK_M, BLOCK_N = 128, 128
    q_attn_arg = q_attn_arg.npu()
    k_attn_arg = k_attn_arg.npu()
    cu_seqlens_q = cu_seqlens_q.npu()
    cu_seqlens_k = cu_seqlens_k.npu()       

    scale = 1.0 / (head_dim ** 0.5)

    rtol, atol = 5e-3, 5e-3
    print(f"\n======================== fwd acc begin ====================")
    result = FlashAttentionFunc.apply(
        q,
        k,
        v,
        q_attn_arg,
        k_attn_arg,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        scale,
        1,
        False,
    )
    print(f"======================== fwd acc end ====================")

    torch.npu.synchronize()
    print(f"\n======================== prof begin ====================")
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
    )

    with torch_npu.profiler.profile(
            activities=[  # torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU],
            with_stack=False,  # 采集torch 算子的函数调用栈的开关，该参数选填，默认关闭
            record_shapes=False,  # 采集torch 算子的input shape和input type的开关，该参数选填，默认关闭
            profile_memory=False,  # 采集memory相关数据的开关，该参数选填，默认关闭
            schedule=torch_npu.profiler.schedule(wait=1,
                                                 warmup=1,
                                                 active=30,
                                                 repeat=1,
                                                 skip_first=1),
            # warmup默认为0，老版本torch_npu包该参数为必填项
            experimental_config=experimental_config,  # 该参数选填，默认为Level0
            # 产生的profling文件的位置
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./prof_dir")
    ) as prof:
        # prof.start()
        for i in range(10):
            #mask_tensor_new = generate_mask_fn_vectorized(cu_seqlens_q, cu_seqlens_k, bs, max_seqlen_q, max_seqlen_k, q_attn_arg, k_attn_arg)
            #mask_tensor_new = mask_tensor_new.npu()
            #print(f"{mask_tensor_new.shape=}")
            torch.npu.synchronize()
            result = FlashAttentionFunc.apply(
                q,
                k,
                v,
                q_attn_arg,
                k_attn_arg,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                scale,
                1,
                False,
            )
            # result.backward(do)
            torch.npu.synchronize()  # 确保 kernel 真正执行完
            prof.step()
    print(f"======================== prof end ====================")


if __name__ == "__main__":
    # test_case()
    # test_case_ascendc()
    test_varlen_case()
    # test_main()

# 0527包测试：8.96ms
