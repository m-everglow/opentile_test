"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import pytest
import torch
import torch_npu
import triton
import triton.language as tl

DEVICE = "npu"

import triton.runtime.driver as driver
from typing import List

device = torch.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
AICORE_NUM = properties["num_aicore"]

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@triton.jit
def _attn_fwd_inner_fp8(acc, l_i, m_i, q, # q: fp8 
                    K_block_ptr, V_block_ptr, ATTEN_MASK, #
                    scale_q_value, scale_k_block_ptr, scale_v_block_ptr, # scales
                    stride_am, start_m, qk_scale: tl.constexpr,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr,
                    ):
    if STAGE == 1:
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        attn_mask_ptr = tl.make_block_ptr(
            base = ATTEN_MASK,
            shape=(N_CTX, N_CTX),
            strides=(stride_am, 1),
            offsets=(start_m * BLOCK_M, lo),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0)
        )
    else:
        lo, hi = 0, N_CTX
        
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)

      
        trans_k = tl.trans(k)
        qk = tl.dot(q, trans_k)
        # q, k dequant
        scale_k_value = tl.load(scale_k_block_ptr)
        qk = qk * scale_q_value
        qk = qk * scale_k_value
        # ------------------------------

        if STAGE == 2:
            curr_mask_ptr = attn_mask_ptr
            mask = tl.load(curr_mask_ptr)
            qk = qk * qk_scale + tl.where(mask, -1.0e4, 0)
            # TODO(CONVERTER_PROPAGATE_NAN): restore propagate_nan=True after the Converter upgrade.
            m_ij = tl.maximum(m_i, tl.max(qk, 1), propagate_nan=tl.PropagateNan.ALL)
            qk -= m_ij[:, None]
            attn_mask_ptr = tl.advance(attn_mask_ptr, (0, BLOCK_N))
        else:
            qk = qk * qk_scale
            # TODO(CONVERTER_PROPAGATE_NAN): restore propagate_nan=True after the Converter upgrade.
            m_ij = tl.maximum(m_i, tl.max(qk, 1), propagate_nan=tl.PropagateNan.ALL)
            qk = qk - m_ij[:, None]


        scale_k_block_ptr = tl.advance(scale_k_block_ptr, (1, 0))

        p = tl.math.exp(qk)
        p_cast = p.to(q.type) # p_cast: fp8
        v = tl.load(V_block_ptr)
        pv = tl.dot(p_cast, v)

        scale_v_value = tl.load(scale_v_block_ptr)
        pv = pv * scale_v_value
        scale_v_block_ptr = tl.advance(scale_v_block_ptr, (1, 0))

        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp(m_i - m_ij)
        #alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None] + pv

        m_i = m_ij 
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_fp8(Q, K, V, ATTEN_MASK, M, Out, sm_scale: tl.constexpr, sparse_start_idx, #
              SCALE_Q, SCALE_K, SCALE_V, #
              stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qk: tl.constexpr,  #
              stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kk: tl.constexpr,  #
              stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vk: tl.constexpr,  #
              stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr,  #
              stride_am: tl.constexpr, #
              stride_sqz: tl.constexpr, stride_sqh: tl.constexpr, stride_sqm: tl.constexpr, stride_sqk: tl.constexpr, #
              stride_skz: tl.constexpr, stride_skh: tl.constexpr,
              Z: tl.constexpr, H: tl.constexpr, 
              N_CTX: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              NUM_BLOCKS_PER_CORE: tl.constexpr,
              NUM_BLOCKS: tl.constexpr,
              NUM_BLOCKS_M: tl.constexpr,
              AICORE_NUM: tl.constexpr,
              PER_BLOCK_SIZE_Q: tl.constexpr,
              PER_BLOCK_SIZE_KV: tl.constexpr,
              ):
    pid = tl.program_id(0)
    block_start = pid * NUM_BLOCKS_PER_CORE
    NUM_BLOCKS_hz = NUM_BLOCKS // NUM_BLOCKS_M 
    task_m_idx = 0
    task_hz_idx = 0

    if STAGE == 3:
        # use indices calculated by load balancer in causal stage
        start_block = tl.load(sparse_start_idx + pid)
        end_block = tl.load(sparse_start_idx + pid + 1)
        step = 1
    else:
        start_block, end_block, step = pid, NUM_BLOCKS, AICORE_NUM
    
    for block_idx in range(start_block, end_block, step):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        scale_offset = off_z.to(tl.int64) * stride_sqz + off_h.to(tl.int64) * stride_sqh
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        scale_offset_q  = off_z.to(tl.int64) * stride_sqz + off_h.to(tl.int64) * stride_sqh
        scale_offset_kv = off_z.to(tl.int64) * stride_skz + off_h.to(tl.int64) * stride_skh
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),

            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,

            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vn, stride_vk),

            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        SCALE_Q_block_ptr = tl.make_block_ptr(
            base=SCALE_Q + scale_offset_q,
            shape=(N_CTX // PER_BLOCK_SIZE_Q, 1),
            strides=(stride_sqm, stride_sqk),
            offsets=(task_m_idx * BLOCK_M // PER_BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_M // PER_BLOCK_SIZE_Q, 1),
            order=(1, 0),
        )
        SCALE_K_block_ptr = tl.make_block_ptr(
            base=SCALE_K + scale_offset_kv,
            shape=(N_CTX // PER_BLOCK_SIZE_KV, 1),
            strides=(stride_sqm, stride_sqk),
            offsets=(0, 0),
            block_shape=(BLOCK_N // PER_BLOCK_SIZE_KV, 1),
            order=(1, 0),
        )
        SCALE_V_block_ptr = tl.make_block_ptr(
            base=SCALE_V + scale_offset_kv,
            shape=(N_CTX // PER_BLOCK_SIZE_KV, 1),
            strides=(stride_sqm, stride_sqk),
            offsets=(0, 0),
            block_shape=(BLOCK_N // PER_BLOCK_SIZE_KV, 1),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        # initialize offsets
        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # load q: it will stay in SRAM throughout
        q = tl.load(Q_block_ptr)
        scale_q_value = tl.load(SCALE_Q_block_ptr)
        # stage 1: off-band
        # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
        # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE

        if STAGE & 1:
            acc, l_i, m_i = _attn_fwd_inner_fp8(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, ATTEN_MASK, #
                                            scale_q_value, SCALE_K_block_ptr, SCALE_V_block_ptr, #
                                            stride_am, task_m_idx, sm_scale,  #
                                            BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                            4 - STAGE, offs_m, offs_n, N_CTX  #
                                            )
        # stage 2: on-band

        if STAGE & 2:
            # barrier makes it easier for compielr to schedule the
            # two loops independently
            acc, l_i, m_i = _attn_fwd_inner_fp8(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, ATTEN_MASK, #
                                            scale_q_value, SCALE_K_block_ptr, SCALE_V_block_ptr, #
                                            stride_am, task_m_idx, sm_scale,  #
                                            BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                            2, offs_m, offs_n, N_CTX  #
                                            )
        # epilogue
        m_i += tl.math.log(l_i)
        acc = acc / l_i[:, None]
        m_ptrs = M + task_hz_idx * N_CTX + offs_m

        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))

# define a cache dict
SPARSE_INDEX_CACHE = {}

class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, scale_q, scale_k, scale_v, atten_mask, causal, sm_scale, BM, BN, PER_BLOCK_SIZE_Q, 
                PER_BLOCK_SIZE_KV):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        o = torch.empty_like(q.to(torch.float32))

        # stage = 3
        stage = 3 if causal else 1
        extra_kern_args = {}
        num_cores = AICORE_NUM
        NUM_BLOCKS_M = triton.cdiv(q.shape[2], BM)
        NUM_BLOCKS = NUM_BLOCKS_M * q.shape[0] * q.shape[1]
        NUM_BLOCKS_PER_CORE = triton.cdiv(NUM_BLOCKS, num_cores)
        cache_key = (q.shape[0], q.shape[1], q.shape[2], k.shape[2], BM, BN, causal, num_cores)

        if cache_key in SPARSE_INDEX_CACHE:
            sparse_start_idx = SPARSE_INDEX_CACHE[cache_key]
        else:
            sparse_valid_array = [0] * (q.shape[2] // BM)   
            sparse_start_idx = [NUM_BLOCKS] * (num_cores+1)
            init_sparse_valid_array(sparse_valid_array,k.shape[2],BM,BN,s1_sparse_valid_size=q.shape[2],s2_sparse_valid_size=0)
            set_sparse_start_idx(
                sparse_valid_array,
                sparse_start_idx,
                total_size=NUM_BLOCKS,
                core_num=num_cores,
                max_core_num=num_cores,
            )
            sparse_start_idx = torch.tensor(sparse_start_idx, device=q.device, dtype=torch.int32)
            SPARSE_INDEX_CACHE[cache_key] = sparse_start_idx

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        _attn_fwd_fp8[(num_cores,)](
            q, k, v, atten_mask, M, o, sm_scale,
            sparse_start_idx,
            scale_q, scale_k, scale_v,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[2],
            scale_q.stride(0), scale_q.stride(1), scale_q.stride(2), scale_q.stride(3),
            scale_k.stride(0), scale_k.stride(1),
            q.shape[0], q.shape[1], N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            BLOCK_M = BM,
            BLOCK_N = BN,
            STAGE=stage,
            NUM_BLOCKS_PER_CORE=NUM_BLOCKS_PER_CORE,
            NUM_BLOCKS=NUM_BLOCKS,
            NUM_BLOCKS_M=NUM_BLOCKS_M,
            AICORE_NUM = num_cores,
            PER_BLOCK_SIZE_Q=PER_BLOCK_SIZE_Q,
            PER_BLOCK_SIZE_KV=PER_BLOCK_SIZE_KV,
            num_stages=2,
            multibuffer=True,
            enable_auto_bind_sub_block=True,
            sync_solver=True,
            limit_auto_multi_buffer_of_local_buffer="no-limit",
            set_workspace_multibuffer = 2,
            **extra_kern_args)


        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

import math
attention = _attention.apply

def init_sparse_valid_array(sparse_valid_array: List[int],
    s2_size, BLOCK_S1, BLOCK_S2,s1_sparse_valid_size,s2_sparse_valid_size):
    INVALID_ROW_SPARSE_RATIO = 6
    assert len(sparse_valid_array) > 0 ,'init_sparse_valid_array: Sparse valid array size should be larger than 0.'

    s2_num_blocks = math.ceil(s2_size / BLOCK_S2)
    valid_s1_size = math.ceil(s1_sparse_valid_size / BLOCK_S1)
    valid_s2_size = math.ceil(s2_sparse_valid_size / BLOCK_S2)
    for i in range(len(sparse_valid_array)):
        reduce_blocks = 0 if i < valid_s1_size else (triton.cdiv((i + 1) * BLOCK_S1 - s1_sparse_valid_size , BLOCK_S2) - 1)
        add_blocks = min(s2_num_blocks - valid_s2_size,
            math.ceil(((i + 1) * BLOCK_S1 + s2_sparse_valid_size) / BLOCK_S2) - valid_s2_size)
        valid_block_num = valid_s2_size - reduce_blocks + add_blocks
        sparse_valid_array[i] = valid_block_num  if valid_block_num > 0 else INVALID_ROW_SPARSE_RATIO
    return True

def set_sparse_start_idx(sparse_valid_array,
                        sparse_start_idx,
                        total_size,
                        core_num,
                        max_core_num):
    """
    设置稀疏起始索引
    
    params:
        sparse_valid_array: 稀疏有效数组
        multi_core_params: 多核参数对象
        max_core_num: 最大核心数
    """
    valid_aiv_num = min(core_num, max_core_num)
    assert valid_aiv_num > 0,"valid_aiv_num should be greater than 0"

    for idx in range(max_core_num):
        sparse_start_idx[idx] = total_size

    if total_size <= valid_aiv_num:
        for idx in range(total_size):
            sparse_start_idx[idx] = idx
        return True

    sparse_array_sum = sum(sparse_valid_array)
    load_total = sparse_array_sum * (total_size // len(sparse_valid_array))
    partition_result = [total_size] * (valid_aiv_num + 1)
    last_valid_partition_result = [total_size] * (valid_aiv_num + 1)
    
    load_each_core_lower_bound = load_total // valid_aiv_num - 1
    max_sparse_valid = max(sparse_valid_array)
    load_each_core_upper_bound = triton.cdiv(load_total, valid_aiv_num) + max_sparse_valid

    
    # binary search for load-balancing
    while load_each_core_lower_bound + 1 < load_each_core_upper_bound:
        load_max = load_each_core_lower_bound + (load_each_core_upper_bound - load_each_core_lower_bound) // 2
        if (load_max * valid_aiv_num >= load_total and
            partition_sparse_data(sparse_valid_array, sparse_array_sum, 
                                 total_size, load_max, partition_result)):
            load_each_core_upper_bound = load_max
            last_valid_partition_result, partition_result = partition_result, last_valid_partition_result
        else:
            load_each_core_lower_bound = load_max

    for idx in range(valid_aiv_num):
        sparse_start_idx[idx] = last_valid_partition_result[idx]

def partition_sparse_data(sparse_rolling_array: List[int],
                         sparse_rolling_array_sum: int,
                         sparse_array_size: int,
                         load_max_each_core: int,
                         partition_result: List[int]) -> bool:
    """
    将稀疏数据分配到多个核心
    
    参数:
        sparse_rolling_array: 稀疏滚动数组 sparse_valid_
        sparse_rolling_array_sum: 稀疏滚动数组的和 sparse_valid_sum
        sparse_array_size: 稀疏数组大小
        load_max_each_core: 每个核心的最大负载
        partition_result: 分区结果（会被修改）
    
    返回:
        bool: 分配是否成功
    """

    assert partition_result,"partition_result should be greater than 0"
    assert sparse_rolling_array_sum > 0,"sparse_rolling_array_sum should be greater than 0"
    
    # 计算每个核心的基础负载
    s1_outer_cut_each_core = load_max_each_core // sparse_rolling_array_sum
    s1_outer_load_each_core = s1_outer_cut_each_core * sparse_rolling_array_sum
    s1_outer_num_each_core = s1_outer_cut_each_core * len(sparse_rolling_array)
    
    target_core_num = len(partition_result)
    core_idx = 0
    rolling_idx = 0
    load_size = s1_outer_load_each_core
    
    # 初始化第一个核心的起始位置
    partition_result[0] = 0
    
    # 分配剩余的数据到各个核心
    i = s1_outer_num_each_core
    while i < sparse_array_size:
        # 处理rolling_idx回绕
        if rolling_idx >= len(sparse_rolling_array):
            rolling_idx = 0

        load_next = sparse_rolling_array[rolling_idx]
        need_one_more_core = (load_size + load_next) > load_max_each_core
        
        if need_one_more_core and core_idx >= (target_core_num - 1):
            return False
        
        if need_one_more_core:
            core_idx += 1
            partition_result[core_idx] = i
            i += s1_outer_num_each_core - 1
            rolling_idx -= 1
            load_size = s1_outer_load_each_core
        else:
            load_size += load_next 
        i += 1
        rolling_idx += 1
    for j in range(core_idx + 1, target_core_num):
        partition_result[j] = sparse_array_size
    
    return True

