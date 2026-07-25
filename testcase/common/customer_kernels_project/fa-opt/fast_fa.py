import torch
import torch_npu
import triton
import triton.language as tl
import numpy as np

import triton.language.extra.cann.extension as al
import triton.extension.buffer.language as bl
import triton.runtime.driver as driver

try:
    from .fast_fa_vf import *
except ImportError:
    from fast_fa_vf import *

DEVICE = "npu"
debug = True

def get_npu_aicore_num():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)["num_aicore"]

@triton.jit
def vec_prefree_s_ub():
    al.sync_block_set("vector", "cube", 2 ,al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    al.sync_block_set("vector", "cube", 2 ,al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

@triton.jit
def vec_prefree_pv_ub():
    al.sync_block_set("vector", "cube", 10 ,al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    al.sync_block_set("vector", "cube", 10 ,al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

@triton.jit
def vec_postwait_p_l1():
    al.sync_block_wait("cube", "vector", 6 ,al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)
    al.sync_block_wait("cube", "vector", 6 ,al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)

@triton.jit
def cube_prefree_p_l1():
    al.sync_block_set("cube", "vector", 6 ,al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)
    al.sync_block_set("cube", "vector", 6 ,al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)

@triton.jit
def cube_postwait_s_ub():
    al.sync_block_wait("vector", "cube", 2 ,al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    al.sync_block_wait("vector", "cube", 2 ,al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

@triton.jit
def cube_postwait_pv_ub():
    al.sync_block_wait("vector", "cube", 10 ,al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    al.sync_block_wait("vector", "cube", 10 ,al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

@triton.jit
def _qk_matmul(q, K_block_ptr, qk_ub_ping, qk_ub_pong, qk_l0c, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr, sid, mask_q):
        k = tl.load(K_block_ptr, mask=mask_q)
        trans_k = tl.trans(k)
        qk = tl.dot(q, trans_k)
        bl.to_buffer(qk, bind_buffer=qk_l0c)
        al.sync_block_wait("vector", "cube", 2 ,al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

        if (sid & 1) == 0:
            qk_ub = bl.to_tensor(qk_ub_ping)
        else:
            qk_ub = bl.to_tensor(qk_ub_pong)

        al.fixpipe(qk, bl.to_buffer(qk_ub, al.ascend_address_space.UB), al.FixpipeDMAMode.NZ2ND, al.FixpipeDualDstMode.ROW_SPLIT)

        al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)

@triton.jit
def _pv_matmul(p_l1_ping, p_l1_pong, pv_ub_ping, pv_ub_pong, pv_l0c, V_block_ptr, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, pvid, mask_q):
    v = tl.load(V_block_ptr, mask=mask_q)

    #wait for vec to complete p's transfer
    al.sync_block_wait("vector", "cube", 4,al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE1)

    if (pvid & 1) == 0:
        p_l1 = bl.to_tensor(p_l1_ping,target_shape=[BLOCK_M, BLOCK_N])
        pv_ub = bl.to_tensor(pv_ub_ping)
    else:
        p_l1 = bl.to_tensor(p_l1_pong,target_shape=[BLOCK_M, BLOCK_N])
        pv_ub = bl.to_tensor(pv_ub_pong)

    pv = tl.dot(p_l1, v)
    bl.to_buffer(pv, bind_buffer=pv_l0c)

    # m free P buffer to vec
    al.sync_block_set("cube", "vector", 6, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)

    # fixpipe allocate PV buffer from vec
    al.sync_block_wait("vector", "cube", 10, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    al.fixpipe(pv, bl.to_buffer(pv_ub, al.ascend_address_space.UB), al.FixpipeDMAMode.NZ2ND,al.FixpipeDualDstMode.ROW_SPLIT)

    # fixpipe indicate to vec that PV transfer completes
    al.sync_block_set("cube", "vector", 8, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)

@triton.jit
def _flash_update(pv_ub_ping, pv_ub_pong, alpha_tb0, alpha_tb1, alpha_tb2, acc_buffer, v_s1_idx_mod3, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, pvid, update_acc):
    al.sync_block_wait("cube", "vector", 8, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
    if (pvid & 1) == 1:
        pv = bl.to_tensor(pv_ub_ping)
    else:
        pv = bl.to_tensor(pv_ub_pong)
    if (pvid-3) % 3 == 0:
        alpha = bl.to_tensor(alpha_tb0)
    elif (pvid-3) % 3 == 1:
        alpha = bl.to_tensor(alpha_tb1)
    else:
        alpha = bl.to_tensor(alpha_tb2)

    if update_acc:
        acc_tensor = bl.to_tensor(acc_buffer)
        acc_tensor = acc_tensor * alpha[:, None] + pv
        bl.to_buffer(acc_tensor, bind_buffer=acc_buffer)
    else:
        if (pvid & 1) == 1:
            al.copy(pv_ub_ping, acc_buffer)
        else:
            al.copy(pv_ub_pong, acc_buffer)

    al.sync_block_set("vector", "cube", 10, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

@triton.jit
def update_s2_loop(taskId_mod3, cur_s2_idx, s2_idx_1, s2_idx_2, s2_idx_3, s2_idx_4):
    if taskId_mod3 == 0:
        s2_idx_1, s2_idx_2, s2_idx_3, s2_idx_4 = cur_s2_idx, s2_idx_2, s2_idx_3, s2_idx_4
    elif taskId_mod3 == 1:
        s2_idx_1, s2_idx_2, s2_idx_3, s2_idx_4 = s2_idx_1, cur_s2_idx, s2_idx_3, s2_idx_4
    elif taskId_mod3 == 2:
        s2_idx_1, s2_idx_2, s2_idx_3, s2_idx_4 = s2_idx_1, s2_idx_2, cur_s2_idx, s2_idx_4
    else:
        s2_idx_1, s2_idx_2, s2_idx_3, s2_idx_4 = s2_idx_1, s2_idx_2, s2_idx_3, cur_s2_idx
    return s2_idx_1, s2_idx_2, s2_idx_3, s2_idx_4

@triton.jit
def is_need_update(taskId_mod3, s2_idx_1, s2_idx_2, s2_idx_3, s2_idx_4):
    if taskId_mod3 == 0:
        is_need = not s2_idx_1 == 0
    elif taskId_mod3 == 1:
        is_need = not s2_idx_2 == 0
    elif taskId_mod3 == 2:
        is_need = not s2_idx_3 == 0
    else:
        is_need = not s2_idx_4 == 0
    return is_need

@triton.jit
def is_last_skv(taskId_mod3, s2_idx_1, s2_size_1, s2_idx_2, s2_size_2, s2_idx_3, s2_size_3, s2_idx_4, s2_size_4):
    if taskId_mod3 == 0:
        is_reach = s2_idx_1 == s2_size_1 - 1
    elif taskId_mod3 == 1:
        is_reach = s2_idx_2 == s2_size_2 - 1
    elif taskId_mod3 == 2:
        is_reach = s2_idx_3 == s2_size_3 - 1
    else:
        is_reach = s2_idx_4 == s2_size_4 - 1
    return is_reach

@triton.jit
def is_first_skv_loop(taskId_mod3, s2_idx_1, s2_idx_2, s2_idx_3, s2_idx_4):
    if taskId_mod3 == 0:
        is_first = s2_idx_1 == 0
    elif taskId_mod3 == 1:
        is_first = s2_idx_2 == 0
    elif taskId_mod3 == 2:
        is_first = s2_idx_3 == 0
    else:
        is_first = s2_idx_4  == 0
    return is_first

@triton.jit
def get_s_offset(taskId_mod3, head_num, seqlen, stride,
                b_idx, n_idx, s_idx):
    s_offset = (b_idx.to(tl.int64) * head_num + n_idx.to(tl.int64)) * seqlen +  s_idx.to(tl.int64) * stride
    return s_offset.to(tl.int64)

@triton.jit
def get_cur_task(taskId_mod3,
                b_idx_1, n_idx_1, s1_idx_1, s2_idx_1, s2_size_1,
                b_idx_2, n_idx_2, s1_idx_2, s2_idx_2, s2_size_2,
                b_idx_3, n_idx_3, s1_idx_3, s2_idx_3, s2_size_3,
                b_idx_4, n_idx_4, s1_idx_4, s2_idx_4, s2_size_4):
    
    if taskId_mod3 == 0:
        cur_b_idx, cur_n_idx, cur_s1_idx, cur_s2_idx, cur_s2_size = b_idx_1, n_idx_1, s1_idx_1, s2_idx_1, s2_size_1
    elif taskId_mod3 == 1:
        cur_b_idx, cur_n_idx, cur_s1_idx, cur_s2_idx, cur_s2_size = b_idx_2, n_idx_2, s1_idx_2, s2_idx_2, s2_size_2
    elif taskId_mod3 == 2:
        cur_b_idx, cur_n_idx, cur_s1_idx, cur_s2_idx, cur_s2_size = b_idx_3, n_idx_3, s1_idx_3, s2_idx_3, s2_size_3
    else:
        cur_b_idx, cur_n_idx, cur_s1_idx, cur_s2_idx, cur_s2_size = b_idx_4, n_idx_4, s1_idx_4, s2_idx_4, s2_size_4
    return cur_b_idx, cur_n_idx, cur_s1_idx, cur_s2_idx, cur_s2_size

@triton.jit
def update_pos(s1_cur_idx, s1_step, batch_size, head_num, NUM_BLOCKS_M, STAGE, N_CTX, BLOCK_N, CAUSAL_TYPE: tl.constexpr):
    bn_idx = s1_cur_idx // NUM_BLOCKS_M         # total batch_offset * head_num_offset
    s1_idx = s1_cur_idx % NUM_BLOCKS_M          # remain s_offset
    b_idx = bn_idx // head_num                  # batch_offset
    n_idx = bn_idx % head_num                   # head_num_offset
    total_kv_blocks = (N_CTX + BLOCK_N - 1) // BLOCK_N
    if STAGE == 1:
        if CAUSAL_TYPE == 0:
            lo, hi = 0, s1_idx + 1
        else:
            lo, hi = s1_idx, total_kv_blocks
    else:
        lo, hi = 0, total_kv_blocks

    return b_idx, n_idx, s1_idx, lo, hi

@triton.jit
def update_task(taskId, task_cnt, v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4):
    s1_task_mod3 = task_cnt % 3
    if taskId & 3 == 0:
        v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4 = s1_task_mod3, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4
    elif taskId & 3 == 1:
        v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4 = v_s1_task_mod3_1, s1_task_mod3, v_s1_task_mod3_3, v_s1_task_mod3_4
    elif taskId & 3 == 2:
        v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4 = v_s1_task_mod3_1, v_s1_task_mod3_2, s1_task_mod3, v_s1_task_mod3_4
    else:
        v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4 = v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, s1_task_mod3

    return v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4

@triton.jit
def get_s_task(taskId, v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4):
    if taskId & 3 == 0:
        cur_s1_task_mod3= v_s1_task_mod3_1
    elif taskId & 3 == 1:
        cur_s1_task_mod3 = v_s1_task_mod3_2
    elif taskId & 3 == 2:
        cur_s1_task_mod3 = v_s1_task_mod3_3
    else:
        cur_s1_task_mod3 = v_s1_task_mod3_4

    return cur_s1_task_mod3

@triton.jit
def create_task(taskId, b_idx, n_idx, s1_idx, s2_idx, s2_size,
                b_idx_1, n_idx_1, s1_idx_1, s2_idx_1, s2_size_1,
                b_idx_2, n_idx_2, s1_idx_2, s2_idx_2, s2_size_2,
                b_idx_3, n_idx_3, s1_idx_3, s2_idx_3, s2_size_3,
                b_idx_4, n_idx_4, s1_idx_4, s2_idx_4, s2_size_4
                ):
    if taskId & 3 == 0:
        b_idx_1, n_idx_1, s1_idx_1, s2_idx_1, s2_size_1 = b_idx, n_idx, s1_idx, s2_idx, s2_size
    elif taskId & 3 == 1:
        b_idx_2, n_idx_2, s1_idx_2, s2_idx_2, s2_size_2 = b_idx, n_idx, s1_idx, s2_idx, s2_size
    elif taskId & 3 == 2:
        b_idx_3, n_idx_3, s1_idx_3, s2_idx_3, s2_size_3 = b_idx, n_idx, s1_idx, s2_idx, s2_size
    else:
        b_idx_4, n_idx_4, s1_idx_4, s2_idx_4, s2_size_4 = b_idx, n_idx, s1_idx, s2_idx, s2_size

    return (b_idx_1, n_idx_1, s1_idx_1, s2_idx_1, s2_size_1,
            b_idx_2, n_idx_2, s1_idx_2, s2_idx_2, s2_size_2,
            b_idx_3, n_idx_3, s1_idx_3, s2_idx_3, s2_size_3,
            b_idx_4, n_idx_4, s1_idx_4, s2_idx_4, s2_size_4)

@triton.jit
def create_and_get_basic_pos(s1_cur_idx, s1_step, batch, head_num, s1_start, NUM_BLOCKS_M, STAGE, N_CTX, BLOCK_N, CAUSAL_TYPE: tl.constexpr):
    b_idx, n_idx, s1_idx, s2_lo, s2_size = \
        update_pos(s1_cur_idx, s1_step, batch, head_num, NUM_BLOCKS_M, STAGE, N_CTX, BLOCK_N, CAUSAL_TYPE)

    return (b_idx, n_idx, s1_idx, s2_lo, s2_size)

@triton.jit
def _attn_fwd(
    Q, K, V, ATTEN_MASK, M, Out, 
    cu_seqlens_q, cu_seqlens_k,
    max_q_len, max_k_len,
    sparse_start_idx, sm_scale: tl.constexpr,
    stride_qt: tl.constexpr, stride_qn: tl.constexpr, stride_qd: tl.constexpr,
    stride_kt: tl.constexpr, stride_kn: tl.constexpr, stride_kd: tl.constexpr,
    stride_vt: tl.constexpr, stride_vn: tl.constexpr, stride_vd: tl.constexpr,
    stride_ot: tl.constexpr, stride_on: tl.constexpr, stride_od: tl.constexpr,
    stride_mask0: tl.constexpr, stride_mask1: tl.constexpr,
    Z: tl.constexpr, H: tl.constexpr, 
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    CAUSAL_TYPE: tl.constexpr,
    #NUM_BLOCKS_PER_CORE: tl.constexpr,
    #NUM_BLOCKS: tl.constexpr,
    #NUM_BLOCKS_M: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    ORI_MAX_Q_LEN = max_q_len
    ORI_MAX_K_LEN = max_k_len
    pid = tl.program_id(0)
    task_nums = tl.cdiv(max_q_len, BLOCK_M)
    #task_nums = task_nums * H
    #task_nums = task_nums % CORE_NUM
    #max_q_len = tl.where(task_nums == 0, max_q_len + BLOCK_M, max_q_len)
    align_num = 14
    task_nums = task_nums % align_num
    target_max = tl.cdiv(tl.cdiv(max_q_len, BLOCK_M), align_num)
    target_max = target_max * align_num * BLOCK_M + BLOCK_M
    max_q_len = tl.where(task_nums == 1, max_q_len, target_max)

    NUM_BLOCKS_M = tl.cdiv(max_q_len, BLOCK_M)  # 非对齐跳过
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H

    # warm up stage nums
    preload = 3
    # 输入数据类型 内部数据cast用
    cast_dtype = Q.dtype.element_ty

    # pingpong 数据分块
    qk_ub_ping = bl.alloc(tl.float32, [BLOCK_M // 2, BLOCK_N], al.ascend_address_space.UB)
    qk_ub_pong = bl.alloc(tl.float32, [BLOCK_M // 2, BLOCK_N], al.ascend_address_space.UB)

    p_l1_ping = bl.alloc(cast_dtype, [ BLOCK_N // 16, BLOCK_M // 16, 16, 16], al.ascend_address_space.L1)
    p_l1_pong = bl.alloc(cast_dtype, [ BLOCK_N // 16, BLOCK_M // 16, 16, 16], al.ascend_address_space.L1)

    pv_ub_ping = bl.alloc(tl.float32, [BLOCK_M // 2, HEAD_DIM], al.ascend_address_space.UB, is_mem_unique=True)
    pv_ub_pong = bl.alloc(tl.float32, [ BLOCK_M // 2, HEAD_DIM], al.ascend_address_space.UB, is_mem_unique=True)

    # 外层Sq循环控制信息 结合 {preload}大小 处理cool down块
    multi_core_limit = NUM_BLOCKS
    last_loop = 0
    last_second_loop = 0
    last_third_loop = 0
    if True or STAGE == 0:
        last_third_loop = (NUM_BLOCKS - pid + CORE_NUM - 1)  // CORE_NUM * CORE_NUM + pid # 84
        last_second_loop = last_third_loop + CORE_NUM # 112
        last_loop = last_second_loop + CORE_NUM # 140
        multi_core_limit += 3 * CORE_NUM # 57 + 84 = 141
        start_block, end_block, step = pid, multi_core_limit, CORE_NUM  # (0, 141, 28)

    with al.scope(core_mode="cube"):
        '''
        cube keeps own taskInfo[4]
        mark cur_task's batch_idx, head_num_idx, seq_q_idx, seq_kv_idx, seq_kv_loop_size
        c1 always use task_info[cur_idx] as task producer
        c2 always use task_info[cur_idx-2] as task consumer
        the Deferred Consumption masks synchronization issues
        task_id marks the task_idx as (task_id - consumer_stage)
        TODO: use class to Optimize when triton-ascend supports the feature :>
        '''
        b_idx_1, n_idx_1, s1_idx_1, s2_idx_1, s2_size_1 = 0, 0, 0, 0, 0 # task_1 V1
        b_idx_2, n_idx_2, s1_idx_2, s2_idx_2, s2_size_2 = 0, 0, 0, 0, 0 # task_2 C1
        b_idx_3, n_idx_3, s1_idx_3, s2_idx_3, s2_size_3 = 0, 0, 0, 0, 0
        b_idx_4, n_idx_4, s1_idx_4, s2_idx_4, s2_size_4 = 0, 0, 0, 0, 0
        cur_b_idx, cur_n_idx, cur_s1_idx, cur_s2_lo, cur_s2_size = 0, 0, 0, 0, 0
        taskId = 0

        cube_prefree_p_l1()
        # =================== Q sequence length loop ===================
        for sq_loop_idx in range(start_block, end_block, step):  # (0, 28, 56, 84, 112, 140)
            # =================== 判断是否cool down阶段 ===================
            is_last_loop = sq_loop_idx == last_loop  # false
            is_last_second_loop = sq_loop_idx == last_second_loop 
            is_last_third_loop = sq_loop_idx == last_third_loop
            not_last = not is_last_loop
            not_last_two = not is_last_loop and not is_last_second_loop
            not_last_three = (not is_last_loop and not is_last_second_loop) and not is_last_third_loop

            # =================== 获取 producer 信息 ===================
            if not_last_three:
                (cur_b_idx, cur_n_idx, cur_s1_idx, cur_s2_lo, cur_s2_size) = \
                    create_and_get_basic_pos(
                        sq_loop_idx, CORE_NUM, Z, H, pid, NUM_BLOCKS_M, STAGE, max_q_len, BLOCK_N, CAUSAL_TYPE
                    )

            q_len_start = tl.load(cu_seqlens_q + cur_b_idx)
            q_len_end = tl.load(cu_seqlens_q + cur_b_idx + 1)
            cur_q_len = q_len_end - q_len_start
            # TODO: cross_attn q_len != k_len
            k_len_start = q_len_start
            k_len_end = q_len_end
            cur_k_len = cur_q_len
            if not_last_three and STAGE != 1:
                cur_s2_lo = 0
            cur_s2_size = (cur_k_len + BLOCK_N - 1) // BLOCK_N

            cur_task_valid = cur_s1_idx * BLOCK_M < cur_q_len
            if cur_task_valid or not not_last_three:
                # =================== 最后preload次循环解决cool down问题 所以发射次数固定 ===================
                if not not_last_three:
                    cur_s2_lo = 0
                    cur_s2_size = 1
                
                # softmax max sum 使用, 3 buffer 的索引
                s1_task_mod3 = ((sq_loop_idx - pid) // CORE_NUM) % 3

                # =================== l0c pingpong 需要写在for循环内 配合编译选项生效 ===================
                qk_l0c = bl.alloc(tl.float32, [BLOCK_M, BLOCK_N], al.ascend_address_space.L0C, is_mem_unique=True)
                pv_l0c = bl.alloc(tl.float32, [BLOCK_M, HEAD_DIM], al.ascend_address_space.L0C, is_mem_unique=True)

                # =================== 常驻Q依赖 ===================
                q_l1_keep = bl.alloc(Q.dtype.element_ty, [BLOCK_M, HEAD_DIM], al.ascend_address_space.L1)
                # =================== KV sequence length loop ===================
                cur_s2_loop_size = cur_s2_size - cur_s2_lo
                for skv_loop_idx in range(0, cur_s2_loop_size):
                    actual_skv_idx = cur_s2_lo + skv_loop_idx
                    # create and push task to producer stack
                    if not_last_three:
                        (b_idx_1, n_idx_1, s1_idx_1, s2_idx_1, s2_size_1,
                        b_idx_2, n_idx_2, s1_idx_2, s2_idx_2, s2_size_2,
                        b_idx_3, n_idx_3, s1_idx_3, s2_idx_3, s2_size_3,
                        b_idx_4, n_idx_4, s1_idx_4, s2_idx_4, s2_size_4) = \
                                create_task(taskId, cur_b_idx, cur_n_idx, cur_s1_idx, actual_skv_idx, cur_s2_size,
                                            b_idx_1, n_idx_1, s1_idx_1, s2_idx_1, s2_size_1,
                                            b_idx_2, n_idx_2, s1_idx_2, s2_idx_2, s2_size_2,
                                            b_idx_3, n_idx_3, s1_idx_3, s2_idx_3, s2_size_3,
                                            b_idx_4, n_idx_4, s1_idx_4, s2_idx_4, s2_size_4)

                    # q_rs = get_s_offset(taskId & 3, H, N_CTX, BLOCK_M, cur_b_idx, cur_n_idx, cur_s1_idx) + tl.arange(0, BLOCK_M)[:, None]
                    # mask_h = (cur_s1_idx * BLOCK_M + tl.arange(0, BLOCK_M)) < N_CTX
                    # mask_w = tl.arange(0, HEAD_DIM) < HEAD_DIM
                    # mask_q = mask_h[:, None] & mask_w[None, :]
                    # q_cs = tl.arange(0, HEAD_DIM)[None, :]
                    # q_ptr = Q + q_rs * stride_qm + q_cs * stride_qk
                    q_rs = (q_len_start * H + cur_s1_idx * BLOCK_M * H + tl.arange(0, BLOCK_M) * H + cur_n_idx)[:, None]
                    q_cs = tl.arange(0, HEAD_DIM)[None, :]
                    mask_h = (cur_s1_idx * BLOCK_M + tl.arange(0, BLOCK_M)) < cur_q_len
                    mask_w = tl.arange(0, HEAD_DIM) < HEAD_DIM
                    mask_q = mask_h[:, None] & mask_w[None, :]
                    q_ptr = Q + q_rs * HEAD_DIM + q_cs

                    # k_rs = get_s_offset(taskId & 3, H, cur_k_len, BLOCK_M, cur_b_idx, cur_n_idx, skv_loop_idx) + tl.arange(0, BLOCK_N)[:, None]
                    # k_cs = tl.arange(0, HEAD_DIM)[None, :]
                    # mask_h = (skv_loop_idx * BLOCK_N + tl.arange(0, BLOCK_N)) < N_CTX
                    # mask_w = tl.arange(0, HEAD_DIM) < HEAD_DIM
                    # mask_k = mask_h[:, None] & mask_w[None, :]
                    # k_ptr = K + k_rs * stride_kn + k_cs * stride_kk
                    k_rs = (k_len_start * H + actual_skv_idx * BLOCK_N * H + tl.arange(0, BLOCK_N) * H + cur_n_idx)[:, None]
                    k_cs = tl.arange(0, HEAD_DIM)[None, :]
                    mask_h = (actual_skv_idx * BLOCK_N + tl.arange(0, BLOCK_N)) < cur_k_len
                    mask_w = tl.arange(0, HEAD_DIM) < HEAD_DIM
                    mask_k = mask_h[:, None] & mask_w[None, :]
                    k_ptr = K + k_rs * HEAD_DIM + k_cs
                    if (not is_last_loop and not is_last_second_loop) and not is_last_third_loop:
                        # Q 常驻 L1 适配逻辑
                        if skv_loop_idx == 0:
                            q = tl.load(q_ptr, mask=mask_q)
                            bl.to_buffer(tensor=q, bind_buffer=q_l1_keep)
                        else:
                            q = bl.to_tensor(q_l1_keep)
                        _qk_matmul(
                            q, k_ptr, qk_ub_ping, qk_ub_pong, qk_l0c, HEAD_DIM, BLOCK_N, taskId, mask_k
                        )

                    # get c2 task
                    c2_use_b_idx, c2_use_n_idx, c2_use_s1_idx, c2_use_s2_idx, c2_use_s2_size = get_cur_task((taskId+2) & 3,
                                                                            b_idx_1, n_idx_1, s1_idx_1, s2_idx_1, s2_size_1,
                                                                            b_idx_2, n_idx_2, s1_idx_2, s2_idx_2, s2_size_2,
                                                                            b_idx_3, n_idx_3, s1_idx_3, s2_idx_3, s2_size_3,
                                                                            b_idx_4, n_idx_4, s1_idx_4, s2_idx_4, s2_size_4)

                    c2_k_len_start = tl.load(cu_seqlens_k + c2_use_b_idx)
                    c2_cur_k_len = tl.load(cu_seqlens_k + c2_use_b_idx + 1) - c2_k_len_start
                    v_rs = (c2_k_len_start * H + c2_use_s2_idx * BLOCK_N * H + tl.arange(0, BLOCK_N) * H + c2_use_n_idx)[:, None]
                    v_cs = tl.arange(0, HEAD_DIM)[None, :]
                    v_ptr = V + v_rs * HEAD_DIM + v_cs

                    mask_h = (c2_use_s2_idx * BLOCK_N + tl.arange(0, BLOCK_N)) < c2_cur_k_len
                    mask_w = tl.arange(0, HEAD_DIM) < HEAD_DIM
                    mask_v = mask_h[:, None] & mask_w[None, :]
                    
                    if taskId > 1 and not_last:
                        _pv_matmul(
                            p_l1_ping,p_l1_pong, pv_ub_ping, pv_ub_pong,
                            pv_l0c, v_ptr, HEAD_DIM,BLOCK_M, BLOCK_N, taskId-2, mask_v
                        )

                    taskId += 1

        cube_postwait_s_ub()
        cube_postwait_pv_ub()

    with al.scope(core_mode="vector"):
        '''
        vector keeps own taskInfo[4]
        mark cur_task's batch_idx, head_num_idx, seq_q_idx, seq_kv_idx, seq_kv_loop_size
        v1 always use task_info[cur_idx-1] as task producer
        v2 always use task_info[cur_idx-3] as task consumer
        TODO: use class to Optimize when triton-ascend supports the feature :>
        v_s1_task_mod3_x is used as m_i&l_i control, update by s1_task_cnt
        '''
        v_b_idx_1, v_n_idx_1, v_s1_idx_1, v_s2_idx_1, v_s2_size_1 = 0, 0, 0, 0, 0 # task_1 V1
        v_b_idx_2, v_n_idx_2, v_s1_idx_2, v_s2_idx_2, v_s2_size_2 = 0, 0, 0, 0, 0 # task_2 C1
        v_b_idx_3, v_n_idx_3, v_s1_idx_3, v_s2_idx_3, v_s2_size_3 = 0, 0, 0, 0, 0
        v_b_idx_4, v_n_idx_4, v_s1_idx_4, v_s2_idx_4, v_s2_size_4 = 0, 0, 0, 0, 0
        v_cur_b_idx, v_cur_n_idx, v_cur_s1_idx, v_cur_s2_lo, v_cur_s2_size = 0, 0, 0, 0, 0

        v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4 = 0, 0, 0, 0
        s1_task_cnt = 0
        vtaskId = 0

        # =================== use 3 buffer to keep data in UB ===================
        # softmax max
        m_i_tb0 = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
        m_i_tb1 = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
        m_i_tb2 = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
        # softmax sum
        l_i_tb0 = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
        l_i_tb1 = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
        l_i_tb2 = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
        # exp(max - maxi)
        alpha_tb0 = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB, is_mem_unique=True)
        alpha_tb1 = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB, is_mem_unique=True)
        alpha_tb2 = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB, is_mem_unique=True)
        # flash update lse
        acc_buffer = bl.alloc(tl.float32, [BLOCK_M // 2, HEAD_DIM], al.ascend_address_space.UB, is_mem_unique=True)

        vec_prefree_s_ub()
        vec_prefree_pv_ub()

        # =================== Q sequence length loop ===================
        for sq_loop_idx in range(start_block, end_block, step):
            # =================== 循环控制和taskInfo[4]与Cube完全相同 等待支持后可以合并为一套 ===================
            v_is_last_loop = sq_loop_idx == last_loop
            v_is_last_second_loop = sq_loop_idx == last_second_loop
            v_is_last_third_loop = sq_loop_idx == last_third_loop
            v_not_last = not v_is_last_loop
            v_not_last_two = not v_is_last_loop and not v_is_last_second_loop
            v_not_last_three = (not v_is_last_loop and not v_is_last_second_loop) and not v_is_last_third_loop

            if v_not_last_three:
                (v_cur_b_idx, v_cur_n_idx, v_cur_s1_idx, v_cur_s2_lo, v_cur_s2_size) = \
                        create_and_get_basic_pos(sq_loop_idx, CORE_NUM, Z, H, pid, NUM_BLOCKS_M, STAGE, max_q_len, BLOCK_N, CAUSAL_TYPE)

            q_len_start = tl.load(cu_seqlens_q + v_cur_b_idx)
            q_len_end = tl.load(cu_seqlens_q + v_cur_b_idx + 1)
            cur_q_len = q_len_end - q_len_start
            # TODO: cross_attn q_len != k_len
            k_len_start = q_len_start
            k_len_end = q_len_end
            cur_k_len = cur_q_len
            if v_not_last_three and STAGE != 1:
                v_cur_s2_lo = 0
            v_cur_s2_size = (cur_k_len + BLOCK_N - 1) // BLOCK_N

            v_cur_task_valid = v_cur_s1_idx * BLOCK_M < cur_q_len
            if v_cur_task_valid or not v_not_last_three:
                if not v_not_last_three:
                    v_cur_s2_lo = 0
                    v_cur_s2_size = 1
                if v_cur_task_valid:
                    s1_task_cnt += 1

                # =================== KV sequence length loop ===================
                v_cur_s2_loop_size = v_cur_s2_size - v_cur_s2_lo
                for skv_loop_idx in range(0, v_cur_s2_loop_size):
                    v_actual_skv_idx = v_cur_s2_lo + skv_loop_idx
                    # create and push task to producer stack
                    if v_not_last_three:
                        (v_b_idx_1, v_n_idx_1, v_s1_idx_1, v_s2_idx_1, v_s2_size_1,
                        v_b_idx_2, v_n_idx_2, v_s1_idx_2, v_s2_idx_2, v_s2_size_2,
                        v_b_idx_3, v_n_idx_3, v_s1_idx_3, v_s2_idx_3, v_s2_size_3,
                        v_b_idx_4, v_n_idx_4, v_s1_idx_4, v_s2_idx_4, v_s2_size_4) = \
                                create_task(vtaskId, v_cur_b_idx, v_cur_n_idx, v_cur_s1_idx, v_actual_skv_idx, v_cur_s2_size,
                                            v_b_idx_1, v_n_idx_1, v_s1_idx_1, v_s2_idx_1, v_s2_size_1,
                                            v_b_idx_2, v_n_idx_2, v_s1_idx_2, v_s2_idx_2, v_s2_size_2,
                                            v_b_idx_3, v_n_idx_3, v_s1_idx_3, v_s2_idx_3, v_s2_size_3,
                                            v_b_idx_4, v_n_idx_4, v_s1_idx_4, v_s2_idx_4, v_s2_size_4)
                        v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4 = \
                                update_task(vtaskId, s1_task_cnt, v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4)

                    # get v1 task
                    v1_use_b_idx, v1_use_n_idx, v1_use_s1_idx, v1_use_s2_idx, v1_use_s2_size = get_cur_task((vtaskId-1) & 3,
                                                                        v_b_idx_1, v_n_idx_1, v_s1_idx_1, v_s2_idx_1, v_s2_size_1,
                                                                        v_b_idx_2, v_n_idx_2, v_s1_idx_2, v_s2_idx_2, v_s2_size_2,
                                                                        v_b_idx_3, v_n_idx_3, v_s1_idx_3, v_s2_idx_3, v_s2_size_3,
                                                                        v_b_idx_4, v_n_idx_4, v_s1_idx_4, v_s2_idx_4, v_s2_size_4)
                    need_do_v1 = vtaskId > 0 and v_not_last_two
                    if need_do_v1:
                        '''
                        1. flash softmax need update in colum block from second to last
                        2. qk's N_CTX is same, mask only need in last skv block
                        '''
                        v1_last_skv = v1_use_s2_idx == v1_use_s2_size - 1
                        if CAUSAL_TYPE == 0:
                            v1_need_update = v1_use_s2_idx != 0
                        else:
                            v1_need_update = v1_use_s2_idx != v1_use_s1_idx
                        v1_s1_task_mod3 = get_s_task(vtaskId-1, v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4)

                        sub_vec_id = al.sub_vec_id()
                        mask_b = v1_use_b_idx.to(tl.int64) * ORI_MAX_Q_LEN * ORI_MAX_K_LEN
                        mask_row = v1_use_s1_idx.to(tl.int64) * BLOCK_M + (sub_vec_id * BLOCK_M // 2).to(tl.int64)  + tl.arange(0, BLOCK_M // 2)[:, None]
                        mask_col = v1_use_s2_idx.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    
                        mask_ptr = ATTEN_MASK + mask_b + mask_row * stride_mask0 + mask_col * stride_mask1
                        m_q_len_start = tl.load(cu_seqlens_q + v1_use_b_idx)
                        m_q_len_end = tl.load(cu_seqlens_q + v1_use_b_idx + 1)
                        cur_m_q_len = m_q_len_end - m_q_len_start
                        mask_h = (v1_use_s1_idx * BLOCK_M + (sub_vec_id * BLOCK_M // 2) +  tl.arange(0, BLOCK_M // 2)) < cur_m_q_len
                        mask_w = (v1_use_s2_idx.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)) < cur_m_q_len
                        atten_mask = tl.load(mask_ptr, mask=(mask_h[:, None] & mask_w[None, :]), other=0)
                        # atten_mask = ~atten_mask
                        process_v1(qk_ub_ping, qk_ub_pong, p_l1_ping, p_l1_pong, atten_mask,
                            m_i_tb0, m_i_tb1, m_i_tb2, l_i_tb0, l_i_tb1, l_i_tb2, alpha_tb0, alpha_tb1, alpha_tb2,
                            sm_scale, vtaskId, v1_s1_task_mod3, Q.dtype.element_ty, 1, v1_need_update, BLOCK_M, BLOCK_N, 1)

                    # get v2 task
                    v2_use_b_idx, v2_use_n_idx, v2_use_s1_idx, v2_use_s2_idx, v2_use_s2_size = get_cur_task((vtaskId+1) & 3,
                                                                        v_b_idx_1, v_n_idx_1, v_s1_idx_1, v_s2_idx_1, v_s2_size_1,
                                                                        v_b_idx_2, v_n_idx_2, v_s1_idx_2, v_s2_idx_2, v_s2_size_2,
                                                                        v_b_idx_3, v_n_idx_3, v_s1_idx_3, v_s2_idx_3, v_s2_size_3,
                                                                        v_b_idx_4, v_n_idx_4, v_s1_idx_4, v_s2_idx_4, v_s2_size_4)
                    if vtaskId > 2:
                        if CAUSAL_TYPE == 0:
                            update_acc = v2_use_s2_idx != 0
                        else:
                            update_acc = v2_use_s2_idx != v2_use_s1_idx
                        v2_s1_task_mod3 = get_s_task(vtaskId-3, v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4)
                        _flash_update(pv_ub_ping, pv_ub_pong, alpha_tb0, alpha_tb1, alpha_tb2, acc_buffer, v2_s1_task_mod3, BLOCK_M, HEAD_DIM, vtaskId, update_acc)

                    v2_last_skv_v2 = v2_use_s2_idx == v2_use_s2_size - 1
                    # after sq row done, do flash softmax div sum
                    if v2_last_skv_v2:
                        v2_s1_task_mod3 = get_s_task(vtaskId-3, v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4)
                        if v2_s1_task_mod3 == 0:
                            l_i = bl.to_tensor(l_i_tb0)
                            m_i = bl.to_tensor(m_i_tb0)
                        elif v2_s1_task_mod3 == 1:
                            l_i = bl.to_tensor(l_i_tb1)
                            m_i = bl.to_tensor(m_i_tb1)
                        else:
                            l_i = bl.to_tensor(l_i_tb2)
                            m_i = bl.to_tensor(m_i_tb2)
                        m_i += tl.math.log(l_i)
                        acc = bl.to_tensor(acc_buffer)
                        acc = acc / l_i[:, None]
                        
                        sub_vec_id = al.sub_vec_id()
                        v2_q_len_start = tl.load(cu_seqlens_q + v2_use_b_idx)
                        v2_cur_q_len = tl.load(cu_seqlens_q + v2_use_b_idx + 1) - v2_q_len_start
                        out_offset = (v2_q_len_start * H + v2_use_s1_idx * BLOCK_M * H + sub_vec_id  * (BLOCK_M // 2) * H + v2_use_n_idx)
                        m_ptrs = M + out_offset  + tl.arange(0, BLOCK_M // 2) * H
                        mask_h = (v2_use_s1_idx * BLOCK_M + sub_vec_id * (BLOCK_M // 2) + tl.arange(0, BLOCK_M // 2)) < v2_cur_q_len
                        tl.store(m_ptrs, m_i, mask=mask_h)

                        o_rs = out_offset + tl.arange(0, BLOCK_M//2)[:, None] * H
                        o_cs = tl.arange(0, HEAD_DIM)[None, :]
                        o_ptrs = Out + o_rs * HEAD_DIM + o_cs
                        
                        mask_w = tl.arange(0, HEAD_DIM) < HEAD_DIM
                        mask_o = mask_h[:, None] & mask_w[None, :]

                        # tl.device_print("temp_acc:", sub_vec_id)
                        tl.store(o_ptrs, acc.to(Out.type.element_ty), mask=mask_o)

                    vtaskId += 1

        vec_postwait_p_l1()

def triton_attn_fwd(
    q, k, v, atten_mask, 
    cu_seqlens_q, cu_seqlens_k, max_q_len, max_k_len,
    causal, causal_type, sm_scale, BM, BN
):
    # qkv [T, N, D]
    # atten_mask [bsz, max_seq_q, max_seq_k]

    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    stage = 1 if causal else 0
    extra_kern_args = {}

    num_cores = get_npu_aicore_num()
    bsz = cu_seqlens_q.shape[0] - 1
    H = q.shape[1]
    NUM_BLOCKS_M = triton.cdiv(max_q_len, BM)
    NUM_BLOCKS = NUM_BLOCKS_M * bsz * H
    NUM_BLOCKS_PER_CORE = triton.cdiv(NUM_BLOCKS, num_cores)
    grid = min(num_cores, NUM_BLOCKS)
    o = torch.zeros_like(q)

    sparse_start_idx = [0]
    task = 0
    for i in range(num_cores):
        if i == 0:
            task += 3
            sparse_start_idx.append(task)
        else:
            task += 2
            sparse_start_idx.append(task)
    sparse_start_idx = torch.tensor(sparse_start_idx, device=q.device, dtype=torch.int32)
    M = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
    _attn_fwd[(grid,)](
        q, k, v, atten_mask, M, o, 
        cu_seqlens_q, cu_seqlens_k,
        max_q_len, max_k_len,
        sparse_start_idx, sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2), 
        atten_mask.stride(1), atten_mask.stride(2),
        bsz, H,
        HEAD_DIM=HEAD_DIM_K,  # 64
        BLOCK_M = BM, # 32
        BLOCK_N = BN, # 32
        STAGE=stage,
        CAUSAL_TYPE=causal_type,
        NUM_BLOCKS_PER_CORE=NUM_BLOCKS_PER_CORE,
        NUM_BLOCKS=NUM_BLOCKS,
        NUM_BLOCKS_M=NUM_BLOCKS_M,
        CORE_NUM=num_cores,
        debug=True,
        multibuffer=True,
        sync_solver=True,
        disable_auto_inject_block_sync=True,
        limit_auto_multi_buffer_of_local_buffer="no-limit",
        **extra_kern_args)

    return o, M


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
	if TYPE == 1:
		triu_causal = (q_offset[:, None] <= k_offset[None, :]).to(tl.int32)
		attn_args_mask = ((q_attn_arg[:, None] == k_attn_arg[None, :]).to(tl.int32) |
				(k_attn_arg[None, :] == 0).to(tl.int32)).to(tl.int32)
		return (
				(triu_causal &
				attn_args_mask) |
				(q_offset[:, None] == k_offset[None, :]).to(tl.int32))
	if TYPE == 2:
		tril_causal = (q_offset[:, None] >= k_offset[None, :])
		return ((tril_causal & ((q_attn_arg[:, None] == k_attn_arg[None, :]) | (k_attn_arg[None, :] == 0))) | (
					q_offset[:, None] == k_offset[None, :]))


@triton.jit
def gen_fa_mask_kernel(
		output_ptr,
		q_attn_arg_ptr, k_attn_arg_ptr,
		cu_seqlens_q, cu_seqlens_k,
		MASK_FN: tl.constexpr,
		BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
		AICORE_NUM: tl.constexpr,
		MAX_Q_LEN: tl.constexpr,
		MAX_K_LEN: tl.constexpr,
		BATCH_SIZE: tl.constexpr,
):
	ORI_MAX_Q_LEN = MAX_Q_LEN
	ORI_MAX_K_LEN = MAX_K_LEN
	pid = tl.program_id(0)
	task_nums = tl.cdiv(MAX_Q_LEN, BLOCK_M)
	task_nums = task_nums % 7
	MAX_Q_LEN = tl.where(task_nums == 0, MAX_Q_LEN + BLOCK_M, MAX_Q_LEN)
	NUM_BLOCKS_M = tl.cdiv(MAX_Q_LEN, BLOCK_M)  # 非对齐跳过
	NUM_BLOCKS = NUM_BLOCKS_M * BATCH_SIZE 
	zero_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int8)
	start_block, end_block, step = pid, NUM_BLOCKS, AICORE_NUM
	for block_idx in range(start_block, end_block, step):
		task_hz_idx = block_idx // NUM_BLOCKS_M
		start_m = block_idx % NUM_BLOCKS_M
		start_b = task_hz_idx

		q_start1 = tl.load(cu_seqlens_q + start_b)
		q_end = tl.load(cu_seqlens_q + start_b + 1)
		q_len = q_end - q_start1
		if start_m * BLOCK_M < q_len:
			k_start1 = tl.load(cu_seqlens_k + start_b)
			k_end = tl.load(cu_seqlens_k + start_b + 1)
			k_len = k_end - k_start1

			begin = start_m * BLOCK_M
			if begin.to(tl.int64) < k_len.to(tl.int64):
				end = k_len

				offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

				q_start = q_start1.to(tl.int64)
				k_start = k_start1.to(tl.int64)
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
				mask_out_ptr = tl.make_block_ptr(
					base=output_ptr + start_b * ORI_MAX_Q_LEN * ORI_MAX_K_LEN,
					shape=(q_len, k_len),
					strides=(ORI_MAX_K_LEN, 1),
					offsets=(start_m * BLOCK_M, begin),
					block_shape=(BLOCK_M, BLOCK_N),
					order=(1, 0)
				)
				if begin > 0:
					zero_mask_out_ptr = tl.make_block_ptr(
						base=output_ptr + start_b * ORI_MAX_Q_LEN * ORI_MAX_K_LEN,
						shape=(q_len, begin),
						strides=(ORI_MAX_K_LEN, 1),
						offsets=(start_m * BLOCK_M, 0),
						block_shape=(BLOCK_M, BLOCK_N),
						order=(1, 0)
					)
					for start_n in range(0, begin, BLOCK_N):
						store_if(zero_mask_out_ptr, zero_block, False, False)
						zero_mask_out_ptr = tl.advance(zero_mask_out_ptr, (0, BLOCK_N))
				q_attn_arg = load_if(q_attn_arg_block_ptr, False, True)

				for start_n in range(begin, end, BLOCK_N):
					start_n = tl.multiple_of(start_n, BLOCK_N)
					k_attn_arg = load_if(k_attn_arg_block_ptr, False, True)
					offset_n = start_n + tl.arange(0, BLOCK_N)
					mask = mask_fn(q_attn_arg, k_attn_arg, offset_m, offset_n, MASK_FN).to(tl.int8)

					store_if(mask_out_ptr, mask, False, False)
					mask_out_ptr = tl.advance(mask_out_ptr, (0, BLOCK_N))


def print_mismatched_positions(tensor1, tensor2, atol=1e-3, rtol=1e-2, max_print=20):
    """
    打印两个tensor不相等的位置及其值
    :param tensor1: 第一个tensor
    :param tensor2: 第二个tensor
    :param atol: 绝对误差容差
    :param rtol: 相对误差容差
    :param max_print: 最大打印不匹配位置数量
    """
    # 计算绝对误差
    abs_diff = torch.abs(tensor1 - tensor2)
    
    # 计算相对误差（避免除以0）
    denom = torch.abs(tensor2) + atol
    rel_diff = abs_diff / denom
    
    # 找到不匹配的位置
    mismatch_mask = (abs_diff > atol) | (rel_diff > rtol)
    
    # 获取不匹配位置的索引
    mismatch_indices = torch.where(mismatch_mask)
    
    num_mismatches = len(mismatch_indices[0])
    print(f"\n=== 不匹配位置统计 ===")
    print(f"总元素数: {tensor1.numel()}")
    print(f"不匹配元素数: {num_mismatches}")
    print(f"不匹配比例: {num_mismatches / tensor1.numel() * 100:.2f}%")
    
    if num_mismatches > 0:
        print(f"\n=== 前{min(max_print, num_mismatches)}个不匹配位置 ===")
        for i in range(min(max_print, num_mismatches)):
            idx = tuple(idx[i].item() for idx in mismatch_indices)
            val1 = tensor1[idx].item()
            val2 = tensor2[idx].item()
            diff = abs(val1 - val2)
            print(f"位置 {idx}: triton={val1:.6f}, native={val2:.6f}, 绝对误差={diff:.6f}")
    
    return num_mismatches


def torch_native_attention(q, k, v, scale, mask_fn, q_attn_arg=None, k_attn_arg=None, cu_seqlens_q=None, cu_seqlens_k=None):
    """
    PyTorch原生实现的Attention，与Triton kernel完全一致
    支持多batch且每个batch序列长度不等
    :param q: [total_q_seq, q_head, qk_dim]
    :param k: [total_kv_seq, kv_head, qk_dim]
    :param v: [total_kv_seq, kv_head, v_dim]
    :param scale: 缩放因子
    :param mask_fn: 掩码函数类型
    :param q_attn_arg: query的attn_arg
    :param k_attn_arg: key的attn_arg
    :param cu_seqlens_q: query的序列长度累积和 [batch+1]
    :param cu_seqlens_k: key的序列长度累积和 [batch+1]
    :return: output [total_q_seq, q_head, v_dim]
    """
    q_len, q_head, qk_dim = q.shape
    k_len, kv_head, v_dim = v.shape
    
    # 如果q_head != kv_head，需要处理grouped query attention
    if q_head != kv_head:
        head_group = q_head // kv_head
        k = k.unsqueeze(1).repeat(1, head_group, 1, 1).reshape(k_len, q_head, qk_dim)
        v = v.unsqueeze(1).repeat(1, head_group, 1, 1).reshape(k_len, q_head, v_dim)
    
    # 创建索引偏移
    q_offset = torch.arange(q_len, device=q.device)
    k_offset = torch.arange(k_len, device=k.device)
    
    # 计算注意力分数
    k_transposed = k.transpose(0, 1).transpose(1, 2)  # [q_head, qk_dim, k_len]
    q_reshaped = q.transpose(0, 1)  # [q_head, q_len, qk_dim]
    scores = torch.matmul(q_reshaped, k_transposed) * scale  # [q_head, q_len, k_len]
    
    # 应用batch内的序列边界掩码（防止跨batch attention）
    if cu_seqlens_q is not None and cu_seqlens_k is not None:
        batch_size = len(cu_seqlens_q) - 1
        batch_mask = torch.zeros(q_len, k_len, device=q.device, dtype=torch.bool)
        
        for b in range(batch_size):
            q_start, q_end = cu_seqlens_q[b].item(), cu_seqlens_q[b+1].item()
            k_start, k_end = cu_seqlens_k[b].item(), cu_seqlens_k[b+1].item()
            batch_mask[q_start:q_end, k_start:k_end] = True
        
        # 初始时将所有位置设为无效，然后根据batch_mask设置有效位置
        scores = scores.masked_fill(~batch_mask.unsqueeze(0), float('-inf'))
    
    # 应用掩码（完全按照Triton kernel的mask_fn实现）
    if mask_fn == 1 or mask_fn == 2:
        # 根据Triton kernel中的mask_fn实现
        tril_causal = q_offset[:, None] >= k_offset[None, :]  # q >= k
        triu_causal = q_offset[:, None] <= k_offset[None, :]  # q <= k
        
        if mask_fn == 1:
            # TYPE 1: (triu_causal & ((q_attn_arg == k_attn_arg) | (k_attn_arg == 0))) | diagonal
            causal_mask = triu_causal
        else:
            # TYPE 2: (tril_causal & ((q_attn_arg == k_attn_arg) | (k_attn_arg == 0))) | diagonal
            causal_mask = tril_causal
        
        # 应用attn_arg条件
        if q_attn_arg is not None and k_attn_arg is not None:
            arg_mask = (q_attn_arg[:, None] == k_attn_arg[None, :]) | (k_attn_arg[None, :] == 0)
            mask = causal_mask & arg_mask
        else:
            mask = causal_mask
        
        # 对角线总是保留
        diag_mask = q_offset[:, None] == k_offset[None, :]
        final_mask = mask | diag_mask
        
        # 对分数应用掩码
        scores = scores.masked_fill(~final_mask.unsqueeze(0), float('-inf'))
    
    # 计算softmax
    attn_weights = torch.softmax(scores, dim=-1)  # [q_head, q_len, k_len]
    
    # 乘以v
    v_reshaped = v.transpose(0, 1)  # [q_head, k_len, v_dim]
    output = torch.matmul(attn_weights, v_reshaped)  # [q_head, q_len, v_dim]
    
    # 转置回 [q_len, q_head, v_dim]
    output = output.transpose(0, 1)
    
    return output

def compare(triton_output, native_output, q):
    # 计算误差
    max_abs_error = torch.max(torch.abs(triton_output - native_output)).item()
    mean_abs_error = torch.mean(torch.abs(triton_output - native_output)).item()
    max_rel_error = torch.max(torch.abs(triton_output - native_output) / (torch.abs(native_output) + 1e-6)).item()
    
    print(f"Max Absolute Error: {max_abs_error:.6e}")
    print(f"Mean Absolute Error: {mean_abs_error:.6e}")
    print(f"Max Relative Error: {max_rel_error:.6e}")
    
    # 检查是否通过
    atol = 1e-2
    rtol = 1e-2
    passed = torch.allclose(triton_output, native_output, atol=atol, rtol=rtol)
    print(f"Test {q.shape=}, {'PASSED' if passed else 'FAILED'} (atol={atol}, rtol={rtol})")
    if not passed:
        # 打印不匹配位置
        # print_mismatched_positions(triton_output, native_output, atol=atol, rtol=rtol)
        
        passed = torch.allclose(triton_output[:64,...], native_output[:64,...], atol=atol, rtol=rtol)
        if not passed:
            # 额外打印[:64]和[64:]的统计
            print(f"\n=== [:64] 区域 ===")
            print_mismatched_positions(triton_output[:64,...], native_output[:64,...], atol=atol, rtol=rtol)
        passed = torch.allclose(triton_output[64:,...], native_output[64:,...], atol=atol, rtol=rtol)
        if not passed:
            print(f"\n=== [64:] 区域 ===")
            print_mismatched_positions(triton_output[64:,...], native_output[64:,...], atol=atol, rtol=rtol)


def test_acc_customer_mask_fn():
    # 生成随机输入
    torch.manual_seed(42)
    device = DEVICE
    test_cases = [
        ([384, 640, 192, 448], [384, 640, 192, 448], 8, 8, 64, 64, 1),
    ]
    
    for i, (q_seq_lens, k_seq_lens, q_head, kv_head, qk_dim, v_dim, mask_fn) in enumerate(test_cases):
        batch_size = len(q_seq_lens)
        total_q_seq = sum(q_seq_lens)
        total_k_seq = sum(k_seq_lens)
        max_q_seq = max(q_seq_lens)
        max_k_seq = max(k_seq_lens)
        
        q = torch.randn(total_q_seq, q_head, qk_dim, device=device, dtype=torch.float16)
        k = torch.randn(total_k_seq, kv_head, qk_dim, device=device, dtype=torch.float16)
        v = torch.randn(total_k_seq, kv_head, v_dim, device=device, dtype=torch.float16)
        
        # 生成cu_seqlens（累积和）
        cu_seqlens_q = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
        cu_seqlens_k = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
        for b in range(batch_size):
            cu_seqlens_q[b+1] = cu_seqlens_q[b] + q_seq_lens[b]
            cu_seqlens_k[b+1] = cu_seqlens_k[b] + k_seq_lens[b]
        
        # 生成attn_arg（用于稀疏注意力）
        q_attn_arg = torch.zeros(total_q_seq, device=device, dtype=torch.int32)
        k_attn_arg = torch.zeros(total_k_seq, device=device, dtype=torch.int32)
        
        # 计算scale
        scale = 1.0 / (qk_dim ** 0.5)

        q_attn_arg, k_attn_arg = q_attn_arg.to(torch.int32), k_attn_arg.to(torch.int32),
        cu_seqlens_q, cu_seqlens_k = cu_seqlens_q.to(torch.int32), cu_seqlens_k.to(torch.int32),
        mask_tensor = torch.empty((batch_size, max_q_seq, max_k_seq), dtype=torch.bool, device=device) 
        BM, BN = 128, 128
        gen_fa_mask_kernel[(56,)](
            mask_tensor,
            q_attn_arg, k_attn_arg,
            cu_seqlens_q, cu_seqlens_k,
            mask_fn,
            BM,
            BN,
            AICORE_NUM = 56,
            MAX_Q_LEN = max_q_seq,
            MAX_K_LEN = max_k_seq,
            BATCH_SIZE = batch_size,
            multibuffer=True,
            num_stages=2,
        )

        # Triton实现
        causal = True
        causal_type = 1
        triton_output, L = triton_attn_fwd(
            q, k, v, mask_tensor,
            cu_seqlens_q, cu_seqlens_k, max_q_seq, max_k_seq,
            causal, causal_type, scale, BM, BN
        )
        
        # PyTorch原生实现（支持不等长序列）
        native_output = torch_native_attention(
            q, k, v, scale, mask_fn, q_attn_arg, k_attn_arg, cu_seqlens_q, cu_seqlens_k
        )
        
        # compare(triton_output, native_output, q)
        rtol = 0.0
        atol = 1e-2

        diff_golden_fa = (triton_output - native_output).abs()
        print(f"diff_golden_fa (Max Diff): {diff_golden_fa.max().item()}")
        assert torch.allclose(triton_output, native_output, atol=atol, rtol=rtol)
        print("compare success!")
        

def test_acc(Z, H, N_CTX, HEAD_DIM, causal, causal_type, dtype, BM ,BN):
    assert causal_type == 0, "torch_npu only support causal_type 0"
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()).npu()
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()).npu()
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()).npu()
    
    sm_scale = 0.5

    M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE)).npu()
    atten_mask = None
    atten_mask_golden = None
    sparse_mode= 0
    if causal:
        atten_mask = torch.triu(torch.ones(Z, N_CTX, N_CTX, device=DEVICE), diagonal=1).bool().npu()
        compressed_len = 2048
        atten_mask_golden = torch.triu(torch.ones(compressed_len, compressed_len, device=DEVICE), diagonal=1).bool().npu()
        sparse_mode = 2

    ref_out, ref_softmaxmax,ref_softmaxsum = torch_npu.npu_fusion_attention(
        q, k, v, H,
        padding_mask=None,
        atten_mask=atten_mask_golden,
        scale=sm_scale,
        keep_prob=1.0,
        input_layout='BNSD',
        pre_tockens=65535,
        next_tockens=65535,
        sparse_mode=sparse_mode,
    )[0:3]

    # BNSD -> TND
    q = q.permute(0, 2, 1, 3).contiguous().view(-1, H, HEAD_DIM)
    k = k.permute(0, 2, 1, 3).contiguous().view(-1, H, HEAD_DIM)
    v = v.permute(0, 2, 1, 3).contiguous().view(-1, H, HEAD_DIM)
    cu_seqlens_q_list = [0] + [N_CTX] * Z
    cu_seqlens_k_list = [0] + [N_CTX] * Z
    max_q_len = max(cu_seqlens_q_list)
    max_k_len = max(cu_seqlens_k_list)
    cu_seqlens_q = np.cumsum(cu_seqlens_q_list).tolist()
    cu_seqlens_k = np.cumsum(cu_seqlens_k_list).tolist()
    cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=DEVICE)
    cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=DEVICE)
    atten_mask = ~atten_mask
    tri_out, L = triton_attn_fwd(
        q, k, v, atten_mask,
        cu_seqlens_q, cu_seqlens_k, max_q_len, max_k_len,
        causal, causal_type, sm_scale, BM, BN
    )
    tri_out = tri_out.to(q.dtype)

    # TND -> BNSD
    tri_out = tri_out.view(Z, N_CTX, H, HEAD_DIM)
    tri_out = tri_out.permute(0, 2, 1, 3).contiguous()
    L = L.view(Z, N_CTX, H)
    L = L.permute(0, 2, 1).contiguous()

    rtol = 0.0
    atol = 1e-2

    diff_golden_fa = (ref_out - tri_out).abs()
    print(f"diff_golden_fa (Max Diff): {diff_golden_fa.max().item()}")
    assert torch.allclose(ref_out, tri_out, atol=atol, rtol=rtol)
    ref_M = ref_softmaxmax + torch.log(ref_softmaxsum)
    assert torch.allclose(ref_M.mean(axis=-1), L, atol=atol, rtol=rtol)
    print("compare success!")


def test_func(q_cumsum, causal_type=0):
    # input params
    DEVICE = torch.device("npu")

    num_head = 8
    head_dim = 64
    q_seq_list = q_cumsum[1:] - q_cumsum[:-1]
    k_seq_list = q_seq_list

    q_len = sum(q_seq_list)
    k_len = sum(k_seq_list)
    max_seqlen_q = int(np.max(q_seq_list))
    max_seqlen_k = int(np.max(k_seq_list))

    # qkv
    q = torch.randn((q_len, num_head, head_dim), dtype=torch.float16, device=DEVICE)
    k = torch.randn((q_len, num_head, head_dim), dtype=torch.float16, device=DEVICE)
    v = torch.randn((q_len, num_head, head_dim), dtype=torch.float16, device=DEVICE)
    

    q_attn_arg = torch.zeros(q_len, dtype=torch.int32, device="cpu")
    q_attn_arg[0] = 1
    k_attn_arg = torch.zeros(k_len, dtype=torch.int32, device="cpu")
    k_attn_arg[0] = 1
    cu_seqlens_q = q_cumsum.tolist()
    cu_seqlens_k = q_cumsum.tolist()
    cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cpu")
    cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device="cpu")

    q_attn_arg = q_attn_arg.npu()
    k_attn_arg = k_attn_arg.npu()
    cu_seqlens_q = cu_seqlens_q.npu()
    cu_seqlens_k = cu_seqlens_k.npu()    


    scale = 1.0 / (head_dim ** 0.5)
    batch_size = cu_seqlens_q.shape[0] - 1
    if causal_type == 0:
        atten_mask = torch.triu(torch.ones(batch_size, max_seqlen_q, max_seqlen_k, device=DEVICE), diagonal=1).bool().npu()
    else:
        atten_mask = torch.tril(torch.ones(batch_size, max_seqlen_q, max_seqlen_k, device=DEVICE), diagonal=-1).bool().npu()

    # print(q.shape, k.shape, v.shape, atten_mask.shape, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    atten_mask = ~atten_mask
    result = triton_attn_fwd(
        q, k, v, atten_mask,
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        causal=True, causal_type=causal_type, sm_scale=scale, BM=128, BN=128
    )

def test_func_case_final(causal_type=0):
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
        285586, 287605, 288770, 289261, 291526]
    )
    test_func(q_cumsum, causal_type)


if __name__ == "__main__":
    # test_acc(8, 8, 2432, 64, True, 0, torch.bfloat16, 128, 128)
    # torch.npu.synchronize() 
    # print("=============== test acc (causal_type=0) done ============\n")

    # test_func_case_final(causal_type=0)
    # torch.npu.synchronize() 
    # print("=============== test case_final (causal_type=0) done ============\n")

    test_func_case_final(causal_type=1)
    torch.npu.synchronize() 
    print("=============== test case_final (causal_type=1) done ============\n")

    test_acc_customer_mask_fn()
    torch.npu.synchronize()
    print("=============== test acc (custom_mask_fn) done ============\n")

    # profiling
    # experimental_config = torch_npu.profiler._ExperimentalConfig(
    #     aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    #     profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
    # )
    # with torch_npu.profiler.profile(
    #         activities=[  # torch_npu.profiler.ProfilerActivity.CPU,
    #             torch_npu.profiler.ProfilerActivity.NPU],
    #         with_stack=False,  # 采集torch 算子的函数调用栈的开关，该参数选填，默认关闭
    #         record_shapes=False,  # 采集torch 算子的input shape和input type的开关，该参数选填，默认关闭
    #         profile_memory=False,  # 采集memory相关数据的开关，该参数选填，默认关闭
    #         schedule=torch_npu.profiler.schedule(wait=1,
    #                                              warmup=1,
    #                                              active=30,
    #                                              repeat=1,
    #                                              skip_first=1),
    #         experimental_config=experimental_config,  # 该参数选填，默认为Level0
    #         on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./prof_dir")
    # ) as prof:
    #     for i in range(10):
    #         print(f"xxxxxxxxxx step: {i} begin xxxxxxxx")
    #         torch.npu.synchronize()
    #         test_func(q_cumsum=None)
    #         torch.npu.synchronize()  # 确保 kernel 真正执行完
    #         print(f"xxxxxxxxxx step: {i} end xxxxxxxx")
    #         prof.step()
