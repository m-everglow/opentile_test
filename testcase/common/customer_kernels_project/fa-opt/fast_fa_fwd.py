import torch
import torch_npu
import triton
import triton.language as tl

import triton.language.extra.cann.extension as al
import triton.extension.buffer.language as bl
import triton.runtime.driver as driver

from fast_fa_fwd_vf import *

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
def update_pos(s1_cur_idx, s1_step, batch_size, head_num, NUM_BLOCKS_M, STAGE, N_CTX, BLOCK_N):
    bn_idx = s1_cur_idx // NUM_BLOCKS_M         # total batch_offset * head_num_offset
    s1_idx = s1_cur_idx % NUM_BLOCKS_M          # remain s_offset
    b_idx = bn_idx // head_num                  # batch_offset
    n_idx = bn_idx % head_num                   # head_num_offset
    if STAGE == 1:
        lo, hi = 0, s1_idx + 1
    else:
        lo, hi = 0, (N_CTX + BLOCK_N - 1) // BLOCK_N

    return b_idx, n_idx, s1_idx, hi

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
def create_and_get_basic_pos(s1_cur_idx, s1_step, batch, head_num, s1_start, NUM_BLOCKS_M, STAGE, N_CTX, BLOCK_N):
    b_idx, n_idx, s1_idx, s2_size = \
        update_pos(s1_cur_idx, s1_step, batch, head_num, NUM_BLOCKS_M, STAGE, N_CTX, BLOCK_N)

    return (b_idx, n_idx, s1_idx, s2_size)

@triton.jit
def _attn_fwd(Q, K, V, ATTEN_MASK, M, Out, sparse_start_idx, sm_scale: tl.constexpr,  #
              stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qk: tl.constexpr,  #
              stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kk: tl.constexpr,  #
              stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vk: tl.constexpr,  #
              stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr,  #
              stride_mask0: tl.constexpr, stride_mask1: tl.constexpr,
              stride_am: tl.constexpr,
              Z: tl.constexpr, H: tl.constexpr, 
              N_CTX: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              NUM_BLOCKS_PER_CORE: tl.constexpr,
              NUM_BLOCKS: tl.constexpr,
              NUM_BLOCKS_M: tl.constexpr,
              CORE_NUM: tl.constexpr,
              ):
    pid = tl.program_id(0)
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
    else:
        start_block = tl.load(sparse_start_idx+pid)  # (0)
        end_block = tl.load(sparse_start_idx+pid+1)  # (3)
        step = 1
        multi_core_limit += 3  # (60)
        last_third_loop = end_block  # 3 
        last_second_loop = end_block + 1 # 4
        last_loop = end_block + 2  # 5
        end_block = end_block + preload # 6

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
        cur_b_idx, cur_n_idx, cur_s1_idx, cur_s2_size = 0, 0, 0, 0
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
                (cur_b_idx, cur_n_idx, cur_s1_idx, cur_s2_size) = \
                    create_and_get_basic_pos(
                        sq_loop_idx, CORE_NUM, Z, H, pid, NUM_BLOCKS_M, STAGE, N_CTX, BLOCK_N
                    )
            # =================== 最后preload次循环解决cool down问题 所以发射次数固定 ===================
            if not not_last_three:
                cur_s2_size = 1
            
            # softmax max sum 使用, 3 buffer 的索引
            s1_task_mod3 = ((sq_loop_idx - pid) // CORE_NUM) % 3

            # =================== l0c pingpong 需要写在for循环内 配合编译选项生效 ===================
            qk_l0c = bl.alloc(tl.float32, [BLOCK_M, BLOCK_N], al.ascend_address_space.L0C, is_mem_unique=True)
            pv_l0c = bl.alloc(tl.float32, [BLOCK_M, HEAD_DIM], al.ascend_address_space.L0C, is_mem_unique=True)

            # =================== 常驻Q依赖 ===================
            q_l1_keep = bl.alloc(Q.dtype.element_ty, [BLOCK_M, HEAD_DIM], al.ascend_address_space.L1)
            # =================== KV sequence length loop ===================
            for skv_loop_idx in range(0, cur_s2_size):
                # create and push task to producer stack
                if not_last_three:
                    (b_idx_1, n_idx_1, s1_idx_1, s2_idx_1, s2_size_1,
                    b_idx_2, n_idx_2, s1_idx_2, s2_idx_2, s2_size_2,
                    b_idx_3, n_idx_3, s1_idx_3, s2_idx_3, s2_size_3,
                    b_idx_4, n_idx_4, s1_idx_4, s2_idx_4, s2_size_4) = \
                            create_task(taskId, cur_b_idx, cur_n_idx, cur_s1_idx, skv_loop_idx, cur_s2_size,
                                        b_idx_1, n_idx_1, s1_idx_1, s2_idx_1, s2_size_1,
                                        b_idx_2, n_idx_2, s1_idx_2, s2_idx_2, s2_size_2,
                                        b_idx_3, n_idx_3, s1_idx_3, s2_idx_3, s2_size_3,
                                        b_idx_4, n_idx_4, s1_idx_4, s2_idx_4, s2_size_4)

                q_rs = get_s_offset(taskId & 3, H, N_CTX, BLOCK_M, cur_b_idx, cur_n_idx, cur_s1_idx) + tl.arange(0, BLOCK_M)[:, None]
                mask_h = (cur_s1_idx * BLOCK_M + tl.arange(0, BLOCK_M)) < N_CTX
                mask_w = tl.arange(0, HEAD_DIM) < HEAD_DIM
                mask_q = mask_h[:, None] & mask_w[None, :]
                q_cs = tl.arange(0, HEAD_DIM)[None, :]
                q_ptr = Q + q_rs * stride_qm + q_cs * stride_qk
                k_rs = get_s_offset(taskId & 3, H, N_CTX, BLOCK_M, cur_b_idx, cur_n_idx, skv_loop_idx) + tl.arange(0, BLOCK_N)[:, None]
                k_cs = tl.arange(0, HEAD_DIM)[None, :]
                
                mask_h = (skv_loop_idx * BLOCK_N + tl.arange(0, BLOCK_N)) < N_CTX
                mask_w = tl.arange(0, HEAD_DIM) < HEAD_DIM
                mask_k = mask_h[:, None] & mask_w[None, :]
                k_ptr = K + k_rs * stride_kn + k_cs * stride_kk
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

                v_rs = get_s_offset((taskId+2) & 3, H, N_CTX, BLOCK_M, c2_use_b_idx, c2_use_n_idx, c2_use_s2_idx) + tl.arange(0, BLOCK_N)[:, None]
                v_cs = tl.arange(0, HEAD_DIM)[None, :]
                v_ptr = V + v_rs * stride_vn + v_cs * stride_vk
                
                mask_h = (c2_use_s2_idx * BLOCK_N + tl.arange(0, BLOCK_N)) < N_CTX
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
        v_cur_b_idx, v_cur_n_idx, v_cur_s1_idx, v_cur_s2_size = 0, 0, 0, 0

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
                (v_cur_b_idx, v_cur_n_idx, v_cur_s1_idx, v_cur_s2_size) = \
                        create_and_get_basic_pos(sq_loop_idx, CORE_NUM, Z, H, pid, NUM_BLOCKS_M, STAGE, N_CTX, BLOCK_N)
            if not v_not_last_three:
                v_cur_s2_size = 1
            s1_task_cnt += 1

            # mask ptr
            if STAGE == 1:
                attn_mask_ptr = tl.make_block_ptr(
                    base = ATTEN_MASK,
                    shape=(N_CTX, N_CTX),
                    strides=(stride_am, 1),
                    offsets=(0, 0),
                    block_shape=(BLOCK_M // 2, BLOCK_N),
                    order=(1, 0)
                )
            else:
                attn_mask_ptr = None

            # =================== KV sequence length loop ===================
            for skv_loop_idx in range(0, v_cur_s2_size):
                # create and push task to producer stack
                if v_not_last_three:
                    (v_b_idx_1, v_n_idx_1, v_s1_idx_1, v_s2_idx_1, v_s2_size_1,
                    v_b_idx_2, v_n_idx_2, v_s1_idx_2, v_s2_idx_2, v_s2_size_2,
                    v_b_idx_3, v_n_idx_3, v_s1_idx_3, v_s2_idx_3, v_s2_size_3,
                    v_b_idx_4, v_n_idx_4, v_s1_idx_4, v_s2_idx_4, v_s2_size_4) = \
                            create_task(vtaskId, v_cur_b_idx, v_cur_n_idx, v_cur_s1_idx, skv_loop_idx, v_cur_s2_size,
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
                    v1_need_update = v1_use_s2_idx != 0
                    v1_s1_task_mod3 = get_s_task(vtaskId-1, v_s1_task_mod3_1, v_s1_task_mod3_2, v_s1_task_mod3_3, v_s1_task_mod3_4)

                    sub_vec_id = al.sub_vec_id()
                    mask_row = v1_use_s1_idx.to(tl.int64) * BLOCK_M + (sub_vec_id * BLOCK_M // 2).to(tl.int64)  + tl.arange(0, BLOCK_M // 2)[:, None]
                    mask_col = v1_use_s2_idx.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
 
                    mask_ptr = ATTEN_MASK + mask_row * stride_mask0 + mask_col * stride_mask1

                    mask_h = (v1_use_s1_idx * BLOCK_M + (sub_vec_id * BLOCK_M // 2) +  tl.arange(0, BLOCK_M // 2)) < N_CTX
                    mask_w = (v1_use_s2_idx.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)) < N_CTX
                    atten_mask = tl.load(mask_ptr, mask=(mask_h[:, None] & mask_w[None, :]), other=1)
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
                    update_acc = v2_use_s2_idx != 0
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
                    out_offset = get_s_offset((vtaskId+1) & 3, H, N_CTX, BLOCK_M, v2_use_b_idx, v2_use_n_idx, v2_use_s1_idx) + sub_vec_id * (BLOCK_M // 2)

                    m_ptrs = M + out_offset  + tl.arange(0, BLOCK_M // 2)
                    
                    mask_h = (v2_use_s1_idx * BLOCK_M + sub_vec_id * (BLOCK_M // 2) + tl.arange(0, BLOCK_M // 2)) < N_CTX
                    tl.store(m_ptrs, m_i, mask=mask_h)

                    o_rs = out_offset + tl.arange(0, BLOCK_M//2)[:, None]   # [BM, 1]
                    o_cs = tl.arange(0, HEAD_DIM)[None, :]
                    o_ptrs = Out + o_rs * stride_om + o_cs * stride_on
                    
                    mask_w = tl.arange(0, HEAD_DIM) < HEAD_DIM
                    mask_o = mask_h[:, None] & mask_w[None, :]

                    # tl.device_print("temp_acc:", sub_vec_id)
                    tl.store(o_ptrs, acc.to(Out.type.element_ty), mask=mask_o)

                vtaskId += 1

        vec_postwait_p_l1()

def triton_attn_fwd(q, k, v, atten_mask,causal, sm_scale, BM, BN):
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    stage = 1 if causal else 0
    extra_kern_args = {}

    num_cores = get_npu_aicore_num()
    NUM_BLOCKS_M = triton.cdiv(q.shape[2], BM)
    NUM_BLOCKS = NUM_BLOCKS_M * q.shape[0] * q.shape[1]
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
    print(f"{sparse_start_idx=}")
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    _attn_fwd[(grid,)](
        q, k, v, atten_mask, M, o, sparse_start_idx, sm_scale, #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
        atten_mask.stride(0), atten_mask.stride(1),
        q.shape[2],#
        q.shape[0], q.shape[1], N_CTX=q.shape[2],
        HEAD_DIM=HEAD_DIM_K,  # 64
        BLOCK_M = BM, # 32
        BLOCK_N = BN, # 32
        STAGE=stage,
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

def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype, BM ,BN):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()).npu()
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()).npu()
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()).npu()
    o = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()).npu()

    sm_scale = 0.5

    M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE)).npu()
    atten_mask = None
    atten_mask_golden = None
    sparse_mode= 0
    if causal:
        atten_mask = torch.triu(torch.ones(N_CTX, N_CTX, device=DEVICE), diagonal=1).bool().npu()
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
    print(ref_out.max().item())
    tri_out, L = triton_attn_fwd(q, k, v,atten_mask, causal, sm_scale, BM, BN)
    tri_out = tri_out.to(q.dtype)

    rtol = 0.0
    atol = 1e-2

    diff_golden_fa = (ref_out - tri_out).abs()
    print(f"diff_golden_fa (Max Diff): {diff_golden_fa.max().item()}")
    assert torch.allclose(ref_out, tri_out, atol=atol, rtol=rtol)
    ref_M = ref_softmaxmax + torch.log(ref_softmaxsum)
    assert torch.allclose(ref_M.mean(axis=-1), L, atol=atol, rtol=rtol)
    print("compare success!")


if __name__ == "__main__":
    test_op(8, 8, 2432, 64, True, torch.bfloat16, 128, 128)
