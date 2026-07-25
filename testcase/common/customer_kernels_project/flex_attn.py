import torch
import triton
import triton.language as tl

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

@triton.jit
def mask_fn(
        offs_m,
        offs_n,
        SEG_IDS,
        MODALITY_INDICATORS,
        DOC_START,
        Q_LEN: tl.constexpr,
        KV_LEN: tl.constexpr,
        MASK_TYPE: tl.constexpr,
        SLIDING_WINDOW: tl.constexpr,
        GLOBAL_WINDOW: tl.constexpr,
):
    q_seg = tl.load(SEG_IDS + offs_m, mask=offs_m < Q_LEN, other=1)[:, None]
    kv_seg = tl.load(SEG_IDS + offs_n, mask=offs_n < KV_LEN, other=1)[None, :]
    q_mod = tl.load(MODALITY_INDICATORS + offs_m, mask=offs_m < Q_LEN, other=1)[:, None]
    kv_mod = tl.load(MODALITY_INDICATORS + offs_n, mask=offs_n < KV_LEN, other=1)[None, :]

    same_doc = q_seg == kv_seg
    causal = offs_m[:, None] >= offs_n[None, :]
    is_img = q_mod > 0
    same_img = is_img & (q_mod == kv_mod)

    if MASK_TYPE == "full":
        return (same_doc & causal) | same_img

    q_doc_start = tl.load(DOC_START + offs_m, mask=offs_m < Q_LEN, other=0)[:, None]
    window = causal & ((offs_m[:, None] - offs_n[None, :]) <= SLIDING_WINDOW)
    use_global = GLOBAL_WINDOW > 0
    glob = causal & (offs_n[None, :] >= q_doc_start) & (offs_n[None, :] < (q_doc_start + GLOBAL_WINDOW))
    sparse = same_doc & (window | (glob & use_global))
    return sparse | same_img

@triton.jit
def flex_attention_kernel(
        Q, K, V,
        KV_NUM_BLKS, KV_IDX, FULL_KV_NUM_BLKS, FULL_KV_IDX,
        SEG_IDS, MODALITY_INDICATORS, DOC_START,
        OUT, LSE,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_out_z, stride_out_h, stride_out_m, stride_out_k,
        stride_lse_z, stride_lse_h, stride_lse_m,
        stride_kv_idx_m,
        SM_SCALE: tl.constexpr,
        QK_HEAD_DIM: tl.constexpr,
        V_HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_TASKS: tl.constexpr,
        NUM_Q_BLOCKS: tl.constexpr,
        Q_HEAD: tl.constexpr,
        SPARSE_Q_BLOCK_SIZE: tl.constexpr,
        SPARSE_KV_BLOCK_SIZE: tl.constexpr,
        Q_LEN: tl.constexpr,
        KV_LEN: tl.constexpr,
        MASK_TYPE: tl.constexpr = "full",
        SLIDING_WINDOW: tl.constexpr = 0,
        GLOBAL_WINDOW: tl.constexpr = 0,
        GQA_SHARED_HEAD: tl.constexpr = 1,
        HAS_FULL_BLOCKS: tl.constexpr = True,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    for task_id in range(pid, NUM_TASKS, num_core):
        q_start = task_id % NUM_Q_BLOCKS
        off_z = (task_id // NUM_Q_BLOCKS) // Q_HEAD
        off_hq = (task_id // NUM_Q_BLOCKS) % Q_HEAD
        off_hkv = off_hq // GQA_SHARED_HEAD

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

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, V_HEAD_DIM], dtype=tl.float32)

        offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, QK_HEAD_DIM)
        offs_v = tl.arange(0, V_HEAD_DIM)

        q = tl.load(Q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                    mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                    other=0.0
                    )

        SPARSE_Q_MULTIPLE = SPARSE_Q_BLOCK_SIZE // BLOCK_M
        SPARSE_KV_MULTIPLE = SPARSE_KV_BLOCK_SIZE // BLOCK_N

        q_sparse_idx = q_start // SPARSE_Q_MULTIPLE
        sparse_kv_num_blks_offset = q_sparse_idx
        sparse_kv_idx_offset = q_sparse_idx * stride_kv_idx_m

        kv_indices = KV_IDX + sparse_kv_idx_offset
        kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)
        block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE,
                                 tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))
        for start_n in range(0, block_n_end):
            blk_idx_in_list = start_n // SPARSE_KV_MULTIPLE
            kv_block = tl.load(kv_indices + blk_idx_in_list)
            kv_start = kv_block * SPARSE_KV_BLOCK_SIZE + (start_n % SPARSE_KV_MULTIPLE) * BLOCK_N

            offs_n_load = kv_start + tl.arange(0, BLOCK_N)
            k = tl.load(K_ptr + offs_n_load[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                        mask=(offs_n_load[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                        other=0.0
                        )
            k = tl.trans(k)

            qk = tl.dot(q, k, input_precision="ieee")
            qk *= SM_SCALE

            mask = mask_fn(
                offs_m,
                offs_n_load,
                SEG_IDS,
                MODALITY_INDICATORS,
                DOC_START,
                Q_LEN=Q_LEN,
                KV_LEN=KV_LEN,
                MASK_TYPE=MASK_TYPE,
                SLIDING_WINDOW=SLIDING_WINDOW,
                GLOBAL_WINDOW=GLOBAL_WINDOW,
            )
            qk = tl.where(mask, qk, float("-inf"))
            qk = tl.where(offs_n_load[None, :] < KV_LEN, qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            masked_out_rows = (m_ij == float("-inf"))
            m_ij_masked = tl.where(masked_out_rows, 0.0, m_ij)

            alpha = tl.math.exp(m_i - m_ij_masked)
            p = tl.math.exp(qk - m_ij_masked[:, None])

            v = tl.load(
                V_ptr + offs_n_load[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                mask=(offs_n_load[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                other=0.0
            )

            pv = tl.dot(p.to(Q.dtype.element_ty), v, input_precision="ieee")
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + pv
            m_i = m_ij

        if HAS_FULL_BLOCKS:
            kv_indices = FULL_KV_IDX + sparse_kv_idx_offset
            kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_kv_num_blks_offset)
            block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE,
                                     tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))
            for start_n in range(0, block_n_end):
                blk_idx_in_list = start_n // SPARSE_KV_MULTIPLE
                kv_block = tl.load(kv_indices + blk_idx_in_list)
                kv_start = kv_block * SPARSE_KV_BLOCK_SIZE + (start_n % SPARSE_KV_MULTIPLE) * BLOCK_N

                offs_n_load = kv_start + tl.arange(0, BLOCK_N)

                k = tl.load(K_ptr + offs_n_load[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                            mask=(offs_n_load[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                            other=0.0
                            )
                k = tl.trans(k)

                qk = tl.dot(q, k, input_precision="ieee")
                qk *= SM_SCALE
                qk = tl.where(offs_n_load[None, :] < KV_LEN, qk, float("-inf"))

                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                masked_out_rows = (m_ij == float("-inf"))
                m_ij_masked = tl.where(masked_out_rows, 0.0, m_ij)

                alpha = tl.math.exp(m_i - m_ij_masked)
                p = tl.math.exp(qk - m_ij_masked[:, None])

                v = tl.load(
                    V_ptr + offs_n_load[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                    mask=(offs_n_load[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                    other=0.0
                )
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
            mask=out_mask,
        )
        lse = m_i + tl.math.log(l_i)
        tl.store(LSE_ptr + offs_m * stride_lse_m, lse,
                 mask=offs_m < Q_LEN,
                 )

@triton.jit
def flex_attention_backward_dq_kernel(
        Q, K, V, DO, LSE, DELTA,
        KV_NUM_BLKS, KV_IDX, FULL_KV_NUM_BLKS, FULL_KV_IDX,
        SEG_IDS, MODALITY_INDICATORS, DOC_START,
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
        NUM_TASKS: tl.constexpr,
        NUM_Q_BLOCKS: tl.constexpr,
        Q_HEAD: tl.constexpr,
        SPARSE_Q_BLOCK_SIZE: tl.constexpr,
        SPARSE_KV_BLOCK_SIZE: tl.constexpr,
        Q_LEN: tl.constexpr,
        KV_LEN: tl.constexpr,
        MASK_TYPE: tl.constexpr = "full",
        SLIDING_WINDOW: tl.constexpr = 0,
        GLOBAL_WINDOW: tl.constexpr = 0,
        GQA_SHARED_HEAD: tl.constexpr = 1,
        HAS_FULL_BLOCKS: tl.constexpr = True,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    for task_id in range(pid, NUM_TASKS, num_core):
        q_start = task_id % NUM_Q_BLOCKS
        off_z = (task_id // NUM_Q_BLOCKS) // Q_HEAD
        off_hq = (task_id // NUM_Q_BLOCKS) % Q_HEAD
        off_hkv = off_hq // GQA_SHARED_HEAD

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

        q = tl.load(Q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                    mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                    other=0.0)
        do = tl.load(DO_ptr + offs_m[:, None] * stride_dom + offs_v[None, :] * stride_dok,
                     mask=(offs_m[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                     other=0.0)
        lse = tl.load(LSE_ptr + offs_m * stride_lse_m, mask=offs_m < Q_LEN, other=0.0)
        delta = tl.load(DELTA_ptr + offs_m * stride_delta_m, mask=offs_m < Q_LEN, other=0.0)

        dq = tl.zeros([BLOCK_M, QK_HEAD_DIM], dtype=tl.float32)

        SPARSE_Q_MULTIPLE = SPARSE_Q_BLOCK_SIZE // BLOCK_M
        SPARSE_KV_MULTIPLE = SPARSE_KV_BLOCK_SIZE // BLOCK_N

        q_sparse_idx = q_start // SPARSE_Q_MULTIPLE
        sparse_kv_num_blks_offset = q_sparse_idx
        sparse_kv_idx_offset = q_sparse_idx * stride_kv_idx_m

        kv_indices = KV_IDX + sparse_kv_idx_offset
        kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)
        block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE,
                                 tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))
        for start_n in range(0, block_n_end):
            blk_idx_in_list = start_n // SPARSE_KV_MULTIPLE
            kv_block = tl.load(kv_indices + blk_idx_in_list)
            kv_start = kv_block * SPARSE_KV_BLOCK_SIZE + (start_n % SPARSE_KV_MULTIPLE) * BLOCK_N

            offs_n_load = kv_start + tl.arange(0, BLOCK_N)

            k = tl.load(K_ptr + offs_n_load[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                        mask=(offs_n_load[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                        other=0.0)
            v = tl.load(V_ptr + offs_n_load[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                        mask=(offs_n_load[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                        other=0.0)

            k_t = tl.trans(k)
            qk = tl.dot(q, k_t, input_precision="ieee")
            qk *= SM_SCALE

            mask = mask_fn(
                offs_m,
                offs_n_load,
                SEG_IDS,
                MODALITY_INDICATORS,
                DOC_START,
                Q_LEN=Q_LEN,
                KV_LEN=KV_LEN,
                MASK_TYPE=MASK_TYPE,
                SLIDING_WINDOW=SLIDING_WINDOW,
                GLOBAL_WINDOW=GLOBAL_WINDOW,
            )

            p = tl.math.exp(qk - lse[:, None])
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
            ds = p * (dp - delta[:, None])
            ds = tl.where(mask, ds, 0.0)
            ds = tl.where(offs_n_load[None, :] < KV_LEN, ds, 0.0)
            ds = ds.to(Q.dtype.element_ty)
            dq += tl.dot(ds, k, input_precision="ieee")

        if HAS_FULL_BLOCKS:
            kv_indices = FULL_KV_IDX + sparse_kv_idx_offset
            kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_kv_num_blks_offset)
            block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE,
                                     tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))
            for start_n in range(0, block_n_end):
                blk_idx_in_list = start_n // SPARSE_KV_MULTIPLE
                kv_block = tl.load(kv_indices + blk_idx_in_list)
                kv_start = kv_block * SPARSE_KV_BLOCK_SIZE + (start_n % SPARSE_KV_MULTIPLE) * BLOCK_N

                offs_n_load = kv_start + tl.arange(0, BLOCK_N)

                k = tl.load(K_ptr + offs_n_load[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                            mask=(offs_n_load[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                            other=0.0)
                v = tl.load(V_ptr + offs_n_load[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                            mask=(offs_n_load[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                            other=0.0)

                k_t = tl.trans(k)
                qk = tl.dot(q, k_t, input_precision="ieee")
                qk *= SM_SCALE
                qk = tl.where(offs_n_load[None, :] < KV_LEN, qk, float("-inf"))

                p = tl.math.exp(qk - lse[:, None])
                dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                ds = p * (dp - delta[:, None])
                ds = tl.where(offs_n_load[None, :] < KV_LEN, ds, 0.0)
                ds = ds.to(Q.dtype.element_ty)
                dq += tl.dot(ds, k, input_precision="ieee")

        dq *= SM_SCALE
        dq_mask = (offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM)
        tl.store(DQ_ptr + offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk,
                 dq, mask=dq_mask)

@triton.jit
def flex_attention_backward_dkdv_kernel(
        Q, K, V, DO, LSE, DELTA,
        Q_NUM_BLKS, Q_IDX, FULL_Q_NUM_BLKS, FULL_Q_IDX,
        SEG_IDS, MODALITY_INDICATORS, DOC_START,
        DK, DV,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_doz, stride_doh, stride_dom, stride_dok,
        stride_lse_z, stride_lse_h, stride_lse_m,
        stride_delta_z, stride_delta_h, stride_delta_m,
        stride_dkz, stride_dkh, stride_dkn, stride_dkk,
        stride_dvz, stride_dvh, stride_dvn, stride_dvk,
        stride_q_idx_n,
        SM_SCALE: tl.constexpr,
        QK_HEAD_DIM: tl.constexpr,
        V_HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_TASKS: tl.constexpr,
        NUM_KV_BLOCKS: tl.constexpr,
        KV_HEAD: tl.constexpr,
        SPARSE_Q_BLOCK_SIZE: tl.constexpr,
        SPARSE_KV_BLOCK_SIZE: tl.constexpr,
        Q_LEN: tl.constexpr,
        KV_LEN: tl.constexpr,
        MASK_TYPE: tl.constexpr = "full",
        SLIDING_WINDOW: tl.constexpr = 0,
        GLOBAL_WINDOW: tl.constexpr = 0,
        GQA_SHARED_HEAD: tl.constexpr = 1,
        HAS_FULL_BLOCKS: tl.constexpr = True,
):
    pid = tl.program_id(0).to(tl.int32)
    num_core = tl.num_programs(0).to(tl.int32)

    for task_id in range(pid, NUM_TASKS, num_core):
        kv_start_blk = task_id % NUM_KV_BLOCKS
        off_z = (task_id // NUM_KV_BLOCKS) // KV_HEAD
        off_hkv = (task_id // NUM_KV_BLOCKS) % KV_HEAD

        dk = tl.zeros([BLOCK_N, QK_HEAD_DIM], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, V_HEAD_DIM], dtype=tl.float32)

        SPARSE_Q_MULTIPLE = SPARSE_Q_BLOCK_SIZE // BLOCK_M
        SPARSE_KV_MULTIPLE = SPARSE_KV_BLOCK_SIZE // BLOCK_N

        kv_sparse_idx = kv_start_blk // SPARSE_KV_MULTIPLE
        sparse_q_num_blks_offset = kv_sparse_idx
        sparse_q_idx_offset = kv_sparse_idx * stride_q_idx_n

        offs_n = kv_start_blk * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, QK_HEAD_DIM)
        offs_v = tl.arange(0, V_HEAD_DIM)

        k_base_offset = off_z * stride_kz + off_hkv * stride_kh
        v_base_offset = off_z * stride_vz + off_hkv * stride_vh
        K_ptr = K + k_base_offset
        V_ptr = V + v_base_offset

        k = tl.load(K_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                    mask=(offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                    other=0.0)
        v = tl.load(V_ptr + offs_n[:, None] * stride_vn + offs_v[None, :] * stride_vk,
                    mask=(offs_n[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                    other=0.0)

        q_num_blocks = tl.load(Q_NUM_BLKS + sparse_q_num_blks_offset)
        for q_blk_iter in range(0, q_num_blocks):
            q_block = tl.load(Q_IDX + sparse_q_idx_offset + q_blk_iter)

            for hq_offset in range(GQA_SHARED_HEAD):
                off_hq = off_hkv * GQA_SHARED_HEAD + hq_offset

                q_base_offset = off_z * stride_qz + off_hq * stride_qh
                do_base_offset = off_z * stride_doz + off_hq * stride_doh
                lse_base_offset = off_z * stride_lse_z + off_hq * stride_lse_h
                delta_base_offset = off_z * stride_delta_z + off_hq * stride_delta_h

                Q_ptr = Q + q_base_offset
                DO_ptr = DO + do_base_offset
                LSE_ptr = LSE + lse_base_offset
                DELTA_ptr = DELTA + delta_base_offset

                for q_inner in range(SPARSE_Q_MULTIPLE):
                    q_row_start = q_block * SPARSE_Q_BLOCK_SIZE + q_inner * BLOCK_M
                    offs_m = q_row_start + tl.arange(0, BLOCK_M)
                    valid_q = q_row_start < Q_LEN

                    q = tl.load(Q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                                mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                                other=0.0)

                    k_t = tl.trans(k)
                    qk = tl.dot(q, k_t, input_precision="ieee")
                    qk *= SM_SCALE

                    mask = mask_fn(
                        offs_m,
                        offs_n,
                        SEG_IDS,
                        MODALITY_INDICATORS,
                        DOC_START,
                        Q_LEN=Q_LEN,
                        KV_LEN=KV_LEN,
                        MASK_TYPE=MASK_TYPE,
                        SLIDING_WINDOW=SLIDING_WINDOW,
                        GLOBAL_WINDOW=GLOBAL_WINDOW,
                    )

                    lse = tl.load(LSE_ptr + offs_m * stride_lse_m, mask=offs_m < Q_LEN, other=0.0)
                    p = tl.math.exp(qk - lse[:, None])
                    p = tl.where(mask, p, 0.0)
                    p = tl.where(offs_n[None, :] < KV_LEN, p, 0.0)
                    p = tl.where(valid_q, p, 0.0)

                    do = tl.load(DO_ptr + offs_m[:, None] * stride_dom + offs_v[None, :] * stride_dok,
                                 mask=(offs_m[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                                 other=0.0)
                    dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do, input_precision="ieee")

                    delta = tl.load(DELTA_ptr + offs_m * stride_delta_m, mask=offs_m < Q_LEN, other=0.0)
                    dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                    ds = p * (dp - delta[:, None])
                    ds = tl.where(mask, ds, 0.0)
                    ds = tl.where(offs_n[None, :] < KV_LEN, ds, 0.0)
                    ds = tl.where(valid_q, ds, 0.0)
                    ds = ds.to(Q.dtype.element_ty)
                    dk += tl.dot(tl.trans(ds), q, input_precision="ieee")

        if HAS_FULL_BLOCKS:
            full_q_num_blocks = tl.load(FULL_Q_NUM_BLKS + sparse_q_num_blks_offset)
            for q_blk_iter in range(0, full_q_num_blocks):
                q_block = tl.load(FULL_Q_IDX + sparse_q_idx_offset + q_blk_iter)

                for hq_offset in range(GQA_SHARED_HEAD):
                    off_hq = off_hkv * GQA_SHARED_HEAD + hq_offset

                    q_base_offset = off_z * stride_qz + off_hq * stride_qh
                    do_base_offset = off_z * stride_doz + off_hq * stride_doh
                    lse_base_offset = off_z * stride_lse_z + off_hq * stride_lse_h
                    delta_base_offset = off_z * stride_delta_z + off_hq * stride_delta_h

                    Q_ptr = Q + q_base_offset
                    DO_ptr = DO + do_base_offset
                    LSE_ptr = LSE + lse_base_offset
                    DELTA_ptr = DELTA + delta_base_offset

                    for q_inner in range(SPARSE_Q_MULTIPLE):
                        q_row_start = q_block * SPARSE_Q_BLOCK_SIZE + q_inner * BLOCK_M
                        offs_m = q_row_start + tl.arange(0, BLOCK_M)
                        valid_q = q_row_start < Q_LEN

                        q = tl.load(Q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                                    mask=(offs_m[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM),
                                    other=0.0)

                        k_t = tl.trans(k)
                        qk = tl.dot(q, k_t, input_precision="ieee")
                        qk *= SM_SCALE
                        qk = tl.where(offs_n[None, :] < KV_LEN, qk, float("-inf"))

                        lse = tl.load(LSE_ptr + offs_m * stride_lse_m, mask=offs_m < Q_LEN, other=0.0)
                        p = tl.math.exp(qk - lse[:, None])
                        p = tl.where(offs_n[None, :] < KV_LEN, p, 0.0)
                        p = tl.where(valid_q, p, 0.0)

                        do = tl.load(DO_ptr + offs_m[:, None] * stride_dom + offs_v[None, :] * stride_dok,
                                     mask=(offs_m[:, None] < Q_LEN) & (offs_v[None, :] < V_HEAD_DIM),
                                     other=0.0)
                        dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do, input_precision="ieee")

                        delta = tl.load(DELTA_ptr + offs_m * stride_delta_m, mask=offs_m < Q_LEN, other=0.0)
                        dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                        ds = p * (dp - delta[:, None])
                        ds = tl.where(offs_n[None, :] < KV_LEN, ds, 0.0)
                        ds = tl.where(valid_q, ds, 0.0)
                        ds = ds.to(Q.dtype.element_ty)
                        dk += tl.dot(tl.trans(ds), q, input_precision="ieee")

        dk *= SM_SCALE

        DK_ptr = DK + off_z * stride_dkz + off_hkv * stride_dkh
        DV_ptr = DV + off_z * stride_dvz + off_hkv * stride_dvh

        dk_store_mask = (offs_n[:, None] < KV_LEN) & (offs_k[None, :] < QK_HEAD_DIM)
        dv_store_mask = (offs_n[:, None] < KV_LEN) & (offs_v[None, :] < V_HEAD_DIM)
        tl.store(DK_ptr + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk,
                 dk, mask=dk_store_mask)
        tl.store(DV_ptr + offs_n[:, None] * stride_dvn + offs_v[None, :] * stride_dvk,
                 dv, mask=dv_store_mask)

class FlexAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            q,
            k,
            v,
            block_mask=None,
            score_mask=None,
            mask_type="full",
            doc_start=None,
            sliding_window=0,
            global_window=0,
    ):
        assert q.dim() == 4, "Q must be 4D tensor"
        assert k.dim() == 4, "K must be 4D tensor"
        assert v.dim() == 4, "V must be 4D tensor"

        Z, Hq, M, D = q.shape
        _, Hkv, N, Dv = k.shape

        GQA_SHARED_HEAD = Hq // Hkv if Hq >= Hkv else 1
        assert k.shape == v.shape, "K and V must have same shape"

        SM_SCALE = 1.0 / (D ** 0.5)

        BLOCK_SIZE = 128
        if D <= 64:
            BLOCK_M = 128
            BLOCK_N = 128
        else:
            BLOCK_M = 64
            BLOCK_N = 64
        SPARSE_Q_BLOCK_SIZE = BLOCK_SIZE
        SPARSE_KV_BLOCK_SIZE = BLOCK_SIZE

        num_q_blocks = (M + BLOCK_M - 1) // BLOCK_M

        output = torch.empty_like(q)
        lse = torch.empty((Z, Hq, M), dtype=torch.float32, device=q.device)

        kv_num_blks = block_mask.kv_num_blocks
        kv_idx = block_mask.kv_indices
        full_kv_num_blks = getattr(block_mask, "full_kv_num_blocks", torch.zeros_like(kv_num_blks))
        full_kv_idx = getattr(block_mask, "full_kv_indices", torch.zeros_like(kv_idx))

        q_num_blks = getattr(block_mask, "q_num_blocks", None)
        q_idx = getattr(block_mask, "q_indices", None)
        assert q_num_blks is not None and q_idx is not None, "q_num_blocks and q_indices must be provided"
        full_q_num_blks = getattr(block_mask, "full_q_num_blocks", torch.zeros_like(q_num_blks))
        full_q_idx = getattr(block_mask, "full_q_indices", torch.zeros_like(q_idx))

        seg_idx = getattr(block_mask, "segment_ids", None)
        modality_indicators = getattr(block_mask, "modality_indicators", None)
        doc_start = doc_start if doc_start is not None else getattr(block_mask, "doc_start", None)
        sliding_window = sliding_window if sliding_window is not None else getattr(block_mask, "sliding_window", None)
        global_window = global_window if global_window is not None else getattr(block_mask, "global_window", None)

        assert seg_idx is not None, "segment_ids must be provided"
        assert modality_indicators is not None, "modality_indicators must be provided"

        if mask_type not in ["full", "sparse"]:
            raise ValueError(f"mask_type must be 'full' or 'sparse', but got {mask_type}")

        if mask_type == "sparse":
            assert doc_start is not None, "doc_start must be provided for sparse mask"
            assert sliding_window is not None, "sliding_window must be provided for sparse mask"
        if doc_start is None:
            doc_start = torch.zeros((M,), dtype=torch.int64, device=q.device)

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

        seg_idx = seg_idx.contiguous()
        modality_indicators = modality_indicators.contiguous()
        doc_start = doc_start.contiguous()

        num_tasks = num_q_blocks * Z * Hq
        grid, num_tasks = _persistent_launch_config(num_tasks)

        kv_num_blks = kv_num_blks.to(torch.int32)
        kv_idx = kv_idx.to(torch.int32)
        full_kv_num_blks = full_kv_num_blks.to(torch.int32)
        full_kv_idx = full_kv_idx.to(torch.int32)
        seg_idx = seg_idx.to(torch.int32)
        modality_indicators = modality_indicators.to(torch.int32)
        doc_start = doc_start.to(torch.int32)
        q_num_blks = q_num_blks.to(torch.int32)
        q_idx = q_idx.to(torch.int32)
        full_q_num_blks = full_q_num_blks.to(torch.int32)
        full_q_idx = full_q_idx.to(torch.int32)

        flex_attention_kernel[grid](
            q, k, v,
            kv_num_blks, kv_idx, full_kv_num_blks, full_kv_idx,
            seg_idx, modality_indicators, doc_start,
            output, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            kv_idx.stride(2),
            SM_SCALE=SM_SCALE,
            QK_HEAD_DIM=D,
            V_HEAD_DIM=Dv,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            NUM_TASKS=num_tasks, NUM_Q_BLOCKS=num_q_blocks, Q_HEAD=Hq,
            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
            Q_LEN=M, KV_LEN=N,
            MASK_TYPE=mask_type,
            SLIDING_WINDOW=0 if sliding_window is None else int(sliding_window),
            GLOBAL_WINDOW=0 if global_window is None else int(global_window),
            GQA_SHARED_HEAD=GQA_SHARED_HEAD,
            HAS_FULL_BLOCKS=True,
        )

        ctx.save_for_backward(q, k, v, output, lse,
                              seg_idx, modality_indicators, doc_start,
                              kv_num_blks, kv_idx, full_kv_num_blks, full_kv_idx,
                              q_num_blks, q_idx, full_q_num_blks, full_q_idx,
                              )
        ctx.mask_type = mask_type
        ctx.sliding_window = sliding_window
        ctx.global_window = global_window
        ctx.gqa_shared_head = GQA_SHARED_HEAD
        ctx.sm_scale = SM_SCALE
        ctx.sparse_q_block_size = SPARSE_Q_BLOCK_SIZE
        ctx.sparse_kv_block_size = SPARSE_KV_BLOCK_SIZE
        ctx.has_full_blocks = True

        return output, lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse=None):
        (
            q, k, v, output, lse,
            seg_idx, modality_indicators, doc_start,
            kv_num_blks, kv_idx, full_kv_num_blks, full_kv_idx,
            q_num_blks, q_idx, full_q_num_blks, full_q_idx,
        ) = ctx.saved_tensors

        Z, Hq, M, D = q.shape
        _, Hkv, N, Dv = k.shape
        GQA_SHARED_HEAD = ctx.gqa_shared_head

        grad_output = grad_output.contiguous()
        delta = (output * grad_output).sum(dim=-1).to(torch.float32).contiguous()

        if D <= 64:
            DQ_BLOCK_M = 128
            DQ_BLOCK_N = 128
        else:
            DQ_BLOCK_M = 64
            DQ_BLOCK_N = 64

        dq = torch.empty_like(q)
        num_q_blocks = (M + DQ_BLOCK_M - 1) // DQ_BLOCK_M
        num_tasks_dq = num_q_blocks * Z * Hq
        grid_dq, num_tasks_dq = _persistent_launch_config(num_tasks_dq)

        flex_attention_backward_dq_kernel[grid_dq](
            q, k, v, grad_output, lse, delta,
            kv_num_blks, kv_idx, full_kv_num_blks, full_kv_idx,
            seg_idx, modality_indicators, doc_start,
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
            BLOCK_M=DQ_BLOCK_M,
            BLOCK_N=DQ_BLOCK_N,
            NUM_TASKS=num_tasks_dq,
            NUM_Q_BLOCKS=num_q_blocks,
            Q_HEAD=Hq,
            SPARSE_Q_BLOCK_SIZE=ctx.sparse_q_block_size,
            SPARSE_KV_BLOCK_SIZE=ctx.sparse_kv_block_size,
            Q_LEN=M,
            KV_LEN=N,
            MASK_TYPE=ctx.mask_type,
            SLIDING_WINDOW=0 if ctx.sliding_window is None else int(ctx.sliding_window),
            GLOBAL_WINDOW=0 if ctx.global_window is None else int(ctx.global_window),
            GQA_SHARED_HEAD=GQA_SHARED_HEAD,
            HAS_FULL_BLOCKS=ctx.has_full_blocks,
        )

        dk = torch.zeros((Z, Hkv, N, D), dtype=torch.float32, device=k.device)
        dv = torch.zeros((Z, Hkv, N, Dv), dtype=torch.float32, device=v.device)

        DKBLOCK_M = 64
        DKBLOCK_N = 64
        num_kv_blocks = (N + DKBLOCK_N - 1) // DKBLOCK_N
        num_tasks_dkdv = num_kv_blocks * Z * Hkv
        grid_dkdv, num_tasks_dkdv = _persistent_launch_config(num_tasks_dkdv)

        flex_attention_backward_dkdv_kernel[grid_dkdv](
            q, k, v, grad_output, lse, delta,
            q_num_blks, q_idx, full_q_num_blks, full_q_idx,
            seg_idx, modality_indicators, doc_start,
            dk, dv,
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
            BLOCK_M=DKBLOCK_M,
            BLOCK_N=DKBLOCK_N,
            NUM_TASKS=num_tasks_dkdv,
            NUM_KV_BLOCKS=num_kv_blocks,
            KV_HEAD=Hkv,
            SPARSE_Q_BLOCK_SIZE=ctx.sparse_q_block_size,
            SPARSE_KV_BLOCK_SIZE=ctx.sparse_kv_block_size,
            Q_LEN=M,
            KV_LEN=N,
            MASK_TYPE=ctx.mask_type,
            SLIDING_WINDOW=0 if ctx.sliding_window is None else int(ctx.sliding_window),
            GLOBAL_WINDOW=0 if ctx.global_window is None else int(ctx.global_window),
            GQA_SHARED_HEAD=GQA_SHARED_HEAD,
            HAS_FULL_BLOCKS=ctx.has_full_blocks,
        )

        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, None, None

def flex_attention(
        q,
        k,
        v,
        block_mask=None,
        score_mask=None,
        mask_type="full",
        doc_start=None,
        sliding_window=0,
        global_window=0,
        return_lse=False,
):
    output, lse = FlexAttentionFunc.apply(q, k, v, block_mask, score_mask, mask_type, doc_start, sliding_window,
                                          global_window)
    if return_lse:
        return output, lse

    return output

def _expand_kv_for_gqa(kv, hq):
    Z, Hkv, S, D = kv.shape
    gqa = hq // Hkv
    return kv.unsqueeze(2).expand(Z, Hkv, gqa, S, D).reshape(Z, hq, S, D)


def _build_block_indices(full_mask, block_size, M, N, device):
    num_q_blocks = (M + block_size - 1) // block_size
    num_kv_blocks = (N + block_size - 1) // block_size

    sparse_kv, full_kv = [], []
    for qb in range(num_q_blocks):
        qs, qe = qb * block_size, min(qb * block_size + block_size, M)
        sb, fb = [], []
        for kvb in range(num_kv_blocks):
            ks, ke = kvb * block_size, min(kvb * block_size + block_size, N)
            blk = full_mask[qs:qe, ks:ke]
            if blk.all():
                fb.append(kvb)
            elif blk.any():
                sb.append(kvb)
        sparse_kv.append(sb)
        full_kv.append(fb)

    sparse_q, full_q = [], []
    for kvb in range(num_kv_blocks):
        ks, ke = kvb * block_size, min(kvb * block_size + block_size, N)
        sq, fq = [], []
        for qb in range(num_q_blocks):
            qs, qe = qb * block_size, min(qb * block_size + block_size, M)
            blk = full_mask[qs:qe, ks:ke]
            if blk.all():
                fq.append(qb)
            elif blk.any():
                sq.append(qb)
        sparse_q.append(sq)
        full_q.append(fq)

    def _pad(lst, ml, pv=-1):
        return (lst + [pv] * (ml - len(lst)))[:ml]

    def _make_idx(lst_list):
        mx = max(len(x) for x in lst_list) if lst_list else 0
        return torch.tensor([_pad(x, max(mx, 1)) for x in lst_list],
                            dtype=torch.int32, device=device).unsqueeze(0).unsqueeze(0)

    def _make_cnt(lst_list):
        return torch.tensor([len(x) for x in lst_list],
                            dtype=torch.int32, device=device).unsqueeze(0).unsqueeze(0)

    return (_make_idx(sparse_kv), _make_cnt(sparse_kv),
            _make_idx(full_kv), _make_cnt(full_kv),
            _make_idx(sparse_q), _make_cnt(sparse_q),
            _make_idx(full_q), _make_cnt(full_q))

def _pytorch_flex_attention(q, k, v, full_mask, mask_type="full",
                            seg_ids=None, modality_indicators=None, doc_start=None,
                            sliding_window=0, global_window=0, return_lse=False):
    Z, Hq, M, D = q.shape
    _, Hkv, N, Dv = k.shape
    GQA = Hq // Hkv
    scale = 1.0 / (D ** 0.5)
    q_f = q.float()
    k_f = _expand_kv_for_gqa(k.float(), Hq)
    v_f = _expand_kv_for_gqa(v.float(), Hq)
    if full_mask is not None:
        m = full_mask.unsqueeze(0).unsqueeze(0).expand(Z, Hq, -1, -1)
    else:
        m = torch.ones(Z, Hq, M, N, dtype=torch.bool, device=q.device)
    s = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
    s = s.masked_fill(~m, float('-inf'))
    s_max = s.max(dim=-1, keepdim=True).values
    p = torch.exp(s - s_max)
    p = p.masked_fill(~m, 0.0)
    l = p.sum(dim=-1, keepdim=True)
    l = torch.where(l == 0.0, 1.0, l)
    out = (p / l) @ v_f
    lse = (s_max + torch.log(l)).squeeze(-1)
    return (out, lse) if return_lse else out

if __name__ == "__main__":
    torch.manual_seed(42)
    device, dtype = "npu", torch.float32

    BATCH, HQ, HKV, M, N, D, Dv, NUM_DOCS = 1, 4, 2, 256, 256, 64, 64, 2
    GQA = HQ // HKV
    DOC_SIZE = M // NUM_DOCS
    segment_ids = torch.zeros(M, dtype=torch.int32, device=device)
    for i in range(NUM_DOCS):
        segment_ids[i * DOC_SIZE:(i + 1) * DOC_SIZE] = i
    modality_indicators = torch.zeros(M, dtype=torch.int32, device=device)
    modality_indicators[DOC_SIZE // 2:DOC_SIZE // 2 + 10] = 1
    doc_start_tensor = torch.zeros(M, dtype=torch.int32, device=device)
    for i in range(NUM_DOCS):
        doc_start_tensor[i * DOC_SIZE:(i + 1) * DOC_SIZE] = i * DOC_SIZE

    q_idx_r = torch.arange(M, device=device)[:, None]
    kv_idx_r = torch.arange(N, device=device)[None, :]
    same_doc = segment_ids[q_idx_r] == segment_ids[kv_idx_r]
    causal = q_idx_r >= kv_idx_r
    is_img = modality_indicators[q_idx_r] > 0
    same_img = is_img & (modality_indicators[q_idx_r] == modality_indicators[kv_idx_r])
    full_mask = (same_doc & causal) | same_img

    full_mask_cpu = full_mask.cpu()

    q = torch.randn(BATCH, HQ, M, D, dtype=dtype, device="npu", requires_grad=True)
    k = torch.randn(BATCH, HKV, N, D, dtype=dtype, device="npu", requires_grad=True)
    v = torch.randn(BATCH, HKV, N, Dv, dtype=dtype, device="npu", requires_grad=True)

    BLOCK_SIZE = 128
    (kv_idx_t, kv_num_t, full_kv_idx_t, full_kv_num_t,
     q_idx_t, q_num_t, full_q_idx_t, full_q_num_t) = _build_block_indices(
        full_mask, BLOCK_SIZE, M, N, "npu")
    class SimpleBlockMask:
        pass
    bm = SimpleBlockMask()
    bm.kv_num_blocks = kv_num_t
    bm.kv_indices = kv_idx_t
    bm.full_kv_num_blocks = full_kv_num_t
    bm.full_kv_indices = full_kv_idx_t
    bm.q_num_blocks = q_num_t
    bm.q_indices = q_idx_t
    bm.full_q_num_blocks = full_q_num_t
    bm.full_q_indices = full_q_idx_t
    bm.segment_ids = segment_ids.npu()
    bm.modality_indicators = modality_indicators.npu()
    bm.doc_start = doc_start_tensor.npu()

    out_triton, _ = flex_attention(q, k, v, block_mask=bm, mask_type="full", return_lse=True)

    q_cpu = q.detach().cpu().requires_grad_(True)
    k_cpu = k.detach().cpu().requires_grad_(True)
    v_cpu = v.detach().cpu().requires_grad_(True)
    ref_out = _pytorch_flex_attention(q_cpu, k_cpu, v_cpu, full_mask_cpu, return_lse=False)
    fwd_diff = (out_triton.cpu() - ref_out).abs().max().item()

    do = torch.randn_like(out_triton)
    out_triton.backward(do)
    dq_t = q.grad.clone().cpu()
    dk_t = k.grad.clone().cpu()
    dv_t = v.grad.clone().cpu()

    ref_out.backward(do.cpu())
    dq_r = q_cpu.grad.float()
    dk_r = k_cpu.grad.float()
    dv_r = v_cpu.grad.float()

    dq_diff = (dq_t.float() - dq_r).abs().max().item()
    dk_diff = (dk_t.float() - dk_r).abs().max().item()
    dv_diff = (dv_t.float() - dv_r).abs().max().item()

    tol = 1e-2
    ok = fwd_diff < tol and dq_diff < tol and dk_diff < tol and dv_diff < tol
    assert ok