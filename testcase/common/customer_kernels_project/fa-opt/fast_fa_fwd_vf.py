import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
import triton.extension.buffer.language as bl


@triton.jit
def softmax_with_mask_with_update_BN_128(vtaskId, qk, sm_scale, attn_mask_ptr, m_i_buffer, l_i_buffer, alpha_buffer, p_nz_buffer, qk_scale, cast_dtype, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, STAGE: tl.constexpr):
    sub_vec_id = al.sub_vec_id()
    m_i = bl.to_tensor(m_i_buffer)
    l_i = bl.to_tensor(l_i_buffer)
    alpha = bl.to_tensor(alpha_buffer)
    p_nz = bl.to_tensor(p_nz_buffer)

    l_ij = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
    l_ij = bl.to_tensor(l_ij)
    tmp_max = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
    tmp_max = bl.to_tensor(tmp_max)

    atten_mask = tl.zeros((BLOCK_M // 2, BLOCK_N), dtype=tl.int8)

    if STAGE == 1:
        # curr_mask_ptr = tl.advance(attn_mask_ptr, ((sub_vec_id * BLOCK_M // 2).to(tl.int32), 0))
        # atten_mask = tl.load(attn_mask_ptr, boundary_check=(0, 1), padding_option="zero") # 64*128
        atten_mask = attn_mask_ptr

    BLOCK_N_UNROLL : tl.constexpr = BLOCK_N // 2
    with al.scope(vector_mode="simd", outline=True):
        for loop in range(BLOCK_M // 2):
            qk_loop = al.extract_slice(qk, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop = qk_loop * sm_scale
            if STAGE == 1:
                mask = al.extract_slice(atten_mask, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1]).to(tl.int1) # 64*128 -> 1*64
                qk_loop = qk_loop + tl.where(mask, -1.0e4, 0)
            qk_scale = al.insert_slice(qk_scale, qk_loop, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])

            qk_loop_unroll = al.extract_slice(qk, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop_unroll = qk_loop_unroll * sm_scale
            if STAGE == 1:
                mask_unroll = al.extract_slice(atten_mask, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1]).to(tl.int1)
                qk_loop_unroll = qk_loop_unroll + tl.where(mask_unroll, -1.0e4, 0)
            qk_scale = al.insert_slice(qk_scale, qk_loop_unroll, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])

            row_max = tl.maximum(qk_loop, qk_loop_unroll, propagate_nan=tl.PropagateNan.ALL)
            row_max_agg = tl.max(row_max, 1, propagate_nan=True)
            
            tmp_max = al.insert_slice(tmp_max, row_max_agg, [loop], [1], [1])
        m_ij = tl.maximum(m_i, tmp_max, propagate_nan=tl.PropagateNan.ALL)

        al.debug_barrier(al.SYNC_IN_VF.VST_VLD)

        for loop in range(BLOCK_M // 2):
            m_ij_loop = al.extract_slice(m_ij, [loop], [1], [1])

            qk_loop = al.extract_slice(qk_scale, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop_unroll = al.extract_slice(qk_scale, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])

            qk_loop = qk_loop - m_ij_loop[:, None]
            qk_loop_unroll = qk_loop_unroll - m_ij_loop[:, None]

            p_loop = tl.math.exp(qk_loop)
            p_loop_unroll = tl.math.exp(qk_loop_unroll)
            
            p_loop_reshape = p_loop.reshape(BLOCK_N_UNROLL // 16, 1, 16)
            p_cast_loop = p_loop_reshape.to(cast_dtype)
            p_nz = al.insert_slice(p_nz, p_cast_loop, [0, loop, 0], [BLOCK_N_UNROLL//16, 1, 16], [1, 1, 1])
            
            p_loop_unroll_reshape = p_loop_unroll.reshape(BLOCK_N_UNROLL // 16, 1, 16)
            p_cast_loop_unroll = p_loop_unroll_reshape.to(cast_dtype)
            p_nz = al.insert_slice(p_nz, p_cast_loop_unroll, [BLOCK_N_UNROLL//16, loop, 0], [BLOCK_N_UNROLL//16, 1, 16], [1, 1, 1])
            
            row_sum = p_loop + p_loop_unroll
            l_ij_loop = tl.sum(row_sum, 1)
            l_ij = al.insert_slice(l_ij, l_ij_loop, [loop], [1], [1])

    with al.scope(vector_mode="simd", outline=True, no_inline=True):
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

    al.copy(bl.to_buffer(m_ij, al.ascend_address_space.UB), m_i_buffer)

    bl.to_buffer(l_i, bind_buffer=l_i_buffer)
    bl.to_buffer(p_nz, bind_buffer=p_nz_buffer)
    bl.to_buffer(alpha, bind_buffer=alpha_buffer)

@triton.jit
def softmax_no_mask_with_update(vtaskId, qk, sm_scale,  m_i_buffer, l_i_buffer, alpha_buffer, p_nz_buffer, qk_scale, cast_dtype, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, STAGE: tl.constexpr):
    sub_vec_id = al.sub_vec_id()
    m_i = bl.to_tensor(m_i_buffer)
    l_i = bl.to_tensor(l_i_buffer)
    alpha = bl.to_tensor(alpha_buffer)
    p_nz = bl.to_tensor(p_nz_buffer)

    l_ij = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
    l_ij = bl.to_tensor(l_ij)
    tmp_max = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
    tmp_max = bl.to_tensor(tmp_max)

    BLOCK_N_UNROLL : tl.constexpr = BLOCK_N // 2
    with al.scope(vector_mode="simd", outline=True):
        for loop in range(BLOCK_M // 2):
            qk_loop = al.extract_slice(qk, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop = qk_loop * sm_scale
            qk_scale = al.insert_slice(qk_scale, qk_loop, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])

            qk_loop_unroll = al.extract_slice(qk, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop_unroll = qk_loop_unroll * sm_scale
            qk_scale = al.insert_slice(qk_scale, qk_loop_unroll, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])

            row_max = tl.maximum(qk_loop, qk_loop_unroll, propagate_nan=tl.PropagateNan.ALL)
            row_max_agg = tl.max(row_max, 1, propagate_nan=True)
            
            tmp_max = al.insert_slice(tmp_max, row_max_agg, [loop], [1], [1])

        m_ij = tl.maximum(m_i, tmp_max, propagate_nan=tl.PropagateNan.ALL)
       
        al.debug_barrier(al.SYNC_IN_VF.VST_VLD)

        for loop in range(BLOCK_M // 2):
            m_ij_loop = al.extract_slice(m_ij, [loop], [1], [1])

            qk_loop = al.extract_slice(qk_scale, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop_unroll = al.extract_slice(qk_scale, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])

            qk_loop = qk_loop - m_ij_loop[:, None]
            qk_loop_unroll = qk_loop_unroll - m_ij_loop[:, None]

            p_loop = tl.math.exp(qk_loop)
            p_loop_unroll = tl.math.exp(qk_loop_unroll)
            
            p_loop_reshape = p_loop.reshape(BLOCK_N_UNROLL // 16, 1, 16)
            p_cast_loop = p_loop_reshape.to(cast_dtype)
            p_nz = al.insert_slice(p_nz, p_cast_loop, [0, loop, 0], [BLOCK_N_UNROLL//16, 1, 16], [1, 1, 1])
            
            p_loop_unroll_reshape = p_loop_unroll.reshape(BLOCK_N_UNROLL // 16, 1, 16)
            p_cast_loop_unroll = p_loop_unroll_reshape.to(cast_dtype)
            p_nz = al.insert_slice(p_nz, p_cast_loop_unroll, [BLOCK_N_UNROLL//16, loop, 0], [BLOCK_N_UNROLL//16, 1, 16], [1, 1, 1])
            
            row_sum = p_loop + p_loop_unroll
            l_ij_loop = tl.sum(row_sum, 1)
            l_ij = al.insert_slice(l_ij, l_ij_loop, [loop], [1], [1])

    with al.scope(vector_mode="simd", outline=True, no_inline=True):
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

    al.copy(bl.to_buffer(m_ij, al.ascend_address_space.UB), m_i_buffer)

    bl.to_buffer(l_i, bind_buffer=l_i_buffer)
    bl.to_buffer(p_nz, bind_buffer=p_nz_buffer)
    bl.to_buffer(alpha, bind_buffer=alpha_buffer)

@triton.jit
def softmax_with_mask_no_update_BN_128(qk, sm_scale, attn_mask_ptr,  m_i_buffer, l_i_buffer, p_nz_buffer, qk_scale, cast_dtype, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, STAGE: tl.constexpr):
    sub_vec_id = al.sub_vec_id()
    m_i = bl.to_tensor(m_i_buffer)
    l_i = bl.to_tensor(l_i_buffer)
    p_nz = bl.to_tensor(p_nz_buffer)

    l_ij_buffer = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
    l_ij = bl.to_tensor(l_ij_buffer)
    m_ij_buffer = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
    m_ij = bl.to_tensor(m_ij_buffer)

    atten_mask = tl.zeros((BLOCK_M // 2, BLOCK_N), dtype=tl.int8)
    if STAGE == 1:
        # curr_mask_ptr = tl.advance(attn_mask_ptr, ((sub_vec_id * BLOCK_M // 2).to(tl.int32), 0))
        # atten_mask = tl.load(attn_mask_ptr, boundary_check=(0, 1), padding_option="zero") # 64*128
        atten_mask = attn_mask_ptr

    BLOCK_N_UNROLL : tl.constexpr = BLOCK_N // 2
    with al.scope(vector_mode="simd", outline=True):
        for loop in range(BLOCK_M // 2):
            qk_loop = al.extract_slice(qk, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop = qk_loop * sm_scale
            if STAGE == 1:
                mask = al.extract_slice(atten_mask, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1]).to(tl.int1) # 64*128 -> 1*64
                qk_loop = qk_loop + tl.where(mask, -1.0e4, 0)
            qk_scale = al.insert_slice(qk_scale, qk_loop, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])

            qk_loop_unroll = al.extract_slice(qk, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop_unroll = qk_loop_unroll * sm_scale
            if STAGE == 1:
                mask_unroll = al.extract_slice(atten_mask, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1]).to(tl.int1)
                qk_loop_unroll = qk_loop_unroll + tl.where(mask_unroll, -1.0e4, 0)
            qk_scale = al.insert_slice(qk_scale, qk_loop_unroll, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])

            row_max = tl.maximum(qk_loop, qk_loop_unroll, propagate_nan=tl.PropagateNan.ALL)
            row_max_agg = tl.max(row_max, 1, propagate_nan=True)
            
            m_ij = al.insert_slice(m_ij, row_max_agg, [loop], [1], [1])

        al.debug_barrier(al.SYNC_IN_VF.VST_VLD)

        for loop in range(BLOCK_M // 2):
            m_ij_loop = al.extract_slice(m_ij, [loop], [1], [1])

            qk_loop = al.extract_slice(qk_scale, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop_unroll = al.extract_slice(qk_scale, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])

            qk_loop = qk_loop - m_ij_loop[:, None]
            qk_loop_unroll = qk_loop_unroll - m_ij_loop[:, None]

            p_loop = tl.math.exp(qk_loop)
            p_loop_unroll = tl.math.exp(qk_loop_unroll)
            
            p_loop_reshape = p_loop.reshape(BLOCK_N_UNROLL // 16, 1, 16)
            p_cast_loop = p_loop_reshape.to(cast_dtype)
            p_nz = al.insert_slice(p_nz, p_cast_loop, [0, loop, 0], [BLOCK_N_UNROLL//16, 1, 16], [1, 1, 1])
            
            p_loop_unroll_reshape = p_loop_unroll.reshape(BLOCK_N_UNROLL // 16, 1, 16)
            p_cast_loop_unroll = p_loop_unroll_reshape.to(cast_dtype)
            p_nz = al.insert_slice(p_nz, p_cast_loop_unroll, [BLOCK_N_UNROLL//16, loop, 0], [BLOCK_N_UNROLL//16, 1, 16], [1, 1, 1])
            
            row_sum = p_loop + p_loop_unroll
            l_ij_loop = tl.sum(row_sum, 1)
            l_ij = al.insert_slice(l_ij, l_ij_loop, [loop], [1], [1])

    al.copy(m_ij_buffer, m_i_buffer)
    al.copy(l_ij_buffer, l_i_buffer)

    bl.to_buffer(p_nz, bind_buffer=p_nz_buffer)


@triton.jit
def softmax_with_mask_with_update_BN_64(vtaskId, qk, sm_scale, attn_mask_ptr, m_i_buffer, l_i_buffer, alpha_buffer, p_nz_buffer, qk_scale, cast_dtype, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, STAGE: tl.constexpr):
    sub_vec_id = al.sub_vec_id()
    m_i = bl.to_tensor(m_i_buffer)
    l_i = bl.to_tensor(l_i_buffer)
    alpha = bl.to_tensor(alpha_buffer)
    p_nz = bl.to_tensor(p_nz_buffer)

    l_ij = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
    l_ij = bl.to_tensor(l_ij)
    tmp_max = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
    tmp_max = bl.to_tensor(tmp_max)

    atten_mask = tl.zeros((BLOCK_M // 2, BLOCK_N), dtype=tl.int8)

    if STAGE == 1:
        # curr_mask_ptr = tl.advance(attn_mask_ptr, ((sub_vec_id * BLOCK_M // 2).to(tl.int32), 0))
        # atten_mask = tl.load(attn_mask_ptr, boundary_check=(0, 1), padding_option="zero") # 64*128
        atten_mask = attn_mask_ptr

    BLOCK_N_UNROLL : tl.constexpr = BLOCK_N // 2
    with al.scope(vector_mode="simd", outline=True):

        for loop in range(BLOCK_M // 2):
            qk_loop = al.extract_slice(qk, [loop, 0], [1, BLOCK_N], [1, 1])
            qk_loop = qk_loop * sm_scale
            if STAGE == 1:
                mask = al.extract_slice(atten_mask, [loop, 0], [1, BLOCK_N], [1, 1]).to(tl.int1) # 64*128 -> 1*64
                qk_loop = qk_loop + tl.where(mask, -1.0e4, 0)
            qk_scale = al.insert_slice(qk_scale, qk_loop, [loop, 0], [1, BLOCK_N], [1, 1])
            row_max = tl.max(qk_loop, 1, propagate_nan=True)

            tmp_max = al.insert_slice(tmp_max, row_max, [loop], [1], [1])
        
        m_ij = tl.maximum(m_i, tmp_max, propagate_nan=tl.PropagateNan.ALL)
        al.debug_barrier(al.SYNC_IN_VF.VST_VLD)        
        for loop in range(BLOCK_M // 2):
            m_ij_loop = al.extract_slice(m_ij, [loop], [1], [1])

            qk_loop = al.extract_slice(qk_scale, [loop, 0], [1, BLOCK_N], [1, 1])

            qk_loop = qk_loop - m_ij_loop[:, None]

            p_loop = tl.math.exp(qk_loop)
            
            p_loop_reshape = p_loop.reshape(BLOCK_N // 16, 1, 16)
            p_cast_loop = p_loop_reshape.to(cast_dtype)
            p_nz = al.insert_slice(p_nz, p_cast_loop, [0, loop, 0], [BLOCK_N//16, 1, 16], [1, 1, 1])
            
            l_ij_loop = tl.sum(p_loop, 1)
            l_ij = al.insert_slice(l_ij, l_ij_loop, [loop], [1], [1])

    with al.scope(vector_mode="simd", outline=True, no_inline=True):
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

    al.copy(bl.to_buffer(m_ij, al.ascend_address_space.UB), m_i_buffer)

    bl.to_buffer(l_i, bind_buffer=l_i_buffer)
    bl.to_buffer(p_nz, bind_buffer=p_nz_buffer)
    bl.to_buffer(alpha, bind_buffer=alpha_buffer)
    
@triton.jit
def softmax_no_mask_no_update(qk, sm_scale,  m_i_buffer, l_i_buffer, p_nz_buffer, qk_scale, cast_dtype, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, STAGE: tl.constexpr):
    sub_vec_id = al.sub_vec_id()
    m_i = bl.to_tensor(m_i_buffer)
    l_i = bl.to_tensor(l_i_buffer)
    p_nz = bl.to_tensor(p_nz_buffer)

    l_ij_buffer = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
    l_ij = bl.to_tensor(l_ij_buffer)
    m_ij_buffer = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
    m_ij = bl.to_tensor(m_ij_buffer)

    BLOCK_N_UNROLL : tl.constexpr = BLOCK_N // 2
    with al.scope(vector_mode="simd", outline=True):
        for loop in range(BLOCK_M // 2):
            qk_loop = al.extract_slice(qk, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop = qk_loop * sm_scale
            qk_scale = al.insert_slice(qk_scale, qk_loop, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])

            qk_loop_unroll = al.extract_slice(qk, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop_unroll = qk_loop_unroll * sm_scale
            qk_scale = al.insert_slice(qk_scale, qk_loop_unroll, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])

            row_max = tl.maximum(qk_loop, qk_loop_unroll, propagate_nan=tl.PropagateNan.ALL)
            row_max_agg = tl.max(row_max, 1, propagate_nan=True)
            
            m_ij = al.insert_slice(m_ij, row_max_agg, [loop], [1], [1])

        al.debug_barrier(al.SYNC_IN_VF.VST_VLD)

        for loop in range(BLOCK_M // 2):
            m_ij_loop = al.extract_slice(m_ij, [loop], [1], [1])

            qk_loop = al.extract_slice(qk_scale, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])
            qk_loop_unroll = al.extract_slice(qk_scale, [loop, BLOCK_N_UNROLL], [1, BLOCK_N_UNROLL], [1, 1])

            qk_loop = qk_loop - m_ij_loop[:, None]
            qk_loop_unroll = qk_loop_unroll - m_ij_loop[:, None]

            p_loop = tl.math.exp(qk_loop)
            p_loop_unroll = tl.math.exp(qk_loop_unroll)
            
            p_loop_reshape = p_loop.reshape(BLOCK_N_UNROLL // 16, 1, 16)
            p_cast_loop = p_loop_reshape.to(cast_dtype)
            p_nz = al.insert_slice(p_nz, p_cast_loop, [0, loop, 0], [BLOCK_N_UNROLL//16, 1, 16], [1, 1, 1])
            
            p_loop_unroll_reshape = p_loop_unroll.reshape(BLOCK_N_UNROLL // 16, 1, 16)
            p_cast_loop_unroll = p_loop_unroll_reshape.to(cast_dtype)
            p_nz = al.insert_slice(p_nz, p_cast_loop_unroll, [BLOCK_N_UNROLL//16, loop, 0], [BLOCK_N_UNROLL//16, 1, 16], [1, 1, 1])
            
            row_sum = p_loop + p_loop_unroll
            l_ij_loop = tl.sum(row_sum, 1)
            l_ij = al.insert_slice(l_ij, l_ij_loop, [loop], [1], [1])

    al.copy(l_ij_buffer, l_i_buffer)
    al.copy(m_ij_buffer, m_i_buffer)

    bl.to_buffer(p_nz, bind_buffer=p_nz_buffer)

@triton.jit
def softmax_with_mask_no_update_BN_64(qk, sm_scale, attn_mask_ptr,  m_i_buffer, l_i_buffer, p_nz_buffer, qk_scale, cast_dtype, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, STAGE: tl.constexpr):
    sub_vec_id = al.sub_vec_id()
    m_i = bl.to_tensor(m_i_buffer)
    l_i = bl.to_tensor(l_i_buffer)
    p_nz = bl.to_tensor(p_nz_buffer)

    l_ij_buffer = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
    l_ij = bl.to_tensor(l_ij_buffer)
    m_ij_buffer = bl.alloc(tl.float32, [BLOCK_M // 2], al.ascend_address_space.UB)
    m_ij = bl.to_tensor(m_ij_buffer)

    atten_mask = tl.zeros((BLOCK_M // 2, BLOCK_N), dtype=tl.int8)
    if STAGE == 1:
        # curr_mask_ptr = tl.advance(attn_mask_ptr, ((sub_vec_id * BLOCK_M // 2).to(tl.int32), 0))
        # atten_mask = tl.load(attn_mask_ptr, boundary_check=(0, 1), padding_option="zero") # 64*128
        atten_mask = attn_mask_ptr

    BLOCK_N_UNROLL : tl.constexpr = BLOCK_N // 2
    with al.scope(vector_mode="simd", outline=True):
        for loop in range(BLOCK_M // 2):
            qk_loop = al.extract_slice(qk, [loop, 0], [1, BLOCK_N], [1, 1])
            qk_loop = qk_loop * sm_scale
            if STAGE == 1:
                mask = al.extract_slice(atten_mask, [loop, 0], [1, BLOCK_N], [1, 1]).to(tl.int1) # 64*128 -> 1*64
                qk_loop = qk_loop + tl.where(mask, -1.0e4, 0)
            qk_scale = al.insert_slice(qk_scale, qk_loop, [loop, 0], [1, BLOCK_N], [1, 1])
            row_max = tl.max(qk_loop, 1, propagate_nan=True)
            
            m_ij = al.insert_slice(m_ij, row_max, [loop], [1], [1])

        al.debug_barrier(al.SYNC_IN_VF.VST_VLD)

        for loop in range(BLOCK_M // 2):
            m_ij_loop = al.extract_slice(m_ij, [loop], [1], [1])

            qk_loop = al.extract_slice(qk_scale, [loop, 0], [1, BLOCK_N], [1, 1])

            qk_loop = qk_loop - m_ij_loop[:, None]

            p_loop = tl.math.exp(qk_loop)
            
            p_loop_reshape = p_loop.reshape(BLOCK_N // 16, 1, 16)
            p_cast_loop = p_loop_reshape.to(cast_dtype)
            p_nz = al.insert_slice(p_nz, p_cast_loop, [0, loop, 0], [BLOCK_N//16, 1, 16], [1, 1, 1])
            
            l_ij_loop = tl.sum(p_loop, 1)
            l_ij = al.insert_slice(l_ij, l_ij_loop, [loop], [1], [1])

    al.copy(m_ij_buffer, m_i_buffer)
    al.copy(l_ij_buffer, l_i_buffer)

    bl.to_buffer(p_nz, bind_buffer=p_nz_buffer)


@triton.jit
def softmax_vf_select(vtaskId, need_mask, need_update, qk, sm_scale, attn_mask_ptr, m_i_buffer, l_i_buffer, alpha_buffer, p_nz_buffer, qk_scale, cast_dtype, BLOCK_M, BLOCK_N, STAGE):
    if need_update:
        if BLOCK_N == 128:
            softmax_with_mask_with_update_BN_128(vtaskId, qk, sm_scale, attn_mask_ptr, m_i_buffer, l_i_buffer, alpha_buffer, p_nz_buffer, qk_scale, cast_dtype, BLOCK_M, BLOCK_N, STAGE)
        else:
            softmax_with_mask_with_update_BN_64(vtaskId, qk, sm_scale, attn_mask_ptr, m_i_buffer, l_i_buffer, alpha_buffer, p_nz_buffer, qk_scale, cast_dtype, BLOCK_M, BLOCK_N, STAGE)
    else:
        if BLOCK_N == 128:
            softmax_with_mask_no_update_BN_128(qk, sm_scale, attn_mask_ptr, m_i_buffer, l_i_buffer, p_nz_buffer, qk_scale, cast_dtype, BLOCK_M, BLOCK_N, STAGE)
        else:
            softmax_with_mask_no_update_BN_64(qk, sm_scale, attn_mask_ptr, m_i_buffer, l_i_buffer, p_nz_buffer, qk_scale, cast_dtype, BLOCK_M, BLOCK_N, STAGE)

@triton.jit
def process_v1(qk_ub_ping, qk_ub_pong, p_l1_ping, p_l1_pong, attn_mask_ptr,
                m_i_tb0, m_i_tb1, m_i_tb2, l_i_tb0, l_i_tb1, l_i_tb2, alpha_tb0, alpha_tb1, alpha_tb2,
                sm_scale, vtaskId, v_s1_task_mod3, cast_dtype,
                need_mask, need_update,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, STAGE: tl.constexpr
                ):
    al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
    sub_vec_id = al.sub_vec_id()

    p_nz = bl.alloc(cast_dtype, [BLOCK_N // 16, BLOCK_M // 32 * 16, 16], al.ascend_address_space.UB)
    al.multibuffer(p_nz, 2)

    qk_scale = bl.alloc(tl.float32, [BLOCK_M // 2, BLOCK_N], al.ascend_address_space.UB)
    qk_scale = bl.to_tensor(qk_scale)

    if (vtaskId & 1) == 1:
        qk = bl.to_tensor(qk_ub_ping)
    else:
        qk = bl.to_tensor(qk_ub_pong)

    if v_s1_task_mod3 == 0 and (vtaskId-1) % 3 == 0:
        softmax_vf_select(vtaskId, need_mask, need_update, qk, sm_scale, attn_mask_ptr, m_i_tb0, l_i_tb0, alpha_tb0, p_nz, qk_scale, cast_dtype, BLOCK_M, BLOCK_N, STAGE)
    elif v_s1_task_mod3 == 0 and (vtaskId-1) % 3 == 1:
        softmax_vf_select(vtaskId, need_mask, need_update, qk, sm_scale, attn_mask_ptr, m_i_tb0, l_i_tb0, alpha_tb1, p_nz, qk_scale, cast_dtype, BLOCK_M, BLOCK_N, STAGE)
    elif v_s1_task_mod3 == 0 and (vtaskId-1) % 3 == 2:
        softmax_vf_select(vtaskId, need_mask, need_update, qk, sm_scale, attn_mask_ptr, m_i_tb0, l_i_tb0, alpha_tb2, p_nz, qk_scale, cast_dtype, BLOCK_M, BLOCK_N, STAGE)
    elif v_s1_task_mod3 == 1 and (vtaskId-1) % 3 == 0:
        softmax_vf_select(vtaskId, need_mask, need_update, qk, sm_scale, attn_mask_ptr, m_i_tb1, l_i_tb1, alpha_tb0, p_nz, qk_scale, cast_dtype, BLOCK_M, BLOCK_N, STAGE)
    elif v_s1_task_mod3 == 1 and (vtaskId-1) % 3 == 1:
        softmax_vf_select(vtaskId, need_mask, need_update, qk, sm_scale, attn_mask_ptr, m_i_tb1, l_i_tb1, alpha_tb1, p_nz, qk_scale, cast_dtype, BLOCK_M, BLOCK_N, STAGE)
    elif v_s1_task_mod3 == 1 and (vtaskId-1) % 3 == 2:
        softmax_vf_select(vtaskId, need_mask, need_update, qk, sm_scale, attn_mask_ptr, m_i_tb1, l_i_tb1, alpha_tb2, p_nz, qk_scale, cast_dtype, BLOCK_M, BLOCK_N, STAGE)
    elif v_s1_task_mod3 == 2 and (vtaskId-1) % 3 == 0:
        softmax_vf_select(vtaskId, need_mask, need_update, qk, sm_scale, attn_mask_ptr, m_i_tb2, l_i_tb2, alpha_tb0, p_nz, qk_scale, cast_dtype, BLOCK_M, BLOCK_N, STAGE)
    elif v_s1_task_mod3 == 2 and (vtaskId-1) % 3 == 1:
        softmax_vf_select(vtaskId, need_mask, need_update, qk, sm_scale, attn_mask_ptr, m_i_tb2, l_i_tb2, alpha_tb1, p_nz, qk_scale, cast_dtype, BLOCK_M, BLOCK_N, STAGE)
    else:
        softmax_vf_select(vtaskId, need_mask, need_update, qk, sm_scale, attn_mask_ptr, m_i_tb2, l_i_tb2, alpha_tb2, p_nz, qk_scale, cast_dtype, BLOCK_M, BLOCK_N, STAGE)

    al.sync_block_set("vector", "cube", 2 ,al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

    al.sync_block_wait("cube", "vector", 6 ,al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)
    p_nz = bl.to_tensor(p_nz)
    if (vtaskId & 1) == 1:
        p_l1_ping_sub = bl.subview(p_l1_ping, [0, sub_vec_id * ((BLOCK_M // 2) // 16), 0, 0], [BLOCK_N // 16, (BLOCK_M // 2) // 16, 16, 16], [1, 1, 1, 1])
        al.copy_from_ub_to_l1(bl.to_buffer(p_nz.reshape(BLOCK_N // 16, BLOCK_M // 32, 16, 16), al.ascend_address_space.UB), p_l1_ping_sub)
    else:
        p_l1_pong_sub = bl.subview(p_l1_pong, [0, sub_vec_id * ((BLOCK_M // 2) // 16), 0, 0], [BLOCK_N // 16, (BLOCK_M // 2) // 16, 16, 16], [1, 1, 1, 1])
        al.copy_from_ub_to_l1(bl.to_buffer(p_nz.reshape(BLOCK_N // 16, BLOCK_M // 32, 16, 16), al.ascend_address_space.UB), p_l1_pong_sub)
    # tl.debug_barrier()
    al.sync_block_set("vector", "cube", 4,al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE1)
