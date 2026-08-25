import os

import math
import torch
import torch_npu
import pytest

import sys
from fa_forward_fp8 import *
# q_shape:  torch.Size([128, 8, 1024, 128])
# q_scale_shape:  torch.Size([128, 8, 8, 1])
# k_shape:  torch.Size([128, 8, 1024, 128])
# k_scale_shape:  torch.Size([128, 8, 8, 1])

# +++++++++++++++++++ zj reference ++++++++++++++++++++++++++++++++++
def dequant_qkv(
    qkv_fp8: torch.Tensor,
    scale: torch.Tensor,
    token_block: int = 128,
    head_block: int = 128,
) -> torch.Tensor:
    B, H, N_CTX, HEAD_DIM = qkv_fp8.shape
    scale_expanded = (
        scale.unsqueeze(-1).unsqueeze(-1)
        .expand(-1, -1, -1, token_block, -1, head_block)
        .reshape(B, H, N_CTX, HEAD_DIM)
    )
    return (qkv_fp8.float() * scale_expanded).to(torch.float32)


def _attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    sm_scale: float,
    num_kv_heads: int = None,
) -> torch.Tensor:
    """Reference BF16 attention，支持 GQA（num_kv_heads < q.shape[1]）"""
    num_q_heads = q.shape[1]
    if num_kv_heads is None:
        num_kv_heads = num_q_heads
    if num_kv_heads < num_q_heads:
        # GQA: expand K,V (B, H_kv, N, D) -> (B, H_q, N, D)，每 H_q/H_kv 个 Q head 共享同一 KV head
        num_groups = num_q_heads // num_kv_heads
        k = k.repeat_interleave(num_groups, dim=1)  # (B, H_q, N_CTX, HEAD_DIM)
        v = v.repeat_interleave(num_groups, dim=1)
    # q,k,v: (B, H, N_CTX, HEAD_DIM)
    qk = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    if causal:
        N_CTX = q.shape[2]
        mask = torch.triu(torch.ones(N_CTX, N_CTX, device=q.device, dtype=torch.bool), diagonal=1)
        qk = qk.masked_fill(mask, float("-inf"))
    p = torch.softmax(qk.float(), dim=-1).to(q.dtype)
    o = torch.matmul(p, v)
    return o


# ++++++++++++++++++++++Ascend reference+++++++++++++++++++++++++++++++++
def tsoftmax(x):
    """稳定的softmax实现"""
    x_max = torch.max(x, dim=-1, keepdim=True)[0]
    x_sub = x - x_max
    p = torch.exp(x_sub)
    p_cast = p.to(torch.float8_e4m3fn)
    p_cast = p_cast.to(torch.float32)
    x_sum = torch.sum(p, dim=-1, keepdim=True)
    return p_cast,x_sum


def block_dequant(res, dequant_matrix_s1, dequant_matrix_s2, block_size_s1=128, block_size_s2=128):
    B, Nkv, G, S1, S2 = res.shape
    result = torch.zeros_like(res)

    for b in range(B):
        for n in range(Nkv):
            for g in range(G):
                for i in range((S1 + block_size_s1 - 1) // block_size_s1):
                    start_s1 = i * block_size_s1
                    end_s1 = min(start_s1 + block_size_s1, S1)

                    for j in range((S2 + block_size_s2 - 1) // block_size_s2):
                        start_s2 = j * block_size_s2
                        end_s2 = min(start_s2 + block_size_s2, S2)

                        block = res[b, n, g, start_s1:end_s1, start_s2:end_s2]

                        s1_g_idx = 0 if dequant_matrix_s1.shape[2] == 1 else g
                        s2_g_idx = 0 if dequant_matrix_s2.shape[2] == 1 else g

                        dequant_value_s1 = dequant_matrix_s1[b, n, s1_g_idx, i, 0]
                        dequant_value_s2 = dequant_matrix_s2[b, n, s2_g_idx, j, 0]

                        result[b, n, g, start_s1:end_s1, start_s2:end_s2] = (
                            block * dequant_value_s1 * dequant_value_s2
                        )
    return result


def chunked_matmul_with_quant(left, right, scale, per_block_size=128):
    B, N2, G1, S1, S2 = left.shape
    _, _, G2, S2_r, D = right.shape
    G_out = max(G1, G2)

    result = torch.zeros((B, N2, G_out, S1, D), dtype=left.dtype, device=left.device)

    for b in range(B):
        for n in range(N2):
            for g in range(G_out):
                g_left  = 0 if G1 == 1 else g
                g_right = 0 if G2 == 1 else g

                for start in range(0, S2, per_block_size):
                    end = min(start + per_block_size, S2)
                    left_block  = left[b, n, g_left,  :, start:end]
                    right_block = right[b, n, g_right, start:end, :]
                    block_result = torch.matmul(left_block, right_block)

                    scale_g_idx = 0 if scale.shape[2] == 1 else g
                    scale_val = scale[b, n, scale_g_idx, start // per_block_size, 0]
                    result[b, n, g] += block_result * scale_val

    return result


def tforward_npu(q_quant, k_quant, v_quant, drop_mask, atten_mask, pse, scale,
                 keep_prob, is_skip_invalid_row, pse_type, dscale_q, dscale_k, dscale_v,
                 per_block_size_q=128, per_block_size_kv=128):
 
    device = q_quant.device
    dtype = torch.float32
 
    q_fp8 = q_quant.to(dtype)
    k_fp8 = k_quant.to(dtype)
    v_fp8 = v_quant.to(dtype)
 
    qkk = torch.matmul(q_fp8, k_fp8.transpose(-1, -2))
    qk = block_dequant(qkk, dscale_q, dscale_k,
                       block_size_s1=per_block_size_q,
                       block_size_s2=per_block_size_kv)
 
    if pse_type == 1:
        if pse is not None:
            qk = (qk + pse) * scale
        else:
            qk = qk * scale
    else:
        qk = qk * scale
        if pse is not None:
            qk = qk + pse
    if atten_mask is not None:
        mask_expanded = atten_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        mask_expanded = mask_expanded.expand(qk.shape)
        qk = torch.where(mask_expanded, torch.tensor(-1e4, device=device), qk)
 
    p_fp8, x_sum = tsoftmax(qk)
 
    pv_scale = chunked_matmul_with_quant(p_fp8, v_fp8, dscale_v,
                                         per_block_size=per_block_size_kv)
    y = pv_scale / x_sum
    print(f"torch pv dscale:pv_scale")
    return y


DEVICE="npu"

def block_quantize(tensor, data_type=torch.float8_e4m3fn, per_block_size=128):
    if data_type == torch.float8_e5m2:
        FP8_MAX = 57344.0
    elif data_type == torch.float8_e4m3fn:
        FP8_MAX = 448.0
    else:
        raise ValueError(f"{data_type} Not support block quant")
 
    B, N, S, D = tensor.shape
 
    reshaped_input = tensor.view(B, N, S // per_block_size, per_block_size, D) # (B, N, S, D) -> (B, N, S//128, 128, D)
    flattened_block = reshaped_input.flatten(start_dim=-2)  # (B, N, G, 128*D)
    max_val = torch.max(torch.abs(flattened_block), dim=-1).values
    scale_val = FP8_MAX / max_val.clamp(min=1e-12)
    scale_expanded = scale_val.view(B, N, -1, 1, 1) 
    scaled_data = reshaped_input * scale_expanded
    quantized_data = scaled_data.to(data_type)
    quantized_tensor = quantized_data.view(B, N, S, D)
    scale = scale_val.unsqueeze(-1)
    d_scale = 1 / scale
    return d_scale, quantized_tensor


@pytest.mark.parametrize("Z,H,N_CTX,HEAD_DIM,causal,dtype,BM,BN", [
        # ============================ fp8 cases ===================================
        # [128, 8, 8192, 128, False, torch.float32, 128,128],
        # [128, 8, 8192, 64, False, torch.float32, 128,128],
        [128, 8, 1024, 128, False, torch.float32, 128,256],
        [128, 8, 1024, 64, False, torch.float32, 128,256],
        # [128, 8, 8192, 128, True, torch.float32, 128,128],
        # [128, 8, 8192, 64, True, torch.float32, 128,128],
        [128, 8, 1024, 128, True, torch.float32, 128,128],
        [128, 8, 1024, 64, True, torch.float32, 256,128],
    ])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype,BM ,BN):
    torch.manual_seed(20)
    PER_BLOCK_SIZE_Q = BM
    PER_BLOCK_SIZE_KV = BN
    # assert PER_BLOCK_SIZE == BM and BM == BN
    q = torch.ones((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).requires_grad_()
    k = torch.ones((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).requires_grad_()
    v = torch.ones((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).requires_grad_()


    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    atten_mask = None
    if causal:
        atten_mask = torch.triu(torch.ones(N_CTX, N_CTX, device=DEVICE), diagonal=1).bool()

    dscale_q, q_quant = block_quantize(q, torch.float8_e4m3fn, per_block_size=PER_BLOCK_SIZE_Q)
    dscale_k, k_quant = block_quantize(k, torch.float8_e4m3fn, per_block_size=PER_BLOCK_SIZE_KV)
    dscale_v, v_quant = block_quantize(v, torch.float8_e4m3fn, per_block_size=PER_BLOCK_SIZE_KV)


    tri_out = attention(
        q_quant, k_quant, v_quant, dscale_q, dscale_k, dscale_v, atten_mask, causal, sm_scale, BM, BN, PER_BLOCK_SIZE_Q,
        PER_BLOCK_SIZE_KV
    )

    ref_out = tforward_npu(
        q_quant=q_quant.unsqueeze(2),
        k_quant=k_quant.unsqueeze(2),
        v_quant=v_quant.unsqueeze(2),
        drop_mask=None,
        atten_mask=atten_mask,
        pse=None,
        scale=sm_scale,
        keep_prob=None,
        is_skip_invalid_row=None,
        pse_type=None,
        dscale_q=dscale_q.unsqueeze(2),
        dscale_k=dscale_k.unsqueeze(2),
        dscale_v=dscale_v.unsqueeze(2),
        per_block_size_q=PER_BLOCK_SIZE_Q,
        per_block_size_kv=PER_BLOCK_SIZE_KV,
    )


    if ref_out.dim() == 5:
        ref_out = ref_out.squeeze(2)
    diff = (tri_out - ref_out).abs().max()
    print(diff)
    assert diff < 0.1
    print("compare success!")

