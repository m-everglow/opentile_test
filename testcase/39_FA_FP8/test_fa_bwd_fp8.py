"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""
import math
from pathlib import Path
import pytest
from collections import namedtuple

import torch
import torch_npu
import triton
import triton.language as tl

LAYOUT = "BNSD"
PER_BLOCK_SIZE_Q = 128
PER_BLOCK_SIZE_KV = 128
SEED = 42
DEVICE = 'npu'

import triton.runtime.driver as driver

device = torch.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
AICORE_NUM = properties["num_aicore"]

current_dir = Path(__file__).resolve().parent

CaseInput = namedtuple('CaseInput',
                       ['do', 'q', 'k', 'v', 'o', 'mask', 'l', 'd_scale_q', 'd_scale_k', 'd_scale_v',
                        'ds_scale', 'scale'])
CaseOutput = namedtuple('CaseOutput', ['dq', 'dk', 'dv'])

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def fag_fp8_reference(
    do, q, k, v, o, lse,
    d_scale_q, d_scale_k, d_scale_v,
    ds_scale,
    sm_scale, CAUSAL, HEAD_DIM, BLOCK_M, BLOCK_N,
    BLOCK_SIZE_Q=PER_BLOCK_SIZE_Q,   # 128
    BLOCK_SIZE_KV=PER_BLOCK_SIZE_KV, # 256
):
    """
    PyTorch 实现

    Inputs :
        do        : [B, H, N_CTX, D]  float16
        q         : [B, H, N_CTX, D]  FP8（block 量化，未 dequant）
        k         : [B, H, N_CTX, D]  FP8（block 量化，未 dequant）
        v         : [B, H, N_CTX, D]  FP8（block 量化，未 dequant）
        o         : [B, H, N_CTX, D]  float16（forward 输出）
        lse       : [B, H, N_CTX]     float32（log-sum-exp）
        d_scale_q : [B, H, N_CTX//BLOCK_SIZE_Q,  1]  dequant scale
        d_scale_k : [B, H, N_CTX//BLOCK_SIZE_KV, 1]  dequant scale
        d_scale_v : [B, H, N_CTX//BLOCK_SIZE_KV, 1]  dequant scale
    """
    BATCH, N_HEAD, N_CTX, _ = q.shape

    # ── Step 1: ─────────────────────────────────────
    # delta[b,h,m] = sum_d( O[b,h,m,d] * dO[b,h,m,d] )
    delta = (o.to(torch.float32) * do.to(torch.float32)).sum(dim=-1)  # [B, H, N_CTX]
    lse_f32 = lse.to(torch.float32) # [B, H, N_CTX]

    # ── Step 2: 提前 dequant v─────────
    # v_block[b,h,i] *= d_scale_v[b,h,i]  （每 BLOCK_N 行共享一个 scale）
    v_f32 = (
        v.to(torch.float32)
         .reshape(BATCH, N_HEAD, N_CTX // BLOCK_N, BLOCK_N * HEAD_DIM)
        * d_scale_v
    ).reshape(BATCH, N_HEAD, N_CTX, HEAD_DIM) # [B, H, N_CTX, D] float32

    # ── Step 3: FP8 → float32 ────────
    q_f32 = q.to(torch.float32)   # [B, H, N_CTX, D]
    k_f32 = k.to(torch.float32)   # [B, H, N_CTX, D]
    do_f32 = do.to(torch.float32) # [B, H, N_CTX, D]

    # ── Step 4:  ───────────────────────────────────────
    # squeeze 最后一维 → [B, H, N//BLOCK_SIZE_*]
    dscale_q = d_scale_q.squeeze(-1).to(torch.float32)  # [B, H, N_CTX//BLOCK_SIZE_Q]
    dscale_k = d_scale_k.squeeze(-1).to(torch.float32)  # [B, H, N_CTX//BLOCK_SIZE_KV]

    # ── Step 5: 主循环 ────────────────────────────────────────
    dq = torch.zeros(BATCH, N_HEAD, N_CTX, HEAD_DIM, dtype=torch.float32).npu()
    dk = torch.zeros(BATCH, N_HEAD, N_CTX, HEAD_DIM, dtype=torch.float32).npu()
    dv = torch.zeros(BATCH, N_HEAD, N_CTX, HEAD_DIM, dtype=torch.float32).npu()

    num_n_blocks = N_CTX // BLOCK_N   # 外层：遍历 K/V 列块
    num_m_blocks = N_CTX // BLOCK_M   # 内层：遍历 Q 行块

    for b in range(BATCH):
        for h in range(N_HEAD):

            # ── 外层：N 块 ──
            for ni in range(num_n_blocks):
                sn = ni * BLOCK_N
                en = sn + BLOCK_N

                k_blk = k_f32[b, h, sn:en, :]   # [BLOCK_N, D]  FP8 raw
                v_blk = v_f32[b, h, sn:en, :]   # [BLOCK_N, D]  已 dequant

                # d_scale_k 整个 N 块共用同一标量
                ds_k = dscale_k[b, h, sn // BLOCK_SIZE_KV]  # 标量

                dv_acc = torch.zeros(BLOCK_N, HEAD_DIM, dtype=torch.float32).npu()
                dk_acc = torch.zeros(BLOCK_N, HEAD_DIM, dtype=torch.float32).npu()

                # ── 内层：M 块 ──
                for mi in range(num_m_blocks):
                    sm = mi * BLOCK_M
                    em = sm + BLOCK_M

                    # d_scale_q 整个 M 块共用同一标量
                    ds_q = dscale_q[b, h, sm // BLOCK_SIZE_Q]  # 标量

                    q_blk  = q_f32 [b, h, sm:em, :]   # [BLOCK_M, D]  FP8 raw
                    do_blk = do_f32[b, h, sm:em, :]   # [BLOCK_M, D]
                    m_vec  = lse_f32[b, h, sm:em]     # [BLOCK_M]
                    d_vec  = delta  [b, h, sm:em]     # [BLOCK_M]

                    # ── compute_s: FP8 raw matmul ──
                    s = q_blk @ k_blk.T # [BLOCK_M, BLOCK_N]

                    # ── calculate_quantitative_s: dequant ──
                    s = s * ds_q * ds_k

                    # ── compute_p: softmax 分子 ──
                    # p = exp(s * sm_scale - lse)
                    p = torch.exp(s * sm_scale - m_vec[:, None]) # [BLOCK_M, BLOCK_N]

                    # ── compute_dv: dV += P^T @ dO ──
                    dv_acc += p.T @ do_blk # [BLOCK_N, D]

                    # ── compute_dp: dP = dO @ V^T ──
                    dp = do_blk @ v_blk.T # [BLOCK_M, BLOCK_N]

                    # ── compute_ds: dS = P * (dP - delta) * sm_scale ────────
                    ds = p * (dp - d_vec[:, None]) * sm_scale  # [BLOCK_M, BLOCK_N]

                    # ── 使用外部传入的 ds_scale 作为缩放因子 ──
                    ds_scale_scalar = ds_scale.item()
                    ds_fp8 = (ds * ds_scale_scalar).to(q.dtype).to(torch.float32)

                    # ── dQ += ds_fp8 @ K * (ds_k / ds_scale_scalar) ──
                    dq[b, h, sm:em, :] += (ds_fp8 @ k_blk) * (ds_k / ds_scale_scalar)

                    # ── dK += ds_fp8^T @ Q * (ds_q / ds_scale_scalar) ──
                    dk_acc += (ds_fp8.T @ q_blk) * (ds_q / ds_scale_scalar)

                dk[b, h, sn:en, :] = dk_acc
                dv[b, h, sn:en, :] = dv_acc

    return dq.to(torch.float16), dk.to(torch.float16), dv.to(torch.float16)
    

def block_quantize(tensor, block_size, data_type=torch.float8_e4m3fn):
    if data_type == torch.float8_e5m2:
        FP8_MAX = 57344.0
    elif data_type == torch.float8_e4m3fn:
        FP8_MAX = 448.0
    else:
        raise ValueError(f"{data_type} Not support block quant")

    B, N, S, D = tensor.shape

    reshaped_input = tensor.view(B, N, S // block_size, block_size, D) # # [B, N, S // block_size, block_size, D]
    flattened_block = reshaped_input.flatten(start_dim=-2)  # [B, N, S // block_size, block_size * D]
    max_val = torch.max(torch.abs(flattened_block), dim=-1).values # [B, N, S // block_size]
    scale_val = FP8_MAX / max_val.clamp(min=1e-12) # [B, N, S // block_size]
    scale_expanded = scale_val.view(B, N, -1, 1, 1) # [B, N, S // block_size, 1, 1]
    scaled_data = reshaped_input * scale_expanded # [B, N, S // block_size, block_size, D]
    quantized_data = scaled_data.to(data_type) # [B, N, S // block_size, block_size, D], torch.float8_e4m3fn
    quantized_tensor = quantized_data.view(B, N, S, D) # [B, N, S, D], torch.float8_e4m3fn
    scale = scale_val.unsqueeze(-1) # [B, N, S // block_size, 1], torch.float32
    d_scale = 1 / scale # [B, N, S // block_size, 1], torch.float32 
    return d_scale, quantized_tensor


def block_quantize_forward(tensor, data_type=torch.float8_e4m3fn, per_block_size=128):
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


def tforward_npu(q_quant, k_quant, v_quant, drop_mask, atten_mask, pse, scale,
                 keep_prob, is_skip_invalid_row, pse_type, dscale_q, dscale_k, dscale_v):
    """
    分块 Flash-Attention forward，避免实例化完整 [B,N,G,S1,S1] 注意力矩阵导致 OOM。

    输入形状:
        q_quant / k_quant / v_quant : [B, N2, G, S, D]  (FP8)
        dscale_q / dscale_k / dscale_v : [B, N2, G, S//128, 1]
        atten_mask : [S, S] bool，True 表示需要被 mask 掉的位置

    输出:
        y         : [B, N2, G, S, D]  float32
        x_max_out : [B, N2, G, S, 1]  每行的全局最大值（log-sum-exp 用）
        x_sum_out : [B, N2, G, S, 1]  归一化分母（已乘 exp(max)）
    """
    device = q_quant.device
    dtype = torch.float32

    B, N2, G, S, D = q_quant.shape

    BLOCK_Q  = 128
    BLOCK_KV = 128

    y_out       = torch.zeros((B, N2, G, S, D),  dtype=dtype, device=device)
    x_max_out   = torch.full ((B, N2, G, S, 1), float('-inf'), dtype=dtype, device=device)
    x_sum_out   = torch.zeros((B, N2, G, S, 1), dtype=dtype, device=device)

    def dequant_q_block(q_raw, b, n, g, q_start):
        q_fp = q_raw.to(dtype)
        # dscale_q shape: [B, N2, G, S//128, 1]
        block_idx = q_start // BLOCK_KV
        s_g = 0 if dscale_q.shape[2] == 1 else g
        dq = dscale_q[b, n, s_g, block_idx, 0]
        return q_fp * dq

    def dequant_k_block(k_raw, b, n, g, kv_start):
        k_fp = k_raw.to(dtype)
        block_idx = kv_start // BLOCK_KV
        s_g = 0 if dscale_k.shape[2] == 1 else g
        dk = dscale_k[b, n, s_g, block_idx, 0]
        return k_fp * dk

    def dequant_v_block(v_raw, b, n, g, kv_start):
        v_fp = v_raw.to(dtype)
        block_idx = kv_start // BLOCK_KV
        s_g = 0 if dscale_v.shape[2] == 1 else g
        dv = dscale_v[b, n, s_g, block_idx, 0]
        return v_fp * dv

    for q_start in range(0, S, BLOCK_Q):
        q_end    = min(q_start + BLOCK_Q, S)
        chunk_q  = q_end - q_start

        # shape: [B, N2, G, chunk_q, D] / [B, N2, G, chunk_q, 1]
        acc      = torch.zeros((B, N2, G, chunk_q, D),  dtype=dtype, device=device)
        m_i      = torch.full ((B, N2, G, chunk_q, 1), float('-inf'), dtype=dtype, device=device)
        l_i      = torch.zeros((B, N2, G, chunk_q, 1), dtype=dtype, device=device)

        q_tile_raw = q_quant[:, :, :, q_start:q_end, :]  # [B, N2, G, chunk_q, D]

        for kv_start in range(0, S, BLOCK_KV):
            kv_end   = min(kv_start + BLOCK_KV, S)

            k_tile_raw = k_quant[:, :, :, kv_start:kv_end, :]  # [B, N2, G, chunk_kv, D]
            v_tile_raw = v_quant[:, :, :, kv_start:kv_end, :]

            # dscale_q: [B, N2, G, S//128, 1]
            q_block_idx = q_start // BLOCK_KV
            k_block_idx = kv_start // BLOCK_KV

            dq_scale = dscale_q[:, :, :, q_block_idx : q_block_idx + 1, :]   # [B,N2,G,1,1]
            dk_scale = dscale_k[:, :, :, k_block_idx : k_block_idx + 1, :]
            dv_scale = dscale_v[:, :, :, k_block_idx : k_block_idx + 1, :]

            q_fp  = q_tile_raw.to(dtype) * dq_scale   # [B, N2, G, chunk_q,  D]
            k_fp  = k_tile_raw.to(dtype) * dk_scale   # [B, N2, G, chunk_kv, D]
            v_fp  = v_tile_raw.to(dtype) * dv_scale   # [B, N2, G, chunk_kv, D]

            # --- QK^T ---  [B, N2, G, chunk_q, chunk_kv]
            qk = torch.matmul(q_fp, k_fp.transpose(-1, -2))

            # --- PSE & scale ---
            if pse_type == 1:
                if pse is not None:
                    pse_tile = pse[..., q_start:q_end, kv_start:kv_end]
                    qk = (qk + pse_tile) * scale
                else:
                    qk = qk * scale
            else:
                qk = qk * scale
                if pse is not None:
                    pse_tile = pse[..., q_start:q_end, kv_start:kv_end]
                    qk = qk + pse_tile

            # --- Attention mask ---
            if atten_mask is not None:
                # atten_mask: [S, S]
                mask_tile = atten_mask[q_start:q_end, kv_start:kv_end] # [chunk_q, chunk_kv]
                mask_tile = mask_tile.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(qk)
                qk = torch.where(mask_tile, torch.tensor(-1e4, dtype=dtype, device=device), qk)

            # m_new = max(m_i, rowmax(qk))
            m_new = torch.maximum(m_i, qk.max(dim=-1, keepdim=True).values)  # [B,N2,G,chunk_q,1]

            alpha  = torch.exp(m_i - m_new)   # [B,N2,G,chunk_q,1]
            acc    = acc * alpha         # rescale

            p_tile = torch.exp(qk - m_new) # [B,N2,G,chunk_q,chunk_kv]

            p_tile_fp8 = p_tile.to(torch.float8_e4m3fn).to(dtype)

            # 累积 PV
            acc  = acc + torch.matmul(p_tile_fp8, v_fp) # [B,N2,G,chunk_q,D]
            l_i  = l_i * alpha + p_tile.sum(dim=-1, keepdim=True)
            m_i  = m_new

        y_out    [:, :, :, q_start:q_end, :] = acc / l_i
        x_max_out[:, :, :, q_start:q_end, :] = m_i
        x_sum_out[:, :, :, q_start:q_end, :] = l_i

    return y_out, x_max_out, x_sum_out


def gen(Z, H, N_CTX, HEAD_DIM, CAUSAL, dtype):
    torch.manual_seed(SEED)
    device = DEVICE

    do = torch.empty(Z, H, N_CTX, HEAD_DIM, dtype=torch.float32, device=device).normal_(mean=0.0, std=0.5)
    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=torch.float32, device=device).normal_(mean=0.0, std=0.5)
    k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=torch.float32, device=device).normal_(mean=0.0, std=0.5)
    v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=torch.float32, device=device).normal_(mean=0.0, std=0.5)
    q_forward = q
    k_forward = k
    v_forward = v
    atten_mask = None
    sparse_mode = 0
    d_scale_q = None
    d_scale_k = None
    d_scale_v = None
    if CAUSAL:
        atten_mask = torch.triu(torch.ones(N_CTX, N_CTX, device=device, dtype=torch.bool), diagonal=1).to(DEVICE)
        sparse_mode = 2
    sm_scale = 1 / math.sqrt(HEAD_DIM)

    # d_scale_do, do = block_quantize(do, dtype)
    d_scale_q, q = block_quantize(q, PER_BLOCK_SIZE_Q, dtype)
    d_scale_k, k = block_quantize(k, PER_BLOCK_SIZE_KV, dtype)
    d_scale_v, v = block_quantize(v, PER_BLOCK_SIZE_KV, dtype)
    d_scale_q = d_scale_q.to(DEVICE)
    d_scale_k = d_scale_k.to(DEVICE)
    d_scale_v = d_scale_v.to(DEVICE)
    do = do.to(torch.float16)
    PER_BLOCK_SIZE=128
    dscale_q, q_quant = block_quantize_forward(q_forward, torch.float8_e4m3fn, per_block_size=PER_BLOCK_SIZE)
    dscale_k, k_quant = block_quantize_forward(k_forward, torch.float8_e4m3fn, per_block_size=PER_BLOCK_SIZE)
    dscale_v, v_quant = block_quantize_forward(v_forward, torch.float8_e4m3fn, per_block_size=PER_BLOCK_SIZE)
    out, softmax_max, softmax_sum = tforward_npu(
            q_quant=q_quant.unsqueeze(2),  # add group dimension to adapt torch fa
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
            dscale_v=dscale_v.unsqueeze(2)
        )
    out = out.squeeze(2)
    softmax_max = softmax_max.squeeze(2)
    softmax_sum = softmax_sum.squeeze(2)
    l = softmax_max[..., 0] + torch.log(softmax_sum[..., 0])
    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)
    '''
    out, softmax_max, softmax_sum, _, seed, offset, numels = torch_npu.npu_quant_fusion_attention(
        q, k, v, H,
        LAYOUT,
        atten_mask=torch.triu(
            torch.ones(2048,
                       2048,
                       device=DEVICE),
            diagonal=1).bool() if CAUSAL else None,
        d_scale_q=d_scale_q,
        d_scale_k=d_scale_k,
        d_scale_v=d_scale_v,
        scale=sm_scale,
        sparse_mode=sparse_mode)
    #import pdb;pdb.set_trace()
    l = softmax_max[..., 0] + torch.log(softmax_sum[..., 0])
    dq, dk, dv, _, _, _, _ = torch_npu.npu_quant_fusion_attention_grad(
        q, k, v, do, H, LAYOUT,
        atten_mask=torch.triu(
            torch.ones(2048, 2048, device=DEVICE),
            diagonal=1).bool() if CAUSAL else None,
        d_scale_q=d_scale_q,
        d_scale_k=d_scale_k,
        d_scale_v=d_scale_v,
        softmax_max=softmax_max,
        softmax_sum=softmax_sum, attention_in=out,
        scale_value=sm_scale,
        sparse_mode=sparse_mode, seed=seed,
        offset=offset, numels=numels)
    '''
    dq = None
    dk = None
    dv = None

    # FP8_MAX = 448.0 if dtype == torch.float8_e4m3fn else 57344.0
    # ds_scale = torch.tensor([FP8_MAX], dtype=torch.float32, device=DEVICE)
    ds_scale = torch.tensor([3.5], dtype=torch.float32, device=DEVICE)

    return (CaseInput(do, q, k, v, out, atten_mask, l, d_scale_q, d_scale_k, d_scale_v, ds_scale, sm_scale),
            CaseOutput(dq, dk, dv))


@triton.jit
def _attn_backward_preprocess(O, DO,
                         Delta,
                         Z, H, N_CTX,
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
                         NUM_CORES: tl.constexpr
                         ):
    pid = tl.program_id(0)

    # task 维度展开 (bhid = Z*H) × (m_block = N_CTX/BLOCK_M)
    NUM_MBLOCKS: tl.constexpr = N_CTX // BLOCK_M
    NUM_TASKS: tl.constexpr = (Z * H) * NUM_MBLOCKS # row: Z*H, col: NUM_MBLOCKS

    off_n = tl.arange(0, HEAD_DIM)

    for task in range(pid, NUM_TASKS, NUM_CORES):
        off_hz = task // NUM_MBLOCKS          # 0 .. Z*H-1, #row
        m_blk  = task - off_hz * NUM_MBLOCKS  # task % NUM_MBLOCKS, #col

        off_m = m_blk * BLOCK_M + tl.arange(0, BLOCK_M) # every col has BLOCK_M length

        o  = tl.load(O  + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]) #[BLOCK_M, BLOCK_DIM]
        do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32) #[BLOCK_M, BLOCK_DIM]

        delta = tl.sum(o * do, axis=1) # [BLOCK_M]

        tl.store(Delta + off_hz * N_CTX + off_m, delta) #[BLOCK_M]


@triton.jit
def compute_s(
    q, k,
):
    kT = tl.trans(k)
    qk = tl.dot(q, kT) # [BLOCK_M, BLOCK_N]
    return qk
    

@triton.jit
def calculate_quantitative_s(
    s, 
    d_scale_q_scalar, d_scale_k_scalar,
):
    s = s * d_scale_q_scalar * d_scale_k_scalar
    return s


@triton.jit
def compute_p(
    M, s_quant, sm_scale,
    BLOCK_M,
    start_m,
):
    offs_m = start_m + tl.arange(0, BLOCK_M)
    m = tl.load(M + offs_m).to(tl.float32)
    p = tl.math.exp(s_quant * sm_scale - m[:, None])
    return p


@triton.jit
def compute_dv(
    DO,
    p, 
    HEAD_DIM,
    BLOCK_M,
    start_m,
    stride_tok: tl.constexpr, 
    stride_d: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    p = p.to(tl.float16)
    dv = tl.dot(tl.trans(p), do)
    return dv


@triton.jit
def compute_dp(
    DO,
    v,
    HEAD_DIM,
    BLOCK_M,
    start_m,
    stride_tok: tl.constexpr,
    stride_d: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dp = tl.dot(do, tl.trans(v))
    return dp


@triton.jit
def compute_ds(
    p, dp, D,
    sm_scale,
    BLOCK_M,
    start_m
):
    offs_m = start_m + tl.arange(0, BLOCK_M)
    Di = tl.load(D + offs_m)
    ds = p * (dp - Di[:, None]) * sm_scale
    return ds


@triton.jit
def fag_fp8_kernel(Q_origin, K_origin, V_origin, DO_origin, DQ_origin,
              DK_origin, DV_origin, M_origin, D_origin,
              sm_scale,
              D_SCALE_Q_origin, D_SCALE_K_origin, D_SCALE_V_origin,
              DS_SCALE_origin,
              stride_z: tl.constexpr, stride_h: tl.constexpr, 
              stride_tok: tl.constexpr, stride_d: tl.constexpr,
              Z: tl.constexpr, H: tl.constexpr, N_CTX: tl.constexpr,
              BLOCK_M1: tl.constexpr,
              BLOCK_N1: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              BLOCK_SIZE_Q: tl.constexpr,
              BLOCK_SIZE_KV: tl.constexpr,
              NUM_CORES: tl.constexpr):
    
    pid_core = tl.program_id(0)
    
    NUM_BLOCKS: tl.constexpr = N_CTX // BLOCK_N1
    NUM_BH: tl.constexpr = Z * H
    NUM_TASKS: tl.constexpr = NUM_BH * NUM_BLOCKS
    
    offs_k = tl.arange(0, HEAD_DIM)
    for task in range(pid_core, NUM_TASKS, NUM_CORES):
        bhid = task // NUM_BLOCKS
        pid = task - bhid * NUM_BLOCKS
        
        off_chz = (bhid * N_CTX).to(tl.int64)
        adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
        
        # Offset pointers for batch/head
        Q = Q_origin + adj
        K = K_origin + adj
        V = V_origin + adj
        DO = DO_origin + adj
        DQ = DQ_origin + adj
        DK = DK_origin + adj
        DV = DV_origin + adj
        M = M_origin + off_chz
        D = D_origin + off_chz
        
        # 每个 (batch, head) 对应 N_CTX//BLOCK_SIZE_Q 个 d_scale_q 条目
        off_chz_q = (bhid * N_CTX // BLOCK_SIZE_Q).to(tl.int64)
        off_chz_kv = (bhid * N_CTX // BLOCK_SIZE_KV).to(tl.int64)
        D_SCALE_Q = D_SCALE_Q_origin + off_chz_q
        D_SCALE_K = D_SCALE_K_origin + off_chz_kv
        D_SCALE_V = D_SCALE_V_origin + off_chz_kv

        ds_scale_scalar = tl.load(DS_SCALE_origin)

        # 当前块的位置
        block_n = pid  # 当前处理的 N 块索引
        start_n = block_n * BLOCK_N1
        offs_n = start_n + tl.arange(0, BLOCK_N1)
        
        # 加载当前 N 块的 K 和 V (这些在整个计算中保持不变)
        k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
        v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
        
        # 初始化 dK 和 dV 累加器
        dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
        # 需要遍历所有的 M 块
        num_blocks = N_CTX // BLOCK_M1

        # 提前加载当前 N 块对应的 k dequant scale（整块共用一个 scale）
        # BLOCK_N1 == BLOCK_SIZE_KV，start_n // BLOCK_SIZE_KV 即唯一下标
        d_scale_k_scalar = tl.load(D_SCALE_K + start_n // BLOCK_SIZE_KV)  # 标量
        for m_idx in range(num_blocks):
            curr_m = m_idx * BLOCK_M1
            offs_m = curr_m + tl.arange(0, BLOCK_M1)

            d_scale_q_scalar = tl.load(D_SCALE_Q + curr_m // BLOCK_SIZE_Q)  # 标量
            q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

            s = compute_s(
                q, k
            )

            s_quant = calculate_quantitative_s(
                s,
                d_scale_q_scalar, d_scale_k_scalar,
            )

            p = compute_p(M, s_quant, sm_scale, BLOCK_M1, curr_m)

            dv += compute_dv(DO, p, HEAD_DIM, BLOCK_M1, curr_m, stride_tok, stride_d)

            dp = compute_dp(DO, v, HEAD_DIM, BLOCK_M1, curr_m, stride_tok, stride_d)

            ds = compute_ds(p, dp, D, sm_scale, BLOCK_M1, curr_m)

            ds = ds * ds_scale_scalar
            ds = ds.to(Q_origin.dtype.element_ty)

            dq = tl.dot(ds, k) * (d_scale_k_scalar / ds_scale_scalar)
            tl.extra.cann.extension.compile_hint(dq, "enable_fast_tf32_mul")

            dk_out = tl.dot(tl.trans(ds), q) * (d_scale_q_scalar / ds_scale_scalar)
            dk += dk_out

            dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
            tl.atomic_add(dq_ptrs, dq)
        
        # 写回 dK 和 dV
        dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
        tl.store(dk_ptrs, dk)
        
        dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
        tl.store(dv_ptrs, dv)


def fag_fp8(do, q, k, v, atten_mask, o, lse, d_scale_q, d_scale_k, d_scale_v,
            ds_scale, sm_scale, CAUSAL, HEAD_DIM, BLOCK_M, BLOCK_N):
    assert o.dtype in (torch.float16, torch.float32), \
        "preprocess kernel expects float16/float32 O"
    assert do.is_contiguous()
    assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
    BATCH, N_HEAD, N_CTX = q.shape[:3]
    PRE_BLOCK = 64
    num_cores = AICORE_NUM
    arg_k = k

    assert N_CTX % PRE_BLOCK == 0
    pre_grid = (num_cores,)
    delta = torch.empty_like(lse)

    _attn_backward_preprocess[pre_grid](
        o, do, delta,
        BATCH, N_HEAD, N_CTX,
        BLOCK_M=PRE_BLOCK, HEAD_DIM=HEAD_DIM,
        NUM_CORES=num_cores,
    )

    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.empty_like(k, dtype=torch.float32, device='cpu').to(q.device)
    dv = torch.empty_like(v, dtype=torch.float32, device='cpu').to(q.device)

    v = v.to(torch.float32).reshape(BATCH, N_HEAD, N_CTX // BLOCK_N, BLOCK_N * HEAD_DIM) * d_scale_v
    v = v.reshape(BATCH, N_HEAD, N_CTX, HEAD_DIM).to(do.dtype)

    grid = (num_cores,)
    fag_fp8_kernel[grid](
        q, arg_k, v, do, dq, dk, dv,
        lse, delta, sm_scale,
        d_scale_q, d_scale_k, d_scale_v,
        ds_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        BATCH, N_HEAD, N_CTX,
        BLOCK_M1=BLOCK_M,
        BLOCK_N1=BLOCK_N,
        HEAD_DIM=HEAD_DIM,
        BLOCK_SIZE_Q=PER_BLOCK_SIZE_Q,
        BLOCK_SIZE_KV=PER_BLOCK_SIZE_KV,
        NUM_CORES=num_cores,
        enable_auto_bind_sub_block=True,
        limit_auto_multi_buffer_of_local_buffer="no-l0c",
    )
    dq = dq.to(torch.float16)
    dk = dk.to(torch.float16)
    dv = dv.to(torch.float16)
    return dq, dk, dv


GOLDEN_MAP = {
    (128, torch.float8_e4m3fn): "2",
    (64, torch.float8_e4m3fn): "3",
    (128, torch.float8_e5m2): "6",
    (64, torch.float8_e5m2): "7",
}


@pytest.mark.parametrize(
    "Z, H, N_CTX, HEAD_DIM, causal, dtype, BM, BN",
    [
        #(128, 8, 8192, 128, False, torch.float8_e4m3fn, 32, 128),
        #(128, 8, 8192, 64, False, torch.float8_e4m3fn, 32, 128),
        (128, 8, 1024, 128, False, torch.float8_e4m3fn, 64, 128),
        (128, 8, 1024, 64, False, torch.float8_e4m3fn, 64, 128),
        #(128, 8, 8192, 128, False, torch.float8_e5m2, 32, 128),
        #(128, 8, 8192, 64, False, torch.float8_e5m2, 32, 128),
        (128, 8, 1024, 128, False, torch.float8_e5m2, 64, 128),
        (128, 8, 1024, 64, False, torch.float8_e5m2, 64, 128),
    ],
)
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype, BM, BN):
    # Filter out non-integer cases; N_CTX must be divisible by BM and BN, and HEAD_DIM must be divisible by 16.
    if N_CTX % BM != 0 or N_CTX % BN != 0 or HEAD_DIM % 16 != 0:
        pytest.skip("Skipping non-divisible case")
    
    # FP8 input and gloden
    case_input, case_output = gen(Z, H, N_CTX, HEAD_DIM, causal, dtype)
    # run flash attention backward
    dq, dk, dv = fag_fp8(case_input.do, case_input.q, case_input.k, case_input.v, case_input.mask, case_input.o,
                         case_input.l, case_input.d_scale_q, case_input.d_scale_k, case_input.d_scale_v,
                         case_input.ds_scale, case_input.scale, causal, HEAD_DIM, BM, BN)
    file_id = GOLDEN_MAP.get((HEAD_DIM, dtype))
    use_reference = True
    if file_id:
        file_names = [f"{name}_golden_torch_{file_id}.pt" for name in ["dq", "dk", "dv"]]
        file_paths = [current_dir / fname for fname in file_names]
        if all(p.exists() for p in file_paths):
            for name, val, path in zip(["dq", "dk", "dv"], [dq, dk, dv], file_paths):
                # file_path = os.path.join(current_dir, f"{name}_golden_torch_{file_id}.pt")
                golden = torch.load(path, map_location="cpu")
                val = val.cpu()
                torch.testing.assert_close(val, golden, atol=1e-2, rtol=1e-2, equal_nan=True)
            use_reference = False
        else:
            print(f"Golden files for ID {file_id} missing, falling back to reference function.")
 
    if use_reference:
        print("Running fag_fp8_reference to calculate Golden...")
        dq_ref, dk_ref, dv_ref = fag_fp8_reference( 
            case_input.do, case_input.q, case_input.k, case_input.v, 
            case_input.o,  case_input.l, 
            case_input.d_scale_q, case_input.d_scale_k, case_input.d_scale_v, 
            case_input.ds_scale,
            case_input.scale, causal, HEAD_DIM, BM, BN, 
        ) 

        # 比对计算结果
        torch.testing.assert_close(dv, dv_ref, atol=1e-2, rtol=1e-2, equal_nan=True) 
        torch.testing.assert_close(dq, dq_ref, atol=1e-2, rtol=1e-2, equal_nan=True) 
        torch.testing.assert_close(dk, dk_ref, atol=1e-2, rtol=1e-2, equal_nan=True)
