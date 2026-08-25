import torch
import torch_npu
import numpy as np
import random
import pytest
import sysconfig
import os
import sys
import time
import traceback
import torch.nn.functional as F
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from op.hstu_triton_fwd import triton_hstu_attention_fwd
from precision_calcu import *

def jagged_to_dense(jagged_tensor, seq_lens, head_nums, attn_dim):
    if isinstance(seq_lens, torch.Tensor):
        seq_lens_t = seq_lens
        max_len = seq_lens.max().item()
        batch_size = len(seq_lens)
    else:
        seq_lens_t = torch.tensor(seq_lens, device=jagged_tensor.device)
        max_len = max(seq_lens)
        batch_size = len(seq_lens)

    device = jagged_tensor.device
    
    jagged_flat = jagged_tensor.view(-1, head_nums, attn_dim)

    dense_tensor = torch.zeros(
        batch_size, max_len, head_nums, attn_dim, 
        dtype=jagged_tensor.dtype, device=device
    )

    range_row = torch.arange(max_len, device=device).unsqueeze(0)
    lens_col = seq_lens_t.unsqueeze(1)
    mask = range_row < lens_col

    dense_tensor[mask] = jagged_flat

    return dense_tensor

def dense_to_jagged(q, dense_tensor, seq_lens):
    device = dense_tensor.device
    B, N, H, D = dense_tensor.shape

    if not isinstance(seq_lens, torch.Tensor):
        seq_lens_t = torch.tensor(seq_lens, device=device)
    else:
        seq_lens_t = seq_lens.to(device)
    range_row = torch.arange(N, device=device).unsqueeze(0)
    lens_col = seq_lens_t.unsqueeze(1)
    mask = range_row < lens_col
    jagged_tensor = dense_tensor[mask]

    return jagged_tensor

def golden_op_exec_low(q, k, v, seq_offset, seq_offset_k, silu_scale, alpha, max_seq_len_q, max_seq_len_k):
        (_, head_nums_q, head_dim), data_type = q.shape, q.dtype
        head_nums_k = k.shape[1]
        head_dim_v = v.shape[2]
        batch_size = seq_offset.shape[0] - 1
        if data_type == torch.float8_e4m3fn:
            data_type = torch.float16
        if head_nums_q != head_nums_k:
            assert head_nums_q % head_nums_k == 0, (f"head_num_q ({head_nums_q}) must be divisible by "
                                                    f"head_num_k({head_nums_k}) ")
        h_qk_ratio = head_nums_q // head_nums_k

        seq_lens = np.zeros((batch_size,)).astype(np.int64)
        seq_lens_k = np.zeros((batch_size,)).astype(np.int64)
        print("zeros done")
        for batch_id in range(batch_size):
            seq_lens[batch_id] = seq_offset[batch_id + 1] - seq_offset[batch_id]
            seq_lens_k[batch_id] = seq_offset_k[batch_id + 1] - seq_offset_k[batch_id]

        silu_scale = 1 / max_seq_len_q if silu_scale == 0 else silu_scale
        print("seq_lens_k done")

        q_dens = jagged_to_dense(q.to(data_type), seq_lens, head_nums_q, head_dim).to(data_type).to("cpu")
        print("q_dens jagged_to_dense done")
        k_dens = jagged_to_dense(k.to(data_type), seq_lens_k, head_nums_k, head_dim).to(data_type).to("cpu")
        print("k_dens jagged_to_dense done")
        v_dens = jagged_to_dense(v.to(data_type), seq_lens_k, head_nums_k, head_dim_v).to(data_type).to("cpu")
        print("v_dens jagged_to_dense done")
        k_dens_expanded = k_dens.repeat_interleave(h_qk_ratio, dim=2)
        v_dens_expanded = v_dens.repeat_interleave(h_qk_ratio, dim=2)

        q_dens = q_dens.permute(0, 2, 1, 3).npu()
        k_dens = k_dens_expanded.permute(0, 2, 3, 1).npu()
        print("permute done")
        # Matmul放在npu上加速计算
        qk_attn = torch.matmul(q_dens, k_dens).to(torch.float32).cpu()
        print("matmul done")

        qk_attn = F.silu(qk_attn * alpha) * silu_scale
        print("silu done")

        v_dens = v_dens_expanded.permute(0, 2, 1, 3).cpu()
        if q.dtype == torch.float8_e4m3fn:
            qk_attn = qk_attn.to(q.dtype)
        qk_attn = qk_attn.to(data_type)
        attn_output = torch.matmul(qk_attn.npu(), v_dens.npu()).to("cpu")
        print("v_dens done")
        attn_output = attn_output.permute(0, 2, 1, 3)

        attn_output = dense_to_jagged(q, attn_output, seq_lens)

        torch.npu.synchronize()
        return attn_output.to(data_type).reshape(-1)


def golden_op_exec_high(q, k, v, seq_offset, seq_offset_k, silu_scale, alpha, max_seq_len_q, max_seq_len_k):
        (_, head_nums_q, head_dim), data_type = q.shape, q.dtype
        head_nums_k = k.shape[1]
        head_dim_v = v.shape[2]
        batch_size = seq_offset.shape[0] - 1

        data_type = torch.float32
        if head_nums_q != head_nums_k:
            assert head_nums_q % head_nums_k == 0, (f"head_num_q ({head_nums_q}) must be divisible by "
                                                    f"head_num_k({head_nums_k}) ")
        h_qk_ratio = head_nums_q // head_nums_k

        seq_lens = np.zeros((batch_size,)).astype(np.int64)
        seq_lens_k = np.zeros((batch_size,)).astype(np.int64)
        print("zeros done")
        for batch_id in range(batch_size):
            seq_lens[batch_id] = seq_offset[batch_id + 1] - seq_offset[batch_id]
            seq_lens_k[batch_id] = seq_offset_k[batch_id + 1] - seq_offset_k[batch_id]

        silu_scale = 1 / max_seq_len_q if silu_scale == 0 else silu_scale
        print("seq_lens_k done")

        q_dens = jagged_to_dense(q.to(data_type), seq_lens, head_nums_q, head_dim).to(data_type).to("cpu")
        print("q_dens jagged_to_dense done")
        k_dens = jagged_to_dense(k.to(data_type), seq_lens_k, head_nums_k, head_dim).to(data_type).to("cpu")
        print("k_dens jagged_to_dense done")
        v_dens = jagged_to_dense(v.to(data_type), seq_lens_k, head_nums_k, head_dim_v).to(data_type).to("cpu")
        print("v_dens jagged_to_dense done")
        k_dens_expanded = k_dens.repeat_interleave(h_qk_ratio, dim=2)
        v_dens_expanded = v_dens.repeat_interleave(h_qk_ratio, dim=2)

        q_dens = q_dens.permute(0, 2, 1, 3).npu()
        k_dens = k_dens_expanded.permute(0, 2, 3, 1).npu()
        print("permute done")
        # Matmul放在npu上加速计算
        qk_attn = torch.matmul(q_dens, k_dens).to(torch.float32).cpu()
        print("matmul done")

        qk_attn = F.silu(qk_attn * alpha) * silu_scale
        print("silu done")

        v_dens = v_dens_expanded.permute(0, 2, 1, 3).cpu()
        if q.dtype == torch.float8_e4m3fn:
            qk_attn = qk_attn.to(q.dtype)
        qk_attn = qk_attn.to(data_type)
        attn_output = torch.matmul(qk_attn.npu(), v_dens.npu()).to("cpu")
        print("v_dens done")
        attn_output = attn_output.permute(0, 2, 1, 3)

        attn_output = dense_to_jagged(q, attn_output, seq_lens)

        torch.npu.synchronize()
        return attn_output.to(data_type).reshape(-1)

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ===== 偶现精度问题排查辅助 =====
DEBUG_DUMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "_debug_dumps")
os.makedirs(DEBUG_DUMP_DIR, exist_ok=True)


def _analyze_mare_position(golden, actual, top_k=5):
    """分析 MARE 最大值指向的位置（batch/head/dim/seq），用于判断错误是否稳定。
    返回 [(value, idx_tuple, pct_err), ...]"""
    g = golden.to(torch.float32)
    a = actual.to(torch.float32)
    flat_err = torch.abs(a - g) / (torch.abs(g) + 1e-7)
    flat_err = flat_err.flatten()
    # 过滤掉 golden=0 的极端位置（MARE 在 0 处分母是 MIN_ERR，本来就大）
    vals, idxs = torch.topk(flat_err, k=min(top_k, flat_err.numel()))
    B, H, D = golden.shape[0], golden.shape[2], golden.shape[3]
    # shape: [num_tokens, num_heads_q, linear_dim]
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        tok = i // (H * D)
        rem = i % (H * D)
        head = rem // D
        dim = rem % D
        out.append((v, (tok, head, dim)))
    return out


def _bucket_error_counts(golden, actual, buckets=(0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 10.0)):
    """把误差分桶统计：<0.01%, <0.1%, <1%, <10%, <50%, <100%, >100%。
    帮助判断是少数极端错还是普遍偏。"""
    g = golden.to(torch.float32)
    a = actual.to(torch.float32)
    rel_err = (torch.abs(a - g) / (torch.abs(g) + 1e-7)).flatten()
    out = {}
    for thr in buckets:
        out[f"<{thr}"] = int((rel_err < thr).sum().item())
    out["nan"] = int(torch.isnan(actual).sum().item())
    out["inf"] = int(torch.isinf(actual).sum().item())
    out["total"] = int(rel_err.numel())
    return out


def _dump_failed_iter(case_tag, q, k, v, seq_offset_q, seq_offset_k,
                     golden_output, sim_output, triton_output, mare_positions):
    """把失败 iter 的完整输入输出 + 错误位置 dump 到磁盘。"""
    payload = {
        "case_tag": case_tag,
        "q": q.detach().cpu(),
        "k": k.detach().cpu(),
        "v": v.detach().cpu(),
        "seq_offset_q": seq_offset_q.detach().cpu(),
        "seq_offset_k": seq_offset_k.detach().cpu(),
        "golden_output": golden_output.detach().cpu(),
        "sim_output": sim_output.detach().cpu(),
        "triton_output": triton_output.detach().cpu(),
        "mare_positions": mare_positions,
    }
    fname = f"failed_{case_tag}_{int(time.time()*1000)}.pt"
    fpath = os.path.join(DEBUG_DUMP_DIR, fname)
    torch.save(payload, fpath)
    return fpath

def get_or_generate_data(
    batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k, 
    attention_dim, linear_dim, device, dtype
):
    # 配置缓存目录
    cache_dir = "./data_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # 构建唯一文件名
    file_name = (
        f"bs{batch_size}_hq{num_heads_q}_hk{num_heads_k}_"
        f"sq{seq_len_q}_sk{seq_len_k}_ad{attention_dim}_ld{linear_dim}_dt{dtype}.pt"
    )
    file_path = os.path.join(cache_dir, file_name)

    print(f"\n[INFO] Checking data cache: {file_path}")

    if os.path.exists(file_path):
        print(f"[INFO] Cache hit. Loading data...")
        data = torch.load(file_path, map_location=device)
        q = data["q"]
        k = data["k"]
        v = data["v"]
        seq_offset_q = data["seq_offset_q"]
        seq_offset_k = data["seq_offset_k"]
        print(f"[INFO] Data loaded successfully.")
    else:
        print(f"[INFO] Cache miss. Generating new data...")
        set_seed(42)
        
        # 在 CPU 上生成元数据
        seq_lens_k = torch.full((batch_size,), seq_len_k, dtype=torch.int64)
        seq_lens_q = torch.full((batch_size,), seq_len_q, dtype=torch.int64)
        zero = torch.tensor([0], dtype=torch.int64)
        seq_offset_k = torch.cat([zero, torch.cumsum(seq_lens_k, dim=0)])
        seq_offset_q = torch.cat([zero, torch.cumsum(seq_lens_q, dim=0)])
        
        num_tokens_k = seq_offset_k[-1].item()
        num_tokens_q = seq_offset_q[-1].item()

        print(f"[INFO] Total tokens Q: {num_tokens_q}, Total tokens K: {num_tokens_k}")

        q = torch.rand([num_tokens_q, num_heads_q, attention_dim]).to(dtype)
        k = torch.rand([num_tokens_k, num_heads_k, attention_dim]).to(dtype)
        v = torch.rand([num_tokens_k, num_heads_k, linear_dim]).to(dtype)
        
        # 保存所有数据到文件
        save_dict = {
            "q": q, "k": k, "v": v,
            "seq_offset_q": seq_offset_q, "seq_offset_k": seq_offset_k
        }
        torch.save(save_dict, file_path)
        print(f"[INFO] Data saved to {file_path}")
        
        # 移动到计算设备
        q = q.to(device)
        k = k.to(device)
        v = v.to(device)
        seq_offset_q = seq_offset_q.to(device)
        seq_offset_k = seq_offset_k.to(device)

    print(f"[INFO] Input shapes - Q: {q.shape}, K: {k.shape}, V: {v.shape}")
    print(f"[INFO] Device: {device}, Dtype: {dtype}")

    return q, k, v, seq_offset_q, seq_offset_k

@pytest.mark.parametrize(
    "batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k, attention_dim, linear_dim",
    [
        (2, 8, 4, 256, 64, 64, 64), # GQA
        (2, 8, 4, 64, 64, 64, 64),
        (16, 4, 4, 32, 499, 64, 32), # (v_d != qk_d)
        (5, 8, 4, 128, 256, 96, 128), # (v_d != qk_d)
        (4, 4, 4, 512, 1024, 72, 64), # (v_d != qk_d)
        (4, 4, 4, 512, 1024, 72, 80), # (v_d != qk_d)
        (2, 4, 1, 1001, 901, 128, 48), # (v_d != qk_d)
        (16, 4, 4, 501, 1000, 64, 64), # delta_q
        (8, 8, 8, 8000, 8000, 128, 128),
        (8, 8, 8, 8000, 8000, 256, 256),
        (2048, 4, 4, 52, 1000, 64, 64), # delta_q
        (2048, 2, 2, 32, 32, 256, 256),
        (2048, 2, 2, 64, 32, 256, 256),
        (96, 2, 2, 512, 3072, 256, 256), # delta_q
        (96, 2, 2, 512, 3072, 512, 512), # delta_q
        (1, 2, 2, 32, 32, 256, 256),
        (2, 8, 4, 32, 32, 32, 32),
    ],
)
@pytest.mark.parametrize("dtype_str", ["fp16", "bf16", "fp8"])
def test_triton_matches_golden(
    batch_size: int,
    num_heads_q: int,
    num_heads_k: int,
    seq_len_q: int,
    seq_len_k: int,
    attention_dim: int,
    linear_dim: int,
    dtype_str: str,
):
    # ===========prepare input===========
    set_seed(42)
    errors = [] 
    device = torch.device("npu")
    alpha = 1.0 / (attention_dim ** 0.5)
    type_mapper = {"fp16":torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
    dtype = type_mapper.get(dtype_str)
    q, k, v, seq_offset_q, seq_offset_k = get_or_generate_data(
        batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k, 
        attention_dim, linear_dim, device, dtype
    )


    golden_output = golden_op_exec_high(
        q=q,
        k=k,
        v=v,
        silu_scale=1.0 / seq_len_q,
        alpha=alpha,
        seq_offset=seq_offset_q,
        seq_offset_k=seq_offset_k,
        max_seq_len_q=seq_len_q,
        max_seq_len_k=seq_len_k,
    ).view([-1, num_heads_q, linear_dim]).to("npu")


    sim_output = golden_op_exec_low(
        q=q,
        k=k,
        v=v,
        silu_scale=1.0 / seq_len_q,
        alpha=alpha,
        seq_offset=seq_offset_q,
        seq_offset_k=seq_offset_k,
        max_seq_len_q=seq_len_q,
        max_seq_len_k=seq_len_k,
    ).view([-1, num_heads_q, linear_dim]).to("npu")
    
    triton_output = triton_hstu_attention_fwd(
        q=q,
        k=k,
        v=v,
        max_seq_len_q=seq_len_q,
        max_seq_len_k=seq_len_k,
        seq_offsets_q=seq_offset_q,
        seq_offsets_k=seq_offset_k,
        num_context=None,
        num_target=None,
        alpha=alpha,
        silu_scale=1.0 / seq_len_q,
    ).view([-1, num_heads_q, linear_dim]).to(device)

    # ===== 排查 1: Triton 算子 iter-to-iter 一致性 =====
    # 同一份输入跑两次，看 Triton 内部是否一致——不一致就是 triton 内部 race
    triton_output_2 = triton_hstu_attention_fwd(
        q=q, k=k, v=v,
        max_seq_len_q=seq_len_q, max_seq_len_k=seq_len_k,
        seq_offsets_q=seq_offset_q, seq_offsets_k=seq_offset_k,
        num_context=None, num_target=None,
        alpha=alpha, silu_scale=1.0 / seq_len_q,
    ).view([-1, num_heads_q, linear_dim]).to(device)
    triton_diff = (triton_output.float() - triton_output_2.float()).abs().max().item()
    print(f"[DEBUG] triton-output self-diff (max abs): {triton_diff:.6e}")

    # ===== 排查 2: NaN/Inf 检测 =====
    n_nan = int(torch.isnan(triton_output).sum().item())
    n_inf = int(torch.isinf(triton_output).sum().item())
    print(f"[DEBUG] triton nan count: {n_nan}, inf count: {n_inf}")

    if golden_output is None:
        return

    try:
        assert compare_cv(golden_output.npu(), sim_output.npu(), triton_output.npu())
    except AssertionError as e:
        case_tag = (f"{dtype_str}-{batch_size}-{num_heads_q}-{num_heads_k}"
                    f"-{seq_len_q}-{seq_len_k}-{attention_dim}-{linear_dim}")
        print(f"\n{'='*60}")
        print(f"[FAIL] case={case_tag}")
        print(f"[FAIL] triton self-diff (2x same input): {triton_diff:.6e}")
        print(f"[FAIL] nan={n_nan}, inf={n_inf}")

        # ===== 排查 3: MARE 峰值位置 =====
        mare_pos = _analyze_mare_position(golden_output, triton_output, top_k=5)
        print(f"[FAIL] top-5 MARE positions (tok, head, dim) -> rel_err:")
        num_tokens_q = int(seq_offset_q[-1].item())
        for rel_err, pos in mare_pos:
            tok, head, dim = pos
            print(f"        {pos} -> {rel_err:.6f}")
            if tok < num_tokens_q:
                batch_idx = tok // seq_len_q
                local_pos = tok % seq_len_q
                batch_offset = int(seq_offset_q[batch_idx].item())
                print(f"          (batch={batch_idx}, batch_offset={batch_offset}, local_seq_pos={local_pos}, head={head}, dim={dim})")
            else:
                print(f"          (out of range: tok={tok})")

        # ===== 排查 4: 误差分桶统计 =====
        bucket = _bucket_error_counts(golden_output, triton_output)
        print(f"[FAIL] error bucket distribution: {bucket}")

        # ===== 排查 5: dump 数据用于离线分析 =====
        fpath = _dump_failed_iter(
            case_tag, q, k, v, seq_offset_q, seq_offset_k,
            golden_output, sim_output, triton_output, mare_pos)
        print(f"[FAIL] dumped to: {fpath}")
        print(f"{'='*60}\n")
        raise
