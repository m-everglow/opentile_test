import torch
import torch_npu
import numpy as np
import random
import pytest
import sysconfig
import os
import sys
import torch.nn.functional as F
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from op.hstu_triton import triton_hstu_bwd
# torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libhstu_dense_ops.so")
ENABLE_DATACACHE = False
# torch.set_printoptions(threshold=float('inf'))

def debug_print(*args, **kwargs):
    print("[Debug]", *args, **kwargs)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def allclose(tensor: torch.Tensor, other: torch.Tensor, atol: float, rtol: float) -> bool:
    assert tensor.shape == other.shape
    diff = (torch.abs(tensor - other) > atol)
    diff_count = torch.sum(diff)
    return (diff_count / tensor.numel()) < rtol

def compare_backward_tensors(golden, triton, tensor_name, ATOL, RTOL, errors):
    is_close = torch.isclose(golden, triton, atol=ATOL, rtol=RTOL)
    
    if not torch.all(is_close):
        # 找到不匹配的位置
        mismatch_indices = torch.nonzero(~is_close, as_tuple=False)
        diff = (golden - triton).abs()
        max_diff = diff.max().item()
        
        # 收集详细的误差信息
        error_msg = []
        error_msg.append(f"Backward {tensor_name}: Max diff {max_diff} exceeds tolerance {ATOL}")
        error_msg.append(f"  Total mismatches: {len(mismatch_indices)} out of {golden.numel()}")
        
        # 限制打印的数量，避免太多输出
        max_errors_to_show = min(5, len(mismatch_indices))
        
        if max_errors_to_show > 0:
            error_msg.append(f"  Showing first {max_errors_to_show} mismatches:")
            
            # 收集前几个不匹配的详细信息
            for i in range(max_errors_to_show):
                idx = mismatch_indices[i]
                idx_tuple = tuple(idx.tolist())
                
                golden_val = golden[idx_tuple].item()
                triton_val = triton[idx_tuple].item()
                abs_diff = diff[idx_tuple].item()
                rel_diff = abs_diff / (abs(golden_val) + 1e-8)
                
                error_msg.append(
                    f"    Index {idx_tuple}: "
                    f"Golden={golden_val:.6e}, "
                    f"Triton={triton_val:.6e}, "
                    f"Abs diff={abs_diff:.6e}, "
                    f"Rel diff={rel_diff:.6e}"
                )
            
            # 如果还有更多不匹配，添加备注
            if len(mismatch_indices) > max_errors_to_show:
                error_msg.append(f"    ... and {len(mismatch_indices) - max_errors_to_show} more mismatches")
            
            # 添加统计信息
            abs_diffs = diff[~is_close].flatten()
            rel_diffs = abs_diffs / torch.abs(golden[~is_close].flatten()).clamp(min=1e-8)
            
            error_msg.append(f"  Mismatch statistics:")
            error_msg.append(f"    Mean absolute diff: {abs_diffs.mean().item():.6e}")
            error_msg.append(f"    Max absolute diff: {abs_diffs.max().item():.6e}")
            error_msg.append(f"    Mean relative diff: {rel_diffs.mean().item():.6e}")
            error_msg.append(f"    Max relative diff: {rel_diffs.max().item():.6e}")
        
        errors.append("\n".join(error_msg))

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


def dense_to_jagged(jagged_tensor, dense_tensor, seq_lens):
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


def golden_op_exec_bwd(
        grad,
        q,
        k,
        v,
        bias,
        mask,
        max_seq_len,
        max_seq_len_k,
        seq_offset_q,
        seq_offset_k,
        mask_type,
        silu_scale,
        enable_bias,
        data_type,
        alpha
):
    head_nums_q = q.shape[1]
    head_dim_qk = q.shape[2]
    head_nums_k = k.shape[1]
    head_dim_v = v.shape[2]
    batch_size = seq_offset_q.shape[0] - 1
    if data_type == torch.float8_e4m3fn:
        data_type = torch.float32

    if head_nums_q != head_nums_k:
        assert head_nums_q % head_nums_k == 0, (f"head_num_q ({head_nums_q}) must be divisible by "
                                                f"head_num_k({head_nums_k}) ")
    h_qk_ratio = head_nums_q // head_nums_k

    seq_lens_q = np.zeros((batch_size,)).astype(np.int64)
    seq_lens_k = np.zeros((batch_size,)).astype(np.int64)
    for batch_id in range(batch_size):
        seq_lens_q[batch_id] = seq_offset_q[batch_id + 1] - seq_offset_q[batch_id]
        seq_lens_k[batch_id] = seq_offset_k[batch_id + 1] - seq_offset_k[batch_id]

    max_seq_len = max(max_seq_len, max_seq_len_k)
    # 如果cpu执行太慢，可以开启ENABLE_DATACACHE并在NPU单次执行每个shape，获取golden缓存数据
    grad_dens = jagged_to_dense(grad.to(data_type), seq_lens_q, head_nums_q, head_dim_v).to("npu")
    q_dens = jagged_to_dense(q.to(data_type), seq_lens_q, head_nums_q, head_dim_qk).to("npu")
    k_dens = jagged_to_dense(k.to(data_type), seq_lens_k, head_nums_k, head_dim_qk).to("npu")
    v_dens = jagged_to_dense(v.to(data_type), seq_lens_k, head_nums_k, head_dim_v).to("npu")

    k_dens_expanded = k_dens.repeat_interleave(h_qk_ratio, dim=2)
    v_dens_expanded = v_dens.repeat_interleave(h_qk_ratio, dim=2)

    qk = torch.matmul(q_dens.permute(0, 2, 1, 3), k_dens_expanded.permute(0, 2, 3, 1))
    gv = torch.matmul(grad_dens.permute(0, 2, 1, 3), v_dens_expanded.permute(0, 2, 3, 1))

    qk = qk.float()
    gv = gv.float()

    if mask_type == 0 or mask_type == 3:
        mask = mask.to("npu")
        mask = mask.float()

    if enable_bias:
        bias = bias.to("npu")
        bias = bias.float()
        qkb = qk + bias
    else:
        qkb = qk
    qkb = qkb * alpha
    real_silu_scale = 1 / max_seq_len if silu_scale == 0.0 else silu_scale

    if mask_type == 0 or mask_type == 3:
        score = F.silu(qkb) * real_silu_scale * mask
    else:
        score = F.sigmoid(qkb) * qkb * real_silu_scale

    score = score.to(data_type)
    v_grad_dens = torch.matmul(score.permute(0, 1, 3, 2), grad_dens.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    # debug_print(f"score.dtype {score.dtype} grad_dens.dtype {grad_dens.dtype} v_grad_dens.dtype {v_grad_dens.dtype}")

    if mask_type == 0 or mask_type == 3:
        bias_grad = gv * real_silu_scale * mask * F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))
    else:
        bias_grad = gv * real_silu_scale * F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))
    bias_grad = bias_grad * alpha
    bias_grad = bias_grad.to(data_type)
    k_grad_dens = torch.matmul(bias_grad.permute(0, 1, 3, 2), q_dens.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    q_grad_dens = torch.matmul(bias_grad, k_dens_expanded.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

    bias_grad = bias_grad.cpu()
    q_grad_dens = q_grad_dens.cpu()
    q_grad = dense_to_jagged(q, q_grad_dens, seq_lens_q)
    k_grad_dens = k_grad_dens.cpu()
    if h_qk_ratio > 1:
        # shape is [B, L_k, H_q, D] -> reshape to [B, L_k, H_k, Ratio, D] -> sum(dim=3)
        B, L_k, H_q, D = k_grad_dens.shape
        k_grad_dens = k_grad_dens.view(B, L_k, head_nums_k, h_qk_ratio, D).sum(dim=3)
    k_grad = dense_to_jagged(k, k_grad_dens, seq_lens_k)
    v_grad_dens = v_grad_dens.cpu()
    if h_qk_ratio > 1:
        # shape is [B, L_k, H_q, D] -> reshape to [B, L_k, H_k, Ratio, D] -> sum(dim=3)
        B, L_k, H_q, D = v_grad_dens.shape
        v_grad_dens = v_grad_dens.view(B, L_k, head_nums_k, h_qk_ratio, D).sum(dim=3)
    v_grad = dense_to_jagged(v, v_grad_dens, seq_lens_k)

    torch.npu.synchronize()

    return q_grad, k_grad, v_grad, bias_grad


def prepare_data(
    batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k,
    attention_dim, linear_dim, device, dtype
):
    seq_lens_k = torch.full((batch_size,), seq_len_k, device=device, dtype=torch.int64)
    seq_lens_q = torch.full((batch_size,), seq_len_q, device=device, dtype=torch.int64)

    zero = torch.tensor([0], dtype=seq_lens_k.dtype, device=device)
    seq_offset_k = torch.cat([zero, torch.cumsum(seq_lens_k, dim=0)])
    seq_offset_q = torch.cat([zero, torch.cumsum(seq_lens_q, dim=0)])

    num_tokens_k = seq_offset_k[-1].item()
    num_tokens_q = seq_offset_q[-1].item()
    # q = torch.rand([num_tokens_q, num_heads_q, attention_dim]).to(dtype)
    # k = torch.rand([num_tokens_k, num_heads_k, attention_dim]).to(dtype)
    # v = torch.rand([num_tokens_k, num_heads_k, linear_dim]).to(dtype)
    # dout = torch.rand([num_tokens_q, num_heads_q, linear_dim]).to(dtype)

    # q = torch.ones([num_tokens_q, num_heads_q, attention_dim], device=device).to(dtype)
    # k = torch.ones([num_tokens_k, num_heads_k, attention_dim], device=device).to(dtype)
    # v = torch.ones([num_tokens_k, num_heads_k, linear_dim], device=device).to(dtype)
    # dout = torch.ones([num_tokens_q, num_heads_q, linear_dim], device=device).to(dtype)

    q = torch.empty([num_tokens_q, num_heads_q, attention_dim], device=device).uniform_(-1, 1).to(dtype).requires_grad_()
    k = torch.empty([num_tokens_k, num_heads_k, attention_dim], device=device).uniform_(-1, 1).to(dtype).requires_grad_()
    v = torch.empty([num_tokens_k, num_heads_k, linear_dim], device=device).uniform_(-1, 1).to(dtype).requires_grad_()
    dout = torch.empty([num_tokens_q, num_heads_q, linear_dim], device=device).uniform_(-1, 1).to(dtype)
    # 移动到计算设备
    # q = q.to(device).requires_grad_()
    # k = k.to(device).requires_grad_()
    # v = v.to(device).requires_grad_()
    # dout = dout.to(device)
    seq_offset_q = seq_offset_q.to(device)
    seq_offset_k = seq_offset_k.to(device)
    debug_print("prepare done")

    return dout, q, k, v, seq_offset_q, seq_offset_k


def run_golden(
    dout, q, k, v,
    seq_offset_q, seq_offset_k, seq_len_q, seq_len_k, alpha,
    dtype, device, use_asc_golden, golden_filename
):
    cache_dir = "./hstu_bwc_golden_output"
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, golden_filename)
    if ENABLE_DATACACHE and os.path.exists(file_path):
        if dtype == torch.float8_e4m3fn:
            dtype = torch.float32
        data = torch.load(file_path, map_location="cpu")
        dq_golden = data['dq'].to(device=device, dtype=dtype)
        dk_golden = data['dk'].to(device=device, dtype=dtype)
        dv_golden = data['dv'].to(device=device, dtype=dtype)
        debug_print(f"Golden Data Loaded.")
        return dq_golden, dk_golden, dv_golden

    if use_asc_golden:
        dq_golden, dk_golden, dv_golden, _ = torch.ops.mxrec.hstu_jagged_backward(
            dout,
            q,
            k,
            v,
            None,
            None,
            mask_type=2,
            max_seq_len=seq_len_q,
            max_seq_len_k=seq_len_q,
            silu_scale=1.0 / seq_len_q,
            seq_offset=seq_offset_q,
            seq_offset_k=seq_offset_q,
            num_context=None,
            num_target=None,
            target_group_size=None,
            alpha=alpha,
        )
        debug_print("Asc golden done")
    else:
        dq_golden, dk_golden, dv_golden, _ = golden_op_exec_bwd(
            dout,
            q,
            k,
            v,
            None,
            None,
            max_seq_len=seq_len_q,
            max_seq_len_k=seq_len_k,
            seq_offset_q=seq_offset_q,
            seq_offset_k=seq_offset_k,
            mask_type=2,
            silu_scale=1.0 / seq_len_q,
            enable_bias=False,
            data_type=dtype,
            alpha=alpha,
        )
        dq_golden = dq_golden.to(device)
        dk_golden = dk_golden.to(device)
        dv_golden = dv_golden.to(device)
        debug_print("golden done")

    if ENABLE_DATACACHE:
        golden_data = {
            "dq": dq_golden.cpu(),
            "dk": dk_golden.cpu(),
            "dv": dv_golden.cpu(),
        }
        torch.save(golden_data, file_path)
        debug_print("golden data saved.")

    return dq_golden, dk_golden, dv_golden

ENABLE_TRITON = True
@pytest.mark.parametrize(
    "dtype_str, batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k, attention_dim, linear_dim, use_asc_golden",
    [
    # ============== CI test case torch.float16 ================
        ("fp16", 2, 8, 4, 256, 64, 64, 64, False),      # test case # GQA
        ("fp16", 16, 4, 4, 32, 499, 64, 32, False),     # test case # (v_d != qk_d)
        ("fp16", 5, 8, 4, 128, 256, 96, 128, False),    # test case # (v_d != qk_d)
        ("fp16", 4, 4, 4, 512, 1024, 72, 64, False),    # test case # (v_d != qk_d)
        ("fp16", 4, 4, 4, 512, 1024, 72, 80, False),    # test case # (v_d != qk_d)
        ("fp16", 2, 4, 1, 1001, 901, 128, 48, False),   # test case # (v_d != qk_d)
        ("fp16", 16, 4, 4, 501, 1000, 64, 64, False),   # test case # delta_q
        ("fp16", 96, 2, 2, 512, 3072, 256, 256, False), # test case # delta_q
        ("fp16", 96, 2, 2, 512, 3072, 512, 512, False), # test case # delta_q
        ("fp16", 2048, 4, 4, 52, 1000, 64, 64, False),  # test case # delta_q
        ("fp16", 8, 8, 8, 8000, 8000, 128, 128, True), # test case
        ("fp16", 8, 8, 8, 8000, 8000, 256, 256, True), # test case
        ("fp16", 2048, 2, 2, 32, 32, 256, 256, True),  # test case
    # ============== CI test case torch.bfloat16 ================
        ("bf16", 2, 8, 4, 256, 64, 64, 64, False),      # test case # GQA
        ("bf16", 16, 4, 4, 32, 499, 64, 32, False),     # test case # (v_d != qk_d)
        ("bf16", 5, 8, 4, 128, 256, 96, 128, False),    # test case # (v_d != qk_d)
        ("bf16", 4, 4, 4, 512, 1024, 72, 64, False),    # test case # (v_d != qk_d)
        ("bf16", 4, 4, 4, 512, 1024, 72, 80, False),    # test case # (v_d != qk_d)
        ("bf16", 2, 4, 1, 1001, 901, 128, 48, False),   # test case # (v_d != qk_d)
        ("bf16", 16, 4, 4, 501, 1000, 64, 64, False),   # test case # delta_q
        ("bf16", 96, 2, 2, 512, 3072, 256, 256, False), # test case # delta_q
        ("bf16", 96, 2, 2, 512, 3072, 512, 512, False), # test case # delta_q
        ("bf16", 2048, 4, 4, 52, 1000, 64, 64, False),  # test case # delta_q
        ("bf16", 8, 8, 8, 8000, 8000, 128, 128, True), # test case
        ("bf16", 8, 8, 8, 8000, 8000, 256, 256, True), # test case
        ("bf16", 2048, 2, 2, 32, 32, 256, 256, True),  # test case
    # ============== debug case torch.bfloat16 ================
        # ("fp16", 2, 8, 8, 32, 32, 32, 32, False), # debug case
        # ("fp16", 2, 8, 4, 32, 32, 32, 32, False), # debug case
        # ("fp16", 2, 8, 2, 32, 32, 32, 32, False), # debug case
        # ("fp16", 2, 2, 2, 32, 32, 32, 32, True), # debug case
        # ("fp16", 2, 2, 2, 40, 32, 32, 32, True), # debug case
        # ("fp16", 1, 1, 1, 16, 16, 16, 16, True), # debug case
        # ("fp16", 1, 1, 1, 24, 16, 16, 16, False), # debug case
        # ("fp16", 1, 1, 1, 16, 24, 16, 16, False), # debug case
        # ("fp16", 1, 1, 1, 16, 30, 16, 16, False), # debug case
        # ("fp16", 1, 1, 1, 16, 176, 16, 16, False), # debug case
        # ("fp16", 1, 1, 1, 32, 30, 16, 16, False), # debug case
        # ("fp16", 1, 1, 1, 30, 32, 16, 16, False), # debug case
        # ("fp16", 1, 2, 2, 24, 16, 16, 16, False), # debug case
        # ("fp16", 16, 4, 4, 24, 16, 16, 16, False), # debug case
        # ("fp16", 16, 4, 4, 32, 16, 64, 32, False), # debug case
        # ("fp16", 16, 4, 4, 32, 176, 64, 32, False), # debug case
        # ("fp16", 16, 4, 4, 32, 512, 64, 32, False), # debug case
        # ("fp16", 16, 4, 4, 32, 496, 64, 32, False), # debug case
        # ("fp16", 16, 4, 4, 32, 480, 64, 32, False), # debug case
        # ("fp16", 1, 2, 2, 32, 32, 256, 256, False),
        # ("fp16", 1, 1, 1, 32, 32, 64, 64, True),
        # ("fp16", 1, 1, 1, 64, 64, 16, 16, True),
        # ("fp16", 1, 1, 1, 16, 16, 32, 32, True),
        # ("fp16", 1, 1, 1, 16, 16, 64, 64, True),
        # ("fp16", 1, 1, 1, 16, 16, 128, 128, True)
    ],
)
def test_triton_matches_asc_bwd(
    batch_size: int,
    num_heads_q: int,
    num_heads_k: int,
    seq_len_q: int,
    seq_len_k: int,
    attention_dim: int,
    linear_dim: int,
    use_asc_golden: bool,
    dtype_str: str,
):
    # ===========prepare input===========
    set_seed(42)
    errors = []
    device = torch.device("npu")
    type_mapper = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
    dtype = type_mapper.get(dtype_str)
    debug_print(f"dtype {dtype}")
    alpha = 1.0 / (attention_dim ** 0.5)
    golden_filename = (
        f"cpu_bs{batch_size}_hq{num_heads_q}_hk{num_heads_k}_"
        f"sq{seq_len_q}_sk{seq_len_k}_ad{attention_dim}_ld{linear_dim}_dt{dtype}.pt"
    )


    dout, q, k, v, seq_offset_q, seq_offset_k = prepare_data(
        batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k,
        attention_dim, linear_dim, device, dtype
    )

   # ===========backward===========
    dq_golden, dk_golden, dv_golden = run_golden(
        dout, q, k, v,
        seq_offset_q, seq_offset_k, seq_len_q, seq_len_k,
        alpha, dtype, device, use_asc_golden, golden_filename
    )

    if ENABLE_TRITON:
        out, dq_triton, dk_triton, dv_triton = triton_hstu_bwd(
            N=seq_len_q,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            dout=dout,
            seq_offsets_q=seq_offset_q,
            seq_offsets_k=seq_offset_k,
            causal=False
        )
        debug_print("triton_hstu_jagged_backward done")

        ATOL, RTOL = 1e-3, 1e-3
        if dtype == torch.bfloat16:
            ATOL, RTOL = 1e-2, 1e-2
        elif dtype == torch.float8_e4m3fn:
            dq_golden = dq_golden.to(torch.float32)
            dk_golden = dk_golden.to(torch.float32)
            dv_golden = dv_golden.to(torch.float32)
            dq_triton = dq_triton.to(torch.float32)
            dk_triton = dk_triton.to(torch.float32)
            dv_triton = dv_triton.to(torch.float32)
            ATOL = 5e-3
            RTOL = 5e-3

        debug_print("dq_triton: ", dq_triton)
        debug_print("dq_golden: ", dq_golden)
        # debug_print("dk_triton: ", dk_triton)
        # debug_print("dk_golden: ", dk_golden)
        # debug_print("dv_triton: ", dv_triton)
        # debug_print("dv_golden: ", dv_golden)

        # 分别比较三个梯度张量
        compare_backward_tensors(dq_golden, dq_triton, "dQ", ATOL, RTOL, errors)
        compare_backward_tensors(dk_golden, dk_triton, "dK", ATOL, RTOL, errors)
        compare_backward_tensors(dv_golden, dv_triton, "dV", ATOL, RTOL, errors)

    if len(errors) > 0:
        pytest.fail("ERROR \n".join(errors))
