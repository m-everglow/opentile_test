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
from op.hstu_triton_bwd import triton_hstu_attention_backward

# torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libhstu_dense_ops.so")
ENABLE_DATACACHE = True
ENABLE_TRITON = True

def compare(a, b):
    name = "triton"

    print(f"info: 开始计算 {name} 精度。")
    ratio_threshold = 0.005
    threshold_diff = 0.005
    if a.shape == b.shape:
        # 两个张量shape相同，开始对比
        if torch.all(torch.eq(a, b)):
            print(f"info: {name} 计算结果与标杆完全相同。")
            ratio_diff = 0
            diff = 0
            ratio = 1
            max = 0
            sum = 0
        else:
            # 统计两个张量的值差距超过5‰的元素数量和占比
            diff = torch.abs(a.sub(b))
            min = torch.empty(a.shape, dtype=torch.float16).to(a.device)
            min.fill_(0.000025)
            max = torch.max(torch.abs(a), torch.abs(b))
            threshold = torch.max(max.mul(ratio_threshold), min)
            mask = diff > threshold
            num_diff = torch.sum(mask)
            ratio_diff = num_diff / torch.numel(a)
            if ratio_diff > 0.1:
                print(
                    f"error: {name} 计算结果有{num_diff}个元素的偏差超过{ratio_threshold:.2%}，占比为{ratio_diff:.4%}，超过阈值，用例执行失败！！！")
                pytest.fail(f"error: {name} 计算结果有{num_diff}个元素的偏差超过{ratio_threshold:.2%}，占比为{ratio_diff:.4%}，超过阈值，用例执行失败！！！")
            else:
                print(
                    f"info: {name} 计算结果有{num_diff}个元素的偏差超过{ratio_threshold:.2%}，占比为{ratio_diff:.4%} 。")
            ratio = (1 - ratio_diff).cpu().detach().numpy()
            # log(a.max(), b.max())
            max = torch.max(diff).cpu().detach().numpy()
            sum = torch.sum(diff).cpu().detach().numpy()
        print(f"{name} diff_max : {max:.8f}")
        print(f"{name} diff_sum : {sum:.6f}")
    else:
        print(f"error: {name}计算结果错误，shape与标杆不匹配，用例执行失败！！！")
        pytest.fail(f"error: {name}计算结果错误，shape与标杆不匹配，用例执行失败！！！")
        print(f"debug: a {a.shape}")
        print(f"debug: b {b.shape}")
    return

def debug_print(*args, **kwargs):
    print("[Debug]", *args, **kwargs)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def allclose(tensor: torch.Tensor, other: torch.Tensor, atol: float, rtol: float) -> bool:
    assert tensor.shape == other.shape
    name = "triton"
    ratio_threshold = 0.005
    threshold_diff = 0.005    
    a = tensor
    b = other
    if a.shape == b.shape:
        # 两个张量shape相同，开始对比
        if torch.all(torch.eq(a, b)):
            print(f"info: {name} 计算结果与标杆完全相同。")
            ratio_diff = 0
            diff = 0
            ratio = 1
            max = 0
            sum = 0
        else:
            # 统计两个张量的值差距超过5‰的元素数量和占比
            diff = torch.abs(a.sub(b))
            min = torch.empty(a.shape, dtype=torch.float16).to(a.device)
            min.fill_(0.000025)
            max = torch.max(torch.abs(a), torch.abs(b))
            threshold = torch.max(max.mul(ratio_threshold), min)
            mask = diff > threshold
            num_diff = torch.sum(mask)
            ratio_diff = num_diff / torch.numel(a)
            if ratio_diff > 0.1:
                print(
                    f"error: {name} 计算结果有{num_diff}个元素的偏差超过{ratio_threshold:.2%}，占比为{ratio_diff:.4%}，超过阈值，用例执行失败！！！")
                return False
            else:
                print(
                    f"info: {name} 计算结果有{num_diff}个元素的偏差超过{ratio_threshold:.2%}，占比为{ratio_diff:.4%} 。")
                return True
            ratio = (1 - ratio_diff).cpu().detach().numpy()
            # log(a.max(), b.max())
            max = torch.max(diff).cpu().detach().numpy()
            sum = torch.sum(diff).cpu().detach().numpy()
        print(f"{name} diff_max : {max:.8f}")
        print(f"{name} diff_sum : {sum:.6f}")
    else:
        print(f"error: {name}计算结果错误，shape与标杆不匹配，用例执行失败！！！")
        pytest.fail(f"error: {name}计算结果错误，shape与标杆不匹配，用例执行失败！！！")
        print(f"debug: a {a.shape}")
        print(f"debug: b {b.shape}")
        return False
    return True

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
    grad_dens = jagged_to_dense(grad.to(data_type), seq_lens_q, head_nums_q, head_dim_v).to("cpu")
    q_dens = jagged_to_dense(q.to(data_type), seq_lens_q, head_nums_q, head_dim_qk).to("cpu")
    k_dens = jagged_to_dense(k.to(data_type), seq_lens_k, head_nums_k, head_dim_qk).to("cpu")
    v_dens = jagged_to_dense(v.to(data_type), seq_lens_k, head_nums_k, head_dim_v).to("cpu")

    k_dens_expanded = k_dens.repeat_interleave(h_qk_ratio, dim=2)
    v_dens_expanded = v_dens.repeat_interleave(h_qk_ratio, dim=2)

    qk = torch.matmul(q_dens.permute(0, 2, 1, 3).npu(), k_dens_expanded.permute(0, 2, 3, 1).npu()).cpu()
    gv = torch.matmul(grad_dens.permute(0, 2, 1, 3).npu(), v_dens_expanded.permute(0, 2, 3, 1).npu()).cpu()

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

    if q.dtype == torch.float8_e4m3fn:
        score = score.to(q.dtype)
    score = score.to(data_type)
    v_grad_dens = torch.matmul(score.permute(0, 1, 3, 2).npu(), grad_dens.permute(0, 2, 1, 3).npu()).permute(0, 2, 1, 3).cpu()
    # debug_print(f"score.dtype {score.dtype} grad_dens.dtype {grad_dens.dtype} v_grad_dens.dtype {v_grad_dens.dtype}")

    if mask_type == 0 or mask_type == 3:
        bias_grad = gv * real_silu_scale * mask * F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))
    else:
        bias_grad = gv * real_silu_scale * F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))
    bias_grad = bias_grad * alpha
    if q.dtype == torch.float8_e4m3fn:
        bias_grad = bias_grad.to(q.dtype)
    bias_grad = bias_grad.to(data_type)
    k_grad_dens = torch.matmul(bias_grad.permute(0, 1, 3, 2).npu(), q_dens.permute(0, 2, 1, 3).npu()).permute(0, 2, 1, 3).cpu()
    q_grad_dens = torch.matmul(bias_grad.npu(), k_dens_expanded.permute(0, 2, 1, 3).npu()).permute(0, 2, 1, 3).cpu()

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
    q = torch.empty([num_tokens_q, num_heads_q, attention_dim], device=device).uniform_(-1, 1).to(dtype)
    k = torch.empty([num_tokens_k, num_heads_k, attention_dim], device=device).uniform_(-1, 1).to(dtype)
    v = torch.empty([num_tokens_k, num_heads_k, linear_dim], device=device).uniform_(-1, 1).to(dtype)
    dout = torch.empty([num_tokens_q, num_heads_q, linear_dim], device=device).uniform_(-1, 1).to(dtype)
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
            None, # mask
            None, # attn_bias
            2, # mask_type
            seq_len_q, # ? max_seq_len_q
            seq_len_k, # ? max_seq_len_k
            (1.0 / seq_len_q), # silu_scale
            seq_offset_q, # seq_offset_q
            seq_offset_k, # seq_offset_k
            None, # num_context
            None, # num_target
            None, # target_group_size
            alpha, # alpha
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

@pytest.mark.parametrize(
    "batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k, attention_dim, linear_dim, use_asc_golden",
    [
        (2, 8, 4, 256, 64, 64, 64, False),      # GQA
        (16, 4, 4, 32, 499, 64, 32, False),     # (v_d != qk_d)
        (5, 8, 4, 128, 256, 96, 128, False),    # (v_d != qk_d)
        (4, 4, 4, 512, 1024, 72, 64, False),    # (v_d != qk_d)
        (4, 4, 4, 512, 1024, 72, 80, False),    # (v_d != qk_d)
        (2, 4, 1, 1001, 901, 128, 48, False),   # (v_d != qk_d)
        (16, 4, 4, 501, 1000, 64, 64, False),   # delta_q
        (96, 2, 2, 512, 3072, 256, 256, False), # delta_q
        (96, 2, 2, 512, 3072, 512, 512, False), # delta_q
        (2048, 4, 4, 52, 1000, 64, 64, False),  # delta_q
        (8, 8, 8, 8000, 8000, 128, 128, False),
        (8, 8, 8, 8000, 8000, 256, 256, False),
        (2048, 2, 2, 32, 32, 256, 256, False),
        # (1, 2, 2, 32, 32, 256, 256, True),
        # (1, 1, 1, 32, 32, 64, 64, True),
        # (1, 1, 1, 64, 64, 16, 16, True),
        # (1, 1, 1, 16, 16, 32, 32, True),
        # (1, 1, 1, 16, 16, 64, 64, True),
        # (1, 1, 1, 16, 16, 128, 128, True)
    ],
)
@pytest.mark.parametrize("dtype_str", ["fp8"]) # TODO: fp8暂不支持
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
        dq_triton, dk_triton, dv_triton = triton_hstu_attention_backward(
            dout=dout,
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
        )
        debug_print("triton_hstu_attention_backward done")


    compare(dq_golden, dq_triton)


# @pytest.mark.parametrize(
#     "batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k, attention_dim, linear_dim",
#     [
#         (8, 8, 8, 8000, 8000, 128, 128),
#         (8, 8, 8, 8000, 8000, 256, 256),
#         (2048, 2, 2, 32, 32, 256, 256),
#     ],
# )
# @pytest.mark.parametrize("dtype_str", ["fp16", "bf16", "fp8"]) # TODO: fp8暂不支持
# def test_asc_bwd(
#     batch_size: int,
#     num_heads_q: int,
#     num_heads_k: int,
#     seq_len_q: int,
#     seq_len_k: int,
#     attention_dim: int,
#     linear_dim: int,
#     dtype_str: str,
# ):
#     # ===========prepare input===========
#     set_seed(42)
#     errors = []
#     device = torch.device("npu")
#     type_mapper = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
#     dtype = type_mapper.get(dtype_str)
#     debug_print(f"dtype {dtype}")
#     alpha = 1.0 / (attention_dim ** 0.5)
#     golden_filename = (
#         f"cpu_bs{batch_size}_hq{num_heads_q}_hk{num_heads_k}_"
#         f"sq{seq_len_q}_sk{seq_len_k}_ad{attention_dim}_ld{linear_dim}_dt{dtype}.pt"
#     )
#     dout, q, k, v, seq_offset_q, seq_offset_k = prepare_data(
#         batch_size, num_heads_q, num_heads_k, seq_len_q, seq_len_k,
#         attention_dim, linear_dim, device, dtype
#     )
#    # ===========asc backward===========
#     run_golden(
#         dout, q, k, v,
#         seq_offset_q, seq_offset_k, seq_len_q, seq_len_k,
#         alpha, dtype, device, True, golden_filename
#     )
