import pytest
import torch
import torch_npu
import triton
import triton.language as tl
# from maybe_triton_jit import maybe_triton_jit
from triton.language.extra.cann import libdevice
# import triton.language.extra.ascend.libdevice as libdevice
from utils import is_hopper, is_ampere

def get_fwd_config():
    if is_hopper():
        return [triton.Config({"BLOCK_SIZE": 2048}, num_stages=2, num_warps=4)]
    elif is_ampere():
        return [triton.Config({"BLOCK_SIZE": 2048}, num_stages=4, num_warps=8)]
    else:
        return [triton.Config({"BLOCK_SIZE": 2048}, num_stages=2, num_warps=16)]

def get_bwd_config():
    if is_hopper():
        return [triton.Config({"BLOCK_SIZE": 2048}, num_stages=4, num_warps=4)]
    elif is_ampere():
        return [triton.Config({"BLOCK_SIZE": 2048}, num_stages=3, num_warps=8)]
    else:
        return [triton.Config({"BLOCK_SIZE": 512}, num_stages=2, num_warps=8)]

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": BM, "multibuffer": MB})
        for BM in [16384, 8192, 4096, 2048, 1024, 512]
        for MB in [True, False]
    ],
    key=["n_elements"]
)
@triton.jit
def softcap_fwd_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    softcap,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = softcap * (libdevice.tanh(x.to(tl.float32) / softcap)).to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": BM, "multibuffer": MB})
        for BM in [16384, 8192, 4096, 2048, 1024, 512]
        for MB in [True, False]
    ],
    key=["n_elements"]
)
@triton.jit
def softcap_bwd_kernel(
    dy_ptr,
    x_ptr,
    dx_ptr,
    n_elements,
    softcap,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    dy = tl.load(dy_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = libdevice.tanh(x.to(tl.float32) / softcap).to(x.dtype)
    dx = dy * (1 - y * y)
    tl.store(dx_ptr + offsets, dx, mask=mask)

class Softcap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, softcap):
        assert x.is_contiguous(), "Tensors must be contiguous"
        assert x.dtype in [torch.float32, torch.bfloat16, torch.float16]
        numel = x.numel()
        y = torch.empty_like(x)
        grid = lambda META: (triton.cdiv(numel, META["BLOCK_SIZE"]),)
        softcap_fwd_kernel[grid](x, y, numel, softcap, debug=True)
        ctx.save_for_backward(x)
        ctx.softcap = softcap
        return y

    @staticmethod
    def backward(ctx, dy):
        x = ctx.saved_tensors[0]
        numel = x.numel()
        dx = torch.empty_like(x)
        grid = lambda META: (triton.cdiv(numel, META["BLOCK_SIZE"]),)
        softcap_bwd_kernel[grid](dy, x, dx, numel, ctx.softcap)
        return dx, None

# 新增性能检查相关函数（保持不变）
import os
import csv
from pathlib import Path

def find_op_statistic_csv(result_dir: str) -> str:
    result_path = Path(result_dir)
    if not result_path.exists():
        raise FileNotFoundError(f"结果目录 {result_dir} 不存在")
    for root, dirs, files in os.walk(result_path):
        if "ASCEND_PROFILER_OUTPUT" in root and "op_statistic.csv" in files:
            csv_path = os.path.join(root, "op_statistic.csv")
            print(f"找到性能统计文件: {csv_path}")
            return csv_path
    raise FileNotFoundError(f"在 {result_dir} 及其子目录中未找到 op_statistic.csv 文件")

def get_op_avg_time(csv_path: str, op_name: str) -> float:
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["OP Type"].strip() == op_name:
                avg_time = float(row["Avg Time(us)"])
                print(f"OP {op_name} 的平均耗时: {avg_time} us")
                return avg_time
    raise ValueError(f"在 {csv_path} 中未找到 OP Type 为 {op_name} 的记录")

def check_performance_degradation():
    FWD_BASELINE = 6.01
    BWD_BASELINE = 7.562
    MAX_DEGRADATION_RATIO = 1.1

    csv_path = find_op_statistic_csv("./result_dir")
    fwd_avg_time = get_op_avg_time(csv_path, "softcap_fwd_kernel")
    bwd_avg_time = get_op_avg_time(csv_path, "softcap_bwd_kernel")

    fwd_max_allowed = FWD_BASELINE * MAX_DEGRADATION_RATIO
    bwd_max_allowed = BWD_BASELINE * MAX_DEGRADATION_RATIO

    assert fwd_avg_time <= fwd_max_allowed, \
        f"❌ softcap_fwd_kernel 性能劣化超过10%！基准值: {FWD_BASELINE} us, 实际值: {fwd_avg_time} us, 允许最大值: {fwd_max_allowed} us"
    assert bwd_avg_time <= bwd_max_allowed, \
        f"❌ softcap_bwd_kernel 性能劣化超过10%！基准值: {BWD_BASELINE} us, 实际值: {bwd_avg_time} us, 允许最大值: {bwd_max_allowed} us"
    
    print("✅ 所有性能检查通过！")
    print(f"   - softcap_fwd_kernel: {fwd_avg_time:.3f} us (≤ {fwd_max_allowed:.3f} us)")
    print(f"   - softcap_bwd_kernel: {bwd_avg_time:.3f} us (≤ {bwd_max_allowed:.3f} us)")

@pytest.mark.functiontest
def test_get_last_loc_kernel():
    dtype = torch.float16
    softcap = 50.0
    x = torch.randn((1024, 1024), dtype=dtype, device="npu").requires_grad_()

    y = Softcap.apply(x, softcap)
    torch.npu.synchronize()

    dy = torch.randn_like(y)
    y.backward(dy)
    triton_dx, x.grad = x.grad.clone(), None

    # --------------------- 核心修改：将print改为断言 ---------------------
    ref = softcap * torch.tanh(x.to(torch.float32) / softcap).to(dtype)
    ref.backward(dy)
    torch_dx, x.grad = x.grad.clone(), None
    atol = 1e-3
    rtol = 1e-3


    # 前向结果断言（失败时抛出明确异常，包含最大误差）
    assert torch.allclose(y, ref, atol=atol, rtol=rtol), \
        f"❌ [Fwd]Triton and Torch differ! atol={atol}, rtol={rtol}, max absolute error: {torch.max(torch.abs(y - ref)):.6f}"
    print("✅ [Fwd]Triton and Torch match")

    # 反向结果断言（失败时抛出明确异常，包含最大误差）
    assert torch.allclose(triton_dx, torch_dx, atol=atol, rtol=rtol), \
        f"❌ [Bwd]Triton and Torch differ! atol={atol}, rtol={rtol}, max absolute error: {torch.max(torch.abs(triton_dx - torch_dx)):.6f}"
    print("✅ [Bwd]Triton and Torch match")


@pytest.mark.perftest
def test_perf():
    dtype = torch.float16
    softcap = 50.0
    x = torch.randn((1024, 1024), dtype=dtype, device="npu").requires_grad_()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
        )

    with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.NPU],
                with_stack=False,
                record_shapes=False,
                profile_memory=False,
                schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=30, repeat=1, skip_first=1),
                experimental_config=experimental_config,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir")
        ) as prof:
            for i in range(30):
                y = Softcap.apply(x, softcap)
                torch.npu.synchronize()
                prof.step()

                dy = torch.randn_like(y)
                y.backward(dy)
                triton_dx, x.grad = x.grad.clone(), None

    # 执行性能断言检查
    check_performance_degradation()


if __name__ == '__main__':
    test_get_last_loc_kernel()
    test_perf()