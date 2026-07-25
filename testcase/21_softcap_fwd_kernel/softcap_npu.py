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

# @triton.autotune(
#     configs=get_fwd_config(),
#     key=[],
# )

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": BM, "multibuffer": MB, "BLOCK_NUM": BN})
        for BM in [18725, 16384, 8192, 4096, 2048, 1024, 512]
        for MB in [True, False]
        for BN in [64, 32, 16, 8, 4, 2, 1]
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
    BLOCK_NUM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    for i in range(BLOCK_NUM):
        # 0. calc addr of ptr
        block_start = pid * BLOCK_SIZE * BLOCK_NUM + i * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # 1. load x
        x = tl.load(x_ptr + offsets, mask=mask)

        # 2. softcap compute
        # y = softcap * (tl.tanh(x.to(tl.float32) / softcap)).to(x.dtype)
        y = softcap * (libdevice.tanh(x.to(tl.float32) / softcap)).to(x.dtype)

        # 3. store y
        tl.store(y_ptr + offsets, y, mask=mask)


# @triton.autotune(
#     configs=get_bwd_config(),
#     key=[],
# )
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": BM, "multibuffer": MB, "BLOCK_NUM": BN})
        for BM in [18725, 16384, 8192, 4096, 2048, 1024, 512]
        for MB in [True, False]
        for BN in [64, 32, 16, 8, 4, 2, 1]
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
    BLOCK_NUM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    for i in range(BLOCK_NUM):
        # 0. calc addr of ptr
        block_start = pid * BLOCK_SIZE * BLOCK_NUM + i * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # 1. load dy & x
        dy = tl.load(dy_ptr + offsets, mask=mask)
        x = tl.load(x_ptr + offsets, mask=mask)

        # 2. softcap backward compute
        y = libdevice.tanh(x.to(tl.float32) / softcap).to(x.dtype)

        # y = tl.tanh(x.to(tl.float32) / softcap).to(x.dtype)

        dx = dy * (1 - y * y)

        # 3. store dx
        tl.store(dx_ptr + offsets, dx, mask=mask)


class Softcap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, softcap):
        assert x.is_contiguous(), "Tensors must be contiguous"
        assert x.dtype in [
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ]
        numel = x.numel()
        # Allocates output.
        y = torch.empty_like(x)
        grid = lambda META: (triton.cdiv(numel, META["BLOCK_SIZE"] * META["BLOCK_NUM"]),)
        softcap_fwd_kernel[grid](
            x,
            y,
            numel,
            softcap,
            debug=True,
            enable_vf_fusion=True,
        )

        ctx.save_for_backward(x)
        ctx.softcap = softcap
        return y

    @staticmethod
    def backward(ctx, dy):
        x = ctx.saved_tensors[0]
        numel = x.numel()
        dx = torch.empty_like(x)
        grid = lambda META: (triton.cdiv(numel, META["BLOCK_SIZE"] * META["BLOCK_NUM"]),)
        softcap_bwd_kernel[grid](
            dy,
            x,
            dx,
            numel,
            ctx.softcap,
            enable_vf_fusion=True,
        )

        return dx, None


if __name__ == "__main__":
    dtype = torch.float16
    softcap = 50.0
    x = torch.randn((1024, 1024), dtype=dtype, device="npu").requires_grad_()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
        )

    with torch_npu.profiler.profile(
                activities=[  # torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
                with_stack=False,  # 采集torch 算子的函数调用栈的开关，该参数选填，默认关闭
                record_shapes=False,  # 采集torch 算子的input shape和input type的开关，该参数选填，默认关闭
                profile_memory=False,  # 采集memory相关数据的开关，该参数选填，默认关闭
                schedule=torch_npu.profiler.schedule(wait=1,
                                                    warmup=1,
                                                    active=30,
                                                    repeat=1,
                                                    skip_first=1),
                # warmup默认为0，老版本torch_npu包该参数为必填项
                experimental_config=experimental_config,  # 该参数选填，默认为Level0
                # 产生的profling文件的位置
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir")
        ) as prof:
            # prof.start()
            for i in range(30):
                # fn_triton(*args)
                # 这里填实际的上板运算
                y = Softcap.apply(x, softcap)

                torch.npu.synchronize()  # 确保 kernel 真正执行完
                prof.step()

                dy = torch.randn_like(y)
                y.backward(dy)
                triton_dx, x.grad = x.grad.clone(), None


    ref = softcap * torch.tanh(x.to(torch.float32) / softcap).to(dtype)
    ref.backward(dy)
    torch_dx, x.grad = x.grad.clone(), None
    atol = 1e-3
    rtol = 1e-3
    if torch.allclose(y, ref, atol=atol, rtol=rtol):
        print("✅ [Fwd]Triton and Torch match")
    else:
        print("❌ [Fwd]Triton and Torch differ")
    if torch.allclose(triton_dx, torch_dx, atol=atol, rtol=rtol):
        print("✅ [Bwd]Triton and Torch match")
    else:
        print("❌ [Bwd]Triton and Torch differ")

    # 改为自动化断言，失败直接抛异常，成功无输出
    assert torch.allclose(y, ref, atol=atol, rtol=rtol), "[Fwd] Triton 和 Torch 前向计算结果不匹配"
    assert torch.allclose(triton_dx, torch_dx, atol=atol, rtol=rtol), "[Bwd] Triton 和 Torch 反向梯度计算结果不匹配"