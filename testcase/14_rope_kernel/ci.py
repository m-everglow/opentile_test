import os
import torch
import torch_npu
import triton
import triton.language as tl
from triton.language.extra.ascend import libdevice
import pytest


# BLOCK_M_VALUES = (100, 75, 50, 25, 10, 5, 2)
# MULTIBUFFER_VALUES = (True, False)
# BLOCK_M_VALUES = (128, 64, 32, 16, 8, 4, 2)
BLOCK_M_VALUES = (16, )
MULTIBUFFER_VALUES = (False, )
ROPE_CONFIGS = [
    pytest.param(
        block_m,
        multibuffer,
        id=f"BLOCK_M={block_m}-multibuffer={'on' if multibuffer else 'off'}",
    )
    for block_m in BLOCK_M_VALUES
    for multibuffer in MULTIBUFFER_VALUES
]


@triton.jit
def rope_kernel(
    in_ptr, pos_ptr, cu_seqlens, out_ptr,
    head: tl.constexpr, base,
    DIM: tl.constexpr,
    max_seq_len,
    REVERSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    start_b = tl.program_id(0)
    begin = tl.load(cu_seqlens + start_b)
    len = tl.load(cu_seqlens + start_b + 1) - begin
    tasks = tl.cdiv(max_seq_len, BLOCK_M)
    for start_m in range(tasks):
        if start_m * BLOCK_M < len:
            begin = begin.to(tl.int64)
            x0_block_ptr = tl.make_block_ptr(
                base = in_ptr + begin * head * DIM,
                shape = (len, head, DIM),
                strides = (head * DIM, DIM, 1),
                offsets = (start_m * BLOCK_M, 0, 0),
                block_shape = (BLOCK_M, head, DIM // 2),
                order = (2, 1, 0)
            )
            y0_block_ptr = tl.make_block_ptr(
                base = in_ptr + begin * head * DIM,
                shape = (len, head, DIM),
                strides = (head * DIM, DIM, 1),
                offsets = (start_m * BLOCK_M, 0, DIM // 2),
                block_shape = (BLOCK_M, head, DIM // 2),
                order = (2, 1, 0)
            )
            x1_block_ptr = tl.make_block_ptr(
                base = out_ptr + begin * head * DIM,
                shape = (len, head, DIM),
                strides = (head * DIM, DIM, 1),
                offsets = (start_m * BLOCK_M, 0, 0),
                block_shape = (BLOCK_M, head, DIM // 2),
                order = (2, 1, 0)
            )
            y1_block_ptr = tl.make_block_ptr(
                base = out_ptr + begin * head * DIM,
                shape = (len, head, DIM),
                strides = (head * DIM, DIM, 1),
                offsets = (start_m * BLOCK_M, 0, DIM // 2),
                block_shape = (BLOCK_M, head, DIM // 2),
                order = (2, 1, 0)
            )
            pos_block_ptr = tl.make_block_ptr(
                base = pos_ptr + begin,
                shape = (len,),
                strides = (1,),
                offsets = (start_m * BLOCK_M,),
                block_shape = (BLOCK_M,),
                order = (0,)
            )

            x0 = tl.load(x0_block_ptr, boundary_check=(0,))         # x0/y0    = 16x4x64xf32
            y0 = tl.load(y0_block_ptr, boundary_check=(0,))
            pos = tl.load(pos_block_ptr, boundary_check=(0,))

            offset_n = tl.arange(0, DIM // 2)
            # print("yty test")
            # print(base)
            inv_freq = libdevice.pow(base, -2.0 / DIM * offset_n)       # inv_freq = 64xf32
            # inv_freq = tl.pow(base, -2.0 / DIM * offset_n)
            freqs = pos[:, None] * inv_freq[None, :]        # pos       = 16xf32 -> 16x1xf32 -> 16x64xf32
                                                            # inv_freq  = 64xf32 -> 1x64xf32 -> 16x64xf32
                                                            # freqs     = 16x64xf32
            sin = libdevice.sin(freqs)                      # sin/cos     = 16x64xf32
            cos = libdevice.cos(freqs)
            # sin = tl.sin(freqs)
            # cos = tl.cos(freqs)
            if REVERSE:
                sin = -sin
            x1 = x0 * cos[:, None, :] - y0 * sin[:, None, :]    # sin/cos     = 16x64xf32 -> 16x1x64xf32 -> 16x4x64xf32
            y1 = x0 * sin[:, None, :] + y0 * cos[:, None, :]
            dtype = in_ptr.type.element_ty
            tl.store(x1_block_ptr, x1.to(dtype), boundary_check=(0,))
            tl.store(y1_block_ptr, y1.to(dtype), boundary_check=(0,))


def rope_impl(input, position, offset, max_len, base=10000., reverse=False, block_m=100):
    len, head, dim = input.size()
    out = input.new_empty(len, head, dim)
    # print(f"==========> {offset=}")
    bs = offset.size(0) - 1
    grid = (bs,)
    rope_kernel[grid](input, position, offset, out, head, base, dim, max_len, reverse, block_m)
    return out


class RopeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, position, offset, max_len, base = 10000., reverse = False):
        ctx.position = position
        ctx.offset = offset
        ctx.max_len = max_len
        ctx.base = base
        ctx.reverse = reverse
        return rope_impl(input, position, offset, max_len, base, reverse)
    
    @staticmethod
    def backward(ctx, do):
        return rope_impl(do, ctx.position, ctx.offset, ctx.max_len, ctx.base, not ctx.reverse), None, None, None, None, None


class RotaryPositionalEmbeddings(torch.nn.Module):
    def __init__(self, d: int, base: int = 10_000):
        super().__init__()
        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        if (
            self.cos_cached is not None
            and x.shape[0] <= self.cos_cached.shape[0]
        ):
            return
        seq_len = x.shape[0]
        theta = (
            1.0
            / (
                self.base
                ** (torch.arange(0, self.d, 2).float() / self.d)
            )
        ).to(x.device)
        seq_idx = torch.arange(seq_len, device=x.device).float()
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
        idx_theta = torch.cat([idx_theta, idx_theta], dim=1)
        self.cos_cached = idx_theta.cos()[:, None, None, :]
        self.sin_cached = idx_theta.sin()[:, None, None, :]

    def _neg_half(self, x: torch.Tensor):
        half_dim = self.d // 2
        return torch.cat(
            [-x[:, :, :, half_dim:], x[:, :, :, :half_dim]],
            dim=-1,
        )

    def forward(self, x: torch.Tensor):
        self._build_cache(x)
        return (
            x * self.cos_cached[:x.shape[0]]
            + self._neg_half(x) * self.sin_cached[:x.shape[0]]
        )


def pad(input_tensor, sizes, max_len):
    padded_sequences = []
    remaining = input_tensor
    for size in sizes.tolist():
        sequence = remaining[:size]
        remaining = remaining[size:]
        padding = torch.zeros(
            max_len - size,
            sequence.size(1),
            sequence.size(2),
            dtype=sequence.dtype,
        )
        padded_sequences.append(torch.cat([sequence, padding]))
    return torch.stack(padded_sequences)


def unpad(input_tensor, sizes):
    return torch.cat(
        [
            input_tensor[index, :size]
            for index, size in enumerate(sizes.tolist())
        ],
    )


@pytest.mark.parametrize("block_m, multibuffer", ROPE_CONFIGS)
def test_rope_kernel(block_m, multibuffer):
    DEVICE = torch.device("npu")
    # MAX_LEN = 100
    # BS = 100
    # HEAD = 8
    # DIM = 64
    MAX_LEN = 16
    BS = 4
    HEAD = 4
    DIM = 128
    torch.manual_seed(0)
    size = torch.randint(MAX_LEN  - 4, [BS]) + 4
    offset = torch.nn.functional.pad(torch.cumsum(size, 0), [1, 0])
    all_len = size.sum()
    print(f"\n[debug] {all_len=}")
    input = torch.randn(all_len, HEAD, DIM)
    print(f"[debug] {input.shape=}")
    pos = []
    for sz in size.cpu().numpy():
        pos += [torch.arange(sz)]
    pos = torch.cat(pos)

    v = rope_impl(input.to(DEVICE), pos.to(DEVICE), offset.to(DEVICE), MAX_LEN, base=2., block_m=block_m)

    # experimental_config = torch_npu.profiler._ExperimentalConfig(
    #         aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    #         profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
    #     )

    # with torch_npu.profiler.profile(
    #             activities=[  # torch_npu.profiler.ProfilerActivity.CPU,
    #                 torch_npu.profiler.ProfilerActivity.NPU],
    #             with_stack=False,  # 采集torch 算子的函数调用栈的开关，该参数选填，默认关闭
    #             record_shapes=False,  # 采集torch 算子的input shape和input type的开关，该参数选填，默认关闭
    #             profile_memory=False,  # 采集memory相关数据的开关，该参数选填，默认关闭
    #             schedule=torch_npu.profiler.schedule(wait=1,
    #                                                 warmup=1,
    #                                                 active=30,
    #                                                 repeat=1,
    #                                                 skip_first=1),
    #             # warmup默认为0，老版本torch_npu包该参数为必填项
    #             experimental_config=experimental_config,  # 该参数选填，默认为Level0
    #             # 产生的profling文件的位置
    #             on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./prof_dir")
    #     ) as prof:
    #         # prof.start()
    #         for i in range(30):
    #             # fn_triton(*args)
    #             v = rope_impl(input.to(DEVICE), pos.to(DEVICE), offset.to(DEVICE), MAX_LEN, base=2.)
    #             torch.npu.synchronize()  # 确保 kernel 真正执行完
    #             prof.step()

    pad_input = pad(input, size, MAX_LEN)
    rope = RotaryPositionalEmbeddings(DIM, base=2)
    roped = rope(pad_input.transpose(0,1).to(DEVICE)).transpose(0,1)
    unpad_rope = unpad(roped, size)
    
    # print(unpad_rope)
    # print(v)

    # test_common.print_max_error(input, unpad_rope, v)

    # print(torch.allclose(unpad_rope, v, rtol=1e-3, atol=1e-3))
    # print(torch.abs(unpad_rope-v).max())

    # print(torch.abs(unpad_rope-v).max())
    # assert torch.allclose(unpad_rope, v, rtol=1e-3, atol=1e-3)

    atol = 1e-3
    rtol = 1e-3
    if not torch.allclose(v, unpad_rope, atol=atol, rtol=rtol):
        abs_diff = torch.abs(v - unpad_rope)
        tolerance = atol + rtol * torch.abs(unpad_rope)

        # 与 torch.allclose 的判断规则一致，并处理 NaN。
        mismatch_mask = (abs_diff > tolerance) | torch.isnan(abs_diff)
        mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=False)

        print("\n[ERROR] torch.allclose failed")
        print(f"result.shape:       {v.shape}")
        print(f"golden.shape:       {unpad_rope.shape}")
        print(f"mismatch count:     {mismatch_indices.shape[0]}")
        print(f"max abs diff:       {abs_diff.nan_to_num().max().item()}")
        print(f"mean abs diff:      {abs_diff.nan_to_num().mean().item()}")

        # 避免数据量过大，只打印前 20 个失败元素。
        max_print = 20
        print(f"\nFirst {min(max_print, mismatch_indices.shape[0])} mismatches:")

        for index in mismatch_indices[:max_print]:
            index_tuple = tuple(index.tolist())

            actual = v[index_tuple].item()
            expected = unpad_rope[index_tuple].item()
            diff = abs_diff[index_tuple].item()
            allowed = tolerance[index_tuple].item()

            print(
                f"index={index_tuple}, "
                f"result={actual}, "
                f"golden={expected}, "
                f"abs_diff={diff}, "
                f"tolerance={allowed}"
            )

        pytest.fail(
            f"padded_gather output mismatch: "
            f"{mismatch_indices.shape[0]} elements differ"
        )
