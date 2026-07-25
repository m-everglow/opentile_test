import os
import torch
import torch_npu
import triton
import triton.language as tl
from triton.language.extra.cann import libdevice
# from utils import is_hopper, is_ampere
# import test_common

# def is_hopper():
#     return torch.cuda.is_available() and torch.cuda.get_device_properties(0).major == 9
#
# def get_configs():
#     if is_hopper():
#         return [triton.Config({"BLOCK_M": 32}, num_stages=2, num_warps=4)]
#     elif is_ampere():
#         return [triton.Config({"BLOCK_M": 32}, num_stages=1, num_warps=4)]
#     else:
#         return [triton.Config({"BLOCK_M": 16}, num_stages=3, num_warps=16)]
#
# if os.environ.get("TRITON_DEBUG") == "1":
#     configs = [triton.Config({"BLOCK_M": 4}, num_stages=1, num_warps=1)]

# @triton.autotune(get_configs(), key = ["DIM", "REVERSE"])
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": BM, "multibuffer": MB})
        for BM in [100, 75, 50, 25, 10, 5, 2]
        for MB in [True, False]
    ],
    key=["DIM", "REVERSE"]
)
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

            x0 = tl.load(x0_block_ptr, boundary_check=(0,))
            y0 = tl.load(y0_block_ptr, boundary_check=(0,))
            pos = tl.load(pos_block_ptr, boundary_check=(0,))

            offset_n = tl.arange(0, DIM // 2)
            # print("yty test")
            # print(base)
            inv_freq = libdevice.pow(base, -2.0 / DIM * offset_n)
            # inv_freq = tl.pow(base, -2.0 / DIM * offset_n)
            freqs = pos[:, None] * inv_freq[None, :]
            sin = tl.sin(freqs)
            cos = tl.cos(freqs)
            if REVERSE:
                sin = -sin
            x1 = x0 * cos[:, None, :] - y0 * sin[:, None, :]
            y1 = x0 * sin[:, None, :] + y0 * cos[:, None, :]
            dtype = in_ptr.type.element_ty
            tl.store(x1_block_ptr, x1.to(dtype), boundary_check=(0,))
            tl.store(y1_block_ptr, y1.to(dtype), boundary_check=(0,))

def rope_impl(input, position, offset, max_len, base = 10000., reverse = False):
    len, head, dim = input.size()
    out = input.new_empty(len, head, dim)
    # print(f"==========> {offset=}")
    bs = offset.size(0) - 1
    grid = (bs,)
    rope_kernel[grid](input, position, offset, out, head, base, dim, max_len, reverse)
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


if __name__ == "__main__":
  class RotaryPositionalEmbeddings(torch.nn.Module):
    def __init__(self, d: int, base: int = 10_000):
      super().__init__()
      self.base = base
      self.d = d
      self.cos_cached = None
      self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
      if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
        return
      seq_len = x.shape[0]
      theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device) # THETA = 10,000^(-2*i/d) or 1/10,000^(2i/d)
      seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device) #Position Index -> [0,1,2...seq-1]
      idx_theta = torch.einsum('n,d->nd', seq_idx, theta)  #Calculates m*(THETA) = [ [0, 0...], [THETA_1, THETA_2...THETA_d/2], ... [seq-1*(THETA_1), seq-1*(THETA_2)...] ]
      idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1) # [THETA_1, THETA_2...THETA_d/2] -> [THETA_1, THETA_2...THETA_d]
      self.cos_cached = idx_theta2.cos()[:, None, None, :] #Cache [cosTHETA_1, cosTHETA_2...cosTHETA_d]
      self.sin_cached = idx_theta2.sin()[:, None, None, :] #cache [sinTHETA_1, sinTHETA_2...sinTHETA_d]

    def _neg_half(self, x: torch.Tensor):
      d_2 = self.d // 2 #
      return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1) # [x_1, x_2,...x_d] -> [-x_d/2, ... -x_d, x_1, ... x_d/2]


    def forward(self, x: torch.Tensor):
      self._build_cache(x)
      neg_half_x = self._neg_half(x)
      print(x.size())
      x_rope = (x * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]]) # [x_1*cosTHETA_1 - x_d/2*sinTHETA_d/2, ....]
      return x_rope

  def pad(x, size, max_len):
      st = []
      for sz in size:
          y = x[:sz]
          x = x[sz:]
          y = torch.cat([y, torch.zeros(max_len - sz, y.size(1), y.size(2))])
          st += [y]
      return torch.stack(st)

  def unpad(x, size):
      st = []
      for i, sz in enumerate(size):
          y = x[i, :sz]
          st += [y]
      return torch.cat(st)

if __name__ == "__main__":
    DEVICE = torch.device("npu")
    MAX_LEN = 100
    BS = 100
    HEAD = 8
    DIM = 64
    torch.manual_seed(0)
    size = torch.randint(MAX_LEN  - 4, [BS]) + 4
    offset = torch.nn.functional.pad(torch.cumsum(size, 0), [1, 0])
    all_len = size.sum()
    input = torch.randn(all_len, HEAD, DIM)
    print(input.shape)
    pos = []
    for sz in size.cpu().numpy():
        pos += [torch.arange(sz)]
    pos = torch.cat(pos)

    v = rope_impl(input.to(DEVICE), pos.to(DEVICE), offset.to(DEVICE), MAX_LEN, base=2.)

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
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./prof_dir")
        ) as prof:
            # prof.start()
            for i in range(30):
                # fn_triton(*args)
                v = rope_impl(input.to(DEVICE), pos.to(DEVICE), offset.to(DEVICE), MAX_LEN, base=2.)
                torch.npu.synchronize()  # 确保 kernel 真正执行完
                prof.step()

    pad_input = pad(input, size, MAX_LEN)
    rope = RotaryPositionalEmbeddings(DIM, base=2)
    roped = rope(pad_input.transpose(0,1).to(DEVICE)).transpose(0,1)
    unpad_rope = unpad(roped, size)
    
    print(unpad_rope)
    print(v)

    # test_common.print_max_error(input, unpad_rope, v)

    print(torch.allclose(unpad_rope, v, rtol=1e-3, atol=1e-3))
    print(torch.abs(unpad_rope-v).max())