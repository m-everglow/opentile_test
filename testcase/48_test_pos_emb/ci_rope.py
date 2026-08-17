"""Triton/OpenTile fixed-length RoPE forward and backward."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.runtime import driver


@triton.jit
def rope_kernel(
    x,
    out,
    cos,
    sin,
    stride_xb,
    stride_xh,
    stride_xs,
    stride_xd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    stride_cs,
    batch: tl.constexpr,
    heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    inverse: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    programs = tl.num_programs(axis=0)
    tasks = batch * heads * seq_len
    half = head_dim // 2

    for task in range(pid, tasks, programs):
        seq = task % seq_len
        head = (task // seq_len) % heads
        batch_id = task // (seq_len * heads)
        dims = tl.arange(0, BLOCK_D)
        mask = dims < half

        base_x = batch_id * stride_xb + head * stride_xh + seq * stride_xs
        base_o = batch_id * stride_ob + head * stride_oh + seq * stride_os
        x1 = tl.load(x + base_x + dims * stride_xd, mask=mask, other=0.0)
        x2 = tl.load(x + base_x + (dims + half) * stride_xd, mask=mask, other=0.0)
        c = tl.load(cos + seq * stride_cs + dims, mask=mask, other=0.0)
        s = tl.load(sin + seq * stride_cs + dims, mask=mask, other=0.0)

        if inverse:
            y1 = x1 * c + x2 * s
            y2 = x2 * c - x1 * s
        else:
            y1 = x1 * c - x2 * s
            y2 = x2 * c + x1 * s

        tl.store(out + base_o + dims * stride_od, y1, mask=mask)
        tl.store(out + base_o + (dims + half) * stride_od, y2, mask=mask)


def _vector_cores() -> int:
    target = driver.active.get_current_target()
    backend = str(target.backend)
    if backend != "opentile" and not backend.startswith("opentile_"):
        raise RuntimeError(f"expected OpenTile, got {backend!r}")
    if driver.active.get_active_torch_device().type != "npu":
        raise RuntimeError("OpenTile did not select an NPU")
    props = driver.active.utils.get_device_properties(torch.npu.current_device())
    return int(props["num_vectorcore"])


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool) -> torch.Tensor:
    if x.ndim != 4 or x.shape[-1] % 2:
        raise ValueError("x must be [batch, heads, sequence, even_head_dim]")
    batch, heads, seq_len, head_dim = x.shape
    if cos.shape != (1, seq_len, head_dim) or sin.shape != cos.shape:
        raise ValueError("cos and sin must be [1, sequence, head_dim]")
    out = torch.full_like(x, float("nan"))
    block_d = triton.next_power_of_2(head_dim // 2)
    rope_kernel[(_vector_cores(),)](
        x,
        out,
        cos,
        sin,
        *x.stride(),
        *out.stride(),
        cos.stride(-2),
        batch,
        heads,
        seq_len,
        head_dim=head_dim,
        inverse=inverse,
        BLOCK_D=block_d,
    )
    return out
