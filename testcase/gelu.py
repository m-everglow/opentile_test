"""End-to-end launcher for the production Mojo GELU Triton kernel.

Kernel source:
  HighPriority50Operators/mojo_opset-master/
  mojo_opset/backends/ttx/kernels/npu/gelu.py
  SHA256 f8f159562e2dfe7e00b3087335338698dd728e4428cea0c4b4230af312d7992c

Only the runtime-only autotune/libentry decorators are omitted.  The helper
and forward-kernel bodies below are kept identical to the production source.
"""

import torch
import triton
import triton.language as tl

from triton.compiler import ASTSource
from triton.language.extra.cuda import libdevice


# The production source spells this as tl.math.tanh.  OpenTileConverter exposes
# the same Triton operation through libdevice.
tl.math.tanh = libdevice.tanh


@triton.jit
def gelu_tanh_approx(x):
    """GELU activation using tanh approximation"""
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / π)
    x_cubed = x * x * x
    tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    return 0.5 * x * (1 + tl.math.tanh(tanh_arg))


@triton.jit
def _gelu_fwd_kernel(
    x,
    y,
    stride_row,
    n_rows,
    n_cols,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            x_ptrs = x + rows_off[:, None] * stride_row + cols_off[None, :]
            y_ptrs = y + rows_off[:, None] * stride_row + cols_off[None, :]

            x_chunk = tl.load(x_ptrs, mask=block_mask, other=0.0)

            x_f32 = x_chunk.to(tl.float32)
            y_f32 = gelu_tanh_approx(x_f32)

            y_chunk = y_f32.to(x_chunk.dtype)

            tl.store(y_ptrs, y_chunk, mask=block_mask)


def _num_vector_cores():
    return triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]


def gelu_forward(x: torch.Tensor) -> torch.Tensor:
    """Compile and execute the production `_gelu_fwd_kernel` on OpenTile."""
    if x.dtype != torch.bfloat16:
        raise TypeError(f"GELU case requires torch.bfloat16, got {x.dtype}")

    original_shape = x.shape
    n_cols = original_shape[-1]
    x_2d = x.contiguous().reshape(-1, n_cols)
    n_rows = x_2d.shape[0]
    y = torch.empty_like(x_2d)

    source = ASTSource(
        fn=_gelu_fwd_kernel,
        signature={
            "x": "*bf16",
            "y": "*bf16",
            "stride_row": "i64",
            "n_rows": "i32",
            "n_cols": "i32",
            "BLOCK_SIZE_N": "constexpr",
            "BLOCK_SIZE_M": "constexpr",
        },
        constexprs={"BLOCK_SIZE_N": 128, "BLOCK_SIZE_M": 1},
    )
    compiled = triton.compile(source, options={"num_warps": 1, "num_stages": 1})
    grid_size = min(_num_vector_cores(), n_rows)
    compiled[(grid_size, 1, 1)](x_2d, y, x_2d.stride(0), n_rows, n_cols)
    return y.reshape(original_shape)

