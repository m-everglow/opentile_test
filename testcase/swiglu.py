"""End-to-end launcher for the production Mojo SwiGLU Triton kernel.

Kernel source:
  HighPriority50Operators/mojo_opset-master/
  mojo_opset/backends/ttx/kernels/npu/swiglu.py
  SHA256 dede713ea2302fd2bceb0a152a38b467a6b6f102caa2460cf3f9bcc1125447a4

Only the runtime-only autotune/libentry decorators are omitted.  The helper
and forward-kernel bodies below are kept identical to the production source.
"""

import torch
import triton
import triton.language as tl

from triton.compiler import ASTSource


@triton.jit
def silu(x):
    """SiLU activation function: x * sigmoid(x)"""
    return x * tl.sigmoid(x)


@triton.jit
def _swiglu_fwd_kernel(
    a,
    b,
    c,
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

            a_ptrs = a + rows_off[:, None] * stride_row + cols_off[None, :]
            b_ptrs = b + rows_off[:, None] * stride_row + cols_off[None, :]
            c_ptrs = c + rows_off[:, None] * stride_row + cols_off[None, :]

            a_chunk = tl.load(a_ptrs, mask=block_mask, other=0.0)
            b_chunk = tl.load(b_ptrs, mask=block_mask, other=0.0)

            a_f32 = a_chunk.to(tl.float32)
            silu_a = silu(a_f32)

            c_chunk = silu_a.to(a_chunk.dtype) * b_chunk

            tl.store(c_ptrs, c_chunk, mask=block_mask)


def _num_vector_cores():
    return triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]


def swiglu_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compile and execute the production `_swiglu_fwd_kernel` on OpenTile."""
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError(f"SwiGLU case requires torch.bfloat16, got {a.dtype} and {b.dtype}")
    if a.shape != b.shape:
        raise ValueError(f"SwiGLU inputs must have the same shape, got {a.shape} and {b.shape}")

    original_shape = a.shape
    n_cols = original_shape[-1]
    a_2d = a.contiguous().reshape(-1, n_cols)
    b_2d = b.contiguous().reshape(-1, n_cols)
    n_rows = a_2d.shape[0]
    c = torch.empty_like(a_2d)

    source = ASTSource(
        fn=_swiglu_fwd_kernel,
        signature={
            "a": "*bf16",
            "b": "*bf16",
            "c": "*bf16",
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
    compiled[(grid_size, 1, 1)](a_2d, b_2d, c, a_2d.stride(0), n_rows, n_cols)
    return c.reshape(original_shape)

