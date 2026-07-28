"""Standalone NPU SiLU kernel + kernel-faithful golden.

Run directly:
  python3 test_silu_forward_backward_diff.py

Run with pytest:
  pytest -s test_silu_forward_backward_diff.py
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_BACKEND", "opentile")
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")

import torch
import triton
import triton.language as tl

SEED = 2026071840
SHAPES = ((128, 128), (999, 9999), (1024, 10240))
DTYPES = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
}
TOLERANCES = {
    "f32": (1e-5, 1e-5),
    "f16": (1e-3, 1e-3),
    "bf16": (5e-3, 5e-3),
}
VEC_ALIGN_BYTES = 256
COL_BLOCKING_THRESHOLD = 4096


@triton.jit
def silu_activation(x):
    return x * tl.sigmoid(x)


@triton.jit
def _silu_fwd_kernel(
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
            y_f32 = silu_activation(x_f32)
            y_chunk = y_f32.to(x_chunk.dtype)
            tl.store(y_ptrs, y_chunk, mask=block_mask)


@triton.jit
def _silu_bwd_kernel(
    dy,
    x,
    dx,
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
            dy_ptrs = dy + rows_off[:, None] * stride_row + cols_off[None, :]
            x_ptrs = x + rows_off[:, None] * stride_row + cols_off[None, :]
            dx_ptrs = dx + rows_off[:, None] * stride_row + cols_off[None, :]
            dy_chunk = tl.load(dy_ptrs, mask=block_mask, other=0.0)
            x_chunk = tl.load(x_ptrs, mask=block_mask, other=0.0)
            x_f32 = x_chunk.to(tl.float32)
            sigmoid_x = tl.sigmoid(x_f32)
            dsilu_dx = sigmoid_x * (1 + x_f32 * (1 - sigmoid_x))
            dx_chunk = dy_chunk * dsilu_dx.to(dy_chunk.dtype)
            tl.store(dx_ptrs, dx_chunk, mask=block_mask)

def _num_vector_cores():
    return triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]

def _setup_npu() -> torch.device:
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("torch_npu is required on the real-hardware host") from exc
    device_id = int(os.environ.get("OPENTILE_TEST_DEVICE", "0"))
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is false")
    torch.npu.set_device(device_id)
    return torch.device(f"npu:{device_id}")


def _block_size_n(tensor: torch.Tensor, n_cols: int) -> int:
    if n_cols > COL_BLOCKING_THRESHOLD:
        return 2048
    row_bytes = tensor.element_size() * n_cols
    aligned_bytes = ((row_bytes + VEC_ALIGN_BYTES - 1) // VEC_ALIGN_BYTES) * VEC_ALIGN_BYTES
    return aligned_bytes // tensor.element_size()


def _compare(name: str, actual: torch.Tensor, golden: torch.Tensor, dtype: str) -> None:
    atol, rtol = TOLERANCES[dtype]
    actual_f32 = actual.detach().cpu().float()
    golden_f32 = golden.detach().cpu().float()
    absolute = (actual_f32 - golden_f32).abs()
    tolerance = atol + rtol * golden_f32.abs()
    passed = absolute <= tolerance
    print(
        f"{name} dtype={dtype} shape={tuple(actual.shape)} "
        f"pass={int(passed.sum())}/{passed.numel()} bad={int((~passed).sum())} "
        f"max_abs={float(absolute.max()):.9g} mean_abs={float(absolute.mean()):.9g} "
        f"atol={atol} rtol={rtol}"
    )
    torch.testing.assert_close(actual_f32, golden_f32, atol=atol, rtol=rtol)


def run_one(dtype_name: str, shape: tuple[int, int], device: torch.device) -> None:
    torch_dtype = DTYPES[dtype_name]
    rows, cols = shape

    # Generate on CPU exactly as the current packaged real-hardware case does.
    torch.manual_seed(SEED)
    x_cpu = torch.rand(
        rows,
        cols,
        dtype=torch_dtype,
        device="cpu",
        requires_grad=True,
    )
    with torch.no_grad():
        x_f32 = x_cpu.float()
        sigmoid_f32 = torch.sigmoid(x_f32)
        y_golden = (x_f32 * sigmoid_f32).to(torch_dtype)
        dy_cpu = torch.rand_like(y_golden)
        derivative_f32 = sigmoid_f32 * (1.0 + x_f32 * (1.0 - sigmoid_f32))
        dx_golden = dy_cpu * derivative_f32.to(torch_dtype)

    x = x_cpu.detach().to(device)
    dy = dy_cpu.to(device)
    y = torch.empty_like(x)
    dx = torch.empty_like(x)
    block_size_n = _block_size_n(x, cols)
    block_dim = min(_num_vector_cores(), rows)
    grid = (block_dim,)

    _silu_fwd_kernel[grid](
        x,
        y,
        x.stride(0),
        rows,
        cols,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_M=1,
    )
    torch.npu.synchronize()
    _compare(f"Y_{dtype_name}_{rows}x{cols}", y, y_golden, dtype_name)

    _silu_bwd_kernel[grid](
        dy,
        x,
        dx,
        dy.stride(0),
        rows,
        cols,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_M=1,
    )
    torch.npu.synchronize()
    _compare(f"DX_{dtype_name}_{rows}x{cols}", dx, dx_golden, dtype_name)
    del x, dy, y, dx, x_cpu, dy_cpu, y_golden, dx_golden
    torch.npu.empty_cache()


def test_silu_forward_backward_diff() -> None:
    device = _setup_npu()
    for dtype_name in ("f32", "f16", "bf16"):
        for shape in SHAPES:
            run_one(dtype_name, shape, device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("all", *DTYPES), default="all")
    parser.add_argument(
        "--shape",
        choices=("all", *(f"{rows}x{cols}" for rows, cols in SHAPES)),
        default="all",
    )
    args = parser.parse_args()
    device = _setup_npu()
    dtype_names = tuple(DTYPES) if args.dtype == "all" else (args.dtype,)
    shapes = (
        SHAPES
        if args.shape == "all"
        else (tuple(int(value) for value in args.shape.split("x")),)
    )
    for dtype_name in dtype_names:
        for shape in shapes:
            run_one(dtype_name, shape, device)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
