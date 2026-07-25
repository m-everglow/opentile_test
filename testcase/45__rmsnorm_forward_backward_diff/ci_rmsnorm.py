"""Focused Triton/OpenTile RMSNorm forward and backward kernels.

This is derived from mojo_opset's Ascend NPU implementation.  The initial
OpenTile scope deliberately keeps the non-atomic n_cols <= 2048 path.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.runtime import driver


@triton.jit
def rmsnorm_fwd_kernel(
    y_ptr,
    x_ptr,
    w_ptr,
    rstd_ptr,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, grid_size):
        rows = row_task_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows < n_rows
        x_rows = x_ptr + rows[:, None] * n_cols

        square_sum = tl.zeros((BLOCK_SIZE_M,), tl.float32)
        for col_start in range(0, n_cols, BLOCK_SIZE_N):
            cols = col_start + tl.arange(0, BLOCK_SIZE_N)
            mask = rows_mask[:, None] & (cols[None, :] < n_cols)
            x = tl.load(x_rows + cols[None, :], mask=mask, other=0.0).to(tl.float32)
            square_sum += tl.sum(x * x, axis=1)

        rstd = tl.rsqrt(square_sum / n_cols + eps)
        tl.store(rstd_ptr + rows, rstd, mask=rows_mask)

        y_rows = y_ptr + rows[:, None] * n_cols
        for col_start in range(0, n_cols, BLOCK_SIZE_N):
            cols = col_start + tl.arange(0, BLOCK_SIZE_N)
            mask = rows_mask[:, None] & (cols[None, :] < n_cols)
            x = tl.load(x_rows + cols[None, :], mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
            y = x * rstd[:, None] * w[None, :]
            tl.store(y_rows + cols[None, :], y, mask=mask)


@triton.jit
def rmsnorm_bwd_kernel(
    dy_ptr,
    dx_ptr,
    x_ptr,
    w_ptr,
    rstd_ptr,
    dw_partial_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    cols = tl.arange(0, BLOCK_SIZE_N)
    cols_mask = cols < n_cols
    w = tl.load(w_ptr + cols, mask=cols_mask, other=0.0).to(tl.float32)
    dw = tl.zeros((BLOCK_SIZE_N,), tl.float32)

    for row_task_id in range(pid, num_row_tasks, grid_size):
        rows = row_task_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows < n_rows
        mask = rows_mask[:, None] & cols_mask[None, :]
        offsets = rows[:, None] * n_cols + cols[None, :]

        dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + rows, mask=rows_mask, other=0.0)

        normed_x = x * rstd[:, None]
        m = dy * w[None, :]
        dot = tl.sum(m * x, axis=1)
        dx = rstd[:, None] * m
        dx = dx - (rstd * rstd * rstd)[:, None] * (dot / n_cols)[:, None] * x

        tl.store(dx_ptr + offsets, dx, mask=mask)
        dw += tl.sum(dy * normed_x, axis=0)

    tl.store(dw_partial_ptr + pid * n_cols + cols, dw, mask=cols_mask)


def _opentile_device_and_vector_cores() -> Tuple[torch.device, int]:
    target = driver.active.get_current_target()
    backend = str(target.backend)
    if backend != "opentile" and not backend.startswith("opentile_"):
        raise RuntimeError(f"expected an OpenTile target, got {backend!r}")

    device = driver.active.get_active_torch_device()
    if device.type != "npu":
        raise RuntimeError(f"expected OpenTile to select an NPU, got {device}")

    device_index = torch.npu.current_device()
    properties = driver.active.utils.get_device_properties(device_index)
    return torch.device("npu", device_index), int(properties["num_vectorcore"])


def _launch_config(n_cols: int) -> Tuple[int, int]:
    if n_cols > 2048:
        raise NotImplementedError(
            "the focused OpenTile test covers the non-atomic n_cols <= 2048 path"
        )
    block_n = triton.next_power_of_2(n_cols)
    block_m = max(1, min(4, 4096 // block_n))
    return block_n, block_m


def rmsnorm_forward(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, vector_cores = _opentile_device_and_vector_cores()
    if x.ndim != 2 or not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("the focused port requires contiguous 2-D input and weight")
    n_rows, n_cols = x.shape
    if weight.shape != (n_cols,) or weight.dtype != torch.float32:
        raise ValueError("weight must be float32 with shape (x.shape[-1],)")

    block_n, block_m = _launch_config(n_cols)
    y = torch.full_like(x, float("nan"))
    rstd = torch.full((n_rows,), float("nan"), dtype=torch.float32, device=x.device)
    rmsnorm_fwd_kernel[(vector_cores,)](
        y,
        x,
        weight,
        rstd,
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_M=block_m,
    )
    return y, rstd


def rmsnorm_backward(
    dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, rstd: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, vector_cores = _opentile_device_and_vector_cores()
    if dy.shape != x.shape or not dy.is_contiguous():
        raise ValueError("dy must be contiguous and have the same shape as x")
    n_rows, n_cols = x.shape
    block_n, block_m = _launch_config(n_cols)

    dx = torch.full_like(x, float("nan"))
    dw_partial = torch.zeros(
        (vector_cores, n_cols), dtype=torch.float32, device=weight.device
    )
    rmsnorm_bwd_kernel[(vector_cores,)](
        dy,
        dx,
        x,
        weight,
        rstd,
        dw_partial,
        n_rows,
        n_cols,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_M=block_m,
    )
    return dx, dw_partial.sum(dim=0)
