"""Triton/OpenTile RMSNorm inference kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.runtime import driver


@triton.jit
def rmsnorm_kernel(x, y, weight, n_rows, n_cols, eps, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    programs = tl.num_programs(axis=0)
    tasks = (n_rows + BLOCK_M - 1) // BLOCK_M
    for task in range(pid, tasks, programs):
        rows = task * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < n_rows
        square_sum = tl.zeros((BLOCK_M,), tl.float32)
        for start in range(0, n_cols, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            mask = row_mask[:, None] & (cols[None, :] < n_cols)
            values = tl.load(x + rows[:, None] * n_cols + cols[None, :], mask=mask, other=0.0).to(tl.float32)
            square_sum += tl.sum(values * values, axis=1)
        rrms = tl.rsqrt(square_sum / n_cols + eps)
        for start in range(0, n_cols, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            mask = row_mask[:, None] & (cols[None, :] < n_cols)
            values = tl.load(x + rows[:, None] * n_cols + cols[None, :], mask=mask, other=0.0).to(tl.float32)
            w = tl.load(weight + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
            tl.store(y + rows[:, None] * n_cols + cols[None, :], values * rrms[:, None] * w[None, :], mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    target = driver.active.get_current_target()
    backend = str(target.backend)
    if (backend != "opentile" and not backend.startswith("opentile_")) or driver.active.get_active_torch_device().type != "npu":
        raise RuntimeError("expected the OpenTile NPU route")
    rows, cols = x.shape
    props = driver.active.utils.get_device_properties(torch.npu.current_device())
    block_n = min(2048, triton.next_power_of_2(cols))
    block_m = max(1, min(4, 4096 // block_n))
    y = torch.full_like(x, float("nan"))
    rmsnorm_kernel[(int(props["num_vectorcore"]),)](x, y, weight, rows, cols, eps, BLOCK_N=block_n, BLOCK_M=block_m)
    return y
