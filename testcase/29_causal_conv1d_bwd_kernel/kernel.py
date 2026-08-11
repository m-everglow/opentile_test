# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in LICENSE.

from __future__ import annotations

import torch
import triton
import triton.language as tl


W = 4
DEFAULT_BT = 64
_MAX_GRID = 65535
SUPPORTED_DTYPES = {
    torch.float32,
    torch.bfloat16,
    torch.float16,
}
SUPPORTED_SHAPES = {
    (1, 64, 64),
    (2, 65, 70),
}


@triton.jit
def causal_conv1d_bwd_kernel(
    x,
    y,
    weight,
    dy,
    dx,
    dw,
    db,
    T,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    stride_y_n,
    stride_y_t,
    stride_y_d,
    stride_dy_n,
    stride_dy_t,
    stride_dy_d,
    stride_dx_n,
    stride_dx_t,
    stride_dx_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    GRID_D: tl.constexpr,
    GRID_T: tl.constexpr,
    CHUNK_OFFSET: tl.constexpr,
    NT: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    linear_pid = tl.program_id(0)
    i_d = linear_pid % GRID_D
    remaining_pid = linear_pid // GRID_D
    i_t = remaining_pid % GRID_T + CHUNK_OFFSET
    i_b = remaining_pid // GRID_T
    i_tg = i_b * NT + i_t

    p_x = x + i_b.to(tl.int64) * stride_x_n
    p_y = y + i_b.to(tl.int64) * stride_y_n
    p_dy = dy + i_b.to(tl.int64) * stride_dy_n
    p_dx = dx + i_b.to(tl.int64) * stride_dx_n
    o_d = i_d * BD + tl.arange(0, BD)
    o_t = i_t * BT + tl.arange(0, BT)
    o_w = tl.arange(0, W)
    m_d = o_d < D
    m_t = o_t < T

    b_x = tl.load(
        p_x
        + o_t[:, None] * stride_x_t
        + o_d[None, :] * stride_x_d,
        mask=m_t[:, None] & m_d[None, :],
        other=0.0,
    ).to(tl.float32)
    b_w = tl.load(
        weight + o_d[:, None] * W + o_w[None, :],
        mask=m_d[:, None],
        other=0.0,
    ).to(tl.float32)
    b_dx = tl.zeros((BT, BD), dtype=tl.float32)
    b_db = tl.zeros((BD,), dtype=tl.float32)
    for i_w in tl.static_range(0, W):
        o_dy = o_t + i_w
        m_dy = (o_dy < T)[:, None] & m_d[None, :]
        b_dy = tl.load(
            p_dy
            + o_dy[:, None] * stride_dy_t
            + o_d[None, :] * stride_dy_d,
            mask=m_dy,
            other=0.0,
        ).to(tl.float32)
        if ACTIVATION == "silu":
            b_y = tl.load(
                p_y
                + o_dy[:, None] * stride_y_t
                + o_d[None, :] * stride_y_d,
                mask=m_dy,
                other=0.0,
            ).to(tl.float32)
            b_sigmoid = tl.sigmoid(b_y)
            b_dy *= b_sigmoid * (
                1.0 + b_y * (1.0 - b_sigmoid)
            )
        b_weight = tl.sum(
            b_w * (o_w == (W - i_w - 1))[None, :],
            axis=1,
        )
        b_dx += b_dy * b_weight[None, :]
        b_dw = tl.sum(b_dy * b_x, axis=0)
        tl.store(
            dw + i_tg * D * W + o_d * W + W - i_w - 1,
            b_dw,
            mask=m_d,
        )
        if i_w == 0:
            b_db += tl.sum(b_dy, axis=0)

    tl.store(
        p_dx
        + o_t[:, None] * stride_dx_t
        + o_d[None, :] * stride_dx_d,
        b_dx.to(dx.dtype.element_ty),
        mask=m_t[:, None] & m_d[None, :],
    )
    tl.store(
        db + i_tg * D + o_d,
        b_db,
        mask=m_d,
    )


def _chunk_size(time: int) -> int:
    if time in (1, 2, 4, 8, 16, 32, 64):
        return min(DEFAULT_BT, time)
    return min(triton.next_power_of_2(time), DEFAULT_BT)


def _bwd_tile_config(time: int, dim: int) -> tuple[int, int]:
    pow2_time = triton.next_power_of_2(time)
    floor_time = pow2_time if pow2_time == time else pow2_time // 2
    block_time = min(64, floor_time)
    block_dim = min(
        16,
        max(8, triton.next_power_of_2(dim)),
    )
    return block_dim, block_time


def _max_time_chunks(grid_dim: int, batch: int) -> int:
    denominator = grid_dim * batch
    if denominator > _MAX_GRID:
        raise RuntimeError(
            f"grid dim0*batch={denominator} exceeds {_MAX_GRID}"
        )
    return max(1, _MAX_GRID // max(denominator, 1))


def _launch_bwd(
    x: torch.Tensor,
    y_pre: torch.Tensor | None,
    weight: torch.Tensor,
    dy: torch.Tensor,
    bias: torch.Tensor,
    activation: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, time, dim = x.shape
    block_dim, block_time = _bwd_tile_config(time, dim)
    num_time_chunks = triton.cdiv(time, block_time)
    grid_dim = triton.cdiv(dim, block_dim)
    max_time_chunks = _max_time_chunks(grid_dim, batch)
    partial_count = batch * num_time_chunks
    dx = torch.full_like(x, float("nan"))
    dw_partial = torch.full(
        (partial_count, dim, W),
        float("nan"),
        dtype=torch.float32,
        device=x.device,
    )
    db_partial = torch.full(
        (partial_count, dim),
        float("nan"),
        dtype=torch.float32,
        device=x.device,
    )
    y = x if y_pre is None else y_pre

    for chunk_offset in range(0, num_time_chunks, max_time_chunks):
        chunk_count = min(
            max_time_chunks,
            num_time_chunks - chunk_offset,
        )
        causal_conv1d_bwd_kernel[
            (grid_dim * chunk_count * batch,)
        ](
            x=x,
            y=y,
            weight=weight,
            dy=dy,
            dx=dx,
            dw=dw_partial,
            db=db_partial,
            T=time,
            stride_x_n=x.stride(0),
            stride_x_t=x.stride(1),
            stride_x_d=x.stride(2),
            stride_y_n=y.stride(0),
            stride_y_t=y.stride(1),
            stride_y_d=y.stride(2),
            stride_dy_n=dy.stride(0),
            stride_dy_t=dy.stride(1),
            stride_dy_d=dy.stride(2),
            stride_dx_n=dx.stride(0),
            stride_dx_t=dx.stride(1),
            stride_dx_d=dx.stride(2),
            D=dim,
            W=W,
            BT=block_time,
            BD=block_dim,
            GRID_D=grid_dim,
            GRID_T=chunk_count,
            CHUNK_OFFSET=chunk_offset,
            NT=num_time_chunks,
            ACTIVATION=activation,
        )

    dw = dw_partial.sum(dim=0).to(weight.dtype)
    db = db_partial.sum(dim=0).to(bias.dtype)
    return dx, dw, db


def active_opentile_npu() -> tuple[torch.device, int, int]:
    target = triton.runtime.driver.active.get_current_target()
    backend = str(target.backend)
    if backend != "opentile" and not backend.startswith("opentile_"):
        raise RuntimeError(f"expected OpenTile backend, got {backend!r}")

    device = torch.device(
        triton.runtime.driver.active.get_active_torch_device()
    )
    if device.type != "npu":
        raise RuntimeError(f"expected a physical NPU, got {device}")
    device_index = device.index
    if device_index is None:
        device_index = int(torch.npu.current_device())
    properties = triton.runtime.driver.active.utils.get_device_properties(
        device_index
    )
    ai_cores = int(properties["num_aicore"])
    vector_cores = int(properties["num_vectorcore"])
    if ai_cores <= 0 or vector_cores <= 0:
        raise RuntimeError(
            "target must report positive AI Core and Vector Core counts"
        )
    return device, ai_cores, vector_cores


def _validate_inputs(
    x: torch.Tensor,
    y_pre: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor,
    dy: torch.Tensor,
    activation: str | None,
) -> None:
    if activation not in (None, "silu"):
        raise ValueError(
            f"activation must be None or 'silu', got {activation!r}"
        )
    if x.device.type != "npu":
        raise ValueError(f"x must be on an NPU, got {x.device}")
    tensors = (weight, bias, dy)
    if y_pre is not None:
        tensors += (y_pre,)
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("all inputs must be on the same NPU")
    if x.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"unsupported dtype: {x.dtype}")
    if any(tensor.dtype != x.dtype for tensor in tensors):
        raise TypeError("all inputs must have the same dtype")
    if x.ndim != 3:
        raise ValueError(f"x must have shape [B,T,D], got {x.shape}")
    if tuple(x.shape) not in SUPPORTED_SHAPES:
        raise ValueError(
            f"unsupported shape {tuple(x.shape)}; "
            f"expected one of {sorted(SUPPORTED_SHAPES)}"
        )
    batch, time, dim = x.shape
    if dy.shape != (batch, time, dim):
        raise ValueError("dy shape must match x")
    if weight.shape != (dim, W):
        raise ValueError(f"weight must have shape ({dim},{W})")
    if bias.shape != (dim,):
        raise ValueError(f"bias must have shape ({dim},)")
    if activation == "silu":
        if y_pre is None or y_pre.shape != x.shape:
            raise ValueError("silu requires y_pre with the same shape as x")
    elif y_pre is not None:
        raise ValueError("y_pre must be None when activation is None")
    if not all(
        tensor.is_contiguous()
        for tensor in (x, weight, bias, dy)
    ):
        raise ValueError("x, weight, bias, and dy must be contiguous")
    if y_pre is not None and not y_pre.is_contiguous():
        raise ValueError("y_pre must be contiguous")


def causal_conv1d_bwd(
    *,
    x: torch.Tensor,
    y_pre: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor,
    dy: torch.Tensor,
    activation: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_inputs(x, y_pre, weight, bias, dy, activation)
    return _launch_bwd(
        x=x,
        y_pre=y_pre,
        weight=weight,
        dy=dy,
        bias=bias,
        activation=activation,
    )