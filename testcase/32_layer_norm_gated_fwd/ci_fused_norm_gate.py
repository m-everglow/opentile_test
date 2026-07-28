# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in this directory.
# For a list of all upstream contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Standalone fused norm + gate kernels for Triton/OpenTile on Ascend NPU.

The implementation starts from FLA's triton-ascend fused_norm_gate backend.
It keeps the one-dimensional grid-stride schedule and explicit pointer
arithmetic, while freezing a conservative one-row tile for OpenTile bring-up.
The executable validation case focuses on LayerNorm + affine + SiLU gated
forward on a physical Ascend NPU.
"""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401 -- registers torch.npu
import triton
import triton.language as tl
import triton.runtime.driver as driver


_ACTIVATION_SILU = 0
_ACTIVATION_SIGMOID = 1
_MAX_FEATURE_SIZE = 1024
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _activation_id(activation: str) -> int:
    if activation in ("swish", "silu"):
        return _ACTIVATION_SILU
    if activation == "sigmoid":
        return _ACTIVATION_SIGMOID
    raise ValueError(f"Unsupported activation: {activation}")


def assert_opentile_backend() -> str:
    """Return the active OpenTile backend name or reject a wrong route.

    OpenTile revisions expose either the legacy aggregate name ``opentile`` or
    a target-specific ``opentile_*`` name.  The active torch device is the
    authoritative discriminator between an Ascend and non-Ascend target.
    """
    target = driver.active.get_current_target()
    backend = str(getattr(target, "backend", ""))
    if backend != "opentile" and not backend.startswith("opentile_"):
        raise RuntimeError(
            "fused_norm_gate requires a Triton/OpenTile target, "
            f"got {backend!r}"
        )
    active_device = driver.active.get_active_torch_device()
    if active_device.type != "npu":
        raise RuntimeError(f"OpenTile selected an unexpected torch device: {active_device}")
    return backend


def get_npu_properties(device_index: int | None = None) -> dict:
    """Return and validate the active device's AI/vector core properties."""
    if device_index is None:
        device_index = torch.npu.current_device()
    properties = dict(driver.active.utils.get_device_properties(device_index))
    for name in ("num_aicore", "num_vectorcore"):
        if name not in properties or int(properties[name]) <= 0:
            raise RuntimeError(
                f"Triton driver returned invalid {name!r} for npu:{device_index}: {properties}"
            )
    return properties


@triton.jit
def layer_norm_gated_fwd_kernel(
    x,
    g,
    y,
    w,
    b,
    residual,
    residual_out,
    mean,
    rstd,
    eps,
    T,
    NS,
    D: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    program = tl.program_id(0)
    cols = tl.arange(0, BD)
    col_mask = cols < D
    cols_i64 = cols.to(tl.int64)

    if HAS_WEIGHT:
        values_w = tl.load(w + cols_i64, mask=col_mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        values_b = tl.load(b + cols_i64, mask=col_mask, other=0.0).to(tl.float32)

    for row_id in range(program, T, NS):
        row = row_id.to(tl.int64)
        offsets = row * D + cols_i64
        values_x = tl.load(x + offsets, mask=col_mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            values_x += tl.load(residual + offsets, mask=col_mask, other=0.0).to(tl.float32)
        if STORE_RESIDUAL_OUT:
            tl.store(
                residual_out + offsets,
                values_x.to(residual_out.dtype.element_ty),
                mask=col_mask,
            )

        if not IS_RMS_NORM:
            value_mean = tl.sum(values_x, axis=0) / D
            tl.store(mean + row, value_mean)
            values_centered = tl.where(col_mask, values_x - value_mean, 0.0)
            value_var = tl.sum(values_centered * values_centered, axis=0) / D
        else:
            values_centered = tl.where(col_mask, values_x, 0.0)
            value_var = tl.sum(values_centered * values_centered, axis=0) / D

        value_rstd = 1.0 / tl.sqrt(value_var + eps)
        tl.store(rstd + row, value_rstd)

        if not IS_RMS_NORM:
            values_norm = (values_x - value_mean) * value_rstd
        else:
            values_norm = values_x * value_rstd
        values_y = values_norm * values_w if HAS_WEIGHT else values_norm
        if HAS_BIAS:
            values_y += values_b

        values_g = tl.load(g + offsets, mask=col_mask, other=0.0).to(tl.float32)
        values_sigmoid = tl.sigmoid(values_g)
        if ACTIVATION == 0:
            values_y *= values_g * values_sigmoid
        else:
            values_y *= values_sigmoid

        tl.store(y + offsets, values_y.to(y.dtype.element_ty), mask=col_mask)


@triton.jit
def layer_norm_gated_bwd_kernel(
    x,
    g,
    w,
    b,
    y,
    dy,
    dx,
    dg,
    dw_partial,
    db_partial,
    dresidual,
    dresidual_in,
    mean,
    rstd,
    T,
    NS,
    D: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    program = tl.program_id(0)
    cols = tl.arange(0, BD)
    col_mask = cols < D
    cols_i64 = cols.to(tl.int64)

    if HAS_WEIGHT:
        values_w = tl.load(w + cols_i64, mask=col_mask, other=0.0).to(tl.float32)
        values_dw = tl.zeros((BD,), dtype=tl.float32)
    if HAS_BIAS:
        values_b = tl.load(b + cols_i64, mask=col_mask, other=0.0).to(tl.float32)
        values_db = tl.zeros((BD,), dtype=tl.float32)

    for row_id in range(program, T, NS):
        row = row_id.to(tl.int64)
        offsets = row * D + cols_i64
        values_x = tl.load(x + offsets, mask=col_mask, other=0.0).to(tl.float32)
        if not IS_RMS_NORM:
            value_mean = tl.load(mean + row)
        value_rstd = tl.load(rstd + row)
        if not IS_RMS_NORM:
            values_xhat = (values_x - value_mean) * value_rstd
        else:
            values_xhat = values_x * value_rstd
        values_xhat = tl.where(col_mask, values_xhat, 0.0)

        values_ungated = values_xhat * values_w if HAS_WEIGHT else values_xhat
        if HAS_BIAS:
            values_ungated += values_b
        if RECOMPUTE_OUTPUT:
            tl.store(y + offsets, values_ungated.to(y.dtype.element_ty), mask=col_mask)

        values_g = tl.load(g + offsets, mask=col_mask, other=0.0).to(tl.float32)
        values_dy = tl.load(dy + offsets, mask=col_mask, other=0.0).to(tl.float32)
        values_sigmoid = tl.sigmoid(values_g)
        if ACTIVATION == 0:
            values_gate_grad = values_sigmoid * (1.0 + values_g * (1.0 - values_sigmoid))
            values_dg = values_dy * values_ungated * values_gate_grad
            values_dy *= values_g * values_sigmoid
        else:
            values_dg = values_dy * values_ungated * values_sigmoid * (1.0 - values_sigmoid)
            values_dy *= values_sigmoid
        tl.store(dg + offsets, values_dg.to(dg.dtype.element_ty), mask=col_mask)

        if HAS_WEIGHT:
            values_dw += tl.where(col_mask, values_dy * values_xhat, 0.0)
            values_wdy = values_dy * values_w
        else:
            values_wdy = values_dy
        if HAS_BIAS:
            values_db += tl.where(col_mask, values_dy, 0.0)

        if not IS_RMS_NORM:
            value_c1 = tl.sum(values_xhat * values_wdy, axis=0) / D
            value_c2 = tl.sum(values_wdy, axis=0) / D
            values_dx = (
                values_wdy - (values_xhat * value_c1 + value_c2)
            ) * value_rstd
        else:
            value_c1 = tl.sum(values_xhat * values_wdy, axis=0) / D
            values_dx = (values_wdy - values_xhat * value_c1) * value_rstd

        if HAS_DRESIDUAL:
            values_dx += tl.load(
                dresidual + offsets,
                mask=col_mask,
                other=0.0,
            ).to(tl.float32)
        if STORE_DRESIDUAL:
            tl.store(
                dresidual_in + offsets,
                values_dx.to(dresidual_in.dtype.element_ty),
                mask=col_mask,
            )
        tl.store(dx + offsets, values_dx.to(dx.dtype.element_ty), mask=col_mask)

    partial_base = program.to(tl.int64) * D + cols_i64
    if HAS_WEIGHT:
        tl.store(dw_partial + partial_base, values_dw, mask=col_mask)
    if HAS_BIAS:
        tl.store(db_partial + partial_base, values_db, mask=col_mask)


@triton.jit
def reduce_norm_parameter_grads_kernel(
    partial,
    output,
    NS,
    D: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    rows = tl.arange(0, BN)
    cols = tl.arange(0, BD)
    mask = (rows[:, None] < NS) & (cols[None, :] < D)
    offsets = rows[:, None].to(tl.int64) * D + cols[None, :].to(tl.int64)
    values = tl.load(partial + offsets, mask=mask, other=0.0).to(tl.float32)
    reduced = tl.sum(values, axis=0)
    tl.store(
        output + cols.to(tl.int64),
        reduced.to(output.dtype.element_ty),
        mask=cols < D,
    )


def _validate_main_inputs(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
) -> tuple[int, int]:
    if x.device.type != "npu":
        raise ValueError(f"x must be on an NPU device, got {x.device}")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"unsupported x dtype: {x.dtype}")
    if x.ndim < 2 or x.numel() == 0:
        raise ValueError(f"x must be a non-empty tensor with at least 2 dimensions, got {x.shape}")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if g.shape != x.shape or g.device != x.device or g.dtype != x.dtype:
        raise ValueError("g must have the same shape, device, and dtype as x")
    if not g.is_contiguous():
        raise ValueError("g must be contiguous")

    feature_size = x.shape[-1]
    if feature_size <= 0 or feature_size > _MAX_FEATURE_SIZE:
        raise ValueError(
            f"feature size must be in [1, {_MAX_FEATURE_SIZE}], got {feature_size}"
        )
    rows = x.numel() // feature_size

    for name, value in (("weight", weight), ("bias", bias)):
        if value is None:
            continue
        if value.shape != (feature_size,):
            raise ValueError(f"{name} must have shape ({feature_size},), got {value.shape}")
        if value.device != x.device or value.dtype != x.dtype or not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous and match x device/dtype")
    if residual is not None:
        if residual.shape != x.shape or residual.device != x.device:
            raise ValueError("residual must have the same shape and device as x")
        if residual.dtype not in _SUPPORTED_DTYPES or not residual.is_contiguous():
            raise ValueError("residual must be contiguous and use a supported floating dtype")
    return rows, feature_size


def _validate_output(
    name: str,
    output: torch.Tensor,
    shape: torch.Size | tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if output.shape != shape or output.device != device or output.dtype != dtype:
        raise ValueError(
            f"{name} must have shape={tuple(shape)}, device={device}, dtype={dtype}; "
            f"got shape={tuple(output.shape)}, device={output.device}, dtype={output.dtype}"
        )
    if not output.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _launch_shape(x: torch.Tensor, rows: int, feature_size: int) -> tuple[int, int, int]:
    assert_opentile_backend()
    properties = get_npu_properties(x.device.index)
    programs = min(rows, int(properties["num_vectorcore"]))
    block_feature = triton.next_power_of_2(feature_size)
    return programs, feature_size, block_feature


def layer_norm_gated_fwd(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None = None,
    activation: str = "swish",
    eps: float = 1e-5,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    residual_dtype: torch.dtype | None = None,
    is_rms_norm: bool = False,
    *,
    out: torch.Tensor | None = None,
    mean_out: torch.Tensor | None = None,
    rstd_out: torch.Tensor | None = None,
    residual_out_buffer: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Run fused LayerNorm/RMSNorm followed by SiLU or sigmoid gating.

    The OpenTile validation contract covers FP32, BF16, and FP16 LayerNorm +
    SiLU with weight and bias, without residual. Optional buffers are exposed
    so the focused test can poison outputs and prove complete writes.
    """
    rows, feature_size = _validate_main_inputs(x, g, weight, bias, residual)
    activation_id = _activation_id(activation)
    output_dtype = x.dtype if out_dtype is None else out_dtype
    if output_dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"unsupported output dtype: {output_dtype}")

    if out is None:
        out = torch.empty_like(x, dtype=output_dtype)
    _validate_output("out", out, x.shape, x.device, output_dtype)

    if is_rms_norm:
        if mean_out is not None:
            raise ValueError("mean_out must be None for RMSNorm")
        mean = None
    else:
        mean = (
            torch.empty((rows,), dtype=torch.float32, device=x.device)
            if mean_out is None
            else mean_out
        )
        _validate_output("mean_out", mean, (rows,), x.device, torch.float32)

    rstd = (
        torch.empty((rows,), dtype=torch.float32, device=x.device)
        if rstd_out is None
        else rstd_out
    )
    _validate_output("rstd_out", rstd, (rows,), x.device, torch.float32)

    if residual is not None:
        residual_dtype = residual.dtype
    needs_residual_out = residual is not None or (
        residual_dtype is not None and residual_dtype != x.dtype
    )
    if needs_residual_out:
        target_residual_dtype = x.dtype if residual_dtype is None else residual_dtype
        residual_out = (
            torch.empty_like(x, dtype=target_residual_dtype)
            if residual_out_buffer is None
            else residual_out_buffer
        )
        _validate_output(
            "residual_out_buffer",
            residual_out,
            x.shape,
            x.device,
            target_residual_dtype,
        )
    else:
        if residual_out_buffer is not None:
            raise ValueError("residual_out_buffer is unused without residual or dtype conversion")
        residual_out = None

    programs, _, block_feature = _launch_shape(x, rows, feature_size)
    x_2d = x.view(rows, feature_size)
    g_2d = g.view(rows, feature_size)
    out_2d = out.view(rows, feature_size)
    residual_2d = residual.view(rows, feature_size) if residual is not None else x_2d
    residual_out_2d = (
        residual_out.view(rows, feature_size) if residual_out is not None else out_2d
    )
    weight_arg = weight if weight is not None else x_2d
    bias_arg = bias if bias is not None else x_2d
    mean_arg = mean if mean is not None else rstd

    layer_norm_gated_fwd_kernel[(programs,)](
        x=x_2d,
        g=g_2d,
        y=out_2d,
        w=weight_arg,
        b=bias_arg,
        residual=residual_2d,
        residual_out=residual_out_2d,
        mean=mean_arg,
        rstd=rstd,
        eps=eps,
        T=rows,
        NS=programs,
        D=feature_size,
        BD=block_feature,
        ACTIVATION=activation_id,
        IS_RMS_NORM=is_rms_norm,
        STORE_RESIDUAL_OUT=residual_out is not None,
        HAS_RESIDUAL=residual is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
    )
    saved_x = residual_out if residual_out is not None else x
    return out, mean, rstd, saved_x


def layer_norm_gated_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None = None,
    activation: str = "swish",
    eps: float = 1e-5,
    mean: torch.Tensor | None = None,
    rstd: torch.Tensor | None = None,
    dresidual: torch.Tensor | None = None,
    has_residual: bool = False,
    is_rms_norm: bool = False,
    x_dtype: torch.dtype | None = None,
    recompute_output: bool = False,
    *,
    dx_out: torch.Tensor | None = None,
    dg_out: torch.Tensor | None = None,
    dw_out: torch.Tensor | None = None,
    db_out: torch.Tensor | None = None,
) -> tuple:
    """Run the standalone backward path for the admitted fused norm contract."""
    del eps  # rstd already captures epsilon from forward
    if has_residual:
        raise NotImplementedError(
            "the standalone backward bring-up does not admit residual/prenorm"
        )
    if x_dtype is not None and x_dtype != x.dtype:
        raise NotImplementedError(
            "the standalone backward bring-up requires x_dtype to match saved x"
        )
    rows, feature_size = _validate_main_inputs(x, g, weight, bias, None)
    if dy.shape != x.shape or dy.device != x.device or dy.dtype != x.dtype:
        raise ValueError("dy must have the same shape, device, and dtype as x")
    if not dy.is_contiguous():
        raise ValueError("dy must be contiguous")
    if rstd is None:
        raise ValueError("rstd from layer_norm_gated_fwd is required")
    _validate_output("rstd", rstd, (rows,), x.device, torch.float32)
    if is_rms_norm:
        if mean is not None:
            raise ValueError("mean must be None for RMSNorm")
    else:
        if mean is None:
            raise ValueError("mean from layer_norm_gated_fwd is required for LayerNorm")
        _validate_output("mean", mean, (rows,), x.device, torch.float32)
    if dresidual is not None:
        _validate_output("dresidual", dresidual, x.shape, x.device, x.dtype)

    output_dtype = x.dtype if x_dtype is None else x_dtype
    if output_dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"unsupported gradient dtype: {output_dtype}")
    dx = torch.empty_like(x, dtype=output_dtype) if dx_out is None else dx_out
    dg = torch.empty_like(g, dtype=output_dtype) if dg_out is None else dg_out
    _validate_output("dx_out", dx, x.shape, x.device, output_dtype)
    _validate_output("dg_out", dg, g.shape, g.device, output_dtype)

    if weight is not None:
        dw = torch.empty_like(weight) if dw_out is None else dw_out
        _validate_output("dw_out", dw, weight.shape, weight.device, weight.dtype)
    else:
        if dw_out is not None:
            raise ValueError("dw_out is unused when weight is None")
        dw = None
    if bias is not None:
        db = torch.empty_like(bias) if db_out is None else db_out
        _validate_output("db_out", db, bias.shape, bias.device, bias.dtype)
    else:
        if db_out is not None:
            raise ValueError("db_out is unused when bias is None")
        db = None

    programs, _, block_feature = _launch_shape(x, rows, feature_size)
    partial_dw = (
        torch.empty((programs, feature_size), dtype=torch.float32, device=x.device)
        if weight is not None
        else rstd
    )
    partial_db = (
        torch.empty((programs, feature_size), dtype=torch.float32, device=x.device)
        if bias is not None
        else rstd
    )
    recomputed = torch.empty_like(dy) if recompute_output else None
    dresidual_in = (
        torch.empty_like(x, dtype=output_dtype)
        if has_residual and output_dtype != x.dtype
        else None
    )

    x_2d = x.view(rows, feature_size)
    g_2d = g.view(rows, feature_size)
    dy_2d = dy.view(rows, feature_size)
    dx_2d = dx.view(rows, feature_size)
    dg_2d = dg.view(rows, feature_size)
    weight_arg = weight if weight is not None else x_2d
    bias_arg = bias if bias is not None else x_2d
    recomputed_arg = (
        recomputed.view(rows, feature_size) if recomputed is not None else dy_2d
    )
    dresidual_arg = (
        dresidual.view(rows, feature_size) if dresidual is not None else dx_2d
    )
    dresidual_in_arg = (
        dresidual_in.view(rows, feature_size) if dresidual_in is not None else dx_2d
    )
    mean_arg = mean if mean is not None else rstd

    layer_norm_gated_bwd_kernel[(programs,)](
        x=x_2d,
        g=g_2d,
        w=weight_arg,
        b=bias_arg,
        y=recomputed_arg,
        dy=dy_2d,
        dx=dx_2d,
        dg=dg_2d,
        dw_partial=partial_dw,
        db_partial=partial_db,
        dresidual=dresidual_arg,
        dresidual_in=dresidual_in_arg,
        mean=mean_arg,
        rstd=rstd,
        T=rows,
        NS=programs,
        D=feature_size,
        BD=block_feature,
        ACTIVATION=_activation_id(activation),
        IS_RMS_NORM=is_rms_norm,
        STORE_DRESIDUAL=dresidual_in is not None,
        HAS_DRESIDUAL=dresidual is not None,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
        RECOMPUTE_OUTPUT=recomputed is not None,
    )

    block_programs = triton.next_power_of_2(programs)
    if dw is not None:
        reduce_norm_parameter_grads_kernel[(1,)](
            partial_dw,
            dw,
            programs,
            D=feature_size,
            BN=block_programs,
            BD=block_feature,
        )
    if db is not None:
        reduce_norm_parameter_grads_kernel[(1,)](
            partial_db,
            db,
            programs,
            D=feature_size,
            BN=block_programs,
            BD=block_feature,
        )

    if has_residual and output_dtype == x.dtype:
        dresidual_in = dx
    result = (dx, dg, dw, db, dresidual_in)
    return result if recomputed is None else (*result, recomputed)
