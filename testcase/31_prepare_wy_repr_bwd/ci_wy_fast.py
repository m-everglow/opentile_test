# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Focused prepare_wy_repr_bwd kernels for Triton/OpenTile on Ascend NPU.

This standalone port starts from FLA's exact
``fla/ops/gated_delta_rule/backends/triton_ascend/wy_fast.py`` baseline.
It preserves the split-kernel/FP32-scratch algorithm, multidimensional launch,
grouped-head reduction, and fixed/variable-length semantics. OpenTile-specific
changes are limited to runtime routing, diagnostics, simulator-safe allocation,
and integer widths required by the block-pointer frontend.
"""

from __future__ import annotations

import ctypes
import os
import time
import warnings

import torch
import torch_npu  # noqa: F401 -- registers the torch.npu device
import triton
import triton.language as tl
import triton.runtime.driver as driver


# Keep every kernel tensor on the explicit NPU device.  Triton discovers the
# installed active driver from the visible runtime; this Python input does not
# force a backend through environment overrides.
DEVICE = torch.device("npu")


def assert_opentile_backend() -> str:
    """Return the active OpenTile backend name or reject a wrong route."""
    target = driver.active.get_current_target()
    backend = str(getattr(target, "backend", ""))
    if backend != "opentile" and not backend.startswith("opentile_"):
        raise RuntimeError(
            "prepare_wy_repr_bwd requires a Triton/OpenTile target, "
            f"got {backend!r}"
        )
    active_device = driver.active.get_active_torch_device()
    if active_device.type != "npu":
        raise RuntimeError(
            f"OpenTile selected an unexpected torch device: {active_device}"
        )
    return backend


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


def get_npu_properties():
    device = torch.npu.current_device()
    properties = dict(driver.active.utils.get_device_properties(device))
    for name in ("num_aicore", "num_vectorcore"):
        if name not in properties or int(properties[name]) <= 0:
            raise RuntimeError(
                f"Triton driver returned invalid {name!r} for {DEVICE}: {properties}"
            )
    return properties


ASCEND_MAX_GRID_DIM = 65535
_FALLBACK_UB_CAPACITY_BITS = 65536 * 8
_NUM_WARPS = 2
_PREPARE_BWD_MEM_MULT = 18.0
_SAFETY_MARGIN = 0.75
_FALLBACK_TILE = 8
_MAX_TILE_BWD = 32
_ub_capacity_bits: int | None = None


def _get_ub_capacity_bits() -> int:
    global _ub_capacity_bits
    if _ub_capacity_bits is not None:
        return _ub_capacity_bits

    env_capacity = os.environ.get("ASCEND_UB_CAPACITY_BITS")
    if env_capacity is not None:
        try:
            capacity_bits = int(env_capacity)
        except ValueError:
            capacity_bits = 0
        if capacity_bits > 0:
            _ub_capacity_bits = capacity_bits
            return capacity_bits

    try:
        from tbe.common.platform import (  # type: ignore[import-not-found]
            get_soc_spec,
            set_current_compile_soc_info,
        )

        device = torch.npu.current_device()
        set_current_compile_soc_info(torch.npu.get_device_name(device))
        ub_size_bytes = int(get_soc_spec("UB_SIZE"))
        if ub_size_bytes <= 0:
            raise ValueError(f"invalid UB_SIZE: {ub_size_bytes}")
        _ub_capacity_bits = ub_size_bytes * 8
    except Exception as error:
        warnings.warn(
            "Using the conservative 64 KiB UB fallback because runtime UB "
            f"capacity detection failed: {error}. Set "
            "ASCEND_UB_CAPACITY_BITS to override.",
            stacklevel=2,
        )
        _ub_capacity_bits = _FALLBACK_UB_CAPACITY_BITS
    return _ub_capacity_bits


def _get_bwd_axis_tile(BT: int, dim: int) -> int:
    desired = triton.next_power_of_2(dim)
    safe_bits = int(_get_ub_capacity_bits() * _SAFETY_MARGIN)
    max_raw = max(
        1,
        int(
            safe_bits
            // (_PREPARE_BWD_MEM_MULT * float(BT) * 4 * 8)
        ),
    )
    safe_power_of_two = triton.next_power_of_2(max_raw + 1) // 2
    block = min(desired, safe_power_of_two)
    block = max(_FALLBACK_TILE, block)
    return min(block, min(_MAX_TILE_BWD, desired))


def _get_bwd_tiles(BT: int, K: int, V: int) -> tuple[int, int]:
    return _get_bwd_axis_tile(BT, K), _get_bwd_axis_tile(BT, V)


def _max_grid_axis_chunks(
    axis_size: int,
    other_grid_product: int,
    *,
    max_grid: int = ASCEND_MAX_GRID_DIM,
) -> int:
    del axis_size
    return max(1, max_grid // max(other_grid_product, 1))


def _kernel_trace_enabled() -> bool:
    return os.environ.get("WY_BWD_TRACE_KERNELS", "0").lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def _kernel_trace_name(kernel) -> str:
    name = getattr(kernel, "__name__", None)
    if name:
        return str(name)
    function = getattr(kernel, "fn", None)
    name = getattr(function, "__name__", None)
    return str(name) if name else type(kernel).__name__


def _launch_wy_kernel(kernel, *, NT: int, bh_total: int, kernel_kwargs: dict) -> None:
    trace = _kernel_trace_enabled()
    kernel_name = _kernel_trace_name(kernel)
    source_chunk_indices = kernel_kwargs.get("chunk_indices")
    cu_seqlens = kernel_kwargs.get("cu_seqlens")
    max_nt = _max_grid_axis_chunks(NT, bh_total)
    for nt_offset in range(0, NT, max_nt):
        nt_length = min(max_nt, NT - nt_offset)
        if cu_seqlens is not None and source_chunk_indices is not None:
            kernel_kwargs["chunk_indices"] = source_chunk_indices[
                nt_offset : nt_offset + nt_length
            ]
            kernel_kwargs["NT_OFFSET"] = 0
        else:
            kernel_kwargs["NT_OFFSET"] = nt_offset

        max_bh = _max_grid_axis_chunks(bh_total, nt_length)
        for bh_offset in range(0, bh_total, max_bh):
            bh_length = min(max_bh, bh_total - bh_offset)
            kernel_kwargs["BH_OFFSET"] = bh_offset
            trace_context = (
                f"kernel={kernel_name} grid=({nt_length},{bh_length}) "
                f"nt_offset={nt_offset} bh_offset={bh_offset} "
                f"NT={NT} BH_TOTAL={bh_total} "
                f"USE_G={kernel_kwargs.get('USE_G', 'n/a')}"
            )
            if trace:
                print(f"[KERNEL-TRACE] BEGIN {trace_context}", flush=True)
                print(
                    f"[KERNEL-TRACE] PRE_SYNC_BEGIN {trace_context}",
                    flush=True,
                )
                torch.npu.synchronize()
                print(
                    f"[KERNEL-TRACE] PRE_SYNC_END {trace_context}",
                    flush=True,
                )
                print(
                    f"[KERNEL-TRACE] LAUNCH_BEGIN {trace_context}",
                    flush=True,
                )
                started = time.perf_counter()
            kernel[(nt_length, bh_length)](**kernel_kwargs)
            if trace:
                print(
                    f"[KERNEL-TRACE] LAUNCH_RETURN {trace_context}",
                    flush=True,
                )
                print(
                    f"[KERNEL-TRACE] POST_SYNC_BEGIN {trace_context}",
                    flush=True,
                )
                torch.npu.synchronize()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                print(
                    f"[KERNEL-TRACE] END {trace_context} "
                    f"elapsed_ms={elapsed_ms:.3f}",
                    flush=True,
                )


# Triton's block-pointer frontend requires ``offsets`` and ``block_shape`` to
# be 32-bit. Every stage therefore keeps logical chunk index ``i_t`` as i32.
# ``i_b``/``i_h`` are widened separately so pointer bases built from ``bos``
# still use 64-bit address arithmetic.
@triton.jit(do_not_specialize=["T"])
def prepare_wy_repr_bwd_k_npu(
    k, beta, g, A, dw, dk, dA_scr, db, dg,
    cu_seqlens, chunk_indices, T,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
    USE_G: tl.constexpr, IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    i_t = (tl.program_id(0) + NT_OFFSET).to(tl.int32)
    i_bh = (tl.program_id(1) + BH_OFFSET).to(tl.int32)
    i_b = (i_bh // HV).to(tl.int64)
    i_h = (i_bh % HV).to(tl.int64)
    T_local = T
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T_local = (eos - bos).to(tl.int32)
    else:
        bos = i_b * T

    p_b = tl.make_block_ptr(beta + (bos * HV + i_h), (T_local,), (HV,), (i_t * BT,), (BT,), (0,))
    p_db = tl.make_block_ptr(db + (bos * HV + i_h), (T_local,), (HV,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(A + (bos * HV + i_h) * BT, (BT, T_local), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))
    p_dA = tl.make_block_ptr(dA_scr + (bos * HV + i_h) * BT, (BT, T_local), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))

    b_b = tl.load(p_b, boundary_check=(0,))
    b_db = tl.zeros([BT], dtype=tl.float32)
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)

    if USE_G:
        p_g = tl.make_block_ptr(g + (bos * HV + i_h), (T_local,), (HV,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_exp = exp2(b_g)
        b_dg = tl.zeros([BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h // (HV // H)) * K, (T_local, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0),
        )
        p_dk = tl.make_block_ptr(dk + (bos * HV + i_h) * K, (T_local, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dw = tl.make_block_ptr(dw + (bos * HV + i_h) * K, (T_local, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if USE_G:
            b_kbg = b_k * (b_b * b_g_exp)[:, None]
        else:
            b_kbg = b_k * b_b[:, None]
        b_dw = tl.load(p_dw, boundary_check=(0, 1))
        b_dA += tl.dot(b_dw, tl.trans(b_kbg).to(b_dw.dtype))
        b_dkbg = tl.dot(b_A.to(b_dw.dtype), b_dw)
        if USE_G:
            b_dk = b_dkbg * (b_g_exp * b_b)[:, None]
            b_db += tl.sum(b_dkbg * b_k * b_g_exp[:, None], 1)
            b_dg += tl.sum(b_dkbg * b_kbg, 1)
        else:
            b_dk = b_dkbg * b_b[:, None]
            b_db += tl.sum(b_dkbg * b_k, 1)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))
    if USE_G:
        p_dg = tl.make_block_ptr(dg + (bos * HV + i_h), (T_local,), (HV,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


@triton.jit(do_not_specialize=["T"])
def prepare_wy_repr_bwd_v_npu(
    v, beta, A, du, dv, dA_scr, db,
    cu_seqlens, chunk_indices, T,
    HV: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    i_t = (tl.program_id(0) + NT_OFFSET).to(tl.int32)
    i_bh = (tl.program_id(1) + BH_OFFSET).to(tl.int32)
    i_b = (i_bh // HV).to(tl.int64)
    i_h = (i_bh % HV).to(tl.int64)
    T_local = T
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T_local = (eos - bos).to(tl.int32)
    else:
        bos = i_b * T

    p_b = tl.make_block_ptr(beta + (bos * HV + i_h), (T_local,), (HV,), (i_t * BT,), (BT,), (0,))
    p_db = tl.make_block_ptr(db + (bos * HV + i_h), (T_local,), (HV,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(A + (bos * HV + i_h) * BT, (BT, T_local), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))
    p_dA = tl.make_block_ptr(dA_scr + (bos * HV + i_h) * BT, (BT, T_local), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))

    b_b = tl.load(p_b, boundary_check=(0,))
    b_db = tl.load(p_db, boundary_check=(0,)).to(tl.float32)
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_dA = tl.load(p_dA, boundary_check=(0, 1)).to(tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos * HV + i_h) * V, (T_local, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (bos * HV + i_h) * V, (T_local, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_du = tl.make_block_ptr(du + (bos * HV + i_h) * V, (T_local, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_du = tl.load(p_du, boundary_check=(0, 1))
        b_dA += tl.dot(b_du, tl.trans(b_vb))
        b_dvb = tl.dot(b_A, b_du)
        b_dv = b_dvb * b_b[:, None]
        b_db += tl.sum(b_dvb * b_v, 1)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))


@triton.jit(do_not_specialize=["T"])
def prepare_wy_repr_bwd_da_mask_npu(
    dA_scr,
    cu_seqlens, chunk_indices, T,
    HV: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    i_t = (tl.program_id(0) + NT_OFFSET).to(tl.int32)
    i_bh = (tl.program_id(1) + BH_OFFSET).to(tl.int32)
    i_b = (i_bh // HV).to(tl.int64)
    i_h = (i_bh % HV).to(tl.int64)
    T_local = T
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T_local = (eos - bos).to(tl.int32)
    else:
        bos = i_b * T

    p_dA = tl.make_block_ptr(dA_scr + (bos * HV + i_h) * BT, (BT, T_local), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_dA = tl.load(p_dA, boundary_check=(0, 1)).to(tl.float32)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T_local
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA, 0)
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def prepare_wy_repr_bwd_da_dot1_npu(
    A, dA_scr, dA_mid,
    cu_seqlens, chunk_indices, T,
    HV: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    i_t = (tl.program_id(0) + NT_OFFSET).to(tl.int32)
    i_bh = (tl.program_id(1) + BH_OFFSET).to(tl.int32)
    i_b = (i_bh // HV).to(tl.int64)
    i_h = (i_bh % HV).to(tl.int64)
    T_local = T
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T_local = (eos - bos).to(tl.int32)
    else:
        bos = i_b * T

    p_A = tl.make_block_ptr(A + (bos * HV + i_h) * BT, (BT, T_local), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))
    p_in = tl.make_block_ptr(dA_scr + (bos * HV + i_h) * BT, (BT, T_local), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))
    p_out = tl.make_block_ptr(dA_mid + (bos * HV + i_h) * BT, (BT, T_local), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_dA = tl.load(p_in, boundary_check=(0, 1)).to(tl.float32)
    b_out = tl.dot(b_dA, b_A.to(tl.float32))
    tl.store(p_out, b_out.to(p_out.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def prepare_wy_repr_bwd_da_dot2_npu(
    A, dA_mid, dA_out,
    cu_seqlens, chunk_indices, T,
    HV: tl.constexpr, BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    i_t = (tl.program_id(0) + NT_OFFSET).to(tl.int32)
    i_bh = (tl.program_id(1) + BH_OFFSET).to(tl.int32)
    i_b = (i_bh // HV).to(tl.int64)
    i_h = (i_bh % HV).to(tl.int64)
    T_local = T
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T_local = (eos - bos).to(tl.int32)
    else:
        bos = i_b * T

    p_A = tl.make_block_ptr(A + (bos * HV + i_h) * BT, (BT, T_local), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))
    p_in = tl.make_block_ptr(dA_mid + (bos * HV + i_h) * BT, (BT, T_local), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))
    p_out = tl.make_block_ptr(dA_out + (bos * HV + i_h) * BT, (BT, T_local), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_dA = tl.load(p_in, boundary_check=(0, 1)).to(tl.float32)
    b_dA = tl.dot(b_A.to(tl.float32), b_dA)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T_local
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, -b_dA, 0)
    tl.store(p_out, b_dA.to(p_out.dtype.element_ty), boundary_check=(0, 1))


_DG_BLK = 16


@triton.jit(do_not_specialize=["T"])
def prepare_wy_repr_bwd_da_gate_npu(
    g, dA_out,
    cu_seqlens, chunk_indices, T,
    HV: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    i_t = (tl.program_id(0) + NT_OFFSET).to(tl.int32)
    i_bh = (tl.program_id(1) + BH_OFFSET).to(tl.int32)
    i_b = (i_bh // HV).to(tl.int64)
    i_h = (i_bh % HV).to(tl.int64)
    T_local = T
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T_local = (eos - bos).to(tl.int32)
    else:
        bos = i_b * T

    n_sub = BT // BC

    for r in range(n_sub):
        i_tr = i_t * BT + r * BC
        p_gr = tl.make_block_ptr(g + (bos * HV + i_h), (T_local,), (HV,), (i_tr,), (BC,), (0,))
        b_gr = tl.load(p_gr, boundary_check=(0,)).to(tl.float32)
        for c in range(n_sub):
            i_tc = i_t * BT + c * BC
            p_dA = tl.make_block_ptr(
                dA_out + (bos * HV + i_h) * BT, (BT, T_local), (1, HV * BT),
                (r * BC, i_t * BT + c * BC), (BC, BC), (0, 1),
            )
            b_dA = tl.load(p_dA, boundary_check=(0, 1)).to(tl.float32)
            p_gc = tl.make_block_ptr(g + (bos * HV + i_h), (T_local,), (HV,), (i_tc,), (BC,), (0,))
            b_gc = tl.load(p_gc, boundary_check=(0,)).to(tl.float32)
            b_diff = b_gr[:, None] - b_gc[None, :]
            b_gate = exp2(b_diff)
            b_prod = b_dA * b_gate
            b_dA = tl.where(b_prod == b_prod, b_prod, 0.0)
            tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def prepare_wy_repr_bwd_finalize_k_npu(
    k, beta, dA_out, dk, db,
    cu_seqlens, chunk_indices, T,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    i_t = (tl.program_id(0) + NT_OFFSET).to(tl.int32)
    i_bh = (tl.program_id(1) + BH_OFFSET).to(tl.int32)
    i_b = (i_bh // HV).to(tl.int64)
    i_h = (i_bh % HV).to(tl.int64)
    T_local = T
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T_local = (eos - bos).to(tl.int32)
    else:
        bos = i_b * T

    p_b = tl.make_block_ptr(beta + (bos * HV + i_h), (T_local,), (HV,), (i_t * BT,), (BT,), (0,))
    p_db = tl.make_block_ptr(db + (bos * HV + i_h), (T_local,), (HV,), (i_t * BT,), (BT,), (0,))
    p_dA = tl.make_block_ptr(dA_out + (bos * HV + i_h) * BT, (BT, T_local), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))

    b_b = tl.load(p_b, boundary_check=(0,))
    b_db = tl.load(p_db, boundary_check=(0,)).to(tl.float32)
    b_dA = tl.load(p_dA, boundary_check=(0, 1)).to(tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h // (HV // H)) * K, (T_local, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0),
        )
        p_dk = tl.make_block_ptr(dk + (bos * HV + i_h) * K, (T_local, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_kb = b_k * b_b[:, None]
        b_dkb = tl.dot(b_dA, b_k)
        b_db += tl.sum(b_dkb * b_k, 1)
        b_dk = b_dkb * b_b[:, None] + tl.trans(tl.dot(tl.trans(b_kb), b_dA))
        b_dk += tl.load(p_dk, boundary_check=(0, 1)).to(tl.float32)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))


@triton.jit(do_not_specialize=["T"])
def prepare_wy_repr_bwd_finalize_a2_npu(
    k, beta, a2_scr,
    cu_seqlens, chunk_indices, T,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    i_t = (tl.program_id(0) + NT_OFFSET).to(tl.int32)
    i_bh = (tl.program_id(1) + BH_OFFSET).to(tl.int32)
    i_b = (i_bh // HV).to(tl.int64)
    i_h = (i_bh % HV).to(tl.int64)
    T_local = T
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T_local = (eos - bos).to(tl.int32)
    else:
        bos = i_b * T

    p_b = tl.make_block_ptr(beta + (bos * HV + i_h), (T_local,), (HV,), (i_t * BT,), (BT,), (0,))
    p_a2 = tl.make_block_ptr(a2_scr + (bos * HV + i_h) * BT, (BT, T_local), (1, HV * BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_b = tl.load(p_b, boundary_check=(0,))
    b_A2 = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h // (HV // H)) * K, (T_local, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A2 += tl.dot(b_k, tl.trans(b_k))
    b_A2 *= b_b[:, None]
    tl.store(p_a2, b_A2.to(p_a2.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def prepare_wy_repr_bwd_finalize_dg_npu(
    dA_out, a2_scr, dg, col_acc_scr,
    cu_seqlens, chunk_indices, T,
    HV: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    i_t = (tl.program_id(0) + NT_OFFSET).to(tl.int32)
    i_bh = (tl.program_id(1) + BH_OFFSET).to(tl.int32)
    i_b = (i_bh // HV).to(tl.int64)
    i_h = (i_bh % HV).to(tl.int64)
    T_local = T
    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T_local = (eos - bos).to(tl.int32)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t.to(tl.int64)
        bos = i_b * T

    n_sub = BT // BC
    col_off = (i_tg * HV + i_h) * BT
    p_col0 = tl.make_block_ptr(col_acc_scr + col_off, (BT,), (1,), (0,), (BT,), (0,))
    tl.store(p_col0, tl.zeros([BT], dtype=tl.float32), boundary_check=(0,))

    for r in range(n_sub):
        i_tr = i_t * BT + r * BC
        p_dg_r = tl.make_block_ptr(dg + (bos * HV + i_h), (T_local,), (HV,), (i_tr,), (BC,), (0,))
        b_dg_r = tl.load(p_dg_r, boundary_check=(0,)).to(tl.float32)
        for c in range(n_sub):
            p_dA = tl.make_block_ptr(
                dA_out + (bos * HV + i_h) * BT, (BT, T_local), (1, HV * BT),
                (r * BC, i_t * BT + c * BC), (BC, BC), (0, 1),
            )
            p_a2 = tl.make_block_ptr(
                a2_scr + (bos * HV + i_h) * BT, (BT, T_local), (1, HV * BT),
                (r * BC, i_t * BT + c * BC), (BC, BC), (0, 1),
            )
            b_dA = tl.load(p_dA, boundary_check=(0, 1)).to(tl.float32)
            b_a2 = tl.load(p_a2, boundary_check=(0, 1)).to(tl.float32)
            prod = b_dA * b_a2
            b_dg_r += tl.sum(prod, axis=1)
            p_col = tl.make_block_ptr(
                col_acc_scr + col_off, (BT,), (1,), (c * BC,), (BC,), (0,),
            )
            b_col = tl.load(p_col, boundary_check=(0,)).to(tl.float32)
            b_col += tl.sum(prod, axis=0)
            tl.store(p_col, b_col.to(p_col.dtype.element_ty), boundary_check=(0,))
        tl.store(p_dg_r, b_dg_r.to(p_dg_r.dtype.element_ty), boundary_check=(0,))

    p_dg = tl.make_block_ptr(dg + (bos * HV + i_h), (T_local,), (HV,), (i_t * BT,), (BT,), (0,))
    p_col = tl.make_block_ptr(col_acc_scr + col_off, (BT,), (1,), (0,), (BT,), (0,))
    b_dg = tl.load(p_dg, boundary_check=(0,)).to(tl.float32)
    b_col = tl.load(p_col, boundary_check=(0,)).to(tl.float32)
    b_dg -= b_col
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


def _prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    lengths = torch.diff(cu_seqlens)
    chunk_counts = torch.div(
        lengths + chunk_size - 1,
        chunk_size,
        rounding_mode="floor",
    )
    sequence_ids = torch.repeat_interleave(
        torch.arange(
            chunk_counts.numel(),
            dtype=chunk_counts.dtype,
            device=chunk_counts.device,
        ),
        chunk_counts,
    )
    chunk_starts = torch.cat(
        (
            torch.zeros(
                1,
                dtype=chunk_counts.dtype,
                device=chunk_counts.device,
            ),
            chunk_counts.cumsum(0)[:-1],
        )
    )
    local_chunk_ids = torch.arange(
        sequence_ids.numel(),
        dtype=chunk_counts.dtype,
        device=chunk_counts.device,
    ) - chunk_starts[sequence_ids]
    return torch.stack((sequence_ids, local_chunk_ids), dim=1).to(cu_seqlens)


def _validate_contract(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    g: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
    chunk_indices: torch.LongTensor | None,
) -> tuple[int, int, int, int, int, int, int]:
    if k.ndim != 4 or v.ndim != 4 or beta.ndim != 3 or A.ndim != 4:
        raise ValueError("expected k/v rank 4, beta rank 3, and A rank 4")
    B, T, H, K = k.shape
    if min(B, T, H, K) <= 0:
        raise ValueError(f"k dimensions must be positive, got {tuple(k.shape)}")
    if v.shape[:2] != (B, T):
        raise ValueError("k and v must have the same batch and sequence dimensions")
    HV, V = v.shape[2:]
    if min(HV, V) <= 0:
        raise ValueError(f"v head/feature dimensions must be positive, got {tuple(v.shape)}")
    if HV % H != 0:
        raise ValueError(f"HV={HV} must be divisible by H={H}")
    BT = A.shape[-1]
    if BT <= 0:
        raise ValueError(f"BT must be positive, got {BT}")
    if BT & (BT - 1):
        raise ValueError(f"BT must be a power of two, got {BT}")
    if beta.shape != (B, T, HV):
        raise ValueError(f"beta must have shape {(B, T, HV)}, got {tuple(beta.shape)}")
    if A.shape != (B, T, HV, BT):
        raise ValueError(f"A must have shape {(B, T, HV, BT)}, got {tuple(A.shape)}")
    if dw.shape != (B, T, HV, K):
        raise ValueError(f"dw must have shape {(B, T, HV, K)}, got {tuple(dw.shape)}")
    if du.shape != (B, T, HV, V):
        raise ValueError(f"du must have shape {(B, T, HV, V)}, got {tuple(du.shape)}")
    if g is not None and g.shape != (B, T, HV):
        raise ValueError(f"g must have shape {(B, T, HV)}, got {tuple(g.shape)}")
    if g is not None and BT % _DG_BLK != 0:
        raise ValueError(f"gated backward requires BT divisible by {_DG_BLK}, got {BT}")
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError("packed variable-length inputs require B=1")
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must be a one-dimensional tensor with at least two entries")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"cu_seqlens must be int32 or int64, got {cu_seqlens.dtype}")
    if chunk_indices is not None:
        if chunk_indices.ndim != 2 or chunk_indices.shape[1] != 2:
            raise ValueError(
                f"chunk_indices must have shape [N, 2], got {tuple(chunk_indices.shape)}"
            )
        if chunk_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"chunk_indices must be int32 or int64, got {chunk_indices.dtype}")

    named = {
        "k": k,
        "v": v,
        "beta": beta,
        "A": A,
        "dw": dw,
        "du": du,
    }
    if g is not None:
        named["g"] = g
    if cu_seqlens is not None:
        named["cu_seqlens"] = cu_seqlens
    if chunk_indices is not None:
        named["chunk_indices"] = chunk_indices
    current_device = torch.npu.current_device()
    if k.device.type != DEVICE.type or (
        k.device.index is not None and k.device.index != current_device
    ):
        raise ValueError(f"k must be on the current NPU {current_device}, got {k.device}")
    for name, tensor in named.items():
        if tensor.device.type != DEVICE.type:
            raise ValueError(f"{name} must be on an NPU, got {tensor.device}")
        if tensor.device != k.device:
            raise ValueError(f"{name} must be on the same NPU as k ({k.device}), got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    for name in ("k", "v", "beta", "A", "dw", "du"):
        if not named[name].is_floating_point():
            raise ValueError(f"{name} must have a floating-point dtype, got {named[name].dtype}")
    if g is not None and not g.is_floating_point():
        raise ValueError(f"g must have a floating-point dtype, got {g.dtype}")
    return B, T, H, HV, K, V, BT


_acl = None


def _runtime_memcpy_enabled() -> bool:
    return bool(os.environ.get("DAV3510_SIM_RUN_DIR")) or os.environ.get(
        "OPENTILE_USE_RUNTIME_MEMCPY", "0"
    ).lower() in {"1", "true", "on", "yes"}


def _copy_host_to_device(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Use ACL RT memcpy only for dav_3510, which has no ACLNN Copy/Fill."""
    global _acl
    if dst.shape != src.shape or dst.dtype != src.dtype:
        raise ValueError("runtime memcpy requires identical shape and dtype")
    nbytes = src.numel() * src.element_size()
    if _acl is None:
        _acl = ctypes.CDLL("libascendcl.so")
        _acl.aclrtMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        _acl.aclrtMemcpy.restype = ctypes.c_int
    result = _acl.aclrtMemcpy(
        ctypes.c_void_p(dst.data_ptr()),
        nbytes,
        ctypes.c_void_p(src.data_ptr()),
        nbytes,
        1,
    )
    if result != 0:
        raise RuntimeError(f"aclrtMemcpy H2D failed with error code {result}")


def _poison(shape: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
    if not _runtime_memcpy_enabled():
        return torch.full(shape, float("nan"), dtype=like.dtype, device=like.device)
    host = torch.full(shape, float("nan"), dtype=like.dtype, device="cpu")
    result = torch.empty(shape, dtype=like.dtype, device=like.device)
    _copy_host_to_device(result, host)
    return result


def _zeros(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Allocate zeroed scratch without relying on simulator ACLNN Fill."""
    if not _runtime_memcpy_enabled():
        return torch.zeros(shape, dtype=dtype, device=device)
    host = torch.zeros(shape, dtype=dtype, device="cpu")
    result = torch.empty(shape, dtype=dtype, device=device)
    _copy_host_to_device(result, host)
    return result


def prepare_wy_repr_bwd_npu(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    g: torch.Tensor = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, HV, K, V, BT = _validate_contract(
        k,
        v,
        beta,
        A,
        dw,
        du,
        g,
        cu_seqlens,
        chunk_indices,
    )
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = _prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK, BV = _get_bwd_tiles(BT, K, V)
    use_g = g is not None
    is_varlen = cu_seqlens is not None

    dk = _poison((B, T, HV, K), k)
    dv = _poison(tuple(v.shape), v)
    dg = _poison(tuple(g.shape), g) if use_g else None
    db = _poison(tuple(beta.shape), beta)
    g_arg = g if use_g else beta
    dg_arg = dg if use_g else beta
    dA_scr = _zeros(tuple(A.shape), dtype=torch.float32, device=k.device)
    dA_mid = _zeros(tuple(A.shape), dtype=torch.float32, device=k.device)
    dA_out = _zeros(tuple(A.shape), dtype=torch.float32, device=k.device)
    a2_scr = (
        _zeros(tuple(A.shape), dtype=torch.float32, device=k.device)
        if use_g
        else None
    )
    col_acc_scr = (
        _zeros((B, NT, HV, BT), dtype=torch.float32, device=k.device)
        if use_g
        else None
    )

    base = dict(
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        BT=BT,
        IS_VARLEN=is_varlen,
        num_warps=_NUM_WARPS,
    )
    _launch_wy_kernel(
        prepare_wy_repr_bwd_k_npu,
        NT=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            k=k, beta=beta, g=g_arg, A=A, dw=dw,
            dk=dk, dA_scr=dA_scr, db=db, dg=dg_arg,
            H=H, HV=HV, K=K, BK=BK, USE_G=use_g,
            **base,
        ),
    )
    _launch_wy_kernel(
        prepare_wy_repr_bwd_v_npu,
        NT=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            v=v, beta=beta, A=A, du=du, dv=dv, dA_scr=dA_scr, db=db,
            HV=HV, V=V, BV=BV,
            **base,
        ),
    )
    _launch_wy_kernel(
        prepare_wy_repr_bwd_da_mask_npu,
        NT=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            dA_scr=dA_scr,
            HV=HV,
            **base,
        ),
    )
    _launch_wy_kernel(
        prepare_wy_repr_bwd_da_dot1_npu,
        NT=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            A=A, dA_scr=dA_scr, dA_mid=dA_mid,
            HV=HV,
            **base,
        ),
    )
    _launch_wy_kernel(
        prepare_wy_repr_bwd_da_dot2_npu,
        NT=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            A=A, dA_mid=dA_mid, dA_out=dA_out,
            HV=HV,
            **base,
        ),
    )
    if use_g:
        _launch_wy_kernel(
            prepare_wy_repr_bwd_da_gate_npu,
            NT=NT,
            bh_total=B * HV,
            kernel_kwargs=dict(
                g=g_arg, dA_out=dA_out,
                HV=HV, BC=_DG_BLK,
                **base,
            ),
        )
    _launch_wy_kernel(
        prepare_wy_repr_bwd_finalize_k_npu,
        NT=NT,
        bh_total=B * HV,
        kernel_kwargs=dict(
            k=k, beta=beta, dA_out=dA_out, dk=dk, db=db,
            H=H, HV=HV, K=K, BK=BK,
            **base,
        ),
    )
    if use_g:
        _launch_wy_kernel(
            prepare_wy_repr_bwd_finalize_a2_npu,
            NT=NT,
            bh_total=B * HV,
            kernel_kwargs=dict(
                k=k, beta=beta, a2_scr=a2_scr,
                H=H, HV=HV, K=K, BK=BK,
                **base,
            ),
        )
        _launch_wy_kernel(
            prepare_wy_repr_bwd_finalize_dg_npu,
            NT=NT,
            bh_total=B * HV,
            kernel_kwargs=dict(
                dA_out=dA_out, a2_scr=a2_scr, dg=dg_arg, col_acc_scr=col_acc_scr,
                HV=HV, BC=_DG_BLK,
                **base,
            ),
        )
    if H != HV:
        dk = dk.view(B, T, H, HV // H, K).sum(3)
    return dk, dv, db, dg


# Match the generic FLA public name while retaining the ``_npu`` alias used
# by the exact triton-ascend baseline.
prepare_wy_repr_bwd = prepare_wy_repr_bwd_npu
bwd_prepare_wy_repr = prepare_wy_repr_bwd_npu
