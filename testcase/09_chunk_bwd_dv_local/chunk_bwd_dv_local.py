"""Standalone Triton/OpenTile port of FLA chunk_bwd_dv_local."""

from __future__ import annotations

from functools import cache
from typing import Any

import torch
import triton
import triton.language as tl
from triton.runtime import driver


@triton.jit(do_not_specialize=["T"])
def chunk_bwd_kernel_dv_local(
    q,
    k,
    g,
    do,
    dv,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
):
    linear_pid = tl.program_id(0)
    i_t = linear_pid % NT
    i_b = linear_pid // NT
    bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    for i_h in range(HV):
        q_head = q + (bos * H + i_h // (HV // H)) * K
        k_head = k + (bos * H + i_h // (HV // H)) * K
        do_head = do + (bos * HV + i_h) * V
        dv_head = dv + (bos * HV + i_h) * V

        g_head = g + (i_b * HV + i_h) * T
        b_g = tl.load(g_head + o_t, mask=m_t, other=0.0).to(tl.float32)

        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_k = tl.make_block_ptr(
                k_head,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            p_q = tl.make_block_ptr(
                q_head,
                (K, T),
                (1, H * K),
                (i_k * BK, i_t * BT),
                (BK, BT),
                (0, 1),
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_A += tl.dot(b_k, b_q) * scale

        b_A *= tl.exp(b_g[None, :] - b_g[:, None])
        m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
        b_A = tl.where(m_A, b_A, 0.0).to(do_head.dtype.element_ty)

        for i_v in range(tl.cdiv(V, BV)):
            p_do = tl.make_block_ptr(
                do_head,
                (T, V),
                (HV * V, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            p_dv = tl.make_block_ptr(
                dv_head,
                (T, V),
                (HV * V, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
            tl.store(
                p_dv,
                b_dv.to(p_dv.dtype.element_ty),
                boundary_check=(0, 1),
            )


@cache
def _device_properties() -> tuple[torch.device, int, int, str]:
    target = driver.active.get_current_target()
    backend = str(target.backend)
    if backend != "opentile" and not backend.startswith("opentile_"):
        raise RuntimeError(f"expected OpenTile target, got {backend!r}")
    active_device = driver.active.get_active_torch_device()
    if active_device.type != "npu":
        raise RuntimeError(f"expected NPU device, got {active_device}")
    logical_device = torch.npu.current_device()
    properties: dict[str, Any] = driver.active.utils.get_device_properties(
        logical_device
    )
    num_aicore = int(properties["num_aicore"])
    num_vectorcore = int(properties["num_vectorcore"])
    if num_aicore <= 0 or num_vectorcore <= 0:
        raise RuntimeError(f"invalid NPU properties: {properties!r}")
    return active_device, num_aicore, num_vectorcore, backend


def apply_chunk_bwd_dv_local(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    if q.dtype != torch.bfloat16:
        raise ValueError("source contract requires BF16 q")
    if k.dtype != q.dtype or do.dtype != q.dtype or g.dtype != q.dtype:
        raise ValueError("q, k, do, and g must share the BF16 dtype")
    if q.ndim != 4 or k.ndim != 4 or do.ndim != 4 or g.ndim != 3:
        raise ValueError("expected q/k/do rank 4 and g rank 3")
    if q.shape != k.shape:
        raise ValueError("q and k must have identical [B, T, H, K] shapes")
    B, T, H, K = q.shape
    if do.shape[:2] != (B, T):
        raise ValueError("do must share q batch and sequence dimensions")
    HV, V = do.shape[2:]
    if g.shape != (B, T, HV):
        raise ValueError("g must have shape [B, T, HV]")
    if HV < H or HV % H:
        raise ValueError("HV must be an integer multiple of H")
    if chunk_size != 64:
        raise ValueError("this focused source configuration requires chunk_size=64")
    if K <= 0 or V <= 0:
        raise ValueError("K and V must be positive")
    if not q.is_contiguous() or not k.is_contiguous() or not do.is_contiguous():
        raise ValueError("q, k, and do must be contiguous")

    active_device, num_aicore, num_vectorcore, backend = _device_properties()
    BT = chunk_size
    BK = min(max(triton.next_power_of_2(K), 16), 256)
    BV = min(max(triton.next_power_of_2(V), 16), 256)
    NT = triton.cdiv(T, BT)
    if scale is None:
        scale = K**-0.5

    g_head_major = g.transpose(1, 2).contiguous()
    dv = torch.full_like(do, float("nan"))
    physical_grid = (B * NT,)
    chunk_bwd_kernel_dv_local[physical_grid](
        q,
        k,
        g_head_major,
        do,
        dv,
        scale,
        T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NT=NT,
    )
    print(
        "[OPENTILE_E2E] op=chunk_bwd_dv_local "
        f"route_ok backend={backend} active_device={active_device} "
        f"logical_device={torch.npu.current_device()} "
        f"num_aicore={num_aicore} num_vectorcore={num_vectorcore} "
        f"kernel_mode=mix physical_grid={physical_grid} "
        f"logical_grid=({NT}, {B}) BT={BT} BK={BK} BV={BV}"
    )
    return dv
