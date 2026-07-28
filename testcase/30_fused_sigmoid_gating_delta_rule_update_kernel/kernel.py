from __future__ import annotations

import torch
import triton
import triton.language as tl


STAGE1_SHAPE = (1, 1, 4, 8, 128, 128)
STAGE2_SHAPE = (1, 1, 16, 32, 256, 256)
STAGE3_SHAPE = (1, 1, 32, 64, 144, 144)
SUPPORTED_SHAPES = (STAGE1_SHAPE, STAGE2_SHAPE, STAGE3_SHAPE)


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0_source"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Original fused sigmoid-gating delta-rule recurrence."""
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all_tokens = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all_tokens = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all_tokens + bos) * HV + i_hv) * V + o_v
    p_A_log = A_log + i_hv
    p_a = a + bos * HV + i_hv
    p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        safe_idx = tl.where(idx < 0, 0, idx)
        p_h0 = (
            h0_source
            + safe_idx * HV * K * V
            + i_hv * K * V
            + o_k[:, None] * V
            + o_v[None, :]
        )
        loaded_h0 = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
        b_h += tl.where(idx < 0, tl.zeros_like(loaded_h0), loaded_h0)

    for i in range(0, T):
        b_q = tl.load(p_q + i * H * K, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k + i * H * K, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v + i * HV * V, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b + i * HV).to(tl.float32)
        b_A_log = tl.load(p_A_log).to(tl.float32)
        b_a = tl.load(p_a + i * HV).to(tl.float32)
        b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x
        b_beta = tl.sigmoid(b_b)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)

        b_q = b_q * scale
        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o + i * HV * V, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def active_opentile_npu() -> tuple[torch.device, int, int]:
    target = triton.runtime.driver.active.get_current_target()
    backend = str(target.backend)
    if backend != "opentile" and not backend.startswith("opentile_"):
        raise RuntimeError(f"expected OpenTile backend, got {backend!r}")

    device = torch.device(triton.runtime.driver.active.get_active_torch_device())
    if device.type != "npu":
        raise RuntimeError(f"expected a physical NPU device, got {device}")
    if device.index is None:
        device = torch.device("npu", int(torch.npu.current_device()))

    properties = triton.runtime.driver.active.utils.get_device_properties(
        device.index
    )
    num_aicore = int(properties["num_aicore"])
    num_vectorcore = int(properties["num_vectorcore"])
    if num_aicore <= 0 or num_vectorcore <= 0:
        raise RuntimeError(
            f"invalid core properties: aic={num_aicore}, aiv={num_vectorcore}"
        )
    return device, num_aicore, num_vectorcore


def _validate_inputs(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    if q.ndim != 4 or k.shape != q.shape:
        raise ValueError("q and k must have shape [B,T,H,K]")
    if v.ndim != 4 or v.shape[:2] != q.shape[:2]:
        raise ValueError("v must have shape [B,T,HV,V]")

    batch, time, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    shape = (batch, time, heads, value_heads, key_dim, value_dim)
    if shape not in SUPPORTED_SHAPES:
        raise ValueError(
            f"supported shapes are {SUPPORTED_SHAPES}, got {shape}"
        )
    if value_heads % heads != 0:
        raise ValueError("HV must be divisible by H")
    if a.shape != (batch * time, value_heads) or b.shape != a.shape:
        raise ValueError("a and b must have shape [B*T,HV]")
    if A_log.shape != (value_heads,) or dt_bias.shape != (value_heads,):
        raise ValueError("A_log and dt_bias must have shape [HV]")
    if initial_state_source.shape != (
        batch,
        value_heads,
        key_dim,
        value_dim,
    ):
        raise ValueError("initial_state_source must have shape [B,HV,K,V]")
    if initial_state_indices.shape != (batch,):
        raise ValueError("initial_state_indices must have shape [B]")
    if initial_state_indices.dtype != torch.int32:
        raise ValueError("initial_state_indices must use torch.int32")

    floating = (A_log, a, dt_bias, q, k, v, b, initial_state_source)
    if any(tensor.dtype != torch.bfloat16 for tensor in floating):
        raise ValueError("stage one is BF16-only")
    if any(not tensor.is_contiguous() for tensor in floating):
        raise ValueError("all floating tensors must be contiguous")
    if not initial_state_indices.is_contiguous():
        raise ValueError("initial_state_indices must be contiguous")
    return shape


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, time, heads, value_heads, key_dim, value_dim = (
        _validate_inputs(
            A_log,
            a,
            dt_bias,
            q,
            k,
            v,
            b,
            initial_state_source,
            initial_state_indices,
        )
    )
    device, _, _ = active_opentile_npu()
    tensors = (
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        initial_state_source,
        initial_state_indices,
    )
    if any(tensor.device != device for tensor in tensors):
        raise ValueError(f"all tensors must be on active device {device}")

    block_key = triton.next_power_of_2(key_dim)
    block_value = min(triton.next_power_of_2(value_dim), 64)
    n_key_tiles = triton.cdiv(key_dim, block_key)
    n_value_tiles = triton.cdiv(value_dim, block_value)
    if n_key_tiles != 1:
        raise AssertionError("NK > 1 is not supported")

    output = torch.full(
        (n_key_tiles, *v.shape),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    grid = (n_key_tiles, n_value_tiles, batch * value_heads)
    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log,
        a,
        dt_bias,
        softplus_beta,
        softplus_threshold,
        q,
        k,
        v,
        b,
        output,
        initial_state_source,
        initial_state_indices,
        None,
        key_dim**-0.5,
        time,
        B=batch,
        H=heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BK=block_key,
        BV=block_value,
        USE_QK_L2NORM_IN_KERNEL=True,
    )
    return output.squeeze(0), initial_state_source
