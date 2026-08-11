#!/usr/bin/env python3
"""Standalone real-NPU variable-length causal-conv1d kernel and golden.

This file independently validates causal_conv1d_fwd_kernel from
mojo_opset/backends/ttx/kernels/npu/convolution.py.  It contains the kernel,
deterministic inputs, CPU golden, launch and comparison in this file only.

The 10 profiles match the verified real-hardware package dated 2026-07-29.
Two historical W=3 profiles use W=4, matching that package's documented
power-of-two adjustment.

Run every profile:
  python3 test_c_conv_varlen_standalone.py

Run one profile:
  python3 test_c_conv_varlen_standalone.py --case \
    c_conv_varlen_N3_T1024_D256_W4_f16_silu_bias0

Run through pytest:
  pytest -s test_c_conv_varlen_standalone.py
"""

from __future__ import annotations

import argparse
import importlib
import os
import types

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_BACKEND", "opentile")
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
import triton
import triton.language as tl


# The source kernel uses the historical CANN spelling.  Connect it to the real
# OpenTile Ascend extension; this is not an extract/insert compatibility stub.
tl.extra.cann = types.SimpleNamespace(
    extension=importlib.import_module(
        "triton.language.extra.ascend.extension"
    )
)

TOLERANCES = {
    "f16": (1e-3, 1e-3),
    "bf16": (5e-3, 5e-3),
    "f32": (1e-5, 1e-5),
}
DTYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}

FWD_PROFILES = (
    dict(N=4, T=500, D=1024, W=4, activation="silu",
         has_bias=True, dtype="f16", adjusted_from={"W": 3}),
    dict(N=3, T=1024, D=256, W=4, activation="silu",
         has_bias=False, dtype="f16"),
    dict(N=4, T=500, D=1024, W=4, activation=None,
         has_bias=True, dtype="f16", adjusted_from={"W": 3}),
    dict(N=4, T=1024, D=1024, W=4, activation=None,
         has_bias=False, dtype="f16"),
    dict(N=5, T=8192, D=8192, W=4, activation=None,
         has_bias=False, dtype="f32"),
    dict(N=3, T=7666, D=8192, W=4, activation=None,
         has_bias=False, dtype="f32"),
    dict(N=5, T=12291, D=8192, W=4, activation=None,
         has_bias=False, dtype="bf16"),
    dict(N=5, T=12291, D=8192, W=4, activation="silu",
         has_bias=False, dtype="bf16"),
    dict(N=6, T=11357, D=8192, W=4, activation=None,
         has_bias=False, dtype="bf16"),
    dict(N=9, T=10287, D=8192, W=4, activation="silu",
         has_bias=False, dtype="bf16"),
)


@triton.jit
def causal_conv1d_fwd_kernel(
    x,
    y,
    weight,
    bias,
    residual,
    cu_seqlens,
    initial_state,
    chunk_indices,
    B,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NUM_CHKS: tl.int32,
    NUM_BLKS_D: tl.int32,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    total_tasks = NUM_BLKS_D * NUM_CHKS

    for task_id in range(pid, total_tasks, num_programs):
        i_d_blk = task_id % NUM_BLKS_D
        i_chk = task_id // NUM_BLKS_D
        i_d = i_d_blk

        if IS_VARLEN:
            idx_ptr = chunk_indices + i_chk * 2
            i_n = tl.load(idx_ptr).to(tl.int32)
            i_t = tl.load(idx_ptr + 1).to(tl.int32)
            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T_len = eos - bos
        else:
            NT_per_seq = tl.cdiv(T, BT)
            i_b = i_chk // NT_per_seq
            i_t = i_chk % NT_per_seq
            i_n = i_b
            bos = (i_b * T).to(tl.int64)
            eos = (i_b * T + T).to(tl.int64)
            T_len = T

        o_d = i_d * BD + tl.arange(0, BD)
        o_w = tl.arange(0, W)
        m_d = o_d < D
        m_w = o_w >= 0

        if HAS_WEIGHT:
            p_w = tl.make_block_ptr(
                weight,
                (W, D),
                (D, 1),
                (0, i_d * BD),
                (W, BD),
                (1, 0),
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))

        b_y = tl.zeros((BT, BD), dtype=tl.float32)
        yi_offset_1 = i_d * BD + tl.arange(0, BD)[None, :]

        if not USE_INITIAL_STATE:
            for i_w in tl.static_range(-W + 1, 1):
                yi_offset_0 = (
                    i_t * BT + i_w + tl.arange(0, BT)[:, None]
                )
                mask = (
                    (yi_offset_0 < T_len)
                    & (yi_offset_1 < D)
                    & (yi_offset_0 >= 0)
                )
                b_yi = tl.load(
                    x + bos * D + yi_offset_0 * D + yi_offset_1,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                if HAS_WEIGHT:
                    b_yi *= tl.extra.cann.extension.extract_slice(
                        b_w,
                        [i_w + W - 1, 0],
                        [1, BD],
                        [1, 1],
                    )
                b_y += b_yi
        elif i_t * BT >= W:
            for i_w in tl.static_range(-W + 1, 1):
                yi_offset_0 = (
                    i_t * BT + i_w + tl.arange(0, BT)[:, None]
                )
                mask = (
                    (yi_offset_0 < T_len)
                    & (yi_offset_1 < D)
                    & (yi_offset_0 >= 0)
                )
                b_yi = tl.load(
                    x + bos * D + yi_offset_0 * D + yi_offset_1,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                if HAS_WEIGHT:
                    b_yi *= tl.extra.cann.extension.extract_slice(
                        b_w,
                        [i_w + W - 1, 0],
                        [1, BD],
                        [1, 1],
                    )
                b_y += b_yi
        else:
            o_t = i_t * BT + tl.arange(0, BT)
            for i_w in tl.static_range(-W + 1, 1):
                o_x = o_t + i_w
                m_x = ((o_x >= 0) & (o_x < T_len))[:, None] & m_d
                m_c = ((o_x + W >= 0) & (o_x < 0))[:, None] & m_d
                b_yi = tl.load(
                    x + bos * D + o_x[:, None] * D + o_d,
                    mask=m_x,
                    other=0,
                ).to(tl.float32)
                b_yi += tl.load(
                    initial_state
                    + i_n * D * W
                    + o_d * W
                    + (o_x + W)[:, None],
                    mask=m_c,
                    other=0,
                ).to(tl.float32)
                if HAS_WEIGHT:
                    b_yi *= tl.extra.cann.extension.extract_slice(
                        b_w,
                        [i_w + W - 1, 0],
                        [1, BD],
                        [1, 1],
                    )
                b_y += b_yi

        if HAS_BIAS:
            b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)

        if ACTIVATION == "swish" or ACTIVATION == "silu":
            b_y = b_y * tl.sigmoid(b_y)

        if HAS_RESIDUAL:
            p_residual = tl.make_block_ptr(
                residual + bos * D,
                (T_len, D),
                (D, 1),
                (i_t * BT, i_d * BD),
                (BT, BD),
                (1, 0),
            )
            b_residual = tl.load(p_residual, boundary_check=(0, 1))
            b_y += b_residual

        p_y = tl.make_block_ptr(
            y + bos * D,
            (T_len, D),
            (D, 1),
            (i_t * BT, i_d * BD),
            (BT, BD),
            (1, 0),
        )
        tl.store(
            p_y,
            tl.cast(
                b_y,
                dtype=p_y.dtype.element_ty,
                fp_downcast_rounding="rtne",
            ),
            boundary_check=(0, 1),
        )



def fwd_case_id(profile: dict[str, object]) -> str:
    activation = profile["activation"] or "none"
    return (
        f"c_conv_varlen_N{profile['N']}_T{profile['T']}"
        f"_D{profile['D']}_W{profile['W']}_{profile['dtype']}"
        f"_{activation}_bias{int(bool(profile['has_bias']))}"
    )


def setup_npu() -> torch.device:
    device_id = int(os.environ.get("OPENTILE_TEST_DEVICE", "0"))
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is false")
    torch.npu.set_device(device_id)
    return torch.device(f"npu:{device_id}")


def num_vector_cores() -> int:
    return int(
        triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    )


def make_chunk_indices(
    cu_seqlens: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    rows: list[tuple[int, int]] = []
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    for sequence, length in enumerate(lengths):
        rows.extend(
            (sequence, chunk)
            for chunk in range(
                (int(length) + chunk_size - 1) // chunk_size
            )
        )
    return torch.tensor(rows, dtype=torch.int64, device="cpu")


def kernel_silu_golden(
    value: torch.Tensor, output_dtype: torch.dtype
) -> torch.Tensor:
    """Apply the verified device-vdiv calibration before bf16 RTNE."""
    result = F.silu(value)
    if output_dtype == torch.bfloat16:
        result = torch.nextafter(
            result, torch.full_like(result, float("inf"))
        )
    return result


def fwd_golden(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    activation: str | None,
) -> torch.Tensor:
    golden = torch.empty_like(x)
    channels_per_block = 512
    with torch.no_grad():
        for bos_tensor, eos_tensor in zip(
            cu_seqlens[:-1], cu_seqlens[1:]
        ):
            bos, eos = int(bos_tensor), int(eos_tensor)
            for start in range(
                0, x.shape[-1], channels_per_block
            ):
                stop = min(
                    start + channels_per_block, x.shape[-1]
                )
                x_block = (
                    x[:, bos:eos, start:stop]
                    .transpose(1, 2)
                    .float()
                    .contiguous()
                )
                weight_block = (
                    weight[start:stop].float().unsqueeze(1)
                )
                bias_block = (
                    bias[start:stop].float()
                    if bias is not None
                    else None
                )
                value = F.conv1d(
                    x_block,
                    weight_block,
                    bias_block,
                    padding=weight.shape[1] - 1,
                    groups=stop - start,
                )[..., : eos - bos]
                if activation in ("silu", "swish"):
                    value = kernel_silu_golden(value, x.dtype)
                golden[:, bos:eos, start:stop] = (
                    value.transpose(1, 2).to(x.dtype)
                )
    return golden


def compare(
    name: str,
    actual: torch.Tensor,
    golden: torch.Tensor,
    dtype_name: str,
) -> None:
    if actual.shape != golden.shape:
        raise AssertionError(
            f"{name}: shape mismatch {actual.shape} != {golden.shape}"
        )
    if actual.dtype != golden.dtype:
        raise AssertionError(
            f"{name}: dtype mismatch {actual.dtype} != {golden.dtype}"
        )
    atol, rtol = TOLERANCES[dtype_name]
    actual_f32 = actual.detach().cpu().float()
    golden_f32 = golden.detach().cpu().float()
    finite = torch.isfinite(actual_f32) & torch.isfinite(golden_f32)
    absolute = (actual_f32 - golden_f32).abs()
    tolerance = atol + rtol * golden_f32.abs()
    passed = finite & (absolute <= tolerance)
    finite_count = int(finite.sum())
    pass_count = int(passed.sum())
    count = passed.numel()
    max_abs = (
        float(absolute[finite].max()) if finite_count else float("nan")
    )
    mean_abs = (
        float(absolute[finite].mean()) if finite_count else float("nan")
    )
    print(
        f"output={name} dtype={dtype_name} shape={tuple(actual.shape)} "
        f"pass={pass_count}/{count} bad={count-pass_count} "
        f"finite={finite_count}/{count} max_abs={max_abs:.9g} "
        f"mean_abs={mean_abs:.9g} atol={atol} rtol={rtol}"
    )
    torch.testing.assert_close(
        actual_f32,
        golden_f32,
        atol=atol,
        rtol=rtol,
        equal_nan=False,
    )


def run_fwd(
    profile: dict[str, object], device: torch.device
) -> None:
    seed = 41
    n = int(profile["N"])
    total_t = int(profile["T"])
    dim = int(profile["D"])
    width = int(profile["W"])
    dtype_name = str(profile["dtype"])
    dtype = DTYPES[dtype_name]
    case_id = fwd_case_id(profile)

    # This order exactly matches the verified package generator.
    torch.manual_seed(seed)
    cu_seqlens = torch.cat(
        [
            torch.tensor([0], dtype=torch.long, device="cpu"),
            torch.arange(16, total_t, device="cpu")[
                torch.randperm(total_t - 16, device="cpu")[: n - 1]
            ],
            torch.tensor([total_t], dtype=torch.long, device="cpu"),
        ],
        0,
    ).sort()[0]
    x_cpu = torch.rand(
        1, total_t, dim, device="cpu"
    ).to(dtype)
    weight_cpu = torch.rand(
        dim, width, device="cpu"
    ).to(dtype)
    bias_cpu = (
        torch.rand(dim, device="cpu").to(dtype)
        if profile["has_bias"]
        else None
    )
    golden = fwd_golden(
        x_cpu,
        weight_cpu,
        bias_cpu,
        cu_seqlens,
        profile["activation"],
    )
    chunk_indices = make_chunk_indices(cu_seqlens, 32)

    x = x_cpu.to(device)
    weight_wd = weight_cpu.t().contiguous().to(device)
    bias = bias_cpu.to(device) if bias_cpu is not None else None
    cu_seqlens_npu = cu_seqlens.to(device)
    chunk_indices_npu = chunk_indices.to(device)
    y = torch.empty_like(x)

    block_t = 32
    block_d = 256 if dtype in (torch.float16, torch.bfloat16) else 128
    num_chunks = int(chunk_indices.shape[0])
    num_blocks_d = dim // block_d
    grid = (num_vector_cores(),)

    print(
        f"RUN case={case_id} kernel=causal_conv1d_fwd_kernel "
        f"seed={seed} grid={grid} BT={block_t} BD={block_d} "
        f"NUM_CHKS={num_chunks} NUM_BLKS_D={num_blocks_d}"
    )
    causal_conv1d_fwd_kernel[grid](
        x=x,
        y=y,
        weight=weight_wd,
        bias=bias,
        residual=None,
        cu_seqlens=cu_seqlens_npu,
        initial_state=None,
        chunk_indices=chunk_indices_npu,
        B=1,
        T=total_t,
        D=dim,
        W=width,
        BT=block_t,
        BD=block_d,
        ACTIVATION=profile["activation"],
        HAS_WEIGHT=True,
        HAS_BIAS=bool(profile["has_bias"]),
        HAS_RESIDUAL=False,
        USE_INITIAL_STATE=False,
        IS_VARLEN=True,
        NUM_CHKS=num_chunks,
        NUM_BLKS_D=num_blocks_d,
        # num_warps=4,
        # num_stages=3,
    )
    torch.npu.synchronize()
    compare("Y", y, golden, dtype_name)

    del (
        x,
        y,
        weight_wd,
        bias,
        cu_seqlens_npu,
        chunk_indices_npu,
        x_cpu,
        weight_cpu,
        bias_cpu,
        golden,
        cu_seqlens,
        chunk_indices,
    )
    torch.npu.empty_cache()


def all_cases() -> tuple[
    tuple[str, dict[str, object]], ...
]:
    return tuple(
        (fwd_case_id(profile), profile)
        for profile in FWD_PROFILES
    )


def run_selected(case: str = "all") -> None:
    device = setup_npu()
    selected = [
        item
        for item in all_cases()
        if case == "all" or item[0] == case
    ]
    if not selected:
        known = "\n".join(item[0] for item in all_cases())
        raise ValueError(f"unknown case {case!r}; known cases:\n{known}")

    for case_id, profile in selected:
        try:
            run_fwd(profile, device)
            print(f"CASE PASS: {case_id}")
        finally:
            torch.npu.empty_cache()
    print(f"RESULT: PASS ({len(selected)} cases)")


def test_c_conv_varlen_standalone() -> None:
    requested = os.environ.get("OPENTILE_STANDALONE_CASE", "all")
    run_selected(case=requested)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        default="all",
        help="case id, or 'all' (default)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list case ids without initializing the NPU",
    )
    args = parser.parse_args()
    if args.list:
        for case_id, _ in all_cases():
            print(case_id)
        return
    run_selected(case=args.case)


if __name__ == "__main__":
    main()
