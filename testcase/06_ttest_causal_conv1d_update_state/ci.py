#!/usr/bin/env python3
"""Standalone real-NPU causal-conv1d state-update kernel and golden.

This file independently validates causal_conv1d_update_kernel_bdt_fwd from
mojo_opset/backends/ttx/kernels/npu/convolution.py.  It contains the kernel,
deterministic inputs, CPU golden, launch and comparison in this file only.

The 7 profiles match the verified real-hardware package dated 2026-07-29.
Historical W=3/W=4 profiles use W=1, matching that package's documented
power-of-two adjustment.

Run every profile:
  python3 test_causal_conv1d_update_state_standalone.py

Run one profile:
  python3 test_causal_conv1d_update_state_standalone.py --case \
    test_causal_conv1d_update_state_B1_T32_D32_W1_f16_none

Run through pytest:
  pytest -s test_causal_conv1d_update_state_standalone.py
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

UPDATE_PROFILES = (
    dict(B=1, T=12291, D=8192, W=1, activation="swish",
         dtype="f16", adjusted_from={"W": 4}),
    dict(B=1, T=5000, D=2048, W=1, activation="swish",
         dtype="f16", adjusted_from={"W": 4}),
    dict(B=2, T=64, D=128, W=1, activation="swish",
         dtype="f16", adjusted_from={"W": 3}),
    dict(B=2, T=128, D=128, W=1, activation="swish",
         dtype="f16", adjusted_from={"W": 4}),
    dict(B=2, T=64, D=128, W=1, activation=None,
         dtype="f16", adjusted_from={"W": 3}),
    dict(B=3, T=1446, D=256, W=1, activation=None,
         dtype="f16", adjusted_from={"W": 4}),
    dict(B=1, T=32, D=32, W=1, activation=None,
         dtype="f16", adjusted_from={"W": 4}),
)


@triton.jit()
def causal_conv1d_update_kernel_bdt_fwd(
    x_ptr,
    conv_state_ptr,
    conv_state_update_ptr,
    weight_ptr,
    bias_ptr,
    conv_state_indices_ptr,
    out_ptr,
    batch: tl.constexpr,
    dim: tl.constexpr,
    state_len: tl.constexpr,
    seq_len: tl.constexpr,
    width: tl.constexpr,
    out_len: tl.constexpr,
    x_batch_stride: tl.constexpr,
    conv_batch_stride: tl.constexpr,
    out_batch_stride: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    T_CHK_SIZE: tl.constexpr,
    D_CHK_SIZE: tl.constexpr,
    NUM_T_CHK: tl.constexpr,
    NUM_D_CHK: tl.constexpr,
    ST_STORE_HEAD_TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pnum = tl.num_programs(0)
    total_task = batch * NUM_D_CHK * NUM_T_CHK

    for task_id in tl.range(pid, total_task, pnum):
        di = task_id % NUM_D_CHK
        bti = task_id // NUM_D_CHK
        bi = bti // NUM_T_CHK
        ti = bti % NUM_T_CHK

        w_off_d = (
            di * D_CHK_SIZE + tl.arange(0, D_CHK_SIZE)[:, None]
        )
        w_off_w = tl.arange(0, width)[None, :]
        w = tl.load(
            weight_ptr + w_off_d * width + w_off_w
        ).to(tl.float32)

        if ti == 0:
            st_b = tl.load(
                tl.make_block_ptr(
                    conv_state_ptr + bi * state_len * dim,
                    shape=(dim, state_len),
                    strides=(state_len, 1),
                    offsets=(
                        di * D_CHK_SIZE,
                        state_len - (width - 1),
                    ),
                    block_shape=(
                        D_CHK_SIZE,
                        (width - 1) + T_CHK_SIZE,
                    ),
                    order=(1, 0),
                ),
                boundary_check=(0, 1),
                padding_option="zero",
            )
            x_b_tmp = tl.load(
                tl.make_block_ptr(
                    x_ptr + bi * dim * seq_len,
                    shape=(dim, seq_len),
                    strides=(seq_len, 1),
                    offsets=(di * D_CHK_SIZE, 0),
                    block_shape=(D_CHK_SIZE, T_CHK_SIZE),
                    order=(1, 0),
                ),
                boundary_check=(0, 1),
                padding_option="zero",
            )
            x_b = tl.extra.cann.extension.insert_slice(
                st_b,
                x_b_tmp,
                (0, width - 1),
                (D_CHK_SIZE, T_CHK_SIZE),
                (1, 1),
            )
        else:
            x_b = tl.load(
                tl.make_block_ptr(
                    x_ptr + bi * dim * seq_len,
                    shape=(dim, seq_len),
                    strides=(seq_len, 1),
                    offsets=(
                        di * D_CHK_SIZE,
                        ti * T_CHK_SIZE - (width - 1),
                    ),
                    block_shape=(
                        D_CHK_SIZE,
                        T_CHK_SIZE + width - 1,
                    ),
                    order=(1, 0),
                ),
                boundary_check=(0, 1),
                padding_option="zero",
            )

        out_block = tl.zeros(
            (D_CHK_SIZE, T_CHK_SIZE), dtype=tl.float32
        )
        x_b = x_b.to(tl.float32)

        new_state_start_off = seq_len - state_len
        t_start_off = ti * T_CHK_SIZE - (width - 1)
        t_end_off = (ti + 1) * T_CHK_SIZE
        if t_end_off >= new_state_start_off:
            nst_off_y0 = (
                di * D_CHK_SIZE
                + tl.arange(0, D_CHK_SIZE)[:, None]
            )
            nst_off_y1 = tl.arange(0, state_len)[None, :]
            nst_mask = (
                (nst_off_y0 < dim) & (nst_off_y1 < state_len)
            )
            block_ptr_st = (
                bi * dim * state_len
                + nst_off_y0 * state_len
                + nst_off_y1
            )

            if new_state_start_off < 0:
                cs_src_idx = seq_len + nst_off_y1
                cs_mask = (
                    nst_mask
                    & (cs_src_idx < state_len)
                    & (cs_src_idx >= 0)
                )
                cs_off = (
                    bi * dim * state_len
                    + nst_off_y0 * state_len
                    + cs_src_idx
                )
                cs_val = tl.load(
                    conv_state_ptr + cs_off,
                    mask=cs_mask,
                    other=0,
                ).to(conv_state_update_ptr.dtype.element_ty)
                tl.store(
                    conv_state_update_ptr + block_ptr_st,
                    cs_val,
                    mask=cs_mask,
                )

            src_col_off = (
                new_state_start_off - t_start_off + nst_off_y1
            )
            src_valid = (
                (src_col_off >= 0)
                & (src_col_off < T_CHK_SIZE + width - 1)
            )
            x_st_off = (
                bi * dim * seq_len
                + nst_off_y0 * seq_len
                + (new_state_start_off + nst_off_y1)
            )
            x_st_mask = (
                nst_mask
                & src_valid
                & ((new_state_start_off + nst_off_y1) < seq_len)
                & ((new_state_start_off + nst_off_y1) >= 0)
            )
            x_st = tl.load(
                x_ptr + x_st_off,
                mask=x_st_mask,
                other=0,
            ).to(conv_state_update_ptr.dtype.element_ty)
            tl.store(
                conv_state_update_ptr + block_ptr_st,
                x_st,
                mask=x_st_mask,
            )

        for owi in tl.range(0, width):
            new_x = tl.extra.cann.extension.extract_slice(
                x_b,
                (0, owi),
                (D_CHK_SIZE, T_CHK_SIZE),
                (1, 1),
            )
            w_chl_wi = tl.extra.cann.extension.extract_slice(
                w,
                (0, owi),
                (D_CHK_SIZE, 1),
                (1, 1),
            )
            out_block += new_x * w_chl_wi

        if SILU_ACTIVATION:
            out_block = out_block / (1.0 + tl.exp(-out_block))
        out_block = out_block.to(out_ptr.type.element_ty)
        out_off_d = (
            di * D_CHK_SIZE + tl.arange(0, D_CHK_SIZE)[:, None]
        )
        out_off_t = (
            ti * T_CHK_SIZE + tl.arange(0, T_CHK_SIZE)[None, :]
        )
        out_mask = (out_off_d < dim) & (out_off_t < out_len)
        out_off = (
            bi * dim * out_len
            + out_off_d * out_len
            + out_off_t
        )
        tl.store(out_ptr + out_off, out_block, mask=out_mask)


def update_case_id(profile: dict[str, object]) -> str:
    activation = profile["activation"] or "none"
    return (
        f"test_causal_conv1d_update_state_B{profile['B']}"
        f"_T{profile['T']}_D{profile['D']}_W{profile['W']}"
        f"_{profile['dtype']}_{activation}"
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


def run_update(
    profile: dict[str, object], device: torch.device
) -> None:
    seed = 42
    batch = int(profile["B"])
    seq_len = int(profile["T"])
    dim = int(profile["D"])
    width = int(profile["W"])
    dtype_name = str(profile["dtype"])
    dtype = DTYPES[dtype_name]
    case_id = update_case_id(profile)

    # This order exactly matches the verified package generator.
    torch.manual_seed(seed)
    hidden_cpu = torch.randn(
        batch,
        dim,
        seq_len,
        dtype=dtype,
        device="cpu",
    )
    state_cpu = torch.randn(
        batch,
        dim,
        width,
        dtype=dtype,
        device="cpu",
    )
    weight_cpu = torch.randn(
        dim,
        width,
        dtype=dtype,
        device="cpu",
    )

    state_golden = hidden_cpu[:, :, -width:].contiguous()
    with torch.no_grad():
        value = F.conv1d(
            torch.cat([state_cpu, hidden_cpu], dim=-1).to(
                weight_cpu.dtype
            ),
            weight_cpu.unsqueeze(1),
            bias=None,
            padding=0,
            groups=dim,
        )[:, :, -seq_len:]
        if profile["activation"] in ("silu", "swish"):
            value = F.silu(value)
        out_golden = value.to(hidden_cpu.dtype).contiguous()

    hidden = hidden_cpu.to(device)
    state = state_cpu.to(device)
    weight = weight_cpu.to(device)
    state_update = torch.empty_like(state)
    out = torch.empty_like(hidden)

    t_chunk = 256
    d_chunk = 16
    num_t_chunks = triton.cdiv(seq_len, t_chunk)
    num_d_chunks = triton.cdiv(dim, d_chunk)
    head_tile = (
        width
        if (seq_len % t_chunk) > width
        else (width - seq_len % t_chunk) % t_chunk
    )
    grid = (num_vector_cores(),)

    print(
        f"RUN case={case_id} "
        f"kernel=causal_conv1d_update_kernel_bdt_fwd "
        f"seed={seed} grid={grid} T_CHK_SIZE={t_chunk} "
        f"D_CHK_SIZE={d_chunk} NUM_T_CHK={num_t_chunks} "
        f"NUM_D_CHK={num_d_chunks}"
    )
    causal_conv1d_update_kernel_bdt_fwd[grid](
        hidden,
        state,
        state_update,
        weight,
        None,
        None,
        out,
        batch=batch,
        dim=dim,
        state_len=width,
        seq_len=seq_len,
        width=width,
        out_len=seq_len,
        x_batch_stride=dim * seq_len,
        conv_batch_stride=dim * width,
        out_batch_stride=dim * seq_len,
        HAS_BIAS=False,
        SILU_ACTIVATION=profile["activation"] in ("silu", "swish"),
        T_CHK_SIZE=t_chunk,
        D_CHK_SIZE=d_chunk,
        NUM_T_CHK=num_t_chunks,
        NUM_D_CHK=num_d_chunks,
        ST_STORE_HEAD_TILE_SIZE=int(head_tile),
        # multibuffer=True,
        # limit_auto_multi_buffer_of_local_buffer="no-limit",
        # num_warps=4,
        # num_stages=3,
    )
    torch.npu.synchronize()
    compare("conv_state_update", state_update, state_golden, dtype_name)
    compare("out", out, out_golden, dtype_name)

    del (
        hidden,
        state,
        weight,
        state_update,
        out,
        hidden_cpu,
        state_cpu,
        weight_cpu,
        state_golden,
        out_golden,
    )
    torch.npu.empty_cache()


def all_cases() -> tuple[
    tuple[str, dict[str, object]], ...
]:
    return tuple(
        (update_case_id(profile), profile)
        for profile in UPDATE_PROFILES
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
            run_update(profile, device)
            print(f"CASE PASS: {case_id}")
        finally:
            torch.npu.empty_cache()
    print(f"RESULT: PASS ({len(selected)} cases)")


def test_causal_conv1d_update_state_standalone() -> None:
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
