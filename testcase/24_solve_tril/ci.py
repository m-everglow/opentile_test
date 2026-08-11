"""Standalone NPU solve_tril kernel + kernel-faithful golden.

Uses 1-D grid launch to avoid the lowerBroadcast short-row code path
introduced by 7947fb6, which produces wrong L1 base addresses for
multi-tile cases with static strides.

Run directly:
  python3 test_solve_tril_forward_backward_diff_standalone_1d.py

Run with pytest:
  pytest -s --assert=plain test_solve_tril_forward_backward_diff_standalone_1d.py
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
import torch.nn.functional as F
from triton.runtime import driver


_KERNEL_MODES = {
    "_solve_tril_16x16_kernel": "aiv",
    "_merge_16x16_to_32x32_kernel": "mix",
    "_merge_16x16_to_64x64_kernel": "mix",
}


def _install_kernel_mode_launcher() -> None:
    """Supply the expected runtime mode for this mixed AIV/MIX testcase."""
    active_driver = driver.active
    base_launcher = active_driver.launcher_cls
    if getattr(base_launcher, "_solve_tril_kernel_modes", False):
        return

    class SolveTrilLauncher(base_launcher):
        _solve_tril_kernel_modes = True

        def __init__(self, src, metadata):
            super().__init__(src, metadata)
            fn = getattr(src, "fn", None)
            kernel_name = getattr(fn, "__name__", metadata.name)
            mode = _KERNEL_MODES.get(kernel_name)
            if mode is None:
                raise RuntimeError(
                    f"missing runtime mode for solve_tril kernel {kernel_name!r}"
                )
            active_driver.utils.register_cv_mode(metadata.name, mode)

    active_driver.launcher_cls = SolveTrilLauncher

SEED = 2026071840
# (B, T, H, chunk_size) — matches FLA test_solve_tril parametrization
SHAPES = (
    (1, 63, 1, 16),       # single-tile bt16
    (1, 64, 1, 32),       # single-tile bt32
    (1, 64, 1, 64),       # single-tile bt64
    (2, 500, 4, 32),      # fla-original-scale bt32
    (2, 1000, 5, 64),     # fla-original-scale bt64
    (3, 1024, 6, 64),     # larger scale bt64
)
DTYPES = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
}
TOLERANCES = {
    "f32": (1e-4, 1e-4),
    "f16": (1e-3, 1e-3),
    "bf16": (1e-3, 1e-3),
}


# ============================================================
# NPU solve_tril kernels (1-D grid, from fla/ops/utils/backends/triton_ascend/solve_tril.py)
# ============================================================

@triton.jit(do_not_specialize=['T'])
def _solve_tril_16x16_kernel(
    A, Ai, T,
    H: tl.constexpr, BT: tl.constexpr,
    BH: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    pid = tl.program_id(0)
    i_t = pid // BH + NT_OFFSET
    i_bh = pid % BH + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H
    bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    A = A + (bos * H + i_h) * BT
    Ai = Ai + (bos * H + i_h) * 16

    offset = (i_t * 16) % BT
    p_A = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * 16, offset), (16, 16), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
    b_A = tl.where(m_A, b_A, 0)
    b_A = -b_A

    for i in range(2, min(16, T - i_t * 16)):
        b_a = -tl.load(A + (i_t * 16 + i) * H * BT + o_i + offset)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        b_A = tl.where((o_i == i)[:, None], b_a, b_A)
    b_A += m_I

    p_Ai = tl.make_block_ptr(Ai, (T, 16), (H * 16, 1), (i_t * 16, 0), (16, 16), (1, 0))
    tl.store(p_Ai, b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def _merge_16x16_to_32x32_kernel(
    A, Ai, T,
    H: tl.constexpr, BT: tl.constexpr,
    BH: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    pid = tl.program_id(0)
    i_t = pid // BH + NT_OFFSET
    i_bh = pid % BH + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H
    bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    p_A_11 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_A_22 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)

    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H * BT + o_i)
        b_a_11 = tl.where(o_i < i, b_a_11, 0.)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)
        b_a_22 = tl.where(o_i < i - 16, b_a_22, 0.)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)

    b_Ai_11 += m_I
    b_Ai_22 += m_I

    p_A_21 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21), b_Ai_11)

    p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
    tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
    tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))


@triton.jit(do_not_specialize=['T'])
def _merge_16x16_to_64x64_kernel(
    A, Ai, T,
    H: tl.constexpr, BT: tl.constexpr,
    BH: tl.constexpr,
    NT_OFFSET: tl.constexpr, BH_OFFSET: tl.constexpr,
):
    pid = tl.program_id(0)
    i_t = pid // BH + NT_OFFSET
    i_bh = pid % BH + BH_OFFSET
    i_b, i_h = i_bh // H, i_bh % H
    bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    p_A_11 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_A_22 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    p_A_33 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
    p_A_44 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
    b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_33 = tl.load(p_A_33, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_44 = tl.load(p_A_44, boundary_check=(0, 1)).to(tl.float32)

    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H * BT + o_i)
        b_a_11 = tl.where(o_i < i, b_a_11, 0.)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)
        b_a_22 = tl.where(o_i < i - 16, b_a_22, 0.)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a_33 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 32)
        b_a_33 = tl.where(o_i < i - 32, b_a_33, 0.)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a_44 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 48)
        b_a_44 = tl.where(o_i < i - 48, b_a_44, 0.)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    p_A_21 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_A_31 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
    p_A_32 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
    p_A_41 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
    p_A_42 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
    p_A_43 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
    b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
    b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
    b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
    b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
    b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)

    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21), b_Ai_11)
    b_Ai_32 = -tl.dot(tl.dot(b_Ai_33, b_A_32), b_Ai_22)
    b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43), b_Ai_33)
    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11) + tl.dot(b_A_32, b_Ai_21),
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22) + tl.dot(b_A_43, b_Ai_32),
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11) + tl.dot(b_A_42, b_Ai_21) + tl.dot(b_A_43, b_Ai_31),
    )

    p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    p_Ai_33 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
    p_Ai_44 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
    p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_Ai_31 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
    p_Ai_32 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
    p_Ai_41 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
    p_Ai_42 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
    p_Ai_43 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
    tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
    tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
    tl.store(p_Ai_33, b_Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
    tl.store(p_Ai_44, b_Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
    tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
    tl.store(p_Ai_31, b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
    tl.store(p_Ai_32, b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
    tl.store(p_Ai_41, b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
    tl.store(p_Ai_42, b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
    tl.store(p_Ai_43, b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))


def _setup_npu() -> torch.device:
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("torch_npu is required on the real-hardware host") from exc
    device_id = int(os.environ.get("OPENTILE_TEST_DEVICE", "0"))
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is false")
    torch.npu.set_device(device_id)
    _install_kernel_mode_launcher()
    return torch.device(f"npu:{device_id}")


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


def _launch_kernel(A: torch.Tensor, BT: int) -> torch.Tensor:
    """Launch solve_tril kernel on NPU with 1-D grid. A shape: [B, T, H, BT]"""
    B, T, H, _ = A.shape
    NT = triton.cdiv(T, BT)
    BH = B * H
    Ai = torch.zeros_like(A)
    if BT == 16:
        kernel = _solve_tril_16x16_kernel
    elif BT == 32:
        kernel = _merge_16x16_to_32x32_kernel
    elif BT == 64:
        kernel = _merge_16x16_to_64x64_kernel
    else:
        raise ValueError(f"BT={BT} not supported, must be 16/32/64")

    grid = (NT * BH,)
    kernel[grid](
        A, Ai, T,
        H=H, BT=BT,
        BH=BH,
        NT_OFFSET=0, BH_OFFSET=0,
    )
    return Ai


def run_one(dtype_name: str, shape: tuple[int, int, int, int], device: torch.device) -> None:
    torch_dtype = DTYPES[dtype_name]
    B, T, H, chunk_size = shape

    torch.manual_seed(SEED)
    # FLA test: do not randomly initialize A otherwise the inverse is not stable
    k = F.normalize(torch.randn((B, H, T, 64), dtype=torch.float32, device="cpu"), dim=-1)
    padding_size = (chunk_size - T % chunk_size) % chunk_size
    k_padded = F.pad(k, (0, 0, 0, padding_size, 0, 0, 0, 0))
    k_padded = k_padded.reshape(B, H, -1, chunk_size, 64)
    # A: [B, H, NT, chunk_size, chunk_size], strictly lower triangular
    A_5d = (k_padded @ k_padded.transpose(-1, -2)).tril(-1)

    # Kernel input: [B, T, H, BT]
    A_in = A_5d.reshape(B, H, -1, chunk_size)[:, :, :T, :].transpose(1, 2).contiguous()

    # Golden must use the same dtype-quantized input consumed by the kernel.
    # Converting only the final reference to bf16 compares against information
    # that is no longer present in A_dev and creates deterministic one-ULP errors.
    ref_input_5d = A_5d.to(torch_dtype).float()
    ref_5d = torch.inverse(ref_input_5d + torch.eye(chunk_size, dtype=torch.float32))
    ref = ref_5d.reshape(B, H, -1, chunk_size)[:, :, :T, :].transpose(1, 2).contiguous()
    ref = ref.to(torch_dtype)

    A_dev = A_in.to(torch_dtype).to(device).contiguous()
    Ai = _launch_kernel(A_dev, chunk_size)
    torch.npu.synchronize()
    _compare(f"solve_tril_{dtype_name}_B{B}T{T}H{H}bt{chunk_size}", Ai, ref, dtype_name)

    del A_dev, Ai, A_in, A_5d, ref, ref_5d, k
    torch.npu.empty_cache()


def test_solve_tril_forward_backward_diff_standalone() -> None:
    device = _setup_npu()
    for dtype_name in ("f32", "f16", "bf16"):
        for shape in SHAPES:
            run_one(dtype_name, shape, device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("all", *DTYPES), default="all")
    parser.add_argument(
        "--shape",
        choices=("all", *(f"{b}x{t}x{h}x{cs}" for b, t, h, cs in SHAPES)),
        default="all",
    )
    args = parser.parse_args()
    device = _setup_npu()
    dtype_names = tuple(DTYPES) if args.dtype == "all" else (args.dtype,)
    shapes = (
        SHAPES
        if args.shape == "all"
        else (tuple(int(v) for v in args.shape.split("x")),)
    )
    for dtype_name in dtype_names:
        for shape in shapes:
            run_one(dtype_name, shape, device)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
