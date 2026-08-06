"""Standalone NPU softcap kernel + kernel-faithful golden.

Run directly:
  python3 test_softcap_forward_backward_diff_standalone.py

Run with pytest:
  pytest -s --assert=plain test_softcap_forward_backward_diff_standalone.py

The golden reproduces the kernel's exact arithmetic: tanh is computed as
2*sigmoid(2z)-1 and the forward result is rounded to the input dtype *before*
multiplying by softcap (matching `softcap * tanh_x.to(x.dtype)`).  Using
torch.tanh without the intermediate dtype rounding differs from the kernel by
up to 0.5 bf16 ULP on rounding boundaries, which softcap=50 amplifies past the
bf16 tolerance on large shapes.
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

SEED = 2026071840
SHAPES = ((128, 128), (999, 9999), (1024, 10240))
DTYPES = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
}
TOLERANCES = {
    "f32": (1e-5, 1e-5),
    "f16": (1e-3, 1e-3),
    "bf16": (5e-3, 5e-3),
}
SOFTCAP = 50.0
BLOCK_SIZE = 512


@triton.jit
def _softcap_fwd_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    softcap,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    for block_id in range(pid, num_blocks, grid_size):
        block_start = block_id * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        # tanh(z) = 2*sigmoid(2z) - 1
        z = x.to(tl.float32) / softcap
        tanh_x = 2.0 * tl.sigmoid(2.0 * z) - 1.0
        y = softcap * tanh_x.to(x.dtype)
        tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _softcap_bwd_kernel(
    dy_ptr,
    x_ptr,
    dx_ptr,
    n_elements,
    softcap,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    for block_id in range(pid, num_blocks, grid_size):
        block_start = block_id * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        dy = tl.load(dy_ptr + offsets, mask=mask)
        x = tl.load(x_ptr + offsets, mask=mask)
        # Fix: keep tanh and derivative in FP32, preserve dy*softcap rounding.
        # tanh(z) = 2*sigmoid(2z) - 1
        z = x.to(tl.float32) / softcap
        tanh_x = 2.0 * tl.sigmoid(2.0 * z) - 1.0
        scaled_dy = (dy.to(tl.float32) * softcap).to(dy.dtype).to(tl.float32)
        dx = scaled_dy * (1.0 - tanh_x * tanh_x) / softcap
        tl.store(dx_ptr + offsets, dx, mask=mask)


def _num_vector_cores():
    return triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]


def _setup_npu() -> torch.device:
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("torch_npu is required on the real-hardware host") from exc
    device_id = int(os.environ.get("OPENTILE_TEST_DEVICE", "0"))
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is false")
    torch.npu.set_device(device_id)
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


def _softcap_tanh_f32(x_f32: torch.Tensor) -> torch.Tensor:
    """Kernel-faithful tanh: 2*sigmoid(2z)-1 in FP32."""
    z = x_f32 / SOFTCAP
    return 2.0 * torch.sigmoid(2.0 * z) - 1.0


def run_one(dtype_name: str, shape: tuple[int, int], device: torch.device) -> None:
    torch_dtype = DTYPES[dtype_name]
    rows, cols = shape
    numel = rows * cols

    torch.manual_seed(SEED)
    x_cpu = torch.randn(rows, cols, dtype=torch_dtype, device="cpu", requires_grad=True)
    with torch.no_grad():
        # Golden forward, kernel-faithful:
        #   y = softcap * tanh(x.float()/softcap).to(x.dtype)
        # The intermediate .to(x.dtype) rounding happens BEFORE multiplying by
        # softcap (matches `softcap * tanh_x.to(x.dtype)`), so reproduce it.
        x_f32 = x_cpu.float()
        tanh_x = _softcap_tanh_f32(x_f32)
        tanh_rounded = tanh_x.to(torch_dtype)          # kernel's intermediate cast
        y_golden = (SOFTCAP * tanh_rounded.float()).to(torch_dtype)

        dy_cpu = torch.randn_like(y_golden)
        # Golden backward, kernel-faithful:
        #   dx = round(dy*softcap) * (1 - tanh_fp32^2) / softcap
        dy_f32 = dy_cpu.float()
        tanh_f32 = _softcap_tanh_f32(x_f32)
        scaled_dy = (dy_f32 * SOFTCAP).to(torch_dtype).float()
        dx_golden = (scaled_dy * (1.0 - tanh_f32 * tanh_f32) / SOFTCAP).to(torch_dtype)

    x = x_cpu.detach().to(device).contiguous()
    dy = dy_cpu.to(device).contiguous()
    y = torch.empty_like(x)
    dx = torch.empty_like(x)
    block_dim = min(_num_vector_cores(), triton.cdiv(numel, BLOCK_SIZE))
    grid = (block_dim,)

    _softcap_fwd_kernel[grid](
        x,
        y,
        numel,
        SOFTCAP,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()
    _compare(f"Y_{dtype_name}_{rows}x{cols}", y, y_golden, dtype_name)

    _softcap_bwd_kernel[grid](
        dy,
        x,
        dx,
        numel,
        SOFTCAP,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()
    _compare(f"DX_{dtype_name}_{rows}x{cols}", dx, dx_golden, dtype_name)
    del x, dy, y, dx, x_cpu, dy_cpu, y_golden, dx_golden
    torch.npu.empty_cache()


def test_softcap_forward_backward_diff_standalone() -> None:
    device = _setup_npu()
    for dtype_name in ("f32", "f16", "bf16"):
        for shape in SHAPES:
            run_one(dtype_name, shape, device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("all", *DTYPES), default="all")
    parser.add_argument(
        "--shape",
        choices=("all", *(f"{rows}x{cols}" for rows, cols in SHAPES)),
        default="all",
    )
    args = parser.parse_args()
    device = _setup_npu()
    dtype_names = tuple(DTYPES) if args.dtype == "all" else (args.dtype,)
    shapes = (
        SHAPES
        if args.shape == "all"
        else (tuple(int(value) for value in args.shape.split("x")),)
    )
    for dtype_name in dtype_names:
        for shape in shapes:
            run_one(dtype_name, shape, device)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()