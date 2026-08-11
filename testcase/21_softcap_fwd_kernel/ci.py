"""Daily A5 E2E accuracy gate for ``softcap_fwd_kernel``.

Source contract:
  HighPriority50Operators/testcase/21_softcap_fwd_kernel/softcap_npu.py

The Triton kernel body, FP16 dtype, softcap=50, FP32 tanh evaluation and
FP16 rounding point are preserved.  The source's profiler and backward pass
are intentionally outside this forward-kernel gate.  Autotuning is replaced
by the deterministic BLOCK_SIZE=2048/BLOCK_NUM=1 specialization already
validated on A5.

Coverage:
  * (1024, 1024): original full-shape contract, 512 programs.
  * (1048613,): non-multiple-of-2048 masked load/store tail, 513 programs.
"""

import os


os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_BACKEND", "opentile")
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")


def _npu_available():
    try:
        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


if not _npu_available():
    pytest.skip(
        "torch_npu is available, but no NPU device is visible",
        allow_module_level=True,
    )

_DEVICE_INDEX = int(
    os.environ.get("OPENTILE_TEST_DEVICE", "0").rsplit(":", 1)[-1]
)
torch.npu.set_device(_DEVICE_INDEX)

import triton
import triton.language as tl
# from triton.language.extra.cuda import libdevice
from triton.language.extra.ascend import libdevice


BLOCK_SIZE = 2048
BLOCK_NUM = 1
SOFTCAP = 50.0
ATOL = 1.0e-3
RTOL = 1.0e-3


@triton.jit
def softcap_fwd_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    softcap,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_NUM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    for i in range(BLOCK_NUM):
        block_start = pid * BLOCK_SIZE * BLOCK_NUM + i * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = softcap * (libdevice.tanh(x.to(tl.float32) / softcap)).to(x.dtype)
        tl.store(y_ptr + offsets, y, mask=mask)


def _select_device():
    return torch.device(f"npu:{_DEVICE_INDEX}")


def _run_case(shape, seed, case_name):
    device = _select_device()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x_cpu = torch.randn(shape, dtype=torch.float16, generator=generator)

    # Preserve the source computation's FP32 tanh followed by FP16 rounding.
    expected = (
        SOFTCAP
        * torch.tanh(x_cpu.to(torch.float32) / SOFTCAP).to(torch.float16)
    )

    x = x_cpu.to(device)
    actual = torch.empty_like(x)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE * BLOCK_NUM),)
    softcap_fwd_kernel[grid](
        x,
        actual,
        n_elements,
        SOFTCAP,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_NUM=BLOCK_NUM,
        debug=True,
    )
    torch.npu.synchronize()

    actual_cpu = actual.cpu()
    finite = int(torch.isfinite(actual_cpu).sum().item())
    if finite != n_elements:
        raise AssertionError(
            f"{case_name}: non-finite output "
            f"finite={finite}/{n_elements}"
        )

    difference = (actual_cpu - expected).abs()
    close = torch.isclose(actual_cpu, expected, atol=ATOL, rtol=RTOL)
    passed = int(close.sum().item())
    max_abs = float(difference.max().item())
    mean_abs = float(difference.float().mean().item())
    if passed != n_elements:
        bad_flat = int((~close).reshape(-1).nonzero()[0].item())
        actual_value = float(actual_cpu.reshape(-1)[bad_flat].item())
        expected_value = float(expected.reshape(-1)[bad_flat].item())
        raise AssertionError(
            f"{case_name}: compare failed pass={passed}/{n_elements} "
            f"first_bad={bad_flat} actual={actual_value:.8g} "
            f"expected={expected_value:.8g} "
            f"max_abs={max_abs:.8g} mean_abs={mean_abs:.8g} "
            f"atol={ATOL} rtol={RTOL}"
        )

    print(
        f"[CI_PASS] kernel=softcap_fwd_kernel case={case_name} "
        f"shape={tuple(shape)} dtype=fp16 seed={seed} "
        f"programs={grid[0]} pass={passed}/{n_elements} "
        f"max_abs={max_abs:.8g} mean_abs={mean_abs:.8g} "
        f"atol={ATOL} rtol={RTOL}",
        flush=True,
    )


@pytest.mark.parametrize(
    "shape,seed,case_name",
    [
        ((1024, 1024), 2101, "full_1024x1024"),
        ((1048613,), 2102, "tail_1048613"),
    ],
    ids=["full_1024x1024", "tail_1048613"],
)
def test_softcap_fwd_kernel(shape, seed, case_name):
    _run_case(shape, seed, case_name)


if __name__ == "__main__":
    for parameters in (
        ((1024, 1024), 2101, "full_1024x1024"),
        ((1048613,), 2102, "tail_1048613"),
    ):
        _run_case(*parameters)
