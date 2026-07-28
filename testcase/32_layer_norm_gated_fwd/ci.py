"""Physical-NPU numerical checks for LayerNorm + SiLU gated forward."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
os.environ.setdefault("OPENTILE_KERNEL_MODE", "aiv")
os.environ.setdefault("OPENTILE_ENABLE_APPROX", "0")
os.environ.setdefault("OPENTILE_ENABLE_FTZ", "0")

import pytest  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401 -- registers torch.npu
import torch.nn.functional as F  # noqa: E402


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))


def _read_test_seed() -> int:
    raw_seed = os.environ.get("OPENTILE_TEST_SEED", "42")
    try:
        return int(raw_seed)
    except ValueError as error:
        raise ValueError(
            f"OPENTILE_TEST_SEED must be an integer, got {raw_seed!r}"
        ) from error


TEST_SEED = _read_test_seed()
TOLERANCE = {
    torch.float32: 1e-5,
    torch.bfloat16: 5e-3,
    torch.float16: 1e-3,
}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "on", "yes"}


def _npu_available() -> bool:
    try:
        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


if not _npu_available():
    raise RuntimeError(
        "test_fused_norm_gate.py requires a visible physical Ascend NPU"
    )

DEVICE = torch.device("npu", torch.npu.current_device())

from ci_fused_norm_gate import (  # noqa: E402
    assert_opentile_backend,
    get_npu_properties,
    layer_norm_gated_fwd,
)


def _compile_only() -> bool:
    return _env_flag("TRITON_COMPILE_ONLY") or _env_flag("OPENTILE_COMPILE_ONLY")


def _to_npu(value: torch.Tensor) -> torch.Tensor:
    return value.contiguous().to(DEVICE)


def _to_cpu(value: torch.Tensor) -> torch.Tensor:
    return value.cpu()


def _poisoned_npu(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return _to_npu(torch.full(shape, float("nan"), dtype=dtype))


def _assert_device_route() -> None:
    backend = assert_opentile_backend()
    properties = get_npu_properties()
    assert int(properties["num_aicore"]) > 0
    assert int(properties["num_vectorcore"]) > 0
    print(
        f"[ROUTE] backend={backend} "
        f"device={DEVICE} num_aicore={properties['num_aicore']} "
        f"num_vectorcore={properties['num_vectorcore']}",
        flush=True,
    )


def _make_inputs(
    rows: int,
    feature_size: int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Generate a fresh, directly typed random sample for one pytest case."""
    generator = torch.Generator(device="cpu").manual_seed(TEST_SEED)
    inputs = {
        "x": torch.randn(
            rows, feature_size, generator=generator, dtype=dtype
        ).mul_(0.5),
        "g": torch.randn(
            rows, feature_size, generator=generator, dtype=dtype
        ).mul_(0.5),
        "weight": torch.randn(
            feature_size, generator=generator, dtype=dtype
        ).mul_(0.1).add_(1.0),
        "bias": torch.randn(
            feature_size, generator=generator, dtype=dtype
        ).mul_(0.1),
    }
    return {name: value.contiguous() for name, value in inputs.items()}


def _reference(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values_x = x.float()
    mean = values_x.mean(dim=-1)
    centered = values_x - mean[:, None]
    rstd = torch.rsqrt(centered.square().mean(dim=-1) + eps)
    output = (
        (centered * rstd[:, None] * weight.float() + bias.float())
        * F.silu(g.float())
    ).to(x.dtype)
    return output, mean, rstd


def _assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> None:
    actual_cpu = _to_cpu(actual).float()
    expected_cpu = expected.cpu().float()
    assert actual_cpu.shape == expected_cpu.shape
    assert torch.isfinite(actual_cpu).all(), f"{name} contains non-finite values"
    assert torch.isfinite(expected_cpu).all(), (
        f"{name} reference contains non-finite values"
    )
    diff = (actual_cpu - expected_cpu).abs()
    tolerance = atol + rtol * expected_cpu.abs()
    mismatches = int((diff > tolerance).sum().item())
    max_abs = float(diff.max().item())
    relative = diff / expected_cpu.abs().clamp_min(1e-12)
    max_rel = float(relative.max().item())
    print(
        f"[NUMERICS] {name}: max_abs={max_abs:.8e} "
        f"max_rel={max_rel:.8e} mismatches={mismatches}/{actual_cpu.numel()}",
        flush=True,
    )
    assert mismatches == 0


@pytest.mark.parametrize(
    "rows,feature_size,dtype",
    [
        # ============================ FP32 cases ============================
        pytest.param(8, 64, torch.float32, id="fp32-t8-d64-aligned"),
        # pytest.param(200, 48, torch.float32, id="fp32-t200-d48-tail"),
        # ============================ BF16 cases ============================
        pytest.param(8, 64, torch.bfloat16, id="bf16-t8-d64-aligned"),
        # pytest.param(200, 48, torch.bfloat16, id="bf16-t200-d48-tail"),
        # ============================ FP16 cases ============================
        pytest.param(8, 64, torch.float16, id="fp16-t8-d64-aligned"),
        # pytest.param(200, 48, torch.float16, id="fp16-t200-d48-tail"),
    ],
)
def test_layer_norm_gated_fwd_opentile(
    rows: int,
    feature_size: int,
    dtype: torch.dtype,
) -> None:
    """Execute LayerNorm + affine + SiLU gate through Triton/OpenTile."""
    _assert_device_route()
    eps = 1e-5
    tolerance = TOLERANCE[dtype]
    print(
        f"[DATA] seed={TEST_SEED} dtype={dtype} "
        f"shape=({rows}, {feature_size})",
        flush=True,
    )
    inputs = _make_inputs(rows, feature_size, dtype)
    npu = {name: _to_npu(value) for name, value in inputs.items()}

    output = _poisoned_npu((rows, feature_size), dtype)
    mean = _poisoned_npu((rows,), torch.float32)
    rstd = _poisoned_npu((rows,), torch.float32)
    actual, actual_mean, actual_rstd, saved_x = layer_norm_gated_fwd(
        npu["x"],
        npu["g"],
        npu["weight"],
        npu["bias"],
        activation="silu",
        eps=eps,
        is_rms_norm=False,
        out=output,
        mean_out=mean,
        rstd_out=rstd,
    )
    if _compile_only():
        return

    torch.npu.synchronize()
    assert actual.data_ptr() == output.data_ptr()
    assert actual_mean is not None
    assert actual_mean.data_ptr() == mean.data_ptr()
    assert actual_rstd.data_ptr() == rstd.data_ptr()
    assert saved_x.data_ptr() == npu["x"].data_ptr()

    expected, expected_mean, expected_rstd = _reference(
        inputs["x"],
        inputs["g"],
        inputs["weight"],
        inputs["bias"],
        eps,
    )
    assert actual.dtype == expected.dtype == dtype
    case = f"{dtype}.t{rows}.d{feature_size}"
    _assert_close(
        f"{case}.y",
        actual,
        expected,
        atol=tolerance,
        rtol=tolerance,
    )
    _assert_close(
        f"{case}.mean",
        actual_mean,
        expected_mean,
        atol=tolerance,
        rtol=tolerance,
    )
    _assert_close(
        f"{case}.rstd",
        actual_rstd,
        expected_rstd,
        atol=tolerance,
        rtol=tolerance,
    )


if __name__ == "__main__":
    # Normal execution goes through pytest. Uncomment exactly one call only
    # when isolating a specific dtype/shape failure, then comment it again.
    # test_layer_norm_gated_fwd_opentile(8, 64, torch.float32)
    # test_layer_norm_gated_fwd_opentile(200, 48, torch.float16)
    pass
