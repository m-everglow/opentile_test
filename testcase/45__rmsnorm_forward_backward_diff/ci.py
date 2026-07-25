from __future__ import annotations

import os

import pytest
import torch
import torch_npu  # noqa: F401
from triton.runtime import driver

from ci_rmsnorm import rmsnorm_backward
from ci_rmsnorm import rmsnorm_forward


EPS = 1e-6
TEST_SEED = int(os.environ.get("OPENTILE_TEST_SEED", "2026"))
TOLERANCE = {
    torch.float32: 1e-5,
    torch.bfloat16: 5e-3,
}
CASES = [
    pytest.param((32, 1024), torch.float32, id="fp32-aligned-32x1024"),
    pytest.param((77, 489), torch.float32, id="fp32-tail-77x489"),
    pytest.param((32, 1024), torch.bfloat16, id="bf16-aligned-32x1024"),
    pytest.param((77, 489), torch.bfloat16, id="bf16-tail-77x489"),
]


def _reference(
    x: torch.Tensor, weight: torch.Tensor, dy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_f32 = x.float()
    dy_f32 = dy.float()
    rstd = torch.rsqrt(x_f32.square().mean(dim=-1) + EPS)
    y = (x_f32 * rstd[:, None] * weight[None, :]).to(x.dtype)

    m = dy_f32 * weight[None, :]
    dx = (
        rstd[:, None] * m
        - x_f32
        * rstd[:, None].pow(3)
        * (m * x_f32).mean(dim=-1, keepdim=True)
    ).to(x.dtype)
    dw = (dy_f32 * x_f32 * rstd[:, None]).sum(dim=0)
    return y, dx, dw


@pytest.mark.parametrize("shape,dtype", CASES)
def test_rmsnorm_forward_backward_diff(shape: tuple[int, int], dtype: torch.dtype):
    target = driver.active.get_current_target()
    backend = str(target.backend)
    assert backend == "opentile" or backend.startswith("opentile_")
    assert driver.active.get_active_torch_device().type == "npu"

    generator = torch.Generator(device="cpu").manual_seed(TEST_SEED)
    x_cpu = torch.randn(shape, generator=generator, dtype=dtype)
    weight_cpu = torch.randn(shape[-1], generator=generator, dtype=torch.float32)
    dy_cpu = torch.randn(shape, generator=generator, dtype=dtype)
    expected_y, expected_dx, expected_dw = _reference(x_cpu, weight_cpu, dy_cpu)

    device = torch.device("npu", torch.npu.current_device())
    x = x_cpu.to(device)
    weight = weight_cpu.to(device)
    dy = dy_cpu.to(device)
    actual_y, rstd = rmsnorm_forward(x, weight, EPS)
    actual_dx, actual_dw = rmsnorm_backward(dy, x, weight, rstd)
    torch.npu.synchronize()

    actual_y_cpu = actual_y.cpu()
    actual_dx_cpu = actual_dx.cpu()
    actual_dw_cpu = actual_dw.cpu()
    assert torch.isfinite(actual_y_cpu).all(), "forward left NaN/sentinel values"
    assert torch.isfinite(actual_dx_cpu).all(), "backward left NaN/sentinel values"

    tolerance = TOLERANCE[dtype]
    print(f"OpenTile backend={backend}, seed={TEST_SEED}, shape={shape}, dtype={dtype}")
    for name, actual, expected in (
        ("y", actual_y_cpu.float(), expected_y.float()),
        ("dx", actual_dx_cpu.float(), expected_dx.float()),
        ("dw", actual_dw_cpu.float(), expected_dw.float()),
    ):
        difference = (actual - expected).abs()
        denominator = expected.abs().clamp_min(torch.finfo(torch.float32).tiny)
        mismatch = ~torch.isclose(actual, expected, atol=tolerance, rtol=tolerance)
        print(
            f"{name}: max_abs={difference.max().item():.8g}, "
            f"max_rel={(difference / denominator).max().item():.8g}, "
            f"mismatches={mismatch.sum().item()}/{actual.numel()}"
        )
        torch.testing.assert_close(
            actual, expected, atol=tolerance, rtol=tolerance, equal_nan=False
        )


if __name__ == "__main__":
    # test_rmsnorm_forward_backward_diff((32, 1024), torch.float32)
    pass
