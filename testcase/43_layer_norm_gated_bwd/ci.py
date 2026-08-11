# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Focused numerical validation for the standalone OpenTile specialization."""

import os
from dataclasses import dataclass

import pytest
import torch
import triton
import triton.runtime.driver as driver

from ci_kernel import layer_norm_gated_bwd_kernel

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


FP32 = torch.float32
ATOL = 1e-5
RTOL = 1e-5
EPS = 1e-5
TEST_SEED = int(os.environ.get("OPENTILE_TEST_SEED", "2026"))
CASES = [(1, 16), (3, 15)]


@dataclass(frozen=True)
class OpenTileRuntime:
    backend: str
    device: torch.device
    num_aicore: int
    num_vectorcore: int


def get_opentile_runtime() -> OpenTileRuntime:
    if torch_npu is None:
        raise RuntimeError("the physical-NPU test requires torch_npu")
    target = driver.active.get_current_target()
    backend = str(target.backend)
    if backend != "opentile" and not backend.startswith("opentile_"):
        raise RuntimeError(f"expected OpenTile target, got {backend!r}")

    active_device = driver.active.get_active_torch_device()
    if active_device.type != "npu":
        raise RuntimeError(f"expected active NPU device, got {active_device}")

    device_index = torch.npu.current_device()
    device = torch.device("npu", device_index)
    properties = driver.active.utils.get_device_properties(device_index)
    num_aicore = int(properties["num_aicore"])
    num_vectorcore = int(properties["num_vectorcore"])
    if num_vectorcore < 1:
        raise RuntimeError(f"invalid vector core count: {num_vectorcore}")
    return OpenTileRuntime(
        backend=backend,
        device=device,
        num_aicore=num_aicore,
        num_vectorcore=num_vectorcore,
    )


def make_inputs(case: tuple[int, int], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    if dtype != FP32:
        raise ValueError(f"this standalone contract supports only {FP32}, got {dtype}")
    t, d = case
    generator = torch.Generator(device="cpu").manual_seed(TEST_SEED)
    x = torch.randn((t, d), generator=generator, dtype=dtype)
    g = torch.randn((t, d), generator=generator, dtype=dtype)
    w = torch.randn((d,), generator=generator, dtype=dtype)
    b = torch.randn((d,), generator=generator, dtype=dtype)
    dy = torch.randn((t, d), generator=generator, dtype=dtype)
    mean = x.mean(dim=-1, dtype=torch.float32)
    variance = ((x - mean[:, None]) ** 2).mean(dim=-1, dtype=torch.float32)
    rstd = torch.rsqrt(variance + EPS)
    return {"x": x, "g": g, "w": w, "b": b, "dy": dy, "mean": mean, "rstd": rstd}


def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    x = inputs["x"]
    g = inputs["g"]
    w = inputs["w"]
    b = inputs["b"]
    dy = inputs["dy"]
    mean = inputs["mean"]
    rstd = inputs["rstd"]

    xhat = (x - mean[:, None]) * rstd[:, None]
    y = xhat * w[None, :] + b[None, :]
    sigmoid_g = torch.sigmoid(g)
    dsilu = sigmoid_g * (1 + g * (1 - sigmoid_g))
    dg = dy * y * dsilu
    dnorm = dy * g * sigmoid_g
    dw = torch.sum(dnorm * xhat, dim=0, dtype=torch.float32)
    db = torch.sum(dnorm, dim=0, dtype=torch.float32)
    wdy = dnorm * w[None, :]
    c1 = torch.mean(xhat * wdy, dim=1, dtype=torch.float32)
    c2 = torch.mean(wdy, dim=1, dtype=torch.float32)
    dx = (wdy - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
    return {"dx": dx, "dg": dg, "dw": dw, "db": db}


def launch_case(
    case: tuple[int, int],
    cpu_inputs: dict[str, torch.Tensor],
    runtime: OpenTileRuntime,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    t, d = case
    inputs = {name: value.to(runtime.device) for name, value in cpu_inputs.items()}
    block_d = triton.next_power_of_2(d)
    block_t = 1
    num_programs = min(runtime.num_vectorcore, t)

    dx = torch.full((t, d), float("nan"), dtype=FP32, device=runtime.device)
    dg = torch.full((t, d), float("nan"), dtype=FP32, device=runtime.device)
    partial_dw = torch.full((num_programs, d), float("nan"), dtype=torch.float32, device=runtime.device)
    partial_db = torch.full((num_programs, d), float("nan"), dtype=torch.float32, device=runtime.device)

    layer_norm_gated_bwd_kernel[(num_programs,)](
        inputs["x"],
        inputs["g"],
        inputs["w"],
        inputs["b"],
        inputs["dy"],
        dx,
        dg,
        partial_dw,
        partial_db,
        inputs["mean"],
        inputs["rstd"],
        t,
        num_programs,
        D=d,
        BD=block_d,
        BT=block_t,
    )
    actual = {
        "dx": dx,
        "dg": dg,
        "dw": partial_dw.sum(dim=0),
        "db": partial_db.sum(dim=0),
    }
    return actual, reference(cpu_inputs)


def report_and_compare(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_cpu = actual.detach().float().cpu()
    expected_cpu = expected.detach().float().cpu()
    absolute = torch.abs(actual_cpu - expected_cpu)
    relative = absolute / torch.clamp(torch.abs(expected_cpu), min=1e-12)
    close = torch.isclose(actual_cpu, expected_cpu, atol=ATOL, rtol=RTOL)
    mismatch_count = int((~close).sum().item())
    print(
        f"{name}: max_abs={absolute.max().item():.9g} "
        f"max_rel={relative.max().item():.9g} mismatches={mismatch_count}/{actual_cpu.numel()}"
    )
    assert torch.isfinite(actual_cpu).all(), f"{name} contains non-finite values"
    torch.testing.assert_close(actual_cpu, expected_cpu, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "case,dtype",
    [
        pytest.param((1, 16), FP32, id="fp32-aligned-t1-d16"),
        pytest.param((3, 15), FP32, id="fp32-tail-t3-d15"),
    ],
)
def test_opentile_layer_norm_gated_bwd(case: tuple[int, int], dtype: torch.dtype) -> None:
    runtime = get_opentile_runtime()
    print(
        f"route backend={runtime.backend} device={runtime.device} "
        f"aic={runtime.num_aicore} aiv={runtime.num_vectorcore} seed={TEST_SEED} shape={case} dtype={dtype}"
    )
    cpu_inputs = make_inputs(case, dtype)
    actual, expected = launch_case(case, cpu_inputs, runtime)
    torch.npu.synchronize()
    for name in ("dx", "dg", "dw", "db"):
        report_and_compare(name, actual[name], expected[name])


if __name__ == "__main__":
    # test_opentile_layer_norm_gated_bwd((1, 16), FP32)
    pass
