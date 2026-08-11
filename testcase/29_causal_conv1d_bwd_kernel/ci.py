from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from kernel import (
    active_opentile_npu,
    causal_conv1d_bwd,
    causal_conv1d_bwd_kernel,
)


assert causal_conv1d_bwd_kernel is not None


W = 4
SEED = int(os.environ.get("OPENTILE_TEST_SEED", "42"))

DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}
TOLERANCES = {
    torch.float32: (1e-5, 1e-5),
    torch.bfloat16: (5e-3, 5e-3),
    torch.float16: (1e-3, 1e-3),
}
CASES = [
    pytest.param("fp32", "none", 1, 64, 64, id="fp32-none-aligned"),
    pytest.param("fp32", "none", 2, 65, 70, id="fp32-none-tail"),
    pytest.param("fp32", "silu", 1, 64, 64, id="fp32-silu-aligned"),
    pytest.param("fp32", "silu", 2, 65, 70, id="fp32-silu-tail"),
    pytest.param("bf16", "none", 1, 64, 64, id="bf16-none-aligned"),
    pytest.param("bf16", "none", 2, 65, 70, id="bf16-none-tail"),
    pytest.param("bf16", "silu", 1, 64, 64, id="bf16-silu-aligned"),
    pytest.param("bf16", "silu", 2, 65, 70, id="bf16-silu-tail"),
    pytest.param("fp16", "none", 1, 64, 64, id="fp16-none-aligned"),
    pytest.param("fp16", "none", 2, 65, 70, id="fp16-none-tail"),
    pytest.param("fp16", "silu", 1, 64, 64, id="fp16-silu-aligned"),
    pytest.param("fp16", "silu", 2, 65, 70, id="fp16-silu-tail"),
]


def _compile_only() -> bool:
    true_values = {"1", "true", "on", "yes"}
    return any(
        os.environ.get(name, "").lower() in true_values
        for name in ("TRITON_COMPILE_ONLY", "OPENTILE_COMPILE_ONLY")
    )


def _inputs(
    batch: int,
    time: int,
    dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    x = torch.randn(
        (batch, time, dim),
        dtype=dtype,
        generator=generator,
    )
    weight = torch.randn(
        (dim, W),
        dtype=dtype,
        generator=generator,
    )
    bias = torch.randn(
        (dim,),
        dtype=dtype,
        generator=generator,
    )
    dy = torch.randn(
        (batch, time, dim),
        dtype=dtype,
        generator=generator,
    )
    return x, weight, bias, dy


def _preactivation(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    y = F.conv1d(
        x.float().transpose(1, 2),
        weight.float().unsqueeze(1),
        bias.float(),
        padding=W - 1,
        groups=x.shape[-1],
    )[..., : x.shape[1]]
    return y.transpose(1, 2).to(x.dtype).contiguous()


def _reference(
    x: torch.Tensor,
    y_pre: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor,
    dy: torch.Tensor,
    activation: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_ref = x.float().detach().requires_grad_(True)
    weight_ref = weight.float().detach().requires_grad_(True)
    bias_ref = bias.float().detach().requires_grad_(True)
    conv = F.conv1d(
        x_ref.transpose(1, 2),
        weight_ref.unsqueeze(1),
        bias_ref,
        padding=W - 1,
        groups=x.shape[-1],
    )[..., : x.shape[1]].transpose(1, 2)

    dy_conv = dy.float()
    if activation == "silu":
        assert y_pre is not None
        y_quantized = y_pre.float()
        sigmoid_y = torch.sigmoid(y_quantized)
        dy_conv = dy_conv * sigmoid_y * (
            1.0 + y_quantized * (1.0 - sigmoid_y)
        )

    dx, dw, db = torch.autograd.grad(
        conv,
        (x_ref, weight_ref, bias_ref),
        grad_outputs=dy_conv,
    )
    return (
        dx.to(x.dtype),
        dw.to(weight.dtype),
        db.to(bias.dtype),
    )


@pytest.mark.parametrize(
    "dtype_name,activation_name,batch,time,dim",
    CASES,
)
def test_causal_conv1d_bwd(
    dtype_name: str,
    activation_name: str,
    batch: int,
    time: int,
    dim: int,
) -> None:
    dtype = DTYPES[dtype_name]
    activation = None if activation_name == "none" else activation_name
    x, weight, bias, dy = _inputs(batch, time, dim, dtype)
    y_pre = _preactivation(x, weight, bias) if activation == "silu" else None
    expected = _reference(x, y_pre, weight, bias, dy, activation)

    device, ai_cores, vector_cores = active_opentile_npu()
    actual = causal_conv1d_bwd(
        x=x.to(device),
        y_pre=y_pre.to(device) if y_pre is not None else None,
        weight=weight.to(device),
        bias=bias.to(device),
        dy=dy.to(device),
        activation=activation,
    )
    if _compile_only():
        return

    torch.npu.synchronize()
    atol, rtol = TOLERANCES[dtype]
    print(
        f"seed={SEED} device={device} ai_cores={ai_cores} "
        f"vector_cores={vector_cores} dtype={dtype_name} "
        f"activation={activation_name} shape=({batch},{time},{dim})"
    )
    failures: list[str] = []
    for name, actual_tensor, expected_tensor in zip(
        ("dx", "dw", "db"),
        actual,
        expected,
        strict=True,
    ):
        host_actual = actual_tensor.cpu()
        difference = (host_actual.float() - expected_tensor.float()).abs()
        relative = (
            difference / expected_tensor.float().abs().clamp_min(1e-12)
        )
        mismatches = ~torch.isclose(
            host_actual.float(),
            expected_tensor.float(),
            atol=atol,
            rtol=rtol,
        )
        print(
            f"{name}: max_abs_error={difference.max().item():.9g} "
            f"max_rel_error={relative.max().item():.9g} "
            f"mismatch_count={mismatches.sum().item()}/"
            f"{expected_tensor.numel()}"
        )
        if not torch.isfinite(host_actual).all():
            failures.append(f"{name}: contains non-finite or unwritten data")
            continue
        try:
            torch.testing.assert_close(
                host_actual.float(),
                expected_tensor.float(),
                atol=atol,
                rtol=rtol,
            )
        except AssertionError as error:
            failures.append(f"{name}: {error}")
    assert not failures, "\n\n".join(failures)
