from __future__ import annotations

import os
from typing import Optional

import pytest
import torch

from kernel import active_opentile_npu, fused_penalty_temp


SEED = int(os.environ.get("OPENTILE_TEST_SEED", "2026"))


def _compile_only() -> bool:
    true_values = {"1", "true", "on", "yes"}
    return any(
        os.environ.get(name, "").lower() in true_values
        for name in ("TRITON_COMPILE_ONLY", "OPENTILE_COMPILE_ONLY")
    )


def _reference(
    logits: torch.Tensor,
    freqs: torch.Tensor,
    is_present: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    repetition_penalty: torch.Tensor,
    temperature: Optional[torch.Tensor],
) -> torch.Tensor:
    output = logits.clone()
    effective_freqs = torch.where(
        is_present[:, None] != 0, freqs, torch.zeros_like(freqs)
    ).float()
    output -= frequency_penalty[:, None] * effective_freqs
    output -= presence_penalty[:, None] * (effective_freqs > 0).float()
    repeated = effective_freqs > 0
    output = torch.where(
        repeated & (output > 0),
        output / repetition_penalty[:, None],
        output,
    )
    output = torch.where(
        repeated & (output < 0),
        output * repetition_penalty[:, None],
        output,
    )
    if temperature is not None:
        output /= temperature[:, None]
    return output


CASES = [
    pytest.param(1, 1024, True, id="aligned-b1-v1024"),
    pytest.param(2, 1024, True, id="aligned-b2-v1024"),
    pytest.param(1, 1025, True, id="tail-b1-v1025"),
    pytest.param(2, 1025, True, id="tail-b2-v1025"),
]



@pytest.mark.parametrize("batch,vocab,use_temperature", CASES)
def test_fused_penalty_temp(batch: int, vocab: int, use_temperature: bool):
    device, vector_cores = active_opentile_npu()
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    logits_cpu = torch.randn(
        (batch, vocab), generator=generator, dtype=torch.float32
    )
    freqs_cpu = torch.randint(
        0, 5, (batch, vocab), generator=generator, dtype=torch.int32
    )
    sparse_mask = torch.rand(
        (batch, vocab), generator=generator, dtype=torch.float32
    ) > 0.15
    freqs_cpu[sparse_mask] = 0

    # Include one absent row when possible to exercise the optional-frequencies path.
    is_present_cpu = torch.ones(batch, dtype=torch.float32)
    if batch > 1:
        is_present_cpu[-1] = 0
    frequency_penalty_cpu = torch.linspace(-0.35, 0.25, batch)
    presence_penalty_cpu = torch.linspace(0.4, -0.2, batch)
    repetition_penalty_cpu = torch.linspace(1.5, 0.75, batch)
    temperature_cpu = (
        torch.linspace(0.8, 1.25, batch) if use_temperature else None
    )

    expected = _reference(
        logits_cpu,
        freqs_cpu,
        is_present_cpu,
        frequency_penalty_cpu,
        presence_penalty_cpu,
        repetition_penalty_cpu,
        temperature_cpu,
    )
    actual = fused_penalty_temp(
        logits_cpu.to(device),
        freqs_cpu.to(device),
        is_present_cpu.to(device),
        frequency_penalty_cpu.to(device),
        presence_penalty_cpu.to(device),
        repetition_penalty_cpu.to(device),
        None if temperature_cpu is None else temperature_cpu.to(device),
    )

    print(
        f"seed={SEED} device={device} vector_cores={vector_cores} "
        f"shape=({batch}, {vocab}) temperature={use_temperature}"
    )
    if _compile_only():
        return

    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    difference = (actual_cpu - expected).abs()
    relative = difference / expected.abs().clamp_min(1e-12)
    mismatches = ~torch.isclose(actual_cpu, expected, atol=1e-5, rtol=1e-5)
    print(
        f"max_abs_error={difference.max().item():.9g} "
        f"max_rel_error={relative.max().item():.9g} "
        f"mismatch_count={mismatches.sum().item()}/{expected.numel()}"
    )
    assert torch.isfinite(actual_cpu).all()
    torch.testing.assert_close(actual_cpu, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    # test_fused_penalty_temp(1, 1024, True)
    pass
