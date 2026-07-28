from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from kernel import (
    STAGE1_SHAPE,
    STAGE2_SHAPE,
    STAGE3_SHAPE,
    SUPPORTED_SHAPES,
    _validate_inputs,
    active_opentile_npu,
    fused_sigmoid_gating_delta_rule_update,
)


SEED = int(os.environ.get("OPENTILE_TEST_SEED", "2026"))
DTYPE = torch.bfloat16
ATOL = 5e-3
RTOL = 5e-3
PACKAGE_DIR = Path(__file__).resolve().parent

CASES = [
    pytest.param(*STAGE1_SHAPE, id="bf16-aligned-stage1"),
    pytest.param(*STAGE2_SHAPE, id="bf16-aligned-d256-stage2"),
    pytest.param(*STAGE3_SHAPE, id="bf16-tail-d144-stage3"),
]


def _compile_only() -> bool:
    true_values = {"1", "true", "on", "yes"}
    return any(
        os.environ.get(name, "").lower() in true_values
        for name in ("TRITON_COMPILE_ONLY", "OPENTILE_COMPILE_ONLY")
    )


def _random_tensor(
    shape: tuple[int, ...], generator: torch.Generator
) -> torch.Tensor:
    return torch.empty(shape, dtype=DTYPE).uniform_(
        -1.0, 1.0, generator=generator
    )


def _inputs(
    batch: int,
    time: int,
    heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    return (
        _random_tensor((value_heads,), generator),
        _random_tensor((batch * time, value_heads), generator),
        _random_tensor((value_heads,), generator),
        _random_tensor((batch, time, heads, key_dim), generator),
        _random_tensor((batch, time, heads, key_dim), generator),
        _random_tensor((batch, time, value_heads, value_dim), generator),
        _random_tensor((batch * time, value_heads), generator),
        torch.ones(
            (batch, value_heads, key_dim, value_dim),
            dtype=DTYPE,
        ),
        torch.arange(batch, dtype=torch.int32),
    )


def _reference(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, time, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    head_indices = torch.arange(value_heads) // (value_heads // heads)
    query = q.float().index_select(2, head_indices)
    key = k.float().index_select(2, head_indices)
    value = v.float()
    gate_input = a.float().view(batch, time, value_heads)
    beta_input = b.float().view(batch, time, value_heads)
    state_indices = initial_state_indices.long()
    state = initial_state_source.float().clone()
    hidden = state.index_select(0, state_indices)
    output = torch.full(
        (batch, time, value_heads, value_dim),
        float("nan"),
        dtype=DTYPE,
    )
    scale = key_dim**-0.5

    for time_index in range(time):
        x = gate_input[:, time_index] + dt_bias.float()
        beta_x = softplus_beta * x
        softplus_x = torch.where(
            beta_x <= softplus_threshold,
            torch.log(1.0 + torch.exp(beta_x)) / softplus_beta,
            x,
        )
        gate = -torch.exp(A_log.float()) * softplus_x
        beta = torch.sigmoid(beta_input[:, time_index])
        query_t = query[:, time_index]
        key_t = key[:, time_index]
        query_t = query_t / torch.sqrt(
            torch.sum(query_t * query_t, dim=-1, keepdim=True) + 1e-6
        )
        key_t = key_t / torch.sqrt(
            torch.sum(key_t * key_t, dim=-1, keepdim=True) + 1e-6
        )
        hidden = hidden * torch.exp(gate)[..., None, None]
        value_t = value[:, time_index] - torch.einsum(
            "bhkv,bhk->bhv", hidden, key_t
        )
        value_t = value_t * beta[..., None]
        hidden = hidden + key_t[..., :, None] * value_t[..., None, :]
        output[:, time_index] = torch.einsum(
            "bhkv,bhk->bhv", hidden, query_t * scale
        ).to(DTYPE)

    state.index_copy_(0, state_indices, hidden)
    return output, state.to(DTYPE)


def test_stage3_contract_is_skill_compliant() -> None:
    assert DTYPE == torch.bfloat16
    assert (ATOL, RTOL) == (5e-3, 5e-3)
    assert tuple(case.values for case in CASES) == SUPPORTED_SHAPES

    source = (PACKAGE_DIR / "kernel.py").read_text(encoding="utf-8")
    assert "def fused_sigmoid_gating_delta_rule_update_kernel(" in source
    assert '@triton.jit(do_not_specialize=["T"])' in source
    assert "tl.program_id(0)" in source
    assert "tl.program_id(1)" in source
    assert "tl.program_id(2)" in source
    assert "num_warps=" not in source
    assert "num_stages=" not in source


def test_stage1_rejects_dtype_widening() -> None:
    inputs = list(_inputs(*STAGE1_SHAPE))
    inputs[3] = inputs[3].float()
    with pytest.raises(ValueError, match="BF16-only"):
        _validate_inputs(*inputs[:3], *inputs[3:])


def test_stage2_accepts_original_d256_aligned_shape() -> None:
    assert SUPPORTED_SHAPES == (STAGE1_SHAPE, STAGE2_SHAPE, STAGE3_SHAPE)
    inputs = _inputs(*STAGE2_SHAPE)
    assert _validate_inputs(*inputs[:3], *inputs[3:]) == STAGE2_SHAPE


def test_stage3_accepts_bf16_d144_tail_shape() -> None:
    inputs = _inputs(*STAGE3_SHAPE)
    assert _validate_inputs(*inputs[:3], *inputs[3:]) == STAGE3_SHAPE
    output, state = _reference(
        *inputs[:3],
        1.0,
        20.0,
        *inputs[3:],
    )
    assert output.shape == (1, 1, 64, 144)
    assert state.shape == (1, 64, 144, 144)
    assert torch.isfinite(output).all()
    assert torch.isfinite(state).all()


def test_cpu_reference_writes_complete_outputs() -> None:
    inputs = _inputs(*STAGE1_SHAPE)
    output, state = _reference(
        *inputs[:3],
        1.0,
        20.0,
        *inputs[3:],
    )
    assert output.shape == (1, 1, 8, 128)
    assert state.shape == (1, 8, 128, 128)
    assert output.dtype == DTYPE
    assert state.dtype == DTYPE
    assert torch.isfinite(output).all()
    assert torch.isfinite(state).all()


@pytest.mark.parametrize(
    "batch,time,heads,value_heads,key_dim,value_dim",
    CASES,
)
def test_fused_sigmoid_gating_delta_rule_update(
    batch: int,
    time: int,
    heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
) -> None:
    device, num_aicore, num_vectorcore = active_opentile_npu()
    inputs = _inputs(
        batch,
        time,
        heads,
        value_heads,
        key_dim,
        value_dim,
    )
    expected_o, expected_state = _reference(
        *inputs[:3],
        1.0,
        20.0,
        *inputs[3:],
    )
    actual_inputs = [tensor.to(device) for tensor in inputs]
    actual_o, actual_state = fused_sigmoid_gating_delta_rule_update(
        *actual_inputs[:3],
        1.0,
        20.0,
        *actual_inputs[3:],
    )

    if _compile_only():
        return

    torch.npu.synchronize()
    grid = (1, (value_dim + 63) // 64, batch * value_heads)
    print(
        f"seed={SEED} dtype=bf16 "
        f"shape={(batch, time, heads, value_heads, key_dim, value_dim)} "
        f"device={device} num_aicore={num_aicore} "
        f"num_vectorcore={num_vectorcore} grid={grid}"
    )
    failures: list[str] = []
    for name, actual, expected in (
        ("o", actual_o.cpu(), expected_o),
        ("h0_source", actual_state.cpu(), expected_state),
    ):
        actual_f32 = actual.float()
        expected_f32 = expected.float()
        difference = (actual_f32 - expected_f32).abs()
        relative = difference / expected_f32.abs().clamp_min(1e-12)
        mismatches = ~torch.isclose(
            actual_f32,
            expected_f32,
            atol=ATOL,
            rtol=RTOL,
        )
        print(
            f"{name}: max_abs_error={difference.max().item():.9g} "
            f"max_rel_error={relative.max().item():.9g} "
            f"mismatch_count={mismatches.sum().item()}/{expected.numel()}"
        )
        if not bool(torch.isfinite(actual_f32).all()):
            failures.append(f"{name} contains non-finite data")
            continue
        try:
            torch.testing.assert_close(
                actual_f32,
                expected_f32,
                atol=ATOL,
                rtol=RTOL,
            )
        except AssertionError as error:
            failures.append(f"{name}:\n{error}")

    if failures:
        pytest.fail("\n\n".join(failures), pytrace=False)
