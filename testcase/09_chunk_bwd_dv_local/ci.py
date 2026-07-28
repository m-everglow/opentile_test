from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from chunk_bwd_dv_local import apply_chunk_bwd_dv_local


SEED = int(os.environ.get("OPENTILE_TEST_SEED", "2026"))
DTYPE = torch.bfloat16
MAX_RMS_ERROR_RATIO = 5e-3
CHUNK_SIZE = 64
CASES = [
    pytest.param(1, 128, 2, 2, 128, 128, id="bf16-aligned"),
    pytest.param(1, 65, 2, 2, 88, 80, id="bf16-tail"),
]


def _error_metrics(
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> tuple[float, float]:
    expected = expected.float()
    actual = actual.float()
    difference = actual - expected
    max_absolute_error = difference.abs().max().item()
    rms_error = difference.square().mean().sqrt()
    reference_rms = expected.square().mean().sqrt()
    rms_error_ratio = (rms_error / (reference_rms + 1e-8)).item()
    return max_absolute_error, rms_error_ratio


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    B, T, H, _ = q.shape
    HV, V = do.shape[2:]
    q = q.float()
    k = k.float()
    do = do.float()
    g = g.float()
    dv = torch.zeros((B, T, HV, V), dtype=torch.float32)

    for i_b in range(B):
        for start in range(0, T, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, T)
            length = end - start
            causal = torch.triu(torch.ones((length, length), dtype=torch.bool))
            for i_h in range(HV):
                q_head = q[i_b, start:end, i_h // (HV // H)]
                k_head = k[i_b, start:end, i_h // (HV // H)]
                do_head = do[i_b, start:end, i_h]
                g_head = g[i_b, start:end, i_h]

                attention = torch.matmul(k_head, q_head.transpose(-1, -2))
                attention *= scale
                attention *= torch.exp(g_head[None, :] - g_head[:, None])
                attention.masked_fill_(~causal, 0.0)
                dv[i_b, start:end, i_h] = torch.matmul(attention, do_head)
    return dv.to(DTYPE)


@pytest.mark.parametrize("B,T,H,HV,K,V", CASES)
def test_chunk_bwd_dv_local(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    q = torch.randn((B, T, H, K), generator=generator, dtype=DTYPE)
    k = torch.randn((B, T, H, K), generator=generator, dtype=DTYPE)
    do = torch.randn((B, T, HV, V), generator=generator, dtype=DTYPE)
    g = F.logsigmoid(torch.randn((B, T, HV), generator=generator, dtype=DTYPE))
    scale = K**-0.5
    expected = _reference(q, k, do, g, scale)

    logical_device = torch.npu.current_device()
    device = torch.device("npu", logical_device)
    actual = apply_chunk_bwd_dv_local(
        q.to(device),
        k.to(device),
        do.to(device),
        g.to(device),
        scale=scale,
        chunk_size=CHUNK_SIZE,
    )
    torch.npu.synchronize()
    actual_cpu = actual.cpu()

    assert torch.isfinite(actual_cpu).all()
    max_absolute_error, rms_error_ratio = _error_metrics(
        expected,
        actual_cpu,
    )
    print(
        "[OPENTILE_E2E] op=chunk_bwd_dv_local "
        f"max_abs_error={max_absolute_error:.6f} "
        f"rms_error_ratio={rms_error_ratio:.6f} "
        f"limit={MAX_RMS_ERROR_RATIO:.6f}"
    )
    assert rms_error_ratio < MAX_RMS_ERROR_RATIO, (
        f"RMS error ratio {rms_error_ratio:.6f} exceeds "
        f"{MAX_RMS_ERROR_RATIO:.6f}; max absolute error "
        f"is {max_absolute_error:.6f}"
    )
    print(
        "[OPENTILE_E2E] op=chunk_bwd_dv_local "
        f"seed={SEED} dtype={DTYPE} "
        f"shape={(B, T, H, HV, K, V)}"
    )


if __name__ == "__main__":
    # test_chunk_bwd_dv_local(1, 128, 2, 2, 128, 128)
    pass
