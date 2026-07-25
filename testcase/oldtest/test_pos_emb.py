from __future__ import annotations

import os

import pytest
import torch
import torch_npu  # noqa: F401

from rope import apply_rope


SEED = int(os.environ.get("OPENTILE_TEST_SEED", "2026"))
CASES = [
    pytest.param(1, 124, 8, 2, 128, torch.float16, id="fp16-aligned"),
    pytest.param(1, 124, 2, 1, 88, torch.float16, id="fp16-tail"),
    pytest.param(1, 124, 8, 2, 128, torch.bfloat16, id="bf16-aligned"),
    pytest.param(1, 124, 2, 1, 88, torch.bfloat16, id="bf16-tail"),
]
TOL = {torch.float16: 1e-3, torch.bfloat16: 5e-3}


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool):
    x1, x2 = x.chunk(2, dim=-1)
    c = cos[..., : x1.shape[-1]]
    s = sin[..., : x1.shape[-1]]
    if inverse:
        return torch.cat((x1 * c + x2 * s, x2 * c - x1 * s), dim=-1)
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)


@pytest.mark.parametrize("bs,seqlen,q_heads,k_heads,head_dim,dtype", CASES)
def test_pos_emb(bs, seqlen, q_heads, k_heads, head_dim, dtype):
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    q = torch.randn((bs, seqlen, q_heads, head_dim), generator=gen, dtype=dtype).transpose(1, 2)
    k = torch.randn((bs, seqlen, k_heads, head_dim), generator=gen, dtype=dtype).transpose(1, 2)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(seqlen, dtype=torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[None, :, :], emb.sin()[None, :, :]

    expected_q = _rotate(q.float(), cos, sin, False).to(dtype)
    expected_k = _rotate(k.float(), cos, sin, False).to(dtype)
    dq = torch.rand(q.shape, generator=gen, dtype=dtype).transpose(1, 2).transpose(1, 2)
    dk = torch.rand(k.shape, generator=gen, dtype=dtype).transpose(1, 2).transpose(1, 2)
    expected_dq = _rotate(dq.float(), cos, sin, True).to(dtype)
    expected_dk = _rotate(dk.float(), cos, sin, True).to(dtype)

    device = torch.device("npu", torch.npu.current_device())
    q_out = apply_rope(q.to(device), cos.to(device), sin.to(device), False)
    k_out = apply_rope(k.to(device), cos.to(device), sin.to(device), False)
    dq_out = apply_rope(dq.to(device), cos.to(device), sin.to(device), True)
    dk_out = apply_rope(dk.to(device), cos.to(device), sin.to(device), True)
    torch.npu.synchronize()

    tol = TOL[dtype]
    for actual, expected in ((q_out, expected_q), (k_out, expected_k), (dq_out, expected_dq), (dk_out, expected_dk)):
        actual_cpu = actual.cpu()
        assert torch.isfinite(actual_cpu).all()
        torch.testing.assert_close(actual_cpu.float(), expected.float(), atol=tol, rtol=tol)
    print(f"seed={SEED}, dtype={dtype}, shape={(bs, seqlen, q_heads, k_heads, head_dim)}")


if __name__ == "__main__":
    # test_pos_emb(1, 124, 8, 2, 128, torch.float16)
    pass
