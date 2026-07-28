from __future__ import annotations

import math
import os

import torch
import torch_npu  # noqa: F401

from ci_sdpa import sdpa


SEED = int(os.environ.get("OPENTILE_TEST_SEED", "2026"))


def _diffusion_mask(seq_length: int, block_size: int) -> torch.Tensor:
    total = seq_length * 2
    rows = torch.arange(total)[:, None]
    cols = torch.arange(total)[None, :]
    row_block = rows // block_size
    col_block = cols // block_size
    same_block = row_block == col_block
    upper_to_lower = (cols >= seq_length) & (rows < seq_length) & (((cols - seq_length) // block_size) < row_block)
    lower_history = (rows >= seq_length) & (cols >= seq_length) & (col_block < row_block)
    return same_block | upper_to_lower | lower_history


def test_sdpa():
    # This is the exact source-test specialization.
    batch, q_heads, kv_heads, head_dim = 1, 5, 1, 128
    source_seq_length, block_size = 2048, 32
    seq_len = source_seq_length * 2
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    q = torch.randn((batch, q_heads, seq_len, head_dim), generator=gen, dtype=torch.bfloat16)
    k = torch.randn((batch, kv_heads, seq_len, head_dim), generator=gen, dtype=torch.bfloat16)
    v = torch.randn((batch, kv_heads, seq_len, head_dim), generator=gen, dtype=torch.bfloat16)
    mask = _diffusion_mask(source_seq_length, block_size)

    device = torch.device("npu", torch.npu.current_device())
    q_npu, k_npu, v_npu, mask_npu = q.to(device), k.to(device), v.to(device), mask.to(device)
    scale = 1.0 / math.sqrt(head_dim)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q_npu,
        k_npu,
        v_npu,
        attn_mask=mask_npu,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
        enable_gqa=True,
    )
    actual = sdpa(q_npu, k_npu, v_npu, mask_npu, scale)
    torch.npu.synchronize()
    actual_cpu, expected_cpu = actual.cpu(), expected.cpu()
    assert torch.isfinite(actual_cpu).all()
    difference = (actual_cpu.float() - expected_cpu.float()).abs()
    print(f"seed={SEED}, max_abs={difference.max().item():.8g}, shape={tuple(actual.shape)}")
    torch.testing.assert_close(actual_cpu.float(), expected_cpu.float(), atol=5e-3, rtol=5e-3)


if __name__ == "__main__":
    # test_sdpa()
    pass

