from __future__ import annotations

import math
import os

import pytest
import torch
import torch_npu  # noqa: F401

from dllm_attention_up_fwd import apply_dllm_attention_up_fwd


SEED = int(os.environ.get("OPENTILE_TEST_SEED", "2026"))
DTYPE = torch.bfloat16
ATOL = 5e-3
RTOL = 5e-3
CASES = [
    pytest.param(128, 5, 1, 128, 2, id="bf16-aligned-s128"),
    pytest.param(65, 5, 1, 128, 2, id="bf16-tail-s65"),
]


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlen: torch.Tensor,
    scale: float,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_heads = q.shape[1]
    k = k.repeat_interleave(q_heads // k.shape[1], dim=1)
    v = v.repeat_interleave(q_heads // v.shape[1], dim=1)
    output = torch.zeros_like(q)
    lse = torch.zeros((q.shape[0], q_heads), dtype=torch.float32)
    sequence_half = q.shape[0] // 2
    block_r = 32
    block_c = 128

    def consume(
        block_q: torch.Tensor,
        block_o: torch.Tensor,
        block_m: torch.Tensor,
        block_l: torch.Tensor,
        block_k: torch.Tensor,
        block_v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # tl.dot(BF16, BF16) accumulates into FP32.  Keep that rounding point
        # explicit instead of producing a BF16 QK matrix and casting it later.
        scores = torch.matmul(block_q.float(), block_k.float().T) * scale
        if mask is not None:
            scores.masked_fill_(~mask, -1.0e6)
        next_m = torch.maximum(block_m, scores.max(dim=-1).values)
        probabilities = torch.exp(scores - next_m[:, None])
        previous_scale = torch.exp(block_m - next_m)
        next_l = (
            previous_scale * block_l + probabilities.sum(dim=-1)
        )
        next_o = previous_scale[:, None] * block_o
        next_o += torch.matmul(
            probabilities.to(DTYPE).float(), block_v.float()
        )
        return next_o, next_m, next_l

    start = 0
    for end_tensor in cu_seqlen:
        end = int(end_tensor.item())
        length = end - start
        row_tiles = (length + block_r - 1) // block_r

        for idx_r in range(row_tiles):
            row_begin = start + idx_r * block_r
            row_end = min(row_begin + block_r, end)
            row_positions = torch.arange(row_begin - start, row_end - start)
            row_groups = row_positions // block_size

            for head in range(q_heads):
                block_q = q[row_begin:row_end, head]
                block_o = torch.zeros(
                    (row_end - row_begin, q.shape[2]),
                    dtype=torch.float32,
                )
                block_m = torch.full(
                    (row_end - row_begin,), -1.0e6, dtype=torch.float32
                )
                block_l = torch.zeros(
                    (row_end - row_begin,), dtype=torch.float32
                )

                # Current upper-half 32-row tile.
                upper_groups = (
                    torch.arange(row_begin - start, row_end - start)
                    // block_size
                )
                upper_mask = (
                    row_groups[:, None] == upper_groups[None, :]
                )
                block_o, block_m, block_l = consume(
                    block_q,
                    block_o,
                    block_m,
                    block_l,
                    k[row_begin:row_end, head],
                    v[row_begin:row_end, head],
                    upper_mask,
                )

                # Current lower-half 32-row tile.
                lower_begin = sequence_half + row_begin
                lower_end = sequence_half + row_end
                lower_mask = (
                    row_groups[:, None] > upper_groups[None, :]
                )
                block_o, block_m, block_l = consume(
                    block_q,
                    block_o,
                    block_m,
                    block_l,
                    k[lower_begin:lower_end, head],
                    v[lower_begin:lower_end, head],
                    lower_mask,
                )

                # Earlier 32-row tiles inside the current 128-row group.
                local_tile_begin = (
                    idx_r * block_r // block_c * block_c // block_r
                )
                for idx_tile_r in range(local_tile_begin, idx_r):
                    key_begin = start + idx_tile_r * block_r
                    key_end = min(key_begin + block_r, end)
                    block_o, block_m, block_l = consume(
                        block_q,
                        block_o,
                        block_m,
                        block_l,
                        k[
                            sequence_half
                            + key_begin : sequence_half
                            + key_end,
                            head,
                        ],
                        v[
                            sequence_half
                            + key_begin : sequence_half
                            + key_end,
                            head,
                        ],
                    )

                # Complete earlier 128-row groups.
                for idx_c in range(idx_r * block_r // block_c):
                    key_begin = start + idx_c * block_c
                    key_end = min(key_begin + block_c, end)
                    block_o, block_m, block_l = consume(
                        block_q,
                        block_o,
                        block_m,
                        block_l,
                        k[
                            sequence_half
                            + key_begin : sequence_half
                            + key_end,
                            head,
                        ],
                        v[
                            sequence_half
                            + key_begin : sequence_half
                            + key_end,
                            head,
                        ],
                    )

                output[row_begin:row_end, head] = block_o / block_l[:, None]
                lse[row_begin:row_end, head] = torch.log(block_l) + block_m
        start = end
    return output, lse


@pytest.mark.parametrize("sequence_half,q_heads,kv_heads,head_dim,block_size", CASES)
def test_dllm_attention_up_fwd(
    sequence_half: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    block_size: int,
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    total_sequence = sequence_half * 2
    q = torch.randn(
        (total_sequence, q_heads, head_dim),
        generator=generator,
        dtype=DTYPE,
    )
    k = torch.randn(
        (total_sequence, kv_heads, head_dim),
        generator=generator,
        dtype=DTYPE,
    )
    v = torch.randn(
        (total_sequence, kv_heads, head_dim),
        generator=generator,
        dtype=DTYPE,
    )
    cu_seqlen = torch.tensor([sequence_half], dtype=torch.int32)
    scale = 1.0 / math.sqrt(head_dim)
    expected, expected_lse = _reference(q, k, v, cu_seqlen, scale, block_size)

    logical_device = torch.npu.current_device()
    device = torch.device("npu", logical_device)
    actual, actual_lse = apply_dllm_attention_up_fwd(
        q.to(device),
        k.to(device),
        v.to(device),
        cu_seqlen.to(device),
        scale,
        block_size,
    )
    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    actual_lse_cpu = actual_lse.cpu()

    assert torch.isfinite(actual_cpu).all()
    assert torch.isfinite(actual_lse_cpu).all()
    torch.testing.assert_close(
        actual_cpu.to(DTYPE).float(),
        expected.float(),
        atol=ATOL,
        rtol=RTOL,
    )
    torch.testing.assert_close(
        actual_lse_cpu[:sequence_half],
        expected_lse[:sequence_half],
        atol=ATOL,
        rtol=RTOL,
    )
    assert torch.count_nonzero(actual_cpu[sequence_half:]) == 0
    assert torch.count_nonzero(actual_lse_cpu[sequence_half:]) == 0
    print(
        "[OPENTILE_E2E] op=dllm_attention_up_fwd "
        f"seed={SEED} dtype={DTYPE} sequence_half={sequence_half} "
        f"q_heads={q_heads} kv_heads={kv_heads} head_dim={head_dim} "
        f"block_size={block_size}"
    )


if __name__ == "__main__":
    # test_dllm_attention_up_fwd(128, 5, 1, 128, 2)
    pass