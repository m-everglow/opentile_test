"""Standalone f32 e2e for the TOPP filter kernel.

This is the self-contained form of
``test_sampling.py::test_topp_filter``.  The Mojo wrapper imports are
intentionally replaced with the Triton kernel used by the Ascend NPU backend.
"""

import os

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _top_p_filter_kernel(
    sorted_logits_ptr,
    output_ptr,
    top_p,
    filter_value,
    min_tokens_to_keep,
    stride_logits_b,
    stride_logits_k,
    stride_out_b,
    stride_out_k,
    TOP_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, TOP_K)

    row_logits_ptr = sorted_logits_ptr + pid * stride_logits_b
    logits = tl.load(row_logits_ptr + offsets * stride_logits_k)

    logits_max = tl.max(logits, 0)
    numerator = tl.exp(logits - logits_max)
    probs = numerator / tl.sum(numerator, 0)
    cum_probs = tl.cumsum(probs, 0)
    to_remove = (cum_probs - probs) > top_p
    to_remove = tl.where(offsets < min_tokens_to_keep, False, to_remove)

    filtered_logits = tl.where(to_remove, filter_value, logits)
    filtered_max = tl.max(filtered_logits, 0)
    filtered_numerator = tl.exp(filtered_logits - filtered_max)
    filtered_probs = filtered_numerator / tl.sum(filtered_numerator, 0)

    row_out_ptr = output_ptr + pid * stride_out_b
    tl.store(row_out_ptr + offsets * stride_out_k, filtered_probs)


def top_p_filter(
    logits: torch.Tensor,
    top_p: float,
    min_tokens_to_keep: int,
    top_k: int,
):
    """Inline replacement for MojoTopPFilter's Triton implementation."""
    sorted_logits, sorted_indices = torch.topk(logits.float(), top_k)
    output_probs = torch.empty_like(sorted_logits)

    _top_p_filter_kernel[(logits.shape[0],)](
        sorted_logits,
        output_probs,
        top_p,
        -float("inf"),
        min_tokens_to_keep,
        sorted_logits.stride(0),
        sorted_logits.stride(1),
        output_probs.stride(0),
        output_probs.stride(1),
        TOP_K=top_k,
    )
    return output_probs, sorted_indices


def top_p_filter_ref(
    logits: torch.Tensor,
    top_p: float,
    min_tokens_to_keep: int,
    top_k: int,
):
    """Independent torch reference for the filter probability output."""
    sorted_logits, sorted_indices = torch.topk(logits.float(), top_k)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)
    to_remove = (cum_probs - probs) > top_p
    to_remove[:, :min_tokens_to_keep] = False
    filtered_logits = sorted_logits.masked_fill(to_remove, -float("inf"))
    return torch.softmax(filtered_logits, dim=-1), sorted_indices


TEST_SEED = int(os.environ.get("OPENTILE_TEST_SEED", "2026"))


@pytest.mark.parametrize(
    "shape,top_k,top_p,min_tokens_to_keep",
    [pytest.param((20, 151936), 1024, 0.75, 1, id="b20-v151936-k1024")],
)
def test_topp_filter(shape, top_k, top_p, min_tokens_to_keep):
    # CPU generation makes this test deterministic.  TOP_K is a power of two
    # because Triton requires the direct tl.arange extent to be a power of two.
    generator = torch.Generator(device="cpu").manual_seed(TEST_SEED)
    logits = torch.randn(shape, generator=generator, dtype=torch.float32)
    ref_probs, ref_indices = top_p_filter_ref(
        logits, top_p, min_tokens_to_keep, top_k
    )

    logits_npu = logits.to("npu")
    torch.npu.synchronize()
    got_probs, got_indices = top_p_filter(
        logits_npu, top_p, min_tokens_to_keep, top_k
    )
    torch.npu.synchronize()

    got_probs = got_probs.cpu()
    got_indices = got_indices.cpu()
    torch.testing.assert_close(got_probs, ref_probs, atol=1e-2, rtol=1e-2)
    assert torch.equal(got_indices.sort(dim=-1).values, ref_indices.sort(dim=-1).values)
    print(f"PASS | seed={TEST_SEED} | shape={shape} | top_k={top_k}")
