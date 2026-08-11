"""Standalone tests for the OpenTile top-p sampling implementation."""

import functools

import pytest
import torch
import torch.nn.functional as F
import torch_npu

from ci_topp_sampling import top_p_sampling_impl


def bypass_not_implemented(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NotImplementedError:
            pytest.skip("Not implemented on this backend, skipped.")
            return None

    return wrapper


def torch_top_p_sampling_forward(
    logits,
    top_p,
    filter_value,
    min_tokens_to_keep,
    rand_top_k,
):
    logits = logits.to(torch.float32)
    top_k = min(rand_top_k, logits.size(-1))
    sorted_topk_logits, sorted_topk_indices = torch.topk(
        logits,
        top_k,
    )

    cumulative_probs = sorted_topk_logits.softmax(dim=-1).cumsum(
        dim=-1
    )
    sorted_indices_to_remove = cumulative_probs > top_p
    if min_tokens_to_keep > 1:
        sorted_indices_to_remove[
            ..., : min_tokens_to_keep - 1
        ] = 0
    sorted_indices_to_remove[..., 1:] = (
        sorted_indices_to_remove[..., :-1].clone()
    )
    sorted_indices_to_remove[..., 0] = 0
    filtered_logits = sorted_topk_logits.masked_fill(
        sorted_indices_to_remove,
        filter_value,
    )

    final_probs_dist = F.softmax(filtered_logits, dim=-1)
    select_index = torch.multinomial(
        final_probs_dist,
        num_samples=1,
    )
    next_tokens = torch.gather(
        sorted_topk_indices,
        dim=-1,
        index=select_index,
    )
    next_probs = torch.gather(
        final_probs_dist,
        dim=-1,
        index=select_index,
    )
    return next_probs, next_tokens


DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]

TEST_CASES = [
    ((1, 32000), 1024, 0.9, 1),
    ((2, 32000), 64, 0.5, 1),
    ((8, 65024), 128, 0.6, 1),
    ((8, 65024), 128, 0.6, 1),
    ((16, 32000), 2048, 0.9, 1),
    ((32, 32000), 128, 0.6, 2),
    ((128, 32000), 2048, 0.9, 1),
    ((512, 32000), 1024, 0.5, 2),
    ((1024, 151936), 1024, 0.75, 1),
    ((16, 92544), 1024, 0.75, 1),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape, topk, topp, min_tokens_to_keep",
    TEST_CASES,
)
@bypass_not_implemented
def test_topp_sampling_accuracy(
    shape,
    topk,
    topp,
    min_tokens_to_keep,
    dtype,
):
    logits = torch.randn(shape, dtype=dtype).npu()

    next_probs, next_tokens = top_p_sampling_impl(
        logits,
        top_p=topp,
        min_tokens_to_keep=min_tokens_to_keep,
        rand_top_k=topk,
    )
    next_probs_ref, next_tokens_ref = (
        torch_top_p_sampling_forward(
            logits,
            top_p=topp,
            filter_value=float("-inf"),
            min_tokens_to_keep=min_tokens_to_keep,
            rand_top_k=topk,
        )
    )

    assert next_probs.shape == next_probs_ref.shape
    assert next_tokens.shape == next_tokens_ref.shape