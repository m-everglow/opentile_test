"""Standalone f32 e2e for join-probability reject sampling ("magic").

This is a self-contained extraction of
``test_sampling.py::test_join_prob_reject_sampler``.  It deliberately keeps
the rank-1 gather ``spec_offset * vocab_size + draft_token_ids`` that exercises
the non-contiguous pointer-tile lowering path.
"""

import os

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _join_prob_reject_sampler_kernel(
    output_token_ids_ptr,
    output_accept_lens_ptr,
    draft_token_ids_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    uniform_random_ptr,
    cum_probs_ptr,
    cum_rand_ptr,
    max_spec_len: tl.constexpr,
    vocab_size: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    batch_draft_token_ids_ptr = draft_token_ids_ptr + batch_idx * max_spec_len
    batch_draft_probs_ptr = draft_probs_ptr + batch_idx * max_spec_len
    batch_target_probs_ptr = (
        target_probs_ptr + batch_idx * (max_spec_len + 1) * vocab_size
    )
    batch_uniform_random_ptr = uniform_random_ptr + batch_idx * max_spec_len
    batch_cum_probs_ptr = cum_probs_ptr + batch_idx * max_spec_len
    batch_cum_rand_ptr = cum_rand_ptr + batch_idx * max_spec_len
    batch_output_token_ids_ptr = output_token_ids_ptr + batch_idx * (max_spec_len + 1)
    batch_output_accept_lens_ptr = output_accept_lens_ptr + batch_idx

    spec_offset = tl.arange(0, max_spec_len)
    uniform_rand = tl.load(batch_uniform_random_ptr + spec_offset)
    draft_token_ids = tl.load(batch_draft_token_ids_ptr + spec_offset)
    draft_probs = tl.load(batch_draft_probs_ptr + spec_offset)
    target_probs = tl.load(
        batch_target_probs_ptr + spec_offset * vocab_size + draft_token_ids
    )

    ratio = tl.clamp(target_probs / draft_probs, 0, 1)
    cum_probs = tl.cumprod(ratio, axis=0)
    cum_rands = tl.cumprod(uniform_rand, axis=0)
    tl.store(batch_cum_probs_ptr + spec_offset, cum_probs)
    tl.store(batch_cum_rand_ptr + spec_offset, cum_rands)

    accept_len = 0
    found_accepted_prefix = False
    for pos in range(0, max_spec_len):
        if not found_accepted_prefix:
            index = max_spec_len - pos - 1
            cum_prob = tl.load(batch_cum_probs_ptr + index)
            cum_rand = tl.load(batch_cum_rand_ptr + index)
            if cum_prob >= cum_rand:
                accept_len = index + 1
                found_accepted_prefix = True

    tl.store(
        batch_output_token_ids_ptr + spec_offset,
        draft_token_ids,
        mask=spec_offset < accept_len,
    )
    tl.store(batch_output_accept_lens_ptr, accept_len)


def join_prob_reject_sampler(
    target_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    uniform_random: torch.Tensor,
):
    """Inline replacement for MojoJoinProbRejectSampling's Triton path."""
    batch_size, _, vocab_size = target_probs.shape
    spec_step = draft_probs.shape[1]
    device = target_probs.device

    # The kernel leaves unaccepted token positions untouched.  Zero-init makes
    # that externally observable contract deterministic, as in the source test
    # after it masks positions at or beyond accept_len.
    output_token_ids = torch.zeros(
        (batch_size, spec_step + 1), dtype=torch.int32, device=device
    )
    output_accept_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)
    # All cumulative positions are required outputs; NaN catches missing writes.
    cum_probs = torch.full((batch_size, spec_step), float("nan"), device=device)
    cum_rand = torch.full((batch_size, spec_step), float("nan"), device=device)

    _join_prob_reject_sampler_kernel[(batch_size,)](
        output_token_ids,
        output_accept_lens,
        draft_tokens,
        draft_probs,
        target_probs,
        uniform_random,
        cum_probs,
        cum_rand,
        max_spec_len=spec_step,
        vocab_size=vocab_size,
    )
    return output_token_ids, output_accept_lens, cum_probs, cum_rand


def join_prob_reject_sampler_ref(
    target_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    uniform_random: torch.Tensor,
):
    """Independent torch reference, including the diagnostic cumulative buffers."""
    batch_size, _, _ = target_probs.shape
    spec_step = draft_probs.shape[1]
    selected = torch.gather(
        target_probs[:, :spec_step, :], 2, draft_tokens.unsqueeze(-1)
    ).squeeze(-1)
    ratios = torch.clamp(selected / draft_probs, 0, 1)
    cum_probs = torch.cumprod(ratios, dim=1)
    cum_rand = torch.cumprod(uniform_random, dim=1)

    output_token_ids = torch.zeros((batch_size, spec_step + 1), dtype=torch.int32)
    output_accept_lens = torch.zeros(batch_size, dtype=torch.int32)
    for batch_idx in range(batch_size):
        accepted = torch.nonzero(
            cum_probs[batch_idx] >= cum_rand[batch_idx], as_tuple=False
        )
        accept_len = 0 if accepted.numel() == 0 else accepted[-1].item() + 1
        output_accept_lens[batch_idx] = accept_len
        output_token_ids[batch_idx, :accept_len] = draft_tokens[
            batch_idx, :accept_len
        ].to(torch.int32)
    return output_token_ids, output_accept_lens, cum_probs, cum_rand


TEST_SEED = int(os.environ.get("OPENTILE_TEST_SEED", "42"))


@pytest.mark.parametrize(
    "batch_size,vocab_size,spec_step",
    [pytest.param(15, 155136, 4, id="b15-v155136-s4")],
)
def test_magic(batch_size, vocab_size, spec_step):
    # This follows the source test's seed and generation order, but uses an
    # explicit CPU Generator so no external/global RNG state is required.
    generator = torch.Generator(device="cpu").manual_seed(TEST_SEED)
    target_probs = torch.randn(
        (batch_size, 1 + spec_step, vocab_size), generator=generator, dtype=torch.float32
    )
    draft_tokens = torch.randint(
        0, vocab_size, (batch_size, spec_step), generator=generator
    )
    draft_probs = torch.ones((batch_size, spec_step), dtype=torch.float32)
    uniform_random = torch.rand(
        (batch_size, spec_step), generator=generator, dtype=torch.float32
    )

    ref = join_prob_reject_sampler_ref(
        target_probs, draft_tokens, draft_probs, uniform_random
    )
    torch.npu.synchronize()
    got = join_prob_reject_sampler(
        target_probs.to("npu"),
        draft_tokens.to("npu"),
        draft_probs.to("npu"),
        uniform_random.to("npu"),
    )
    torch.npu.synchronize()
    got_ids, got_lens, got_cum_probs, got_cum_rand = (value.cpu() for value in got)
    ref_ids, ref_lens, ref_cum_probs, ref_cum_rand = ref

    torch.testing.assert_close(got_cum_probs, ref_cum_probs, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got_cum_rand, ref_cum_rand, atol=1e-6, rtol=1e-6)
    assert torch.equal(got_lens, ref_lens), (
        f"accept_len mismatch: ref={ref_lens.tolist()} got={got_lens.tolist()}"
    )
    assert torch.equal(got_ids, ref_ids), "accepted draft-token prefix mismatch"
    print(f"PASS | seed={TEST_SEED} | b={batch_size} | vocab={vocab_size}")
