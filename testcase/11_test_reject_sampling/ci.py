"""Standalone e2e test for reject_sampler kernel (speculative decoding).

No external deps beyond torch + triton.  Run directly with:

    TRITON_ALWAYS_COMPILE=1 OPENTILE_KERNEL_MODE=aiv \
      python3 -m pytest -q -s test_reject_e2e.py
"""

import os
import pytest
import torch
import triton
import triton.language as tl

# ── Triton kernel (inline) ──────────────────────────────────────────────

@triton.jit()
def _reject_sampler_kernel(
    output_token_ids_ptr,
    output_accept_lens_ptr,
    draft_token_ids_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    uniform_random_ptr,
    max_spec_len: tl.constexpr,
    vocab_size: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    batch_draft_token_ids_ptr = draft_token_ids_ptr + batch_idx * max_spec_len
    batch_draft_probs_ptr = draft_probs_ptr + batch_idx * max_spec_len
    batch_target_probs_ptr = (
        target_probs_ptr + batch_idx * (max_spec_len + 1) * vocab_size
    )
    batch_output_token_ids_ptr = (
        output_token_ids_ptr + batch_idx * (max_spec_len + 1)
    )
    batch_output_accept_lens_ptr = output_accept_lens_ptr + batch_idx
    batch_uniform_random = tl.load(uniform_random_ptr + batch_idx)

    accept_len = 0
    rejected = False
    for pos in range(0, max_spec_len):
        if not rejected:
            draft_token_id = tl.load(batch_draft_token_ids_ptr + pos)
            draft_prob = tl.load(batch_draft_probs_ptr + pos)
            target_prob = tl.load(
                batch_target_probs_ptr + pos * vocab_size + draft_token_id
            )
            if (
                draft_prob > 0
                and target_prob / draft_prob >= batch_uniform_random
            ):
                accept_len += 1
                tl.store(batch_output_token_ids_ptr + pos, draft_token_id)
            else:
                rejected = True

    tl.store(batch_output_accept_lens_ptr, accept_len)


# ── Launcher ────────────────────────────────────────────────────────────

def reject_sampler(
    target_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    rand_vals: torch.Tensor,
):
    batch_size, _, vocab_size = target_probs.shape
    spec_step = draft_probs.shape[1]

    output_token_ids = torch.zeros(
        (batch_size, spec_step + 1), device=target_probs.device, dtype=torch.int32
    )
    output_accept_lens = torch.zeros(
        batch_size, device=target_probs.device, dtype=torch.int32
    )

    grid = (batch_size,)
    _reject_sampler_kernel[grid](
        output_token_ids_ptr=output_token_ids,
        output_accept_lens_ptr=output_accept_lens,
        draft_token_ids_ptr=draft_tokens,
        draft_probs_ptr=draft_probs,
        target_probs_ptr=target_probs,
        uniform_random_ptr=rand_vals,
        max_spec_len=spec_step,
        vocab_size=vocab_size,
    )
    return output_token_ids, output_accept_lens


# ── Torch reference ─────────────────────────────────────────────────────

def reject_sampler_ref(
    target_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    rand_vals: torch.Tensor,
):
    batch_size, _, vocab_size = target_probs.shape
    spec_step = draft_probs.shape[1]

    output_token_ids = torch.zeros(
        (batch_size, spec_step + 1), dtype=torch.int32
    )
    output_accept_lens = torch.zeros(batch_size, dtype=torch.int32)

    for b in range(batch_size):
        accept_len = 0
        rejected = False
        for pos in range(spec_step):
            if not rejected:
                token_id = draft_tokens[b, pos].item()
                draft_p = draft_probs[b, pos].item()
                target_p = target_probs[b, pos, token_id].item()
                rand = rand_vals[b, 0].item()
                if draft_p > 0 and target_p / draft_p >= rand:
                    accept_len += 1
                    output_token_ids[b, pos] = token_id
                else:
                    rejected = True
        output_accept_lens[b] = accept_len

    return output_token_ids, output_accept_lens


# ── Test ────────────────────────────────────────────────────────────────

TEST_SEED = int(os.environ.get("OPENTILE_TEST_SEED", "2026"))


@pytest.mark.parametrize("batch_size,vocab_size,spec_step", [
    pytest.param(15, 155136, 3, id="b15-v155136-s3"),
])
def test_reject(batch_size, vocab_size, spec_step):
    gen = torch.Generator(device="cpu").manual_seed(TEST_SEED)
    rng_seed = 42

    target_probs = torch.randn(
        (batch_size, spec_step + 1, vocab_size), generator=gen, dtype=torch.float32
    )
    draft_tokens = torch.randint(
        0, vocab_size, (batch_size, spec_step), generator=gen
    )
    draft_probs = torch.ones((batch_size, spec_step), dtype=torch.float32)

    # Generate rand_vals on CPU so ref and kernel use identical values.
    torch.manual_seed(rng_seed)
    rand_vals = torch.rand(batch_size, 1)

    ref_ids, ref_lens = reject_sampler_ref(
        target_probs, draft_tokens, draft_probs, rand_vals
    )
    device = "npu"
    stream = torch.npu.current_stream()
    tp_npu = target_probs.to(device)
    dt_npu = draft_tokens.to(device)
    dp_npu = draft_probs.to(device)
    rv_npu = rand_vals.to(device)
    stream.synchronize()
    ttx_ids, ttx_lens = reject_sampler(tp_npu, dt_npu, dp_npu, rv_npu)
    torch.npu.synchronize()
    ttx_ids, ttx_lens = ttx_ids.cpu(), ttx_lens.cpu()

    mask = torch.arange(spec_step + 1).expand(batch_size, -1)
    ref_ids = ref_ids * (mask < ref_lens.unsqueeze(-1)).to(ref_ids.dtype)
    ttx_ids = ttx_ids * (mask < ttx_lens.unsqueeze(-1)).to(ttx_ids.dtype)

    # Compare only accepted positions (within accept_len)
    for b in range(batch_size):
        rl = ref_lens[b].item()
        tl = ttx_lens[b].item()
        if rl != tl:
            print(f"BATCH {b}: ref_len={rl} ttx_len={tl}")
            for p in range(min(rl, spec_step)):
                rid = ref_ids[b, p].item()
                tid = ttx_ids[b, p].item()
                print(f"  pos={p} ref_token={rid} ttx_token={tid}")

    assert torch.equal(ref_lens, ttx_lens), f"len mismatch: ref={ref_lens.tolist()} ttx={ttx_lens.tolist()}"
    assert torch.equal(ref_ids, ttx_ids), f"token mismatch"
    print(f"PASS | seed={TEST_SEED}")


if __name__ == "__main__":
    # test_reject(15, 155136, 3)
    pass
