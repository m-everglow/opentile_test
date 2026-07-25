"""Standalone e2e test for store_label_cache kernel (scatter store).

No external deps beyond torch + triton.  Run directly with:

    TRITON_ALWAYS_COMPILE=1 OPENTILE_KERNEL_MODE=aiv \
      python3 -m pytest -q -s test_store_e2e.py
"""

import os
import pytest
import torch
import triton
import triton.language as tl

# ── Triton kernel (inline) ──────────────────────────────────────────────

@triton.jit
def _store_label_cache_kernel(
    label_cache_ptr,
    key_lr_ptr,
    block_idx_list_ptr,
    token_idx_list_ptr,
    head_num: tl.constexpr,
    head_dim: tl.constexpr,
    token_num: tl.constexpr,
    l_stride_b: tl.constexpr,
    l_stride_h: tl.constexpr,
    l_stride_t: tl.constexpr,
    l_stride_d: tl.constexpr,
    k_stride_s: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    BATCH_BLOCK_NUM: tl.constexpr,
):
    pid_b = tl.program_id(0)
    b_start = pid_b * BATCH_BLOCK_NUM
    b_end = tl.minimum(b_start + BATCH_BLOCK_NUM, token_num)
    b = tl.arange(0, BATCH_BLOCK_NUM) + b_start
    b_3d = b[:, None, None]
    h = tl.arange(0, head_num)
    h_3d = h[None, :, None]
    d = tl.arange(0, head_dim)
    d_3d = d[None, None, :]

    block_idx = tl.load(block_idx_list_ptr + b_3d, mask=(b_3d < b_end), other=0)
    token_idx = tl.load(token_idx_list_ptr + b_3d, mask=(b_3d < b_end), other=0)

    label_cache_addr = (
        block_idx * l_stride_b + h_3d * l_stride_h
        + token_idx * l_stride_t + d_3d * l_stride_d
    )
    key_lr_offset = (
        b_3d * k_stride_s + h_3d * k_stride_h + d_3d * k_stride_d
    )
    valid_mask = (b_3d < b_end) & (h_3d < head_num) & (d_3d < head_dim)

    key_lr_data = tl.load(key_lr_ptr + key_lr_offset, mask=valid_mask, other=0.0)
    tl.store(label_cache_ptr + label_cache_addr, key_lr_data, mask=valid_mask)


# ── Launcher ────────────────────────────────────────────────────────────

def store_label_cache(
    label_cache: torch.Tensor,
    key_lr: torch.Tensor,
    block_idxs: torch.Tensor,
    token_idxs: torch.Tensor,
):
    batch, head_num, block_size, head_dim = label_cache.shape
    token_num = block_idxs.shape[0]
    BATCH_BLOCK_NUM = 16
    num_programs = (token_num + BATCH_BLOCK_NUM - 1) // BATCH_BLOCK_NUM
    grid = (num_programs,)

    _store_label_cache_kernel[grid](
        label_cache_ptr=label_cache,
        key_lr_ptr=key_lr,
        block_idx_list_ptr=block_idxs,
        token_idx_list_ptr=token_idxs,
        head_num=head_num,
        head_dim=head_dim,
        token_num=token_num,
        l_stride_b=label_cache.stride(0),
        l_stride_h=label_cache.stride(1),
        l_stride_t=label_cache.stride(2),
        l_stride_d=label_cache.stride(3),
        k_stride_s=key_lr.stride(0),
        k_stride_h=key_lr.stride(1),
        k_stride_d=key_lr.stride(2),
        BATCH_BLOCK_NUM=BATCH_BLOCK_NUM,
    )
    return label_cache


# ── Torch reference ─────────────────────────────────────────────────────

def store_label_cache_ref(label_cache, key_lr, block_idxs, token_idxs):
    label_cache[block_idxs, :, token_idxs, :] = key_lr
    return label_cache


# ── Test ────────────────────────────────────────────────────────────────

TEST_SEED = int(os.environ.get("OPENTILE_TEST_SEED", "2026"))


@pytest.mark.parametrize("label_cache_shape,key_lr_shape,token_num", [
    pytest.param((256, 1, 512, 128), (1, 128), 16, id="small"),
])
def test_store(label_cache_shape, key_lr_shape, token_num):
    gen = torch.Generator(device="cpu").manual_seed(TEST_SEED)

    slot_mapping = torch.randperm(token_num, generator=gen)
    label_cache = torch.zeros(label_cache_shape, dtype=torch.float32)
    key_lr = torch.randn((token_num, *key_lr_shape), generator=gen, dtype=torch.float32)
    block_idxs = slot_mapping // 512
    token_idxs = slot_mapping % 512

    ref = store_label_cache_ref(
        label_cache.clone(), key_lr.clone(), block_idxs, token_idxs
    )
    out = store_label_cache(
        label_cache.clone(), key_lr.clone(), block_idxs, token_idxs
    )
    torch.npu.synchronize()

    torch.testing.assert_close(out.float(), ref.float(), atol=1e-5, rtol=1e-5)
    print(f"PASS | seed={TEST_SEED}")


if __name__ == "__main__":
    # test_store((256, 1, 512, 128), (1, 128), 16)
    pass
