

import functools
import math

import pytest
import torch

from tests.utils import auto_switch_platform
from tests.utils import bypass_not_implemented
from mojo_opset import MojoSdpa
from mojo_opset.backends.ttx.kernels import sdpa_fwd_impl, sdpa_bwd_impl

dtypes = [torch.float16, torch.bfloat16, torch.float32]


def generate_diffusion_attention_mask(
    seq_length: int,
    block_size: int,
) -> torch.Tensor:
    total_length = seq_length * 2
    attn_mask = torch.zeros(total_length, total_length, dtype=torch.int8)

    for i in range(total_length):
        for j in range(total_length):
            block_i = i // block_size
            block_j = j // block_size
            if block_i == block_j:
                attn_mask[i, j] = 1

            if j >= seq_length and i < seq_length and ((j - seq_length) // block_size) < block_i:
                attn_mask[i, j] = 1

            if i >= seq_length and j >= seq_length and block_j < block_i:
                attn_mask[i, j] = 1

    return attn_mask.to(torch.bool)


def check_tol_diff(
    norm: torch.Tensor,
    ref: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    ptol: float = 1.0,
    mixed_tol: bool = False,
):
    """
    Args:
        norm: The computed/estimated value to be validated.
        ref: The reference/ground truth value for comparison.
        atol: The absolute tolerance.
        rtol: The relative tolerance.
        ptol: The percentage tolerance. When match_ratio >= ptol is considered to pass.
        mixed_tol: If true, atol, rtol and ptol are ignored.
    """
    if isinstance(norm, tuple) or isinstance(norm, list):
        for norm_i, ref_i in zip(norm, ref):
            check_tol_diff(norm_i, ref_i, atol, rtol, ptol, mixed_tol)
        return

    if mixed_tol:
        mask = ref.abs() < 1.0
        tmpatol = tmprtol = 2**-6
        torch.testing.assert_close(norm[mask], ref[mask], atol=tmpatol, rtol=0)
        torch.testing.assert_close(norm[~mask], ref[~mask], atol=0, rtol=tmprtol)

    elif ptol != 1.0:
        assert ptol < 1.0, f"{ptol=} should <= 1.0"

        matches = torch.isclose(norm, ref, rtol=rtol, atol=atol)
        total = matches.numel()
        match = int(torch.sum(matches))
        mismatch = total - match
        match_ratio = match / total

        assert match_ratio >= ptol, (
            f"{match_ratio=:.5%} ({match=} / {mismatch=} / {total=}) is under {ptol=:%}, Please Check!"
        )

    else:
        torch.testing.assert_close(norm.to(torch.float32), ref.to(torch.float32), atol=atol, rtol=rtol)


def generate_test_data(
    bsz: int,
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
    seq_length: int,
    block_size: int,
    dtype: torch.dtype,
):
    query = torch.randn(bsz, q_head_num, seq_length * 2, head_dim, dtype=dtype, requires_grad=False).npu()
    key = torch.randn(bsz, kv_head_num, seq_length * 2, head_dim, dtype=dtype, requires_grad=False).npu()
    value = torch.randn(bsz, kv_head_num, seq_length * 2, head_dim, dtype=dtype, requires_grad=False).npu()
    blockwise_diffusion_attn_mask = generate_diffusion_attention_mask(seq_length, block_size).npu()
    return query, key, value, blockwise_diffusion_attn_mask, q_head_num != kv_head_num


CONFIG = [
    (1, 5, 1, 128, 2048, 32,),
    (2, 8, 8, 64, 512, 16,),
    (4, 16, 4, 128, 1024, 64,),
    (1, 32, 1, 128, 4096, 128,),
    (8, 12, 12, 64, 2048, 32,),
    (2, 6, 2, 128, 768, 48,),
    (1, 1, 1, 128, 128, 16,),  # head_dim不符合kernel要求需改写
    (3, 15, 5, 64, 8192, 256,),
    (1, 8, 8, 64, 1024, 1,),  # head_dim不符合kernel要求需改写
    (2, 9, 3, 128, 3072, 64,)
]


@pytest.mark.parametrize(
    "bsz, q_head_num, kv_head_num, head_dim, seq_length, block_size, dtype",
    [
        pytest.param(bsz, q_head_num, kv_head_num, head_dim, seq_length, block_size, dtype)
        for bsz, q_head_num, kv_head_num, head_dim, seq_length, block_size in CONFIG
        for dtype in dtypes
    ],
)
@auto_switch_platform(set_perf=True)
def test_sdpa(
    bsz: int,
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
    seq_length: int,
    block_size: int,
    dtype: torch.dtype,
):
    query, key, value, blockwise_diffusion_attn_mask, enable_gqa = generate_test_data(
        bsz, q_head_num, kv_head_num, head_dim, seq_length, block_size, dtype
    )
    diffusion_attn = MojoSdpa(
        scale=1.0 / math.sqrt(query.shape[-1]), enable_gqa=enable_gqa
    )
    perf(lambda: diffusion_attn(query, key, value, blockwise_diffusion_attn_mask))


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16]
)
@pytest.mark.parametrize(
    "bsz, q_head_num, kv_head_num, head_dim, seq_length, block_size",
    [(1, 5, 1, 128, 2048, 32,),
     (2, 8, 8, 64, 512, 16),
     (4, 16, 4, 128, 1024, 64),
     (1, 32, 1, 128, 4096, 128),
     (8, 12, 12, 64, 2048, 32),
     (2, 6, 2, 128, 768, 48),
     (1, 1, 1, 128, 128, 16),
     (3, 15, 5, 64, 8192, 256),
     (1, 8, 8, 64, 1024, 1),
     (2, 9, 3, 128, 3072, 64),],
)
@auto_switch_platform(set_perf=True)
def test_sdpa_fwd_func_perf(
    bsz,
    q_head_num,
    kv_head_num,
    head_dim,
    seq_length,
    block_size,
    dtype: torch.dtype,
):
    query, key, value, blockwise_diffusion_attn_mask, enable_gqa = generate_test_data(
        bsz, q_head_num, kv_head_num, head_dim, seq_length, block_size, dtype
    )
    scale = 1.0 / math.sqrt(query.shape[-1])
    perf(lambda: sdpa_fwd_impl(
        q=query,
        k=key,
        v=value,
        mask=blockwise_diffusion_attn_mask,
        scale=scale,
        gqa_enabled=enable_gqa,
    ))

@pytest.mark.parametrize(
    "dtype", [torch.bfloat16]
)
@pytest.mark.parametrize(
    "bsz, q_head_num, kv_head_num, head_dim, seq_length, block_size",
    [(1, 5, 1, 128, 2048, 32,),
     (2, 8, 8, 64, 512, 16),
     (4, 16, 4, 128, 1024, 64),
     (1, 32, 1, 128, 4096, 128),
     (8, 12, 12, 64, 2048, 32),
     (2, 6, 2, 128, 768, 48),
     (1, 1, 1, 128, 128, 16),
     (3, 15, 5, 64, 8192, 256),
     (1, 8, 8, 64, 1024, 1),
     (2, 9, 3, 128, 3072, 64),],
)
@auto_switch_platform(set_perf=True)
def test_sdpa_bwd_func_perf(
    bsz,
    q_head_num,
    kv_head_num,
    head_dim,
    seq_length,
    block_size,
    dtype: torch.dtype,
):
    query, key, value, blockwise_diffusion_attn_mask, enable_gqa = generate_test_data(
        bsz, q_head_num, kv_head_num, head_dim, seq_length, block_size, dtype
    )
    scale = 1.0 / math.sqrt(query.shape[-1])

    query.requires_grad = True
    key.requires_grad = True
    value.requires_grad = True

    o, lse = sdpa_fwd_impl(
        q=query,
        k=key,
        v=value,
        mask=blockwise_diffusion_attn_mask,
        scale=scale,
        gqa_enabled=enable_gqa,
    )

    grad_output = torch.randn_like(o)

    perf(lambda: sdpa_bwd_impl(
        o=o,
        do=grad_output,
        q=query,
        k=key,
        v=value,
        lse=lse,
        mask=blockwise_diffusion_attn_mask,
        scale=scale,
        gqa_enabled=enable_gqa,
    ))


@pytest.mark.parametrize(
    "bsz, q_head_num, kv_head_num, head_dim, seq_length, block_size, dtype",
    [
        pytest.param(bsz, q_head_num, kv_head_num, head_dim, seq_length, block_size, dtype)
        for bsz, q_head_num, kv_head_num, head_dim, seq_length, block_size in CONFIG
        for dtype in dtypes
    ],
)
@bypass_not_implemented
def test_sdpa_accuracy(
    bsz: int,
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
    seq_length: int,
    block_size: int,
    dtype: torch.dtype,
):
    query, key, value, blockwise_diffusion_attn_mask, enable_gqa = generate_test_data(
        bsz, q_head_num, kv_head_num, head_dim, seq_length, block_size, dtype
    )
    diffusion_attn_ref = MojoSdpa._registry.get("torch")(
        scale=1.0 / math.sqrt(query.shape[-1]), enable_gqa=enable_gqa
    )
    diffusion_attn = MojoSdpa(
        scale=1.0 / math.sqrt(query.shape[-1]), enable_gqa=enable_gqa
    )
    diffusion_attn_ref.forward_diff_with(diffusion_attn, query, key, value, blockwise_diffusion_attn_mask)

@pytest.mark.parametrize(
    "dtype", [torch.bfloat16]
)
@pytest.mark.parametrize(
    "bsz, q_head_num, kv_head_num, head_dim, seq_length, block_size",
    [(1, 5, 1, 128, 2048, 32,),
     (2, 8, 8, 64, 512, 16),
     (4, 16, 4, 128, 1024, 64),
     (1, 32, 1, 128, 4096, 128),
     (8, 12, 12, 64, 2048, 32),
     (2, 6, 2, 128, 768, 48),
     (1, 1, 1, 128, 128, 16),
     (3, 15, 5, 64, 8192, 256),
     (1, 8, 8, 64, 1024, 1),
     (2, 9, 3, 128, 3072, 64),],
)
def test_sdpa_fwd_func(
    bsz,
    q_head_num,
    kv_head_num,
    head_dim,
    seq_length,
    block_size,
    dtype: torch.dtype,
):
    atol: float = 1e-2
    rtol: float = 1e-2
    ptol: float = 1.0
    random_seed: int = 42
    mixed_tol: bool = False

    query, key, value, blockwise_diffusion_attn_mask, enable_gqa = generate_test_data(
        bsz, q_head_num, kv_head_num, head_dim, seq_length, block_size, dtype
    )

    scale = 1.0 / math.sqrt(query.shape[-1])

    # Test our implementation
    o, lse = sdpa_fwd_impl(
        q=query.npu(),
        k=key.npu(),
        v=value.npu(),
        mask=blockwise_diffusion_attn_mask.npu(),
        scale=scale,
        gqa_enabled=enable_gqa,
    )

    # Get reference
    o_ref = torch.nn.functional.scaled_dot_product_attention(
        query.npu(),
        key.npu(),
        value.npu(),
        attn_mask=blockwise_diffusion_attn_mask.npu(),
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
        enable_gqa=enable_gqa,
    )

    assert o is not None, "forward should return a non-None value."
    assert o_ref is not None, "comparison operator should return a non-None value."
    assert lse is not None, "lse should return a non-None value."

    check_tol_diff(o, o_ref, atol, rtol, ptol, mixed_tol)

@pytest.mark.parametrize(
    "dtype", [torch.bfloat16]
)
@pytest.mark.parametrize(
    "bsz, q_head_num, kv_head_num, head_dim, seq_length, block_size",
    [(1, 5, 1, 128, 2048, 32,),
     (2, 8, 8, 64, 512, 16),
     (4, 16, 4, 128, 1024, 64),
     (1, 32, 1, 128, 4096, 128),
     (8, 12, 12, 64, 2048, 32),
     (2, 6, 2, 128, 768, 48),
     (1, 1, 1, 128, 128, 16),
     (3, 15, 5, 64, 8192, 256),
     (1, 8, 8, 64, 1024, 1),
     (2, 9, 3, 128, 3072, 64),],
)
def test_sdpa_bwd_func(
    bsz,
    q_head_num,
    kv_head_num,
    head_dim,
    seq_length,
    block_size,
    dtype: torch.dtype,
):
    atol: float = 1e-2
    rtol: float = 1e-2
    ptol: float = 0.95  # Slightly lower tolerance for backward pass
    mixed_tol: bool = True

    # Generate test data
    query, key, value, blockwise_diffusion_attn_mask, enable_gqa = generate_test_data(
        bsz, q_head_num, kv_head_num, head_dim, seq_length, block_size, dtype
    )

    scale = 1.0 / math.sqrt(query.shape[-1])

    # Make tensors require grad
    query.requires_grad = True
    key.requires_grad = True
    value.requires_grad = True

    # Forward pass with PyTorch for reference gradients
    o_ref = torch.nn.functional.scaled_dot_product_attention(
        query.npu(),
        key.npu(),
        value.npu(),
        attn_mask=blockwise_diffusion_attn_mask.npu(),
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
        enable_gqa=enable_gqa,
    )

    # Generate random gradients
    grad_output = torch.randn_like(o_ref)

    # Backward pass with PyTorch
    o_ref.backward(grad_output)
    dq_ref = query.grad
    dk_ref = key.grad
    dv_ref = value.grad

    # Reset gradients
    query.grad = None
    key.grad = None
    value.grad = None

    # Forward pass with our implementation
    o, lse = sdpa_fwd_impl(
        q=query.npu(),
        k=key.npu(),
        v=value.npu(),
        mask=blockwise_diffusion_attn_mask.npu(),
        scale=scale,
        gqa_enabled=enable_gqa,
    )

    # Backward pass with our implementation
    dq, dk, dv = sdpa_bwd_impl(
        o=o,
        do=grad_output,
        q=query.npu(),
        k=key.npu(),
        v=value.npu(),
        lse=lse,
        mask=blockwise_diffusion_attn_mask.npu(),
        scale=scale,
        gqa_enabled=enable_gqa,
    )

    # Check results
    assert dq is not None, "dq should return a non-None value."
    assert dk is not None, "dk should return a non-None value."
    assert dv is not None, "dv should return a non-None value."
    assert dq_ref is not None, "dq_ref should return a non-None value."
    assert dk_ref is not None, "dk_ref should return a non-None value."
    assert dv_ref is not None, "dv_ref should return a non-None value."

    check_tol_diff(dq, dq_ref.npu(), atol, rtol, ptol, mixed_tol)
    check_tol_diff(dk, dk_ref.npu(), atol, rtol, ptol, mixed_tol)
    check_tol_diff(dv, dv_ref.npu(), atol, rtol, ptol, mixed_tol)
