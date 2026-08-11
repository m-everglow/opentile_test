# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
import os
import numpy as np
import pytest
import torch

# from megablocks import ops

import torch
import triton
import triton.language as tl

os.environ['TRITON_ALL_BLOCKS_PARALLEL'] = '1'


def assert_is_tensor(x, ndim):
    if x.ndim != ndim:
        raise ValueError(f'Expected {ndim}-tensor but got {x.ndim}-tensor')


def assert_is_matrix(x):
    assert_is_tensor(x, 2)


def assert_is_vector(x):
    if x.ndim != 1:
        raise ValueError(f'Expected 1-tensor but got {x.ndim}-tensor')


def assert_equal(a, b):
    if a != b:
        raise ValueError(f'Expected dimensions to be equal but got {a} and {b}.',)


def histc_manual(input_tensor: torch.Tensor, bins: int, min: float, max: float):
    x = input_tensor.flatten()
    step = (max - min) / bins
    boundaries = torch.linspace(min + step, max - step, bins - 1, device=x.device)
    bin_indices = torch.bucketize(x, boundaries)
    hist = torch.bincount(bin_indices, minlength=bins)
    return hist


# a: (tokens, hidden_size), real.
# b: (num_experts, expert_capacity, num_columns), real.
# indices: (tokens * top_k), integer.
# weights: (tokens * top_k), real.
# bins: (num_experts), integer.

@triton.jit
def _binned_copy(
    a,
    b,
    num_experts,
    expert_capacity,
    indices,
    weights,
    bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
    A_TO_B: tl.constexpr,
    SCALE: tl.constexpr,
):
    # Load our indices into the output.
    expert_idx = tl.program_id(0) # i
    entry_idx = tl.program_id(1) # j

    # Calculate our offset into the output.
    index_b = expert_idx * expert_capacity + entry_idx # i * ec + j

    # Load the index bounds for our bin and calculate
    # the number of tokens assigned to our expert.
    start = 0
    if expert_idx > 0:
        start = tl.load(bins + expert_idx - 1)
    end = tl.load(bins + expert_idx) # bins[i]
    num_tokens = end - start

    # Calculate our offset into the input. If we don't
    # have an input exit early.
    if entry_idx >= num_tokens:
        return
    index_a = tl.load(indices + start + entry_idx) # indices[start + j]

    # Offset the input and output pointers.
    #
    # If we're going from A to B, divide the input index to copy
    # the same input repeatedly. If we're going from B to A we
    # need to reduce the result. Using atomics is slow, so we
    # do the reduce step in a second kernel.
    offset = index_a // TOP_K if A_TO_B else index_a # index
    a += tl.multiple_of(offset * NUM_COLUMNS, NUM_COLUMNS)
    b += tl.multiple_of(index_b * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

    # Load the scale, if requested.
    scale = tl.load(weights + index_a) if SCALE else 1

    # Swap the pointers depending on the direction.
    #
    # NOTE: We need to zero the output in both directions.
    iptr = a if A_TO_B else b
    optr = b if A_TO_B else a

    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        # mask = offsets == offsets
        x = tl.load(iptr + offsets, mask=mask)
        x = x.to(tl.float32) * scale.to(tl.float32)

        tl.store(optr + offsets, x.to(optr.dtype.element_ty), mask=mask)

        offsets += BLOCK_X


def binned_gather(x, indices, weights, bins, expert_capacity, top_k):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bins)
    assert_equal(indices.shape[0], x.shape[0] * top_k)

    if weights is not None:
        assert_equal(weights.shape[0], x.shape[0] * top_k)

    num_experts = bins.shape[0]
    out = torch.zeros((num_experts, expert_capacity, x.shape[1]), dtype=x.dtype, device=x.device)
    _binned_copy[(num_experts, expert_capacity)](
        x,
        out,
        num_experts,
        expert_capacity,
        indices,
        weights,
        bins,
        NUM_COLUMNS=x.shape[1],
        A_TO_B=True,
        TOP_K=top_k,
        SCALE=weights is not None,
        BLOCK_X=64
    )
    return out


def binned_scatter(x, indices, weights, bins, top_k):
    # Validate the input shapes.
    assert_is_tensor(x, 3)
    assert_is_vector(indices)
    assert_is_vector(bins)
    assert_equal(bins.shape[0], x.shape[0])

    if weights is not None:
        assert_equal(indices.shape[0], weights.shape[0])

    num_experts, expert_capacity, hidden_size = x.shape
    tokens = indices.shape[0] // top_k
    out = torch.zeros((tokens, top_k, hidden_size), dtype=x.dtype, device=x.device)
    _binned_copy[(num_experts, expert_capacity)](
        out,
        x,
        num_experts,
        expert_capacity,
        indices,
        weights,
        bins,
        NUM_COLUMNS=hidden_size,
        A_TO_B=False,
        TOP_K=top_k,
        SCALE=weights is not None,
        BLOCK_X=64
    )

    # Reduce along the top-k dimension, if needed.
    return out.sum(dim=1) if top_k > 1 else out.view(tokens, hidden_size)


# a: (tokens, hidden_size), real.
# b: (num_experts, expert_capacity, num_columns), real.
# indices: (tokens * top_k), integer.
# weights: (tokens * top_k), real.
# bins: (num_experts), integer.





BINNED_GATHER_TESTS = (
    (4, 2, 1, 2),
#     (8, 2, 2, 1),
#     (2, 32, 1, 2),
    (4, 64, 2, 2),
    (4, 256, 2, 4),
#     (128, 64, 2, 64),
#     (1024, 1536, 4, 4),
#     (1024, 1536, 64, 4),
#     (1024, 1536, 128, 4),
#     (1024, 1536, 128, 4),
#     (16384, 1, 128, 2),
#     (16384, 768, 4, 1),
#     (16384, 768, 64, 4),
#     (16384, 768, 128, 4),
)

# (4, 2, 2, 1),
# (4, 2, 2, 2),
# (4, 2, 2, 4),
# (1024, 1536, 4, 1),
# (1024, 1536, 4, 2),
# (1024, 1536, 4, 4),
# (1024, 1536, 64, 1),
# (1024, 1536, 64, 2),
# (1024, 1536, 64, 4),
# (1024, 1536, 128, 1),
# (1024, 1536, 128, 2),
# (1024, 1536, 128, 4),
# (16384, 768, 4, 1),
# (16384, 768, 4, 2),
# (16384, 768, 4, 4),
# (16384, 768, 64, 1),
# (16384, 768, 64, 2),
# (16384, 768, 64, 4),
# (16384, 768, 128, 1),
# (16384, 768, 128, 2),
# (16384, 768, 128, 4),


@pytest.mark.gpu
@pytest.mark.parametrize(('sl', 'hs', 'ne', 'top_k'), BINNED_GATHER_TESTS)
def test_binned_gather(sl: int, hs: int, ne: int, top_k: int):
    # NOTE: Capacity factor == 1.
    ec = (sl * top_k) // ne
    SEED = 2026
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    # Create the data and indices.
    x = torch.randn((sl, hs), generator = generator).npu().half()

    # Randomly assign tokens to experts.
    top_expert = torch.randint(0, ne, (sl * top_k,), generator = generator).npu().int()
    _, indices = torch.sort(top_expert)
    bin_ids, indices = torch.sort(top_expert)
    tokens_per_expert = histc_manual(top_expert, ne, 0, ne - 1).to(torch.int32)
    bins = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)
    # weights = torch.rand((sl * top_k,)).npu().half()
    torch.set_printoptions(
    threshold=torch.inf,      # 不省略
    edgeitems=5,              # 每维开头和结尾显示的元素数（省略时有效）
    linewidth=120,            # 每行的宽度（字符数）
    sci_mode=False,           # 禁用科学计数法（可选）
    precision=4               # 小数位数
    )
    def binned_gather_golden(
        x: torch.Tensor,
        indices: torch.Tensor,
        bins: torch.Tensor,
        ec: int,
        top_k: int,
    ):
        x = x.cpu().numpy()
        indices = indices.cpu().numpy()
        bins = bins.cpu().numpy()
        start = 0
        out = np.zeros((ne, ec, hs))
        for i in range(ne):
            # if i > 0:
            #     start = bins[i - 1]
            end = bins[i]
            for j in range(min(ec, end - start)):
                index = indices[start + j] // top_k
                out[i, j, :] = x[index, :]
            start = end
        return torch.from_numpy(out).cpu().half()
    out = binned_gather(x, indices, None, bins, ec, top_k)
    expected_out = binned_gather_golden(x, indices, bins, ec, top_k)
    # assert torch.all(torch.eq(out, expected_out))
    assert np.testing.assert_allclose(
        out.cpu(),
        expected_out.cpu(),
        rtol=1e-3,
        atol=1e-3,
    ) is None



_BINNED_SCATTER_TESTS = (
    (4, 2, 1, 2),
    (4, 2, 2, 1),
    (4, 512, 2, 2),
#     (1024, 1, 4, 2),
#     (1024, 1536, 4, 4),
#     (1024, 1536, 64, 4),
#     (1024, 1536, 128, 4),
#     (1024, 1536, 128, 4),
#     (16384, 1, 128, 2),
#     (16384, 768, 4, 1),
#     (16384, 768, 64, 4),
#     (16384, 768, 128, 4),
)

# (4, 2, 2, 1),
# (4, 2, 2, 2),
# (4, 2, 2, 4),
# (1024, 1536, 4, 1),
# (1024, 1536, 4, 2),
# (1024, 1536, 4, 4),
# (1024, 1536, 64, 1),
# (1024, 1536, 64, 2),
# (1024, 1536, 64, 4),
# (1024, 1536, 128, 1),
# (1024, 1536, 128, 2),
# (1024, 1536, 128, 4),
# (16384, 768, 4, 1),
# (16384, 768, 4, 2),
# (16384, 768, 4, 4),
# (16384, 768, 64, 1),
# (16384, 768, 64, 2),
# (16384, 768, 64, 4),
# (16384, 768, 128, 1),
# (16384, 768, 128, 2),
# (16384, 768, 128, 4),


@pytest.mark.gpu
@pytest.mark.parametrize(('sl', 'hs', 'ne', 'top_k'), _BINNED_SCATTER_TESTS)
def testBinnedScatter(sl: int, hs: int, ne: int, top_k: int):
    # NOTE: Capacity factor == 1.
    ec = (sl * top_k) // ne
    SEED = 2026
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    # Create the data and indices.
    x = torch.randn((sl, hs), generator = generator).npu().half()

    # Randomly assign tokens to experts.
    top_expert = torch.randint(0, ne, (sl * top_k,), generator = generator).npu().int()
    _, indices = torch.sort(top_expert)
    # bins = ops.inclusive_cumsum(ops.histogram(top_expert, ne), 0)
    tokens_per_expert = histc_manual(top_expert, ne, 0, ne - 1).to(torch.int32)
    bins = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)

    # Sample weights for the scatter reduce.
    weights = torch.rand((sl * top_k,), generator = generator).npu().half()
    x = binned_gather(x, indices, None, bins, ec, top_k)
    def binned_scatter_golden(
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        bins: torch.Tensor,
        top_k: int,
    ):
        x = x.cpu().numpy()
        indices = indices.cpu().numpy()
        weights = weights.cpu().numpy()
        bins = bins.cpu().numpy()
        start = 0
        out = np.zeros((sl, hs))
        for i in range(ne):
            end = bins[i]
            for j in range(min(ec, end - start)):
                index = indices[start + j]
                scale = weights[index]
                index //= top_k

                out[index, :] += scale * x[i, j, :]
            start = end
        return torch.from_numpy(out).cpu().half()
    out = binned_scatter(x, indices, weights, bins, top_k)
    expected_out = binned_scatter_golden(x, indices, weights, bins, top_k)
    # NOTE: We need to check approximate equality because the
    # scatter reduce uses atomics.
    assert np.testing.assert_allclose(
        out.cpu(),
        expected_out.cpu(),
        rtol=1e-3,
        atol=1e-3,
    ) is None

