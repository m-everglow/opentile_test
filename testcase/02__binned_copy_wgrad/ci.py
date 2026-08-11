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


@triton.jit
def _binned_copy_wgrad(
    x,
    grad,
    wgrad,
    num_experts,
    expert_capacity,
    indices,
    bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    # Load our indices into the output.
    expert_idx = tl.program_id(0)
    entry_idx = tl.program_id(1)

    # Calculate our offset into the output.
    index_x = expert_idx * expert_capacity + entry_idx

    # Load the index bounds for our bin and calculate
    # the number of tokens assigned to our expert.
    start = 0
    if expert_idx > 0:
        start = tl.load(bins + expert_idx - 1)
    end = tl.load(bins + expert_idx)
    num_tokens = end - start

    # Calculate our offset into the input. If we don't
    # have an input exit early.
    if entry_idx >= num_tokens:
        return
    index_out = tl.load(indices + start + entry_idx)

    # Offset the input and output pointers.
    wgrad += index_out
    grad += tl.multiple_of((index_out // TOP_K) * NUM_COLUMNS, NUM_COLUMNS)
    x += tl.multiple_of(index_x * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

    acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        data = tl.load(x + offsets, mask=mask).to(tl.float32)
        scale = tl.load(grad + offsets, mask=mask).to(tl.float32)
        acc += data * scale
        offsets += BLOCK_X

    # Reduce to get the final result and store.
    out = tl.sum(acc).to(wgrad.dtype.element_ty)
    tl.store(wgrad, out)


def binned_scatter_wgrad(x, grad, indices, bins, top_k):
    # Validate the input shapes.
    assert_is_tensor(x, 3)
    assert_is_matrix(grad)
    assert_is_vector(indices)
    assert_is_vector(bins)
    assert_equal(bins.shape[0], x.shape[0])

    num_experts, expert_capacity, hidden_size = x.shape
    tokens = indices.shape[0] // top_k
    out = torch.zeros((tokens * top_k), dtype=x.dtype, device=x.device)
    _binned_copy_wgrad[(num_experts, expert_capacity)](
        x,
        grad,
        out,
        num_experts,
        expert_capacity,
        indices,
        bins,
        NUM_COLUMNS=hidden_size,
        TOP_K=top_k,
        BLOCK_X=64
    )
    return out


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

_BINNED_SCATTER_Wgrad_TESTS = (
#  (4, 2, 2, 1),
#  (4, 2, 2, 2),
#  (4, 2, 2, 4),
    (8, 96, 1, 1),
#     (1, 128, 1, 1),
#     (1, 64, 1, 1),
#     (2, 64, 1, 1),
#     (3, 64, 1, 1),
#     (4, 64, 1, 1),
#     (8, 64, 1, 1),
    (16, 64, 1, 1),
    (32, 64, 1, 1),
#     (16, 128, 1, 1),
#     (32, 128, 1, 1),
#     (64, 128, 1, 1),
#     (1, 64, 1, 4),
#     (1, 64, 4, 4),
#     (2, 64, 2, 2),
#     (8, 64, 8, 1),
#     (1, 128, 1, 2),
#     (2, 128, 2, 2),
#     (16, 1, 1, 1),
#     (8, 16, 1, 2),
#     (1, 512, 1, 1),
    (1024, 1536, 4, 4),
    (1024, 1536, 64, 4),
    (1024, 1536, 128, 4),
    (1024, 1536, 128, 16),
    (16384, 1, 128, 2),
    (16384, 768, 4, 1),
    # (16384, 768, 64, 4),
    # (16384, 768, 128, 64),
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
@pytest.mark.parametrize(('sl', 'hs', 'ne', 'top_k'), _BINNED_SCATTER_Wgrad_TESTS)
def testBinnedScatterWgrad(sl: int, hs: int, ne: int, top_k: int):
    # NOTE: Capacity factor == 1.
    ec = (sl * top_k) // ne
    # SEED = 2026
    # generator = torch.Generator(device="cpu").manual_seed(SEED)
    # Create the data and indices.
    input_data = torch.randn((sl, hs)).to("npu").half()

    # Randomly assign tokens to experts.
    top_expert = torch.randint(0, ne, (sl * top_k,)).to("npu").int()
    _, indices = torch.sort(top_expert)
    tokens_per_expert = histc_manual(top_expert, ne, 0, ne - 1).to(torch.int32)
    bins = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)

    # Sample weights for the scatter reduce.
    weights = torch.rand((sl * top_k,)).to("npu").half()

    binned_gather_result = binned_gather(input_data, indices, None, bins, ec, top_k)
    binned_scatter_result = binned_scatter(binned_gather_result, indices, weights, bins, top_k)
    torch.set_printoptions(
    threshold=torch.inf,      # 不省略
    edgeitems=5,              # 每维开头和结尾显示的元素数（省略时有效）
    linewidth=120,            # 每行的宽度（字符数）
    sci_mode=False,           # 禁用科学计数法（可选）
    precision=4               # 小数位数
    )
    # grads = torch.randn_like(binned_scatter_result, generator = generator)
    grads = torch.randn(binned_scatter_result.shape, dtype=binned_scatter_result.dtype).npu()
    def binned_scatter_wgrad_golden(
        x: torch.Tensor,
        grads: torch.Tensor,
        indices: torch.Tensor,
        bins: torch.Tensor,
        top_k: int,
        ne: int,
        ec: int,
    ):
        x = to_numpy(x).astype(np.float32)
        grads = to_numpy(grads).astype(np.float32)
        indices = to_numpy(indices)
        bins = to_numpy(bins)

        out = np.zeros(indices.shape).astype(np.float32)
        start = 0
        for i in range(ne):
            end = bins[i]
            for j in range(min(ec, end - start)):
                index = indices[start + j]
                grad_idx = index // top_k
                grad = grads[grad_idx]
                out[index] = np.sum(grad.astype(np.float32) * x[i, j, :].astype(np.float32))
            start = end
        return torch.from_numpy(out).npu().half()
    # print("binned_gather_result", binned_gather_result,
    # "grads", grads,
    # "indices", indices,
    # "bins", bins,
    # "top_k", top_k,
    # "ne", ne,
    # "ec", ec, sep='\n')
    # out = binned_scatter_wgrad(binned_gather_result.npu().float(), grads.npu().float(), indices, bins, top_k)
    out = binned_scatter_wgrad(binned_gather_result, grads, indices, bins, top_k)
    expected_out = binned_scatter_wgrad_golden(binned_gather_result, grads, indices, bins, top_k, ne, ec)

    # NOTE: We need to check approximate equality because the
    # scatter reduce uses atomics.
    torch.npu.synchronize()
    # print(out)
    # print(expected_out)
    assert np.testing.assert_allclose(
        out.cpu(),
        expected_out.cpu(),
        rtol=1e-3,
        atol=1e-3,
    ) is None