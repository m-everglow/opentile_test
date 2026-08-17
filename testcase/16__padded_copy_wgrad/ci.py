import os
import numpy as np
import pytest
import time
import shutil
import torch
import triton
import triton.language as tl

os.environ['TRITON_ALL_BLOCKS_PARALLEL'] = '1'

if os.path.exists("./prof_padded_copy_scatter_wgrad_dir"):
    shutil.rmtree("./prof_padded_copy_scatter_wgrad_dir")

def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

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
        raise ValueError(f'Expected dimensions to be equal but got {a} and {b}.')


# x: (tokens, top_k, hidden_size), real
# grad: (tokens, hidden_size), real.
# wgrad: (tokens, top_k), real.
# indices: (tokens * top_k), integer.
# bin_ids: (tokens * top_k), integer.
# bins: (num_experts), integer.
# padded_bins: (num_experts), integer.
@triton.jit
def _padded_copy_wgrad(
    x,
    grad,
    wgrad,
    indices,
    bin_ids,
    bins,
    padded_bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    # Our index into 'tokens * top_k'.
    index_out = tl.load(indices + tl.program_id(0))

    # One threadblock per row in 'a'. Array 'b' has greater or equal
    # number of rows since they could be padded.
    bin_idx = tl.load(bin_ids + tl.program_id(0))

    # Now we know what bin we're assigned to, but we need to know how
    # many threadblocks were assigned to earlier bins so we can offset
    # in our bin properly.
    offset_in_bin = tl.program_id(0)
    if bin_idx > 0:
        offset_in_bin -= tl.load(bins + bin_idx - 1)

    # Load the starting index of our bin in array 'x'.
    index_x = offset_in_bin
    if bin_idx > 0:
        index_x += tl.load(padded_bins + bin_idx - 1)

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


def padded_scatter_wgrad(
    x,
    grad,
    indices,
    bin_ids,
    bins,
    padded_bins,
    top_k,
    block_x,
):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_matrix(grad)
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_is_vector(padded_bins)
    assert_equal(indices.shape[0], bin_ids.shape[0])
    assert_equal(bins.size(), padded_bins.size())

    tokens = indices.shape[0] // top_k
    out = torch.empty((tokens * top_k), dtype=x.dtype, device=x.device)
    _padded_copy_wgrad[(indices.shape[0],)](
        x,
        grad,
        out,
        indices,
        bin_ids,
        bins,
        padded_bins,
        NUM_COLUMNS=x.shape[1],
        TOP_K=top_k,
        BLOCK_X=block_x,
        compile_mode='simt_only',
    )
    return out



def round_up(x: torch.Tensor, value: int):
    assert isinstance(value, int)
    assert x.dtype == torch.int32
    return torch.div(x + (value - 1), value, rounding_mode="trunc") * value


def padded_gather_np(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    bins: torch.Tensor,
    padded_bins: torch.Tensor,
    top_k: int,
):
    x = x.cpu().numpy()
    indices = indices.cpu().numpy()
    bin_ids = bin_ids.cpu().numpy()
    bins = bins.cpu().numpy()
    padded_bins = padded_bins.cpu().numpy()

    out = np.zeros((padded_bins[-1], x.shape[1]))
    in_idx = 0
    for i, end in enumerate(bins):
        out_idx = 0 if i == 0 else padded_bins[i - 1]
        end = bins[i]
        while in_idx < end:
            load_idx = indices[in_idx] // top_k
            out[out_idx, :] = x[load_idx, :]
            in_idx += 1
            out_idx += 1
    return torch.from_numpy(out).npu().half()

def padded_scatter_wgrad_numpy(
    x: torch.Tensor,
    grads: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    bins: torch.Tensor,
    padded_bins: torch.Tensor,
    top_k: int,
):
    x = to_numpy(x).astype(np.float32)
    grads = to_numpy(grads).astype(np.float32)
    indices = to_numpy(indices)
    bin_ids = to_numpy(bin_ids)
    bins = to_numpy(bins)
    padded_bins = to_numpy(padded_bins)

    out = np.zeros(indices.shape).astype(np.float32)
    in_idx = 0
    for i, end in enumerate(bins):
        x_idx = 0 if i == 0 else padded_bins[i - 1]
        while in_idx < end:
            data = x[x_idx, :]
            grad_idx = indices[in_idx] // top_k
            grad = grads[grad_idx]
            out_idx = indices[in_idx]
            out[out_idx] = np.sum(data * grad)
            in_idx += 1
            x_idx += 1
    return torch.from_numpy(out).npu().half()

def round_up(x: torch.Tensor, value: int):
    assert isinstance(value, int)
    assert x.dtype == torch.int32
    return torch.div(x + (value - 1), value, rounding_mode="trunc") * value


def histc_manual(input_tensor: torch.Tensor, bins: int, min: float, max: float):
    x = input_tensor.flatten()
    step = (max - min) / bins
    boundaries = torch.linspace(min + step, max - step, bins - 1, device=x.device)
    bin_indices = torch.bucketize(x, boundaries)
    hist = torch.bincount(bin_indices, minlength=bins)
    return hist


def generate_data(sl: int, hs: int, ne: int, top_k: int):
    x = torch.randn((sl, hs)).npu().half()
    top_expert = torch.randint(0, ne, (sl * top_k,)).npu().int()
    bin_ids, indices = torch.sort(top_expert)
    tokens_per_expert = histc_manual(top_expert, ne, 0, ne - 1).to(torch.int32)
    padded_tokens_per_expert = round_up(tokens_per_expert, 128)
    padded_bins = torch.cumsum(padded_tokens_per_expert, dim=0).to(torch.int32)
    bins = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)
    weights = torch.rand((sl * top_k,)).npu().half()
    grads = torch.randn((sl, hs)).npu().half()
    return x.contiguous(), indices.contiguous(), bin_ids.contiguous(), bins.contiguous(), padded_bins.contiguous(), weights.contiguous(), grads.contiguous()


SHAPES = [
    (1024, 1024, 64, 4),        # 测试 hs = 1024
    (1024, 1536, 64, 4),
    (1024, 1536, 128, 4),
    (16384, 768, 64, 4),
    (16384, 768, 128, 4),
    # (4, 2, 2, 1),
    # (1024, 1536, 128, 4),
    # (16384, 768, 4, 1),
    # (1024, 1, 4, 2),
    # (16384, 1, 128, 2),
    # (1024, 1536, 4, 4)
]

BLOCK_X_VALUES = (64, 128, 256, 512, 1024)


@pytest.mark.parametrize(('sl', 'hs', 'ne', 'top_k'), SHAPES)
@pytest.mark.parametrize(
    'block_x',
    BLOCK_X_VALUES,
    ids=lambda block_x: f'BLOCK_X={block_x}',
)
def test_padded_copy_scatter_wgrad(
    sl: int,
    hs: int,
    ne: int,
    top_k: int,
    block_x: int,
):
    grid = sl * top_k
    if grid > 65535:
        pytest.skip(
            f"Ascend blockDim must not exceed 65535, but grid is {grid}"
        )

    import torch_npu

    total_start = time.time()

    shape_start = time.time()
    print(f"\n{'='*60}")
    print(
        f"Testing shape: sl={sl}, hs={hs}, ne={ne}, top_k={top_k}, "
        f"BLOCK_X={block_x}"
    )
    print(f"{'='*60}")

    x, indices, bin_ids, bins, padded_bins, weights, grads = generate_data(sl, hs, ne, top_k)
    gather_result = padded_gather_np(x, indices, bin_ids, bins, padded_bins, top_k)

    result = padded_scatter_wgrad(
        gather_result, grads, indices, bin_ids, bins, padded_bins, top_k,
        block_x
    )
    golden = padded_scatter_wgrad_numpy(
        gather_result, grads, indices, bin_ids, bins, padded_bins, top_k
    )
    # assert torch.allclose(result, golden, atol=1e-3, rtol=1e-3)
    # print("====accuracy ok===")

    atol = 1e-3
    rtol = 1e-3
    if not torch.allclose(result, golden, atol=atol, rtol=rtol):
        abs_diff = torch.abs(result - golden)
        tolerance = atol + rtol * torch.abs(golden)

        # 与 torch.allclose 的判断规则一致，并处理 NaN。
        mismatch_mask = (abs_diff > tolerance) | torch.isnan(abs_diff)
        mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=False)

        print("\n[ERROR] torch.allclose failed")
        print(f"result.shape:       {result.shape}")
        print(f"golden.shape:       {golden.shape}")
        print(f"mismatch count:     {mismatch_indices.shape[0]}")
        print(f"max abs diff:       {abs_diff.nan_to_num().max().item()}")
        print(f"mean abs diff:      {abs_diff.nan_to_num().mean().item()}")

        # 避免数据量过大，只打印前 20 个失败元素。
        max_print = 20
        print(f"\nFirst {min(max_print, mismatch_indices.shape[0])} mismatches:")

        for index in mismatch_indices[:max_print]:
            index_tuple = tuple(index.tolist())

            actual = result[index_tuple].item()
            expected = golden[index_tuple].item()
            diff = abs_diff[index_tuple].item()
            allowed = tolerance[index_tuple].item()

            print(
                f"index={index_tuple}, "
                f"result={actual}, "
                f"golden={expected}, "
                f"abs_diff={diff}, "
                f"tolerance={allowed}"
            )

        pytest.fail(
            f"padded_gather output mismatch: "
            f"{mismatch_indices.shape[0]} elements differ"
        )

    # for _ in range(5):
    #     out = padded_scatter_wgrad(
    #         gather_result, grads, indices, bin_ids, bins, padded_bins, top_k,
    #         block_x
    #     )
    #     torch_npu.npu.synchronize()

    # prof_dir = (
    #     "./prof_padded_copy_scatter_wgrad_dir/"
    #     f"shape_{sl}_{hs}_{ne}_{top_k}_block_x_{block_x}"
    # )

    # experimental_config = torch_npu.profiler._ExperimentalConfig(
    #     aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    #     profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    #     l2_cache=False
    # )

    # with torch_npu.profiler.profile(
    #     activities=[torch_npu.profiler.ProfilerActivity.NPU],
    #     with_stack=False,
    #     record_shapes=False,
    #     profile_memory=False,
    #     schedule=torch_npu.profiler.schedule(
    #         wait=1, warmup=1, active=20, repeat=1, skip_first=1
    #     ),
    #     experimental_config=experimental_config,
    #     on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir)
    # ) as prof:
    #     for i in range(40):
    #         out = padded_scatter_wgrad(
    #             gather_result, grads, indices, bin_ids, bins, padded_bins,
    #             top_k, block_x
    #         )
    #         torch_npu.npu.synchronize()
    #         prof.step()
    #     prof.stop()

    # shape_elapsed = time.time() - shape_start
    # print(f"Profile saved to: {prof_dir} (took {shape_elapsed:.1f}s)")

    # del x, indices, bin_ids, bins, padded_bins, weights, grads, out

    # total_elapsed = time.time() - total_start
    # print(f"\n\nAll profiling completed in {total_elapsed:.1f}s!")
    # print("Results are in ./prof_padded_copy_scatter_wgrad_dir/shape_* directories")
