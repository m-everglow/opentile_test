"""A5 end-to-end test for the production fused-add LayerNorm forward kernel.

Production kernel:
  HighPriority50Operators/mojo_opset-master/mojo_opset/backends/ttx/kernels/
  npu/fused_add_layernorm.py::_fused_add_layernorm_fwd_kernel
  SHA256 df9565c7bf3596f6b9da3a2f195018559255ca8576b098847b5679f4ec69f6ec

Input contract:
  mojo_opset-master/tests/perf/test_normalization.py::
  test_residual_add_layernorm
  SHA256 e7945bef05a9fddec1a8ddd99060f330a6952cf6d3f3924e24c7d9a88ad9760a

This selects one complete random case from that parameterization:
BF16 [128, 128], eps=1e-5, norm_pos="pre".  The Triton kernel body below is
identical to the production forward body; only heuristics/libentry decorators
are omitted so the case can drive OpenTile through ASTSource explicitly.
"""

import os

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_BACKEND", "opentile")
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")


def _npu_available():
    try:
        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


if not _npu_available():
    pytest.skip("torch_npu is available, but no NPU device is visible", allow_module_level=True)

torch.npu.set_device(int(os.environ.get("OPENTILE_TEST_DEVICE", "0")))

import torch.nn.functional as F  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402


SHAPES = [
    (32, 1024),
    (64, 8192),
    (2, 256),
    (128, 128),
    (256, 256),
    (512, 128),
]
DTYPES = [torch.bfloat16, torch.float32, torch.float16]
EPS = 1e-5
ADD_MODE = "pre"
BLOCK_SIZE_N = 128
# The production heuristic returns 24 for n_cols=128.  This direct ASTSource
# harness uses 16 because Triton's public tl.arange contract requires a
# power-of-two extent.  This changes only row batching, not the kernel body or
# the covered input/output contract.
BLOCK_SIZE_M = 16

DTYPE_TO_PTR = {
    torch.bfloat16: "*bf16",
    torch.float32: "*fp32",
    torch.float16: "*fp16",
}
DTYPE_TO_LABEL = {
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
    torch.float16: "fp16",
}
DTYPE_TOLERANCE = {
    torch.bfloat16: (3e-2, 6e-3),
    torch.float32: (1e-5, 1e-6),
    torch.float16: (3e-2, 6e-3),
}


@triton.jit
def _fused_add_layernorm_fwd_kernel(
    Y_ptr,
    Y_row_stride,
    S_ptr,
    S_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr,
    R_row_stride,
    W_ptr,
    B_ptr,
    Mean_ptr,
    Mean_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    for row_task_id in range(pid, num_row_tasks, grid_size):
        block_start_row = row_task_id * BLOCK_SIZE_M
        rows_off = block_start_row + tl.arange(0, BLOCK_SIZE_M)
        rows_mask = rows_off < n_rows

        X_ptr_row_block = X_ptr + rows_off[:, None] * X_row_stride
        R_ptr_row_block = R_ptr + rows_off[:, None] * R_row_stride
        S_ptr_row_block = S_ptr + rows_off[:, None] * S_row_stride
        Y_ptr_row_block = Y_ptr + rows_off[:, None] * Y_row_stride

        mean_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        var_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            block_mask = rows_mask[:, None] & (cols_off[None, :] < n_cols)

            X_chunk = tl.load(X_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
            R_chunk = tl.load(R_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0)
            S_chunk = X_chunk + R_chunk
            tl.store(S_ptr_row_block + cols_off[None, :], S_chunk, mask=block_mask)

            S_chunk_f32 = S_chunk.to(tl.float32)
            mean_acc += tl.sum(S_chunk_f32, axis=1)
            var_acc += tl.sum(S_chunk_f32 * S_chunk_f32, axis=1)

        mean_vec = mean_acc / n_cols
        var_vec = (var_acc / n_cols) - (mean_vec * mean_vec)
        rstd_vec = tl.rsqrt(var_vec + eps)
        tl.store(Mean_ptr + rows_off * Mean_row_stride, mean_vec, mask=rows_mask)
        tl.store(RSTD_ptr + rows_off * RSTD_row_stride, rstd_vec, mask=rows_mask)

        for col_offset in range(0, n_cols, BLOCK_SIZE_N):
            cols_off = col_offset + tl.arange(0, BLOCK_SIZE_N)
            cols_mask = cols_off < n_cols
            block_mask = rows_mask[:, None] & cols_mask[None, :]

            S_chunk = tl.load(S_ptr_row_block + cols_off[None, :], mask=block_mask, other=0.0).to(tl.float32)
            W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask, other=0.0)
            B_chunk = tl.load(B_ptr + cols_off, mask=cols_mask, other=0.0)

            normed_S_chunk = (S_chunk - mean_vec[:, None]) * rstd_vec[:, None]
            Y_chunk = normed_S_chunk * W_chunk[None, :] + B_chunk[None, :]
            tl.store(Y_ptr_row_block + cols_off[None, :], Y_chunk.to(Y_ptr.dtype.element_ty), mask=block_mask)


def _stage(message):
    print(f"[E2E_STAGE] op=fused_add_layernorm {message}", flush=True)


def _device():
    return torch.device("npu", torch.npu.current_device())


def _compile_and_launch(x, residual, weight, bias):
    dtype = x.dtype
    ptr = DTYPE_TO_PTR[dtype]
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    s = torch.empty_like(x)
    mean = torch.empty(n_rows, dtype=torch.float32, device=x.device)
    rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)
    source = ASTSource(
        fn=_fused_add_layernorm_fwd_kernel,
        signature={
            "Y_ptr": ptr,
            "Y_row_stride": "i64",
            "S_ptr": ptr,
            "S_row_stride": "i64",
            "X_ptr": ptr,
            "X_row_stride": "i64",
            "R_ptr": ptr,
            "R_row_stride": "i64",
            "W_ptr": ptr,
            "B_ptr": ptr,
            "Mean_ptr": "*fp32",
            "Mean_row_stride": "i64",
            "RSTD_ptr": "*fp32",
            "RSTD_row_stride": "i64",
            "n_rows": "i32",
            "n_cols": "i32",
            "eps": "fp32",
            "BLOCK_SIZE_N": "constexpr",
            "BLOCK_SIZE_M": "constexpr",
        },
        constexprs={"BLOCK_SIZE_N": BLOCK_SIZE_N, "BLOCK_SIZE_M": BLOCK_SIZE_M},
    )
    compiled = triton.compile(source, options={"num_warps": 1, "num_stages": 1})
    num_cores = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
    grid = min(num_cores, (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)
    compiled[(grid, 1, 1)](
        y,
        y.stride(0),
        s,
        s.stride(0),
        x,
        x.stride(0),
        residual,
        residual.stride(0),
        weight,
        bias,
        mean,
        mean.stride(0),
        rstd,
        rstd.stride(0),
        n_rows,
        n_cols,
        EPS,
    )
    return y, s, mean, rstd


def _compare(name, actual, expected, atol, rtol):
    actual_f32 = actual.cpu().float()
    expected_f32 = expected.float()
    diff = (actual_f32 - expected_f32).abs()
    finite = int(torch.isfinite(actual_f32).sum().item())
    total = actual_f32.numel()
    max_abs = float(diff.max().item())
    max_idx = int(diff.argmax().item())
    mean_abs = float(diff.mean().item())
    print(
        f"[E2E_DIFF] op=fused_add_layernorm tensor={name} "
        f"max_diff={max_abs:.8g} max_idx={max_idx} mean_diff={mean_abs:.8g}",
        flush=True,
    )
    torch.testing.assert_close(actual_f32, expected_f32, atol=atol, rtol=rtol)
    print(
        f"[E2E_COMPARE] op=fused_add_layernorm tensor={name} pass=1 "
        f"finite={finite}/{total} max_abs={max_abs:.8g} mean_abs={mean_abs:.8g} "
        f"atol={atol} rtol={rtol}",
        flush=True,
    )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_add_layernorm_opentile(shape, dtype):
    n_rows, n_cols = shape
    atol, rtol = DTYPE_TOLERANCE[dtype]
    dtype_label = DTYPE_TO_LABEL[dtype]
    generator = torch.Generator(device="cpu").manual_seed(43)
    x_cpu = torch.randn(shape, dtype=dtype, generator=generator)
    residual_cpu = torch.randn(shape, dtype=dtype, generator=generator)
    weight_cpu = torch.randn((n_cols,), dtype=dtype, generator=generator)
    bias_cpu = torch.randn((n_cols,), dtype=dtype, generator=generator)

    summed_cpu = x_cpu + residual_cpu
    expected_y = F.layer_norm(summed_cpu, [n_cols], weight=weight_cpu, bias=bias_cpu, eps=EPS)
    expected_s = summed_cpu
    summed_f32 = summed_cpu.float()
    expected_mean = summed_f32.mean(dim=-1)
    expected_var = summed_f32.var(dim=-1, unbiased=False)
    expected_rstd = torch.rsqrt(expected_var + EPS)
    _stage(
        f"input_ready shape={shape} dtype={dtype_label} eps={EPS} add_mode={ADD_MODE} "
        f"seed=43 golden=cpu_torch_layer_norm"
    )

    x = x_cpu.to(_device())
    residual = residual_cpu.to(_device())
    weight = weight_cpu.to(_device())
    bias = bias_cpu.to(_device())
    torch.npu.synchronize()
    _stage("h2d_done")

    actual_y, actual_s, actual_mean, actual_rstd = _compile_and_launch(x, residual, weight, bias)
    torch.npu.synchronize()
    _stage("compile_launch_sync_done")

    _compare("Y", actual_y, expected_y, atol, rtol)
    _compare("S", actual_s, expected_s, atol, rtol)
    _compare("Mean", actual_mean, expected_mean, atol, rtol)
    _compare("RSTD", actual_rstd, expected_rstd, atol, rtol)