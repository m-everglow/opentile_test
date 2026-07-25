"""End-to-end OpenTile test for fused_matmul_bwd_x_kernel.

Source of truth:
  /Users/lingli/Downloads/fused_matmul_npu_v3.py
  SHA256 ec5dc1d74ed5c54459f8b70292e2847a28dc6e37bd202e30639151acc3d140fe

The Triton kernel body below is copied from that file without algorithmic
changes.  The test keeps its original FP16 example dimensions:
dy [256, 1024], w [512, 1024] -> dx [256, 512].  The fixed launch uses
BM=128/BN=128/BK=64, one of the source bwd_x_autotune_config() candidates.
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

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


M = 256
N = 512
K = 1024
_SHAPES = [
    (256, 512, 1024),
    (512, 1024, 2048),
]
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 64
GROUP_SIZE_M = 8
NUM_WARPS = 4
NUM_STAGES = 3
SEED = 0

ATOL = 1e-3
RTOL = 1e-2


@triton.jit
def fused_matmul_bwd_x_kernel(
        dy_ptr,
        w_ptr,
        dx_ptr,
        M,
        N,
        K,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
):
    dtype = dx_ptr.type.element_ty
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    if (pid_m * BLOCK_SIZE_M >= M) or (pid_n * BLOCK_SIZE_N >= N):
        return

    offset_dym = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offset_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    accumulator_dx = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offset_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        dy_ptrs = dy_ptr + (offset_dym[:, None] * K + offset_k[None, :])
        dy_mask = (offset_dym[:, None] < M) & (offset_k[None, :] < K)

        w_ptrs = w_ptr + (offset_k[:, None] + offset_wn[None, :] * K)
        w_mask = (offset_k[:, None] < K) & (offset_wn[None, :] < N)

        dy = tl.load(dy_ptrs, mask=dy_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        # bwd x
        accumulator_dx = tl.dot(dy, w, accumulator_dx)
    dx = accumulator_dx.to(dtype)

    offset_dxm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_dxn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dx_ptrs = dx_ptr + offset_dxm[:, None] * N + offset_dxn[None, :]
    dx_mask = (offset_dxm[:, None] < M) & (offset_dxn[None, :] < N)
    tl.store(dx_ptrs, dx, mask=dx_mask)


def _device():
    return torch.device("npu", torch.npu.current_device())


def _stage(message):
    print(f"[E2E_STAGE] op=fused_matmul_bwd_x {message}", flush=True)


def _compare(actual, expected):
    actual_f32 = actual.cpu().float()
    expected_f32 = expected.float()
    diff = (actual_f32 - expected_f32).abs()
    finite = int(torch.isfinite(actual_f32).sum().item())
    total = actual_f32.numel()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    torch.testing.assert_close(actual_f32, expected_f32, atol=ATOL, rtol=RTOL)
    print(
        f"[E2E_COMPARE] op=fused_matmul_bwd_x tensor=dx pass=1 "
        f"finite={finite}/{total} max_abs={max_abs:.8g} mean_abs={mean_abs:.8g} "
        f"atol={ATOL} rtol={RTOL}",
        flush=True,
    )


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_matmul_bwd_x_opentile(shape, dtype):
    M, N, K = shape
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    dy_cpu = torch.randn((M, K), dtype=dtype, generator=generator)
    w_cpu = torch.randn((N, K), dtype=dtype, generator=generator)
    # dx = dy @ w.T.  Use FP32 CPU accumulation, then match the store dtype.
    expected_dx = torch.matmul(dy_cpu.float(), w_cpu.float().transpose(0, 1)).to(dtype)
    dtype_name = {torch.float16: "fp16", torch.bfloat16: "bf16"}[dtype]
    _stage(
        f"input_ready dy_shape=({M},{K}) w_shape=({N},{K}) dtype={dtype_name} "
        f"seed={SEED} golden=cpu_fp32_matmul_transpose_cast_{dtype_name}"
    )

    dy = dy_cpu.to(_device())
    w = w_cpu.to(_device())
    # NaN sentinel makes any unwritten output element fail the finite check.
    dx = torch.full((M, N), float("nan"), dtype=dtype).to(_device())
    torch.npu.synchronize()
    _stage("h2d_done")

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
    )
    fused_matmul_bwd_x_kernel[grid](
        dy,
        w,
        dx,
        M,
        N,
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    torch.npu.synchronize()
    _stage(
        f"compile_launch_sync_done grid={grid} BM={BLOCK_SIZE_M} "
        f"BN={BLOCK_SIZE_N} BK={BLOCK_SIZE_K}"
    )

    _compare(dx, expected_dx)