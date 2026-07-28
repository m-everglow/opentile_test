"""End-to-end OpenTile test for fused_matmul_bwd_b_kernel.

Source of truth:
  /Users/lingli/Downloads/fused_matmul_npu_v3.py
  SHA256 ec5dc1d74ed5c54459f8b70292e2847a28dc6e37bd202e30639151acc3d140fe

The Triton kernel body below is copied from that file without algorithmic
changes.  The test keeps its original FP16 example dimensions:
dy [256, 1024] -> db [1024].  The fixed BM=512/BN=32 launch configuration is
the sole configuration in the source kernel's bwd_b_autotune_config().
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
N = 1024
DTYPE = torch.float16
BLOCK_SIZE_M = 512
BLOCK_SIZE_N = 32
NUM_WARPS = 4
NUM_STAGES = 3
SEED = 0
ATOL = 1e-3
RTOL = 1e-2


@triton.jit
def fused_matmul_bwd_b_kernel(
        dy_ptr,
        db_ptr,
        M,
        N,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
):
    col_idx = tl.program_id(axis=0)
    col_off = col_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_off = tl.arange(0, BLOCK_SIZE_M)
    dy_ptrs = dy_ptr + (row_off[None, :] * N + col_off[:, None])
    sum_b = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    for row_idx in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        mask = (row_off[None, :] < M - row_idx * BLOCK_SIZE_M) & (col_off[:, None] < N)
        dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
        sum_b += dy
        dy_ptrs += BLOCK_SIZE_M * N
    tl.store(db_ptr + col_off, tl.sum(sum_b, 1), mask=col_off < N)


def _device():
    return torch.device("npu", torch.npu.current_device())


def _stage(message):
    print(f"[E2E_STAGE] op=fused_matmul_bwd_b {message}", flush=True)


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
        f"[E2E_COMPARE] op=fused_matmul_bwd_b tensor=db pass=1 "
        f"finite={finite}/{total} max_abs={max_abs:.8g} mean_abs={mean_abs:.8g} "
        f"atol={ATOL} rtol={RTOL}",
        flush=True,
    )


def test_fused_matmul_bwd_b_opentile():
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    dy_cpu = torch.randn((M, N), dtype=DTYPE, generator=generator)
    # The production kernel accumulates FP16 input in FP32 and stores FP16.
    expected_db = dy_cpu.float().sum(dim=0).to(DTYPE)
    _stage(
        f"input_ready dy_shape=({M},{N}) dtype=fp16 seed={SEED} "
        "golden=cpu_fp32_sum_cast_fp16"
    )

    dy = dy_cpu.to(_device())
    # NaN sentinel makes any unwritten output element fail the finite check.
    db = torch.full((N,), float("nan"), dtype=DTYPE).to(_device())
    torch.npu.synchronize()
    _stage("h2d_done")

    grid = (triton.cdiv(N, BLOCK_SIZE_N),)
    fused_matmul_bwd_b_kernel[grid](
        dy,
        db,
        M,
        N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    torch.npu.synchronize()
    _stage(
        f"compile_launch_sync_done grid={grid} BM={BLOCK_SIZE_M} BN={BLOCK_SIZE_N}"
    )

    _compare(db, expected_db)
