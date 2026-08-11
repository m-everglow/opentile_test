"""Daily OpenTile E2E regression for ``fused_matmul_bwd_w_kernel``.

The kernel body is copied verbatim from:

  HighPriority50Operators/testcase/22_fused_matmul_bwd_w_kernel/
  fused_matmul_npu_v3.py

Source SHA256:
  ec5dc1d74ed5c54459f8b70292e2847a28dc6e37bd202e30639151acc3d140fe

The source's BM=128/BN=128/BK=128/SPLIT_K=8/stages=3 autotune
candidate is fixed here so daily results do not depend on autotune selection.
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
    pytest.skip(
        "torch_npu is available, but no NPU device is visible",
        allow_module_level=True,
    )

_DEVICE_INDEX = int(
    os.environ.get("OPENTILE_TEST_DEVICE", "0").rsplit(":", 1)[-1]
)
torch.npu.set_device(_DEVICE_INDEX)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


DTYPE = torch.float16
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 128
GROUP_SIZE_M = 8
SPLIT_K = 8
NUM_STAGES = 3
ATOL = 1e-3
RTOL = 1e-2

# The source case remains the primary gate.  Two compact cases add one- and
# three-iteration K loops plus the two extreme grid orientations without
# repeating another source-sized output tensor.
CASES = (
    pytest.param(512, 1024, 256, 0, id="source_m512_n1024_k256_grid4x8"),
    pytest.param(128, 1024, 128, 2202, id="compact_m128_n1024_k128_grid1x8"),
    pytest.param(1024, 128, 384, 2203, id="compact_m1024_n128_k384_grid8x1"),
)


@triton.jit
def fused_matmul_bwd_w_kernel(
        dy_ptr,
        x_ptr,
        dw_ptr,
        LOCK_W,
        M,
        N,
        K,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        SPLIT_K: tl.constexpr,
):
    dtype = dw_ptr.type.element_ty
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    if ((pid_m * BLOCK_SIZE_M >= M) or (pid_n * BLOCK_SIZE_N >= N)):
        return

    offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offset_dyn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    accumulator_dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for pid_k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offset_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        x_ptrs = x_ptr + (offset_xm[:, None] + offset_k[None, :] * M)
        dy_ptrs = dy_ptr + (offset_k[:, None] * N + offset_dyn[None, :])

        x_mask = (offset_xm[:, None] < M) & (offset_k[None, :] < K)
        dy_mask = (offset_k[:, None] < K) & (offset_dyn[None, :] < N)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        dy = tl.load(dy_ptrs, mask=dy_mask, other=0.0)
        # bwd w
        accumulator_dw = tl.dot(x, dy, accumulator_dw)
    dw = accumulator_dw.to(dtype)

    offset_dwm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_dwn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw_ptrs = dw_ptr + offset_dwm[:, None] * N + offset_dwn[None, :]
    dw_mask = (offset_dwm[:, None] < M) & (offset_dwn[None, :] < N)
    tl.store(dw_ptrs, dw, mask=dw_mask)


def _device():
    return torch.device("npu", torch.npu.current_device())


@pytest.mark.parametrize("m,n,k,seed", CASES)
def test_fused_matmul_bwd_w_opentile(m, n, k, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x_cpu = torch.randn((k, m), dtype=DTYPE, generator=generator)
    dy_cpu = torch.randn((k, n), dtype=DTYPE, generator=generator)
    expected = (x_cpu.float().transpose(0, 1) @ dy_cpu.float()).to(DTYPE)

    device = _device()
    x = x_cpu.to(device)
    dy = dy_cpu.to(device)
    dw = torch.full((m, n), float("nan"), dtype=DTYPE).to(device)
    lock_w = torch.zeros(32 * 1024, dtype=torch.int32).to(device)
    torch.npu.synchronize()

    grid = (
        triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n, BLOCK_SIZE_N),
    )
    fused_matmul_bwd_w_kernel[grid](
        dy,
        x,
        dw,
        lock_w,
        m,
        n,
        k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        SPLIT_K=SPLIT_K,
        num_stages=NUM_STAGES,
    )
    torch.npu.synchronize()

    actual = dw.cpu()
    finite = torch.isfinite(actual)
    assert bool(finite.all()), (
        f"non-finite output: {actual.numel() - int(finite.sum().item())}/"
        f"{actual.numel()}"
    )
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=ATOL,
        rtol=RTOL,
        equal_nan=False,
    )

    diff = (actual.float() - expected.float()).abs()
    print(
        "[E2E_COMPARE] op=fused_matmul_bwd_w tensor=dw "
        f"shape=({m},{n}) k={k} grid={grid[0]} "
        f"pass=1 finite={actual.numel()}/{actual.numel()} "
        f"max_abs={float(diff.max().item()):.8g} "
        f"mean_abs={float(diff.mean().item()):.8g} "
        f"atol={ATOL} rtol={RTOL}",
        flush=True,
    )
