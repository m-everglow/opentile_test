"""Physical-Ascend numerical validation for the complete ``chunk_fwd_o`` kernel."""

import os

import pytest
import torch
import torch_npu  # noqa: F401  # registers torch.npu and Tensor.npu
import triton.runtime.driver as driver

from chunk_fwd_o import chunk_fwd_kernel_o


B = 1
H = 2
HV = 2
K = 64
V = 64
BT = 64
BK = 128
BV = 128
SCALE = K**-0.5
DTYPE = torch.bfloat16
ATOL = 5e-3
RTOL = 5e-3
TEST_SEED = int(os.environ.get("OPENTILE_TEST_SEED", "42"))
COMPILE_ONLY = (
    os.environ.get("TRITON_COMPILE_ONLY") == "1"
    or os.environ.get("OPENTILE_COMPILE_ONLY") == "1"
)

CASES = [
    pytest.param(128, id="bf16-aligned-t128"),
    pytest.param(65, id="bf16-tail-t65"),
]


def _stage(message):
    print(f"[OPENTILE_E2E] op=chunk_fwd_o {message}", flush=True)


@pytest.fixture(scope="session", autouse=True)
def _assert_opentile_npu_route():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch_npu is imported, but no physical NPU is visible")

    target = driver.active.get_current_target()
    backend = str(target.backend)
    if backend != "opentile" and not backend.startswith("opentile_"):
        raise RuntimeError(f"expected OpenTile target, got {backend!r}")

    active_device = driver.active.get_active_torch_device()
    if active_device.type != "npu":
        raise RuntimeError(f"expected OpenTile to select an NPU, got {active_device}")

    logical_device = torch.npu.current_device()
    properties = driver.active.utils.get_device_properties(logical_device)
    _stage(
        "route_ok "
        f"backend={backend} active_device={active_device} "
        f"logical_device={logical_device} "
        f"num_aicore={properties.get('num_aicore')} "
        f"num_vectorcore={properties.get('num_vectorcore')} "
        f"kernel_mode={os.environ.get('OPENTILE_KERNEL_MODE')}"
    )


def _random_bf16(generator, shape, low, high):
    value = torch.rand(shape, dtype=torch.float32, generator=generator)
    return (value * (high - low) + low).to(DTYPE)


def _make_inputs(T):
    generator = torch.Generator(device="cpu").manual_seed(TEST_SEED)
    chunks = (T + BT - 1) // BT
    q = _random_bf16(generator, (B, T, H, K), -0.25, 0.25)
    k = _random_bf16(generator, (B, T, H, K), -0.25, 0.25)
    v = _random_bf16(generator, (B, T, HV, V), -0.25, 0.25)
    h = _random_bf16(generator, (B * chunks, HV, K, V), -0.10, 0.10)
    g = torch.rand((B, T, HV), dtype=torch.float32, generator=generator)
    g = g * 0.20 - 0.20
    return q, k, v, h, g


def _cpu_reference(q, k, v, h, g):
    T = q.shape[1]
    chunks = (T + BT - 1) // BT
    result = torch.zeros((B, T, HV, V), dtype=torch.float32)
    qf = q.float()
    kf = k.float()
    vf = v.float()
    hf = h.float()

    for batch in range(B):
        for head in range(HV):
            q_head = head // (HV // H)
            for chunk, start in enumerate(range(0, T, BT)):
                stop = min(start + BT, T)
                qb = qf[batch, start:stop, q_head]
                kb = kf[batch, start:stop, q_head]
                vb = vf[batch, start:stop, head]
                gb = g[batch, start:stop, head].float()
                state = hf[batch * chunks + chunk, head]

                state_term = torch.matmul(qb, state)
                attention = torch.matmul(qb, kb.transpose(0, 1))
                attention *= torch.exp2(gb[:, None] - gb[None, :])
                rows = torch.arange(stop - start)[:, None]
                cols = torch.arange(stop - start)[None, :]
                attention = torch.where(rows >= cols, attention, 0.0)

                # The kernel rounds A to BF16 before the second Cube matmul.
                recurrent = torch.matmul(attention.to(DTYPE).float(), vb)
                result[batch, start:stop, head] = (
                    state_term * torch.exp2(gb)[:, None] + recurrent
                ) * SCALE

    return result.to(DTYPE)


def _to_npu(tensor, name):
    try:
        result = tensor.npu(non_blocking=False)
    except Exception as exc:
        raise RuntimeError(
            f"CPU->NPU transfer failed for {name} via Tensor.npu(); "
            "verify torch_npu/CANN and visible-device setup"
        ) from exc
    if result.device.type != "npu":
        raise RuntimeError(f"{name} transfer returned unexpected device {result.device}")
    return result


def _launch(q, k, v, h, g):
    T = q.shape[1]
    output = torch.full_like(v, float("nan"))
    dummy_f32 = torch.zeros((1,), dtype=torch.float32, device=q.device)
    dummy_i64 = torch.zeros((1,), dtype=torch.int64, device=q.device)
    logical_grid = ((V + BV - 1) // BV, (T + BT - 1) // BT, B * HV)
    total_programs = logical_grid[0] * logical_grid[1] * logical_grid[2]
    physical_grid = (total_programs,)

    chunk_fwd_kernel_o[physical_grid](
        q,
        k,
        v,
        h,
        g,
        dummy_f32,
        output,
        dummy_i64,
        dummy_i64,
        SCALE,
        T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        USE_G=True,
        USE_G_GAMMA=False,
        STATE_V_FIRST=False,
        IS_VARLEN=False,
        num_warps=8,
        num_stages=3,
    )
    _stage(
        f"launch_submitted T={T} physical_grid={physical_grid} "
        f"logical_grid={logical_grid} programs={total_programs}"
    )
    return output


def _compare(actual, expected):
    actual_f32 = actual.cpu().float()
    expected_f32 = expected.float()
    if actual_f32.shape != expected_f32.shape:
        raise AssertionError(
            f"shape mismatch: actual={actual_f32.shape}, expected={expected_f32.shape}"
        )
    if not torch.isfinite(actual_f32).all():
        non_finite = int((~torch.isfinite(actual_f32)).sum().item())
        tile_coverage = []
        for head in range(HV):
            for start in range(0, actual_f32.shape[1], BT):
                tile = actual_f32[:, start : start + BT, head, :]
                finite = int(torch.isfinite(tile).sum().item())
                tile_coverage.append(f"h{head}:t{start}={finite}/{tile.numel()}")
        raise AssertionError(
            f"OpenTile output has {non_finite} non-finite/unwritten elements; "
            f"tile_coverage={','.join(tile_coverage)}"
        )

    diff = (actual_f32 - expected_f32).abs()
    threshold = ATOL + RTOL * expected_f32.abs()
    mismatch_count = int((diff > threshold).sum().item())
    relative = diff / expected_f32.abs().clamp_min(1e-8)
    _stage(
        "compare "
        f"elements={actual_f32.numel()} mismatches={mismatch_count} "
        f"max_abs={float(diff.max().item()):.8g} "
        f"mean_abs={float(diff.mean().item()):.8g} "
        f"max_rel={float(relative.max().item()):.8g} "
        f"atol={ATOL} rtol={RTOL}"
    )
    torch.testing.assert_close(
        actual_f32,
        expected_f32,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("T", CASES)
def test_chunk_fwd_o_opentile(T):
    q_cpu, k_cpu, v_cpu, h_cpu, g_cpu = _make_inputs(T)
    expected = _cpu_reference(q_cpu, k_cpu, v_cpu, h_cpu, g_cpu)
    _stage(
        f"input_ready seed={TEST_SEED} dtype=bf16 "
        f"B={B} T={T} H={H} HV={HV} K={K} V={V}"
    )

    q = _to_npu(q_cpu, "q")
    k = _to_npu(k_cpu, "k")
    v = _to_npu(v_cpu, "v")
    h = _to_npu(h_cpu, "h")
    g = _to_npu(g_cpu, "g")
    torch.npu.synchronize()
    _stage("h2d_done")

    actual = _launch(q, k, v, h, g)
    if COMPILE_ONLY:
        _stage("compile_only_complete; execution and numerical comparison skipped")
        return

    torch.npu.synchronize()
    _stage("launch_sync_done")
    _compare(actual, expected)


if __name__ == "__main__":
    # test_chunk_fwd_o_opentile(128)
    # test_chunk_fwd_o_opentile(65)
    pass
