"""Upstream-semantics checks for OpenTile ``prepare_wy_repr_bwd``."""

from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
import sys

import pytest


# Compiler/test controls only. Backend and architecture are discovered from
# the visible NPU and active Triton/OpenTile driver.
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
os.environ.setdefault("OPENTILE_KERNEL_MODE", "mix")
os.environ.setdefault("OPENTILE_ENABLE_APPROX", "0")
os.environ.setdefault("OPENTILE_ENABLE_FTZ", "0")

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
import torch.nn.functional as F  # noqa: E402


BT = 64
ERROR_RATIO_LIMIT = 5e-3


def _read_test_seed() -> int:
    raw_seed = os.environ.get("OPENTILE_TEST_SEED", "42")
    try:
        return int(raw_seed)
    except ValueError as error:
        raise ValueError(
            f"OPENTILE_TEST_SEED must be an integer, got {raw_seed!r}"
        ) from error


TEST_SEED = _read_test_seed()
# Keep this matrix identical to the upstream focused operator test. These are
# intentionally ordinary tests, not xfails: unsupported OpenTile paths should
# fail at their real validation, compile, launch, or numerical boundary.
CASES = [
    (2, 128, 2, 2, 64, True, torch.bfloat16),
    (2, 128, 2, 4, 64, True, torch.bfloat16),
    (1, 256, 4, 4, 32, True, torch.float16),
    (2, 128, 2, 2, 64, False, torch.bfloat16),
]
CASE_IDS = [
    f"B{B}-T{T}-H{H}-HV{HV}-D{D}-use_g{use_g}-{dtype}"
    for B, T, H, HV, D, use_g, dtype in CASES
]


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "on", "yes"}


def _trace_phase(name: str, *, use_g: bool) -> None:
    if os.environ.get("WY_BWD_TRACE_KERNELS", "1").lower() in {
        "1",
        "true",
        "on",
        "yes",
    }:
        print(f"[CASE-TRACE] {name} use_g={use_g}", flush=True)


def _npu_available() -> bool:
    try:
        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


if not _npu_available():
    pytest.skip(
        "torch_npu is installed, but no NPU device is visible",
        allow_module_level=True,
    )

device_id = os.environ.get("OPENTILE_TEST_DEVICE")
if device_id is not None:
    torch.npu.set_device(int(device_id))

DEVICE = torch.device("npu", torch.npu.current_device())

from ci_wy_fast import (  # noqa: E402
    assert_opentile_backend,
    get_npu_properties,
    prepare_wy_repr_bwd,
)


_acl = None
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


def _runtime_memcpy_enabled() -> bool:
    return bool(os.environ.get("DAV3510_SIM_RUN_DIR")) or _env_flag(
        "OPENTILE_USE_RUNTIME_MEMCPY"
    )


def _runtime_memcpy(dst: torch.Tensor, src: torch.Tensor, kind: int) -> None:
    global _acl
    if dst.shape != src.shape or dst.dtype != src.dtype:
        raise ValueError("runtime memcpy requires identical shape and dtype")
    if not dst.is_contiguous() or not src.is_contiguous():
        raise ValueError("runtime memcpy requires contiguous tensors")
    nbytes = src.numel() * src.element_size()
    if _acl is None:
        _acl = ctypes.CDLL("libascendcl.so")
        _acl.aclrtMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        _acl.aclrtMemcpy.restype = ctypes.c_int
    result = _acl.aclrtMemcpy(
        ctypes.c_void_p(dst.data_ptr()),
        nbytes,
        ctypes.c_void_p(src.data_ptr()),
        nbytes,
        kind,
    )
    if result != 0:
        raise RuntimeError(f"aclrtMemcpy failed with error code {result}")


def _to_npu(value: torch.Tensor) -> torch.Tensor:
    value = value.contiguous()
    if not _runtime_memcpy_enabled():
        return value.to(DEVICE)
    result = torch.empty(value.shape, dtype=value.dtype, device=DEVICE)
    _runtime_memcpy(result, value, _ACL_MEMCPY_HOST_TO_DEVICE)
    return result


def _to_cpu(value: torch.Tensor) -> torch.Tensor:
    if not _runtime_memcpy_enabled():
        return value.cpu()
    torch.npu.synchronize()
    result = torch.empty(value.shape, dtype=value.dtype, device="cpu")
    _runtime_memcpy(result, value, _ACL_MEMCPY_DEVICE_TO_HOST)
    return result


def _assert_device_route() -> None:
    backend = assert_opentile_backend()
    properties = get_npu_properties()
    assert int(properties["num_aicore"]) > 0
    assert int(properties["num_vectorcore"]) > 0
    print(
        f"[ROUTE] backend={backend} device={DEVICE} "
        f"num_aicore={properties['num_aicore']} "
        f"num_vectorcore={properties['num_vectorcore']}",
        flush=True,
    )


def _chunk_kkt_solve(
    k: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor | None,
    chunk_size: int = BT,
) -> torch.Tensor:
    """Construct the true WY inverse in the input key dtype."""
    B, T, H, _ = k.shape
    HV = beta.shape[2]
    if T % chunk_size != 0:
        raise ValueError(f"T={T} must be divisible by chunk_size={chunk_size}")
    if HV % H != 0:
        raise ValueError(f"HV={HV} must be divisible by H={H}")

    head_index = torch.arange(HV, device=k.device) // (HV // H)
    k_by_value_head = k.index_select(2, head_index)
    mask = torch.tril(
        torch.ones(
            chunk_size,
            chunk_size,
            dtype=torch.float32,
            device=k.device,
        ),
        diagonal=-1,
    )
    identity = torch.eye(
        chunk_size,
        dtype=torch.float32,
        device=k.device,
    )
    inverse_chunks = []
    for start in range(0, T, chunk_size):
        stop = start + chunk_size
        k_chunk = (
            k_by_value_head[:, start:stop]
            .float()
            .permute(0, 2, 1, 3)
        )
        kkt = k_chunk @ k_chunk.transpose(-1, -2)
        beta_chunk = (
            beta[:, start:stop].float().permute(0, 2, 1)
        )
        if g is None:
            gate = 1.0
        else:
            g_chunk = g[:, start:stop].float().permute(0, 2, 1)
            difference = (
                g_chunk[:, :, :, None] - g_chunk[:, :, None, :]
            )
            gate = torch.exp2(
                difference.masked_fill(~mask.bool(), 0.0)
            )
        lower = kkt * gate * beta_chunk[:, :, :, None] * mask
        inverse = torch.linalg.inv(identity + lower)
        inverse_chunks.append(
            inverse.permute(0, 2, 1, 3).to(k.dtype)
        )
    return torch.cat(inverse_chunks, dim=1).contiguous()


def _make_inputs(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    use_g: bool,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | None]:
    generator = torch.Generator(device="cpu").manual_seed(TEST_SEED)
    k = torch.randn(
        B, T, H, D, generator=generator, dtype=dtype
    )
    k = F.normalize(k.float(), p=2, dim=-1).to(dtype).contiguous()
    v = torch.randn(
        B, T, HV, D, generator=generator, dtype=dtype
    ).contiguous()
    beta = (
        torch.rand(B, T, HV, generator=generator, dtype=dtype)
        .float()
        .sigmoid()
        .to(dtype)
        .contiguous()
    )
    g = (
        (
            0.1
            * torch.randn(
                B, T, HV, generator=generator, dtype=torch.float32
            )
        ).contiguous()
        if use_g
        else None
    )
    A = _chunk_kkt_solve(k, beta, g)
    dw = torch.randn(
        B, T, HV, D, generator=generator, dtype=dtype
    ).contiguous()
    du = torch.randn(
        B, T, HV, D, generator=generator, dtype=dtype
    ).contiguous()
    return {
        "k": k,
        "v": v,
        "beta": beta,
        "A": A,
        "dw": dw,
        "du": du,
        "g": g,
    }


def _recompute_reference(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor | None,
    chunk_size: int = BT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent PyTorch expression for the WY forward represented by A."""
    _, T, H, _ = k.shape
    HV = v.shape[2]
    if T % chunk_size != 0:
        raise ValueError(f"T={T} must be divisible by chunk_size={chunk_size}")
    if HV % H != 0:
        raise ValueError(f"HV={HV} must be divisible by H={H}")

    head_index = torch.arange(HV, device=k.device) // (HV // H)
    k_by_value_head = k.index_select(2, head_index)
    w_chunks = []
    u_chunks = []
    for start in range(0, T, chunk_size):
        stop = start + chunk_size
        A_chunk = (
            A[:, start:stop].permute(0, 2, 1, 3).float()
        )
        beta_chunk = beta[:, start:stop].float()
        v_scaled = (
            v[:, start:stop].float()
            * beta_chunk[:, :, :, None]
        ).permute(0, 2, 1, 3)
        k_scaled = (
            k_by_value_head[:, start:stop].float()
            * beta_chunk[:, :, :, None]
        )
        if g is not None:
            k_scaled = (
                k_scaled
                * torch.exp2(g[:, start:stop].float())[:, :, :, None]
            )
        k_scaled = k_scaled.permute(0, 2, 1, 3)
        u_chunks.append(
            (A_chunk @ v_scaled).permute(0, 2, 1, 3)
        )
        w_chunks.append(
            (A_chunk @ k_scaled).permute(0, 2, 1, 3)
        )
    return torch.cat(w_chunks, dim=1), torch.cat(u_chunks, dim=1)


def _prepare_wy_repr_bwd_ref(
    inputs: dict[str, torch.Tensor | None],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Match the true-A operator semantics, including the gate adjoint."""
    k = inputs["k"].detach().float().requires_grad_()
    v = inputs["v"].detach().float().requires_grad_()
    beta = inputs["beta"].detach().float().requires_grad_()
    g_input = inputs["g"]
    g = (
        g_input.detach().float().requires_grad_()
        if g_input is not None
        else None
    )
    A = _chunk_kkt_solve(k, beta, g)
    w, u = _recompute_reference(k, v, beta, A, g)
    loss = (inputs["dw"].float() * w).sum()
    loss += (inputs["du"].float() * u).sum()
    variables = [k, v, beta] + ([g] if g is not None else [])
    gradients = torch.autograd.grad(loss, variables)
    dg = gradients[3] / math.log(2.0) if g is not None else None
    return (
        gradients[0].to(inputs["k"].dtype),
        gradients[1].to(inputs["v"].dtype),
        gradients[2],
        dg,
    )


def _assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    actual_cpu = _to_cpu(actual).float()
    expected_cpu = expected.cpu().float()
    assert actual_cpu.shape == expected_cpu.shape
    assert torch.isfinite(actual_cpu).all(), f"{name} contains non-finite values"
    assert torch.isfinite(expected_cpu).all(), (
        f"{name} reference contains non-finite values"
    )
    difference = actual_cpu - expected_cpu
    max_abs = float(difference.abs().max().item())
    max_rel = float(
        (difference.abs() / expected_cpu.abs().clamp_min(1e-12)).max().item()
    )
    rel_rmse = float(
        difference.square().mean().sqrt().item()
        / (expected_cpu.square().mean().sqrt().item() + 1e-8)
    )
    print(
        f"[NUMERICS] {name}: max_abs={max_abs:.8e} "
        f"max_rel={max_rel:.8e} rel_rmse={rel_rmse:.8e} "
        f"limit={ERROR_RATIO_LIMIT:.8e}",
        flush=True,
    )
    assert rel_rmse < ERROR_RATIO_LIMIT


@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "D", "use_g", "dtype"),
    CASES,
    ids=CASE_IDS,
)
def test_prepare_wy_repr_bwd_opentile(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    use_g: bool,
    dtype: torch.dtype,
) -> None:
    """Require the OpenTile result to preserve upstream operator semantics."""
    _assert_device_route()
    inputs = _make_inputs(B, T, H, HV, D, use_g, dtype)
    npu = {
        name: _to_npu(value) if value is not None else None
        for name, value in inputs.items()
    }
    print(
        f"[CASE] seed={TEST_SEED} dtype={dtype} "
        f"shape=B{B}-T{T}-H{H}-HV{HV}-K{D}-V{D}-BT{BT} "
        f"use_g={use_g}",
        flush=True,
    )
    _trace_phase("PRODUCTION_CALL_BEGIN", use_g=use_g)
    dk, dv, db, dg = prepare_wy_repr_bwd(
        k=npu["k"],
        v=npu["v"],
        beta=npu["beta"],
        A=npu["A"],
        dw=npu["dw"],
        du=npu["du"],
        g=npu["g"],
    )
    _trace_phase("PRODUCTION_CALL_END", use_g=use_g)
    _trace_phase("FINAL_SYNC_BEGIN", use_g=use_g)
    torch.npu.synchronize()
    _trace_phase("FINAL_SYNC_END", use_g=use_g)
    _trace_phase("UPSTREAM_REFERENCE_BEGIN", use_g=use_g)
    expected = _prepare_wy_repr_bwd_ref(inputs)
    _trace_phase("UPSTREAM_REFERENCE_END", use_g=use_g)
    assert dk.dtype == dtype
    assert dv.dtype == dtype
    assert db.dtype == dtype
    for name, actual_value, expected_value in zip(
        ("dk", "dv", "db"),
        (dk, dv, db),
        expected[:3],
    ):
        assert actual_value is not None
        _trace_phase(f"COMPARE_{name}_BEGIN", use_g=use_g)
        _assert_close(name, actual_value, expected_value)
        _trace_phase(f"COMPARE_{name}_END", use_g=use_g)

    if use_g:
        assert dg is not None
        assert expected[3] is not None
        assert dg.dtype == torch.float32
        _trace_phase("COMPARE_dg_BEGIN", use_g=use_g)
        _assert_close("dg", dg, expected[3])
        _trace_phase("COMPARE_dg_END", use_g=use_g)
    else:
        assert dg is None
        assert expected[3] is None
    _trace_phase("TEST_END", use_g=use_g)


if __name__ == "__main__":
    # For temporary single-case diagnosis only, uncomment exactly one line:
    # test_prepare_wy_repr_bwd_opentile(
    #     2, 128, 2, 2, 64, True, torch.bfloat16
    # )
    pass
