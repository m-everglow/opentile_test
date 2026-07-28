# SPDX-License-Identifier: MIT
"""OpenTile/NPU solve_tril regression derived from FLA test_solve_tril.

The Gram-matrix/reference construction follows flash-linear-attention's
``tests/ops/test_solve_tril.py::test_solve_tril``.  On the ACLNN-only target,
``torch.rand`` replaces unsupported NPU ``randn/normal_`` while preserving the
same normalized Gram input contract.  The launch grid is flattened explicitly
because Ascend exposes a one-dimensional hardware block ID.
"""

from __future__ import annotations

import os

# Match the bootstrap used by the existing, working OpenTile E2E cases.
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_BACKEND", "opentile")
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.compiler import ASTSource

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


@triton.jit
def merge_16x16_to_32x32_inverse_kernel(
    A,
    Ai,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
):
    pid = tl.program_id(0)
    i_t = pid // BH
    i_bh = pid % BH
    i_b = i_bh // H
    i_h = i_bh % H

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    bos = i_b * T
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    p_A_11 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_A_22 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    b_Ai_11 = -tl.where(m_A, tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32), 0.0)
    b_Ai_22 = -tl.where(m_A, tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32), 0.0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H * BT + o_i)
        b_a_11 = tl.where(o_i < i, b_a_11, 0.0)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(18, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)
        b_a_22 = tl.where(o_i < i - 16, b_a_22, 0.0)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    b_Ai_11 += m_I
    b_Ai_22 += m_I

    p_A_21 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_21 = -tl.dot(
        tl.dot(b_Ai_22, b_A_21, input_precision="ieee"),
        b_Ai_11,
        input_precision="ieee",
    )

    p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    tl.store(p_Ai_11, b_Ai_11, boundary_check=(0, 1))
    tl.store(p_Ai_21, b_Ai_21, boundary_check=(0, 1))
    tl.store(p_Ai_22, b_Ai_22, boundary_check=(0, 1))


@triton.jit
def merge_16x16_to_64x64_inverse_kernel(
    A,
    Ai,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
):
    pid = tl.program_id(0)
    i_t = pid // BH
    i_bh = pid % BH
    i_b = i_bh // H
    i_h = i_bh % H

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    bos = i_b * T
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    p_A_11 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_A_22 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    p_A_33 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
    p_A_44 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
    b_Ai_11 = -tl.where(m_A, tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32), 0.0)
    b_Ai_22 = -tl.where(m_A, tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32), 0.0)
    b_Ai_33 = -tl.where(m_A, tl.load(p_A_33, boundary_check=(0, 1)).to(tl.float32), 0.0)
    b_Ai_44 = -tl.where(m_A, tl.load(p_A_44, boundary_check=(0, 1)).to(tl.float32), 0.0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H * BT + o_i)
        b_a_11 = tl.where(o_i < i, b_a_11, 0.0)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(18, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)
        b_a_22 = tl.where(o_i < i - 16, b_a_22, 0.0)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(34, min(48, T - i_t * BT)):
        b_a_33 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 32)
        b_a_33 = tl.where(o_i < i - 32, b_a_33, 0.0)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(50, min(64, T - i_t * BT)):
        b_a_44 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 48)
        b_a_44 = tl.where(o_i < i - 48, b_a_44, 0.0)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    p_A_21 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_A_31 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
    p_A_32 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
    p_A_41 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
    p_A_42 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
    p_A_43 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
    b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
    b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
    b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
    b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
    b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)

    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, input_precision="ieee"), b_Ai_11, input_precision="ieee")
    b_Ai_32 = -tl.dot(tl.dot(b_Ai_33, b_A_32, input_precision="ieee"), b_Ai_22, input_precision="ieee")
    b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43, input_precision="ieee"), b_Ai_33, input_precision="ieee")
    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11, input_precision="ieee")
        + tl.dot(b_A_32, b_Ai_21, input_precision="ieee"),
        input_precision="ieee",
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22, input_precision="ieee")
        + tl.dot(b_A_43, b_Ai_32, input_precision="ieee"),
        input_precision="ieee",
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11, input_precision="ieee")
        + tl.dot(b_A_42, b_Ai_21, input_precision="ieee")
        + tl.dot(b_A_43, b_Ai_31, input_precision="ieee"),
        input_precision="ieee",
    )

    p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
    p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
    p_Ai_31 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
    p_Ai_32 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
    p_Ai_33 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
    p_Ai_41 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
    p_Ai_42 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
    p_Ai_43 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
    p_Ai_44 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
    tl.store(p_Ai_11, b_Ai_11, boundary_check=(0, 1))
    tl.store(p_Ai_21, b_Ai_21, boundary_check=(0, 1))
    tl.store(p_Ai_22, b_Ai_22, boundary_check=(0, 1))
    tl.store(p_Ai_31, b_Ai_31, boundary_check=(0, 1))
    tl.store(p_Ai_32, b_Ai_32, boundary_check=(0, 1))
    tl.store(p_Ai_33, b_Ai_33, boundary_check=(0, 1))
    tl.store(p_Ai_41, b_Ai_41, boundary_check=(0, 1))
    tl.store(p_Ai_42, b_Ai_42, boundary_check=(0, 1))
    tl.store(p_Ai_43, b_Ai_43, boundary_check=(0, 1))
    tl.store(p_Ai_44, b_Ai_44, boundary_check=(0, 1))


def solve_tril(A: torch.Tensor) -> torch.Tensor:
    B, T, H, BT = A.shape
    assert BT in (32, 64)
    NT = triton.cdiv(T, BT)
    Ai = torch.zeros_like(A, dtype=torch.float32)
    kernel = merge_16x16_to_32x32_inverse_kernel if BT == 32 else merge_16x16_to_64x64_inverse_kernel

    # Match the working GELU E2E launcher: compile an explicit ASTSource and
    # invoke the compiled kernel with an explicit 3-D runtime grid.
    source = ASTSource(
        fn=kernel,
        signature={
            "A": "*fp32",
            "Ai": "*fp32",
            "T": "i32",
            "H": "constexpr",
            "BT": "constexpr",
            "BH": "constexpr",
        },
        constexprs={"H": H, "BT": BT, "BH": B * H},
    )
    compiled = triton.compile(source, options={"num_warps": 1, "num_stages": 2})
    # Ascend exposes one physical block ID. Flatten (NT, B*H) explicitly.
    compiled[(NT * B * H, 1, 1)](A, Ai, T)
    return Ai


def _test_device() -> torch.device | str:
    if os.getenv("OPENTILE_COMPILE_ONLY") == "1":
        return "npu"
    if torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("torch_npu is unavailable or no NPU device is visible")
    device_id = int(os.environ.get("OPENTILE_TEST_DEVICE", "0"))
    torch.npu.set_device(device_id)
    return torch.device("npu", device_id)


def _sync(tag: str) -> None:
    print(f"[SYNC] {tag}", flush=True)
    torch.npu.synchronize()
    print(f"[SYNC OK] {tag}", flush=True)


@pytest.mark.parametrize(
    ("B", "T", "H", "BT"),
    [
        pytest.param(1, 32, 1, 32, id="single-tile-bt32"),
        pytest.param(1, 64, 1, 64, id="single-tile-bt64"),
        pytest.param(2, 500, 4, 32, id="fla-original-scale-bt32"),
        pytest.param(2, 1000, 5, 64, id="fla-original-scale-bt64"),
    ],
)
def test_solve_tril(B: int, T: int, H: int, BT: int) -> None:
    print("[STEP] 1: _test_device BEGIN", flush=True)
    device = _test_device()
    print(f"[STEP] 1: _test_device PASS device={device}", flush=True)

    print("[STEP] 2: torch.manual_seed BEGIN", flush=True)
    torch.manual_seed(20260723 + BT)
    print("[STEP] 2: torch.manual_seed PASS", flush=True)

    print("[STEP] 3: torch.rand 2-D BF16 BEGIN", flush=True)
    k_seed = torch.rand((B * H * T, 64), dtype=torch.bfloat16, device=device)
    print(f"[STEP] 3: torch.rand 2-D BF16 PASS shape={k_seed.shape} dtype={k_seed.dtype}", flush=True)
    _sync("after bf16 2-D rand")

    print("[STEP] 4: reshape cast normalize BEGIN", flush=True)
    k = F.normalize(
        k_seed.reshape(B, H, T, 64).to(torch.float32) * 2.0 - 1.0,
        dim=-1,
    )
    print(f"[STEP] 4: reshape cast normalize PASS shape={k.shape}", flush=True)
    _sync("after reshape cast normalize")

    print("[STEP] 5: pad matmul tril BEGIN", flush=True)
    padding = (BT - T % BT) % BT
    k_padded = F.pad(k, (0, 0, 0, padding))
    k_padded = k_padded.reshape(B, H, -1, BT, 64)
    A_bhnt = (k_padded @ k_padded.transpose(-1, -2)).tril(-1)
    print(f"[STEP] 5: pad matmul tril PASS shape={A_bhnt.shape}", flush=True)
    _sync("after input construction")

    print("[STEP] 6: torch.inverse reference BEGIN", flush=True)
    ref = torch.inverse(
        A_bhnt + torch.eye(BT, dtype=torch.float32, device=device)[None, None, None]
    )
    ref = ref.reshape(B, H, -1, BT)[:, :, :T, :].contiguous()
    A = A_bhnt.reshape(B, H, -1, BT)[:, :, :T, :].transpose(1, 2).contiguous()
    A_before = A.clone()
    print(f"[STEP] 6: torch.inverse reference PASS shape={ref.shape}", flush=True)
    _sync("after torch reference")

    print(f"[STEP] 7: solve_tril BEGIN triton={triton.__file__}", flush=True)
    actual = solve_tril(A)
    print(f"[STEP] 7: solve_tril PASS shape={actual.shape}", flush=True)
    _sync("after opentile solve_tril")

    if os.getenv("OPENTILE_COMPILE_ONLY") == "1":
        print(f"COMPILE_ONLY_PASS B={B} T={T} H={H} BT={BT}")
        return

    actual_cpu = actual.cpu().transpose(1, 2).contiguous()
    ref_cpu = ref.cpu()
    input_after_cpu = A.cpu()
    input_before_cpu = A_before.cpu()
    assert torch.equal(input_after_cpu, input_before_cpu), "input tensor was modified"
    assert torch.isfinite(actual_cpu).all(), "output contains NaN/Inf"
    torch.testing.assert_close(actual_cpu, ref_cpu, atol=1.0e-4, rtol=1.0e-4)

    abs_err = (actual_cpu - ref_cpu).abs()
    close = torch.isclose(actual_cpu, ref_cpu, atol=1.0e-4, rtol=1.0e-4)
    print(
        f"PRECISION_PASS B={B} T={T} H={H} BT={BT} "
        f"mismatch={(~close).sum().item()} max_abs={abs_err.max().item():.9g}"
    )
