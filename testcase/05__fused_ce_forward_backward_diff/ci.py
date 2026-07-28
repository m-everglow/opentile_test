"""Standalone NPU cross-entropy-prime kernel + packaged-case golden.

This intentionally matches the current real-hardware package scope:
  * input to the Triton kernel is FP32 logits materialized from BF16 linear;
  * outputs are per-row mean loss and pre-target-correction grad-logits;
  * the outer linear backward and target-class correction are not included.

Run directly:
  python3 test_fused_ce_fwd_bwd_diff_standalone.py

Run with pytest:
  pytest -s test_fused_ce_fwd_bwd_diff_standalone.py
"""

from __future__ import annotations

import os

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_BACKEND", "opentile")
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")

import torch
import torch.nn.functional as torch_f
import triton
import triton.language as tl

SEED = 42
BATCH = 2048
HIDDEN_SIZE = 1024
TARGET_SIZE = 4096
IGNORE_INDEX = -100
ATOL = 1e-5
RTOL = 1e-5

@triton.jit
def _cross_entropy_prime_kernel(
    X_ptr,
    Y_ptr,
    loss_ptr,
    n_rows,
    n_cols,
    X_stride_row,
    Y_stride_row,
    loss_stride_row,
    n_non_ignore,
    ignore_index,
    lse_square_scale: tl.constexpr,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,
    OVERWRITE_GRAD_LOGITS: tl.constexpr,
    IS_BACKWARD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for row_idx in range(pid, n_rows, num_programs):
        current_X_ptr = X_ptr + row_idx * X_stride_row
        current_Y_ptr = Y_ptr + row_idx * Y_stride_row
        y = tl.load(current_Y_ptr)

        m = float("-inf")
        d = 0.0
        ori_X_y = 0.0
        if y != ignore_index:
            ori_X_y = tl.load(current_X_ptr + y).cast(tl.float32)

        scaled_x_sum = 0.0
        eps = label_smoothing / n_cols

        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            X_mask = X_offsets < n_cols
            X_block = tl.load(
                current_X_ptr + X_offsets,
                mask=X_mask,
                other=float("-inf"),
            ).cast(tl.float32)
            block_max = tl.max(X_block, axis=0)
            if label_smoothing > 0:
                X_block2 = tl.load(
                    current_X_ptr + X_offsets,
                    mask=X_mask,
                    other=0.0,
                ).cast(tl.float32)
                scaled_x_sum += tl.sum(-eps * X_block2, axis=0).to(tl.float32)
            m_new = tl.maximum(m, block_max)
            d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new), axis=0)
            m = m_new

        lse = m + tl.log(d)

        if OVERWRITE_GRAD_LOGITS:
            for i in range(0, n_cols, BLOCK_SIZE):
                X_offsets = i + tl.arange(0, BLOCK_SIZE)
                X_mask = X_offsets < n_cols
                X_block = tl.load(
                    current_X_ptr + X_offsets,
                    mask=X_mask,
                    other=float("-inf"),
                ).cast(tl.float32)
                X_block = tl.exp(X_block - m) / d
                X_block += 2 * lse_square_scale * lse * X_block
                X_block += -eps
                if reduction == "mean":
                    X_block = X_block / n_non_ignore
                tl.store(current_X_ptr + X_offsets, X_block, mask=X_mask)

        if not IS_BACKWARD:
            current_loss_ptr = loss_ptr + row_idx * loss_stride_row
            loss = lse - ori_X_y
            if label_smoothing > 0:
                smooth_loss = scaled_x_sum + label_smoothing * lse
                loss = loss * (1 - label_smoothing) + smooth_loss
            z_loss = lse_square_scale * lse * lse
            if reduction == "mean":
                loss = loss / n_non_ignore
                z_loss = z_loss / n_non_ignore
            loss += z_loss
            tl.store(current_loss_ptr, loss)

def _num_vector_cores():
    return triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]

def _setup_npu() -> torch.device:
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("torch_npu is required on the real-hardware host") from exc
    device_id = int(os.environ.get("OPENTILE_TEST_DEVICE", "0"))
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is false")
    torch.npu.set_device(device_id)
    return torch.device(f"npu:{device_id}")


def _compare(name: str, actual: torch.Tensor, golden: torch.Tensor) -> None:
    actual_cpu = actual.detach().cpu().float()
    golden_cpu = golden.detach().cpu().float()
    absolute = (actual_cpu - golden_cpu).abs()
    tolerance = ATOL + RTOL * golden_cpu.abs()
    passed = absolute <= tolerance
    print(
        f"{name} shape={tuple(actual.shape)} "
        f"pass={int(passed.sum())}/{passed.numel()} bad={int((~passed).sum())} "
        f"max_abs={float(absolute.max()):.9g} mean_abs={float(absolute.mean()):.9g} "
        f"atol={ATOL} rtol={RTOL}"
    )
    torch.testing.assert_close(actual_cpu, golden_cpu, atol=ATOL, rtol=RTOL)


def run_fused_ce_case() -> None:
    device = _setup_npu()

    # Generate on CPU exactly like the current packaged case.
    torch.manual_seed(SEED)
    input_tensor = torch.randn(
        BATCH,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device="cpu",
        requires_grad=True,
    )
    weight = torch.randn(
        TARGET_SIZE,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device="cpu",
        requires_grad=True,
    )
    target_cpu = torch.randint(
        0,
        TARGET_SIZE,
        (BATCH,),
        dtype=torch.long,
        device="cpu",
    )

    with torch.no_grad():
        logits_cpu = torch_f.linear(input_tensor, weight, None).float().contiguous()
        loss_per_row_golden = (
            torch_f.cross_entropy(logits_cpu, target_cpu, reduction="none") / BATCH
        )
        grad_logits_golden = torch_f.softmax(logits_cpu, dim=-1) / BATCH
        scalar_loss_golden = torch_f.cross_entropy(
            logits_cpu, target_cpu, reduction="mean"
        )

    # X is inout: the kernel overwrites logits with pre-target-correction grad.
    logits = logits_cpu.to(device)
    target = target_cpu.to(device)
    loss_per_row = torch.empty(BATCH, dtype=torch.float32, device=device)
    block_dim = _num_vector_cores()
    grid = (block_dim,)

    _cross_entropy_prime_kernel[grid](
        X_ptr=logits,
        Y_ptr=target,
        loss_ptr=loss_per_row,
        n_rows=BATCH,
        n_cols=TARGET_SIZE,
        X_stride_row=logits.stride(0),
        Y_stride_row=target.stride(0),
        loss_stride_row=loss_per_row.stride(0),
        n_non_ignore=BATCH,
        ignore_index=IGNORE_INDEX,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="mean",
        OVERWRITE_GRAD_LOGITS=True,
        IS_BACKWARD=False,
        BLOCK_SIZE=4096,
    )
    torch.npu.synchronize()

    _compare("grad_logits_before_target_correction", logits, grad_logits_golden)
    _compare("loss_per_row", loss_per_row, loss_per_row_golden)
    actual_sum = loss_per_row.detach().cpu().float().sum()
    torch.testing.assert_close(
        actual_sum, scalar_loss_golden.float(), atol=ATOL, rtol=RTOL
    )
    print(
        f"loss_per_row_sum actual={float(actual_sum):.9g} "
        f"golden={float(scalar_loss_golden):.9g}"
    )

def test_fused_ce_fwd_bwd_diff_standalone() -> None:
    run_fused_ce_case()

def main() -> None:
    run_fused_ce_case()
    print("RESULT: PASS")

if __name__ == "__main__":
    main()
