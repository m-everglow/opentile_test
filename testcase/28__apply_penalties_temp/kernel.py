from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_penalty_temp_kernel(
    logits_ptr,
    freqs_ptr,
    is_present_ptr,
    freq_pen_ptr,
    pres_pen_ptr,
    rep_pen_ptr,
    temp_ptr,
    stride_logits_b,
    stride_logits_v,
    stride_freqs_b,
    stride_freqs_v,
    stride_is_present,
    stride_freq_pen,
    stride_pres_pen,
    stride_rep_pen,
    stride_temp,
    n_batch,
    n_vocab,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    num_vocab_blocks = tl.cdiv(n_vocab, BLOCK_SIZE)
    total_tasks = n_batch * num_vocab_blocks

    for task_id in range(pid, total_tasks, grid_size):
        pid_b = task_id // num_vocab_blocks
        pid_v = task_id % num_vocab_blocks

        is_present_float = tl.load(
            is_present_ptr + pid_b * stride_is_present
        )
        freq_pen = tl.load(freq_pen_ptr + pid_b * stride_freq_pen)
        pres_pen = tl.load(pres_pen_ptr + pid_b * stride_pres_pen)
        rep_pen = tl.load(rep_pen_ptr + pid_b * stride_rep_pen)

        temperature = 1.0
        if temp_ptr is not None:
            temperature = tl.load(temp_ptr + pid_b * stride_temp)

        offs_v = pid_v * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs_v < n_vocab
        logit_ptrs = (
            logits_ptr
            + pid_b * stride_logits_b
            + offs_v * stride_logits_v
        )
        freq_ptrs = (
            freqs_ptr
            + pid_b * stride_freqs_b
            + offs_v * stride_freqs_v
        )

        logits = tl.load(logit_ptrs, mask=mask, other=0.0).to(tl.float32)
        token_freqs = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        if is_present_float != 0.0:
            token_freqs = tl.load(freq_ptrs, mask=mask, other=0.0).to(
                tl.float32
            )
            if freq_pen != 0.0:
                logits = logits - freq_pen * token_freqs
            if pres_pen != 0.0:
                is_present = token_freqs > 0
                logits = logits - pres_pen * is_present.to(tl.float32)
            if rep_pen != 1.0:
                has_freq = token_freqs > 0
                logits = tl.where(
                    has_freq & (logits > 0), logits / rep_pen, logits
                )
                logits = tl.where(
                    has_freq & (logits < 0), logits * rep_pen, logits
                )

        if temp_ptr is not None:
            logits = logits / temperature
        tl.store(logit_ptrs, logits, mask=mask)


def active_opentile_npu() -> tuple[torch.device, int]:
    target = triton.runtime.driver.active.get_current_target()
    backend = str(target.backend)
    if backend != "opentile" and not backend.startswith("opentile_"):
        raise RuntimeError(f"expected OpenTile backend, got {backend!r}")

    device = triton.runtime.driver.active.get_active_torch_device()
    device = torch.device(device)
    if device.type != "npu":
        raise RuntimeError(
            f"expected a physical NPU device, got {device}; "
            "the compile-only OpenTile stub is not valid for this test"
        )

    device_index = device.index
    if device_index is None:
        device_index = int(torch.npu.current_device())
        device = torch.device("npu", device_index)
    properties = triton.runtime.driver.active.utils.get_device_properties(
        device_index
    )
    vector_cores = int(properties["num_vectorcore"])
    if vector_cores <= 0:
        raise RuntimeError(f"invalid num_vectorcore={vector_cores}")
    return device, vector_cores


def fused_penalty_temp(
    logits: torch.Tensor,
    token_freqs: torch.Tensor,
    is_present: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    repetition_penalty: torch.Tensor,
    temperature: Optional[torch.Tensor],
) -> torch.Tensor:
    if logits.dtype != torch.float32 or logits.ndim != 2:
        raise ValueError("logits must be a two-dimensional float32 tensor")
    if token_freqs.dtype != torch.int32 or token_freqs.shape != logits.shape:
        raise ValueError("token_freqs must be int32 with the logits shape")

    device, vector_cores = active_opentile_npu()
    tensors = [
        logits,
        token_freqs,
        is_present,
        frequency_penalty,
        presence_penalty,
        repetition_penalty,
    ]
    if temperature is not None:
        tensors.append(temperature)
    if any(tensor.device != device for tensor in tensors):
        raise ValueError(f"all tensors must be on active device {device}")

    batch_size, n_vocab = logits.shape
    expected_scalars = (batch_size,)
    for name, tensor in (
        ("is_present", is_present),
        ("frequency_penalty", frequency_penalty),
        ("presence_penalty", presence_penalty),
        ("repetition_penalty", repetition_penalty),
    ):
        if tensor.dtype != torch.float32 or tensor.shape != expected_scalars:
            raise ValueError(f"{name} must be float32[{batch_size}]")
    if temperature is not None and (
        temperature.dtype != torch.float32
        or temperature.shape != expected_scalars
    ):
        raise ValueError(f"temperature must be float32[{batch_size}]")

    logits = logits.contiguous()
    token_freqs = token_freqs.contiguous()
    is_present = is_present.contiguous()
    frequency_penalty = frequency_penalty.contiguous()
    presence_penalty = presence_penalty.contiguous()
    repetition_penalty = repetition_penalty.contiguous()
    if temperature is not None:
        temperature = temperature.contiguous()

    stride_temp = 0 if temperature is None else temperature.stride(0)
    _fused_penalty_temp_kernel[(vector_cores,)](
        logits,
        token_freqs,
        is_present,
        frequency_penalty,
        presence_penalty,
        repetition_penalty,
        temperature,
        logits.stride(0),
        logits.stride(1),
        token_freqs.stride(0),
        token_freqs.stride(1),
        is_present.stride(0),
        frequency_penalty.stride(0),
        presence_penalty.stride(0),
        repetition_penalty.stride(0),
        stride_temp,
        batch_size,
        n_vocab,
        BLOCK_SIZE=1024,
    )
    return logits
