"""
minimal_repro.py
=================
独立排查脚本：只跑 triton_hstu_attention_fwd 100 次，定位是否是 Triton 算子内部 race。

判定标准：
  - triton_output 跨次运行 self-diff ≠ 0  →  Triton 内部 race，确认问题在算子
  - triton_output 跨次运行 self-diff = 0   →  Triton 内部确定，问题在 golden / 框架

用法：
  cd /home/VSCode/opentile_test/hstu
  python test/minimal_repro.py [iter_count]
默认 iter_count=100。
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

import torch
import torch_npu
import numpy as np
import random

from op.hstu_triton_fwd import triton_hstu_attention_fwd


# 目标 case: fp8-2048-2-2-64-32-256-256
B, Hq, Hk, Sq, Sk, Ad, Ld = 2048, 2, 2, 64, 32, 256, 256
DTYPE = torch.float8_e4m3fn


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_inputs():
    """build the same inputs as test_hstu_fwd_jagged.py would."""
    set_seed(42)
    num_tokens_q = B * Sq
    num_tokens_k = B * Sk

    q = torch.rand(num_tokens_q, Hq, Ad).to(DTYPE).npu()
    k = torch.rand(num_tokens_k, Hk, Ad).to(DTYPE).npu()
    v = torch.rand(num_tokens_k, Hk, Ld).to(DTYPE).npu()

    zero = torch.tensor([0], dtype=torch.int64).npu()
    seq_offset_q = torch.cat(
        [zero, torch.arange(Sq, num_tokens_q + 1, Sq, dtype=torch.int64).npu()])
    seq_offset_k = torch.cat(
        [zero, torch.arange(Sk, num_tokens_k + 1, Sk, dtype=torch.int64).npu()])

    alpha = 1.0 / (Ad ** 0.5)
    return q, k, v, seq_offset_q, seq_offset_k, alpha


def main(iter_count=100):
    print(f"[INFO] minimal_repro: case=fp8-{B}-{Hq}-{Hk}-{Sq}-{Sk}-{Ad}-{Ld}, iters={iter_count}")
    q, k, v, seq_offset_q, seq_offset_k, alpha = build_inputs()

    ref = None
    diffs = []
    nan_total = 0
    inf_total = 0
    unique_outputs = set()

    for i in range(iter_count):
        out = triton_hstu_attention_fwd(
            q=q, k=k, v=v,
            max_seq_len_q=Sq, max_seq_len_k=Sk,
            seq_offsets_q=seq_offset_q, seq_offsets_k=seq_offset_k,
            num_context=None, num_target=None,
            alpha=alpha, silu_scale=1.0 / Sq,
        )
        torch.npu.synchronize()
        out_f32 = out.float()
        n_nan = int(torch.isnan(out_f32).sum().item())
        n_inf = int(torch.isinf(out_f32).sum().item())
        nan_total += n_nan
        inf_total += n_inf

        if ref is None:
            ref = out.detach().clone()
            diff = 0.0
            print(f"[iter {i:3d}] REF  nan={n_nan} inf={n_inf}")
        else:
            diff = (out_f32 - ref.float()).abs().max().item()
            print(f"[iter {i:3d}] max_diff={diff:.6e}  nan={n_nan}  inf={n_inf}")
            diffs.append(diff)
            # 用 sha1 区分不同的输出（避免存大张量）
            sample = out_f32.flatten()[:8].cpu().tolist()
            unique_outputs.add(tuple(round(x, 4) for x in sample))

    print("\n" + "=" * 60)
    print(f"[SUMMARY] total iters: {iter_count}")
    print(f"[SUMMARY] nan_total: {nan_total}, inf_total: {inf_total}")
    if diffs:
        print(f"[SUMMARY] self-diff: min={min(diffs):.6e}  "
              f"max={max(diffs):.6e}  mean={sum(diffs)/len(diffs):.6e}")
        nonzero = sum(1 for d in diffs if d > 0)
        print(f"[SUMMARY] non-zero diffs: {nonzero}/{len(diffs)}")
    print(f"[SUMMARY] unique 8-sample prefixes: {len(unique_outputs)}")
    if len(unique_outputs) == 1 and (not diffs or max(diffs) == 0):
        print("[SUMMARY] ✅ Triton output is DETERMINISTIC across runs")
        print("[SUMMARY]    → 不是 triton 内部 race，问题在 golden / 框架")
    else:
        print("[SUMMARY] ⚠️  Triton output is NON-DETERMINISTIC across runs")
        print("[SUMMARY]    → 确认是 triton 内部 race，检查：")
        print("[SUMMARY]      - multi-stage pipeline (num_stages)")
        print("[SUMMARY]      - UB 未清零")
        print("[SUMMARY]      - Cube/Vector 跨核同步")
    print("=" * 60)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(n)