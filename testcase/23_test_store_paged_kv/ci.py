"""Standalone NPU store_paged_kv_cache kernel + CPU paged-copy golden.

Run directly:
  python3 test_store_paged_kv_forward_diff_standalone.py

Run with pytest:
  pytest -s --assert=plain test_store_paged_kv_forward_diff_standalone.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_BACKEND", "opentile")
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
os.environ.setdefault("TRITON_DEBUG", "1")
os.environ.setdefault(
    "TRITON_CACHE_DIR", f"/tmp/store_paged_kv_diag_cache_{os.getpid()}"
)

import torch
import triton
import triton.language as tl

SEED = 2026071840
HEAD_DIM = 128
NUM_KV_HEADS = 1
BLOCK_SIZE = 128
CHUNK_SIZE = 64

DTYPES = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
}
TOLERANCES = {
    "f32": (0, 0),
    "f16": (1e-3, 1e-3),
    "bf16": (5e-3, 5e-3),
}

# (seq_len, logical_start, num_batches)
#   - aligned: seq_len fits in one block, logical_start=0
#   - tail-cross-block: logical_start offsets into a second physical block
#   - multi-batch: multiple sequences sharing the cache
CASES = (
    (64, 0, 1),       # aligned, single batch
    (85, 100, 1),     # tail-cross-block, single batch
    (128, 0, 1),      # full block, single batch
    (200, 0, 1),      # multi-block, single batch
    (64, 0, 3),       # multi-batch, each 64 tokens
    (85, 50, 2),      # multi-batch, tail-cross
)


@triton.jit
def _store_paged_kv_cache_kernel(
    k_ptr,
    v_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    cu_seqlens_ptr,
    chunk_indices_ptr,
    stride_k_tok,
    stride_k_head,
    stride_k_dim,
    stride_v_tok,
    stride_v_head,
    stride_v_dim,
    stride_kc_blk,
    stride_kc_head,
    stride_kc_tok,
    stride_kc_dim,
    stride_vc_blk,
    stride_vc_head,
    stride_vc_tok,
    stride_vc_dim,
    stride_bt_batch,
    stride_bt_blk,
    num_kv_heads,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    total_chunks,
    CHUNK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for chunk_id_linear in range(pid, total_chunks, num_programs):
        meta_ptr = chunk_indices_ptr + chunk_id_linear * 3
        batch_idx = tl.load(meta_ptr)
        token_offset_in_seq = tl.load(meta_ptr + 1)
        logical_kv_start = tl.load(meta_ptr + 2)

        seq_start_tok = tl.load(cu_seqlens_ptr + batch_idx)
        global_token_idx = seq_start_tok + token_offset_in_seq

        seq_len_curr = tl.load(cu_seqlens_ptr + batch_idx + 1) - seq_start_tok
        valid_len = seq_len_curr - token_offset_in_seq

        curr_log_pos = logical_kv_start
        curr_kv_pos = global_token_idx

        remain_chunk_len = CHUNK_SIZE
        remain_chunk_len = tl.minimum(remain_chunk_len, valid_len)

        processed = 0
        while processed < remain_chunk_len:
            block_table_idx = curr_log_pos // block_size
            block_inner_off = curr_log_pos % block_size

            physical_block_id = tl.load(block_table_ptr + batch_idx * stride_bt_batch + block_table_idx * stride_bt_blk)

            space_in_block = block_size - block_inner_off
            sub_len = tl.minimum(remain_chunk_len - processed, space_in_block).to(tl.int32)

            offs_sub = tl.arange(0, CHUNK_SIZE)
            mask_sub = offs_sub < sub_len

            offs_d = tl.arange(0, head_dim)

            for h in range(num_kv_heads):
                src_k_ptr = (
                    k_ptr
                    + (curr_kv_pos + offs_sub[:, None]) * stride_k_tok
                    + h * stride_k_head
                    + offs_d[None, :] * stride_k_dim
                )
                k_val = tl.load(src_k_ptr, mask=mask_sub[:, None], other=0.0)

                dst_k_ptr = (
                    key_cache_ptr
                    + physical_block_id * stride_kc_blk
                    + h * stride_kc_head
                    + (block_inner_off + offs_sub[:, None]) * stride_kc_tok
                    + offs_d[None, :] * stride_kc_dim
                )
                tl.store(dst_k_ptr, k_val, mask=mask_sub[:, None])

                src_v_ptr = (
                    v_ptr
                    + (curr_kv_pos + offs_sub[:, None]) * stride_v_tok
                    + h * stride_v_head
                    + offs_d[None, :] * stride_v_dim
                )
                v_val = tl.load(src_v_ptr, mask=mask_sub[:, None], other=0.0)

                dst_v_ptr = (
                    value_cache_ptr
                    + physical_block_id * stride_vc_blk
                    + h * stride_vc_head
                    + (block_inner_off + offs_sub[:, None]) * stride_vc_tok
                    + offs_d[None, :] * stride_vc_dim
                )
                tl.store(dst_v_ptr, v_val, mask=mask_sub[:, None])

            processed += sub_len
            curr_log_pos += sub_len
            curr_kv_pos += sub_len


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


def _chunk_metadata(seq_lens, logical_starts):
    """Build [total_chunks, 3] int32 tensor: (batch_idx, token_offset, logical_kv_start)."""
    rows = []
    for batch_idx, (slen, lstart) in enumerate(zip(seq_lens, logical_starts)):
        for token_offset in range(0, slen, CHUNK_SIZE):
            rows.append((batch_idx, token_offset, lstart + token_offset))
    return torch.tensor(rows, dtype=torch.int32)


def _cpu_reference(
    k: torch.Tensor,
    v: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens,
    logical_starts,
    num_blocks: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU golden: indexed paged-cache copy. Unwritten regions stay NaN sentinel."""
    k_cache = torch.full(
        (num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM),
        float("nan"),
        dtype=dtype,
    )
    v_cache = torch.full_like(k_cache, float("nan"))
    token_offset = 0
    for batch_idx, (slen, lstart) in enumerate(zip(seq_lens, logical_starts)):
        for token in range(slen):
            logical = lstart + token
            table_index = logical // BLOCK_SIZE
            inner = logical % BLOCK_SIZE
            physical = int(block_table[batch_idx, table_index].item())
            k_cache[physical, :, inner, :] = k[token_offset + token]
            v_cache[physical, :, inner, :] = v[token_offset + token]
        token_offset += slen
    return k_cache, v_cache


def _compare_with_nan(
    name: str, actual: torch.Tensor, golden: torch.Tensor, dtype: str
) -> None:
    atol, rtol = TOLERANCES[dtype]
    actual_f = actual.detach().cpu().float()
    golden_f = golden.detach().cpu().float()
    same_nan = torch.isnan(actual_f) & torch.isnan(golden_f)
    finite_pair = torch.isfinite(actual_f) & torch.isfinite(golden_f)
    close = torch.zeros_like(same_nan)
    close[same_nan] = True
    if finite_pair.any():
        close[finite_pair] = torch.isclose(
            actual_f[finite_pair], golden_f[finite_pair], atol=atol, rtol=rtol
        )
    bad_mask = ~close
    mismatch = int(bad_mask.sum().item())
    diff = (actual_f[finite_pair] - golden_f[finite_pair]).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    total = actual_f.numel()
    print(
        f"{name} dtype={dtype} shape={tuple(actual.shape)} "
        f"pass={total - mismatch}/{total} bad={mismatch} "
        f"max_abs={max_abs:.9g} "
        f"atol={atol} rtol={rtol}"
    )
    if mismatch:
        idx = bad_mask.nonzero(as_tuple=False)
        rows = idx[:, [0, 2]]
        unique_rows, counts = torch.unique(rows, dim=0, return_counts=True)
        print(f"{name} BAD_ROWS count={unique_rows.shape[0]}")
        for row, count in zip(unique_rows[:64], counts[:64]):
            block, cache_row = map(int, row.tolist())
            print(
                f"  block={block} row={cache_row} "
                f"bad_dims={int(count)}"
            )
        print(f"{name} FIRST_BAD_ELEMENTS")
        for coord in idx[:32]:
            block, head, cache_row, dim = map(int, coord.tolist())
            actual_value = float(actual_f[block, head, cache_row, dim])
            golden_value = float(golden_f[block, head, cache_row, dim])
            print(
                f"  ({block},{head},{cache_row},{dim}) "
                f"actual={actual_value!r} golden={golden_value!r}"
            )
    return mismatch


def run_one(dtype_name: str, case: tuple[int, int, int], device: torch.device) -> None:
    torch_dtype = DTYPES[dtype_name]
    seq_len_per_batch, logical_start, num_batches = case
    seq_lens = [seq_len_per_batch] * num_batches
    logical_starts = [logical_start] * num_batches
    total_tokens = seq_len_per_batch * num_batches

    torch.manual_seed(SEED)
    k_cpu = torch.randn((total_tokens, NUM_KV_HEADS, HEAD_DIM), dtype=torch_dtype)
    v_cpu = torch.randn_like(k_cpu)

    # Compute num_blocks needed
    max_logical = logical_start + seq_len_per_batch - 1
    table_entries = max_logical // BLOCK_SIZE + 1
    # Each batch owns a disjoint physical-block range in block_table.  Size the
    # cache for every referenced physical block, plus one untouched sentinel
    # block.  Using table_entries + 1 under-allocates multi-batch cases.
    num_blocks = table_entries * num_batches + 1

    # block_table: [num_batches, table_entries]
    block_table_cpu = torch.arange(
        table_entries * num_batches, dtype=torch.int32
    ).reshape(num_batches, table_entries)

    # cu_seqlens: [num_batches + 1]
    cu_seqlens_cpu = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lens), 0).tolist()),
        dtype=torch.int32,
    )

    # chunk_indices: [total_chunks, 3]
    chunk_indices_cpu = _chunk_metadata(seq_lens, logical_starts)
    total_chunks = chunk_indices_cpu.shape[0]

    # CPU golden
    expected_k, expected_v = _cpu_reference(
        k_cpu, v_cpu, block_table_cpu, cu_seqlens_cpu,
        seq_lens, logical_starts, num_blocks, torch_dtype,
    )

    # NPU
    k = k_cpu.to(device)
    v = v_cpu.to(device)
    key_cache = torch.full(expected_k.shape, float("nan"), dtype=torch_dtype, device=device)
    value_cache = torch.full_like(key_cache, float("nan"))
    block_table = block_table_cpu.to(device)
    cu_seqlens = cu_seqlens_cpu.to(device)
    chunk_indices = chunk_indices_cpu.to(device)

    num_cores = _num_vector_cores()
    grid = (num_cores,)

    _store_paged_kv_cache_kernel[grid](
        k,
        v,
        key_cache,
        value_cache,
        block_table,
        cu_seqlens,
        chunk_indices,
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
        block_table.stride(0), block_table.stride(1),
        NUM_KV_HEADS,
        HEAD_DIM,
        BLOCK_SIZE,
        total_chunks,
        CHUNK_SIZE=CHUNK_SIZE,
    )
    torch.npu.synchronize()

    k_bad = _compare_with_nan(f"K_{dtype_name}_seq{seq_len_per_batch}ls{logical_start}b{num_batches}", key_cache, expected_k, dtype_name)
    v_bad = _compare_with_nan(f"V_{dtype_name}_seq{seq_len_per_batch}ls{logical_start}b{num_batches}", value_cache, expected_v, dtype_name)

    del k, v, key_cache, value_cache, block_table, cu_seqlens, chunk_indices
    torch.npu.empty_cache()
    return k_bad, v_bad


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _print_identity_and_artifacts() -> None:
    print("DIAGNOSTIC_IDENTITY")
    for tool in ("tile-opt", "tile-translate", "ccec"):
        found = shutil.which(tool)
        resolved = str(Path(found).resolve()) if found else "NOT_FOUND"
        sha = _sha256(Path(resolved)) if found else "N/A"
        print(f"  {tool}: path={resolved} sha256={sha}")
    print(f"  TRITON_CACHE_DIR={os.environ['TRITON_CACHE_DIR']}")
    cache = Path(os.environ["TRITON_CACHE_DIR"])
    if cache.exists():
        for path in sorted(cache.rglob("*")):
            if path.is_file() and path.suffix in {".o", ".ll", ".mlir", ".log"}:
                print(
                    f"  ARTIFACT path={path} size={path.stat().st_size} "
                    f"sha256={_sha256(path)}"
                )


def test_store_paged_kv_forward_diff_standalone() -> None:
    device = _setup_npu()
    print(
        "DIAGNOSTIC_CASE dtype=f32 seq_len=85 logical_start=100 "
        "batch=1 repeats=5"
    )
    results = []
    for repeat in range(5):
        print(f"DIAGNOSTIC_REPEAT={repeat}")
        results.append(run_one("f32", (85, 100, 1), device))
    _print_identity_and_artifacts()
    print(f"DIAGNOSTIC_RESULTS={results}")
    assert all(k_bad == 0 and v_bad == 0 for k_bad, v_bad in results), results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("all", *DTYPES), default="bf16")
    args = parser.parse_args()
    device = _setup_npu()
    dtype_names = tuple(DTYPES) if args.dtype == "all" else (args.dtype,)
    for dtype_name in dtype_names:
        for case in CASES:
            run_one(dtype_name, case, device)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
