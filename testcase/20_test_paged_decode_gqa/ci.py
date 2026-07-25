"""End-to-end launcher for the production Mojo paged-decode Triton kernel.

Kernel source:
  HighPriority50Operators/mojo_opset-master/
  mojo_opset/backends/ttx/kernels/npu/flash_attention.py::paged_decode_kernel
  SHA256 2e643be5b9d08664c8bf370b663f86c45a78caca0b6c4ffc304fdf10758291f4

Input/golden source:
  mojo_opset-master/tests/accuracy/operators/test_attention.py
  ::generate_paged_decode_data + ::test_paged_decode_gqa
  first case: B8/QH16/KVH4/D128/S1024/block32/BF16, AABB;
  golden=MojoPagedDecodeGQA torch forward path
  SHA256 56d845e90190e64cde249da24f5a31811f0cf273a088cb99495220dc735af315

OpenTileAS lowering provenance:
  PR !297, head 5ac40ce2a7e4d400573d373f979968c5f10f4e49
  reference production LL SHA256
  f8878dae6c3af7031c075ae2ad031e1679e1e3d3e7db2ad4dedbde4424efbcdf

The kernel body is copied verbatim.  Immediately before AST compilation, the
same two-coordinate-to-linear-program-id transformation used by source session
019f4eee-ded3-74b3-bf64-80eef06d016a is applied for the A5 AIV launch ABI.
"""

import hashlib
import math
import os
import time

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_BACKEND", "opentile")
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")

_STAGE_START = time.monotonic()
_LAST_STAGE = "module_import"


def _stage(message):
    global _LAST_STAGE
    _LAST_STAGE = message
    elapsed = time.monotonic() - _STAGE_START
    print(f"[PAGED_STAGE] elapsed={elapsed:.3f}s {message}", flush=True)


# Match the known-good A5 E2E templates: register torch_npu and select the
# logical device before importing Triton/OpenTile kernel machinery.
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
_DEVICE_ID = int(os.environ.get("OPENTILE_TEST_DEVICE", "0"))
_COMPILE_ONLY = os.environ.get("PAGED_DECODE_COMPILE_ONLY") == "1"
if _COMPILE_ONLY:
    _stage("set_device_skipped compile_only=1")
else:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip(
            "torch_npu is available, but no NPU device is visible",
            allow_module_level=True,
        )
    _stage(f"set_device_begin logical_device={_DEVICE_ID}")
    torch.npu.set_device(_DEVICE_ID)
    _stage(f"set_device_done logical_device={_DEVICE_ID}")

import triton
import triton.language as tl

from triton.compiler import ASTSource


def _source_sha256():
    with open(__file__, "rb") as source_file:
        return hashlib.sha256(source_file.read()).hexdigest()


_stage("module_import_done")


@triton.jit
def paged_decode_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    o_ptr,
    seqlens_ptr,
    block_tables_ptr,
    BATCH_SIZE,
    NUM_Q_HEADS,
    NUM_KV_HEADS,
    GQA_INTERLEAVE,
    HEAD_DIM,
    NUM_TOTAL_BLOCKS,
    MAX_NUM_BLOCKS_PER_SEQ,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_k_block,
    stride_k_head,
    stride_k_blksz,
    stride_k_dim,
    stride_v_block,
    stride_v_head,
    stride_v_blksz,
    stride_v_dim,
    stride_ob,
    stride_oh,
    stride_od,
    stride_bt_batch,
    stride_bt_block,
    sm_scale,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    NUM_SHARE_Q_HEADS = NUM_Q_HEADS // NUM_KV_HEADS
    if GQA_INTERLEAVE:
        pid_kh = pid_h % NUM_KV_HEADS
    else:
        pid_kh = pid_h // NUM_SHARE_Q_HEADS

    kv_len = tl.load(seqlens_ptr + pid_b)

    num_logical_blocks = (kv_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N

    q_offset = pid_b * stride_qb + pid_h * stride_qh

    offs_d = tl.arange(0, BLOCK_SIZE_D)
    q_ptrs = q_ptr + q_offset + offs_d * stride_qd
    q = tl.load(q_ptrs)

    m_i = -float("inf")
    l_i = 0.0
    acc_o = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)

    for logical_block_idx in range(0, num_logical_blocks):
        bt_offset = pid_b * stride_bt_batch + logical_block_idx * stride_bt_block
        physical_block_id = tl.load(block_tables_ptr + bt_offset)

        k_block_ptr = tl.make_block_ptr(
            base=k_cache_ptr + pid_kh * stride_k_head,
            shape=(NUM_TOTAL_BLOCKS, BLOCK_SIZE_N, HEAD_DIM),
            strides=(stride_k_block, stride_k_blksz, stride_k_dim),
            offsets=(physical_block_id, 0, 0),
            block_shape=(1, BLOCK_SIZE_N, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=v_cache_ptr + pid_kh * stride_v_head,
            shape=(NUM_TOTAL_BLOCKS, BLOCK_SIZE_N, HEAD_DIM),
            strides=(stride_v_block, stride_v_blksz, stride_v_dim),
            offsets=(physical_block_id, 0, 0),
            block_shape=(1, BLOCK_SIZE_N, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )

        k = tl.load(k_block_ptr)
        v = tl.load(v_block_ptr)

        k = tl.reshape(k, (BLOCK_SIZE_N, BLOCK_SIZE_D))
        v = tl.reshape(v, (BLOCK_SIZE_N, BLOCK_SIZE_D))

        qk = tl.sum(q[None, :] * k, axis=1)

        current_logical_offset = logical_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = current_logical_offset < kv_len

        qk = tl.where(mask, qk, -float("inf"))
        qk *= sm_scale

        m_j = tl.max(qk, axis=0)
        m_new = tl.maximum(m_i, m_j)

        p = tl.exp(qk - m_new)
        l_j = tl.sum(p, axis=0)

        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_j - m_new)

        l_new = alpha * l_i + l_j

        acc_o = acc_o * alpha

        p = p.to(v.dtype)

        acc_o += tl.sum(p[:, None] * v, axis=0)

        l_i = l_new
        m_i = m_new

    acc_o = acc_o / l_i

    o_offset = pid_b * stride_ob + pid_h * stride_oh
    o_ptrs = o_ptr + o_offset + offs_d * stride_od
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty))


def _linearize_grid(kernel):
    original = "    pid_b = tl.program_id(0)\n    pid_h = tl.program_id(1)\n"
    replacement = (
        "    linear_pid = tl.program_id(0)\n"
        "    pid_b = linear_pid // NUM_Q_HEADS\n"
        "    pid_h = linear_pid % NUM_Q_HEADS\n"
    )
    if kernel.src.count(replacement) == 1:
        return
    if kernel.src.count(original) != 1:
        raise RuntimeError("unexpected paged_decode_kernel program-id source")
    kernel._src = kernel.src.replace(original, replacement)
    kernel.hash = None


def _compile_paged_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    gqa_layout: str = "AABB",
) :
    """Compile the production specialization without loading or launching it."""
    if gqa_layout not in ("AABB", "ABAB"):
        raise ValueError(f"unsupported GQA layout: {gqa_layout}")
    if query.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16 or v_cache.dtype != torch.bfloat16:
        raise TypeError("paged decode case requires BF16 query/K/V")
    if seqlens.dtype != torch.int32 or block_tables.dtype != torch.int32:
        raise TypeError("paged decode seqlens and block tables must be int32")

    query = query.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    seqlens = seqlens.contiguous()
    block_tables = block_tables.contiguous()

    batch_size, num_q_heads, head_dim = query.shape
    num_total_blocks, num_kv_heads, block_size, cache_head_dim = k_cache.shape
    if cache_head_dim != head_dim or v_cache.shape != k_cache.shape:
        raise ValueError("incompatible query/K/V cache shapes")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    _linearize_grid(paged_decode_kernel)
    constants = {
        "BATCH_SIZE": batch_size,
        "NUM_Q_HEADS": num_q_heads,
        "NUM_KV_HEADS": num_kv_heads,
        "GQA_INTERLEAVE": int(gqa_layout == "ABAB"),
        "HEAD_DIM": head_dim,
        "NUM_TOTAL_BLOCKS": num_total_blocks,
        "MAX_NUM_BLOCKS_PER_SEQ": block_tables.shape[1],
        "stride_qb": query.stride(0),
        "stride_qh": query.stride(1),
        "stride_qd": query.stride(2),
        "stride_k_block": k_cache.stride(0),
        "stride_k_head": k_cache.stride(1),
        "stride_k_blksz": k_cache.stride(2),
        "stride_k_dim": k_cache.stride(3),
        "stride_v_block": v_cache.stride(0),
        "stride_v_head": v_cache.stride(1),
        "stride_v_blksz": v_cache.stride(2),
        "stride_v_dim": v_cache.stride(3),
        "stride_ob": query.stride(0),
        "stride_oh": query.stride(1),
        "stride_od": query.stride(2),
        "stride_bt_batch": block_tables.stride(0),
        "stride_bt_block": block_tables.stride(1),
        "sm_scale": 1.0 / math.sqrt(head_dim),
        "BLOCK_SIZE_D": triton.next_power_of_2(head_dim),
        "BLOCK_SIZE_N": block_size,
    }
    source = ASTSource(
        fn=paged_decode_kernel,
        signature={
            "q_ptr": "*bf16",
            "k_cache_ptr": "*bf16",
            "v_cache_ptr": "*bf16",
            "o_ptr": "*bf16",
            "seqlens_ptr": "*i32",
            "block_tables_ptr": "*i32",
        },
        constexprs=constants,
    )
    _stage(
        "compile_begin "
        f"B={batch_size} QH={num_q_heads} KVH={num_kv_heads} "
        f"D={head_dim} block={block_size} total_blocks={num_total_blocks} "
        f"max_blocks={block_tables.shape[1]}"
    )
    compiled = triton.compile(source, options={"num_warps": 4, "multibuffer": False})
    _stage("compile_done")
    return compiled, batch_size * num_q_heads


def paged_decode_forward(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    gqa_layout: str = "AABB",
) -> torch.Tensor:
    """Compile and execute the complete production paged-decode kernel."""
    query = query.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    seqlens = seqlens.contiguous()
    block_tables = block_tables.contiguous()
    output = torch.empty_like(query)
    compiled, grid_size = _compile_paged_decode(
        query,
        k_cache,
        v_cache,
        seqlens,
        block_tables,
        gqa_layout=gqa_layout,
    )
    _stage(f"launch_begin grid={grid_size}")
    compiled[(grid_size, 1, 1)](
        query,
        k_cache,
        v_cache,
        output,
        seqlens,
        block_tables,
    )
    _stage("launch_returned_async")
    return output


_TEST_BATCH_SIZE = 8
_TEST_NUM_Q_HEADS = 16
_TEST_NUM_KV_HEADS = 4
_TEST_HEAD_DIM = 128
_TEST_MAX_SEQ_LEN = 1024
_TEST_BLOCK_SIZE = 32


def _test_device():
    return torch.device("npu", _DEVICE_ID)


def _generate_paged_decode_test_data(generator):
    # Match tests/accuracy/operators/test_attention.py: generate random inputs
    # on CPU.  The golden is also computed on CPU before the kernel inputs are
    # copied to NPU.
    _stage("cpu_input_query_randn_begin shape=(8,16,128) dtype=bf16")
    query = torch.randn(
        _TEST_BATCH_SIZE,
        _TEST_NUM_Q_HEADS,
        _TEST_HEAD_DIM,
        dtype=torch.bfloat16,
        generator=generator,
    )
    _stage("cpu_input_query_randn_done")
    _stage("cpu_input_seqlens_randint_begin shape=(8,) dtype=i32 range=[1,1024)")
    seqlens = torch.randint(
        1,
        _TEST_MAX_SEQ_LEN,
        (_TEST_BATCH_SIZE,),
        dtype=torch.int32,
        generator=generator,
    )
    _stage("cpu_input_seqlens_randint_done")
    max_blocks = (
        seqlens.max().item() + _TEST_BLOCK_SIZE - 1
    ) // _TEST_BLOCK_SIZE
    total_blocks = int(
        torch.div(
            seqlens + _TEST_BLOCK_SIZE - 1,
            _TEST_BLOCK_SIZE,
            rounding_mode="floor",
        ).sum().item()
    ) + 10
    _stage(
        "cpu_input_seqlens_shape_done "
        f"max_blocks={max_blocks} total_blocks={total_blocks} "
        f"values={seqlens.tolist()}"
    )
    cache_shape = (
        total_blocks,
        _TEST_NUM_KV_HEADS,
        _TEST_BLOCK_SIZE,
        _TEST_HEAD_DIM,
    )
    _stage(f"cpu_input_k_cache_randn_begin shape={cache_shape} dtype=bf16")
    k_cache = torch.randn(
        cache_shape,
        dtype=torch.bfloat16,
        generator=generator,
    )
    _stage("cpu_input_k_cache_randn_done")
    _stage(f"cpu_input_v_cache_randn_begin shape={cache_shape} dtype=bf16")
    v_cache = torch.randn(
        cache_shape,
        dtype=torch.bfloat16,
        generator=generator,
    )
    _stage("cpu_input_v_cache_randn_done")
    _stage(f"cpu_input_block_tables_begin shape=(8,{max_blocks}) dtype=i64")
    block_tables = torch.zeros(
        _TEST_BATCH_SIZE,
        max_blocks,
        dtype=torch.int64,
    )
    free_blocks = torch.randperm(total_blocks, generator=generator)
    cursor = 0
    for batch in range(_TEST_BATCH_SIZE):
        count = (
            seqlens[batch].item() + _TEST_BLOCK_SIZE - 1
        ) // _TEST_BLOCK_SIZE
        block_tables[batch, :count] = free_blocks[cursor : cursor + count]
        cursor += count
    _stage("cpu_input_block_tables_done")
    return query, k_cache, v_cache, seqlens, block_tables


def _reference_paged_decode(query, k_cache, v_cache, seqlens, block_tables):
    """Golden copied from MojoPagedDecodeGQA's torch forward path (AABB)."""
    batch_size, num_q_heads, head_dim = query.shape
    _, num_kv_heads, block_size, _ = k_cache.shape
    num_share_q_heads = num_q_heads // num_kv_heads
    softmax_scale = 1.0 / math.sqrt(head_dim)
    outputs = torch.zeros(
        batch_size,
        num_q_heads,
        head_dim,
        dtype=query.dtype,
        device=query.device,
    )

    for batch in range(batch_size):
        seq_len = seqlens[batch].item()
        _stage(f"golden_batch_begin batch={batch} seq_len={seq_len}")
        q = query[batch]
        k_ref = torch.zeros(
            seq_len,
            num_kv_heads,
            head_dim,
            dtype=query.dtype,
            device=query.device,
        )
        v_ref = torch.zeros_like(k_ref)
        num_blocks_for_seq = (
            seq_len + block_size - 1
        ) // block_size

        for logical_block in range(num_blocks_for_seq):
            physical_block = block_tables[batch, logical_block].item()
            start_pos = logical_block * block_size
            tokens_in_block = min(block_size, seq_len - start_pos)
            k_slice = k_cache[
                physical_block,
                :,
                :tokens_in_block,
                :,
            ]
            v_slice = v_cache[
                physical_block,
                :,
                :tokens_in_block,
                :,
            ]
            k_ref[start_pos : start_pos + tokens_in_block] = (
                k_slice.permute(1, 0, 2)
            )
            v_ref[start_pos : start_pos + tokens_in_block] = (
                v_slice.permute(1, 0, 2)
            )

        if num_share_q_heads > 1:
            k_ref = k_ref.repeat_interleave(num_share_q_heads, dim=1)
            v_ref = v_ref.repeat_interleave(num_share_q_heads, dim=1)

        attn_scores = (
            torch.einsum("hd,khd->hk", q, k_ref) * softmax_scale
        )
        attn_probs = torch.softmax(
            attn_scores,
            dim=-1,
            dtype=torch.float32,
        ).to(query.dtype)
        outputs[batch] = torch.einsum(
            "hk,khd->hd",
            attn_probs,
            v_ref,
        )
        _stage(
            f"golden_batch_submitted batch={batch} "
            f"logical_blocks={num_blocks_for_seq}"
        )
    return outputs


def test_paged_decode_gqa_opentile():
    import faulthandler

    faulthandler.dump_traceback_later(90, repeat=False)
    try:
        _stage(
            "test_begin "
            f"source_sha256={_source_sha256()} "
            f"OPENTILE_TEST_DEVICE={os.environ.get('OPENTILE_TEST_DEVICE', '0')} "
            f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', 'unset')}"
        )
        # Do not call torch.manual_seed after torch_npu registration: PyTorch
        # forwards that call to torch_npu.manual_seed_all and triggers NPU lazy
        # initialization before the intended H2D stage.  A CPU-only generator
        # preserves deterministic random inputs without touching NPU runtime.
        cpu_generator = torch.Generator()
        cpu_generator.manual_seed(0)
        _stage("cpu_input_generation_begin")
        query, k_cache, v_cache, seqlens, block_tables = (
            _generate_paged_decode_test_data(cpu_generator)
        )
        _stage(
            "cpu_input_generation_done "
            f"seqlen_min={seqlens.min().item()} "
            f"seqlen_max={seqlens.max().item()}"
        )
        _stage("cpu_golden_begin")
        expected = _reference_paged_decode(
            query,
            k_cache,
            v_cache,
            seqlens,
            block_tables,
        )
        _stage("cpu_golden_done")

        device = _test_device()
        _stage(f"h2d_query_begin device={device}")
        query_npu = query.to(device)
        _stage("h2d_query_done")
        _stage("h2d_k_cache_begin")
        k_cache_npu = k_cache.to(device)
        _stage("h2d_k_cache_done")
        _stage("h2d_v_cache_begin")
        v_cache_npu = v_cache.to(device)
        _stage("h2d_v_cache_done")
        _stage("h2d_seqlens_begin")
        seqlens_npu = seqlens.to(device)
        _stage("h2d_seqlens_done")
        _stage("h2d_block_tables_begin dtype=i32")
        block_tables_npu = block_tables.to(device=device, dtype=torch.int32)
        _stage("h2d_block_tables_done")
        _stage("h2d_synchronize_begin")
        torch.npu.synchronize()
        _stage("h2d_synchronize_done")

        actual = paged_decode_forward(
            query_npu,
            k_cache_npu,
            v_cache_npu,
            seqlens_npu,
            block_tables_npu,
            gqa_layout="AABB",
        )
        _stage("kernel_synchronize_begin")
        torch.npu.synchronize()
        _stage("kernel_synchronize_done")

        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        _stage("actual_d2h_begin")
        actual_cpu = actual.cpu()
        _stage("actual_d2h_done")
        expected_cpu = expected
        actual_f32 = actual_cpu.float()
        expected_f32 = expected_cpu.float()
        abs_diff = (actual_f32 - expected_f32).abs()
        close = torch.isclose(actual_f32, expected_f32, atol=2e-2, rtol=2e-2)
        _stage(
            "compare_summary "
            f"finite_actual={torch.isfinite(actual_f32).sum().item()}/{actual_f32.numel()} "
            f"finite_expected={torch.isfinite(expected_f32).sum().item()}/{expected_f32.numel()} "
            f"close={close.sum().item()}/{close.numel()} "
            f"max_abs={abs_diff.max().item():.9g} "
            f"mean_abs={abs_diff.mean().item():.9g}"
        )
        _stage("compare_assert_begin")
        torch.testing.assert_close(
            actual_cpu,
            expected_cpu,
            atol=2e-2,
            rtol=2e-2,
        )
        _stage("compare_pass")
    except BaseException as error:
        error_text = str(error).replace("\n", " ")[:2000]
        print(
            f"[PAGED_ERROR] after={_LAST_STAGE!r} "
            f"type={type(error).__name__} message={error_text}",
            flush=True,
        )
        # torch_npu may block indefinitely in Python/pytest atexit cleanup
        # after an ACL runtime failure.  The compact error above is the single
        # report payload, so terminate this isolated suite subprocess directly.
        os._exit(1)
    finally:
        faulthandler.cancel_dump_traceback_later()
