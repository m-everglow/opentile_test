"""End-to-end launcher for the production Mojo paged-decode Triton kernel.

Kernel source:
  HighPriority50Operators/mojo_opset-master/
  mojo_opset/backends/ttx/kernels/npu/flash_attention.py::paged_decode_kernel
  SHA256 2e643be5b9d08664c8bf370b663f86c45a78caca0b6c4ffc304fdf10758291f4

The kernel body is copied verbatim.  Immediately before AST compilation, the
same two-coordinate-to-linear-program-id transformation used by source session
019f4eee-ded3-74b3-bf64-80eef06d016a is applied for the A5 AIV launch ABI.
"""

import math

import torch
import triton
import triton.language as tl

from triton.compiler import ASTSource


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

    output = torch.empty_like(query)
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
        "stride_ob": output.stride(0),
        "stride_oh": output.stride(1),
        "stride_od": output.stride(2),
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
    compiled = triton.compile(source, options={"num_warps": 4, "multibuffer": False})
    compiled[(batch_size * num_q_heads, 1, 1)](
        query,
        k_cache,
        v_cache,
        o[<0;86;22Mutput,
        seqlens,
        block_tables,
    )
    return output

