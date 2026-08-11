"""Standalone OpenTile implementation of ``recompute_w_u_fwd``.

The original FLA imports have been replaced with the required helper
implementations so this module does not depend on ``src.kernels.fla``.
"""

import os

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice


if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":

    @triton.jit
    def exp(x):
        return tldevice.fast_expf(x.to(tl.float32))

    @triton.jit
    def exp2(x):
        return tldevice.exp2(x.to(tl.float32))

else:

    @triton.jit
    def exp(x):
        return tl.exp(x.to(tl.float32))

    @triton.jit
    def exp2(x):
        return tl.math.exp2(x.to(tl.float32))


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)


def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> torch.LongTensor:
    if cu_seqlens_cpu is not None:
        indices = torch.cat(
            [
                torch.arange(n, device=cu_seqlens.device)
                for n in triton.cdiv(
                    prepare_lens(cu_seqlens_cpu),
                    chunk_size,
                ).tolist()
            ]
        )
        return torch.stack(
            [indices.eq(0).cumsum(0) - 1, indices],
            1,
        ).to(cu_seqlens)
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(
                prepare_lens(cu_seqlens),
                chunk_size,
            ).tolist()
        ]
    )
    return torch.stack(
        [indices.eq(0).cumsum(0) - 1, indices],
        1,
    ).to(cu_seqlens)


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_b = tl.make_block_ptr(
        beta + bos * HV + i_h,
        (T,),
        (HV,),
        (i_t * BT,),
        (BT,),
        (0,),
    )
    b_b = tl.load(p_b, boundary_check=(0,))
    p_A = tl.make_block_ptr(
        A + (bos * HV + i_h) * BT,
        (T, BT),
        (HV * BT, 1),
        (i_t * BT, 0),
        (BT, BT),
        (1, 0),
    )
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * HV + i_h) * V,
            (T, V),
            (HV * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * HV + i_h) * V,
            (T, V),
            (HV * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(
            p_u,
            b_u.to(p_u.dtype.element_ty),
            boundary_check=(0, 1),
        )

    if USE_G:
        p_g = tl.make_block_ptr(
            g + bos * HV + i_h,
            (T,),
            (HV,),
            (i_t * BT,),
            (BT,),
            (0,),
        )
        if USE_EXP2:
            b_g = exp2(tl.load(p_g, boundary_check=(0,)))
        else:
            b_g = exp(tl.load(p_g, boundary_check=(0,)))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h // (HV // H)) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + (bos * HV + i_h) * K,
            (T, K),
            (HV * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_b[:, None]
        if USE_G:
            b_kb *= b_g[:, None]
        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(
            p_w,
            b_w.to(p_w.dtype.element_ty),
            boundary_check=(0, 1),
        )


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = A.shape[-1]
    BK = 128
    BV = 128

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = k.new_empty(B, T, HV, K)
    u = torch.empty_like(v)
    recompute_w_u_fwd_kernel[(NT, B * HV)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        USE_EXP2=use_exp2,
    )
    return w, u