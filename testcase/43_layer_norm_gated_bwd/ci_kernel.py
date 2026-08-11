# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""FP32 OpenTile specialization of ``layer_norm_gated_bwd_kernel``.

The kernel follows the Triton-Ascend grid-stride implementation in
``fla/modules/backends/triton_ascend/fused_norm_gate.py``. This standalone
contract fixes the optional modes to LayerNorm + swish + affine weight/bias,
with no residual input and no output recomputation.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def layer_norm_gated_bwd_kernel(
    x,
    g,
    w,
    b,
    dy,
    dx,
    dg,
    dw,
    db,
    mean,
    rstd,
    T,
    NS,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
):
    i_s = tl.program_id(0)
    cols = tl.arange(0, BD)
    col_mask = cols < D

    b_w = tl.load(w + cols, mask=col_mask).to(tl.float32)
    b_b = tl.load(b + cols, mask=col_mask, other=0.0).to(tl.float32)
    b_dw = tl.zeros((BD,), dtype=tl.float32)
    b_db = tl.zeros((BD,), dtype=tl.float32)

    n_tiles = tl.cdiv(T, BT)
    for i_t in range(i_s, n_tiles, NS):
        rows = i_t * BT + tl.arange(0, BT)
        row_mask = rows < T
        mask = row_mask[:, None] & col_mask[None, :]
        row_off = rows[:, None].to(tl.int64) * D + cols[None, :].to(tl.int64)

        b_x = tl.load(x + row_off, mask=mask, other=0.0).to(tl.float32)
        b_g = tl.load(g + row_off, mask=mask, other=0.0).to(tl.float32)
        b_dy = tl.load(dy + row_off, mask=mask, other=0.0).to(tl.float32)
        b_mean = tl.load(mean + rows, mask=row_mask, other=0.0)
        b_rstd = tl.load(rstd + rows, mask=row_mask, other=0.0)

        b_xhat = (b_x - b_mean[:, None]) * b_rstd[:, None]
        b_xhat = tl.where(mask, b_xhat, 0.0)
        b_y = b_xhat * b_w[None, :] + b_b[None, :]

        b_sigmoid_g = tl.sigmoid(b_g)
        b_dsilu = b_sigmoid_g * (1 + b_g * (1 - b_sigmoid_g))
        b_dg = b_dy * b_y * b_dsilu
        b_dy = b_dy * b_g * b_sigmoid_g

        b_dw += tl.sum(tl.where(mask, b_dy * b_xhat, 0.0), axis=0)
        b_db += tl.sum(tl.where(mask, b_dy, 0.0), axis=0)
        b_wdy = b_dy * b_w[None, :]
        b_c1 = tl.sum(b_xhat * b_wdy, axis=1) / D
        b_c2 = tl.sum(b_wdy, axis=1) / D
        b_dx = (b_wdy - (b_xhat * b_c1[:, None] + b_c2[:, None])) * b_rstd[:, None]

        tl.store(dx + row_off, b_dx.to(dx.dtype.element_ty), mask=mask)
        tl.store(dg + row_off, b_dg.to(dg.dtype.element_ty), mask=mask)

    tl.store(dw + i_s * D + cols, b_dw, mask=col_mask)
    tl.store(db + i_s * D + cols, b_db, mask=col_mask)
