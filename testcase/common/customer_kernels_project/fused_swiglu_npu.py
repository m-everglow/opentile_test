import torch
import torch_npu
import triton
import triton.language as tl
# from maybe_triton_jit import maybe_triton_jit
# from triton.language.extra import libdevice
from utils import is_hopper, is_ampere


def cal_precision(input: torch.tensor, golden: torch.tensor):
    # 最大相对误差
    relative_max = torch.max(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    # print(f"max relative error: {relative_max}")
    # 平均相对误差
    relative_mean = torch.mean(torch.abs(input-golden)/(torch.abs(golden)+float(1e-7)))
    # print(f"mean relative error: {relative_mean}")
    # 均方根误差
    rmse = torch.sqrt(torch.mean((input-golden)**2))
    # print(f"rmse: {rmse}")
    # 小值域绝对误差
    mask = (torch.abs(golden) < float(2e-14) ) & (torch.abs(input-golden) > float(2e-30))
    error = torch.abs(input-golden)
    total_error = torch.sum(error[mask])
    # print(f"small val error: {total_error}")
    return relative_max, relative_mean, rmse


def diff(x, y):
    assert x.shape == y.shape
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    diff_max = torch.max(torch.abs(x - y)).item()
    diff_sum = torch.sum(torch.abs(x - y)).item()
    return f"diff.max: {diff_max:.3f}, diff.avg: {100.0 * diff_sum / (torch.sum(torch.abs(x)).item() + 1e-10):.3f}%"


def naive_torch_swiglu(x, w_g, w_fc, b_g, b_fc):
    gate = torch.nn.functional.silu(torch.nn.functional.linear(x, w_g.T, b_g))
    fc = torch.nn.functional.linear(x, w_fc.T, b_fc)
    y = gate * fc
    return y


@triton.jit
def fast_sigmoid(x):
    return tl.fdiv(1.0, 1.0 + tl.exp(-x))

@triton.jit
def fast_silu(x):
    dtype = x.type.element_ty
    x = x.to(tl.float32)
    return tl.fdiv(x, 1.0 + tl.exp(-x)).to(dtype)


@triton.jit
def fast_silu_bwd(dy, x):
    dtype = x.type.element_ty
    dy = dy.to(tl.float32)
    x = x.to(tl.float32)
    sigmoid = fast_sigmoid(x)
    return (dy * sigmoid * (1 + x * (1 - sigmoid))).to(dtype)


@triton.jit
def atomic_store(out_ptr, acc, mask, LOCKS, SPLIT_K):
    LOCKS = LOCKS + tl.program_id(0)
    COUNT = LOCKS + tl.num_programs(0)
    while tl.atomic_cas(LOCKS, 0, 1) == 1:
        pass
    count = tl.load(COUNT)
    if count == 0:
        tl.store(out_ptr, acc, mask=mask)
    else:
        cur = tl.load(out_ptr, mask=mask, other=0.0)
        tl.store(out_ptr, acc + cur, mask=mask)
    tl.atomic_xchg(COUNT, (count + 1) % SPLIT_K)
    tl.atomic_xchg(LOCKS, 0)


def fwd_autotune_config():
    if is_hopper():
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=3,
                num_warps=4,
            )
        ]
    elif is_ampere():
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=3,
                num_warps=4,
            )
        ]
    else:
        return [
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 128,
            #         "BLOCK_SIZE_N": 64,
            #         "BLOCK_SIZE_K": 32,
            #         "GROUP_SIZE_M": 8,
            #     },
            #     num_stages=3,
            #     num_warps=4,
            # )
             triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=4,
        )
        for bm in [128, 256, 512]
        for bn in [64, 128, 256, 512]
        for bk in [32, 64, 128, 512] #BLOCK_SIZE_M: 128, BLOCK_SIZE_N: 128, BLOCK_SIZE_K: 128
    ]


def bwd_b_autotune_config():
    if is_hopper():
        return [
            triton.Config(
                {"BLOCK_SIZE_M": 512, "BLOCK_SIZE_N": 16}, num_stages=3, num_warps=8
            )
        ]
    elif is_ampere():
        return [
            triton.Config(
                {"BLOCK_SIZE_M": 512, "BLOCK_SIZE_N": 16}, num_stages=4, num_warps=8
            )
        ]
    else:
        return [
            triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
            },
            num_stages=3,
            num_warps=8,
            )
            # M：16 ~ 128，步长 16
            for bm in range(16, 128 + 1, 16)
            # N：16 ~ 64，步长 16
            for bn in range(16, 64 + 1, 16)
        ]


def bwd_x_autotune_config():
    if is_hopper():
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=3,
                num_warps=4,
            )
        ]
    elif is_ampere():
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=3,
                num_warps=4,
            )
        ]
    else:
        return [
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 128,
            #         "BLOCK_SIZE_N": 128,
            #         "BLOCK_SIZE_K": 128,
            #         "GROUP_SIZE_M": 8,
            #     },
            #     num_stages=2,
            #     num_warps=4,
            # )
               triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=4,
        )
        for bm in [128, 256, 512]
        for bn in [64, 128, 256, 512]
        for bk in [32, 64, 128, 512]
        ]


def bwd_w_autotune_config():
    if is_hopper():
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 64,
                    "SPLIT_K": 1,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=3,
                num_warps=4,
            )
        ]
    elif is_ampere():
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 64,
                    "SPLIT_K": 1,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=3,
                num_warps=4,
            )
        ]
    else:
        return [
             triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "SPLIT_K": 1,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=4,
            )
        for bm in [128, 256,  512]
        for bn in [64, 128, 256, 512]
        for bk in [32, 64, 128, 512]
        ]


@triton.autotune(
    configs=fwd_autotune_config(),
    key=["N", "K", "IS_TRAINING"],
)
@triton.jit
def fused_swiglu_fwd_kernel(
    x_ptr,
    w_g_ptr,
    w_fc_ptr,
    b_g_ptr,
    b_fc_ptr,
    y_ptr,
    g_ptr,
    fc_ptr,
    M,
    N,
    K,
    IS_TRAINING: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    dtype = y_ptr.type.element_ty
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    if (pid_m * BLOCK_SIZE_M >= M) or (pid_n * BLOCK_SIZE_N >= N):
        return

    offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offset_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))

    b_g_ptrs = b_g_ptr + offset_wn
    b_fc_ptrs = b_fc_ptr + offset_wn
    b_g = tl.load(b_g_ptrs, mask=offset_wn < N, other=0.0)
    b_fc = tl.load(b_fc_ptrs, mask=offset_wn < N, other=0.0)

    accumulator_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator_fc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_idx in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offset_k = k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        x_ptrs = x_ptr + (offset_xm[:, None] * K + offset_k[None, :])
        x_mask = (offset_xm[:, None] < M) & (offset_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        g_fc_offs = (offset_k[:, None] * N + offset_wn[None, :])
        g_fc_mask = (offset_k[:, None] < K) & (offset_wn[None, :] < N)

        w_g = tl.load(w_g_ptr + g_fc_offs, mask=g_fc_mask, other=0.0)
        w_fc = tl.load(w_fc_ptr + g_fc_offs, mask=g_fc_mask, other=0.0)
        accumulator_g = tl.dot(x, w_g, accumulator_g)
        accumulator_fc = tl.dot(x, w_fc, accumulator_fc)
        # Advance the ptrs to the next K block.
    accumulator_g += b_g[None, :]
    accumulator_fc += b_fc[None, :]
    accumulator_g = accumulator_g.to(dtype)
    accumulator_fc = accumulator_fc.to(dtype)
    silu_g = fast_silu(accumulator_g)
    hadamard_product = silu_g.to(tl.float32) * accumulator_fc.to(tl.float32)
    y = hadamard_product.to(dtype)

    offset_ym = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    y_ptrs = y_ptr + N * offset_ym[:, None] + offset_yn[None, :]
    y_mask = (offset_ym[:, None] < M) & (offset_yn[None, :] < N)
    tl.store(y_ptrs, y, mask=y_mask)
    accumulator_g = accumulator_g.to(dtype)
    accumulator_fc = accumulator_fc.to(dtype)
    if IS_TRAINING:
        g_ptrs = g_ptr + N * offset_ym[:, None] + offset_yn[None, :]
        fc_ptrs = fc_ptr + offset_ym[:, None] * N + offset_yn[None, :]
        tl.store(g_ptrs, accumulator_g, mask=y_mask)
        tl.store(fc_ptrs, accumulator_fc, mask=y_mask)


@triton.autotune(
    configs=bwd_b_autotune_config(),
    key=["N"],
)
@triton.jit
def fused_swiglu_bwd_b_kernel(
    dy_ptr,
    g_ptr,
    fc_ptr,
    dg_ptr,
    dfc_ptr,
    db_g_ptr,
    db_fc_ptr,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    dtype = dy_ptr.type.element_ty
    col_idx = tl.program_id(axis=0)
    col_off = col_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_off = tl.arange(0, BLOCK_SIZE_M)
    dy_ptrs = dy_ptr + (row_off[:, None] * N + col_off[None, :])
    g_ptrs = g_ptr + (row_off[:, None] * N + col_off[None, :])
    fc_ptrs = fc_ptr + (row_off[:, None] * N + col_off[None, :])
    dg_ptrs = dg_ptr + (row_off[:, None] * N + col_off[None, :])
    dfc_ptrs = dfc_ptr + (row_off[:, None] * N + col_off[None, :])
    sum_b_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    sum_b_fc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for row_idx in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        mask = (row_off[:, None] < M - row_idx * BLOCK_SIZE_M) & (col_off[None, :] < N)
        dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptrs, mask=mask, other=0.0)
        fc = tl.load(fc_ptrs, mask=mask, other=0.0).to(tl.float32)
        silu_g = fast_silu(g)
        dg = (dy * fc).to(dtype)
        dg = fast_silu_bwd(dg, g)
        dfc = (dy * silu_g.to(tl.float32)).to(dtype)
        sum_b_g += dg.to(tl.float32)
        sum_b_fc += dfc.to(tl.float32)
        tl.store(dg_ptrs, dg, mask=mask)
        tl.store(dfc_ptrs, dfc, mask=mask)
        dy_ptrs += BLOCK_SIZE_M * N
        g_ptrs += BLOCK_SIZE_M * N
        fc_ptrs += BLOCK_SIZE_M * N
        dg_ptrs += BLOCK_SIZE_M * N
        dfc_ptrs += BLOCK_SIZE_M * N
    tl.store(db_g_ptr + col_off, tl.sum(sum_b_g, 0), mask=col_off < N)
    tl.store(db_fc_ptr + col_off, tl.sum(sum_b_fc, 0), mask=col_off < N)


@triton.autotune(
    configs=bwd_x_autotune_config(),
    key=["N", "K"],
)
@triton.jit
def fused_swiglu_bwd_x_kernel(
    dg_ptr,
    dfc_ptr,
    w_g_ptr,
    w_fc_ptr,
    dx_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    dtype = dx_ptr.type.element_ty
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    if (pid_m * BLOCK_SIZE_M >= M) or (pid_n * BLOCK_SIZE_N >= N):
        return

    offset_dym = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offset_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))

    accumulator_dx_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator_dx_fc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offset_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        dg_ptrs = dg_ptr + (offset_dym[:, None] * K + offset_k[None, :])
        dfc_ptrs = dfc_ptr + (offset_dym[:, None] * K + offset_k[None, :])
        w_g_ptrs = w_g_ptr + (offset_k[:, None] + offset_wn[None, :] * K)
        w_fc_ptrs = w_fc_ptr + (offset_k[:, None] + offset_wn[None, :] * K)

        dg_fc_mask = (offset_dym[:, None] < M) & (offset_k[None, :] < K)
        w_g_fc_mask = (offset_k[:, None] < K) & (offset_wn[None, :] < N)

        dg = tl.load(dg_ptrs, mask=dg_fc_mask, other=0.0)
        dfc = tl.load(dfc_ptrs, mask=dg_fc_mask, other=0.0)
        w_g = tl.load(w_g_ptrs, mask=w_g_fc_mask, other=0.0)
        w_fc = tl.load(w_fc_ptrs, mask=w_g_fc_mask, other=0.0)

        # bwd x for gate
        accumulator_dx_g = tl.dot(dg, w_g, accumulator_dx_g)
        # bwd x for fc
        accumulator_dx_fc = tl.dot(dfc, w_fc, accumulator_dx_fc)
    dx = (
        accumulator_dx_g.to(dtype).to(tl.float32)
        + accumulator_dx_fc.to(dtype).to(tl.float32)
    ).to(dtype)

    offset_dxm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_dxn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dx_ptrs = dx_ptr + offset_dxm[:, None] * N + offset_dxn[None, :]
    dx_mask = (offset_dxm[:, None] < M) & (offset_dxn[None, :] < N)
    tl.store(dx_ptrs, dx, mask=dx_mask)

@triton.autotune(
    configs=bwd_w_autotune_config(),
    key=["M", "N"],
)
@triton.jit
def fused_swiglu_bwd_w_kernel(
    dg_ptr,
    dfc_ptr,
    x_ptr,
    dw_g_ptr,
    dw_fc_ptr,
    LOCK_G,
    LOCK_FC,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    dtype = dw_g_ptr.type.element_ty
    pid = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    if ((pid_m * BLOCK_SIZE_M >= M) or (pid_n * BLOCK_SIZE_N >= N)) or (
        pid_k * BLOCK_SIZE_K >= K
    ):
        return

    offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offset_dyn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))

    accumulator_dw_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator_dw_fc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # for k in range(K, 0, -BLOCK_SIZE_K * SPLIT_K):
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offset_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        x_ptrs = x_ptr + (offset_xm[:, None] + offset_k[None, :] * M)
        x_mask = (offset_xm[:, None] < M) & (offset_k[None, :] < K)
        dg_ptrs = dg_ptr + (offset_k[:, None] * N + offset_dyn[None, :])
        dfc_ptrs = dfc_ptr + (offset_k[:, None] * N + offset_dyn[None, :])
        dg_fc_mask = (offset_k[:, None] < K) & (offset_dyn[None, :] < N)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        dg = tl.load(dg_ptrs, mask=dg_fc_mask, other=0.0)
        dfc = tl.load(dfc_ptrs, mask=dg_fc_mask, other=0.0)

        # bwd w_gate
        accumulator_dw_g = tl.dot(x, dg, accumulator_dw_g)
        # bwd w_fc
        accumulator_dw_fc = tl.dot(x, dfc, accumulator_dw_fc)
    dw_g = accumulator_dw_g.to(dtype)
    dw_fc = accumulator_dw_fc.to(dtype)

    offset_dwm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_dwn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw_g_ptrs = dw_g_ptr + offset_dwm[:, None] * N + offset_dwn[None, :]
    dw_fc_ptrs = dw_fc_ptr + offset_dwm[:, None] * N + offset_dwn[None, :]
    dw_mask = (offset_dwm[:, None] < M) & (offset_dwn[None, :] < N)
    tl.store(dw_g_ptrs, dw_g, mask=dw_mask)
    tl.store(dw_fc_ptrs, dw_fc, mask=dw_mask)
    # if SPLIT_K == 1:
    #     tl.store(dw_g_ptrs, dw_g, mask=dw_mask)
    #     tl.store(dw_fc_ptrs, dw_fc, mask=dw_mask)
    # else:
    #     atomic_store(dw_g_ptrs, dw_g, dw_mask, LOCK_G, SPLIT_K)
    #     atomic_store(dw_fc_ptrs, dw_fc, dw_mask, LOCK_FC, SPLIT_K)


# x: [total_len, dim]
# w_g: [in_dim, out_dim]
# w_fc: [in_dim, out_dim]
# b_g: [out_dim]
# b_fc: [out_dim]
# y: [total_len, out_dim]
class FusedSwiglu(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w_g, w_fc, b_g, b_fc, is_training=True, is_recompute=False):
        # Check constraints.
        assert w_g.shape == w_fc.shape
        assert b_g.shape == b_fc.shape
        assert x.shape[1] == w_g.shape[0], "Incompatible dimensions"
        assert b_g.shape[0] == w_g.shape[1]
        assert (
            x.is_contiguous() and w_g.is_contiguous() and w_fc.is_contiguous()
        ), "Tensors must be contiguous"
        assert x.dtype == w_g.dtype == w_fc.dtype and x.dtype in [
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ]
        total_len, in_dim = x.shape
        out_dim = w_g.shape[1]
        # Allocates output.
        if is_recompute:
            g = x.new_empty(1)
            fc = x.new_empty(1)
        else:
            g = x.new_empty(total_len, out_dim)
            fc = x.new_empty(total_len, out_dim)
        y = x.new_empty(total_len, out_dim)
        grid = lambda META: (
            triton.cdiv(total_len, META["BLOCK_SIZE_M"])
            * triton.cdiv(out_dim, META["BLOCK_SIZE_N"]),
        )
        fused_swiglu_fwd_kernel[grid](
            x,
            w_g,
            w_fc,
            b_g,
            b_fc,
            y,
            g,
            fc,
            total_len,
            out_dim,
            in_dim,
            IS_TRAINING=is_training and not is_recompute,
            enable_auto_bind_sub_block=True,
            enable_flatten=False,
            set_workspace_multibuffer = 2,
            sync_solver=True,
            limit_auto_multi_buffer_of_local_buffer="no-limit",
            multibuffer=True,
        )

        ctx.is_recompute = is_recompute
        ctx.save_for_backward(x, w_g, w_fc, b_g, b_fc, g, fc)
        return y

    @staticmethod
    def backward(ctx, dy):
        device = dy.device
        is_recompute = ctx.is_recompute
        x, w_g, w_fc, b_g, b_fc, g, fc = ctx.saved_tensors
        total_len, out_dim = dy.shape
        in_dim, out_dim = w_g.shape
        dx = torch.empty_like(x)
        dw_g = torch.zeros_like(w_g)
        dw_fc = torch.zeros_like(w_fc)
        db_g = torch.zeros(out_dim, dtype=dy.dtype, device=device)
        db_fc = torch.zeros(out_dim, dtype=dy.dtype, device=device)
        dg = torch.empty_like(g)
        dfc = torch.empty_like(fc)
        if is_recompute:
            dg = dy.new_empty(total_len, out_dim)
            dfc = dy.new_empty(total_len, out_dim)
            # recompute g
            g = torch.nn.functional.linear(x, w_g.T, b_g)
            # recompute fc1
            fc = torch.nn.functional.linear(x, w_fc.T, b_fc)
        # dg & dfc & db_g & db_fc backward

        grid = lambda META: (triton.cdiv(out_dim, META["BLOCK_SIZE_N"]),)
        fused_swiglu_bwd_b_kernel[grid](
            dy,
            g,
            fc,
            dg,
            dfc,
            db_g,
            db_fc,
            total_len,
            out_dim,
            enable_auto_bind_sub_block=False,
            enable_flatten=True,
            multibuffer=True,
        )
        # x backward
        # M: total_len, N: in_dim, K: out_dim (reduce_axis)
        grid = lambda META: (
            triton.cdiv(total_len, META["BLOCK_SIZE_M"])
            * triton.cdiv(in_dim, META["BLOCK_SIZE_N"]),
        )
        fused_swiglu_bwd_x_kernel[grid](
            dg,
            dfc,
            w_g,
            w_fc,
            dx,
            total_len,
            in_dim,
            out_dim,
            enable_auto_bind_sub_block=True,
            enable_flatten=False,
            set_workspace_multibuffer=2,
            sync_solver=True,
            limit_auto_multi_buffer_of_local_buffer="no-limit",
            multibuffer=True,
        )
        # weight backward
        # M: in_dim, N: out_dim, K: total_len (reduce_axis)
        # allocate locks for split-k
        lock_g = torch.zeros(
            32 * 1024, dtype=torch.int32, device=device
        )
        lock_fc = torch.zeros(
            32 * 1024, dtype=torch.int32, device=device
        )
        grid = lambda META: (
            triton.cdiv(in_dim, META["BLOCK_SIZE_M"])
            * triton.cdiv(out_dim, META["BLOCK_SIZE_N"]),
            META["SPLIT_K"],
        )
        fused_swiglu_bwd_w_kernel[grid](
            dg,
            dfc,
            x,
            dw_g,
            dw_fc,
            lock_g,
            lock_fc,
            in_dim,
            out_dim,
            total_len,
            enable_auto_bind_sub_block=True,
            enable_flatten=False,
            set_workspace_multibuffer=2,
            sync_solver=True,
            limit_auto_multi_buffer_of_local_buffer="no-limit",
            multibuffer=True,
        )

        return dx, dw_g, dw_fc, db_g, db_fc, None, None

def third_part_cmp(y, y_gpu, ref, name):
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(y.to("cpu"), ref)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(y_gpu.to("cpu"), ref)
    print(
        f"{name} 最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}")


if __name__ == "__main__":
    dtype = torch.bfloat16
    DEVICE = torch.device("npu")
    torch.manual_seed(1024)

    # >1 Load GPU INPUTDATA
    x = torch.load("/home/tsz/Code/dump-swiglu-200k/x.pt", map_location=torch.device('cpu')).detach().requires_grad_(True)
    w_g = torch.load("/home/tsz/Code/dump-swiglu-200k/w_g.pt", map_location=torch.device('cpu')).detach().requires_grad_(True)
    w_fc = torch.load("/home/tsz/Code/dump-swiglu-200k/w_fc.pt", map_location=torch.device('cpu')).detach().requires_grad_(True)
    b_g = torch.load("/home/tsz/Code/dump-swiglu-200k/b_g.pt", map_location=torch.device('cpu')).detach().requires_grad_(True)
    b_fc = torch.load("/home/tsz/Code/dump-swiglu-200k/b_fc.pt", map_location=torch.device('cpu')).detach().requires_grad_(True)
    dy = torch.load("/home/tsz/Code/dump-swiglu-200k/dy.pt", map_location=torch.device('cpu'))
    x.grad = None
    w_g.grad = None
    w_fc.grad = None
    b_g.grad = None
    b_fc.grad = None

    # 2> FP32 cpu 高精度计算
    x_cpu, w_g_cpu, w_fc_cpu, b_g_cpu, b_fc_cpu = (x.detach().clone().to(torch.float32).to("cpu").requires_grad_(True),
                                                   w_g.detach().clone().to(torch.float32).to("cpu").requires_grad_(True),
                                                   w_fc.detach().clone().to(torch.float32).to("cpu").requires_grad_(True),
                                                   b_g.detach().clone().to(torch.float32).to("cpu").requires_grad_(True),
                                                   b_fc.detach().clone().to(torch.float32).to("cpu").requires_grad_(True))
    dy_cpu = dy.detach().clone().to(torch.float32).to("cpu")

    ref = naive_torch_swiglu(x_cpu, w_g_cpu, w_fc_cpu, b_g_cpu, b_fc_cpu)
    ref.backward(dy_cpu)
    torch_dx = x_cpu.grad.clone()
    torch_dw_g = w_g_cpu.grad.clone()
    torch_dw_fc = w_fc_cpu.grad.clone()
    torch_db_g = b_g_cpu.grad.clone()
    torch_db_fc = b_fc_cpu.grad.clone()

    # 3> NPU BF16 计算
    x = x.to(DEVICE).to(dtype).detach().requires_grad_(True)
    w_g = w_g.to(DEVICE).to(dtype).detach().requires_grad_(True)
    w_fc = w_fc.to(DEVICE).to(dtype).detach().requires_grad_(True)
    b_g = b_g.to(DEVICE).to(dtype).detach().requires_grad_(True)
    b_fc = b_fc.to(DEVICE).to(dtype).detach().requires_grad_(True)
    dy = dy.to(DEVICE).detach().to(dtype)

    y = FusedSwiglu.apply(x, w_g, w_fc, b_g, b_fc, True, False)

    y.backward(dy)
    triton_dx, x.grad = x.grad.clone(), None
    triton_dw_g, w_g.grad = w_g.grad.clone(), None
    triton_dw_fc, w_fc.grad = w_fc.grad.clone(), None
    triton_db_g, b_g.grad = b_g.grad.clone(), None
    triton_db_fc, b_fc.grad = b_fc.grad.clone(), None

    # # 4> GPU BF16 Load
    y_gpu = torch.load("/home/tsz/Code/dump-swiglu-200k/y.pt", map_location=torch.device('cpu')).to(DEVICE)
    dx_gpu = torch.load("/home/tsz/Code/dump-swiglu-200k/triton_dx.pt", map_location=torch.device('cpu')).to(DEVICE)
    dw_g_gpu = torch.load("/home/tsz/Code/dump-swiglu-200k/triton_dw_g.pt", map_location=torch.device('cpu')).to(DEVICE)
    dw_fc_gpu = torch.load("/home/tsz/Code/dump-swiglu-200k/triton_dw_fc.pt", map_location=torch.device('cpu')).to(
        DEVICE)
    db_g_gpu = torch.load("/home/tsz/Code/dump-swiglu-200k/triton_db_g.pt", map_location=torch.device('cpu')).to(DEVICE)
    db_fc_gpu = torch.load("/home/tsz/Code/dump-swiglu-200k/triton_db_fc.pt", map_location=torch.device('cpu')).to(
        DEVICE)

    third_part_cmp(y, y_gpu, ref, "y")
    third_part_cmp(triton_dx, dx_gpu, torch_dx, "dx")
    third_part_cmp(triton_dw_g, dw_g_gpu, torch_dw_g, "dw_g")
    third_part_cmp(triton_dw_fc, dw_fc_gpu, torch_dw_fc, "dw_fc")
    third_part_cmp(triton_db_g, db_g_gpu, torch_db_g, "db_gy")
    third_part_cmp(triton_db_fc, db_fc_gpu, torch_db_fc, "db_fc")

    atol = 1e-2
    rtol = 1e-2
    if torch.allclose(y.to("cpu"), ref.to(dtype), atol=atol, rtol=rtol):
        print("✅ [Fwd]Triton and Torch match")
    else:
        print("❌ [Fwd]Triton and Torch differ")
    if torch.allclose(triton_dx.to("cpu"), torch_dx.to(dtype), atol=atol, rtol=rtol):
        print("✅ [Bwd X]Triton and Torch match")
    else:
        print("❌ [Bwd X]Triton and Torch differ")
    if torch.allclose(triton_dw_g.to("cpu"), torch_dw_g.to(dtype), atol=atol, rtol=rtol):
        print("✅ [Bwd WG]Triton and Torch match")
    else:
        print("❌ [Bwd WG]Triton and Torch differ")
    if torch.allclose(triton_dw_fc.to("cpu"), torch_dw_fc.to(dtype), atol=atol, rtol=rtol):
        print("✅ [Bwd WFC]Triton and Torch match")
    else:
        print("❌ [Bwd WFC]Triton and Torch differ")
    if torch.allclose(triton_db_g.to("cpu"), torch_db_g.to(dtype), atol=atol, rtol=rtol):
        print("✅ [Bwd BG]Triton and Torch match")
    else:
        print("❌ [Bwd BG]Triton and Torch differ")
    if torch.allclose(triton_db_fc.to("cpu"), torch_db_fc.to(dtype), atol=atol, rtol=rtol):
        print("✅ [Bwd BFC]Triton and Torch match")
    else:
        print("❌ [Bwd BFC]Triton and Torch differ")

    assert torch.allclose(y.to("cpu"), ref.to(dtype), atol=atol, rtol=rtol), "[Fwd] Triton 和 Torch 前向结果不匹配"
    assert torch.allclose(triton_dx.to("cpu"), torch_dx.to(dtype), atol=atol, rtol=rtol), "[Bwd X] Triton 和 Torch 梯度不匹配"
    assert torch.allclose(triton_dw_g.to("cpu"), torch_dw_g.to(dtype), atol=atol, rtol=rtol), "[Bwd WG] Triton 和 Torch 梯度不匹配"
    assert torch.allclose(triton_dw_fc.to("cpu"), torch_dw_fc.to(dtype), atol=atol, rtol=rtol), "[Bwd WFC] Triton 和 Torch 梯度不匹配"
    assert torch.allclose(triton_db_g.to("cpu"), torch_db_g.to(dtype), atol=atol, rtol=rtol), "[Bwd BG] Triton 和 Torch 梯度不匹配"
    assert torch.allclose(triton_db_fc.to("cpu"), torch_db_fc.to(dtype), atol=atol, rtol=rtol), "[Bwd BFC] Triton 和 Torch 梯度不匹配"

    experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
        )

    with torch_npu.profiler.profile(
                activities=[  # torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
                with_stack=False,  # 采集torch 算子的函数调用栈的开关，该参数选填，默认关闭
                record_shapes=False,  # 采集torch 算子的input shape和input type的开关，该参数选填，默认关闭
                profile_memory=False,  # 采集memory相关数据的开关，该参数选填，默认关闭
                schedule=torch_npu.profiler.schedule(wait=1,
                                                    warmup=1,
                                                    active=30,
                                                    repeat=1,
                                                    skip_first=1),
                # warmup默认为0，老版本torch_npu包该参数为必填项
                experimental_config=experimental_config,  # 该参数选填，默认为Level0
                # 产生的profling文件的位置
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir")
        ) as prof:
            # prof.start()
            for i in range(30):
                # fn_triton(*args)
                # 这里填实际的上板运算
                y = FusedSwiglu.apply(x, w_g, w_fc, b_g, b_fc, True, False)

                torch.npu.synchronize()  # 确保 kernel 真正执行完
                prof.step()

                dy = torch.randn_like(y)
                y.backward(dy)
                triton_dx, x.grad = x.grad.clone(), None
   
print(f"fused_swiglu_fwd_kernel Best config: {fused_swiglu_fwd_kernel.best_config}") 
print(f"fused_swiglu_bwd_b_kernel Best config: {fused_swiglu_bwd_b_kernel.best_config}")    
print(f"fused_swiglu_bwd_x_kernel Best config: {fused_swiglu_bwd_x_kernel.best_config}") 
print(f"fused_swiglu_bwd_w_kernel Best config: {fused_swiglu_bwd_w_kernel.best_config}") 
