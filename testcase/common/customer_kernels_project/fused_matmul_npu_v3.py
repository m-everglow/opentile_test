import torch
import triton
import triton.language as tl
# from maybe_triton_jit import maybe_triton_jit
import torch_npu

import numpy as np

eval_standard = {
    torch.float32: {
        "rtol": 1e-6,
        "small_value": 1e-6,
        "small_value_atol": 1e-9,
        "etol": 1e-4,
    },
    torch.float16: {
        "rtol": 1e-3,
        "small_value": 1e-3,
        "small_value_atol": 1e-5,
        "etol": 1e-3,
    },
    torch.bfloat16: {
        "rtol": 4e-3,
        "small_value": 1e-3,
        "small_value_atol": 1e-5,
        "etol": 1e-3,
    },
}


def benchmark_compare_close(gold: torch.Tensor, act: torch.Tensor, std: torch.tensor):
    assert act.dtype == std.dtype, "standard tensor's dtype must equal to actual tensor's dtype!"
    if act.dtype == torch.float16 or act.dtype == torch.float32 or act.dtype == torch.bfloat16:
        assert gold.dtype == torch.float32, "golden should be f32"
        assert not (torch.isnan(act).any() or torch.isinf(act).any()), "actual tensor can not have 'inf' or 'nan'"

    gold = gold.cpu()
    act = act.cpu()
    std = std.cpu()

    eps = eval_standard[act.dtype]['small_value']
    atol = eval_standard[act.dtype]['small_value_atol']

    mask = torch.abs(gold) <= eps
    small_count = mask.sum().item()

    def calculate_relative_errors_except_small(tensor):
        re = torch.abs(gold - tensor) / torch.abs(gold)
        return torch.where(mask, 0, re)

    act_re = calculate_relative_errors_except_small(act)
    std_re = calculate_relative_errors_except_small(std)
    act_ae = torch.abs(gold - std)
    std_ae = torch.abs(gold - std)

    # 小值域的定义为golden小于某个阈值 eps
    act_small_error_count = (mask & (act_ae > atol)).sum().item()
    std_small_error_count = (mask & (std_ae > atol)).sum().item()
    act_total = act.numel()
    std_total = std.numel()

    act_small_error_ratio = act_small_error_count / act_total
    std_small_error_ratio = std_small_error_count / std_total

    def calculate_rmse(tensor):
        dlt2 = (tensor - gold) ** 2
        dlt2_except_small_mean = torch.where(mask, 0, dlt2).sum() / small_count
        return torch.sqrt(dlt2_except_small_mean)

    act_rmse = calculate_rmse(act)
    std_rmse = calculate_rmse(std)

    print(f"act_re.max = {act_re.max()}, std_re.max = {std_re.max()}, limit ratio = 10")
    print(f"act_re.sum = {act_re.sum()}, std_re.sum = {std_re.sum()}, limit_ratio = 2")
    print(
        f"act_small_error_ratio = {act_small_error_ratio}, std_small_error_ratio = {std_small_error_ratio}, limit_ratio = 2")
    print(f"act_rmse = {act_rmse}, std_rmse = {std_rmse}, limit_ratio = 2")

    # 条件 1：actual 与 golden 相对误差最大值超过 10 倍 standard 与 golden 相对误差最大值
    assert act_re.max() <= 10 * std_re.max(), "actual re max > stdandard re max's 10 times"

    # 条件 2：actual 与 golden 相对误差均值超过 2 倍 standard 与 golden 相对误差均值
    assert act_re.sum() <= 2 * std_re.sum(), "actual re sum > stdandard re sum's 2 times"

    # 条件 3：actual 小值域 ERROR 占比超过 standard 的两倍
    assert act_small_error_ratio <= 2 * std_small_error_ratio, "act_small_error_ratio > std_small_error_ratio 's 2 times"

    # 条件 4：actual 均方根误差差于 standard 的两倍
    assert act_rmse <= 2 * std_rmse, "act_rmse > std_rmse 's 2 times"

    return True


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
    configs = [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
            },
            num_stages=s,
            num_warps=w,
        )
        for BM in [128]
        for BN in [128]
        for BK in [128]
        for s in [3, 4]
        for w in [4, 8]
    ]
    return configs


def bwd_b_autotune_config():
    configs = [
        triton.Config(
            {"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN}, num_stages=s, num_warps=w
        )
        for BM in [512]  # [128, 256, 512, 1024]
        for BN in [32]  # [16, 32]
        for s in [3]  # [3, 4]
        for w in [4]  # [4, 8]
    ]
    return configs


def bwd_x_autotune_config():
    configs = [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
            },
            num_stages=s,
            num_warps=w,
        )
        for BM in [43, 64, 128]
        for BN in [64, 128]
        for BK in [1024, 32, 64]
        for s in [3, 4]
        for w in [4, 8]
    ]
    return configs


def bwd_w_autotune_config():
    configs = [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "SPLIT_K": SK,
                "GROUP_SIZE_M": 8,
            },
            num_stages=s,
            # num_warps=w,
        )
        for BM in [86, 128]
        for BN in [128]
        for BK in [128, 256]  # [32, 64]
        for SK in [8, 16]
        for s in [3, 4]
        # for w in [4, 8]
    ]
    return configs


@triton.autotune(
    configs=fwd_autotune_config(),
    key=["N", "K"],
)
@triton.jit
def fused_matmul_fwd_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        y_ptr,
        M,
        N,
        K,
        HAS_BIAS: tl.constexpr,
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

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs_base = x_ptr + (offs_am[:, None] * K + offs_k[None, :])
    b_ptrs_base = w_ptr + (offs_k[:, None] * N + offs_bn[None, :])
    msk_m = offs_am < M
    msk_n = offs_bn < N

    if HAS_BIAS:
        offset_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
        b_ptrs = b_ptr + offset_wn
        b = tl.load(b_ptrs, mask=offset_wn < N, other=0.0)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x_ptrs = a_ptrs_base + k * BLOCK_SIZE_K
        w_ptrs = b_ptrs_base + k * BLOCK_SIZE_K * N
        x = tl.load(
            x_ptrs,
            mask=msk_m[:, None] and (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        w = tl.load(
            w_ptrs,
            mask=msk_n[None, :] and (offs_k[:, None] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        accumulator = tl.dot(x, w, accumulator)

    if HAS_BIAS:
        accumulator += tl.broadcast_to(b[None, :], [BLOCK_SIZE_M, BLOCK_SIZE_N])

    y = accumulator.to(dtype)

    offset_ym = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    y_ptrs = y_ptr + offset_ym[:, None] * N + offset_yn[None, :]
    y_mask = (offset_ym[:, None] < M) & (offset_yn[None, :] < N)
    tl.store(y_ptrs, y, mask=y_mask)


@triton.autotune(
    configs=bwd_b_autotune_config(),
    key=["N"],
)
@triton.jit
def fused_matmul_bwd_b_kernel(
        dy_ptr,
        db_ptr,
        M,
        N,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
):
    col_idx = tl.program_id(axis=0)
    col_off = col_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_off = tl.arange(0, BLOCK_SIZE_M)
    dy_ptrs = dy_ptr + (row_off[None, :] * N + col_off[:, None])
    sum_b = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    for row_idx in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        mask = (row_off[None, :] < M - row_idx * BLOCK_SIZE_M) & (col_off[:, None] < N)
        dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
        sum_b += dy
        dy_ptrs += BLOCK_SIZE_M * N
    tl.store(db_ptr + col_off, tl.sum(sum_b, 1), mask=col_off < N)


@triton.autotune(
    configs=bwd_x_autotune_config(),
    key=["N", "K"],
)
@triton.jit
def fused_matmul_bwd_x_kernel(
        dy_ptr,
        w_ptr,
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
    accumulator_dx = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offset_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        dy_ptrs = dy_ptr + (offset_dym[:, None] * K + offset_k[None, :])
        dy_mask = (offset_dym[:, None] < M) & (offset_k[None, :] < K)

        w_ptrs = w_ptr + (offset_k[:, None] + offset_wn[None, :] * K)
        w_mask = (offset_k[:, None] < K) & (offset_wn[None, :] < N)

        dy = tl.load(dy_ptrs, mask=dy_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        # bwd x
        accumulator_dx = tl.dot(dy, w, accumulator_dx)
    dx = accumulator_dx.to(dtype)

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
def fused_matmul_bwd_w_kernel(
        dy_ptr,
        x_ptr,
        dw_ptr,
        LOCK_W,
        M,
        N,
        K,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        SPLIT_K: tl.constexpr,
):
    dtype = dw_ptr.type.element_ty
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    if ((pid_m * BLOCK_SIZE_M >= M) or (pid_n * BLOCK_SIZE_N >= N)):
        return

    offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offset_dyn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    accumulator_dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for pid_k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offset_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        x_ptrs = x_ptr + (offset_xm[:, None] + offset_k[None, :] * M)
        dy_ptrs = dy_ptr + (offset_k[:, None] * N + offset_dyn[None, :])

        x_mask = (offset_xm[:, None] < M) & (offset_k[None, :] < K)
        dy_mask = (offset_k[:, None] < K) & (offset_dyn[None, :] < N)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        dy = tl.load(dy_ptrs, mask=dy_mask, other=0.0)
        # bwd w
        accumulator_dw = tl.dot(x, dy, accumulator_dw)
    dw = accumulator_dw.to(dtype)

    offset_dwm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_dwn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw_ptrs = dw_ptr + offset_dwm[:, None] * N + offset_dwn[None, :]
    dw_mask = (offset_dwm[:, None] < M) & (offset_dwn[None, :] < N)
    tl.store(dw_ptrs, dw, mask=dw_mask)


# x: [total_len, dim]
# w: [in_dim, out_dim]
# b: [out_dim]
# y: [total_len, out_dim]
class FusedMatmul(torch.autograd.Function):
    _lock_w = dict()

    @staticmethod
    def forward(ctx, x, w, b):
        has_bias = b is not None
        # Check constraints.
        assert x.shape[1] == w.shape[0], "Incompatible dimensions"
        if has_bias:
            assert b.shape[0] == w.shape[1]
        assert x.is_contiguous() and w.is_contiguous(), "Tensors must be contiguous"
        assert x.dtype == w.dtype, "Tensors must have the same dtype"
        assert x.dtype in [
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ], "Only float32, bfloat16 and float16 are supported"
        total_len, in_dim = x.shape
        out_dim = w.shape[1]
        # Allocates output.
        y = x.new_empty(total_len, out_dim)
        grid = lambda META: (
            triton.cdiv(total_len, META["BLOCK_SIZE_M"])
            * triton.cdiv(out_dim, META["BLOCK_SIZE_N"]),
        )
        fused_matmul_fwd_kernel[grid](
            x,
            w,
            b,
            y,
            total_len,
            out_dim,
            in_dim,
            HAS_BIAS=has_bias,
            enable_auto_bind_sub_block=True,
            set_workspace_multibuffer=2,
            sync_solver=True,
            limit_auto_multi_buffer_of_local_buffer="no-limit",
            multibuffer=True,
            enable_flatten=True,
        )

        ctx.save_for_backward(x, w)
        ctx.has_bias = has_bias
        return y

    @staticmethod
    def backward(ctx, dy):
        device = dy.device
        x, w = ctx.saved_tensors
        total_len, out_dim = dy.shape
        in_dim, out_dim = w.shape
        # print("Backward shapes - w:", w.shape)
        dx = torch.empty_like(x)
        dw = torch.zeros_like(w)
        db = None
        if ctx.has_bias:
            db = torch.zeros(out_dim, dtype=dy.dtype, device=device)
            # bias backward
            grid = lambda META: (triton.cdiv(out_dim, META["BLOCK_SIZE_N"]),)
            fused_matmul_bwd_b_kernel[grid](
                dy,
                db,
                total_len,
                out_dim,
                enable_auto_bind_sub_block=False,
            )
        # x backward
        # M: total_len, N: in_dim, K: out_dim (reduce_axis)
        grid = lambda META: (
            triton.cdiv(total_len, META["BLOCK_SIZE_M"])
            * triton.cdiv(in_dim, META["BLOCK_SIZE_N"]),
        )
        fused_matmul_bwd_x_kernel[grid](
            dy,
            w,
            dx,
            total_len,
            in_dim,
            out_dim,
            enable_auto_bind_sub_block=False,
        )
        # weight backward
        # M: in_dim, N: out_dim, K: total_len (reduce_axis)
        # allocate locks for split-k
        if device not in FusedMatmul._lock_w:
            FusedMatmul._lock_w[device] = torch.zeros(
                32 * 1024, dtype=torch.int32, device=device
            )
        lock_w = FusedMatmul._lock_w[device]
        grid = lambda META: (
            triton.cdiv(in_dim, META["BLOCK_SIZE_M"])
            * triton.cdiv(out_dim, META["BLOCK_SIZE_N"]),
        )
        # print(f"in_dim: {in_dim}, out_dim: {out_dim}, total_len: {total_len}")
        fused_matmul_bwd_w_kernel[grid](
            dy,
            x,
            dw,
            lock_w,
            in_dim,
            out_dim,
            total_len,
            enable_auto_bind_sub_block=False,
        )

        return dx, dw, db


if __name__ == "__main__":
    dtype = torch.float16

    x = torch.randn((256, 512), dtype=dtype, device="npu").requires_grad_()
    w = torch.randn((512, 1024), dtype=dtype, device="npu").requires_grad_()
    b = torch.randn((1024), dtype=dtype, device="npu").requires_grad_()
    # b = None
    dy = torch.randn((256, 1024), dtype=dtype, device="npu")

    # ====================精度测试============
    y = FusedMatmul.apply(x, w, b)

    y.backward(dy)
    triton_dx, x.grad = x.grad.clone(), None
    triton_dw, w.grad = w.grad.clone(), None
    triton_db = None
    if b is not None:
        triton_db, b.grad = b.grad.clone(), None

    ref = x @ w + (b if b is not None else 0)
    ref.backward(dy)
    torch_dx, x.grad = x.grad.clone(), None
    torch_dw, w.grad = w.grad.clone(), None

    torch_db = None
    if b is not None:
        torch_db, b.grad = b.grad.clone(), None

    atol = 1e-3
    rtol = 1e-2
    if torch.allclose(y, ref.to(torch.float16), atol=atol, rtol=rtol):
        print("✅ [Fwd]Triton and Torch match")
    else:
        print("❌ [Fwd]Triton and Torch differ")
    if torch.allclose(triton_dx, torch_dx.to(torch.float16), atol=atol, rtol=rtol):
        print("✅ [Bwd X]Triton and Torch match")
    else:
        print("❌ [Bwd X]Triton and Torch differ")

    if torch.allclose(triton_dw, torch_dw, atol=atol, rtol=rtol):
        print("✅ [Bwd W]Triton and Torch match")
    else:
        print("❌ [Bwd W]Triton and Torch differ")

    if b is not None:
        if torch.allclose(triton_db, torch_db.to(torch.float16), atol=atol, rtol=rtol):
            print("✅ [Bwd B]Triton and Torch match")
        else:
            print("❌ [Bwd B]Triton and Torch differ")
    
    # 适配用例自动化
    assert torch.allclose(y, ref.to(torch.float16), atol=atol, rtol=rtol), "[Fwd] Triton 和 Torch 前向结果不匹配"
    assert torch.allclose(triton_dx, torch_dx.to(torch.float16), atol=atol, rtol=rtol), "[Bwd X] Triton 和 Torch 梯度不匹配"
    assert torch.allclose(triton_dw, torch_dw, atol=atol, rtol=rtol), "[Bwd W] Triton 和 Torch 梯度不匹配"

    if b is not None:
        assert torch.allclose(triton_db, torch_db.to(torch.float16), atol=atol, rtol=rtol), "[Bwd B] Triton 和 Torch 梯度不匹配"

    # 单算子性能测试
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
    )
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            with_stack=False,
            record_shapes=False,
            profile_memory=False,
            schedule=torch_npu.profiler.schedule(wait=1,
                                                 warmup=1,
                                                 active=30,
                                                 repeat=10,
                                                 skip_first=1),
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir")
    ) as prof:
        for i in range(30):
            y = FusedMatmul.apply(x, w, b)
            y.backward(dy)
            torch_npu.npu.synchronize()
            prof.step()
        prof.stop()