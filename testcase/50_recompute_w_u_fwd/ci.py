"""Standalone tests for ``recompute_w_u_fwd``.

Only the matching standalone operator module is imported.  The reference
implementations and numerical comparison helpers are included below.
"""

import logging
import os
import unittest.mock
import warnings

import pytest
import torch
import torch_npu

from ci_wy_fast import recompute_w_u_fwd


DEVICE = "npu"
MIN_ERR = 1e-7
FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"
logger = logging.getLogger(__name__)


def debug_print(*args, **kwargs):
    print("[DEBUG]: ", *args, **kwargs)


def naive_recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor | None = None,
    use_exp2: bool = False,
):
    B, T, H, K = k.shape
    HV = v.shape[2]
    V = v.shape[3]
    BT = A.shape[-1]

    w = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype)
    u = torch.empty(B, T, HV, V, device=v.device, dtype=v.dtype)
    NT = (T + BT - 1) // BT

    for i_t in range(NT):
        start = i_t * BT
        end = min((i_t + 1) * BT, T)
        A_chunk = A[:, start:end]
        beta_chunk = beta[:, start:end]

        for b in range(B):
            for h in range(HV):
                A_h = A_chunk[b, :, h, : end - start]
                beta_h = beta_chunk[b, :, h]
                k_idx = h // (HV // H) if HV > H else h
                k_h = k[b, start:end, k_idx]
                v_h = v[b, start:end, h]
                k_beta = k_h * beta_h[:, None]
                v_beta = (v_h * beta_h[:, None]).to(v_h.dtype)

                if g is not None:
                    g_h = g[b, start:end, h]
                    if use_exp2:
                        g_factor = torch.exp2(g_h)
                    else:
                        g_factor = torch.exp(g_h)
                    k_beta = k_beta * g_factor[:, None]

                w[b, start:end, h] = torch.matmul(
                    A_h,
                    k_beta.to(k_h.dtype),
                )
                u[b, start:end, h] = torch.matmul(A_h, v_beta)

    return w, u


def naive_recompute_w_u_fwd_high(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor | None = None,
    use_exp2: bool = False,
):
    B, T, H, K = k.shape
    HV = v.shape[2]
    V = v.shape[3]
    BT = A.shape[-1]

    k = k.float()
    v = v.float()
    beta = beta.float()
    A = A.float()
    if g is not None:
        g = g.float()

    w = torch.empty(B, T, HV, K, device=k.device, dtype=torch.float32)
    u = torch.empty(B, T, HV, V, device=v.device, dtype=torch.float32)
    NT = (T + BT - 1) // BT

    for i_t in range(NT):
        start = i_t * BT
        end = min((i_t + 1) * BT, T)
        A_chunk = A[:, start:end]
        beta_chunk = beta[:, start:end]

        for b in range(B):
            for h in range(HV):
                A_h = A_chunk[b, :, h, : end - start]
                beta_h = beta_chunk[b, :, h]
                k_idx = h // (HV // H) if HV > H else h
                k_h = k[b, start:end, k_idx]
                v_h = v[b, start:end, h]
                k_beta = k_h * beta_h[:, None]
                v_beta = v_h * beta_h[:, None]

                if g is not None:
                    g_h = g[b, start:end, h]
                    if use_exp2:
                        g_factor = torch.exp2(g_h)
                    else:
                        g_factor = torch.exp(g_h)
                    k_beta = k_beta * g_factor[:, None]

                w[b, start:end, h] = torch.matmul(A_h, k_beta)
                u[b, start:end, h] = torch.matmul(A_h, v_beta)

    return w, u


def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (
        (x.detach() - y.detach())
        .flatten()
        .square()
        .mean()
        .sqrt()
        .item()
    )
    base = x.detach().flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(
    prefix,
    ref,
    tri,
    ratio,
    warning=False,
    err_atol=1e-6,
):
    abs_atol = get_abs_err(ref, tri)
    msg = (
        f"{prefix:>16} diff: {abs_atol:.6f} "
        f"ratio: {get_err_ratio(ref, tri):.6f}"
    )
    logger.info(msg)
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return
    assert not torch.isnan(ref).any(), f"{prefix}: NaN detected in ref"
    assert not torch.isnan(tri).any(), f"{prefix}: NaN detected in tri"
    if warning or (
        FLA_CI_ENV and (error_rate < 0.01 or abs_atol <= 0.3)
    ):
        if error_rate > ratio:
            warnings.warn(msg, stacklevel=2)
    else:
        assert error_rate < ratio, msg


def get_err_threshold(dtype: torch.dtype):
    err_threshold = 0
    if dtype == torch.bfloat16:
        err_threshold = 2**-8
    if dtype == torch.float16:
        err_threshold = 2**-11
    if dtype == torch.float32:
        err_threshold = 2**-14
    return err_threshold


def get_mare(golden: torch.Tensor, actual: torch.Tensor):
    golden = golden.to(torch.float32)
    abs_error = torch.abs(
        actual.to(torch.float32) - golden
    ) / (torch.abs(golden) + MIN_ERR)
    return torch.max(abs_error.flatten())


def get_mere(golden: torch.Tensor, actual: torch.Tensor):
    golden = golden.to(torch.float32)
    abs_error = torch.abs(
        actual.to(torch.float32) - golden
    ) / (torch.abs(golden) + MIN_ERR)
    return torch.mean(abs_error)


def get_rmse(golden: torch.Tensor, actual: torch.Tensor):
    golden = golden.to(torch.float32)
    sqr_err = torch.pow(actual.to(torch.float32) - golden, 2)
    return torch.sqrt(torch.mean(sqr_err))


def compare_cv(
    golden: torch.Tensor,
    gpu: torch.Tensor,
    actual: torch.Tensor,
    mare_threshold=2,
    mere_threshold=1.2,
    rmse_threshold=1.2,
):
    err_threshold = get_err_threshold(actual.dtype)
    print(f"err_threshold:{err_threshold}")
    mare_npu = get_mare(golden, actual)
    mare_gpu = get_mare(golden, gpu)
    mere_npu = get_mere(golden, actual)
    mere_gpu = get_mere(golden, gpu)
    rmse_npu = get_rmse(golden, actual)
    rmse_gpu = get_rmse(golden, gpu)

    mare_rate = mare_npu / max(mare_gpu, err_threshold)
    mere_rate = mere_npu / max(mere_gpu, err_threshold)
    rmse_rate = rmse_npu / max(rmse_gpu, err_threshold)
    result = (
        mare_rate < mare_threshold
        and mere_rate < mere_threshold
        and rmse_rate < rmse_threshold
    )

    print(f"mare_npu:{mare_npu} mare_gpu:{mare_gpu}")
    print(f"mere_npu:{mere_npu} mere_gpu:{mere_gpu}")
    print(f"rmse_npu:{rmse_npu} rmse_gpu:{rmse_gpu}")
    print(f"MARE:{mare_rate} MERE:{mere_rate} RMSE:{rmse_rate}")
    print(f"new golden cv result:{result}")
    return result


TEST_CASES = [
    (1, 1024, 32, 32, 128, 128, 64, torch.bfloat16),
    (4, 1024, 32, 32, 128, 128, 64, torch.bfloat16),
    (16, 1024, 32, 32, 128, 128, 64, torch.bfloat16),
    (1, 8192, 32, 32, 128, 128, 64, torch.bfloat16),
    (4, 8192, 32, 32, 128, 128, 64, torch.bfloat16),
    (1, 16384, 32, 32, 128, 128, 64, torch.bfloat16),
]


@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "K", "V", "BT", "dtype"),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-HV{}-K{}-V{}-BT{}-{}".format(*test),
        )
        for test in TEST_CASES
    ],
)
@unittest.mock.patch.dict(
    os.environ,
    {"TRITON_ALL_BLOCKS_PARALLEL": "1"},
)
def test_recompute_w_u_fwd(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    BT: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, dtype=dtype, device=DEVICE)
    v = torch.randn(B, T, HV, V, dtype=dtype, device=DEVICE)
    beta = torch.rand(B, T, HV, dtype=dtype, device=DEVICE).sigmoid()
    g = torch.randn(B, T, HV, dtype=dtype, device=DEVICE)
    A = torch.randn(B, T, HV, BT, dtype=dtype, device=DEVICE)

    ref_w, ref_u = naive_recompute_w_u_fwd(
        k,
        v,
        beta,
        A,
        g,
        use_exp2=False,
    )
    tri_w, tri_u = recompute_w_u_fwd(
        k,
        v,
        beta,
        A,
        g,
        use_exp2=False,
    )
    assert_close("w", ref_w, tri_w, 0.005)
    assert_close("u", ref_u, tri_u, 0.005)


@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "K", "V", "BT", "dtype"),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-HV{}-K{}-V{}-BT{}-{}".format(*test),
        )
        for test in TEST_CASES
    ],
)
@unittest.mock.patch.dict(
    os.environ,
    {"TRITON_ALL_BLOCKS_PARALLEL": "1"},
)
def test_recompute_w_u_fwd_cross_platform_acc(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    BT: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    k = torch.randn(B, T, H, K, dtype=dtype, device=DEVICE)
    v = torch.randn(B, T, HV, V, dtype=dtype, device=DEVICE)
    beta = torch.rand(B, T, HV, dtype=dtype, device=DEVICE).sigmoid()
    g = torch.randn(B, T, HV, dtype=dtype, device=DEVICE)
    A = torch.randn(B, T, HV, BT, dtype=dtype, device=DEVICE)

    ref_w, ref_u = naive_recompute_w_u_fwd_high(
        k,
        v,
        beta,
        A,
        g,
        use_exp2=False,
    )
    debug_print("golden done.")
    sim_w, sim_u = naive_recompute_w_u_fwd(
        k,
        v,
        beta,
        A,
        g,
        use_exp2=False,
    )
    debug_print("sim done.")
    tri_w, tri_u = recompute_w_u_fwd(
        k,
        v,
        beta,
        A,
        g,
        use_exp2=False,
    )
    debug_print("triton done.")

    assert compare_cv(
        ref_w,
        sim_w,
        tri_w,
        mare_threshold=10,
        mere_threshold=2.0,
        rmse_threshold=2.0,
    )
    assert compare_cv(
        ref_u,
        sim_u,
        tri_u,
        mare_threshold=10,
        mere_threshold=2.0,
        rmse_threshold=2.0,
    )