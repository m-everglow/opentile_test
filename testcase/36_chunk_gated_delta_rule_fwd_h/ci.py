"""A5 end-to-end test for FLA ``chunk_gated_delta_rule_fwd_h``.

Production source:
  HighPriority50Operators/Q2TritonKernel-main/src/kernels/fla/ops/common/
  chunk_delta_h.py
  SHA256 3bf957707f9018335f19f3a45779aa0e1497723d87f6397f22ddfb29db5b64f9

Input and CPU-golden contract:
  HighPriority50Operators/Q2TritonKernel-main/tests/fla/
  test_chunk_gated_delta_rule_fwd_h.py
  SHA256 738959e533800fb44ab4d79d8d600ea3bb711327dbb24af0356ba339545b32c6

This runs the full CASES matrix from that test (10 shapes), via pytest
parametrize.  The production dispatcher selects
``chunk_gated_delta_rule_fwd_kernel_h_k128_blockdim128`` when K == 128.
This test freezes the original supported autotune choice BV=32,
num_warps=4 and num_stages=1, so the OpenTile entry requests block_dim=128.
The embedded Triton function body is AST-equivalent to the production body;
only heuristics/autotune decorators are omitted for explicit ASTSource
compilation.

All three public forward outputs are compared in full: h, v_new, and
final_state.  Inputs, CPU golden, BF16 dtype, shapes, seed, and the original
relative-error acceptance rule remain unchanged.
"""

import os

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_BACKEND", "opentile")
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")

import pytest
import pathlib

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

GOLDEN_DIR = pathlib.Path("/data/y00939135/test/testcase/36_chunk_gated_delta_rule_fwd_h/golden")
GOLDEN_DIR.mkdir(exist_ok=True)

def _npu_available():
    try:
        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


if not _npu_available():
    pytest.skip("torch_npu is available, but no NPU device is visible", allow_module_level=True)

torch.npu.set_device(int(os.environ.get("OPENTILE_TEST_DEVICE", "0")))

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402


BT = 64
BV = 32
INPUT_SCALE = 0.01
GATE_SCALE = 0.002
ERROR_RATIO = 0.02
ERROR_ATOL = 1e-6

CASES = [
    pytest.param(1, 1024, 32, 128, 128, id="B1-T1024-H32-K128-V128"),
    pytest.param(4, 1024, 32, 128, 128, id="B4-T1024-H32-K128-V128"),
    pytest.param(16, 1024, 32, 128, 128, id="B16-T1024-H32-K128-V128"),
    pytest.param(1, 8192, 32, 128, 128, id="B1-T8192-H32-K128-V128"),
    pytest.param(4, 8192, 32, 128, 128, id="B4-T8192-H32-K128-V128"),
    pytest.param(16, 8192, 8, 128, 128, id="B16-T8192-H8-K128-V128"),
    pytest.param(1, 131072, 4, 128, 128, id="B1-T131072-H4-K128-V128"),
    pytest.param(4, 131072, 2, 128, 128, id="B4-T131072-H2-K128-V128"),
    pytest.param(16, 131072, 2, 128, 128, id="B16-T131072-H2-K128-V128"),
    pytest.param(1, 16384, 4, 128, 128, id="B1-T16384-H4-K128-V128"),
]


@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


@triton.jit
def chunk_gated_delta_rule_fwd_kernel_h_k128_blockdim128(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    K = 128
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    if TRANSPOSE_STATE:
        b_h1 = tl.zeros([BV, 128], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([128, BV], dtype=tl.float32)

    h += (boh * HV + i_h).to(tl.int64) * K*V
    v += (bos * HV + i_h).to(tl.int64) * V
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * HV + i_h).to(tl.int64) * V

    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K*V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K*V

    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 128), (1, 0))
        else:
            p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (128, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        if TRANSPOSE_STATE:
            p_h1 = tl.make_block_ptr(h + i_t_int64 * HV*K*V, (V, K), (K, 1), (i_v * BV, 0), (BV, 128), (1, 0))
        else:
            p_h1 = tl.make_block_ptr(h + i_t_int64 * HV*K*V, (K, V), (V, 1), (0, i_v * BV), (128, BV), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))

        p_w1 = tl.make_block_ptr(w, (T, K), (HV*K, 1), (i_t * BT, 0), (BT, 128), (1, 0))
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        if TRANSPOSE_STATE:
            b_v = tl.dot(b_w1, tl.trans(b_h1).to(b_w1.dtype))
        else:
            b_v = tl.dot(b_w1, b_h1.to(b_w1.dtype))
        p_v = tl.make_block_ptr(v, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(v_new, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(tl.float32)
            p_g = tl.make_block_ptr(g + (bos * HV + i_h).to(tl.int64), (T,), (HV,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            if USE_EXP2:
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_scale = exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                b_g_scale = exp(b_g_last)
            b_h1 *= b_g_scale

        if USE_GK:
            o_k = tl.arange(0, 128)
            b_gk_last1 = tl.load(gk + (bos + last_idx) * HV*K + i_h * K + o_k).to(tl.float32)
            if TRANSPOSE_STATE:
                if USE_EXP2:
                    b_h1 *= exp2(b_gk_last1)[None, :]
                else:
                    b_h1 *= exp(b_gk_last1)[None, :]
            else:
                if USE_EXP2:
                    b_h1 *= exp2(b_gk_last1)[:, None]
                else:
                    b_h1 *= exp(b_gk_last1)[:, None]

        b_v = b_v.to(k.dtype.element_ty)

        p_k1 = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_t * BT), (128, BT), (0, 1))
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        if TRANSPOSE_STATE:
            b_h1 += tl.trans(tl.dot(b_k1, b_v))
        else:
            b_h1 += tl.dot(b_k1, b_v)

    if STORE_FINAL_STATE:
        if TRANSPOSE_STATE:
            p_ht1 = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 128), (1, 0))
        else:
            p_ht1 = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (128, BV), (1, 0))
        tl.store(p_ht1, b_h1.to(p_ht1.dtype.element_ty), boundary_check=(0, 1))


def _stage(message):
    print(f"[E2E_STAGE] op=chunk_gated_delta_rule_fwd_h {message}", flush=True)


def _device():
    return torch.device("npu", torch.npu.current_device())


def _chunk_local_cumsum_cpu(g):
    out = torch.empty_like(g, dtype=torch.float32)
    for start in range(0, g.shape[1], BT):
        end = min(start + BT, g.shape[1])
        out[:, start:end] = g[:, start:end].float().cumsum(dim=1)
    return out


def _make_inputs(B, T, H, K, V):
    generator = torch.Generator(device="cpu").manual_seed(42)
    k = torch.randn((B, T, H, K), dtype=torch.bfloat16, generator=generator) * INPUT_SCALE
    w = torch.randn((B, T, H, K), dtype=torch.bfloat16, generator=generator) * INPUT_SCALE
    u = torch.randn((B, T, H, V), dtype=torch.bfloat16, generator=generator) * INPUT_SCALE
    g_raw = torch.randn((B, T, H), dtype=torch.float32, generator=generator) * GATE_SCALE
    g = _chunk_local_cumsum_cpu(g_raw)
    return k, w, u, g


def _cpu_reference(k, w, u, g):
    B, T, H, K = k.shape
    V = u.shape[-1]
    nt = (T + BT - 1) // BT
    k_f32 = k.float()
    w_f32 = w.float()
    u_f32 = u.float()
    g_f32 = g.float()
    state = torch.zeros((B, H, K, V), dtype=torch.float32)
    h = torch.empty((B, nt, H, K, V), dtype=torch.bfloat16)
    v_new = torch.empty_like(u)

    for chunk_id, start in enumerate(range(0, T, BT)):
        end = min(start + BT, T)
        h[:, chunk_id] = state.to(torch.bfloat16)
        residual = u_f32[:, start:end] - torch.einsum(
            "bthk,bhkv->bthv",
            w_f32[:, start:end],
            state,
        )
        g_chunk = g_f32[:, start:end]
        last_g = g_chunk[:, -1]
        v_new[:, start:end] = residual.to(torch.bfloat16)
        state_v = residual * torch.exp2(last_g[:, None, :] - g_chunk)[..., None]
        state = state * torch.exp2(last_g)[..., None, None]
        state = state + torch.einsum(
            "bthk,bthv->bhkv",
            k_f32[:, start:end],
            state_v,
        )
    return h, v_new, state


def _compile_and_launch(k, w, u, g, B, H, HV, K, V, T):
    nt = (T + BT - 1) // BT
    h = torch.empty((B, nt, H, K, V), dtype=torch.bfloat16, device=k.device)
    v_new = torch.empty_like(u)
    final_state = torch.zeros((B, H, K, V), dtype=torch.float32, device=k.device)

    dummy_f32 = torch.zeros((1,), dtype=torch.float32, device=k.device)
    dummy_i64 = torch.zeros((1,), dtype=torch.int64, device=k.device)
    source = ASTSource(
        fn=chunk_gated_delta_rule_fwd_kernel_h_k128_blockdim128,
        signature={
            "k": "*bf16",
            "v": "*bf16",
            "w": "*bf16",
            "v_new": "*bf16",
            "g": "*fp32",
            "gk": "*fp32",
            "h": "*bf16",
            "h0": "*fp32",
            "ht": "*fp32",
            "cu_seqlens": "*i64",
            "chunk_offsets": "*i64",
            "T": "i32",
            "H": "constexpr",
            "HV": "constexpr",
            "V": "constexpr",
            "BT": "constexpr",
            "BV": "constexpr",
            "USE_G": "constexpr",
            "USE_GK": "constexpr",
            "USE_INITIAL_STATE": "constexpr",
            "STORE_FINAL_STATE": "constexpr",
            "SAVE_NEW_VALUE": "constexpr",
            "USE_EXP2": "constexpr",
            "TRANSPOSE_STATE": "constexpr",
            "IS_VARLEN": "constexpr",
        },
        constexprs={
            "H": H,
            "HV": HV,
            "V": V,
            "BT": BT,
            "BV": BV,
            "USE_G": True,
            "USE_GK": False,
            "USE_INITIAL_STATE": False,
            "STORE_FINAL_STATE": True,
            "SAVE_NEW_VALUE": True,
            "USE_EXP2": True,
            "TRANSPOSE_STATE": False,
            "IS_VARLEN": False,
        },
    )
    _stage(
        "compile_begin "
        "kernel=chunk_gated_delta_rule_fwd_kernel_h_k128_blockdim128 "
        "K=128 BV=32 num_warps=4 num_stages=1 expected_get_block_dim=128"
    )
    compiled = triton.compile(
        source,
        options={"num_warps": 4, "num_stages": 1},
    )
    _stage("compile_done expected_get_block_dim=128")
    grid_v = (V + BV - 1) // BV
    compiled[(grid_v, B * HV, 1)](
        k,
        u,
        w,
        v_new,
        g,
        dummy_f32,
        h,
        dummy_f32,
        final_state,
        dummy_i64,
        dummy_i64,
        T,
    )
    return h, v_new, final_state


def _compare(name, actual, expected):
    actual_f32 = actual.cpu().float().reshape(-1)
    expected_f32 = expected.float().reshape(-1)
    assert actual_f32.numel() == expected_f32.numel()
    assert torch.isfinite(actual_f32).all(), f"{name}: non-finite value in OpenTile output"
    assert torch.isfinite(expected_f32).all(), f"{name}: non-finite value in CPU golden"
    diff = actual_f32 - expected_f32
    max_abs = float(diff.abs().max().item())
    mean_abs = float(diff.abs().mean().item())
    error_rate = float(diff.square().sum().sqrt().item() / (expected_f32.square().sum().sqrt().item() + 1e-8))
    passed = max_abs <= ERROR_ATOL or error_rate < ERROR_RATIO
    print(
        f"[E2E_COMPARE] op=chunk_gated_delta_rule_fwd_h tensor={name} "
        f"pass={int(passed)} finite={actual_f32.numel()}/{actual_f32.numel()} "
        f"max_abs={max_abs:.8g} mean_abs={mean_abs:.8g} error_rate={error_rate:.8g} "
        f"limit={ERROR_RATIO}",
        flush=True,
    )
    assert passed, f"{name}: max_abs={max_abs:.8g}, error_rate={error_rate:.8g}"


@pytest.mark.parametrize("B,T,H,K,V", CASES)
def test_chunk_gated_delta_rule_fwd_h_opentile(B, T, H, K, V):
    golden_file = GOLDEN_DIR / f"golden_B{B}_T{T}_H{H}_K{K}_V{V}.pt"
    HV = H

    # first run to save gold input and output
    # k_cpu, w_cpu, u_cpu, g_cpu = _make_inputs(B, T, H, K, V)
    # expected_h, expected_v_new, expected_final_state = _cpu_reference(k_cpu, w_cpu, u_cpu, g_cpu)
    # torch.save(
    #         {
    #             "input": (k_cpu, w_cpu, u_cpu, g_cpu),
    #             "expected": (expected_h, expected_v_new, expected_final_state),
    #         },
    #         golden_file,
    #     )
    # _stage(f"golden_generated -> {golden_file}")


    # gold ready
    if not golden_file.exists():
        raise FileNotFoundError(
            f"Golden file {golden_file} not found. "
        )
    data = torch.load(golden_file, map_location="cpu")
    k_cpu, w_cpu, u_cpu, g_cpu = data["input"]
    expected_h, expected_v_new, expected_final_state = data["expected"]
    _stage(f"golden_loaded <- {golden_file}")

    _stage(
        f"input_and_golden_ready B={B} T={T} H={H} K={K} V={V} "
        f"dtype=bf16 chunk_size={BT} seed=42 use_g=1 save_new_value=1 "
        f"final_state=1 outputs=h,v_new,final_state kernel=blockdim128"
    )

    k = k_cpu.to(_device())
    w = w_cpu.to(_device())
    u = u_cpu.to(_device())
    g = g_cpu.to(_device())
    torch.npu.synchronize()
    _stage("h2d_done")

    actual_h, actual_v_new, actual_final_state = _compile_and_launch(k, w, u, g, B, H, HV, K, V, T)
    torch.npu.synchronize()
    _stage("compile_launch_sync_done")

    _compare("h", actual_h, expected_h)
    _compare("v_new", actual_v_new, expected_v_new)
    _compare("final_state", actual_final_state, expected_final_state)
