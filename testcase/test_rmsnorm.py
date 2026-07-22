import os
from pathlib import Path
import sys

import pytest


os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ.setdefault("TRITON_BACKEND", "opentile")
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
PYTHON_DIR = REPO_ROOT / "python"
if PYTHON_DIR.exists():
    sys.path.insert(0, str(PYTHON_DIR))
sys.path.insert(0, str(THIS_DIR))


torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

try:
    import torch.distributed.tensor  # noqa: F401
except ImportError:
    pass


def _npu_available():
    try:
        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


if not _npu_available():
    pytest.skip("torch_npu is available, but no NPU device is visible", allow_module_level=True)


_device_id = os.environ.get("OPENTILE_TEST_DEVICE")
if _device_id is not None:
    torch.npu.set_device(int(_device_id))


from rmsnorm import LigerRMSNormFunction  # noqa: E402


EPS = 1e-6
OFFSET = 0.0
CASTING_MODE = "none"
# SHAPE = (128, 4096)
DTYPE = torch.float32
ATOL = 1e-4
RTOL = 1e-4


def _sync(tag):
    print(f"[SYNC] {tag}", flush=True)
    torch.npu.synchronize()
    print(f"[SYNC OK] {tag}", flush=True)

def _device():
    return torch.device("npu", torch.npu.current_device())



def _reference_rms_norm(x, weight):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    y = x * torch.rsqrt(variance + EPS)
    return y * (weight + OFFSET)


def _assert_close(actual, expected):
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=ATOL, rtol=RTOL)

@pytest.mark.parametrize("SHAPE", [(8, 4096), (64, 4096), (32, 4096), (8, 8192), (64, 8192), (32, 8192)])
def test_rms_norm_forward_opentile(SHAPE):
    torch.manual_seed(0)

    x = torch.randn(SHAPE, device=_device(), dtype=DTYPE)
    _sync("after randn x")

    weight = torch.randn(SHAPE[-1], device=_device(), dtype=DTYPE)
    _sync("after randn weight")

    print(
        f"[CASE] shape={SHAPE}, dtype={DTYPE}, casting_mode={CASTING_MODE}, "
        f"x_stride={x.stride()}, weight_shape={weight.shape}",
        flush=True,
    )

    actual = LigerRMSNormFunction.apply(x, weight, EPS, OFFSET, CASTING_MODE, False, None)
    _sync("after opentile rmsnorm forward")

    expected = _reference_rms_norm(x, weight)
    _sync("after torch reference")

    _assert_close(actual, expected)

# def test_rms_norm_backward_opentile():
#     torch.manual_seed(1)
#     x_ref = torch.randn(SHAPE, device=_device(), dtype=DTYPE, requires_grad=True)
#     weight_ref = torch.randn(SHAPE[-1], device=_device(), dtype=DTYPE, requires_grad=True)
#     x_actual = x_ref.detach().clone().requires_grad_(True)
#     weight_actual = weight_ref.detach().clone().requires_grad_(True)

#     expected = _reference_rms_norm(x_ref, weight_ref)
#     actual = LigerRMSNormFunction.apply(
#         x_actual,
#         weight_actual,
#         EPS,
#         OFFSET,
#         CASTING_MODE,
#         False,
#         None,
#     )

#     grad = torch.randn_like(expected)
#     expected.backward(grad)
#     actual.backward(grad)

#     _assert_close(actual, expected)
#     _assert_close(x_actual.grad, x_ref.grad)
#     _assert_close(weight_actual.grad, weight_ref.grad)
