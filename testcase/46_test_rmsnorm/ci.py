import os

import pytest
import torch
import torch_npu  # noqa: F401

from ci_rmsnorm import rmsnorm


SEED = int(os.environ.get("OPENTILE_TEST_SEED", "2026"))
CASES = [
    pytest.param((2, 256), torch.float16, id="fp16-aligned"),
    pytest.param((57, 7338), torch.float16, id="fp16-tail"),
    pytest.param((2, 256), torch.bfloat16, id="bf16-aligned"),
    pytest.param((57, 7338), torch.bfloat16, id="bf16-tail"),
]
TOL = {torch.float16: 1e-3, torch.bfloat16: 5e-3}


@pytest.mark.parametrize("shape,dtype", CASES)
def test_rmsnorm(shape, dtype):
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    x = torch.randn(shape, generator=gen, dtype=dtype)
    weight = torch.randn(shape[-1], generator=gen, dtype=dtype)
    expected = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5) * weight.float()).to(dtype)
    device = torch.device("npu", torch.npu.current_device())
    actual = rmsnorm(x.to(device), weight.to(device), 1e-5)
    torch.npu.synchronize()
    actual = actual.cpu()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=TOL[dtype], rtol=TOL[dtype])
    print(f"seed={SEED}, shape={shape}, dtype={dtype}")


if __name__ == "__main__":
    # test_rmsnorm((2, 256), torch.float16)
    pass
