from __future__ import annotations

from pathlib import Path


def test_triton_smoke_uses_jit_kernel_and_gpu_tensors():
    source = Path("tests/gpu/triton_smoke.py").read_text(encoding="utf-8")

    assert "@triton.jit" in source
    assert 'device="cuda"' in source
    assert "torch.testing.assert_close" in source
    assert "gfx1151" in source
