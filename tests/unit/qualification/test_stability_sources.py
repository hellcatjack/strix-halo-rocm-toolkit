from __future__ import annotations

from pathlib import Path


def test_repeated_start_uses_fresh_child_processes_and_fp16_matmul():
    source = Path("tests/gpu/repeated_start.py").read_text(encoding="utf-8")

    assert "subprocess.run" in source
    assert '"-c"' in source
    assert "torch.float16" in source
    assert "gfx1151" in source


def test_stress_uses_bounded_gemm_and_convolution_with_synchronization():
    source = Path("tests/gpu/stress.py").read_text(encoding="utf-8")

    assert "(4096, 4096)" in source
    assert "(16, 16, 256, 256)" in source
    assert "conv2d" in source
    assert "cuda.synchronize" in source
    assert "max_memory_allocated" in source
