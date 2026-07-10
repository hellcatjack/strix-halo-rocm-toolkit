from __future__ import annotations

from tests.gpu.torch_extension_smoke import CPP_SOURCE, GPU_SOURCE


def test_extension_has_binding_and_gpu_kernel():
    assert "PYBIND11_MODULE" in CPP_SOURCE
    assert "__global__" in GPU_SOURCE
    assert "AT_DISPATCH_FLOATING_TYPES" in GPU_SOURCE
    assert "input.is_cuda()" in GPU_SOURCE
