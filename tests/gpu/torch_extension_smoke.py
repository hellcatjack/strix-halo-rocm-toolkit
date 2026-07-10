from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path


CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor add_one_cuda(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("add_one", &add_one_cuda, "Add one on the GPU");
}
"""

GPU_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void add_one_kernel(const scalar_t* input, scalar_t* output,
                               int64_t count) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = input[index] + static_cast<scalar_t>(1);
  }
}

torch::Tensor add_one_cuda(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "input must be on the GPU");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  auto output = torch::empty_like(input);
  constexpr int threads = 256;
  int blocks = static_cast<int>((input.numel() + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "add_one_cuda", [&] {
    add_one_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
        input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), input.numel());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
"""


def run() -> dict[str, object]:
    import torch
    from torch.utils.cpp_extension import load_inline

    assert torch.cuda.is_available(), "GPU unavailable; CPU fallback is forbidden"
    cache_root_value = os.environ.get("AMD_AI_QUALIFICATION_CACHE")
    cache_root = Path(cache_root_value) if cache_root_value else None
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="torch-extension-",
        dir=cache_root,
    ) as temporary:
        extension_dir = Path(temporary)
        previous = {
            name: os.environ.get(name)
            for name in ("TORCH_EXTENSIONS_DIR", "PYTORCH_ROCM_ARCH", "MAX_JOBS")
        }
        os.environ["TORCH_EXTENSIONS_DIR"] = str(extension_dir)
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx1151"
        os.environ.setdefault("MAX_JOBS", "2")
        try:
            build_started = time.monotonic()
            module = load_inline(
                name="amd_ai_add_one",
                cpp_sources=CPP_SOURCE,
                cuda_sources=GPU_SOURCE,
                functions=None,
                extra_cuda_cflags=["-O2", "--offload-arch=gfx1151"],
                with_cuda=True,
                verbose=True,
            )
            build_seconds = time.monotonic() - build_started
            count = 1 << 20
            input_tensor = torch.randn(count, device="cuda", dtype=torch.float32)
            run_started = time.monotonic()
            output = module.add_one(input_tensor)
            torch.cuda.synchronize()
            run_seconds = time.monotonic() - run_started
            torch.testing.assert_close(
                output,
                input_tensor + 1,
                rtol=0,
                atol=0,
            )
            properties = torch.cuda.get_device_properties(0)
            architecture = str(getattr(properties, "gcnArchName", ""))
            assert architecture.startswith("gfx1151"), architecture
            return {
                "arch": architecture,
                "build_seconds": build_seconds,
                "run_seconds": run_seconds,
                "extension_bytes": _directory_size(extension_dir),
                "count": count,
            }
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _directory_size(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
