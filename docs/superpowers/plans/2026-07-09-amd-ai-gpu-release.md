# AMD AI GPU Qualification and Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Qualify the locked ROCm 7.2.1/PyTorch 2.9.1 stack on Radeon 8060S `gfx1151` and produce an auditable release report, SBOM and immutable image digest.

**Architecture:** Small tests isolate each GPU layer before a release orchestrator runs them in order inside the stable container. Host-side collection snapshots device mappings and kernel logs before and after sustained load. The release gate consumes machine-readable results and refuses a verified release when any required check, lock digest, image label or kernel-log condition fails.

**Tech Stack:** Python 3.12, pytest, ROCm `rocminfo`/`hipcc`, PyTorch 2.9.1, Torch C++ extension loader, Triton 3.5.1, Docker, SPDX 2.3 JSON.

---

## File Map

| Path | Responsibility |
| --- | --- |
| `src/amd_ai/qualification/models.py` | Qualification result and release report types |
| `src/amd_ai/qualification/run.py` | Ordered container/host test orchestration |
| `src/amd_ai/qualification/kernel_log.py` | Before/after kernel log diff and error classification |
| `src/amd_ai/qualification/release.py` | Release gate and immutable artifact report |
| `tests/gpu/torch_smoke.py` | Device, FP16 tensor, GEMM and convolution check |
| `tests/gpu/hip_vector_add.cpp` | Native HIP compiler/runtime check |
| `tests/gpu/torch_extension_smoke.py` | PyTorch C++/HIP extension build and run |
| `tests/gpu/triton_smoke.py` | Minimal Triton kernel compile and run |
| `tests/gpu/stress.py` | Timed repeated GPU workload and memory pressure |
| `tests/gpu/repeated_start.py` | Fresh-process repeated initialization check |
| `tools/generate-sbom.py` | Deterministic SPDX 2.3 OS/Python inventory |
| `profiles/qualification/stable.toml` | Required checks, durations and target identifiers |
| `profiles/qualification/comfy-video.example.toml` | Separate manual application qualification example |
| `tests/unit/qualification/` | Parser, log classifier, command and release-gate tests |
| `tests/hardware/test_release.py` | Target-host qualification launcher |
| `docs/gpu-qualification.md` | Reboot, test, failure evidence and release workflow |
| `reports/releases/` | Git-ignored generated reports and SBOMs |

### Task 1: Define qualification result models and strict profile

**Files:**
- Create: `src/amd_ai/qualification/__init__.py`
- Create: `src/amd_ai/qualification/models.py`
- Create: `profiles/qualification/stable.toml`
- Test: `tests/unit/qualification/test_models.py`

- [ ] **Step 1: Write failing profile and report tests**

```python
# tests/unit/qualification/test_models.py
from pathlib import Path

from amd_ai.qualification.models import CheckResult, QualificationReport, load_profile


def test_stable_profile_locks_target_and_required_checks():
    profile = load_profile(Path("profiles/qualification/stable.toml"))
    assert profile.image == "rocm-pytorch:7.2.1-py3.12-torch2.9.1"
    assert profile.gpu_arch == "gfx1151"
    assert profile.required_checks == (
        "rocm",
        "torch-fp16",
        "hip",
        "torch-extension",
        "triton",
        "repeated-start",
        "stress",
        "kernel-log",
    )


def test_required_failure_blocks_report():
    report = QualificationReport.from_results(
        profile_id="stable-gfx1151",
        results=(
            CheckResult("rocm", True, 0.2, {"arch": "gfx1151"}, ""),
            CheckResult("triton", False, 1.0, {}, "compile failed"),
        ),
        required_checks=("rocm", "triton"),
    )
    assert report.status == "blocked"
```

- [ ] **Step 2: Run and verify qualification models are absent**

Run: `uv run pytest tests/unit/qualification/test_models.py -q`

Expected: collection fails for `amd_ai.qualification.models`.

- [ ] **Step 3: Add the exact stable qualification profile**

```toml
# profiles/qualification/stable.toml
profile_id = "stable-gfx1151"
image = "rocm-pytorch:7.2.1-py3.12-torch2.9.1"
rocm_version = "7.2.1"
torch_version = "2.9.1"
gpu_arch = "gfx1151"
stress_seconds = 300
repeated_starts = 5
required_checks = [
  "rocm",
  "torch-fp16",
  "hip",
  "torch-extension",
  "triton",
  "repeated-start",
  "stress",
  "kernel-log",
]
```

- [ ] **Step 4: Implement immutable models and strict TOML parsing**

Create frozen `QualificationProfile`, `CheckResult`, and `QualificationReport` dataclasses. `load_profile()` uses `tomllib`, rejects unknown keys, requires `stress_seconds` from 60 through 3600, `repeated_starts` from 2 through 20, and requires every check name above exactly once. `QualificationReport.from_results()` blocks missing or failed required checks and otherwise passes. JSON output uses schema version 1 and sorted keys.

- [ ] **Step 5: Run model tests**

Run: `uv run pytest tests/unit/qualification/test_models.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit qualification models**

```bash
git add src/amd_ai/qualification profiles/qualification/stable.toml tests/unit/qualification/test_models.py
git commit -m "feat: define gfx1151 qualification profile"
```

### Task 2: Implement ROCm and PyTorch FP16 smoke checks

**Files:**
- Extend: `src/amd_ai/container/check.py`
- Create: `tests/gpu/torch_smoke.py`
- Test: `tests/unit/qualification/test_torch_smoke.py`

- [ ] **Step 1: Write a CPU-only unit test for result validation**

```python
# tests/unit/qualification/test_torch_smoke.py
import pytest

pytest.importorskip("torch")

from tests.gpu.torch_smoke import validate_result


def test_validate_result_accepts_gfx1151_and_small_errors():
    validate_result({
        "available": True,
        "arch": "gfx1151:sramecc-:xnack-",
        "matmul_max_error": 0.01,
        "conv_max_error": 0.01,
    })


def test_validate_result_rejects_cpu_fallback():
    with pytest.raises(AssertionError, match="GPU unavailable"):
        validate_result({"available": False})
```

- [ ] **Step 2: Run and verify the smoke module is absent**

Run: `uv run pytest tests/unit/qualification/test_torch_smoke.py -q`

Expected: collection fails for `tests.gpu.torch_smoke`.

- [ ] **Step 3: Implement the GPU smoke script**

```python
# tests/gpu/torch_smoke.py
from __future__ import annotations

import json

import torch
import torch.nn.functional as F


def validate_result(result: dict[str, object]) -> None:
    assert result.get("available") is True, "GPU unavailable; CPU fallback is forbidden"
    assert str(result.get("arch", "")).startswith("gfx1151"), result.get("arch")
    assert float(result["matmul_max_error"]) <= 0.2
    assert float(result["conv_max_error"]) <= 0.2


def run() -> dict[str, object]:
    result: dict[str, object] = {
        "available": torch.cuda.is_available(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
    }
    if not torch.cuda.is_available():
        validate_result(result)
    properties = torch.cuda.get_device_properties(0)
    arch = getattr(properties, "gcnArchName", "")
    if not arch:
        arch = next((value for value in torch.cuda.get_arch_list() if value.startswith("gfx1151")), "")
    torch.manual_seed(7)
    left_cpu = torch.randn((1024, 1024), dtype=torch.float32)
    right_cpu = torch.randn((1024, 1024), dtype=torch.float32)
    expected_mm = left_cpu @ right_cpu
    actual_mm = (left_cpu.half().cuda() @ right_cpu.half().cuda()).float().cpu()
    image_cpu = torch.randn((2, 4, 64, 64), dtype=torch.float32)
    kernel_cpu = torch.randn((8, 4, 3, 3), dtype=torch.float32)
    expected_conv = F.conv2d(image_cpu, kernel_cpu, padding=1)
    actual_conv = F.conv2d(image_cpu.half().cuda(), kernel_cpu.half().cuda(), padding=1).float().cpu()
    torch.cuda.synchronize()
    result.update({
        "device": torch.cuda.get_device_name(0),
        "arch": arch,
        "matmul_max_error": (expected_mm - actual_mm).abs().max().item(),
        "conv_max_error": (expected_conv - actual_conv).abs().max().item(),
    })
    validate_result(result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
```

- [ ] **Step 4: Extend full container checks**

`container-check --mode torch` first requires `rocminfo` output to contain a standalone `Name:` value beginning `gfx1151`, then runs the equivalent of `torch_smoke.run()`. It reports device mapping separately from ROCm agent discovery and Torch execution.

- [ ] **Step 5: Run CPU-only unit tests**

Run: `uv run pytest tests/unit/qualification/test_torch_smoke.py -q`

Expected: both validation tests pass without invoking a GPU. A host development environment without Torch skips the module through the explicit `pytest.importorskip("torch")` line.

- [ ] **Step 6: Commit smoke checks**

```bash
git add src/amd_ai/container/check.py tests/gpu/torch_smoke.py tests/unit/qualification/test_torch_smoke.py
git commit -m "test: add gfx1151 FP16 smoke check"
```

### Task 3: Compile and run a native HIP program

**Files:**
- Create: `tests/gpu/hip_vector_add.cpp`
- Create: `tests/gpu/run_hip_smoke.py`
- Test: `tests/unit/qualification/test_hip_command.py`

- [ ] **Step 1: Write the compile-command test**

```python
# tests/unit/qualification/test_hip_command.py
from tests.gpu.run_hip_smoke import compile_argv


def test_hip_compile_targets_native_gfx1151(tmp_path):
    argv = compile_argv(tmp_path / "hip_vector_add.cpp", tmp_path / "hip-vector-add")
    assert argv[0] == "/opt/rocm/bin/hipcc"
    assert "--offload-arch=gfx1151" in argv
    assert "-O2" in argv
```

- [ ] **Step 2: Run and verify the HIP helper is absent**

Run: `uv run pytest tests/unit/qualification/test_hip_command.py -q`

Expected: collection fails for `tests.gpu.run_hip_smoke`.

- [ ] **Step 3: Add the complete vector-add source**

```cpp
// tests/gpu/hip_vector_add.cpp
#include <hip/hip_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>

#define HIP_CHECK(call) do { \
  hipError_t error = call; \
  if (error != hipSuccess) { \
    std::cerr << hipGetErrorString(error) << std::endl; \
    return 2; \
  } \
} while (0)

__global__ void add(const float* a, const float* b, float* output, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) output[index] = a[index] + b[index];
}

int main() {
  constexpr int count = 1 << 20;
  const size_t bytes = count * sizeof(float);
  std::vector<float> a(count, 1.25f), b(count, 2.5f), output(count);
  float *device_a = nullptr, *device_b = nullptr, *device_output = nullptr;
  HIP_CHECK(hipMalloc(&device_a, bytes));
  HIP_CHECK(hipMalloc(&device_b, bytes));
  HIP_CHECK(hipMalloc(&device_output, bytes));
  HIP_CHECK(hipMemcpy(device_a, a.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device_b, b.data(), bytes, hipMemcpyHostToDevice));
  add<<<(count + 255) / 256, 256>>>(device_a, device_b, device_output, count);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(output.data(), device_output, bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(device_a));
  HIP_CHECK(hipFree(device_b));
  HIP_CHECK(hipFree(device_output));
  for (float value : output) if (std::fabs(value - 3.75f) > 1e-6f) return 3;
  std::cout << "HIP vector add PASS" << std::endl;
  return 0;
}
```

- [ ] **Step 4: Implement compile/run helper**

`compile_argv()` returns `/opt/rocm/bin/hipcc -O2 --offload-arch=gfx1151 <source> -o <output>`. The helper compiles into a temporary directory, runs the binary, requires stdout `HIP vector add PASS`, and emits a JSON result with compile/run durations and binary size.

- [ ] **Step 5: Run the command unit test**

Run: `uv run pytest tests/unit/qualification/test_hip_command.py -q`

Expected: pass without invoking `hipcc`.

- [ ] **Step 6: Commit native HIP check**

```bash
git add tests/gpu/hip_vector_add.cpp tests/gpu/run_hip_smoke.py tests/unit/qualification/test_hip_command.py
git commit -m "test: compile native gfx1151 HIP program"
```

### Task 4: Compile a PyTorch C++/HIP extension

**Files:**
- Create: `tests/gpu/torch_extension_smoke.py`
- Test: `tests/unit/qualification/test_extension_source.py`

- [ ] **Step 1: Write a source contract test**

```python
# tests/unit/qualification/test_extension_source.py
from tests.gpu.torch_extension_smoke import CPP_SOURCE, GPU_SOURCE


def test_extension_has_binding_and_gpu_kernel():
    assert "PYBIND11_MODULE" in CPP_SOURCE
    assert "__global__" in GPU_SOURCE
    assert "AT_DISPATCH_FLOATING_TYPES" in GPU_SOURCE
```

- [ ] **Step 2: Run and verify the extension script is absent**

Run: `uv run pytest tests/unit/qualification/test_extension_source.py -q`

Expected: collection fails for `tests.gpu.torch_extension_smoke`.

- [ ] **Step 3: Implement inline extension build and validation**

Use these complete inline sources; PyTorch's ROCm extension path hipifies the CUDA compatibility names and invokes HIPCC:

```python
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
__global__ void add_one_kernel(const scalar_t* input, scalar_t* output, int64_t count) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) output[index] = input[index] + static_cast<scalar_t>(1);
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
```

Build with:

```python
module = torch.utils.cpp_extension.load_inline(
    name="amd_ai_add_one",
    cpp_sources=CPP_SOURCE,
    cuda_sources=GPU_SOURCE,
    functions=None,
    extra_cuda_cflags=["-O2", "--offload-arch=gfx1151"],
    with_cuda=True,
    verbose=True,
)
```

Set `TORCH_EXTENSIONS_DIR` to a temporary project-local directory. Run against a 1,048,576-element float32 GPU tensor and require exact equality with `input + 1`. Emit JSON containing build duration, run duration and extension directory size.

- [ ] **Step 4: Run source tests**

Run: `uv run pytest tests/unit/qualification/test_extension_source.py -q`

Expected: pass.

- [ ] **Step 5: Commit extension qualification**

```bash
git add tests/gpu/torch_extension_smoke.py tests/unit/qualification/test_extension_source.py
git commit -m "test: build PyTorch HIP extension"
```

### Task 5: Compile and run a Triton kernel

**Files:**
- Create: `tests/gpu/triton_smoke.py`
- Test: `tests/unit/qualification/test_triton_source.py`

- [ ] **Step 1: Write the Triton source contract test**

```python
# tests/unit/qualification/test_triton_source.py
from pathlib import Path


def test_triton_smoke_uses_jit_kernel_and_gpu_tensors():
    source = Path("tests/gpu/triton_smoke.py").read_text()
    assert "@triton.jit" in source
    assert 'device="cuda"' in source
    assert "torch.testing.assert_close" in source
```

- [ ] **Step 2: Run and verify the Triton script is absent**

Run: `uv run pytest tests/unit/qualification/test_triton_source.py -q`

Expected: `FileNotFoundError`.

- [ ] **Step 3: Add the complete Triton add kernel**

```python
# tests/gpu/triton_smoke.py
from __future__ import annotations

import json
import time

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(left, right, output, count: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < count
    tl.store(output + offsets, tl.load(left + offsets, mask=mask) + tl.load(right + offsets, mask=mask), mask=mask)


def run() -> dict[str, object]:
    count = 1 << 20
    left = torch.randn(count, device="cuda", dtype=torch.float32)
    right = torch.randn(count, device="cuda", dtype=torch.float32)
    output = torch.empty_like(left)
    started = time.monotonic()
    add_kernel[(triton.cdiv(count, 256),)](left, right, output, count=count, BLOCK=256)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, left + right)
    properties = torch.cuda.get_device_properties(0)
    arch = getattr(properties, "gcnArchName", "")
    if not arch:
        arch = next((value for value in torch.cuda.get_arch_list() if value.startswith("gfx1151")), "")
    assert arch.startswith("gfx1151"), arch
    return {"count": count, "seconds": time.monotonic() - started, "arch": arch}


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
```

- [ ] **Step 4: Run source tests**

Run: `uv run pytest tests/unit/qualification/test_triton_source.py -q`

Expected: pass.

- [ ] **Step 5: Commit Triton qualification**

```bash
git add tests/gpu/triton_smoke.py tests/unit/qualification/test_triton_source.py
git commit -m "test: compile Triton gfx1151 kernel"
```

### Task 6: Add repeated-start, sustained-load and kernel-log checks

**Files:**
- Create: `tests/gpu/repeated_start.py`
- Create: `tests/gpu/stress.py`
- Create: `src/amd_ai/qualification/kernel_log.py`
- Test: `tests/unit/qualification/test_kernel_log.py`

- [ ] **Step 1: Write failing kernel-log classification tests**

```python
# tests/unit/qualification/test_kernel_log.py
from amd_ai.qualification.kernel_log import classify_new_lines


def test_known_gpu_failures_are_blocking():
    findings = classify_new_lines([
        "amdgpu 0000:c5:00.0: MES failed to respond to msg=REMOVE_QUEUE",
        "amdgpu: GPU reset begin!",
        "amdgpu: page fault (src_id:0 ring:24 vmid:3)",
    ])
    assert {finding.code for finding in findings} == {
        "GPU.MES_TIMEOUT",
        "GPU.RESET",
        "GPU.PAGE_FAULT",
    }


def test_unrelated_warning_is_retained_as_evidence_not_blocker():
    findings = classify_new_lines(["usb 1-1: reset high-speed USB device"])
    assert findings == ()
```

- [ ] **Step 2: Run and verify classifier is absent**

Run: `uv run pytest tests/unit/qualification/test_kernel_log.py -q`

Expected: collection fails for `amd_ai.qualification.kernel_log`.

- [ ] **Step 3: Implement exact log diff and classifications**

Capture `sudo -n dmesg --color=never` before and after qualification. The runbook obtains a sudo credential with `sudo -v` before starting; inability to read the kernel log is a blocking `HOST.DMESG_UNAVAILABLE` result rather than a skipped check. Compute a multiset suffix/difference without discarding duplicate new messages. Classify case-insensitively:

```text
MES.*(timeout|failed to respond) -> GPU.MES_TIMEOUT
GPU reset begin                -> GPU.RESET
amdgpu.*page fault             -> GPU.PAGE_FAULT
ring .* timeout                -> GPU.RING_TIMEOUT
failed to load firmware        -> GPU.FIRMWARE
```

Store all new `amdgpu`, `drm`, `kfd`, `ttm` and `firmware` lines as evidence, including nonblocking lines.

- [ ] **Step 4: Implement repeated-start and stress scripts**

`repeated_start.py --count 5` launches five fresh child Python processes. Each imports Torch, checks `torch.cuda.is_available()`, allocates a 1024x1024 FP16 tensor, runs one matmul, synchronizes and exits. Record each duration and reject any child failure or architecture not beginning `gfx1151`.

`stress.py --seconds 300` repeatedly alternates 4096x4096 FP16 GEMM and 16x16x256x256 FP16 convolution, synchronizes every ten iterations, and records iterations, peak allocated bytes and wall time. It catches no GPU exception; any HIP error exits nonzero. The script allocates bounded tensors and does not attempt to consume the entire TTM limit.

- [ ] **Step 5: Run kernel-log unit tests**

Run: `uv run pytest tests/unit/qualification/test_kernel_log.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit stability checks**

```bash
git add tests/gpu/repeated_start.py tests/gpu/stress.py src/amd_ai/qualification/kernel_log.py tests/unit/qualification/test_kernel_log.py
git commit -m "test: detect sustained gfx1151 failures"
```

### Task 7: Orchestrate the complete hardware qualification

**Files:**
- Create: `src/amd_ai/qualification/run.py`
- Extend: `src/amd_ai/cli.py`
- Extend: `bin/container-check`
- Create: `tests/hardware/test_release.py`
- Test: `tests/unit/qualification/test_run.py`

- [ ] **Step 1: Write command-order and device-policy tests**

```python
# tests/unit/qualification/test_run.py
from amd_ai.qualification.run import build_suite_commands


def test_suite_orders_cheap_checks_before_stress():
    commands = build_suite_commands(
        image="rocm-pytorch:7.2.1-py3.12-torch2.9.1",
        gids=(109, 110),
        stress_seconds=300,
        repeated_starts=5,
    )
    names = [command.name for command in commands]
    assert names == [
        "rocm",
        "torch-fp16",
        "hip",
        "torch-extension",
        "triton",
        "repeated-start",
        "stress",
    ]
    for command in commands:
        assert "--privileged" not in command.argv
        assert "--ipc=host" not in command.argv
```

- [ ] **Step 2: Run and verify orchestrator is absent**

Run: `uv run pytest tests/unit/qualification/test_run.py -q`

Expected: collection fails for `amd_ai.qualification.run`.

- [ ] **Step 3: Implement ordered fail-fast commands and report writing**

Every test uses a fresh `docker run --rm` with `/dev/kfd`, `/dev/dri`, actual group GIDs, private IPC and 16 GiB shm on this host. Mount the repository test directory read-only at `/opt/amd-ai/tests` and a project-local writable qualification cache at `reports/qualification-cache`; do not mount model or Hugging Face paths.

Run cheap checks first and stop before stress on any failure. Capture stdout, stderr, return code and monotonic duration. Parse each final JSON line into `CheckResult`. Capture dmesg before the first command and after the last started command, then append `kernel-log` result.

Extend CLI:

```text
container-check --suite stable --profile profiles/qualification/stable.toml --json reports/qualification.json
```

The hardware pytest is marked `hardware` and skips only when `/dev/kfd` is absent; on the target host any GPU failure is a test failure.

- [ ] **Step 4: Run orchestrator unit tests**

Run: `uv run pytest tests/unit/qualification/test_run.py -q`

Expected: all tests pass with fake Docker and dmesg runners.

- [ ] **Step 5: Commit qualification orchestration**

```bash
git add src/amd_ai/qualification/run.py src/amd_ai/cli.py bin/container-check tests/hardware/test_release.py tests/unit/qualification/test_run.py
git commit -m "feat: orchestrate hardware qualification"
```

### Task 8: Generate SPDX inventory and enforce the release gate

**Files:**
- Create: `tools/generate-sbom.py`
- Create: `src/amd_ai/qualification/release.py`
- Test: `tests/unit/qualification/test_sbom.py`
- Test: `tests/unit/qualification/test_release.py`

- [ ] **Step 1: Write failing SBOM and release tests**

```python
# tests/unit/qualification/test_sbom.py
from tests.support.load_script import load_script


def test_spdx_document_has_required_identity(tmp_path):
    module = load_script("tools/generate-sbom.py")
    document = module.build_spdx(
        name="rocm-pytorch:stable",
        namespace="https://example.invalid/spdx/test",
        os_packages=[("rocm-core", "7.2.1.70201-1")],
        python_packages=[("torch", "2.9.1+rocm7.2.1")],
        created="2026-07-09T12:00:00Z",
    )
    assert document["spdxVersion"] == "SPDX-2.3"
    assert {package["name"] for package in document["packages"]} == {"rocm-core", "torch"}
```

```python
# tests/unit/qualification/test_release.py
import pytest

from amd_ai.qualification.release import ReleaseBlocked, verify_release
from tests.unit.qualification.fakes import passing_release_inputs


def test_release_requires_digest_locks_sbom_and_all_checks():
    result = verify_release(passing_release_inputs())
    assert result.status == "verified"
    with pytest.raises(ReleaseBlocked, match="kernel-log"):
        verify_release(passing_release_inputs(failed_check="kernel-log"))
```

- [ ] **Step 2: Run and verify release modules are absent**

Run: `uv run pytest tests/unit/qualification/test_sbom.py tests/unit/qualification/test_release.py -q`

Expected: script load/import failures.

- [ ] **Step 3: Implement deterministic SPDX 2.3 JSON**

`generate-sbom.py` reads `dpkg-query -W` and `importlib.metadata.distributions()`, deduplicates OS and Python package namespaces, sorts by namespace/name/version, and emits a valid SPDX 2.3 JSON document. Every package has a stable `SPDXRef-Package-<sha256-prefix>`, `downloadLocation=NOASSERTION`, `filesAnalyzed=false`, and document `DESCRIBES` relationships. Creation timestamp is supplied by the release orchestrator so tests remain deterministic.

Generate the real inventory inside the qualified image by mounting `tools/generate-sbom.py` read-only at `/opt/amd-ai/generate-sbom.py` and running `/opt/venv/bin/python /opt/amd-ai/generate-sbom.py --name <image tag> --created <report timestamp> --output -`; capture stdout to the host report directory. This ensures `dpkg-query` and `importlib.metadata` observe the image rather than the host.

- [ ] **Step 4: Implement the release gate**

`verify_release()` requires:

```text
stable qualification profile digest
approved design digest
all required CheckResult values passed
verified image profile label
ROCm 7.2.1 and Torch 2.9.1 metadata
local immutable image ID from `docker image inspect .Id`
registry RepoDigest when one exists; absence is recorded as `null` for a local-only release
four primary wheel hashes
ROCm package lock digest
SPDX document digest
Git revision with no tracked-file modifications
```

It writes `reports/releases/<UTC>-gfx1151.json` and copies the SPDX JSON beside it. It never retags an image on failure. On success, run `docker image tag <qualified sha256 image ID> rocm-pytorch:7.2.1-py3.12-torch2.9.1-gfx1151-verified`; then re-inspect the tag and require the same image ID. If the image is later pushed, append its registry RepoDigest to the release report without replacing the recorded local ID.

Expose `python -m amd_ai.qualification.release --qualification PATH --image TAG [--output-dir PATH]`; its `main()` returns 0 only after report/SBOM verification and successful immutable retagging.

- [ ] **Step 5: Run SBOM and release tests**

Run: `uv run pytest tests/unit/qualification/test_sbom.py tests/unit/qualification/test_release.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit release tooling**

```bash
git add tools/generate-sbom.py src/amd_ai/qualification/release.py tests/unit/qualification/test_sbom.py tests/unit/qualification/test_release.py tests/unit/qualification/fakes.py
git commit -m "feat: gate verified gfx1151 releases"
```

### Task 9: Add separate application qualification and final runbook

**Files:**
- Create: `profiles/qualification/comfy-video.example.toml`
- Create: `docs/gpu-qualification.md`
- Modify: `README.md`

- [ ] **Step 1: Add the manual application profile without installing ComfyUI**

```toml
# profiles/qualification/comfy-video.example.toml
profile_id = "manual-comfy-video"
project_path = "/srv/projects/comfy-video"
runs = 2
timeout_seconds = 7200
required_observations = [
  "first-run-seconds",
  "second-run-seconds",
  "peak-allocated-bytes",
  "kernel-log",
]
```

This profile points to a user-created project and contains no repository, model URL, workflow, model mount or cache mount. The operator supplies those in the project TOML.

- [ ] **Step 2: Write the hardware and release runbook**

Document the exact sequence:

```bash
sudo -v
./bin/host-verify --json reports/host-verify.json
./bin/container-check --suite stable \
  --profile profiles/qualification/stable.toml \
  --json reports/qualification.json
uv run python -m amd_ai.qualification.release \
  --qualification reports/qualification.json \
  --image rocm-pytorch:7.2.1-py3.12-torch2.9.1
```

Include failure bundles (`dmesg`, report JSON, image digest, locks, kernel/package snapshot), second-run regression interpretation, how to test a single CWSR/MES workaround without adding it to baseline, and the rule that community patches live only in the affected project image after reproduction and regression testing.

- [ ] **Step 3: Run every non-hardware test**

Run:

```bash
uv run pytest -m "not hardware" -q
```

Expected: zero failures.

- [ ] **Step 4: Run the target-host release suite**

Run after host preparation and reboot:

```bash
sudo -v
uv run pytest tests/hardware/test_release.py -m hardware -v
```

Expected: all eight required checks pass, no blocking new kernel line appears, and the report records `gfx1151`.

- [ ] **Step 5: Commit application profile and qualification docs**

```bash
git add profiles/qualification/comfy-video.example.toml docs/gpu-qualification.md README.md
git commit -m "docs: add GPU and application qualification workflow"
```
