# AMD AI ROCm Images Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build reproducible Ubuntu 24.04 ROCm 7.2.1/Python 3.12 and verified PyTorch 2.9.1 parent images without embedding download caches or unrelated HPC packages.

**Architecture:** Strict profile files identify the complete Torch stack and immutable downloads. Resolver tools populate a local content cache and produce hash-locked manifests; BuildKit mounts that cache as a read-only named context so multi-gigabyte wheels are installed once into the PyTorch parent layer but never copied into a retained wheel layer. Both images are built directly from a digest-pinned Ubuntu base, and GPU-independent image checks run before hardware qualification.

**Tech Stack:** Docker BuildKit, Ubuntu 24.04, ROCm/HIP SDK 7.2.1, Python 3.12 `/opt/venv`, `uv` 0.11.28, AMD CPython 3.12 wheels, PyTorch 2.9.1, TorchVision 0.24.0, TorchAudio 2.9.0, Triton 3.5.1.

---

## File Map

| Path | Responsibility |
| --- | --- |
| `src/amd_ai/image/profile.py` | Strict non-shell Torch profile parser and validation |
| `src/amd_ai/image/lock.py` | Stream downloads, hashes, dependency lock and cache manifest |
| `src/amd_ai/image/build.py` | Deterministic BuildKit argv construction and label checks |
| `src/amd_ai/container/check.py` | Container-local ROCm and Python metadata checks |
| `tools/lock-wheels` | Generate stable profile hashes and wheelhouse |
| `tools/lock-rocm-packages` | Resolve ROCm Debian package versions in Ubuntu 24.04 |
| `profiles/torch/stable.sources.env` | AMD source URLs and fixed component versions |
| `profiles/torch/stable.env` | Generated verified profile with SHA-256 values |
| `profiles/torch/custom.example.env` | Experimental user profile schema example |
| `profiles/torch/stable.requirements.lock` | Hash-locked primary and transitive Python packages |
| `profiles/rocm/7.2.1-packages.lock` | Exact ROCm package versions |
| `profiles/base-images.lock` | Ubuntu and `uv` OCI digests |
| `images/rocm-python/Dockerfile` | No-Torch ROCm/Python/HIP development image |
| `images/rocm-pytorch/Dockerfile` | Verified Torch parent image |
| `images/common/container-check` | In-image check wrapper |
| `images/common/torch-manifest.py` | Build and verify Torch distribution file hashes |
| `bin/image-build` | User-facing image builder |
| `bin/container-check` | Host wrapper for local/current-container checks |
| `tests/unit/image/` | Profile, locking and build argv tests |
| `tests/container/` | GPU-independent Dockerfile/image assertions |
| `docs/image-build.md` | Stable and custom profile build runbook |

### Task 1: Implement the strict Torch profile parser

**Files:**
- Create: `src/amd_ai/image/__init__.py`
- Create: `src/amd_ai/image/profile.py`
- Create: `profiles/torch/stable.sources.env`
- Create: `profiles/torch/custom.example.env`
- Test: `tests/unit/image/test_profile.py`

- [ ] **Step 1: Write failing profile parser tests**

```python
# tests/unit/image/test_profile.py
from pathlib import Path

import pytest

from amd_ai.image.profile import ProfileError, load_profile


def test_repository_stable_profile_is_verified(tmp_path):
    profile_file = tmp_path / "stable.env"
    profile_file.write_text(
        "\n".join([
            "PROFILE_ID=rocm-7.2.1-py3.12-torch-2.9.1",
            "PROFILE_STATUS=verified",
            "ROCM_VERSION=7.2.1",
            "PYTHON_ABI=cp312",
            "PLATFORM=linux/amd64",
            "TORCH_VERSION=2.9.1",
            "TORCH_URL=https://repo.radeon.com/torch.whl",
            f"TORCH_SHA256={'a' * 64}",
            "TORCHVISION_VERSION=0.24.0",
            "TORCHVISION_URL=https://repo.radeon.com/vision.whl",
            f"TORCHVISION_SHA256={'b' * 64}",
            "TORCHAUDIO_VERSION=2.9.0",
            "TORCHAUDIO_URL=https://repo.radeon.com/audio.whl",
            f"TORCHAUDIO_SHA256={'c' * 64}",
            "TRITON_VERSION=3.5.1",
            "TRITON_URL=https://repo.radeon.com/triton.whl",
            f"TRITON_SHA256={'d' * 64}",
        ]) + "\n"
    )
    profile = load_profile(profile_file, allow_verified=True)
    assert profile.profile_id == "rocm-7.2.1-py3.12-torch-2.9.1"
    assert tuple(profile.wheels) == ("torch", "torchvision", "torchaudio", "triton")


@pytest.mark.parametrize("bad_line", [
    "TORCH_URL=http://example.com/torch.whl",
    "TORCH_SHA256=short",
    "EXTRA_KEY=value",
    "TORCH_URL=$(curl attacker)",
    "TORCH_URL=${UNTRUSTED}",
    "TORCH_URL=https://token@example.com/torch.whl",
])
def test_profile_rejects_unsafe_or_unknown_values(tmp_path, bad_line):
    source = Path("profiles/torch/custom.example.env").read_text()
    for key, digest in {
        "TORCH_SHA256": "a" * 64,
        "TORCHVISION_SHA256": "b" * 64,
        "TORCHAUDIO_SHA256": "c" * 64,
        "TRITON_SHA256": "d" * 64,
    }.items():
        source = source.replace(f"{key}={'0' * 64}", f"{key}={digest}")
    lines = [line for line in source.splitlines() if not line.startswith(bad_line.split("=", 1)[0] + "=")]
    path = tmp_path / "bad.env"
    path.write_text("\n".join(lines + [bad_line]) + "\n")
    with pytest.raises(ProfileError):
        load_profile(path, allow_verified=False)
```

- [ ] **Step 2: Run tests and verify the parser is missing**

Run: `uv run pytest tests/unit/image/test_profile.py -q`

Expected: collection fails for `amd_ai.image.profile`.

- [ ] **Step 3: Add exact official source files**

```text
# profiles/torch/stable.sources.env
PROFILE_ID=rocm-7.2.1-py3.12-torch-2.9.1
PROFILE_STATUS=verified
ROCM_VERSION=7.2.1
PYTHON_ABI=cp312
PLATFORM=linux/amd64
TORCH_VERSION=2.9.1
TORCH_URL=https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl
TORCHVISION_VERSION=0.24.0
TORCHVISION_URL=https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl
TORCHAUDIO_VERSION=2.9.0
TORCHAUDIO_URL=https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl
TRITON_VERSION=3.5.1
TRITON_URL=https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl
```

`custom.example.env` contains the same keys plus 64-character all-zero example hashes, uses `PROFILE_ID=custom`, `PROFILE_STATUS=experimental`, and points to the same HTTPS URLs. The parser treats an all-zero digest as invalid, so the example cannot be built until the user replaces all four hashes.

- [ ] **Step 4: Implement a non-evaluating parser**

Create frozen `WheelSpec(name, version, url, sha256)` and `TorchProfile(profile_id, status, rocm_version, python_abi, platform, wheels)` dataclasses. Parse each nonblank/noncomment line with `line.partition("=")`; reject duplicate or unknown keys, whitespace in keys, backticks, `$(`, `${`, non-HTTPS URLs, URL userinfo/query/fragment data, digest values not matching `[0-9a-f]{64}`, and missing required keys. Preserve the wheel order `torch`, `torchvision`, `torchaudio`, `triton`.

For this image family, require `ROCM_VERSION=7.2.1`, `PYTHON_ABI=cp312`, and `PLATFORM=linux/amd64` for stable and custom profiles. A profile targeting another ROCm/Python/platform combination requires a separately locked `rocm-python` base and is rejected by these commands.

Only `load_profile(path, allow_verified=True)` may accept `PROFILE_STATUS=verified`; `path` accepts `str` or `Path`, and the image CLI passes true only when the resolved path is exactly the repository-owned `profiles/torch/stable.env`. Add a module CLI that accepts one profile path, validates with the repository-path rule, and prints `<status> <profile_id>`; this is the command used in Task 2.

- [ ] **Step 5: Run profile tests**

Run: `uv run pytest tests/unit/image/test_profile.py -q`

Expected: all parametrized cases pass.

- [ ] **Step 6: Commit profile parsing**

```bash
git add src/amd_ai/image profiles/torch/stable.sources.env profiles/torch/custom.example.env tests/unit/image/test_profile.py
git commit -m "feat: validate complete Torch profiles"
```

### Task 2: Lock and cache AMD wheels without retaining downloads in image layers

**Files:**
- Create: `src/amd_ai/image/lock.py`
- Create: `tools/lock-wheels`
- Create: `.gitignore`
- Test: `tests/unit/image/test_lock.py`

- [ ] **Step 1: Write failing stream-download and manifest tests**

```python
# tests/unit/image/test_lock.py
import hashlib
from pathlib import Path

from amd_ai.image.lock import hash_file, render_verified_profile


def test_hash_file_streams_and_returns_sha256(tmp_path):
    wheel = tmp_path / "torch.whl"
    wheel.write_bytes(b"amd-wheel-fixture")
    assert hash_file(wheel) == hashlib.sha256(b"amd-wheel-fixture").hexdigest()


def test_render_profile_adds_each_digest_in_component_order(tmp_path):
    source = Path("profiles/torch/stable.sources.env")
    digests = {name: name[0] * 64 for name in ("torch", "torchvision", "torchaudio", "triton")}
    rendered = render_verified_profile(source.read_text(), digests)
    assert rendered.index("TORCH_SHA256=") < rendered.index("TORCHVISION_SHA256=")
    assert "TRITON_SHA256=" + "t" * 64 in rendered
```

- [ ] **Step 2: Run and verify locking code is absent**

Run: `uv run pytest tests/unit/image/test_lock.py -q`

Expected: collection fails for `amd_ai.image.lock`.

- [ ] **Step 3: Implement atomic downloads, hashes and generated profile output**

`download(url, destination)` uses `urllib.request.urlopen` with a 60-second connection timeout, streams 8 MiB chunks into `destination.with_suffix(".part")`, calls `os.fsync`, and atomically renames only after a complete response. Existing files are reused only when their computed digest matches the requested digest; an interrupted `.part` file is replaced.

`hash_file()` uses `hashlib.file_digest(file_handle, "sha256")`. `render_verified_profile()` copies each source key and inserts `<COMPONENT>_SHA256` immediately after its URL. It writes no timestamps, so identical inputs produce identical output.

`tools/lock-wheels` dispatches to `python3 -m amd_ai.image.lock` and supports:

```text
--sources profiles/torch/stable.sources.env
--profile profiles/torch/stable.env
--wheelhouse .cache/wheels/rocm-7.2.1-py3.12-torch-2.9.1
```

It downloads the four official AMD wheels, writes the verified profile, then creates `.cache/locks/stable.in` with the four exact public versions and runs:

```bash
uv pip compile \
  --python-version 3.12 \
  --python-platform x86_64-unknown-linux-gnu \
  --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/ \
  --generate-hashes \
  --no-emit-index-url \
  --output-file profiles/torch/stable.requirements.lock \
  .cache/locks/stable.in
```

The tool requires the resulting four primary versions to contain local suffix `+rocm7.2.1` and rejects a PyPI CPU wheel. It downloads every locked artifact into the same wheelhouse with `python3 -m pip download --require-hashes --dest <wheelhouse> --requirement profiles/torch/stable.requirements.lock --find-links <AMD index>`, then writes `wheelhouse-manifest.json` and `wheelhouse.sha256` containing filename, byte size and SHA-256 sorted by filename. Run `chmod +x tools/lock-wheels`.

Add these entries to `.gitignore`:

```text
.cache/
.venv/
reports/
__pycache__/
*.pyc
```

- [ ] **Step 4: Run unit tests**

Run: `uv run pytest tests/unit/image/test_lock.py -q`

Expected: both tests pass.

- [ ] **Step 5: Generate the real stable lock and verify four primary files**

Run:

```bash
./tools/lock-wheels \
  --sources profiles/torch/stable.sources.env \
  --profile profiles/torch/stable.env \
  --wheelhouse .cache/wheels/rocm-7.2.1-py3.12-torch-2.9.1
uv run python -m amd_ai.image.profile profiles/torch/stable.env
```

Expected: profile validation prints `verified rocm-7.2.1-py3.12-torch-2.9.1`, the manifest contains the four AMD filenames plus locked dependencies, and no wheel is added to Git.

- [ ] **Step 6: Commit generated locks and tooling**

```bash
git add .gitignore src/amd_ai/image/lock.py tools/lock-wheels profiles/torch/stable.env profiles/torch/stable.requirements.lock tests/unit/image/test_lock.py
git commit -m "build: lock AMD PyTorch wheel set"
```

### Task 3: Resolve base-image digests and ROCm Debian package versions

**Files:**
- Create: `tools/lock-rocm-packages`
- Create: `profiles/base-images.lock`
- Create: `profiles/rocm/7.2.1-packages.lock`
- Create: `profiles/rocm/rocm.gpg`
- Create: `profiles/rocm/rocm.gpg.sha256`
- Test: `tests/unit/image/test_rocm_lock.py`

- [ ] **Step 1: Write the lock-format test**

```python
# tests/unit/image/test_rocm_lock.py
from amd_ai.image.lock import parse_package_lock


def test_package_lock_requires_sorted_exact_versions():
    lock = parse_package_lock("hipcc=7.2.1.70201-1\nrocm-core=7.2.1.70201-1\n")
    assert lock == (
        ("hipcc", "7.2.1.70201-1"),
        ("rocm-core", "7.2.1.70201-1"),
    )
```

- [ ] **Step 2: Run and verify the parser test fails**

Run: `uv run pytest tests/unit/image/test_rocm_lock.py -q`

Expected: failure because `parse_package_lock` is absent.

- [ ] **Step 3: Implement strict package lock parsing**

`parse_package_lock()` accepts only sorted unique lines matching `[a-z0-9][a-z0-9+.-]*=[^\s=]+`, rejects unversioned packages, and returns tuples. Add unit cases for duplicates, unsorted lines and whitespace.

- [ ] **Step 4: Implement and run the resolver tool**

`tools/lock-rocm-packages` performs these exact operations:

1. Pull `ubuntu:24.04` for `linux/amd64` and record its `RepoDigest` in `profiles/base-images.lock` as `UBUNTU_24_04=<name>@sha256:<digest>`.
2. Resolve `ghcr.io/astral-sh/uv:0.11.28`, record `UV_IMAGE=<name>@sha256:<digest>`, and set `UV_VERSION=0.11.28` in the same lock.
3. Download `https://repo.radeon.com/rocm/rocm.gpg.key`, dearmor it to `profiles/rocm/rocm.gpg`, and record its SHA-256.
4. Start an ephemeral Ubuntu 24.04 resolver container, register only these repositories:

```text
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2.1 noble main
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/graphics/7.2.1/ubuntu noble main
```

5. Install `rocm-hip-sdk` and `rocm-ml-sdk`, select installed packages whose APT origin is `repo.radeon.com`, and write sorted `name=version` lines to `profiles/rocm/7.2.1-packages.lock`.

Run:

```bash
./tools/lock-rocm-packages --rocm-version 7.2.1 --ubuntu noble
uv run pytest tests/unit/image/test_rocm_lock.py -q
```

Expected: both lock files are nonempty, every ROCm package line contains `=`, and tests pass.

- [ ] **Step 5: Commit supply-chain locks**

```bash
git add tools/lock-rocm-packages profiles/base-images.lock profiles/rocm tests/unit/image/test_rocm_lock.py src/amd_ai/image/lock.py
git commit -m "build: lock ROCm base packages"
```

### Task 4: Build the no-Torch ROCm/Python development image

**Files:**
- Create: `images/rocm-python/Dockerfile`
- Create: `images/common/container-check`
- Create: `src/amd_ai/container/__init__.py`
- Create: `src/amd_ai/container/check.py`
- Test: `tests/container/test_rocm_python_dockerfile.py`

- [ ] **Step 1: Write static Dockerfile contract tests**

```python
# tests/container/test_rocm_python_dockerfile.py
from pathlib import Path


def test_rocm_python_is_clean_development_image():
    text = Path("images/rocm-python/Dockerfile").read_text()
    assert "ARG UBUNTU_BASE" in text
    assert "rocm-hip-sdk" in text
    assert "rocm-ml-sdk" in text
    assert "python3.12-venv" in text
    assert "cmake" in text and "ninja-build" in text and "g++" in text
    assert "python3 -m venv /opt/venv" in text
    assert "rm -rf /var/lib/apt/lists/*" in text
    assert "FROM rocm/pytorch" not in text
    assert "torch" not in text.lower().replace("no-torch", "")
```

- [ ] **Step 2: Run and verify the Dockerfile test fails because the file is absent**

Run: `uv run pytest tests/container/test_rocm_python_dockerfile.py -q`

Expected: `FileNotFoundError`.

- [ ] **Step 3: Create the BuildKit Dockerfile**

The Dockerfile starts with `# syntax=docker/dockerfile:1.7`, declares `ARG UBUNTU_BASE`, and uses `FROM ${UBUNTU_BASE}`. It copies the locked ROCm key/package list and writes the two official 7.2.1 Noble repository lines with `signed-by`.

One cache-mounted APT step installs exact `name=version` entries from the lock plus these Ubuntu packages:

```text
build-essential ca-certificates cmake curl git gnupg g++ make ninja-build
pkg-config python3.12 python3.12-dev python3.12-venv unzip wget xz-utils zip
```

That step uses `--mount=type=cache,target=/var/cache/apt,sharing=locked` and `--mount=type=cache,target=/var/lib/apt/lists,sharing=locked`, removes transient lists before completing, creates `/opt/venv`, and verifies `/opt/rocm/bin/hipcc --version` and `/opt/rocm/bin/rocminfo --version` without requiring a GPU.

Copy `/uv` and `/uvx` from the digest-pinned `UV_IMAGE`. Copy `src/amd_ai` to `/opt/amd-ai/src/amd_ai` and `images/common/container-check` to `/usr/local/bin/container-check`. Set:

```dockerfile
ENV PATH="/opt/venv/bin:/opt/rocm/bin:${PATH}" \
    ROCM_PATH="/opt/rocm" \
    HIP_PATH="/opt/rocm" \
    PYTHONUNBUFFERED="1" \
    PYTHONDONTWRITEBYTECODE="1"
```

Create non-root user `developer` with UID/GID 1000, make `/workspace` its working directory, and leave `/opt/venv` readable/executable but not runtime-writable. Do not set Hugging Face, ComfyUI, MIOpen, Triton or Inductor cache environment variables.

- [ ] **Step 4: Implement the ROCm-only container check**

`container-check` sets `PYTHONPATH=/opt/amd-ai/src` and executes `python3 -m amd_ai.container.check`. `check.py --mode rocm --json -` reports `/opt/rocm/.info/version`, `hipcc --version`, `/dev/kfd`, render nodes and `rocminfo`. Missing `/dev/kfd` is a blocking runtime finding but `--metadata-only` skips device checks for image build validation.

- [ ] **Step 5: Run static tests and build the image**

Run:

```bash
uv run pytest tests/container/test_rocm_python_dockerfile.py -q
UBUNTU_BASE="$(sed -n 's/^UBUNTU_24_04=//p' profiles/base-images.lock)"
UV_IMAGE="$(sed -n 's/^UV_IMAGE=//p' profiles/base-images.lock)"
docker buildx build --load --platform linux/amd64 \
  --build-arg "UBUNTU_BASE=${UBUNTU_BASE}" \
  --build-arg "UV_IMAGE=${UV_IMAGE}" \
  --tag rocm-python:7.2.1-py3.12 \
  --file images/rocm-python/Dockerfile .
docker run --rm rocm-python:7.2.1-py3.12 container-check --mode rocm --metadata-only --json -
```

Expected: test passes, image build exits 0, metadata output reports ROCm 7.2.1, and `python --version` is 3.12.x.

- [ ] **Step 6: Commit the no-Torch image**

```bash
git add images/rocm-python images/common/container-check src/amd_ai/container tests/container/test_rocm_python_dockerfile.py
git commit -m "feat: build ROCm Python development image"
```

### Task 5: Create and verify the immutable Torch file manifest

**Files:**
- Create: `images/common/torch-manifest.py`
- Create: `tests/support/load_script.py`
- Test: `tests/unit/image/test_torch_manifest.py`

- [ ] **Step 1: Write failing manifest tests using tiny fake distributions**

```python
# tests/unit/image/test_torch_manifest.py
import json
from pathlib import Path

from tests.support.load_script import load_script


def test_manifest_detects_changed_distribution_file(tmp_path):
    module = load_script(Path("images/common/torch-manifest.py"))
    package = tmp_path / "torch"
    package.mkdir()
    binary = package / "libtorch.so"
    binary.write_bytes(b"first")
    manifest = tmp_path / "manifest.json"
    module.write_manifest({"torch": [binary]}, manifest)
    assert module.verify_manifest(manifest) == []
    binary.write_bytes(b"second")
    errors = module.verify_manifest(manifest)
    assert errors == [f"changed: {binary}"]
    assert json.loads(manifest.read_text())["schema_version"] == 1
```

- [ ] **Step 2: Run and verify the script is missing**

Run: `uv run pytest tests/unit/image/test_torch_manifest.py -q`

Expected: failure loading `images/common/torch-manifest.py`.

- [ ] **Step 3: Implement deterministic distribution hashing**

The script exposes `write_manifest(distributions, path)` and `verify_manifest(path)`. For production `create`, it obtains `importlib.metadata.distribution()` for `torch`, `torchvision`, `torchaudio`, and `triton`, enumerates existing regular files from each distribution's `files`, hashes each with SHA-256, and writes sorted relative path, size and digest entries plus package version. `verify` reports `missing`, `changed`, or `unexpected version` lines and exits 1 when any exist.

Add this test helper so hyphenated scripts load without modifying their source:

```python
# tests/support/load_script.py
import importlib.util
from pathlib import Path
from types import ModuleType


def load_script(path: str | Path) -> ModuleType:
    script = Path(path)
    spec = importlib.util.spec_from_file_location(script.stem.replace("-", "_"), script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
```

CLI forms:

```text
python torch-manifest.py create /opt/amd-ai/torch-manifest.json
python torch-manifest.py verify /opt/amd-ai/torch-manifest.json
```

- [ ] **Step 4: Run manifest tests**

Run: `uv run pytest tests/unit/image/test_torch_manifest.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the manifest guard**

```bash
git add images/common/torch-manifest.py tests/unit/image/test_torch_manifest.py tests/support/load_script.py
git commit -m "feat: hash protected Torch distributions"
```

### Task 6: Build the verified PyTorch parent layer

**Files:**
- Create: `images/rocm-pytorch/Dockerfile`
- Extend: `src/amd_ai/container/check.py`
- Test: `tests/container/test_rocm_pytorch_dockerfile.py`

- [ ] **Step 1: Write failing Dockerfile and metadata tests**

```python
# tests/container/test_rocm_pytorch_dockerfile.py
from pathlib import Path


def test_torch_image_uses_named_wheel_context_and_manifest():
    text = Path("images/rocm-pytorch/Dockerfile").read_text()
    assert "FROM ${ROCM_PYTHON_BASE}" in text
    assert "--mount=from=wheels" in text
    assert "COPY --from=profile-context" in text
    assert "--require-hashes" in text
    assert "torch-manifest.py create" in text
    assert "ARG PROFILE_STATUS" in text
    assert "pip cache" not in text
```

- [ ] **Step 2: Run and verify the test fails because Dockerfile is absent**

Run: `uv run pytest tests/container/test_rocm_pytorch_dockerfile.py -q`

Expected: `FileNotFoundError`.

- [ ] **Step 3: Create the PyTorch Dockerfile**

Use `ARG ROCM_PYTHON_BASE` then `FROM ${ROCM_PYTHON_BASE}`. Accept `ARG PROFILE_ID`, `PROFILE_STATUS`, `ROCM_VERSION`, and the four public component versions. Copy `/profile.env` and `/requirements.lock` from a read-only BuildKit context named `profile-context`, and copy `torch-manifest.py` from the repository context. Install in one layer with the read-only wheel context:

```dockerfile
RUN --mount=from=wheels,target=/wheels,ro \
    uv pip install \
      --python /opt/venv/bin/python \
      --no-index \
      --find-links /wheels \
      --require-hashes \
      --requirements /opt/amd-ai/profile.requirements.lock \
 && /opt/venv/bin/python /opt/amd-ai/torch-manifest.py create /opt/amd-ai/torch-manifest.json \
 && /opt/venv/bin/python -c "import torch, torchvision, torchaudio, triton; assert torch.version.hip"
```

Apply OCI labels from the validated build args, including `org.amd-ai.profile.status=${PROFILE_STATUS}`. The Dockerfile contains no hardcoded stable status and works for both verified and experimental profiles. Do not enable `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL`; users may opt in per project after qualification.

- [ ] **Step 4: Extend metadata checks**

`container-check --mode torch --metadata-only` imports all four packages, emits both full local versions and public versions obtained by splitting once at `+`, requires nonempty `torch.version.hip`, and runs `torch-manifest.py verify`. It compares public versions to 2.9.1/0.24.0/2.9.0/3.5.1 while retaining AMD local build suffixes as evidence. `--runtime` skips the multi-gigabyte file rehash but still checks the four versions, `torch.version.hip`, `torch.cuda.is_available()`, a device architecture beginning `gfx1151`, and one small synchronized GPU tensor operation. The project build already performs the full manifest verification, so normal container startup uses `--runtime`; qualification can request the full check explicitly.

- [ ] **Step 5: Build and inspect the verified image**

Run:

```bash
ROCM_PYTHON_ID="$(docker image inspect --format '{{.Id}}' rocm-python:7.2.1-py3.12)"
mkdir -p .cache/profile-context/rocm-7.2.1-py3.12-torch-2.9.1
cp profiles/torch/stable.env .cache/profile-context/rocm-7.2.1-py3.12-torch-2.9.1/profile.env
cp profiles/torch/stable.requirements.lock .cache/profile-context/rocm-7.2.1-py3.12-torch-2.9.1/requirements.lock
docker buildx build --load --platform linux/amd64 \
  --build-context wheels=.cache/wheels/rocm-7.2.1-py3.12-torch-2.9.1 \
  --build-context profile-context=.cache/profile-context/rocm-7.2.1-py3.12-torch-2.9.1 \
  --build-arg "ROCM_PYTHON_BASE=${ROCM_PYTHON_ID}" \
  --build-arg PROFILE_ID=rocm-7.2.1-py3.12-torch-2.9.1 \
  --build-arg PROFILE_STATUS=verified \
  --build-arg ROCM_VERSION=7.2.1 \
  --build-arg TORCH_VERSION=2.9.1 \
  --build-arg TORCHVISION_VERSION=0.24.0 \
  --build-arg TORCHAUDIO_VERSION=2.9.0 \
  --build-arg TRITON_VERSION=3.5.1 \
  --tag rocm-pytorch:7.2.1-py3.12-torch2.9.1 \
  --file images/rocm-pytorch/Dockerfile .
docker run --rm rocm-pytorch:7.2.1-py3.12-torch2.9.1 \
  container-check --mode torch --metadata-only --json -
docker history --no-trunc rocm-pytorch:7.2.1-py3.12-torch2.9.1
```

Expected: metadata reports 2.9.1/0.24.0/2.9.0/3.5.1, manifest verification passes, and history has one Torch install layer with no retained wheel-copy layer.

- [ ] **Step 6: Commit the verified image**

```bash
git add images/rocm-pytorch src/amd_ai/container/check.py tests/container/test_rocm_pytorch_dockerfile.py
git commit -m "feat: build verified ROCm PyTorch image"
```

### Task 7: Expose deterministic `image-build` and `container-check` commands

**Files:**
- Create: `src/amd_ai/image/build.py`
- Modify: `src/amd_ai/cli.py`
- Create: `bin/image-build`
- Create: `bin/container-check`
- Test: `tests/unit/image/test_build.py`
- Test: `tests/cli/test_image_commands.py`

- [ ] **Step 1: Write failing BuildKit argv tests**

```python
# tests/unit/image/test_build.py
from amd_ai.image.build import build_torch_argv
from amd_ai.image.profile import load_profile


def test_build_argv_uses_digest_parent_and_named_context():
    profile = load_profile("profiles/torch/stable.env", allow_verified=True)
    argv = build_torch_argv(
        profile=profile,
        parent="sha256:" + "a" * 64,
        wheelhouse=".cache/wheels/rocm-7.2.1-py3.12-torch-2.9.1",
        revision="deadbeef",
    )
    assert "--build-context" in argv
    assert "wheels=.cache/wheels/rocm-7.2.1-py3.12-torch-2.9.1" in argv
    assert "profile-context=.cache/profile-context/rocm-7.2.1-py3.12-torch-2.9.1" in argv
    assert "ROCM_PYTHON_BASE=sha256:" + "a" * 64 in argv
    assert "--load" in argv
```

- [ ] **Step 2: Run and verify build orchestration is absent**

Run: `uv run pytest tests/unit/image/test_build.py tests/cli/test_image_commands.py -q`

Expected: collection fails for `amd_ai.image.build`.

- [ ] **Step 3: Implement exact build validation and argv generation**

Before invoking Docker, validate: BuildKit/buildx exists; lock files parse; every wheelhouse manifest file exists and matches its digest; source Git revision is available or explicitly `unknown`; parent image resolves to a local immutable `.Id` value matching `sha256:<64 hex>`; and a verified profile path is the repository stable profile. Materialize `.cache/profile-context/<profile-id>/profile.env` and `requirements.lock`, then pass it through `--build-context profile-context=<directory>`. An experimental profile always produces an image tag suffixed with its sanitized profile ID and label status `experimental`.

For an experimental profile, run the same resolver as Task 2 against its four exact URLs, first verifying each user-supplied SHA-256, then compiling transitive dependencies for Python 3.12/Linux AMD64 and creating a profile-specific wheelhouse manifest. A custom build cannot reuse the stable requirements lock merely because one public version matches.

The build command uses `docker buildx build --platform linux/amd64 --provenance=mode=max --sbom=true --load`. It passes only declared version/status build args plus `--build-context wheels=<wheelhouse>` and `--build-context profile-context=<generated directory>`; it never exports URLs or hashes through environment variables.

Wire CLI forms:

```text
image-build rocm-python
image-build rocm-pytorch --profile profiles/torch/stable.env
image-build rocm-pytorch --profile /absolute/custom.env --allow-experimental
image-build prune [--apply] [--older-than-hours 168]
```

`image-build prune` scans `amd-ai-project.toml` files under the configured project roots, protects every recorded base image ID, the current stable base IDs, and images used by running containers. It lists only images carrying `org.amd-ai.profile.id` or `org.amd-ai.project.fingerprint`, prints their exact IDs and reported sizes, and defaults to no deletion. With `--apply`, remove only the displayed unreferenced IDs and run `docker buildx prune --force --filter until=168h`; never call `docker system prune`, never delete named volumes, and never delete `.cache/wheels` without a separate explicit filesystem command.

```bash
#!/usr/bin/env bash
# bin/image-build
set -euo pipefail
exec "$(dirname "$0")/_dispatch" image-build "$@"
```

```bash
#!/usr/bin/env bash
# bin/container-check
set -euo pipefail
exec "$(dirname "$0")/_dispatch" container-check "$@"
```

Run `chmod +x bin/image-build bin/container-check`.

- [ ] **Step 4: Run build and CLI unit tests**

Run: `uv run pytest tests/unit/image tests/cli/test_image_commands.py -q`

Expected: all tests pass using fake Docker results, including a prune test proving default mode records no mutating Docker argv and protects a project-referenced base image ID.

- [ ] **Step 5: Commit command wiring**

```bash
git add src/amd_ai/image/build.py src/amd_ai/cli.py bin/image-build bin/container-check tests/unit/image/test_build.py tests/cli/test_image_commands.py
git commit -m "feat: orchestrate locked image builds"
```

### Task 8: Verify clean layers and document stable/custom builds

**Files:**
- Create: `tests/container/test_image_contract.py`
- Create: `docs/image-build.md`
- Test: complete image suite

- [ ] **Step 1: Add image inspection tests**

The test invokes `docker image inspect` and `docker history --no-trunc` and asserts:

```text
rocm-python has no installed torch distribution
rocm-pytorch reports the verified profile labels
/opt/venv is the only directory matching */bin/activate
default user is non-root
no HF_HOME, HF_HUB_CACHE, ComfyUI, MIOpen, Triton or Inductor cache variable is set
`importlib.util.find_spec("comfy")` is `None` and `/workspace/ComfyUI` does not exist
no image layer command copies .whl files into the final image
hipcc, cmake, ninja, g++, Python.h and ROCm headers exist
```

Mark these tests `@pytest.mark.container` and skip with a precise reason when Docker is unavailable.

- [ ] **Step 2: Run all image tests**

Run:

```bash
uv run pytest tests/unit/image tests/container -q
```

Expected: zero failures; skips are permitted only when Docker is absent, not when an image exists but violates the contract.

- [ ] **Step 3: Write the image build runbook**

Document lock refresh, stable builds, custom profile creation, `ALLOW_UNVERIFIED=1`, wheelhouse disk usage, BuildKit cache pruning with preview, image digest recording, metadata-only checks, and why custom Torch starts from `rocm-python` rather than replacing the stable parent. Include the four AMD wheel URLs and official ROCm 7.2.1 package repository URLs.

- [ ] **Step 4: Commit image contract and documentation**

```bash
git add tests/container/test_image_contract.py docs/image-build.md
git commit -m "test: enforce clean ROCm image contracts"
```
