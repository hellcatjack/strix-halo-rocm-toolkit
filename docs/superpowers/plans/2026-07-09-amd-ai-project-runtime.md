# AMD AI Project Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and run one independent image/container per project while sharing the immutable PyTorch parent layer, protecting the baseline stack, and leaving all model/cache mounts under user control.

**Architecture:** Each project has a small TOML manifest and generated Dockerfile. Project dependencies install into inherited `/opt/venv` at build time under exact Torch constraints, followed by a full Torch file-manifest verification. The host runner validates image labels, devices, actual GIDs, UID/GID, mounts and shared-memory sizing before issuing a shell-free `docker run` command.

**Tech Stack:** Python 3.12 `tomllib`, Docker BuildKit, `uv`, TOML project manifests, Linux device nodes and group IDs.

---

## File Map

| Path | Responsibility |
| --- | --- |
| `src/amd_ai/project/config.py` | Strict TOML project config parsing |
| `src/amd_ai/project/init.py` | Project scaffold and immutable base resolution |
| `src/amd_ai/project/dependencies.py` | Lock generation and Torch constraints |
| `src/amd_ai/project/runtime.py` | Device, GID, UID, shm and mount validation |
| `src/amd_ai/project/build.py` | Derived image build and config fingerprint |
| `src/amd_ai/project/run.py` | Image label gate and Docker run orchestration |
| `templates/project/Dockerfile` | Guarded project dependency/image template |
| `templates/project/project-entrypoint` | Runtime GPU/profile gate |
| `templates/project/amd-ai-project.toml` | No-shared-mount default config |
| `templates/project/requirements.in` | Empty project dependency input |
| `templates/project/.dockerignore` | Exclude caches, models, outputs and Git data |
| `bin/project-init` | User-facing scaffold command |
| `bin/project-run` | User-facing build/run command |
| `tests/unit/project/` | Config, scaffold, mount and argv tests |
| `tests/container/test_project_layers.py` | Parent-layer reuse and Torch guard acceptance |
| `docs/project-workflow.md` | Project creation, custom Torch and mount examples |

### Task 1: Parse strict project TOML without implicit shared storage

**Files:**
- Create: `src/amd_ai/project/__init__.py`
- Create: `src/amd_ai/project/config.py`
- Create: `templates/project/amd-ai-project.toml`
- Test: `tests/unit/project/test_config.py`

- [ ] **Step 1: Write failing config tests**

```python
# tests/unit/project/test_config.py
from pathlib import Path

import pytest

from amd_ai.project.config import ConfigError, load_project_config


def test_default_template_has_no_mounts_or_cache_policy():
    config = load_project_config(Path("templates/project/amd-ai-project.toml"))
    assert config.name == "demo"
    assert config.base_profile == "stable"
    assert config.mounts == ()
    assert config.environment == ()
    assert config.command == ("bash",)


def test_duplicate_or_reserved_mount_target_is_rejected(tmp_path):
    path = tmp_path / "amd-ai-project.toml"
    digest = "a" * 64
    path.write_text(
        f'[project]\nname="demo"\nbase_profile="stable"\nimage="demo:runtime"\n'
        f'base_image="sha256:{digest}"\n'
        f'base_digest="sha256:{digest}"\n'
        f'command=["bash"]\n'
        f'[[mounts]]\nsource="/data/a"\ntarget="/opt/venv"\nread_only=true\n'
    )
    with pytest.raises(ConfigError, match="reserved"):
        load_project_config(path)
```

- [ ] **Step 2: Run and verify project config code is absent**

Run: `uv run pytest tests/unit/project/test_config.py -q`

Expected: collection fails for `amd_ai.project.config`.

- [ ] **Step 3: Add the exact default template**

```toml
# templates/project/amd-ai-project.toml
[project]
name = "demo"
base_profile = "stable"
image = "demo:runtime"
base_image = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
base_digest = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
command = ["bash"]
debug = false
```

The template intentionally contains no `[[mounts]]` table and no Hugging Face, ComfyUI, MIOpen, Triton or Inductor variable.

- [ ] **Step 4: Implement immutable config types and validation**

Create frozen dataclasses:

```python
@dataclass(frozen=True)
class MountConfig:
    source: Path
    target: PurePosixPath
    read_only: bool


@dataclass(frozen=True)
class ProjectConfig:
    path: Path
    name: str
    base_profile: str
    image: str
    base_image: str
    base_digest: str
    command: tuple[str, ...]
    debug: bool
    shm_size_gib: int | None
    mounts: tuple[MountConfig, ...]
    environment: tuple[tuple[str, str], ...]
```

Use `tomllib.load()`. Accept only top-level `project`, optional `mounts`, and optional `environment`. Project names match `[a-z0-9][a-z0-9._-]{0,62}`; image values contain no whitespace; both `base_image` and `base_digest` match the same local immutable image ID `sha256:<64 lowercase hex>`; command is a nonempty string array; `shm_size_gib`, when present, is 1 through 128. Resolve relative sources against the config directory. The repository template uses an all-zero syntactically valid digest; `project-init` must replace it, and `project-run` rejects an all-zero digest.

Environment names match `[A-Z_][A-Z0-9_]*` and values are strings without NUL. Sort them by name. Reject overrides of `PATH`, `PYTHONPATH`, `LD_LIBRARY_PATH`, `ROCM_PATH`, `HIP_PATH`, `HOME`, `AMD_AI_PROFILE_ID`, and `AMD_AI_PROFILE_STATUS`; allow user-selected names such as `HF_HOME` and `HF_HUB_CACHE`. The default template has no `[environment]` table.

Mount targets must be absolute, normalized, unique and outside `/opt/venv`, `/opt/rocm`, `/usr/local/bin`, `/opt/amd-ai`, `/dev`, `/proc`, and `/sys`. Reject sources containing newline or comma because Docker `--mount` uses comma-delimited fields. Unknown keys are errors.

- [ ] **Step 5: Run config tests**

Run: `uv run pytest tests/unit/project/test_config.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit project config support**

```bash
git add src/amd_ai/project templates/project/amd-ai-project.toml tests/unit/project/test_config.py
git commit -m "feat: validate project container config"
```

### Task 2: Generate a project scaffold pinned to a parent digest

**Files:**
- Create: `src/amd_ai/project/init.py`
- Create: `templates/project/requirements.in`
- Create: `templates/project/.dockerignore`
- Create: `templates/project/project-entrypoint`
- Test: `tests/unit/project/test_init.py`

- [ ] **Step 1: Write the failing scaffold test**

```python
# tests/unit/project/test_init.py
from amd_ai.project.init import initialize_project
from tests.unit.host.fakes import FakeRunner


def test_scaffold_is_owned_by_project_and_has_no_shared_models(tmp_path):
    runner = FakeRunner.image_digest(
        "rocm-pytorch:7.2.1-py3.12-torch2.9.1",
        "sha256:" + "a" * 64,
    )
    project = initialize_project(
        name="video-lab",
        destination=tmp_path / "video-lab",
        base_profile="stable",
        runner=runner,
    )
    config = (project / "amd-ai-project.toml").read_text()
    assert 'name = "video-lab"' in config
    assert 'base_image = "sha256:' + "a" * 64 + '"' in config
    assert 'base_digest = "sha256:' + "a" * 64 + '"' in config
    combined = "\n".join(path.read_text() for path in project.iterdir() if path.is_file())
    assert "HF_HOME" not in combined
    assert "ComfyUI" not in combined
    assert "[[mounts]]" not in config
```

- [ ] **Step 2: Run and verify scaffold code is missing**

Run: `uv run pytest tests/unit/project/test_init.py -q`

Expected: collection fails for `amd_ai.project.init`.

- [ ] **Step 3: Add deterministic templates**

`requirements.in` contains only this comment:

```text
# Add project-only Python dependencies here. Torch is supplied by the parent image.
```

`.dockerignore` contains:

```text
.git
.venv
.cache
.amd-ai
models
input
output
reports
__pycache__
*.pyc
```

`project-entrypoint` is this executable Python script. It consumes `AMD_AI_PROFILE_STATUS` copied from the parent label at project build time, refuses `experimental` unless `ALLOW_UNVERIFIED=1`, executes the GPU check, and never silently starts in CPU mode:

```python
#!/usr/bin/env python3
import os
import subprocess
import sys


status = os.environ.get("AMD_AI_PROFILE_STATUS", "experimental")
if status != "verified" and os.environ.get("ALLOW_UNVERIFIED") != "1":
    print("experimental or unlabeled Torch profile requires ALLOW_UNVERIFIED=1", file=sys.stderr)
    raise SystemExit(64)
if not sys.argv[1:]:
    print("project command is empty", file=sys.stderr)
    raise SystemExit(64)
check = subprocess.run(
    ["container-check", "--mode", "torch", "--runtime", "--json", "-"],
    check=False,
)
if check.returncode != 0:
    raise SystemExit(check.returncode)
os.execvp(sys.argv[1], sys.argv[1:])
```

- [ ] **Step 4: Implement scaffold generation**

`initialize_project()` maps `stable` to `rocm-pytorch:7.2.1-py3.12-torch2.9.1`; a custom profile must already resolve to a local image label and digest. Resolve the parent with:

```text
docker image inspect --format {{.Id}} <tag>
```

Create the destination with mode `0755`, refuse a nonempty destination, copy templates, replace only the TOML name/image/base profile/base image/base digest fields, and set generated files to the invoking UID/GID when the command runs under sudo. Store the local immutable image ID returned by inspection in both fields; do not require a registry. Do not create model, cache, input or output directories.

- [ ] **Step 5: Run scaffold tests**

Run: `uv run pytest tests/unit/project/test_init.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit scaffold generation**

```bash
git add src/amd_ai/project/init.py templates/project tests/unit/project/test_init.py
git commit -m "feat: scaffold digest-pinned projects"
```

### Task 3: Lock project dependencies and guard the inherited Torch files

**Files:**
- Create: `src/amd_ai/project/dependencies.py`
- Modify: `src/amd_ai/project/init.py`
- Create: `templates/project/Dockerfile`
- Test: `tests/unit/project/test_dependencies.py`
- Test: `tests/container/test_project_dockerfile.py`

- [ ] **Step 1: Write failing dependency and Dockerfile tests**

```python
# tests/unit/project/test_dependencies.py
from amd_ai.project.dependencies import render_torch_constraints


def test_constraints_pin_complete_verified_stack():
    text = render_torch_constraints("profiles/torch/stable.requirements.lock")
    assert "torch==2.9.1" in text
    assert "torchvision==0.24.0" in text
    assert "torchaudio==2.9.0" in text
    assert "triton==3.5.1" in text
    assert len([line for line in text.splitlines() if line]) == 4
```

```python
# tests/container/test_project_dockerfile.py
from pathlib import Path


def test_project_install_cannot_sync_or_replace_parent():
    text = Path("templates/project/Dockerfile").read_text()
    assert "uv pip install" in text
    assert "--constraint /opt/amd-ai/torch-constraints.txt" in text
    assert "torch-manifest.py verify" in text
    assert "AMD_AI_PROFILE_STATUS" in text
    assert "uv pip sync" not in text
    assert "pip uninstall" not in text
```

- [ ] **Step 2: Run and verify files are absent**

Run: `uv run pytest tests/unit/project/test_dependencies.py tests/container/test_project_dockerfile.py -q`

Expected: import failure and `FileNotFoundError`.

- [ ] **Step 3: Implement dependency locking**

`render_torch_constraints()` parses the stable requirements lock and emits exactly the four public versions. `lock_project_dependencies()` runs:

```text
uv pip compile
  --python-version 3.12
  --constraint torch-constraints.txt
  --generate-hashes
  --no-emit-index-url
  --output-file requirements.lock
  requirements.in
```

It rejects a resulting lock that contains a Torch component at a different version or a direct URL for one of the four protected packages. Empty `requirements.in` produces an empty, valid lock with a generated header removed for deterministic output.

Extend `initialize_project()` to write `torch-constraints.txt` from the selected parent profile and call `lock_project_dependencies()` so every completed scaffold contains both files before its first build. Experimental profiles derive the same four constraints from their own validated profile rather than from stable versions.

- [ ] **Step 4: Add the guarded project Dockerfile**

```dockerfile
# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG PROFILE_ID
ARG PROFILE_STATUS
ENV AMD_AI_PROFILE_ID=${PROFILE_ID} \
    AMD_AI_PROFILE_STATUS=${PROFILE_STATUS}

USER root
WORKDIR /workspace
COPY torch-constraints.txt requirements.lock /opt/amd-ai/project-locks/
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    if [ -s /opt/amd-ai/project-locks/requirements.lock ]; then \
      uv pip install \
        --python /opt/venv/bin/python \
        --constraint /opt/amd-ai/project-locks/torch-constraints.txt \
        --require-hashes \
        --requirements /opt/amd-ai/project-locks/requirements.lock; \
    fi \
 && /opt/venv/bin/python /opt/amd-ai/torch-manifest.py verify /opt/amd-ai/torch-manifest.json

COPY --chown=1000:1000 . /workspace
COPY project-entrypoint /usr/local/bin/project-entrypoint
RUN chmod 0755 /usr/local/bin/project-entrypoint \
 && rm -rf /root/.cache/uv
USER 1000:1000
ENTRYPOINT ["/usr/local/bin/project-entrypoint"]
CMD ["bash"]
```

The BuildKit cache mount is not a runtime volume and is absent from final layers. The manifest created in the stable parent verifies every protected distribution file after project dependency installation.

- [ ] **Step 5: Run dependency and template tests**

Run: `uv run pytest tests/unit/project/test_dependencies.py tests/container/test_project_dockerfile.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit dependency protection**

```bash
git add src/amd_ai/project/dependencies.py src/amd_ai/project/init.py templates/project/Dockerfile tests/unit/project/test_dependencies.py tests/container/test_project_dockerfile.py
git commit -m "feat: protect inherited Torch project layer"
```

### Task 4: Validate runtime devices, GIDs, mounts and shared memory

**Files:**
- Create: `src/amd_ai/project/runtime.py`
- Test: `tests/unit/project/test_runtime.py`

- [ ] **Step 1: Write failing runtime policy tests**

```python
# tests/unit/project/test_runtime.py
from pathlib import Path

import pytest

from amd_ai.project.config import MountConfig
from amd_ai.project.runtime import RuntimePolicyError, compute_shm_gib, discover_gpu_access, mount_argv


def test_nominal_128_gib_uses_16_gib_shm():
    assert compute_shm_gib(mem_total_kib=131015488) == 16
    assert compute_shm_gib(mem_total_kib=31 * 1024**2) == 4


def test_device_groups_are_sorted_and_deduplicated(tmp_path, monkeypatch):
    access = discover_gpu_access(
        kfd=Path("/dev/kfd"),
        dri=Path("/dev/dri"),
        stat_gids={"/dev/kfd": 109, "/dev/dri/renderD128": 110, "/dev/dri/renderD129": 110},
    )
    assert access.group_ids == (109, 110)


def test_missing_kfd_is_reported_before_docker():
    with pytest.raises(RuntimePolicyError, match="/dev/kfd"):
        discover_gpu_access(
            kfd=Path("/dev/kfd"),
            dri=Path("/dev/dri"),
            stat_gids={},
        )


def test_mount_argv_preserves_explicit_read_only(tmp_path):
    source = tmp_path / "models"
    source.mkdir()
    mount = MountConfig(source, Path("/models"), True)
    assert mount_argv((mount,)) == (
        "--mount",
        f"type=bind,src={source},dst=/models,readonly",
    )
```

- [ ] **Step 2: Run and verify runtime module is absent**

Run: `uv run pytest tests/unit/project/test_runtime.py -q`

Expected: collection fails for `amd_ai.project.runtime`.

- [ ] **Step 3: Implement deterministic runtime policy**

Normalize `MemTotal` with the same `ceil(memory_gib / 8) * 8` algorithm as TTM. Compute shared memory as `max(4, min(16, nominal_gib // 8))`; values are whole GiB. An explicit project value overrides this calculation.

`discover_gpu_access()` requires `/dev/kfd`, `/dev/dri`, and at least one `renderD*` node. It returns `/dev/kfd` and `/dev/dri` as device mounts plus sorted unique GIDs from KFD and render nodes. It reports missing host devices before Docker starts.

`mount_argv()` verifies each source exists, keeps the configured read-only flag, and emits one `--mount` pair per item. It does not create sources. No default call site supplies a model or cache mount.

- [ ] **Step 4: Run runtime tests**

Run: `uv run pytest tests/unit/project/test_runtime.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit runtime validation**

```bash
git add src/amd_ai/project/runtime.py tests/unit/project/test_runtime.py
git commit -m "feat: validate project GPU runtime access"
```

### Task 5: Build derived project images with a reproducible fingerprint

**Files:**
- Create: `src/amd_ai/project/build.py`
- Test: `tests/unit/project/test_build.py`

- [ ] **Step 1: Write failing fingerprint and build argv tests**

```python
# tests/unit/project/test_build.py
from pathlib import Path

from amd_ai.project.build import build_context_fingerprint, project_build_argv


def test_fingerprint_changes_with_lock_not_ignored_models(tmp_path):
    (tmp_path / "requirements.lock").write_text("alpha==1.0 --hash=sha256:" + "a" * 64 + "\n")
    (tmp_path / "models").mkdir()
    (tmp_path / "models/model.bin").write_bytes(b"large")
    first = build_context_fingerprint(tmp_path)
    (tmp_path / "models/model.bin").write_bytes(b"changed")
    assert build_context_fingerprint(tmp_path) == first
    (tmp_path / "requirements.lock").write_text("alpha==1.1 --hash=sha256:" + "b" * 64 + "\n")
    assert build_context_fingerprint(tmp_path) != first


def test_build_uses_parent_digest():
    argv = project_build_argv(
        context=Path("projects/demo"),
        image="demo:runtime",
        base_image="sha256:" + "a" * 64,
        base_digest="sha256:" + "a" * 64,
        profile_id="rocm-7.2.1-py3.12-torch-2.9.1",
        profile_status="verified",
        fingerprint="f" * 64,
    )
    assert "BASE_IMAGE=sha256:" + "a" * 64 in argv
    assert "PROFILE_STATUS=verified" in argv
    assert "org.amd-ai.project.fingerprint=" + "f" * 64 in argv
```

- [ ] **Step 2: Run and verify project build module is absent**

Run: `uv run pytest tests/unit/project/test_build.py -q`

Expected: collection fails for `amd_ai.project.build`.

- [ ] **Step 3: Implement fingerprinting and BuildKit argv**

Fingerprint only Dockerfile, project-entrypoint, `amd-ai-project.toml`, `requirements.in`, `requirements.lock`, `torch-constraints.txt`, and regular source files not excluded by `.dockerignore`. Hash normalized relative path, mode and contents in lexical order.

Use:

```text
docker buildx build --load --platform linux/amd64
  --build-arg BASE_IMAGE=<local sha256 image id>
  --build-arg PROFILE_ID=<inherited profile id>
  --build-arg PROFILE_STATUS=<verified or experimental>
  --label org.amd-ai.project.fingerprint=<sha256>
  --label org.amd-ai.base.digest=<digest>
  --tag <project image>
  <project directory>
```

Before reusing an image, compare both labels with current values. Read profile ID/status from the parent image labels and pass them as build args so `project-entrypoint` can enforce the same gate inside the container. A mismatch rebuilds unless the user passed `--no-build`, in which case return a stale-image error.

- [ ] **Step 4: Run build tests**

Run: `uv run pytest tests/unit/project/test_build.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit project image building**

```bash
git add src/amd_ai/project/build.py tests/unit/project/test_build.py
git commit -m "feat: build fingerprinted project images"
```

### Task 6: Gate image status and construct the safe Docker run command

**Files:**
- Create: `src/amd_ai/project/run.py`
- Test: `tests/unit/project/test_run.py`

- [ ] **Step 1: Write failing stable, experimental and debug argv tests**

```python
# tests/unit/project/test_run.py
import pytest

from amd_ai.project.run import UnverifiedImage, build_run_argv, require_profile_allowed
from tests.unit.project.fakes import project_config, runtime_access


def test_experimental_image_requires_explicit_environment():
    with pytest.raises(UnverifiedImage):
        require_profile_allowed("experimental", {})
    require_profile_allowed("experimental", {"ALLOW_UNVERIFIED": "1"})


def test_normal_run_is_unprivileged_and_uses_private_ipc(tmp_path):
    argv = build_run_argv(
        config=project_config(tmp_path),
        access=runtime_access(),
        uid=1000,
        gid=1000,
        shm_gib=16,
        environment={},
        terminal=False,
    )
    assert "--privileged" not in argv
    assert "--ipc=host" not in argv
    assert "SYS_PTRACE" not in argv
    assert "--tty" not in argv and "--interactive" not in argv
    assert ("--shm-size", "16g") == tuple(argv[argv.index("--shm-size"):argv.index("--shm-size") + 2])
    assert "109" in argv and "110" in argv


def test_debug_adds_only_ptrace_and_seccomp(tmp_path):
    argv = build_run_argv(
        config=project_config(tmp_path, debug=True),
        access=runtime_access(),
        uid=1000,
        gid=1000,
        shm_gib=16,
        environment={},
        terminal=True,
    )
    assert "SYS_PTRACE" in argv
    assert "seccomp=unconfined" in argv
    assert "--tty" in argv and "--interactive" in argv
    assert "--privileged" not in argv
```

- [ ] **Step 2: Run and verify run orchestration is missing**

Run: `uv run pytest tests/unit/project/test_run.py -q`

Expected: collection fails for `amd_ai.project.run`.

- [ ] **Step 3: Implement label and runtime gates**

Inspect these inherited labels before running: profile ID, profile status, ROCm version, Python version, Torch version, base digest and project fingerprint. Missing profile status is treated as `experimental`. Require `ALLOW_UNVERIFIED` to equal the exact string `1` for experimental images and pass the same variable into the container. Emit each user-declared config environment value as a separate `--env NAME=value` argument after validation; redact values whose names contain `TOKEN`, `SECRET`, `PASSWORD`, or `KEY` from dry-run/log output.

The normal flat argv contains:

```text
docker run --rm [--interactive --tty when stdin and stdout are terminals]
--device /dev/kfd --device /dev/dri
--group-add <each actual gid>
--user <host uid>:<host gid>
--shm-size <computed>g
--workdir /workspace
--env HOME=/workspace/.amd-ai/home
--mount type=bind,src=<project>,dst=/workspace
<each user-declared mount>
<project image>
<configured command>
```

Create `.amd-ai/home` under the project with the host UID/GID before launch. Add `--interactive --tty` only when both stdin and stdout are terminals; noninteractive jobs receive neither flag. Do not emit `--ipc=host`, `--privileged`, `HF_HOME`, `HF_HUB_CACHE` or a model mount. Debug adds only `--cap-add SYS_PTRACE --security-opt seccomp=unconfined`.

- [ ] **Step 4: Run run-policy tests**

Run: `uv run pytest tests/unit/project/test_run.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit run orchestration**

```bash
git add src/amd_ai/project/run.py tests/unit/project/test_run.py tests/unit/project/fakes.py
git commit -m "feat: run isolated GPU project containers"
```

### Task 7: Wire `project-init` and `project-run` commands

**Files:**
- Modify: `src/amd_ai/cli.py`
- Create: `bin/project-init`
- Create: `bin/project-run`
- Test: `tests/cli/test_project_commands.py`

- [ ] **Step 1: Write failing CLI tests**

```python
# tests/cli/test_project_commands.py
from amd_ai.cli import main


def test_project_init_creates_named_directory(tmp_path, fake_docker):
    assert main(["project-init", "demo", "--directory", str(tmp_path / "demo")]) == 0
    assert (tmp_path / "demo/amd-ai-project.toml").is_file()


def test_project_run_dry_run_prints_shell_escaped_command(tmp_path, project_fixture, capsys):
    assert main(["project-run", str(project_fixture), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "docker run" in output
    assert "--device /dev/kfd" in output
    assert "--privileged" not in output
```

- [ ] **Step 2: Run and verify commands are unrecognized**

Run: `uv run pytest tests/cli/test_project_commands.py -q`

Expected: argparse rejects both command names.

- [ ] **Step 3: Add CLI grammar and wrappers**

```text
project-init NAME [--directory PATH] [--base-profile PROFILE_ID]
project-run PROJECT [--build] [--no-build] [--dry-run] [--debug] [--shm-size-gib N]
```

`project-run` builds automatically only when the image is missing or fingerprint/base digest is stale. `--build` forces a rebuild; `--no-build` rejects missing/stale images; the two flags are mutually exclusive. `--dry-run` performs all validation and prints `shlex.join(argv)` without starting Docker.

```bash
#!/usr/bin/env bash
# bin/project-init
set -euo pipefail
exec "$(dirname "$0")/_dispatch" project-init "$@"
```

```bash
#!/usr/bin/env bash
# bin/project-run
set -euo pipefail
exec "$(dirname "$0")/_dispatch" project-run "$@"
```

Run `chmod +x bin/project-init bin/project-run`.

- [ ] **Step 4: Run project CLI tests**

Run: `uv run pytest tests/cli/test_project_commands.py -q`

Expected: all tests pass with fake Docker and fake device stats.

- [ ] **Step 5: Commit command wiring**

```bash
git add src/amd_ai/cli.py bin/project-init bin/project-run tests/cli/test_project_commands.py
git commit -m "feat: expose project container workflow"
```

### Task 8: Prove parent-layer reuse and document mount freedom

**Files:**
- Create: `tests/container/test_project_layers.py`
- Create: `docs/project-workflow.md`

- [ ] **Step 1: Add two minimal fixture projects**

Create `tests/fixtures/projects/alpha` with `requirements.in` containing `safetensors==0.5.3` and `beta` containing `einops==0.8.1`. Generate locks through the project dependency command; neither input mentions Torch.

- [ ] **Step 2: Build both projects and assert layer-prefix identity**

The test obtains `RootFS.Layers` arrays from `docker image inspect` for the stable parent, alpha and beta. Assert every parent layer equals the corresponding prefix element in both project arrays. Run `torch-manifest.py verify` in both derived images and assert `pip show torch` reports the same location and version.

Run:

```bash
uv run pytest tests/container/test_project_layers.py -q
```

Expected: pass; derived images add project layers but do not add a replacement Torch distribution.

- [ ] **Step 3: Write the project workflow document**

Include these concrete examples:

```bash
./bin/project-init comfy-lab
./bin/project-run comfy-lab
ALLOW_UNVERIFIED=1 ./bin/project-run custom-torch-lab
```

Show optional TOML mounts for a private model directory, a user-chosen shared read-only model directory, a writable output directory and a user-chosen Hugging Face cache, including an explicit user-written `[environment]` table with `HF_HOME`. State that none is generated automatically. Document project-local `.amd-ai/home`, custom commands, debug mode, image rebuild behavior, parent-layer disk accounting and safe image/cache cleanup preview.

- [ ] **Step 4: Run the complete project suite**

Run:

```bash
uv run pytest tests/unit/project tests/cli/test_project_commands.py tests/container/test_project_dockerfile.py tests/container/test_project_layers.py -q
```

Expected: zero failures; Docker-dependent tests may skip only when Docker is absent.

- [ ] **Step 5: Commit acceptance tests and documentation**

```bash
git add tests/fixtures/projects tests/container/test_project_layers.py docs/project-workflow.md
git commit -m "test: prove project layer reuse and mount isolation"
```
