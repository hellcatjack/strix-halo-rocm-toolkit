# Docker Capability and OEM 6.17 Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship toolkit `v0.3.0` with runtime-only Docker checks, explicit Buildx gates, an OEM 6.17 host baseline, and two independently resumable reboot checkpoints that protect Ubuntu Desktop systems from kernel and TTM regressions.

**Architecture:** Keep the existing image `Docker` facade, but make detection runtime-only and add Buildx guards only at actual build/publication boundaries. Extend the host snapshot with package, candidate, Buildx, and display facts; split host preparation into typed `kernel` and `tuning` plans; then model those plans as separate installer stages and reboot IDs in state schema 3. Preserve and revalidate the immutable ROCm 7.2.1/PyTorch 2.9.1 images instead of rebuilding them.

**Tech Stack:** Python 3.12 standard library, `dataclasses`, `StrEnum`, Docker Engine/Buildx, Ubuntu APT/systemd/GRUB conventions, pytest 8.4, uv, Bash, Markdown.

---

## File Responsibility Map

| File | Responsibility |
| --- | --- |
| `src/amd_ai/image/build.py` | Runtime Docker detection and explicit image Buildx gates |
| `src/amd_ai/project/build.py`, `src/amd_ai/cli.py` | Project/publication Buildx gates |
| `src/amd_ai/host/models.py`, `parsers.py`, `probe.py` | Typed host capabilities and collection |
| `src/amd_ai/host/policy.py` | OEM 6.17 branch classification |
| `src/amd_ai/host/prepare.py`, `apply.py`, `verify.py` | Kernel/tuning plans, mutation, and separate verification |
| `src/amd_ai/installer/models.py`, `state.py` | New stages and schema-3 migration |
| `src/amd_ai/installer/actions.py`, `privileged.py` | Phase-aware root protocol |
| `src/amd_ai/installer/workflow.py`, `progress.py` | Dual checkpoints, resume, recovery output |
| `profiles/host/tested-kernels.json` | Hardware-gated 6.17.0-1028 promotion |
| `README.md`, `docs/host-operations.md` | v0.3.0 installation and recovery manual |

## Task 1: Decouple Docker Runtime Detection from Buildx

**Files:**
- Modify: `src/amd_ai/image/build.py:57-101`
- Test: `tests/unit/image/test_build.py:286-365`
- Test: `tests/cli/test_image_commands.py`

- [ ] **Step 1: Write the failing detection test**

Add to `tests/unit/image/test_build.py`:

```python
def test_docker_detection_does_not_probe_buildx(monkeypatch):
    calls: list[tuple[str, ...]] = []

    def fake_completed(args, *, check=True):
        del check
        command = tuple(args)
        calls.append(command)
        assert command == (
            "docker", "info", "--format", "{{.ServerVersion}}"
        )
        return subprocess.CompletedProcess(command, 0, "29.1.3\n", "")

    monkeypatch.setattr(build, "_completed", fake_completed)
    docker = build.Docker.detect()

    assert docker.prefix == ("docker",)
    assert docker.server_version == "29.1.3"
    assert calls == [("docker", "info", "--format", "{{.ServerVersion}}")]
```

- [ ] **Step 2: Prove it fails on the current hidden Buildx dependency**

Run:

```bash
uv run pytest tests/unit/image/test_build.py::test_docker_detection_does_not_probe_buildx -q
```

Expected: `FAIL` because detection issues `docker buildx version` or lacks `server_version`.

- [ ] **Step 3: Implement runtime-only detection**

Replace the constructor/detector with:

```python
class Docker:
    def __init__(self, prefix: Sequence[str], server_version: str) -> None:
        self.prefix = tuple(prefix)
        self.server_version = server_version

    @classmethod
    def detect(cls) -> Docker:
        evidence: list[str] = []
        for prefix in (("docker",), ("sudo", "-n", "docker")):
            result = _completed(
                (*prefix, "info", "--format", "{{.ServerVersion}}"),
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return cls(prefix, result.stdout.strip())
            detail = result.stderr.strip() or result.stdout.strip()
            evidence.append(f"{' '.join(prefix)}: {detail or result.returncode}")
        raise BuildError(
            "Docker runtime is unavailable to the current user and sudo -n; "
            + "; ".join(evidence)
        )
```

Update direct constructor uses to supply a version. Fakes exposing only `.prefix` remain valid.

- [ ] **Step 4: Run all runtime-only consumers**

```bash
uv run pytest tests/unit/image/test_build.py tests/unit/doctor \
  tests/cli/test_image_commands.py tests/cli/test_doctor_commands.py -q
```

Expected: all pass; `container-check`, doctor, and release pull do not probe Buildx.

- [ ] **Step 5: Commit**

```bash
git add src/amd_ai/image/build.py tests/unit/image/test_build.py \
  tests/unit/doctor tests/cli/test_image_commands.py tests/cli/test_doctor_commands.py
git commit -m "fix: decouple Docker runtime checks from Buildx"
```

## Task 2: Gate Actual Builds and Publication on Buildx

**Files:**
- Modify: `src/amd_ai/image/build.py:312-488`
- Modify: `src/amd_ai/project/build.py:172-244`
- Modify: `src/amd_ai/cli.py:648-797`
- Test: `tests/unit/image/test_build.py`
- Test: `tests/unit/project/test_build.py`
- Test: `tests/cli/test_release_commands.py`

- [ ] **Step 1: Add failing Buildx guard tests**

```python
def test_require_buildx_reports_healthy_daemon(monkeypatch):
    docker = build.Docker(("docker",), "29.1.3")
    missing = subprocess.CompletedProcess(
        ("docker", "buildx", "version"), 1, "",
        "docker: unknown command: docker buildx",
    )
    monkeypatch.setattr(docker, "capture", lambda args, check=False: missing)
    with pytest.raises(build.BuildError, match="29.1.3.*host repair"):
        docker.require_buildx()
```

In `tests/unit/project/test_build.py`, add one case where a matching image is reused without a Buildx command and one forced-build case where a failed `docker buildx version` raises `ProjectBuildError` before `docker buildx build`.

- [ ] **Step 2: Run the focused tests and observe failure**

```bash
uv run pytest tests/unit/image/test_build.py tests/unit/project/test_build.py -q
```

Expected: new guard tests fail because no explicit API exists.

- [ ] **Step 3: Add explicit image Buildx methods**

Add to `Docker`:

```python
def buildx_version(self) -> str | None:
    result = self.capture(("buildx", "version"), check=False)
    return result.stdout.strip() if result.returncode == 0 else None

def require_buildx(self) -> str:
    result = self.capture(("buildx", "version"), check=False)
    version = result.stdout.strip()
    if result.returncode != 0 or not version:
        evidence = result.stderr.strip() or result.stdout.strip()
        raise BuildError(
            "Docker Buildx is unavailable while Docker runtime "
            f"{self.server_version} is healthy; run the toolkit host repair "
            f"before building ({evidence or 'no Buildx version'})"
        )
    return version
```

Call it before lock/download preparation in both image builders. For `prune_images(apply=True)`, call it before deleting the first image; preview remains runtime-only.

- [ ] **Step 4: Gate project builds after reuse has been ruled out**

Add to `project/build.py` and call immediately before the Buildx build command:

```python
def _require_buildx(runner: Runner, docker_prefix: Sequence[str]) -> str:
    result = runner.run([*docker_prefix, "buildx", "version"], check=False)
    version = result.stdout.strip()
    if result.returncode != 0 or not version:
        evidence = result.stderr.strip() or result.stdout.strip()
        raise ProjectBuildError(
            "Docker Buildx is required to build the project image; "
            "run the toolkit host repair before retrying: "
            + (evidence or "Buildx version unavailable")
        )
    return version
```

`--no-build` and a valid reused project image must not call this helper.

- [ ] **Step 5: Gate publication-specific imagetools operations**

Call `docker.require_buildx()` before push-only or final publication paths that invoke `buildx imagetools`. Keep release verification and dry-run local evidence runtime-only.

- [ ] **Step 6: Run focused regressions**

```bash
uv run pytest tests/unit/image/test_build.py tests/unit/image/test_publish.py \
  tests/unit/project/test_build.py tests/cli/test_release_commands.py \
  tests/cli/test_project_commands.py -q
```

Expected: all pass and every actual build fails before expensive work when Buildx is absent.

- [ ] **Step 7: Commit**

```bash
git add src/amd_ai/image/build.py src/amd_ai/project/build.py src/amd_ai/cli.py \
  tests/unit/image tests/unit/project/test_build.py \
  tests/cli/test_release_commands.py tests/cli/test_project_commands.py
git commit -m "feat: gate Docker builds on explicit Buildx capability"
```

## Task 3: Extend Host Capability Collection

**Files:**
- Modify: `src/amd_ai/host/models.py:22-43`
- Modify: `src/amd_ai/host/parsers.py:78-94`
- Modify: `src/amd_ai/host/probe.py:81-178`
- Modify: `src/amd_ai/host/policy.py:205-230`
- Modify: `tests/unit/host/fakes.py`
- Modify: `tests/fixtures/host/healthy/commands.json`
- Test: `tests/unit/host/test_parsers.py`
- Test: `tests/unit/host/test_probe.py`

- [ ] **Step 1: Add failing parser and probe tests**

```python
def test_parse_apt_candidate():
    assert parse_apt_candidate("  Candidate: 6.17.0-1028.28\n") == "6.17.0-1028.28"
    assert parse_apt_candidate("  Candidate: (none)\n") is None


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        ({"docker-ce-cli"}, DockerDistribution.DOCKER_CE),
        ({"docker.io"}, DockerDistribution.UBUNTU_DOCKER_IO),
        ({"docker.io", "docker-ce"}, DockerDistribution.MIXED),
        (set(), DockerDistribution.EXTERNAL),
    ],
)
def test_classify_docker_distribution(names, expected):
    packages = tuple(InstalledPackage(name, "1") for name in sorted(names))
    assert classify_docker_distribution(packages, runtime_available=True) is expected
```

Extend `test_probe_collects_target_snapshot()` to assert Buildx version, `DOCKER_CE`, candidate `6.17.0-1028.28`, and loaded/active display manager.

- [ ] **Step 2: Run and verify the missing model facts**

```bash
uv run pytest tests/unit/host/test_parsers.py tests/unit/host/test_probe.py -q
```

Expected: failures for missing enum, parser, commands, and snapshot fields.

- [ ] **Step 3: Add typed facts and parsers**

Add:

```python
class DockerDistribution(StrEnum):
    DOCKER_CE = "docker-ce"
    UBUNTU_DOCKER_IO = "ubuntu-docker-io"
    MIXED = "mixed"
    EXTERNAL = "external"
    MISSING = "missing"
```

Extend `HostSnapshot` with `docker_buildx_version`, `docker_buildx_error`, `docker_distribution`, `kernel_oem_617_candidate`, `display_manager_loaded`, and `display_manager_active`. Pure parsers must return `MISSING` whenever the daemon probe fails.

- [ ] **Step 4: Collect each new command once**

```python
buildx_result = self._run([*self._docker_prefix, "buildx", "version"])
candidate_result = self._run(["apt-cache", "policy", "linux-oem-6.17"])
display_load = self._run(
    ["systemctl", "show", "display-manager", "--property=LoadState", "--value"]
)
display_active = self._run(["systemctl", "is-active", "display-manager"])
```

Store bounded Buildx error evidence and expose all facts in report JSON.

- [ ] **Step 5: Update fakes and fixture commands**

Make `FakeRunner.healthy_target()` and `tests/fixtures/host/healthy/commands.json` identify Docker CE, healthy Buildx, OEM candidate, and an active loaded display manager.

- [ ] **Step 6: Run collection tests**

```bash
uv run pytest tests/unit/host/test_parsers.py tests/unit/host/test_probe.py \
  tests/cli/test_host_commands.py -q
```

Expected: all pass and `DockerDistribution` serializes as its string value.

- [ ] **Step 7: Commit**

```bash
git add src/amd_ai/host/models.py src/amd_ai/host/parsers.py \
  src/amd_ai/host/probe.py src/amd_ai/host/policy.py tests/unit/host \
  tests/fixtures/host/healthy tests/cli/test_host_commands.py
git commit -m "feat: inventory Docker Buildx and OEM kernel readiness"
```

## Task 4: Enforce the OEM 6.17 Branch Policy

**Files:**
- Modify: `src/amd_ai/host/policy.py:13-201`
- Test: `tests/unit/host/test_policy.py`
- Defer until Task 12: `profiles/host/tested-kernels.json`

- [ ] **Step 1: Add branch-policy tests with a temporary qualified list**

```python
def tested_kernel_file(tmp_path: Path) -> Path:
    path = tmp_path / "tested-kernels.json"
    path.write_text(
        json.dumps({"schema_version": 1, "kernels": ["6.17.0-1028-oem"]}),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "kernel", ["6.8.0-31-generic", "6.14.0-1020-oem", "6.17.0-14-generic"]
)
def test_old_or_non_oem_kernel_requires_oem_617(kernel):
    report = evaluate_preflight(healthy_snapshot(kernel=kernel))
    assert report.status is Status.CHANGE_REQUIRED
    assert "HOST.OEM_617_REQUIRED" in finding_codes(report)


@pytest.mark.parametrize("kernel", ["6.17.0-1029-oem", "6.19.0-1001-oem"])
def test_unlisted_or_future_oem_is_unverified(kernel, tmp_path):
    report = evaluate_preflight(
        healthy_snapshot(kernel=kernel),
        tested_kernels_path=tested_kernel_file(tmp_path),
    )
    assert report.status is Status.UNVERIFIED
```

Also assert exact `6.17.0-1028-oem` passes with the temporary file.

- [ ] **Step 2: Run and observe the old 6.14 minimum**

```bash
uv run pytest tests/unit/host/test_policy.py -q
```

Expected: failures show `(6, 14, 0, 1018)` and old remediation.

- [ ] **Step 3: Implement branch semantics**

```python
TARGET_OEM_BRANCH = (6, 17)
TARGET_OEM_PACKAGE = "linux-oem-6.17"


def _oem_branch(kernel: str) -> tuple[int, int] | None:
    parsed = _parse_oem_kernel(kernel)
    return None if parsed is None else parsed[:2]
```

Missing/non-OEM/older branches emit `HOST.OEM_617_REQUIRED` with `CHANGE_REQUIRED`. Branch 6.17 and newer consult the tested list; unlisted versions are `UNVERIFIED`. Never request a downgrade for a newer OEM branch.

- [ ] **Step 4: Block only an unavailable upgrade candidate**

When an upgrade is required, require `kernel_oem_617_candidate` to start with `6.17.`. Otherwise emit blocking `HOST.OEM_617_CANDIDATE` with `sudo apt update` and Ubuntu Noble repository remediation.

- [ ] **Step 5: Run policy and CLI tests**

```bash
uv run pytest tests/unit/host/test_policy.py tests/cli/test_host_commands.py -q
```

Expected: all pass; production tested-kernel promotion remains deferred.

- [ ] **Step 6: Commit**

```bash
git add src/amd_ai/host/policy.py tests/unit/host/test_policy.py \
  tests/cli/test_host_commands.py
git commit -m "feat: require the Ubuntu OEM 6.17 kernel branch"
```

## Task 5: Split Kernel and Tuning Plans with Matching Buildx Repair

**Files:**
- Modify: `src/amd_ai/host/models.py:46-61`
- Modify: `src/amd_ai/host/adapters/base.py`
- Modify: `src/amd_ai/host/adapters/ubuntu_2404.py`
- Modify: `src/amd_ai/host/prepare.py:36-263`
- Modify: `src/amd_ai/host/apply.py:132-355`
- Test: `tests/unit/host/test_prepare.py`
- Test: `tests/unit/host/test_apply.py`

- [ ] **Step 1: Add failing phase-isolation tests**

```python
def test_kernel_plan_contains_no_ttm_or_docker_actions():
    plan = create_prepare_plan(
        healthy_snapshot(kernel="6.14.0-1020-oem"),
        target_user="customer",
        phase=HostPlanPhase.KERNEL,
    )
    codes = action_codes(plan)
    assert plan.phase is HostPlanPhase.KERNEL
    assert "APT.INSTALL_OEM_617" in codes
    assert "HOST.REBOOT" in codes
    assert not any(code.startswith("TTM.") for code in codes)
    assert not any(code.startswith("DOCKER.") for code in codes)


def test_tuning_plan_never_installs_a_kernel():
    plan = create_prepare_plan(
        healthy_snapshot(kernel="6.17.0-1028-oem"),
        target_user="customer",
        phase=HostPlanPhase.TUNING,
    )
    assert plan.phase is HostPlanPhase.TUNING
    assert "APT.INSTALL_OEM_617" not in action_codes(plan)
```

Add parametrized expected actions for Docker CE, `docker.io`, mixed, external, and missing distributions.

- [ ] **Step 2: Run and verify the current combined plan fails isolation**

```bash
uv run pytest tests/unit/host/test_prepare.py -q
```

Expected: failures because the plan has no phase and always combines kernel, Docker, and TTM.

- [ ] **Step 3: Add typed phases and split planner entry points**

```python
class HostPlanPhase(StrEnum):
    KERNEL = "kernel"
    TUNING = "tuning"


@dataclass(frozen=True)
class PreparePlan:
    phase: HostPlanPhase
    supported: bool
    target_user: str
    actions: tuple[PlannedAction, ...]
    reboot_required: bool
```

Expose `create_kernel_prepare_plan()` and `create_tuning_prepare_plan()`. Keep this complete public dispatcher:

```python
def create_prepare_plan(
    snapshot: HostSnapshot,
    *,
    target_user: str,
    memory_gib: int | None = None,
    phase: HostPlanPhase | None = None,
) -> PreparePlan:
```

When `phase` is `None`, select kernel only when the current branch needs transition; adapter methods accept the resolved explicit phase.

- [ ] **Step 4: Implement the branch-pinned kernel action**

The plan uses internal `APT.INSTALL_OEM_617`. Privileged apply runs `apt-get update`, reruns `apt-cache policy linux-oem-6.17`, rejects candidate drift, then executes exactly:

```python
(
    "apt-get", "install", "-y", "linux-oem-6.17", "linux-firmware"
)
```

Append `HOST.REBOOT` only when installing the kernel or removing confirmed `amdgpu-dkms`. Never remove old kernel packages, call `apt autoremove`, or change GRUB defaults.

- [ ] **Step 5: Implement package-matched tuning repair**

Use this complete decision:

```python
if snapshot.docker_version is None:
    code = "DOCKER.INSTALL_IF_MISSING"
elif snapshot.docker_buildx_version is not None:
    code = None
elif snapshot.docker_distribution is DockerDistribution.DOCKER_CE:
    code = "DOCKER.INSTALL_BUILDX_PLUGIN"
elif snapshot.docker_distribution is DockerDistribution.UBUNTU_DOCKER_IO:
    code = "DOCKER.INSTALL_UBUNTU_BUILDX"
elif snapshot.docker_distribution is DockerDistribution.MIXED:
    raise HostPlanningError("mixed Docker CE and docker.io packages")
else:
    raise HostPlanningError(
        "Docker runtime is externally managed; install matching Buildx manually"
    )
```

The Ubuntu action installs only `docker-buildx`; the CE action installs only `docker-buildx-plugin` through the fingerprint-verified Docker repository. Reprobe daemon and Buildx after either action.

- [ ] **Step 6: Make the second reboot depend only on TTM persistence**

Set tuning `reboot_required=True` and add `HOST.REBOOT` only when `TTM.SET_AI_MAX` is present. Group or Docker-only changes do not require a reboot. Insert Docker group authorization after any Docker action.

Every nonempty phase starts with its own `BACKUP.SNAPSHOT`; the tuning phase must not reuse the pre-kernel backup as authorization for later host changes.

- [ ] **Step 7: Test candidate drift and post-install reprobes**

Add `test_install_oem_617_rejects_candidate_drift_before_apt_install()` and `test_buildx_repair_reprobes_runtime_and_plugin()` to `test_apply.py`. Assert wrong candidate never reaches `apt-get install`.

```bash
uv run pytest tests/unit/host/test_prepare.py tests/unit/host/test_apply.py -q
```

Expected: all phase, backup, package, and no-autoremove tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/amd_ai/host/models.py src/amd_ai/host/adapters \
  src/amd_ai/host/prepare.py src/amd_ai/host/apply.py tests/unit/host
git commit -m "feat: split kernel activation from host tuning"
```

## Task 6: Add the Desktop-Safe Kernel Verification Gate

**Files:**
- Modify: `src/amd_ai/host/verify.py:14-126`
- Test: `tests/unit/host/test_verify.py`

- [ ] **Step 1: Add failing kernel-only verification tests**

```python
def test_kernel_checkpoint_ignores_ttm_until_tuning():
    report = evaluate_kernel_reboot(
        healthy_snapshot(kernel="6.17.0-1028-oem", ttm_pages_limit=1),
        display_manager_was_active=True,
    )
    assert report.status is Status.PASS
    assert "HOST.TTM_MISMATCH" not in finding_codes(report)


def test_kernel_checkpoint_blocks_display_regression():
    report = evaluate_kernel_reboot(
        healthy_snapshot(display_manager_active=False),
        display_manager_was_active=True,
    )
    assert report.status is Status.BLOCKED
    assert "HOST.DISPLAY_MANAGER_INACTIVE" in finding_codes(report)


@pytest.mark.parametrize(
    "line",
    [
        "amdgpu 0000:c5:00.0: amdgpu: Fatal error during GPU init",
        "amdgpu: probe of 0000:c5:00.0 failed with error -22",
    ],
)
def test_kernel_checkpoint_blocks_fatal_gpu_init(line):
    report = evaluate_kernel_reboot(
        healthy_snapshot(dmesg=line), display_manager_was_active=False
    )
    assert report.status is Status.BLOCKED
    assert "GPU.INIT_FATAL" in finding_codes(report)
```

- [ ] **Step 2: Run and confirm final verification currently enforces TTM too early**

```bash
uv run pytest tests/unit/host/test_verify.py -q
```

Expected: import failure or TTM mismatch at the kernel checkpoint.

- [ ] **Step 3: Implement `evaluate_kernel_reboot()`**

Build from preflight plus current-boot log checks, but never call `compute_ttm_plan()`. Add `GPU.INIT_FATAL` patterns. When the display manager was active before apply, require it loaded and active after reboot; leave headless hosts unchanged.

- [ ] **Step 4: Keep final verification strict and deduplicate scanning**

Retain `evaluate_post_reboot()` for TTM. Move common GPU log scanning into one helper so both reports use identical patterns and do not duplicate findings.

- [ ] **Step 5: Run verification regressions**

```bash
uv run pytest tests/unit/host/test_verify.py \
  tests/unit/qualification/test_kernel_log.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/amd_ai/host/verify.py tests/unit/host/test_verify.py \
  tests/unit/qualification/test_kernel_log.py
git commit -m "feat: verify desktop and GPU before TTM tuning"
```

## Task 7: Migrate Installer State to Schema 3

**Files:**
- Modify: `src/amd_ai/installer/models.py:13-340`
- Modify: `src/amd_ai/installer/state.py:20-390`
- Test: `tests/unit/installer/test_models.py`
- Test: `tests/unit/installer/test_state.py`

- [ ] **Step 1: Write literal schema-2 migration tests**

```python
def test_schema_two_full_state_restarts_host_audit_and_keeps_images(tmp_path):
    path = write_schema_two_state(tmp_path, mode="full", current_stage="COMPLETE")
    migrated = load_state(path)
    assert migrated is not None
    assert migrated.schema_version == 3
    assert migrated.current_stage is InstallStage.BOOTSTRAP
    assert migrated.completed_stage_input_digests == {}
    assert migrated.torch_image_reference.endswith("@sha256:" + "c" * 64)
    assert migrated.kernel_plan_digest is None
    assert migrated.host_plan_digest is None


def test_schema_two_container_state_keeps_compatible_position(tmp_path):
    path = write_schema_two_state(
        tmp_path, mode="container", current_stage="IMAGE_VERIFY"
    )
    migrated = load_state(path)
    assert migrated is not None
    assert migrated.current_stage is InstallStage.IMAGE_VERIFY
    assert InstallStage.RELEASE_RESOLVE.value in migrated.completed_stage_input_digests
```

- [ ] **Step 2: Run and verify schema 2 lacks checkpoint fields**

```bash
uv run pytest tests/unit/installer/test_models.py tests/unit/installer/test_state.py -q
```

Expected: failures because `STATE_SCHEMA_VERSION` is 2.

- [ ] **Step 3: Add schema-3 fields**

Set `STATE_SCHEMA_VERSION = 3` and add:

```python
kernel_plan_digest: str | None = None
kernel_reboot_boot_id: str | None = None
recovery_kernel: str | None = None
display_manager_was_active: bool = False
kernel_verification_status: str | None = None
kernel_kernel: str | None = None
kernel_verification_findings: tuple[str, ...] = ()
```

Validate them with the same digest, UUID, kernel-name, status, and finding-code strictness as final host fields.

- [ ] **Step 4: Implement v1 to v2 to v3 migration**

Preserve immutable image/project/release fields, report paths, target user, and prior Docker-group authorization. Full mode clears completed stages and both plan approvals, then restarts at Bootstrap. Container mode keeps compatible stage position/digests. Unknown keys remain corruption and are preserved under the existing private evidence path.

- [ ] **Step 5: Run round-trip, corruption, and migration tests**

```bash
uv run pytest tests/unit/installer/test_models.py tests/unit/installer/test_state.py \
  tests/cli/test_installer_resume.py -q
```

Expected: schemas 1/2 migrate; valid migration does not emit a Bootstrap digest error.

- [ ] **Step 6: Commit**

```bash
git add src/amd_ai/installer/models.py src/amd_ai/installer/state.py \
  tests/unit/installer/test_models.py tests/unit/installer/test_state.py \
  tests/cli/test_installer_resume.py
git commit -m "feat: migrate installer state for dual host checkpoints"
```

## Task 8: Make Privileged Host Actions Phase-Aware

**Files:**
- Modify: `src/amd_ai/installer/actions.py:95-455,732-825`
- Modify: `src/amd_ai/installer/privileged.py:1-125`
- Test: `tests/unit/installer/test_actions.py`
- Test: `tests/unit/installer/test_privileged.py`

- [ ] **Step 1: Add failing direct and sudo phase tests**

```python
kernel = production.host_plan(
    target_user="developer", phase=HostPlanPhase.KERNEL
)
tuning = production.host_plan(
    target_user="developer", phase=HostPlanPhase.TUNING
)
assert kernel.plan.phase is HostPlanPhase.KERNEL
assert tuning.plan.phase is HostPlanPhase.TUNING
```

Assert the sudo command contains `--phase kernel` and a response with a different phase is rejected.

- [ ] **Step 2: Run and verify protocol support is absent**

```bash
uv run pytest tests/unit/installer/test_actions.py \
  tests/unit/installer/test_privileged.py -q
```

Expected: failures for unsupported arguments and payload identity.

- [ ] **Step 3: Thread phase through plan/apply serialization**

Use this signature:

```python
def host_plan(
    self,
    *,
    target_user: str,
    phase: HostPlanPhase,
    memory_gib: int | None = None,
) -> HostPlanResult:
```

Include `phase` in strict plan payloads. `host_apply()` rejects phase drift before mutation.

- [ ] **Step 4: Add a dedicated kernel verification operation**

```python
def kernel_verify(
    self,
    *,
    target_user: str,
    display_manager_was_active: bool,
) -> Report:
    return evaluate_kernel_reboot(
        self._snapshot(target_user=target_user),
        display_manager_was_active=display_manager_was_active,
    )
```

Expose this as strict helper operation `verify-kernel`; retain `verify` for final TTM verification.

- [ ] **Step 5: Preserve pre-change recovery facts**

Extend `HostPlanResult` with validated `running_kernel` and `display_manager_active`. Root and sudo paths return identical values for workflow persistence.

- [ ] **Step 6: Run action/protocol regressions**

```bash
uv run pytest tests/unit/installer/test_actions.py \
  tests/unit/installer/test_privileged.py tests/cli/test_host_commands.py -q
```

Expected: all pass with exact phase identity.

- [ ] **Step 7: Commit**

```bash
git add src/amd_ai/installer/actions.py src/amd_ai/installer/privileged.py \
  tests/unit/installer/test_actions.py tests/unit/installer/test_privileged.py \
  tests/cli/test_host_commands.py
git commit -m "feat: add phase-aware privileged host actions"
```

## Task 9: Implement the Two-Checkpoint Workflow

**Files:**
- Modify: `src/amd_ai/installer/models.py:32-193`
- Modify: `src/amd_ai/installer/workflow.py:327-905,1060-1130`
- Modify: `src/amd_ai/installer/progress.py:91-113`
- Modify: `tests/unit/installer/fakes.py`
- Test: `tests/unit/installer/test_workflow.py`
- Test: `tests/unit/installer/test_progress.py`

- [ ] **Step 1: Add the exact full-stage order**

```python
FULL_STAGE_ORDER = (
    InstallStage.BOOTSTRAP,
    InstallStage.HOST_PREFLIGHT,
    InstallStage.KERNEL_PLAN,
    InstallStage.KERNEL_CONFIRM,
    InstallStage.KERNEL_APPLY,
    InstallStage.KERNEL_REBOOT_PENDING,
    InstallStage.KERNEL_VERIFY,
    InstallStage.HOST_PLAN,
    InstallStage.HOST_CONFIRM,
    InstallStage.HOST_APPLY,
    InstallStage.REBOOT_PENDING,
    InstallStage.HOST_VERIFY,
    InstallStage.RELEASE_RESOLVE,
    InstallStage.IMAGE_PULL_OR_BUILD,
    InstallStage.IMAGE_VERIFY,
    InstallStage.PROJECT_INIT,
    InstallStage.PROJECT_VERIFY,
    InstallStage.COMPLETE,
)
```

Keep `CONTAINER_STAGE_ORDER` unchanged and add an exact order assertion.

- [ ] **Step 2: Write the failing first-reboot test**

```python
def test_old_kernel_stops_before_tuning_until_kernel_reboot(tmp_path):
    actions = FakeInstallerActions.host_change_requires_two_reboots()
    first = installer_workflow(
        tmp_path,
        actions=actions,
        options=full_options(tmp_path),
        boot_id_reader=lambda: FIRST_BOOT_ID,
    ).run()
    assert first.exit_code == 1
    assert first.state is not None
    assert first.state.current_stage is InstallStage.KERNEL_REBOOT_PENDING
    assert "host_apply" not in actions.calls
```

Add resume cases proving `kernel_verify` precedes `host_plan`, and image acquisition follows final host verification.

- [ ] **Step 3: Add independent noninteractive approvals**

Add `accepted_kernel_plan_digest` to `InstallOptions`; keep `accepted_host_plan_digest` for tuning. Missing acceptance is reported at its confirm stage with the exact digest instead of failing before a plan can be generated. Docker-group approval belongs only to tuning.

- [ ] **Step 4: Persist kernel checkpoint output**

`KERNEL_PLAN` requests phase `kernel`. `KERNEL_APPLY` stores `kernel_reboot_boot_id` only if reboot is required, plus recovery kernel and pre-change display state. `KERNEL_VERIFY` stores kernel verification status/name/findings without populating final host verification fields.

- [ ] **Step 5: Persist tuning checkpoint output**

`HOST_PLAN` requests phase `tuning`. `HOST_APPLY` stores the existing `reboot_boot_id` only for TTM reboot. `HOST_VERIFY` stores final host status/name/findings. Both pending stages use a shared boot-ID helper with different state fields and messages.

- [ ] **Step 6: Add stable labels and physical recovery wording**

```python
InstallStage.KERNEL_PLAN: "生成 OEM 6.17 内核计划",
InstallStage.KERNEL_CONFIRM: "确认内核变更",
InstallStage.KERNEL_APPLY: "安装 OEM 6.17 内核",
InstallStage.KERNEL_REBOOT_PENDING: "等待内核重启",
InstallStage.KERNEL_VERIFY: "验证桌面与 GPU 内核",
InstallStage.HOST_PLAN: "生成主机调优计划",
InstallStage.HOST_APPLY: "应用 Docker 与 TTM 调优",
InstallStage.REBOOT_PENDING: "等待 TTM 重启",
```

The first action message names the recovery kernel and **Advanced options for Ubuntu**. Existing progress footer supplies state/log/resume paths.

- [ ] **Step 7: Handle v0.2 update adoption deliberately**

Allow `0.2.x` to `0.3.x` adoption only for migrated container states whose prior Bootstrap inputs reconstruct exactly. Migrated full states restart at Bootstrap and never reuse old host approval. Reacquisition inspects existing immutable image layers.

- [ ] **Step 8: Run workflow/progress/resume tests**

```bash
uv run pytest tests/unit/installer/test_workflow.py \
  tests/unit/installer/test_progress.py tests/cli/test_installer_resume.py -q
```

Expected: old kernels use ordered checkpoints; healthy 6.17 skips kernel reboot; matching TTM skips the second reboot; resume never replays completed mutation.

- [ ] **Step 9: Commit**

```bash
git add src/amd_ai/installer/models.py src/amd_ai/installer/workflow.py \
  src/amd_ai/installer/progress.py tests/unit/installer/fakes.py \
  tests/unit/installer/test_workflow.py tests/unit/installer/test_progress.py \
  tests/cli/test_installer_resume.py
git commit -m "feat: add kernel and tuning reboot checkpoints"
```

## Task 10: Integrate Buildx Repair through CLI and Fixtures

**Files:**
- Modify: `src/amd_ai/installer/actions.py:430-705`
- Modify: `src/amd_ai/cli.py:128-151,462-530`
- Modify: `install.sh`
- Modify: `tests/fixtures/installer/*.json`
- Modify: `tests/fixtures/installer/README.md`
- Test: `tests/cli/test_installer_commands.py`
- Test: `tests/cli/test_installer_resume.py`

- [ ] **Step 1: Add a runtime-without-Buildx container test**

```python
def test_container_host_check_allows_runtime_without_buildx():
    snapshot = healthy_snapshot(
        docker_buildx_version=None,
        docker_buildx_error="docker: unknown command: docker buildx",
        docker_distribution=DockerDistribution.UBUNTU_DOCKER_IO,
    )
    actions = ProductionInstallerActions(effective_uid=0)
    actions._input_snapshots[InstallStage.CONTAINER_HOST_CHECK] = snapshot
    result = actions.container_host_check()
    assert result.blocked is False
    assert result.facts["docker_buildx_version"] is None
```

Add a pending project-build test expecting `sudo apt install docker-buildx` guidance before BuildKit starts.

- [ ] **Step 2: Run and observe missing facts/arguments**

```bash
uv run pytest tests/unit/installer/test_actions.py \
  tests/cli/test_installer_commands.py tests/cli/test_installer_resume.py -q
```

Expected: new tests fail.

- [ ] **Step 3: Expose both approval digests**

Add `--accept-kernel-plan-digest` beside `--accept-host-plan-digest`; forward both safely through `install.sh`. Help states which reboot boundary each digest authorizes.

- [ ] **Step 4: Make installer remediation package-specific**

Before a required project build, use current host facts. Docker CE prints `sudo apt install docker-buildx-plugin`; Ubuntu `docker.io` prints `sudo apt install docker-buildx`; mixed/external sources print no unsafe package command. Runtime image verification remains allowed.

- [ ] **Step 5: Rewrite fixtures for schema 3 and 18 full stages**

Add Buildx, candidate, and display command responses. `full-host-change.json` models `6.14.0-1020-oem`; `full-rebooted.json` models `6.17.0-1028-oem`. Add AXB35 raw evidence with `Subsystem: Device [2014:801d]`.

- [ ] **Step 6: Run fixture workflows**

```bash
uv run pytest tests/cli/test_installer_commands.py \
  tests/cli/test_installer_resume.py tests/unit/installer/test_actions.py -q
```

Expected: all pass; valid old state never emits `completed stage inputs changed for BOOTSTRAP`.

- [ ] **Step 7: Commit**

```bash
git add src/amd_ai/installer/actions.py src/amd_ai/cli.py install.sh \
  tests/fixtures/installer tests/cli/test_installer_commands.py \
  tests/cli/test_installer_resume.py tests/unit/installer/test_actions.py
git commit -m "feat: integrate kernel and Buildx repair workflow"
```

## Task 11: Publish v0.3.0 Documentation and Version Contracts

**Files:**
- Modify: `src/amd_ai/__init__.py`
- Modify: `tests/test_version.py`
- Modify: `README.md`
- Modify: `docs/host-operations.md`

- [ ] **Step 1: Make version tests fail on the old version**

```python
assert __version__ == "0.3.0"
assert capsys.readouterr().out.strip() == "amd-ai 0.3.0"
```

```bash
uv run pytest tests/test_version.py -q
```

Expected: two failures reporting `0.2.3`.

- [ ] **Step 2: Set toolkit version only**

Set:

```python
__version__ = "0.3.0"
```

Keep stable image release ID `0.2.0` and all image digests unchanged.

- [ ] **Step 3: Rewrite README quick start and recovery**

Use `v0.3.0`, explain both conditional manual reboots, and put GRUB recovery before the first reboot. Add this runtime-only validation:

```bash
IMAGE='ghcr.io/hellcatjack/strix-halo-rocm-pytorch@sha256:dc0bb217474cfd4f602423bd3bf4fe8714b03e900cf3c6b4417b99e622ebcf8b'
docker pull "$IMAGE"
strix-halo-rocm container-check \
  --image "$IMAGE" --mode torch --runtime \
  --json "$HOME/strix-reports/gpu-check.json"
```

State that it does not require Buildx, while project builds do.

- [ ] **Step 4: Rewrite host operations around `linux-oem-6.17`**

Remove claims that 6.14 is supported or that `linux-oem-24.04` is installed. Document candidate validation, matching Buildx repair, conditional display check, retained kernels, and both report checkpoints.

- [ ] **Step 5: Validate docs and version output**

```bash
uv run pytest tests/test_version.py tests/cli/test_installer_commands.py -q
uv run python -m amd_ai.cli --version
./install.sh --help >/tmp/strix-install-help.txt
rg -n "6\.14.*supported|linux-oem-24\.04|v0\.2\.3" \
  README.md docs/host-operations.md
```

Expected: tests pass, version is `amd-ai 0.3.0`, help exits 0, and final `rg` returns 1 with no matches.

- [ ] **Step 6: Commit**

```bash
git add src/amd_ai/__init__.py tests/test_version.py README.md \
  docs/host-operations.md
git commit -m "docs: publish the v0.3.0 OEM 6.17 workflow"
```

## Task 12: Qualify Hardware Before Promoting the Kernel

**Files:**
- Modify after qualification: `profiles/host/tested-kernels.json`
- Test: `tests/unit/host/test_policy.py`
- Verify: `tests/hardware/test_release.py`
- Verify unchanged: `profiles/releases/stable.json`

- [ ] **Step 1: Run focused regressions**

```bash
uv run pytest tests/unit/host tests/unit/installer tests/unit/image \
  tests/unit/project tests/unit/doctor tests/cli/test_host_commands.py \
  tests/cli/test_image_commands.py tests/cli/test_installer_commands.py \
  tests/cli/test_installer_resume.py tests/cli/test_project_commands.py \
  tests/cli/test_release_commands.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the full non-hardware suite**

```bash
uv run pytest -m 'not hardware' -q
```

Expected: no failures and at least the baseline `586 passed, 1 deselected`.

- [ ] **Step 3: Verify the local host identity**

```bash
mkdir -p reports
test "$(uname -r)" = "6.17.0-1028-oem"
lspci -Dnnk -d 1002:1586 | tee reports/v0.3.0-lspci.txt
test -c /dev/kfd
test -c /dev/dri/renderD128
systemctl is-active display-manager
```

Expected: exact kernel, `Kernel driver in use: amdgpu`, both devices, active display.

- [ ] **Step 4: Run stable Torch and full hardware qualification**

```bash
sudo -v
uv run python -m amd_ai.cli container-check \
  --image rocm-pytorch:7.2.1-py3.12-torch2.9.1 \
  --mode torch --runtime --json reports/v0.3.0-torch-runtime.json
uv run pytest tests/hardware/test_release.py -m hardware -v
```

Expected: `gfx1151`, HIP available, synchronized tensor success, stress pass, and clean current-boot GPU log. On failure, stop and do not promote the kernel.

- [ ] **Step 5: Promote only the qualified exact kernel**

After Step 4 passes, write:

```json
{
  "kernels": [
    "6.17.0-1028-oem"
  ],
  "schema_version": 1,
  "source": "https://packages.ubuntu.com/linux-oem"
}
```

Then run:

```bash
uv run pytest tests/unit/host/test_policy.py tests/unit/host/test_verify.py -q
```

Expected: exact 1028 passes, other 6.17 patches are unverified, and 6.14 requires change.

- [ ] **Step 6: Prove stable image references did not drift**

```bash
git diff 15de8d0 -- profiles/releases/stable.json
rg -n "dc0bb217474cfd4f602423bd3bf4fe8714b03e900cf3c6b4417b99e622ebcf8b" \
  profiles/releases/stable.json README.md
```

Expected: no manifest diff and both files contain the immutable Torch digest.

- [ ] **Step 7: Run final repository verification**

```bash
uv run pytest -m 'not hardware' -q
git diff --check
git status --short
```

Expected: tests pass, diff check is silent, and only intended tracked changes remain.

- [ ] **Step 8: Commit qualified baseline**

```bash
git add profiles/host/tested-kernels.json tests/unit/host/test_policy.py \
  tests/unit/host/test_verify.py
git commit -m "release: qualify OEM 6.17 for the v0.3.0 host baseline"
```

- [ ] **Step 9: Review before merge/tag**

Invoke `superpowers:requesting-code-review`, resolve findings through `superpowers:receiving-code-review`, rerun focused and full non-hardware tests, and invoke `superpowers:verification-before-completion`. Do not create or push `v0.3.0` before review and verification pass.
