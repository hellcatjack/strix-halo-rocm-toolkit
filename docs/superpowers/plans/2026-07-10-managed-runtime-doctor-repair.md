# Managed Runtime Doctor and Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run every managed project with a read-only root, detect base or overlay Torch corruption before application startup, classify platform and project failures, and repair only through quarantine plus exact immutable pull/rebuild operations.

**Architecture:** Managed `docker run` arguments expose only the project bind, bounded tmpfs/shm, and explicit mounts as writable. Container checks combine the existing protected-file manifest with an effective import probe that sees the project overlay. A read-only doctor composes release, parent, derived-image, overlay, GPU, and kernel evidence; repair converts repairable findings to exact actions, confirms them, quarantines overlay generations, and rebuilds from the stable digest chain and hash locks.

**Tech Stack:** Python 3.12, Docker Engine, existing `Report`/project build/runtime APIs, `fcntl`, `importlib.metadata`, pytest, ROCm/PyTorch runtime probes.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `src/amd_ai/project/run.py` | Read-only root, bounded tmpfs, protected environment and exact parent identity |
| `src/amd_ai/project/runtime.py` | Deterministic tmpfs sizing |
| `src/amd_ai/overlay/verify.py` | Full-version, distribution-path, module-path, HIP, manifest and shadow verification |
| `src/amd_ai/overlay/repair.py` | Quarantine and offline generation rebuild from the last valid lock |
| `src/amd_ai/container/check.py` | Surface effective-stack findings and retain GPU runtime gate |
| `templates/project/project-entrypoint` | Acquire startup lock, run complete check, block application on failure |
| `src/amd_ai/doctor/models.py` | Diagnostics, report, exact repair action records |
| `src/amd_ai/doctor/checks.py` | Read-only platform/project diagnostic composition |
| `src/amd_ai/doctor/repair.py` | Exact action planning, confirmation and orchestration |
| `tests/unit/doctor/` | Classification and exact repair planning tests |
| `tests/cli/test_doctor_commands.py` | Doctor/repair CLI behavior |
| `tests/container/test_repair_flow.py` | Real shadow, quarantine, image rebuild and GPU recovery tests |

### Task 1: Make Managed Runtime Roots Read-Only with Bounded tmpfs

**Files:**
- Modify: `src/amd_ai/project/runtime.py`
- Modify: `src/amd_ai/project/run.py`
- Modify: `src/amd_ai/project/config.py`
- Modify: `tests/unit/project/test_runtime.py`
- Modify: `tests/unit/project/test_run.py`

- [ ] **Step 1: Write failing tmpfs and run-argv tests**

Add:

```python
@pytest.mark.parametrize(
    ("shm_gib", "expected"),
    [(1, 1), (2, 1), (8, 4), (16, 8), (64, 8), (128, 8)],
)
def test_tmpfs_size_is_bounded(shm_gib, expected):
    assert compute_tmpfs_gib(shm_gib=shm_gib) == expected
```

Extend the normal run test:

```python
assert "--read-only" in argv
tmpfs_index = argv.index("--tmpfs")
assert argv[tmpfs_index + 1] == "/tmp:rw,nosuid,nodev,size=8g,mode=1777"
assert "AMD_AI_PARENT_CONFIG_DIGEST=" + config.base_digest in argv
assert "--privileged" not in argv
```

- [ ] **Step 2: Run focused tests and observe failures**

Run:

```bash
uv run pytest tests/unit/project/test_runtime.py tests/unit/project/test_run.py -q
```

Expected: failures for missing `compute_tmpfs_gib`, `--read-only`, tmpfs, and parent digest environment.

- [ ] **Step 3: Implement bounded tmpfs sizing**

Add:

```python
def compute_tmpfs_gib(*, shm_gib: int) -> int:
    if isinstance(shm_gib, bool) or not isinstance(shm_gib, int):
        raise RuntimePolicyError("shared memory size must be an integer")
    if not 1 <= shm_gib <= 128:
        raise RuntimePolicyError("shared memory size must be from 1 through 128 GiB")
    return min(8, max(1, shm_gib // 2))
```

- [ ] **Step 4: Add immutable runtime arguments**

In `build_run_argv()`, immediately after `--ipc=private`, add `--read-only`. Add:

```python
"--tmpfs",
f"/tmp:rw,nosuid,nodev,size={compute_tmpfs_gib(shm_gib=shm_gib)}g,mode=1777",
```

Pass `AMD_AI_PARENT_CONFIG_DIGEST=<config.base_digest>` as a reserved environment value. Add it to `RESERVED_ENVIRONMENT`; project TOML cannot replace it. Keep project `/workspace` read-write and preserve explicit mount modes. Do not add a config or CLI switch that disables `--read-only`.

- [ ] **Step 5: Run focused tests**

Run: `uv run pytest tests/unit/project/test_runtime.py tests/unit/project/test_run.py -q`

Expected: all focused tests pass.

- [ ] **Step 6: Commit read-only runtime arguments**

```bash
git add src/amd_ai/project/runtime.py src/amd_ai/project/run.py \
  src/amd_ai/project/config.py tests/unit/project/test_runtime.py \
  tests/unit/project/test_run.py
git commit -m "feat: run managed projects with read-only roots"
```

### Task 2: Initialize and Lock the Project Overlay at Startup

**Files:**
- Modify: `src/amd_ai/overlay/transaction.py`
- Modify: `src/amd_ai/cli.py`
- Modify: `templates/project/project-entrypoint`
- Modify: `tests/unit/overlay/test_transaction.py`
- Modify: `tests/cli/test_project_commands.py`
- Modify: `tests/container/test_project_dockerfile.py`

- [ ] **Step 1: Write failing empty-generation initialization tests**

```python
def test_initialize_overlay_creates_one_valid_empty_generation(tmp_path):
    project = tmp_path / "demo"
    project.mkdir()
    paths = OverlayPaths.for_project(project)

    state = initialize_overlay(paths, profile=protected_profile())

    assert state.lock_digest == hashlib.sha256(b"").hexdigest()
    assert resolve_current_generation(paths).name == state.generation_id
    assert (resolve_current_generation(paths) / "site-packages").is_dir()
    assert initialize_overlay(paths, profile=protected_profile()) == state
```

Extend project-run CLI tests to assert `initialize_overlay()` is called before `build_run_argv()` and before the live Docker process.

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
uv run pytest tests/unit/overlay/test_transaction.py \
  tests/cli/test_project_commands.py tests/container/test_project_dockerfile.py -q
```

Expected: missing initializer and ordering assertions fail.

- [ ] **Step 3: Implement idempotent empty generation initialization**

Under the exclusive transaction lock, create an empty input, empty lock, and state bound to the profile ID and parent config digest. Validate an existing current generation instead of replacing it. A missing current link with nonempty generations is not treated as fresh; raise `TransactionError` for doctor/repair classification.

- [ ] **Step 4: Add startup lock and initialization ordering**

`_project_run()` calls `initialize_overlay()` after inspecting and authorizing the project image and before constructing Docker argv. The entrypoint opens `/workspace/.amd-ai/transaction.lock`, takes an exclusive lock only while running startup checks and `mark_generation_healthy()`, releases it, then executes the application. It must not retain the lock across `exec`, because a user must be able to run protected `pip install` from a shell in the running container.

Implement the entrypoint lock with Python `fcntl.flock`; on contention print `OVERLAY.TRANSACTION_INCOMPLETE` and exit 1. After all startup checks pass, mark the current generation healthy and apply the bounded current/previous retention policy from the overlay plan before releasing the lock. Preserve exact profile-status and command-empty checks.

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run pytest tests/unit/overlay/test_transaction.py \
  tests/cli/test_project_commands.py tests/container/test_project_dockerfile.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit startup overlay initialization**

```bash
git add src/amd_ai/overlay/transaction.py src/amd_ai/cli.py \
  templates/project/project-entrypoint tests/unit/overlay/test_transaction.py \
  tests/cli/test_project_commands.py tests/container/test_project_dockerfile.py
git commit -m "feat: initialize and lock project overlays"
```

### Task 3: Verify Full Protected Distribution and Module Identity

**Files:**
- Modify: `src/amd_ai/overlay/verify.py`
- Create: `src/amd_ai/overlay/effective_probe.py`
- Modify: `tests/unit/overlay/test_verify.py`

- [ ] **Step 1: Write failing effective-probe parser tests**

```python
def test_effective_identity_requires_base_distribution_and_module_paths(profile):
    payload = {
        "schema_version": 1,
        "components": {
            name: {
                "distribution_path": f"/opt/venv/lib/python3.12/site-packages/{name}-x.dist-info",
                "module_path": f"/opt/venv/lib/python3.12/site-packages/{name}/__init__.py",
                "version": profile.version_for(name),
            }
            for name in ("torch", "torchvision", "torchaudio", "triton")
        },
        "torch_hip_version": "7.2.1",
    }

    result = validate_effective_probe(payload, profile=profile)

    assert result.torch_hip_version == "7.2.1"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("distribution_path", "/workspace/.amd-ai/current/site-packages/torch.dist-info"),
        ("module_path", "/workspace/torch.py"),
        ("version", "2.9.1"),
    ],
)
def test_effective_identity_rejects_shadow_or_public_only_version(
    profile, valid_probe, field, value
):
    valid_probe["components"]["torch"][field] = value

    with pytest.raises(OverlayVerificationError):
        validate_effective_probe(valid_probe, profile=profile)
```

- [ ] **Step 2: Run verification tests and observe missing probe failure**

Run: `uv run pytest tests/unit/overlay/test_verify.py -q`

Expected: failures for missing effective-probe functions.

- [ ] **Step 3: Implement the probe process**

`effective_probe.py` imports each protected distribution through `importlib.metadata.distribution()`, imports `torch`, `torchvision`, `torchaudio`, and `triton`, and prints one JSON object. For each component include exact `distribution.version`, resolved `distribution.locate_file("")`, and resolved module `__file__`. Include `torch.version.hip`. Catch each import independently and return structured errors; never silently omit a component.

The caller runs:

```python
(
    "/opt/venv/bin/python",
    "-m",
    "amd_ai.overlay.effective_probe",
)
```

with `PYTHONPATH=<candidate site-packages>:/opt/amd-ai/src` and `PYTHONNOUSERSITE=1`.

- [ ] **Step 4: Validate exact full versions and paths**

Load expected full versions from `/opt/amd-ai/torch-manifest.json`, not the public `*_VERSION` profile fields. Require all distribution and module paths to resolve below `/opt/venv`, reject symlink escape, require exact full version string equality, require HIP version `7.2.1` or a `7.2.1-` build suffix, and call `scan_protected_entries()` on the candidate overlay before the probe.

- [ ] **Step 5: Run effective verification tests**

Run: `uv run pytest tests/unit/overlay/test_verify.py -q`

Expected: missing component, import error, path shadow, full-version mismatch, HIP mismatch, and valid probe tests pass.

- [ ] **Step 6: Commit effective identity verification**

```bash
git add src/amd_ai/overlay/verify.py src/amd_ai/overlay/effective_probe.py \
  tests/unit/overlay/test_verify.py
git commit -m "feat: verify effective protected import identity"
```

### Task 4: Integrate Effective Identity into Container Checks

**Files:**
- Modify: `src/amd_ai/container/check.py`
- Modify: `tests/unit/container/test_check.py`
- Modify: `templates/project/project-entrypoint`

- [ ] **Step 1: Write failing finding-code tests**

Add unit cases where the injected effective verifier reports a shadow and where the base file manifest is changed:

```python
assert report.status == Status.BLOCKED
assert {finding.code for finding in report.findings} >= {
    "TORCH.SHADOWED",
    "TORCH.BASE_CHANGED",
}
```

Add a runtime-mode assertion that manifest verification is executed even with `--runtime`.

- [ ] **Step 2: Run container-check unit tests**

Run: `uv run pytest tests/unit/container/test_check.py -q`

Expected: finding-code and runtime-manifest assertions fail.

- [ ] **Step 3: Call both immutable and effective checks**

Always invoke `/opt/amd-ai/torch-manifest.py verify`, including runtime mode. Map a failure to `TORCH.BASE_CHANGED`. Invoke `verify_effective_stack()` under the current `PYTHONPATH`; map path/version/protected-entry failures to `TORCH.SHADOWED`. Keep existing public-version facts for compatibility, but exact full-version validation is authoritative.

Run the existing synchronized gfx1151 GPU operation after static checks. A static failure does not suppress evidence collection, but final status remains blocked.

- [ ] **Step 4: Run container-check tests**

Run: `uv run pytest tests/unit/container/test_check.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit startup effective checks**

```bash
git add src/amd_ai/container/check.py tests/unit/container/test_check.py \
  templates/project/project-entrypoint
git commit -m "feat: block startup on effective Torch drift"
```

### Task 5: Define Diagnostics and Stable Failure Codes

**Files:**
- Create: `src/amd_ai/doctor/__init__.py`
- Create: `src/amd_ai/doctor/models.py`
- Create: `tests/unit/doctor/__init__.py`
- Create: `tests/unit/doctor/test_models.py`

- [ ] **Step 1: Write failing diagnostic report tests**

```python
def test_report_status_uses_highest_disposition_and_serializes_exactly():
    report = DoctorReport.create(
        project="/workspace/demo",
        diagnostics=(
            Diagnostic("OVERLAY.TRANSACTION_INCOMPLETE", DiagnosticDisposition.WARNING, "stale", "tx", "inspect"),
            Diagnostic("TORCH.SHADOWED", DiagnosticDisposition.REPAIRABLE, "shadow", "path", "repair"),
        ),
        facts={"release_id": "0.2.0"},
    )

    assert report.status == "repairable"
    assert report.to_dict()["diagnostics"][1]["code"] == "TORCH.SHADOWED"
```

- [ ] **Step 2: Run tests and confirm missing package failure**

Run: `uv run pytest tests/unit/doctor/test_models.py -q`

Expected: collection fails because `amd_ai.doctor` is missing.

- [ ] **Step 3: Implement diagnostic records**

Use:

```python
class DiagnosticDisposition(StrEnum):
    PASS = "pass"
    WARNING = "warning"
    REPAIRABLE = "repairable"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class Diagnostic:
    code: str
    disposition: DiagnosticDisposition
    summary: str
    evidence: str
    remediation: str


@dataclass(frozen=True)
class RepairAction:
    kind: str
    exact_target: str
    reason_code: str
```

`DoctorReport` includes schema version 1, UTC generated time, optional resolved project path, facts, diagnostics, and derived status. Reject unknown diagnostic codes and any evidence string containing URL userinfo or values from secret-named environment variables after redaction.

- [ ] **Step 4: Run model tests**

Run: `uv run pytest tests/unit/doctor/test_models.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit diagnostic models**

```bash
git add src/amd_ai/doctor tests/unit/doctor
git commit -m "feat: define doctor diagnostics and repair actions"
```

### Task 6: Implement Read-Only Platform and Project Doctor Checks

**Files:**
- Create: `src/amd_ai/doctor/checks.py`
- Create: `tests/unit/doctor/fakes.py`
- Create: `tests/unit/doctor/test_checks.py`

- [ ] **Step 1: Write failing classification tests**

Build table-driven fake states and assert these mappings:

```python
EXPECTED_CODES = {
    "invalid_release": "RELEASE.INVALID",
    "missing_parent": "IMAGE.PARENT_MISSING",
    "tag_drift": "IMAGE.DIGEST_DRIFT",
    "changed_project": "IMAGE.PROJECT_CHANGED",
    "base_manifest": "TORCH.BASE_CHANGED",
    "overlay_shadow": "TORCH.SHADOWED",
    "lock_invalid": "OVERLAY.LOCK_INVALID",
    "incomplete": "OVERLAY.TRANSACTION_INCOMPLETE",
    "gpu": "GPU.RUNTIME_FAILED",
}
```

Assert doctor never calls `pull`, `tag`, `rm`, `build`, `quarantine`, or transaction activation on any state.

- [ ] **Step 2: Run doctor checks and confirm failure**

Run: `uv run pytest tests/unit/doctor/test_checks.py -q`

Expected: collection fails because `amd_ai.doctor.checks` is missing.

- [ ] **Step 3: Implement platform checks**

`doctor_platform()` must:

1. Load and validate the stable manifest.
2. Inspect exact base and torch references without pulling.
3. Classify missing exact references as `IMAGE.PARENT_MISSING` repairable.
4. Compare optional friendly tags to expected config IDs and classify drift as repairable without changing tags.
5. Verify labels and embedded artifact hashes when images exist.
6. Run host preflight read-only checks.
7. Run the existing torch runtime probe only when the exact torch image passes static identity.
8. Classify failed gfx1151/GPU operation as `GPU.RUNTIME_FAILED` blocked.

- [ ] **Step 4: Implement project checks**

`doctor_project()` adds strict config loading, exact parent digest comparison, `validate_project_image_contract()`, project image manifest check, overlay current-link boundary, generation state/input/lock digest consistency, artifact hash validation, protected-entry scan, and effective import probe in a disposable read-only project container. An incomplete non-current generation is warning; a dangling current link or invalid active generation is repairable and blocks startup.

Collect exact parent config ID, project image ID, fingerprint, current generation path, and last valid lock path in facts for repair planning. Redact all Docker output through the existing sensitive-name/URL helper before report creation.

- [ ] **Step 5: Run doctor check tests**

Run: `uv run pytest tests/unit/doctor/test_checks.py -q`

Expected: all classification, no-mutation, evidence, and redaction tests pass.

- [ ] **Step 6: Commit doctor checks**

```bash
git add src/amd_ai/doctor/checks.py tests/unit/doctor
git commit -m "feat: classify platform and project health"
```

### Task 7: Expose Doctor CLI and JSON Reports

**Files:**
- Modify: `src/amd_ai/cli.py`
- Create: `tests/cli/test_doctor_commands.py`

- [ ] **Step 1: Write failing doctor CLI tests**

```python
def test_doctor_without_project_checks_platform(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_doctor", lambda project, manifest: repairable_report())

    code = cli.main(["doctor", "--manifest", "tests/fixtures/releases/stable.json"])

    assert code == 1
    assert "TORCH.SHADOWED" in capsys.readouterr().out


def test_doctor_json_writes_report(monkeypatch, tmp_path):
    output = tmp_path / "doctor.json"
    monkeypatch.setattr(cli, "run_doctor", lambda project, manifest: passing_report())

    assert cli.main(["doctor", "--json", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"
```

- [ ] **Step 2: Run doctor CLI tests**

Run: `uv run pytest tests/cli/test_doctor_commands.py -q`

Expected: parser rejects unknown `doctor`.

- [ ] **Step 3: Add thin doctor command parsing**

Add:

```text
doctor [PROJECT] [--manifest PATH] [--json PATH]
```

Default manifest is `profiles/releases/stable.json`. Print one line per non-pass diagnostic. Return 0 for pass/warning-only, 1 when any repairable finding exists, and 2 when any blocked finding exists or doctor itself cannot collect trustworthy evidence.

- [ ] **Step 4: Run doctor tests**

Run:

```bash
uv run pytest tests/unit/doctor tests/cli/test_doctor_commands.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit doctor CLI**

```bash
git add src/amd_ai/cli.py tests/cli/test_doctor_commands.py
git commit -m "feat: expose classified doctor reports"
```

### Task 8: Plan Only Exact Repair Actions

**Files:**
- Create: `src/amd_ai/doctor/repair.py`
- Create: `tests/unit/doctor/test_repair.py`

- [ ] **Step 1: Write failing exact-action tests**

```python
def test_repair_plan_uses_exact_generation_image_id_and_registry_digest():
    report = repairable_project_report()

    plan = plan_repair(report)

    assert plan.actions == (
        RepairAction(
            "pull-parent",
            "ghcr.io/hellcatjack/strix-halo-rocm-pytorch@sha256:" + "7" * 64,
            "IMAGE.PARENT_MISSING",
        ),
        RepairAction(
            "remove-project-image",
            "sha256:" + "8" * 64,
            "IMAGE.PROJECT_CHANGED",
        ),
        RepairAction(
            "build-project-image",
            "/srv/demo/amd-ai-project.toml",
            "IMAGE.PROJECT_CHANGED",
        ),
        RepairAction(
            "quarantine-overlay",
            "/srv/demo/.amd-ai/generations/20260710T120000Z-a1b2c3d4",
            "TORCH.SHADOWED",
        ),
        RepairAction(
            "rebuild-overlay",
            "/srv/demo/.amd-ai/generations/20260710T120000Z-a1b2c3d4/overlay.requirements.lock",
            "TORCH.SHADOWED",
        ),
        RepairAction(
            "verify-project",
            "/srv/demo/amd-ai-project.toml",
            "TORCH.SHADOWED",
        ),
    )
    assert not any("*" in action.exact_target for action in plan.actions)
```

Add tests that `RELEASE.INVALID` and `GPU.RUNTIME_FAILED` produce no destructive action and keep the plan blocked.

- [ ] **Step 2: Run repair tests and confirm missing planner failure**

Run: `uv run pytest tests/unit/doctor/test_repair.py -q`

Expected: collection fails because repair module is missing.

- [ ] **Step 3: Implement deterministic repair planning**

The planner accepts only facts emitted by doctor and validates every target again. Allowed kinds are:

```python
ALLOWED_REPAIR_KINDS = frozenset(
    {
        "quarantine-overlay",
        "remove-project-image",
        "pull-parent",
        "retag-parent",
        "build-project-image",
        "rebuild-overlay",
        "verify-project",
    }
)
```

Reject mutable image references, non-SHA image IDs, generation paths outside the selected project's `.amd-ai/generations`, duplicate conflicting actions, and any plan containing `prune`. Sort actions by immutable parent pull, exact project image rebuild, overlay quarantine/rebuild, then final verification while preserving each exact reason code.

- [ ] **Step 4: Run repair planning tests**

Run: `uv run pytest tests/unit/doctor/test_repair.py -q`

Expected: all exact-target and blocked-plan tests pass.

- [ ] **Step 5: Commit repair planning**

```bash
git add src/amd_ai/doctor/repair.py tests/unit/doctor/test_repair.py
git commit -m "feat: plan exact immutable repairs"
```

### Task 9: Quarantine and Rebuild the Overlay Offline

**Files:**
- Create: `src/amd_ai/overlay/repair.py`
- Modify: `tests/unit/overlay/test_transaction.py`
- Create: `tests/unit/overlay/test_repair.py`

- [ ] **Step 1: Write failing quarantine tests**

```python
def test_overlay_repair_moves_current_generation_and_replays_lock(tmp_path):
    paths, profile, current = damaged_overlay(tmp_path)
    builder = FakeGenerationBuilder()

    result = repair_overlay(
        paths,
        profile=profile,
        reason_code="TORCH.SHADOWED",
        builder=builder,
    )

    assert result.quarantine.name.endswith("-TORCH.SHADOWED")
    assert (result.quarantine / "generation" / "overlay.requirements.lock").is_file()
    assert builder.lock_text == known_good_lock()
    assert resolve_current_generation(paths) == result.new_generation


def test_failed_overlay_rebuild_keeps_quarantine_and_project_blocked(tmp_path):
    paths, profile, current = damaged_overlay(tmp_path)
    builder = FakeGenerationBuilder(error=TransactionError("install failed"))

    with pytest.raises(TransactionError):
        repair_overlay(paths, profile=profile, reason_code="TORCH.SHADOWED", builder=builder)

    assert paths.current.is_symlink()
    assert not paths.current.exists()
    assert any(paths.quarantine.iterdir())
```

- [ ] **Step 2: Run overlay repair tests**

Run: `uv run pytest tests/unit/overlay/test_repair.py -q`

Expected: collection fails because `amd_ai.overlay.repair` is missing.

- [ ] **Step 3: Implement quarantine under exclusive lock**

Validate the current generation and its metadata digests before moving it. Create `.amd-ai/quarantine/<UTC>-<reason>/`, atomically move the entire generation directory to `generation`, and atomically write `doctor-report.json` plus `quarantine-state.json`. Keep root mirrors as evidence. The current symlink becomes deliberately dangling so every managed startup remains blocked until a new generation passes.

- [ ] **Step 4: Rebuild only from the quarantined successful lock**

Validate every artifact byte against the quarantined lock, build a fresh empty generation using `--no-index --no-deps --require-hashes`, run dependency and effective identity checks, then activate it. Do not resolve, download, edit the lock, or reinstall Torch. On failure retain the dangling current link, quarantine, logs, and root evidence.

- [ ] **Step 5: Run overlay repair tests**

Run:

```bash
uv run pytest tests/unit/overlay/test_repair.py \
  tests/unit/overlay/test_transaction.py -q
```

Expected: successful replay, corrupt artifact, invalid lock, failed install, failed verify, and retry tests pass.

- [ ] **Step 6: Commit overlay repair**

```bash
git add src/amd_ai/overlay/repair.py tests/unit/overlay
git commit -m "feat: quarantine and replay damaged overlays"
```

### Task 10: Execute Parent and Project Image Repairs

**Files:**
- Modify: `src/amd_ai/doctor/repair.py`
- Modify: `src/amd_ai/project/build.py`
- Modify: `tests/unit/doctor/test_repair.py`
- Modify: `tests/unit/project/test_build.py`

- [ ] **Step 1: Write failing execution-order and exact-remove tests**

```python
def test_execute_repair_pulls_parent_removes_exact_project_id_and_rebuilds(plan):
    executor = FakeRepairExecutor()

    execute_repair(plan, executor=executor)

    assert executor.calls == [
        ("pull-and-verify", plan.release.torch.reference),
        ("remove-image-id", "sha256:" + "8" * 64),
        ("build-project", plan.project_path, plan.release.torch.config_digest),
        ("repair-overlay", plan.project_path, "TORCH.SHADOWED"),
        ("doctor", plan.project_path),
    ]
```

Assert a failed parent verification prevents every later call and a failed project build does not touch the overlay.

- [ ] **Step 2: Run repair execution tests**

Run:

```bash
uv run pytest tests/unit/doctor/test_repair.py tests/unit/project/test_build.py -q
```

Expected: failures for missing execution behavior.

- [ ] **Step 3: Implement exact parent and project operations**

For parent repair call the release plan's `pull_and_verify_release()` with exact references, then recreate friendly local aliases only after complete identity validation. For project repair call `docker image rm <exact project image ID>` and `build_or_reuse_project(force=True, no_build=False)` against the config's verified parent config digest. Never call `docker system prune`, `docker image prune`, wildcard remove, or `pip --force-reinstall`.

After build, rerun parent-layer prefix, labels, entrypoint, user, workdir, project fingerprint, and Torch manifest checks already exposed by `project.build`. Preserve project files, `.amd-ai`, models, and every explicit mount source.

- [ ] **Step 4: Require final doctor and GPU/kernel evidence**

Repair returns 0 only if a fresh doctor report has no warning, repairable, or blocked diagnostic; the runtime probe reports gfx1151; and the existing kernel log differential has no new page fault, MES, reset, or fatal GPU finding. A failed final gate leaves the project blocked and all quarantine evidence intact.

- [ ] **Step 5: Run repair tests**

Run:

```bash
uv run pytest tests/unit/doctor/test_repair.py tests/unit/project/test_build.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit immutable repair execution**

```bash
git add src/amd_ai/doctor/repair.py src/amd_ai/project/build.py \
  tests/unit/doctor/test_repair.py tests/unit/project/test_build.py
git commit -m "feat: rebuild damaged project image chain"
```

### Task 11: Expose Confirmed Repair CLI

**Files:**
- Modify: `src/amd_ai/cli.py`
- Modify: `tests/cli/test_doctor_commands.py`

- [ ] **Step 1: Write failing confirmation tests**

```python
def test_repair_prints_exact_actions_and_requires_repair_word(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_doctor", lambda project, manifest: repairable_report())
    monkeypatch.setattr("builtins.input", lambda prompt: "no")

    code = cli.main(["repair", "/srv/demo"])

    assert code == 2
    output = capsys.readouterr().out
    assert "sha256:" in output
    assert "ghcr.io/hellcatjack/strix-halo-rocm-pytorch@sha256:" in output


def test_noninteractive_repair_requires_yes(monkeypatch):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["repair", "/srv/demo", "--non-interactive"])
```

- [ ] **Step 2: Run CLI tests**

Run: `uv run pytest tests/cli/test_doctor_commands.py -q`

Expected: parser rejects unknown `repair`.

- [ ] **Step 3: Add repair parsing, preview, and confirmation**

Add:

```text
repair PROJECT [--manifest PATH] [--yes] [--json PATH]
```

Always run doctor first and print every exact action. Interactive mode requires exact input `REPAIR`; `--yes` is the only non-interactive approval. A blocked/no-action plan returns 2. Write pre-repair and post-repair reports beside the requested JSON path without overwriting either on failure.

- [ ] **Step 4: Run doctor/repair CLI tests**

Run:

```bash
uv run pytest tests/unit/doctor tests/cli/test_doctor_commands.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit repair CLI**

```bash
git add src/amd_ai/cli.py tests/cli/test_doctor_commands.py
git commit -m "feat: expose confirmed immutable repair"
```

### Task 12: Add Container Corruption and Repair Integration Tests

**Files:**
- Create: `tests/container/test_repair_flow.py`

- [ ] **Step 1: Test read-only root and direct base pip failure**

Run a managed project and assert:

```python
assert project.run("touch", "/opt/venv/forbidden").returncode != 0
assert project.run(
    "/opt/venv/bin/python", "-m", "pip", "install", "six==1.17.0"
).returncode != 0
assert project.run("pip", "install", "six==1.17.0").returncode == 0
```

- [ ] **Step 2: Test shadow detection and offline overlay repair**

After a healthy install, manually create `.amd-ai/current/site-packages/torch.py`, assert the next managed startup reports `TORCH.SHADOWED`, run confirmed repair, disconnect resolver network for the repair container, and assert the original lock is restored and startup succeeds.

- [ ] **Step 3: Test exact project image repair**

Build a deliberately changed derived image under the project's friendly tag while retaining the exact damaged image ID. Assert doctor reports `IMAGE.PROJECT_CHANGED`, repair removes only that ID, rebuilds against the configured parent, and leaves a separately tagged unrelated image present.

- [ ] **Step 4: Run non-hardware integration tests**

Run:

```bash
uv run pytest -m container tests/container/test_readonly_overlay.py \
  tests/container/test_repair_flow.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Run a target GPU repair qualification**

On the target host, create and repair an overlay shadow, then run:

```bash
bin/container-check --suite stable \
  --profile profiles/qualification/stable.toml \
  --json reports/repair-qualification.json
```

Expected: all eight checks pass, architecture is gfx1151, and no new blocking kernel-log finding appears.

- [ ] **Step 6: Commit integration coverage**

```bash
git add tests/container/test_repair_flow.py
git commit -m "test: cover immutable repair flow"
```

### Task 13: Run Runtime and Repair Regression Gates

**Files:**
- Modify only files implicated by a failing check

- [ ] **Step 1: Run all focused tests**

Run:

```bash
uv run pytest tests/unit/project tests/unit/container tests/unit/overlay \
  tests/unit/doctor tests/cli/test_project_commands.py \
  tests/cli/test_doctor_commands.py -q
```

Expected: zero failures.

- [ ] **Step 2: Run the full non-hardware suite**

Run: `uv run pytest -m 'not hardware' -q`

Expected: zero failures.

- [ ] **Step 3: Scan for forbidden destructive repair operations**

Run:

```bash
rg -n 'system prune|image prune|--force-reinstall|pip install.*torch|latest' \
  src images templates
```

Expected: no repair implementation contains broad prune, in-place Torch reinstall, or `latest` deployment fallback. Test fixtures and explanatory error messages may contain the searched text only where an assertion explicitly rejects it.

- [ ] **Step 4: Record the clean checkpoint**

```bash
git status --short
git log --oneline -15
```

Expected: clean worktree with runtime, doctor, and repair commits in task order.
