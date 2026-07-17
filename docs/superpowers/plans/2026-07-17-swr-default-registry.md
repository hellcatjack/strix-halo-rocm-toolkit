# SWR-First Registry Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Huawei Cloud SWR the default anonymous stable-image source, fall back to GHCR only for acquisition failures, and preserve exact image identity and installer resume behavior.

**Architecture:** A pure registry-policy module derives immutable SWR or GHCR `StableRelease` candidates without changing any digest. Installer and repair workflows iterate candidates in deterministic order, distinguish acquisition failures from identity failures, and persist the references that were actually verified. Doctor checks accept either trusted repository while all network behavior remains mockable.

**Tech Stack:** Python 3.12 standard library, Docker Engine CLI, Huawei Cloud SWR, GHCR, pytest, existing installer state/progress framework.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `src/amd_ai/installer/registry.py` | Registry choices, trusted repository mapping, immutable candidate derivation |
| `src/amd_ai/installer/models.py` | Validated registry option on `InstallOptions` |
| `src/amd_ai/installer/workflow.py` | Candidate acquisition loop, fallback policy, actual reference persistence |
| `src/amd_ai/installer/actions.py` | Candidate-aware disk estimates and acquisition error classification |
| `src/amd_ai/installer/progress.py` | Registry policy and fallback progress text |
| `src/amd_ai/installer/fixture.py` | Deterministic verified-image output for CLI fixtures |
| `src/amd_ai/cli.py` | `--registry` for install, doctor, and repair |
| `src/amd_ai/doctor/checks.py` | Read-only health checks across trusted registry candidates |
| `src/amd_ai/doctor/repair.py` | SWR-first parent reacquisition and exact-reference planning |
| `tests/unit/installer/test_registry.py` | Pure registry policy tests |
| `tests/unit/installer/test_models.py` | Registry option validation |
| `tests/unit/installer/test_workflow.py` | Fallback, identity blocking, state, and local-build tests |
| `tests/unit/installer/test_actions.py` | Disk estimate source fallback tests |
| `tests/unit/installer/test_progress.py` | User-visible registry plan and diagnostics |
| `tests/unit/installer/fakes.py` | Candidate-aware fake acquisition results |
| `tests/cli/test_installer_commands.py` | CLI default and explicit selection tests |
| `tests/cli/test_doctor_commands.py` | Doctor/repair registry forwarding tests |
| `tests/unit/doctor/test_checks.py` | SWR and GHCR read-only health tests |
| `tests/unit/doctor/test_repair.py` | SWR-first repair acquisition tests |
| `README.md` | China-first quick start and registry override guidance |
| `docs/install.md` | Acquisition policy, progress, resume, and acceptance procedure |
| `docs/release-chain.md` | Canonical release identity versus distribution replicas |
| `docs/releases/v0.3.3.md` | Release notes and upgrade instructions |
| `src/amd_ai/__init__.py` | Toolkit version `0.3.3` |
| `tests/test_version.py` | Version and unchanged stable release contract |

### Task 1: Define The Trusted Registry Policy

**Files:**
- Create: `src/amd_ai/installer/registry.py`
- Create: `tests/unit/installer/test_registry.py`

- [ ] **Step 1: Write failing registry policy tests**

Create `tests/unit/installer/test_registry.py`:

```python
from pathlib import Path

import pytest

from amd_ai.installer.registry import (
    RegistryChoice,
    RegistryPolicyError,
    registry_candidates,
)
from amd_ai.installer.release import load_stable_release


FIXTURE = Path("tests/fixtures/releases/stable.json")
SWR = "swr.cn-east-3.myhuaweicloud.com/hellcat-home"


def test_auto_registry_prefers_swr_then_canonical_ghcr() -> None:
    release = load_stable_release(FIXTURE)

    candidates = registry_candidates(release, RegistryChoice.AUTO)

    assert tuple(candidate.name for candidate in candidates) == ("swr", "ghcr")
    assert candidates[0].release.base.image == (
        f"{SWR}/strix-halo-rocm-python"
    )
    assert candidates[0].release.torch.image == (
        f"{SWR}/strix-halo-rocm-pytorch"
    )
    assert candidates[1].release == release


def test_registry_candidate_changes_only_repository_names() -> None:
    release = load_stable_release(FIXTURE)

    mirrored = registry_candidates(release, "swr")[0].release

    assert mirrored.base.manifest_digest == release.base.manifest_digest
    assert mirrored.base.config_digest == release.base.config_digest
    assert mirrored.base.artifact_digests == release.base.artifact_digests
    assert mirrored.torch.manifest_digest == release.torch.manifest_digest
    assert mirrored.torch.config_digest == release.torch.config_digest
    assert mirrored.torch.artifact_digests == release.torch.artifact_digests


@pytest.mark.parametrize(
    ("choice", "names"),
    (("swr", ("swr",)), ("ghcr", ("ghcr",))),
)
def test_explicit_registry_has_one_candidate(
    choice: str, names: tuple[str, ...]
) -> None:
    release = load_stable_release(FIXTURE)

    assert tuple(
        candidate.name for candidate in registry_candidates(release, choice)
    ) == names


def test_unmapped_custom_release_uses_canonical_source_in_auto() -> None:
    release = load_stable_release(FIXTURE)
    custom = release.__class__(
        **{
            **release.__dict__,
            "base": release.base.__class__(
                image="ghcr.io/example/base",
                manifest_digest=release.base.manifest_digest,
                config_digest=release.base.config_digest,
                artifact_digests=release.base.artifact_digests,
            ),
            "torch": release.torch.__class__(
                image="ghcr.io/example/torch",
                manifest_digest=release.torch.manifest_digest,
                config_digest=release.torch.config_digest,
                artifact_digests=release.torch.artifact_digests,
            ),
        }
    )

    candidates = registry_candidates(custom, "auto")

    assert len(candidates) == 1
    assert candidates[0].name == "ghcr"
    assert candidates[0].release == custom


def test_explicit_swr_rejects_unmapped_release() -> None:
    release = load_stable_release(FIXTURE)
    custom_base = release.base.__class__(
        image="ghcr.io/example/base",
        manifest_digest=release.base.manifest_digest,
        config_digest=release.base.config_digest,
        artifact_digests=release.base.artifact_digests,
    )
    custom = release.__class__(
        **{**release.__dict__, "base": custom_base}
    )

    with pytest.raises(RegistryPolicyError, match="SWR mapping"):
        registry_candidates(custom, "swr")
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run pytest tests/unit/installer/test_registry.py -q
```

Expected: collection fails because `amd_ai.installer.registry` does not exist.

- [ ] **Step 3: Implement the pure registry policy**

Create `src/amd_ai/installer/registry.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from amd_ai.installer.models import StableRelease


class RegistryPolicyError(ValueError):
    pass


class RegistryChoice(StrEnum):
    AUTO = "auto"
    SWR = "swr"
    GHCR = "ghcr"


@dataclass(frozen=True)
class RegistryCandidate:
    name: str
    label: str
    release: StableRelease


SWR_REPOSITORIES: Mapping[str, str] = MappingProxyType(
    {
        "ghcr.io/hellcatjack/strix-halo-rocm-python": (
            "swr.cn-east-3.myhuaweicloud.com/hellcat-home/"
            "strix-halo-rocm-python"
        ),
        "ghcr.io/hellcatjack/strix-halo-rocm-pytorch": (
            "swr.cn-east-3.myhuaweicloud.com/hellcat-home/"
            "strix-halo-rocm-pytorch"
        ),
    }
)


def registry_candidates(
    release: StableRelease,
    choice: RegistryChoice | str = RegistryChoice.AUTO,
) -> tuple[RegistryCandidate, ...]:
    try:
        selected = RegistryChoice(choice)
    except (TypeError, ValueError) as error:
        raise RegistryPolicyError("registry choice is invalid") from error
    candidates: list[RegistryCandidate] = []
    mirrored = _swr_release(release)
    if selected in (RegistryChoice.AUTO, RegistryChoice.SWR):
        if mirrored is None:
            if selected is RegistryChoice.SWR:
                raise RegistryPolicyError(
                    "stable release has no trusted SWR mapping"
                )
        else:
            candidates.append(
                RegistryCandidate("swr", "华为 SWR", mirrored)
            )
    if selected in (RegistryChoice.AUTO, RegistryChoice.GHCR):
        candidates.append(
            RegistryCandidate("ghcr", "GHCR", release)
        )
    return tuple(candidates)


def registry_plan_label(choice: RegistryChoice | str) -> str:
    selected = RegistryChoice(choice)
    if selected is RegistryChoice.AUTO:
        return "auto（华为 SWR 优先，GHCR 回退）"
    if selected is RegistryChoice.SWR:
        return "swr（仅华为 SWR，不回退）"
    return "ghcr（仅 GHCR，不回退）"


def trusted_image_references(
    release: StableRelease,
    *,
    kind: str,
) -> tuple[str, ...]:
    if kind not in {"base", "torch"}:
        raise RegistryPolicyError("release image kind is invalid")
    return tuple(
        getattr(candidate.release, kind).reference
        for candidate in registry_candidates(release, RegistryChoice.AUTO)
    )


def _swr_release(release: StableRelease) -> StableRelease | None:
    base = SWR_REPOSITORIES.get(release.base.image)
    torch = SWR_REPOSITORIES.get(release.torch.image)
    if base is None or torch is None:
        return None
    return replace(
        release,
        base=replace(release.base, image=base),
        torch=replace(release.torch, image=torch),
    )
```

- [ ] **Step 4: Run registry tests and existing release tests**

Run:

```bash
uv run pytest \
  tests/unit/installer/test_registry.py \
  tests/unit/installer/test_release.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the policy module**

```bash
git add src/amd_ai/installer/registry.py tests/unit/installer/test_registry.py
git commit -m "feat: define trusted SWR registry policy"
```

### Task 2: Add Registry Selection To CLI, Models, And Progress

**Files:**
- Modify: `src/amd_ai/installer/models.py`
- Modify: `src/amd_ai/installer/progress.py`
- Modify: `src/amd_ai/cli.py`
- Modify: `tests/unit/installer/test_models.py`
- Modify: `tests/unit/installer/test_progress.py`
- Modify: `tests/cli/test_installer_commands.py`

- [ ] **Step 1: Write failing option and progress tests**

Add to `tests/unit/installer/test_models.py`:

```python
@pytest.mark.parametrize("registry", ("auto", "swr", "ghcr"))
def test_registry_choice_is_valid(registry: str, tmp_path: Path) -> None:
    options = InstallOptions(
        mode=InstallMode.CONTAINER,
        project_dir=tmp_path / "project",
        image_source="pull",
        registry=registry,
    ).validate()

    assert options.registry == registry


def test_unknown_registry_choice_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(InstallerModelError, match="registry"):
        InstallOptions(
            mode=InstallMode.CONTAINER,
            project_dir=tmp_path / "project",
            image_source="pull",
            registry="nearest",
        ).validate()
```

Add `registry="auto"` to `container_plan()` in
`tests/unit/installer/test_progress.py`, then assert:

```python
assert "镜像仓库=auto（华为 SWR 优先，GHCR 回退）" in output
```

Add to `tests/cli/test_installer_commands.py`:

```python
def test_install_registry_defaults_to_auto_and_accepts_explicit_swr() -> None:
    default = cli.build_parser().parse_args(["install"])
    explicit = cli.build_parser().parse_args(
        ["install", "--registry", "swr"]
    )

    assert default.registry == "auto"
    assert explicit.registry == "swr"
```

In `test_install_dispatch_constructs_workflow_once`, add
`"--registry", "ghcr"` and assert:

```python
assert captured["options"].registry == "ghcr"
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
uv run pytest \
  tests/unit/installer/test_models.py \
  tests/unit/installer/test_progress.py \
  tests/cli/test_installer_commands.py -q
```

Expected: failures report missing `registry` fields and CLI argument.

- [ ] **Step 3: Implement validated options and CLI forwarding**

In `InstallOptions` add:

```python
registry: str = "auto"
```

In `InstallOptions.validate()` add:

```python
if self.registry not in ("auto", "swr", "ghcr"):
    raise InstallerModelError(
        "registry must be auto, swr, or ghcr"
    )
```

In `build_parser()` add to the install parser:

```python
install.add_argument(
    "--registry",
    choices=("auto", "swr", "ghcr"),
    default="auto",
)
```

Pass `registry=args.registry` into `InstallOptions`.

In `SessionPlan` add:

```python
registry: str
```

In `InstallerProgress.session_plan()` import `registry_plan_label` and emit:

```python
self._emit(
    "PLAN",
    f"镜像来源={plan.image_source}，"
    f"镜像仓库={registry_plan_label(plan.registry)}，"
    f"stable release={release_id}",
)
```

Pass `self.options.registry` when constructing `SessionPlan` in the workflow.

- [ ] **Step 4: Run option and progress tests**

Run:

```bash
uv run pytest \
  tests/unit/installer/test_models.py \
  tests/unit/installer/test_progress.py \
  tests/cli/test_installer_commands.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit user-facing selection**

```bash
git add \
  src/amd_ai/installer/models.py \
  src/amd_ai/installer/progress.py \
  src/amd_ai/cli.py \
  tests/unit/installer/test_models.py \
  tests/unit/installer/test_progress.py \
  tests/cli/test_installer_commands.py
git commit -m "feat: expose stable image registry selection"
```

### Task 3: Implement SWR-First Acquisition And State Persistence

**Files:**
- Modify: `src/amd_ai/installer/workflow.py`
- Modify: `src/amd_ai/installer/fixture.py`
- Modify: `tests/unit/installer/fakes.py`
- Modify: `tests/unit/installer/test_workflow.py`
- Modify: `tests/cli/test_installer_resume.py`

- [ ] **Step 1: Make the fake return actual verified references**

In `tests/unit/installer/fakes.py`, import
`VerifiedImageIdentity` and `VerifiedReleaseImages`. Replace
`FakeInstallerActions.pull_release()` with:

```python
def pull_release(self, release: StableRelease) -> VerifiedReleaseImages:
    self._raise_if_needed(InstallStage.IMAGE_PULL_OR_BUILD)
    self.calls.append("pull_release")
    self.image_calls.extend(
        (("pull", release.base.reference), ("pull", release.torch.reference))
    )
    error = self.pull_errors.get(release.base.image)
    if error is not None:
        raise error
    return VerifiedReleaseImages(
        base=VerifiedImageIdentity(
            reference=release.base.reference,
            config_digest=release.base.config_digest,
            repo_digests=(release.base.reference,),
            labels={},
        ),
        torch=VerifiedImageIdentity(
            reference=release.torch.reference,
            config_digest=release.torch.config_digest,
            repo_digests=(release.torch.reference,),
            labels={},
        ),
    )
```

Initialize:

```python
self.pull_errors: dict[str, BaseException] = {}
```

Keep compatibility with existing tests by converting assignments to
`actions.pull_error` into `actions.pull_errors[release.base.image]`.

- [ ] **Step 2: Write failing workflow behavior tests**

Add to `tests/unit/installer/test_workflow.py`:

```python
def test_auto_registry_falls_back_from_swr_acquisition_to_ghcr(
    tmp_path: Path,
) -> None:
    actions = FakeInstallerActions.healthy()
    swr = registry_candidates(actions.release, "swr")[0].release
    actions.pull_errors[swr.base.image] = ReleaseAcquisitionError(
        "SWR timeout"
    )

    result = installer_workflow(tmp_path, actions=actions).run()

    assert result.exit_code == 0
    assert actions.image_calls[:2] == [
        ("pull", swr.base.reference),
        ("pull", swr.torch.reference),
    ]
    assert actions.image_calls[2:] == [
        ("pull", actions.release.base.reference),
        ("pull", actions.release.torch.reference),
    ]
    assert result.state is not None
    assert result.state.base_image_reference == actions.release.base.reference
    assert result.state.torch_image_reference == actions.release.torch.reference


def test_successful_swr_pull_persists_verified_swr_references(
    tmp_path: Path,
) -> None:
    actions = FakeInstallerActions.healthy()
    swr = registry_candidates(actions.release, "swr")[0].release

    result = installer_workflow(tmp_path, actions=actions).run()

    assert result.exit_code == 0
    assert result.state is not None
    assert result.state.base_image_reference == swr.base.reference
    assert result.state.torch_image_reference == swr.torch.reference
    assert actions.project_init_kwargs["base_image_reference"] == (
        swr.torch.reference
    )


def test_swr_identity_failure_blocks_without_ghcr_or_build(
    tmp_path: Path,
) -> None:
    actions = FakeInstallerActions.healthy()
    swr = registry_candidates(actions.release, "swr")[0].release
    actions.pull_errors[swr.base.image] = ReleaseIdentityError(
        "config digest changed"
    )

    result = installer_workflow(
        tmp_path,
        actions=actions,
        prompts=FakePrompts(image_fallback="build"),
    ).run()

    assert result.exit_code == 2
    assert actions.image_calls == [
        ("pull", swr.base.reference),
        ("pull", swr.torch.reference),
    ]
    assert "build_local_images" not in actions.calls


@pytest.mark.parametrize("registry", ("swr", "ghcr"))
def test_explicit_registry_never_cross_falls_back(
    tmp_path: Path, registry: str
) -> None:
    actions = FakeInstallerActions.healthy()
    selected = registry_candidates(actions.release, registry)[0].release
    actions.pull_errors[selected.base.image] = ReleaseAcquisitionError(
        "registry unavailable"
    )

    result = installer_workflow(
        tmp_path,
        actions=actions,
        options=replace(container_options(tmp_path), registry=registry),
        prompts=FakePrompts(image_fallback="cancel"),
    ).run()

    assert result.exit_code == 2
    assert len(actions.image_calls) == 2
    assert all(
        reference.startswith(selected.base.image.rsplit("/", 1)[0])
        for _, reference in actions.image_calls
    )
```

Add a resume test that completes with GHCR using installer `0.3.2`, resumes
with registry `auto` using `0.3.3`, and asserts `pull_release` is not replayed
and the recorded GHCR exact references remain unchanged.

- [ ] **Step 3: Run workflow tests and verify RED**

Run:

```bash
uv run pytest \
  tests/unit/installer/test_workflow.py \
  tests/cli/test_installer_resume.py -q
```

Expected: SWR tests fail because the workflow still pulls only the canonical
release and does not apply verified references.

- [ ] **Step 4: Implement the candidate loop**

In `InstallerWorkflow`, add:

```python
def _pull_release_candidates(
    self, release: StableRelease
) -> VerifiedReleaseImages:
    candidates = registry_candidates(release, self.options.registry)
    failures: list[str] = []
    for index, candidate in enumerate(candidates):
        self.progress.detail(
            f"当前仓库={candidate.label}，来源=公开匿名镜像"
        )
        try:
            verified = self.actions.pull_release(candidate.release)
        except ReleaseIdentityError:
            raise
        except ReleaseAcquisitionError as error:
            failures.append(f"{candidate.label}: {error}")
            if index + 1 < len(candidates):
                next_candidate = candidates[index + 1]
                self._status(
                    "WARN",
                    f"{candidate.label} 获取失败，正在回退 "
                    f"{next_candidate.label}：{error}",
                )
            continue
        if not isinstance(verified, VerifiedReleaseImages):
            raise WorkflowError(
                "release pull returned invalid verified identities"
            )
        self.progress.detail(
            f"已采用仓库={candidate.label}，"
            f"base={verified.base.reference}，"
            f"torch={verified.torch.reference}"
        )
        return verified
    raise ReleaseAcquisitionError(
        "all configured registries failed: " + "; ".join(failures)
    )
```

Replace `self.actions.pull_release(release)` with
`self._pull_release_candidates(release)`.

In `_apply_output()`, before the `LocalBuildResult` branch add:

```python
elif (
    stage is InstallStage.IMAGE_PULL_OR_BUILD
    and isinstance(output, VerifiedReleaseImages)
):
    changes.update(
        {
            "base_image_reference": output.base.reference,
            "base_manifest_digest": output.base.reference.rpartition("@")[2],
            "base_config_digest": output.base.config_digest,
            "torch_image_reference": output.torch.reference,
            "torch_manifest_digest": output.torch.reference.rpartition("@")[2],
            "torch_config_digest": output.torch.config_digest,
        }
    )
```

Do not add `registry` to completed-stage input digests. Actual exact
references already bind the completed image-acquisition output, and this keeps
existing trusted checkpoints compatible.

Update `FixtureInstallerActions.pull_release()` to return
`VerifiedReleaseImages` for the release argument using the same construction
as the fake.

- [ ] **Step 5: Run workflow and resume tests**

Run:

```bash
uv run pytest \
  tests/unit/installer/test_workflow.py \
  tests/cli/test_installer_resume.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit acquisition behavior**

```bash
git add \
  src/amd_ai/installer/workflow.py \
  src/amd_ai/installer/fixture.py \
  tests/unit/installer/fakes.py \
  tests/unit/installer/test_workflow.py \
  tests/cli/test_installer_resume.py
git commit -m "feat: prefer SWR for stable image acquisition"
```

### Task 4: Make Disk Estimation Follow Acquisition Policy

**Files:**
- Modify: `src/amd_ai/installer/actions.py`
- Modify: `src/amd_ai/installer/workflow.py`
- Modify: `tests/unit/installer/test_actions.py`
- Modify: `tests/unit/installer/fakes.py`
- Modify: `tests/unit/installer/test_workflow.py`

- [ ] **Step 1: Write failing disk-estimate fallback tests**

Add to `tests/unit/installer/test_actions.py`:

```python
def test_image_disk_estimate_falls_back_from_swr_manifest_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = load_stable_release(RELEASE_FIXTURE)
    calls: list[str] = []

    monkeypatch.setattr(
        actions,
        "_docker_root_and_available",
        lambda runner, prefix: (Path("/var/lib/docker"), 100 * 1024**3),
    )

    def missing(candidate, runner, prefix):
        del runner, prefix
        calls.append(candidate.base.image)
        if "myhuaweicloud.com" in candidate.base.image:
            raise ReleaseAcquisitionError("SWR manifest timeout")
        return 12 * 1024**3

    monkeypatch.setattr(actions, "_missing_release_layer_bytes", missing)

    estimate = ProductionInstallerActions().image_disk_estimate(
        release=release,
        image_source="pull",
        registry="auto",
    )

    assert estimate.payload_bytes == 12 * 1024**3
    assert calls == [
        (
            "swr.cn-east-3.myhuaweicloud.com/hellcat-home/"
            "strix-halo-rocm-python"
        ),
        "ghcr.io/hellcatjack/strix-halo-rocm-python",
    ]


def test_image_disk_estimate_does_not_hide_invalid_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = load_stable_release(RELEASE_FIXTURE)
    monkeypatch.setattr(
        actions,
        "_docker_root_and_available",
        lambda runner, prefix: (Path("/var/lib/docker"), 100 * 1024**3),
    )
    monkeypatch.setattr(
        actions,
        "_missing_release_layer_bytes",
        lambda release, runner, prefix: (_ for _ in ()).throw(
            ActionError("remote layer identity is invalid")
        ),
    )

    with pytest.raises(ActionError, match="identity"):
        ProductionInstallerActions().image_disk_estimate(
            release=release,
            image_source="pull",
            registry="auto",
        )
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
uv run pytest tests/unit/installer/test_actions.py -q
```

Expected: `image_disk_estimate` rejects the new `registry` argument and no
candidate fallback exists.

- [ ] **Step 3: Classify manifest lookup acquisition failures**

In `_missing_release_layer_bytes()`, change only a nonzero
`docker manifest inspect` result to:

```python
raise ReleaseAcquisitionError(
    f"cannot estimate remote image layers for {image.reference}: {evidence}"
)
```

Keep malformed JSON, ambiguous layer data, and invalid digest/size records as
`ActionError`, so they cannot trigger source fallback.

Change `ProductionInstallerActions.image_disk_estimate()` to:

```python
def image_disk_estimate(
    self,
    *,
    release: StableRelease,
    image_source: str,
    registry: str = "auto",
) -> DiskSpaceEstimate:
    location, available = _docker_root_and_available(
        self.runner, self.docker_prefix
    )
    if image_source == "pull":
        failures: list[str] = []
        for candidate in registry_candidates(release, registry):
            try:
                payload = _missing_release_layer_bytes(
                    candidate.release,
                    self.runner,
                    self.docker_prefix,
                )
                break
            except ReleaseAcquisitionError as error:
                failures.append(f"{candidate.label}: {error}")
        else:
            raise ActionError(
                "cannot estimate any configured registry: "
                + "; ".join(failures)
            )
    elif image_source == "build":
        payload = LOCAL_BUILD_ESTIMATE_BYTES
    else:
        raise ActionError("image source must be pull or build")
    return DiskSpaceEstimate(
        location=location,
        payload_bytes=payload,
        available_bytes=available,
    )
```

Pass `registry=self.options.registry` from
`InstallerWorkflow._image_disk_requirement()`. Update the fake method to
accept and record the argument without network access.

- [ ] **Step 4: Run action and workflow tests**

Run:

```bash
uv run pytest \
  tests/unit/installer/test_actions.py \
  tests/unit/installer/test_workflow.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit disk planning behavior**

```bash
git add \
  src/amd_ai/installer/actions.py \
  src/amd_ai/installer/workflow.py \
  tests/unit/installer/test_actions.py \
  tests/unit/installer/fakes.py \
  tests/unit/installer/test_workflow.py
git commit -m "feat: estimate disk across stable registries"
```

### Task 5: Make Doctor And Repair Registry-Aware

**Files:**
- Modify: `src/amd_ai/doctor/checks.py`
- Modify: `src/amd_ai/doctor/repair.py`
- Modify: `src/amd_ai/cli.py`
- Modify: `tests/unit/doctor/test_checks.py`
- Modify: `tests/unit/doctor/test_repair.py`
- Modify: `tests/cli/test_doctor_commands.py`

- [ ] **Step 1: Write failing doctor candidate tests**

Add to `tests/unit/doctor/test_checks.py`:

```python
def test_platform_accepts_verified_swr_parent_images() -> None:
    release = load_stable_release(FIXTURE)
    swr = registry_candidates(release, "swr")[0].release
    backend = FakeDoctorBackend(swr)

    report = doctor_platform(
        manifest_path=FIXTURE,
        backend=backend,
        registry="auto",
    )

    assert report.status == "pass"
    assert report.facts["base_reference"] == swr.base.reference
    assert report.facts["torch_reference"] == swr.torch.reference


def test_platform_uses_canonical_parent_when_swr_is_absent() -> None:
    release = load_stable_release(FIXTURE)
    backend = FakeDoctorBackend(release)

    report = doctor_platform(
        manifest_path=FIXTURE,
        backend=backend,
        registry="auto",
    )

    assert report.status == "pass"
    assert report.facts["torch_reference"] == release.torch.reference
```

No fake-backend implementation change is required:
`FakeDoctorBackend.__init__()` already indexes whichever derived release it
receives and constructs labels from release metadata.

- [ ] **Step 2: Write failing repair tests**

Replace the GHCR-only plan assertion with an allowlisted SWR fact:

```python
def test_repair_plan_accepts_trusted_swr_exact_parent() -> None:
    report = repairable_project_report()
    release = load_stable_release(RELEASE_FIXTURE)
    swr = registry_candidates(release, "swr")[0].release
    report = replace(
        report,
        facts=MappingProxyType(
            {**report.facts, "torch_reference": swr.torch.reference}
        ),
    )

    plan = plan_repair(report)

    assert plan.actions[0] == RepairAction(
        "pull-parent",
        swr.torch.reference,
        "IMAGE.PARENT_MISSING",
    )
```

Add:

```python
def test_system_executor_falls_back_to_ghcr_on_swr_acquisition_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        repair.Docker,
        "detect",
        lambda: SimpleNamespace(prefix=("sudo", "-n", "docker")),
    )
    executor = SystemRepairExecutor(
        manifest_path=RELEASE_FIXTURE,
        registry="auto",
    )
    swr = registry_candidates(executor.release, "swr")[0].release
    calls: list[str] = []

    def pull(candidate, *, docker):
        del docker
        calls.append(candidate.base.image)
        if candidate.base.image == swr.base.image:
            raise ReleaseAcquisitionError("SWR unavailable")
        return VerifiedReleaseImages(
            base=VerifiedImageIdentity(
                reference=candidate.base.reference,
                config_digest=candidate.base.config_digest,
                repo_digests=(candidate.base.reference,),
                labels={},
            ),
            torch=VerifiedImageIdentity(
                reference=candidate.torch.reference,
                config_digest=candidate.torch.config_digest,
                repo_digests=(candidate.torch.reference,),
                labels={},
            ),
        )

    monkeypatch.setattr(repair, "pull_and_verify_release", pull)
    monkeypatch.setattr(
        executor.registry,
        "tag_reference",
        lambda source, target: None,
    )

    executor.pull_and_verify(executor.release)

    assert calls == [swr.base.image, executor.release.base.image]
```

Add:

```python
def test_system_executor_does_not_hide_swr_identity_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        repair.Docker,
        "detect",
        lambda: SimpleNamespace(prefix=("sudo", "-n", "docker")),
    )
    executor = SystemRepairExecutor(
        manifest_path=RELEASE_FIXTURE,
        registry="auto",
    )
    calls: list[str] = []

    def pull(candidate, *, docker):
        del docker
        calls.append(candidate.base.image)
        raise ReleaseIdentityError("config digest changed")

    monkeypatch.setattr(repair, "pull_and_verify_release", pull)

    with pytest.raises(ReleaseIdentityError, match="config digest"):
        executor.pull_and_verify(executor.release)

    assert len(calls) == 1
    assert "myhuaweicloud.com" in calls[0]
```

Import `VerifiedImageIdentity`, `VerifiedReleaseImages`,
`ReleaseAcquisitionError`, and `ReleaseIdentityError` from
`amd_ai.installer.release`.

- [ ] **Step 3: Run doctor tests and verify RED**

Run:

```bash
uv run pytest \
  tests/unit/doctor/test_checks.py \
  tests/unit/doctor/test_repair.py \
  tests/cli/test_doctor_commands.py -q
```

Expected: failures show canonical-only doctor inspection, GHCR-only repair
planning, and missing registry arguments.

- [ ] **Step 4: Implement candidate-aware doctor checks**

Add `registry: str = "auto"` to `doctor_platform()`, `doctor_project()`, and
`run_doctor()`.

For each image kind, derive `registry_candidates(release, registry)` and select
the first candidate whose exact image is present:

```python
selected_release = candidates[0].release
selected_image = getattr(selected_release, kind)
inspection = None
for candidate in candidates:
    candidate_image = getattr(candidate.release, kind)
    observed = backend.inspect_image(candidate_image.reference)
    if observed is not None:
        selected_release = candidate.release
        selected_image = candidate_image
        inspection = observed
        break
```

Use `selected_release` and `selected_image` for static verification and store
the selected exact references in report facts. If no candidate is present,
report the first preferred exact reference as repairable. Use the selected
Torch exact reference for the GPU runtime probe.

- [ ] **Step 5: Implement exact allowlisted repair and acquisition**

Remove `EXACT_REFERENCE_PATTERN`. In `plan_repair()` validate the report fact
with:

```python
allowed = trusted_image_references(release, kind="torch")
if reference not in allowed:
    raise RepairPlanningError(
        "parent repair reference is not a trusted release replica"
    )
```

Add `registry: str = "auto"` to `SystemRepairExecutor.__init__()`. In
`pull_and_verify()` iterate `registry_candidates()` exactly as the installer
does. Continue only after `ReleaseAcquisitionError`; propagate
`ReleaseIdentityError` immediately. Tag `verified.base.reference` and
`verified.torch.reference`, not the canonical manifest references.

Add `--registry auto|swr|ghcr` to doctor and repair parsers and pass it through
`run_doctor()`, `_interactive_doctor()`, and `SystemRepairExecutor`.

- [ ] **Step 6: Run doctor and repair tests**

Run:

```bash
uv run pytest \
  tests/unit/doctor/test_checks.py \
  tests/unit/doctor/test_repair.py \
  tests/cli/test_doctor_commands.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit doctor and repair support**

```bash
git add \
  src/amd_ai/doctor/checks.py \
  src/amd_ai/doctor/repair.py \
  src/amd_ai/cli.py \
  tests/unit/doctor/test_checks.py \
  tests/unit/doctor/test_repair.py \
  tests/cli/test_doctor_commands.py
git commit -m "feat: use trusted registries in doctor repair"
```

### Task 6: Document The China-First Workflow And Bump Toolkit Version

**Files:**
- Modify: `README.md`
- Modify: `docs/install.md`
- Modify: `docs/release-chain.md`
- Create: `docs/releases/v0.3.3.md`
- Modify: `src/amd_ai/__init__.py`
- Modify: `tests/test_version.py`

- [ ] **Step 1: Write the failing version contract**

Change `tests/test_version.py`:

```python
def test_version_constant_and_cli(capsys):
    assert __version__ == "0.3.3"
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == "amd-ai 0.3.3"
```

Keep the stable manifest hash and exact digest assertions unchanged.

- [ ] **Step 2: Run version tests and verify RED**

Run:

```bash
uv run pytest tests/test_version.py -q
```

Expected: version assertion fails with `0.3.2`.

- [ ] **Step 3: Bump version and write release notes**

Set:

```python
__version__ = "0.3.3"
```

Create `docs/releases/v0.3.3.md` with:

```markdown
# v0.3.3

`v0.3.3` makes the public Huawei Cloud SWR replicas the default stable image
source. Stable release `0.2.0`, ROCm 7.2.1, Python 3.12, PyTorch 2.9.1, and
both qualified image digests remain unchanged.

## Default

`--registry auto` tries anonymous SWR exact digests first and falls back to
GHCR only for acquisition failures. Digest, label, or embedded-lock failures
block immediately.

## Overrides

Use `--registry swr` to require SWR or `--registry ghcr` to require GHCR.

## Validation Boundary

Automated tests use fake registries. The US development host does not perform a
cold SWR pull. Cold anonymous pull and GPU runtime acceptance are performed on
the China-based Ubuntu 24.04 / Ryzen AI Max+ 395 test host.
```

- [ ] **Step 4: Update user documentation**

In the README quick start, use:

```bash
./install.sh --mode full \
  --project-dir "$HOME/ai-projects/video-lab" \
  --project-name video-lab \
  --image-source pull
```

Explain that omitted `--registry` means SWR-first `auto`. Add exact overrides:

```bash
./install.sh ... --registry swr
./install.sh ... --registry ghcr
```

Document both SWR exact references:

```text
swr.cn-east-3.myhuaweicloud.com/hellcat-home/strix-halo-rocm-python@sha256:e9991f97f578156c8620fbb587d2d34504eb632f165cc5597deaadaa3e692a12
swr.cn-east-3.myhuaweicloud.com/hellcat-home/strix-halo-rocm-pytorch@sha256:dc0bb217474cfd4f602423bd3bf4fe8714b03e900cf3c6b4417b99e622ebcf8b
```

Update `docs/install.md` and `docs/release-chain.md` with the same fallback and
identity-blocking distinction. State explicitly that no registry credential is
required for users.

- [ ] **Step 5: Run documentation and version checks**

Run:

```bash
uv run pytest tests/test_version.py tests/cli/test_installer_commands.py -q
git diff --check
rg -n "默认.*GHCR|公开 GHCR" README.md docs/install.md docs/release-chain.md
```

Expected: tests pass, `git diff --check` is silent, and every remaining GHCR
statement describes fallback, canonical publication, or an explicit override.

- [ ] **Step 6: Commit docs and version**

```bash
git add \
  README.md \
  docs/install.md \
  docs/release-chain.md \
  docs/releases/v0.3.3.md \
  src/amd_ai/__init__.py \
  tests/test_version.py
git commit -m "docs: publish SWR-first installer workflow"
```

### Task 7: Run Offline Regression And Review

**Files:**
- Review all files changed since `7093ace`

- [ ] **Step 1: Run focused registry regression**

Run:

```bash
uv run pytest \
  tests/unit/installer/test_registry.py \
  tests/unit/installer/test_models.py \
  tests/unit/installer/test_release.py \
  tests/unit/installer/test_actions.py \
  tests/unit/installer/test_progress.py \
  tests/unit/installer/test_workflow.py \
  tests/unit/doctor/test_checks.py \
  tests/unit/doctor/test_repair.py \
  tests/cli/test_installer_commands.py \
  tests/cli/test_installer_resume.py \
  tests/cli/test_doctor_commands.py \
  tests/test_version.py -q
```

Expected: all focused tests pass without contacting SWR.

- [ ] **Step 2: Run the reliable software regression**

Run:

```bash
uv run pytest tests/unit tests/cli tests/test_version.py -q
```

Expected: all tests pass. Do not run a real SWR pull on this US host.

- [ ] **Step 3: Run static repository checks**

Run:

```bash
git diff --check
git status --short
rg -n \
  "swr\\.cn-east-3\\.myhuaweicloud\\.com|--registry|RegistryChoice" \
  src tests README.md docs
```

Expected: no whitespace errors; only intended files are modified; SWR
references occur in the registry policy, tests, and documentation.

- [ ] **Step 4: Review fallback and identity boundaries**

Review the diff and verify:

```text
SWR ReleaseAcquisitionError -> GHCR attempted
SWR ReleaseIdentityError -> immediate block
explicit swr/ghcr -> no cross-registry attempt
successful verification -> actual exact references persisted
completed v0.3.2 state -> no image stage replay
no test command -> real SWR layer download
```

- [ ] **Step 5: Request code review**

Invoke `superpowers:requesting-code-review` and resolve all valid findings.

- [ ] **Step 6: Re-run regression after review changes**

Run:

```bash
uv run pytest tests/unit tests/cli tests/test_version.py -q
git diff --check
```

Expected: all tests pass and the diff check is silent.

- [ ] **Step 7: Commit any review corrections**

If review required changes:

```bash
git add src tests README.md docs
git commit -m "fix: address SWR registry review findings"
```

If no files changed, do not create an empty commit.

### Task 8: China-Host Acceptance And Release Handoff

**Files:**
- Create after remote test: `docs/releases/v0.3.3-china-acceptance.md`

- [ ] **Step 1: Merge or push the feature only after local regression**

After the feature branch is reviewed and local regression is green, fast
forward `main` from the primary worktree:

```bash
git -C /app/imgMaker merge --ff-only feature/swr-default
```

Do not publish `v0.3.3` before this command succeeds.

- [ ] **Step 2: Run cold anonymous acceptance on the China host**

On the China-based Ubuntu 24.04 host, with an empty Docker auth directory:

```bash
EMPTY_CONFIG="$(mktemp -d)"
sudo docker --config "$EMPTY_CONFIG" pull \
  swr.cn-east-3.myhuaweicloud.com/hellcat-home/strix-halo-rocm-pytorch@sha256:dc0bb217474cfd4f602423bd3bf4fe8714b03e900cf3c6b4417b99e622ebcf8b
rm -rf "$EMPTY_CONFIG"
```

Then run the installer without `--registry` and confirm the plan reports
SWR-first `auto`, the final state records SWR exact references, and the
`IMAGE_VERIFY` Torch GPU runtime stage passes.

- [ ] **Step 3: Record acceptance evidence**

Create `docs/releases/v0.3.3-china-acceptance.md` containing:

```markdown
# v0.3.3 China Acceptance

- Host OS: Ubuntu 24.04
- Kernel: 6.17 OEM
- GPU: AMD Ryzen AI Max+ 395 / Radeon 8060S
- Registry: Huawei Cloud SWR cn-east-3
- Authentication: empty Docker config
- Python image digest: `sha256:e9991f97f578156c8620fbb587d2d34504eb632f165cc5597deaadaa3e692a12`
- PyTorch image digest: `sha256:dc0bb217474cfd4f602423bd3bf4fe8714b03e900cf3c6b4417b99e622ebcf8b`
- Installer result: pass
- Torch GPU runtime: pass
- Pull elapsed time: recorded from the test log
```

Replace only the elapsed-time line with the observed value; all other values
must be verified from command output.

- [ ] **Step 4: Commit acceptance evidence and publish**

```bash
git add docs/releases/v0.3.3-china-acceptance.md
git commit -m "test: record v0.3.3 China SWR acceptance"
git tag -a v0.3.3 -m "v0.3.3"
git push origin main
git push origin v0.3.3
```

Before the push, confirm the branch is integrated into `main` and the tag points
to the acceptance-evidence commit.
