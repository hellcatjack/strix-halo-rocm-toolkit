from __future__ import annotations

import json
import subprocess
from collections import namedtuple
from pathlib import Path

import pytest

from amd_ai.host.models import PreparePlan
from amd_ai.installer import actions
from amd_ai.installer.actions import (
    ActionError,
    AnonymousReleaseRegistry,
    ProductionInstallerActions,
    bind_selected_parent,
    prepare_plan_payload,
    validate_local_build_source,
)
from amd_ai.installer.release import load_stable_release
from amd_ai.installer.state import stage_input_digest
from amd_ai.runner import CommandResult
from tests.unit.host.fakes import healthy_snapshot
from tests.unit.installer.fakes import FakeReleaseDocker


RELEASE_FIXTURE = Path("tests/fixtures/releases/stable.json")


def test_host_plan_uses_existing_probe_and_prepare_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    snapshot = healthy_snapshot()

    class Probe:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def collect(self):
            calls.append("HostProbe.collect")
            return snapshot

    expected = PreparePlan(
        supported=True,
        target_user="developer",
        actions=(),
        reboot_required=False,
    )

    def create(snapshot_value, *, target_user: str, memory_gib=None):
        del snapshot_value, memory_gib
        calls.append("create_prepare_plan")
        assert target_user == "developer"
        return expected

    monkeypatch.setattr(actions, "HostProbe", Probe)
    monkeypatch.setattr(actions, "create_prepare_plan", create)
    monkeypatch.setattr(
        actions, "_target_user_group_ids", lambda target_user: (109, 110)
    )

    result = ProductionInstallerActions(effective_uid=0).host_plan(
        target_user="developer"
    )

    assert calls == ["HostProbe.collect", "create_prepare_plan"]
    assert result.plan == expected
    assert result.plan_digest == stage_input_digest(prepare_plan_payload(expected))
    assert result.adapter_id == "ubuntu-24.04"


def test_release_pull_calls_exact_release_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = load_stable_release(RELEASE_FIXTURE)
    docker = FakeReleaseDocker.for_release(release)
    captured: dict[str, object] = {}

    def pull(value, *, docker):
        captured.update(release=value, registry=docker)
        return "verified"

    monkeypatch.setattr(actions, "pull_and_verify_release", pull)

    result = ProductionInstallerActions(release_docker=docker).pull_release(
        release
    )

    assert result == "verified"
    assert captured == {"release": release, "registry": docker}


def test_container_host_check_blocks_missing_docker_and_gpu_permissions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = healthy_snapshot(
        docker_version=None,
        current_group_ids=(),
    )

    class Probe:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def collect(self):
            return snapshot

    monkeypatch.setattr(actions, "HostProbe", Probe)

    result = ProductionInstallerActions().container_host_check()

    assert result.blocked is True
    assert "Docker" in result.message
    assert result.facts["adapter_id"] == "ubuntu-24.04"


def test_local_build_source_requires_exact_clean_checkout(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    for relative in actions.REQUIRED_LOCAL_BUILD_PATHS:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    expected = "a" * 40
    commands: list[tuple[str, ...]] = []

    def clean_run(argv, **kwargs):
        del kwargs
        command = tuple(argv)
        commands.append(command)
        if command[-2:] == ("rev-parse", "HEAD"):
            return subprocess.CompletedProcess(argv, 0, expected + "\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    assert (
        validate_local_build_source(
            source, expected_revision=expected, run=clean_run
        )
        == source.resolve()
    )
    assert commands[-1][-2:] == ("status", "--porcelain")

    def dirty_run(argv, **kwargs):
        result = clean_run(argv, **kwargs)
        if tuple(argv)[-2:] == ("status", "--porcelain"):
            return subprocess.CompletedProcess(argv, 0, " M pyproject.toml\n", "")
        return result

    with pytest.raises(ActionError, match="not clean"):
        validate_local_build_source(
            source, expected_revision=expected, run=dirty_run
        )


def test_image_disk_estimate_uses_missing_remote_layer_bytes(
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
        lambda release, runner, prefix: 12 * 1024**3,
    )

    estimate = ProductionInstallerActions().image_disk_estimate(
        release=release, image_source="pull"
    )

    assert estimate.location == Path("/var/lib/docker")
    assert estimate.payload_bytes == 12 * 1024**3
    assert estimate.available_bytes == 100 * 1024**3


def test_project_disk_estimate_counts_resolved_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    artifacts = project / ".amd-ai/artifacts/sha256"
    artifacts.mkdir(parents=True)
    (artifacts / ("a" * 64)).write_bytes(b"a" * 11)
    (artifacts / ("b" * 64)).write_bytes(b"b" * 13)
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(
        actions.shutil,
        "disk_usage",
        lambda path: Usage(1000, 100, 900),
    )

    estimate = ProductionInstallerActions().project_disk_estimate(
        project_dir=project
    )

    assert estimate.location == project.resolve()
    assert estimate.payload_bytes == 24
    assert estimate.available_bytes == 900


def test_selected_parent_alias_is_bound_only_after_exact_config_check() -> None:
    expected = "sha256:" + "a" * 64
    reference = "ghcr.io/example/torch@sha256:" + "b" * 64

    class Runner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def run(self, args, *, check=True, input_text=None):
            del check, input_text
            call = tuple(args)
            self.calls.append(call)
            if call[1:4] == ("image", "inspect", "--format"):
                return CommandResult(call, 0, expected + "\n", "")
            return CommandResult(call, 0, "", "")

    runner = Runner()

    bind_selected_parent(
        reference=reference,
        config_digest=expected,
        runner=runner,
        docker_prefix=("docker",),
    )

    assert ("docker", "tag", reference, actions.STABLE_TORCH_TAG) in runner.calls


def test_selected_parent_alias_rejects_config_drift_before_tagging() -> None:
    expected = "sha256:" + "a" * 64
    reference = "ghcr.io/example/torch@sha256:" + "b" * 64

    class Runner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def run(self, args, *, check=True, input_text=None):
            del check, input_text
            call = tuple(args)
            self.calls.append(call)
            return CommandResult(call, 0, "sha256:" + "c" * 64 + "\n", "")

    runner = Runner()

    with pytest.raises(ActionError, match="config digest"):
        bind_selected_parent(
            reference=reference,
            config_digest=expected,
            runner=runner,
            docker_prefix=("docker",),
        )

    assert not any(call[1:2] == ("tag",) for call in runner.calls)


def test_default_release_registry_pull_uses_empty_auth_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = AnonymousReleaseRegistry(("docker",))
    captured: list[str] = []
    monkeypatch.setattr(
        registry, "authless_pull", lambda reference: captured.append(reference)
    )

    registry.pull("ghcr.io/example/image@sha256:" + "a" * 64)

    assert captured == ["ghcr.io/example/image@sha256:" + "a" * 64]


def test_nonroot_host_plan_is_recreated_by_audited_sudo_helper() -> None:
    plan = PreparePlan(
        supported=True,
        target_user="developer",
        actions=(),
        reboot_required=True,
    )
    digest = stage_input_digest(prepare_plan_payload(plan))
    response = json.dumps(
        {
            "adapter_id": "ubuntu-24.04",
            "plan": prepare_plan_payload(plan),
            "plan_digest": digest,
            "schema_version": 1,
        }
    )
    captured: list[tuple[str, ...]] = []

    def sudo_run(argv, **kwargs):
        del kwargs
        captured.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, response, "")

    result = ProductionInstallerActions(
        effective_uid=1000,
        non_interactive=True,
        sudo_run=sudo_run,
    ).host_plan(target_user="developer")

    assert result.snapshot is None
    assert result.plan == plan
    assert result.plan_digest == digest
    assert captured[0][:3] == ("sudo", "-n", "--")
    assert "amd_ai.installer.privileged" in captured[0]
    assert captured[0][-3:] == ("--target-user", "developer", "plan")


def test_nonroot_host_apply_passes_only_digest_and_never_reboot() -> None:
    plan = PreparePlan(
        supported=True,
        target_user="developer",
        actions=(),
        reboot_required=True,
    )
    digest = stage_input_digest(prepare_plan_payload(plan))
    host_plan = actions.HostPlanResult(
        snapshot=None,
        plan=plan,
        plan_digest=digest,
        adapter_id="ubuntu-24.04",
    )
    captured: list[tuple[str, ...]] = []

    def sudo_run(argv, **kwargs):
        del kwargs
        captured.append(tuple(argv))
        payload = {
            "facts": {
                "backup_path": "/var/backups/amd-ai/fixture",
                "docker_group_added": False,
                "executed_codes": [],
                "reboot_required": True,
                "skipped_codes": ["HOST.REBOOT"],
            },
            "schema_version": 1,
        }
        return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")

    result = ProductionInstallerActions(
        effective_uid=1000,
        non_interactive=True,
        sudo_run=sudo_run,
    ).host_apply(host_plan, include_docker_group=False)

    assert result.facts["reboot_required"] is True
    command = captured[0]
    assert "--expected-plan-digest" in command
    assert digest in command
    assert "reboot" not in command


def test_nonroot_host_verify_uses_privileged_read_only_helper() -> None:
    report = {
        "command": "host-verify",
        "facts": {"kernel": "6.14.0-1018-oem"},
        "findings": [],
        "generated_at": "2026-07-10T12:00:00Z",
        "schema_version": 1,
        "status": "pass",
    }
    captured: list[tuple[str, ...]] = []

    def sudo_run(argv, **kwargs):
        del kwargs
        captured.append(tuple(argv))
        return subprocess.CompletedProcess(
            argv,
            0,
            json.dumps({"report": report, "schema_version": 1}),
            "",
        )

    result = ProductionInstallerActions(
        effective_uid=1000,
        sudo_run=sudo_run,
    ).host_verify(target_user="developer")

    assert result.status.value == "pass"
    assert captured[0][-1] == "verify"
    assert captured[0][captured[0].index("--target-user") + 1] == "developer"
    assert "apply" not in captured[0]
