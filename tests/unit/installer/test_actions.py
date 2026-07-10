from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from amd_ai.host.models import PreparePlan
from amd_ai.installer import actions
from amd_ai.installer.actions import (
    ActionError,
    ProductionInstallerActions,
    prepare_plan_payload,
    validate_local_build_source,
)
from amd_ai.installer.release import load_stable_release
from amd_ai.installer.state import stage_input_digest
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

    result = ProductionInstallerActions().host_plan(target_user="developer")

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

    def pull(value, registry):
        captured.update(release=value, registry=registry)
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
