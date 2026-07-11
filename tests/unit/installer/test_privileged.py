from __future__ import annotations

import json

import pytest

from amd_ai.host.models import PreparePlan
from amd_ai.installer import privileged
from amd_ai.installer.actions import HostPlanResult, prepare_plan_payload
from amd_ai.installer.state import stage_input_digest
from amd_ai.installer.models import StageResult
from amd_ai.report import Report, Status
from amd_ai.runner import CommandStream


def test_privileged_verify_collects_target_user_groups(
    monkeypatch, capsys
) -> None:
    captured: list[str] = []

    class FakeActions:
        def __init__(self, *, effective_uid: int, **kwargs) -> None:
            del kwargs
            assert effective_uid == 0

        def host_verify(self, *, target_user: str) -> Report:
            captured.append(target_user)
            return Report(
                command="host-verify",
                status=Status.UNVERIFIED,
                generated_at="2026-07-10T20:28:14Z",
                facts={"kernel": "6.17.0-1028-oem"},
                findings=(),
            )

    monkeypatch.setattr(privileged.os, "geteuid", lambda: 0)
    monkeypatch.setattr(privileged, "ProductionInstallerActions", FakeActions)

    code = privileged.main(["--target-user", "developer", "verify"])

    assert code == 0
    assert captured == ["developer"]
    payload = json.loads(capsys.readouterr().out)
    assert payload["report"]["status"] == "unverified"


def test_privileged_verify_requires_target_user(monkeypatch, capsys) -> None:
    class FakeActions:
        def __init__(self, *, effective_uid: int, **kwargs) -> None:
            del kwargs
            assert effective_uid == 0

    monkeypatch.setattr(privileged.os, "geteuid", lambda: 0)
    monkeypatch.setattr(privileged, "ProductionInstallerActions", FakeActions)

    code = privileged.main(["verify"])

    assert code == 2
    assert "requires --target-user" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("progress_args", "progress_visible"),
    [
        (["--progress-mode", "default"], True),
        ([], False),
    ],
)
def test_privileged_apply_keeps_json_on_stdout_and_progress_on_stderr(
    monkeypatch,
    capsys,
    progress_args: list[str],
    progress_visible: bool,
) -> None:
    plan = PreparePlan(
        supported=True,
        target_user="developer",
        actions=(),
        reboot_required=False,
    )
    digest = stage_input_digest(prepare_plan_payload(plan))

    class FakeActions:
        def __init__(
            self,
            *,
            effective_uid: int,
            command_observer,
            **kwargs,
        ) -> None:
            del kwargs
            assert effective_uid == 0
            self.command_observer = command_observer

        def host_plan(self, *, target_user: str, memory_gib=None):
            del memory_gib
            assert target_user == "developer"
            return HostPlanResult(None, plan, digest, "ubuntu-24.04")

        def host_apply(self, host_plan, *, include_docker_group: bool):
            del host_plan, include_docker_group
            self.command_observer.command_output(
                CommandStream.STDOUT, "root-apply-progress\n"
            )
            return StageResult(
                facts={
                    "backup_path": "/var/backups/amd-ai/fixture",
                    "docker_group_added": False,
                    "executed_codes": [],
                    "reboot_required": False,
                    "skipped_codes": [],
                }
            )

    monkeypatch.setattr(privileged.os, "geteuid", lambda: 0)
    monkeypatch.setattr(privileged, "ProductionInstallerActions", FakeActions)
    argv = [
        *progress_args,
        "--target-user",
        "developer",
        "--expected-plan-digest",
        digest,
        "apply",
    ]

    code = privileged.main(argv)
    captured = capsys.readouterr()

    assert code == 0
    assert captured.out.count("\n") == 1
    assert json.loads(captured.out)["schema_version"] == 1
    assert ("root-apply-progress" in captured.err) is progress_visible
    assert "root-apply-progress" not in captured.out
