from __future__ import annotations

import json

from amd_ai.installer import privileged
from amd_ai.report import Report, Status


def test_privileged_verify_collects_target_user_groups(
    monkeypatch, capsys
) -> None:
    captured: list[str] = []

    class FakeActions:
        def __init__(self, *, effective_uid: int) -> None:
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
        def __init__(self, *, effective_uid: int) -> None:
            assert effective_uid == 0

    monkeypatch.setattr(privileged.os, "geteuid", lambda: 0)
    monkeypatch.setattr(privileged, "ProductionInstallerActions", FakeActions)

    code = privileged.main(["verify"])

    assert code == 2
    assert "requires --target-user" in capsys.readouterr().err
