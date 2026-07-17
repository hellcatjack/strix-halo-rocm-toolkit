from __future__ import annotations

import json
from pathlib import Path

import pytest

from amd_ai import cli
from amd_ai.doctor.models import (
    Diagnostic,
    DiagnosticDisposition,
    DoctorReport,
)
from amd_ai.installer.registry import RegistryPolicyError


def test_doctor_without_project_checks_platform(monkeypatch, capsys) -> None:
    captured = {}

    def fake_doctor(project, manifest, *, registry):
        captured.update(
            project=project,
            manifest=manifest,
            registry=registry,
        )
        return repairable_report()

    monkeypatch.setattr(cli, "run_doctor", fake_doctor, raising=False)

    code = cli.main(
        ["doctor", "--manifest", "tests/fixtures/releases/stable.json"]
    )

    assert code == 1
    assert "TORCH.SHADOWED" in capsys.readouterr().out
    assert captured["project"] is None
    assert captured["manifest"] == Path("tests/fixtures/releases/stable.json")
    assert captured["registry"] == "auto"


def test_doctor_json_writes_report(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "doctor.json"
    monkeypatch.setattr(
        cli,
        "run_doctor",
        lambda project, manifest, *, registry: passing_report(),
        raising=False,
    )

    assert cli.main(["doctor", "--json", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"


def test_doctor_blocked_report_returns_two(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "run_doctor",
        lambda project, manifest, *, registry: DoctorReport.create(
            project=None,
            diagnostics=(
                Diagnostic(
                    "GPU.RUNTIME_FAILED",
                    DiagnosticDisposition.BLOCKED,
                    "gpu",
                    "failed",
                    "inspect",
                ),
            ),
            facts={},
        ),
        raising=False,
    )

    assert cli.main(["doctor"]) == 2


def test_repair_prints_exact_actions_and_requires_repair_word(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        cli,
        "run_doctor",
        lambda project, manifest, *, registry: repairable_report(),
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "no")

    code = cli.main(["repair", "/srv/demo"])

    assert code == 2
    output = capsys.readouterr().out
    assert "sha256:" in output
    assert (
        "ghcr.io/hellcatjack/strix-halo-rocm-pytorch@sha256:" in output
    )


def test_doctor_forwards_explicit_registry(monkeypatch) -> None:
    captured = {}

    def fake_doctor(project, manifest, *, registry):
        captured.update(
            project=project,
            manifest=manifest,
            registry=registry,
        )
        return passing_report()

    monkeypatch.setattr(cli, "run_doctor", fake_doctor)

    code = cli.main(["doctor", "--registry", "ghcr"])

    assert code == 0
    assert captured["registry"] == "ghcr"


def test_doctor_forwards_explicit_state_path(monkeypatch, tmp_path) -> None:
    captured = {}
    state_path = tmp_path / "custom-state.json"

    def fake_doctor(project, manifest, *, registry, state_path):
        captured["state_path"] = state_path
        return passing_report()

    monkeypatch.setattr(cli, "run_doctor", fake_doctor)

    code = cli.main(
        [
            "doctor",
            "/srv/demo",
            "--state-path",
            str(state_path),
        ]
    )

    assert code == 0
    assert captured["state_path"] == state_path


def test_repair_forwards_registry_to_doctor_and_executor(
    monkeypatch,
) -> None:
    captured = {}

    def fake_doctor(project, manifest, *, registry):
        captured["doctor_registry"] = registry
        return repairable_report()

    class FakeExecutor:
        def __init__(self, *, manifest_path, registry):
            captured["executor_registry"] = registry

    monkeypatch.setattr(cli, "run_doctor", fake_doctor)
    monkeypatch.setattr(cli, "SystemRepairExecutor", FakeExecutor)
    monkeypatch.setattr(
        cli,
        "execute_repair",
        lambda plan, *, executor: passing_report(),
    )

    code = cli.main(
        ["repair", "/srv/demo", "--yes", "--registry", "swr"]
    )

    assert code == 0
    assert captured == {
        "doctor_registry": "swr",
        "executor_registry": "swr",
    }


def test_doctor_registry_policy_error_returns_two(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        cli,
        "run_doctor",
        lambda project, manifest, *, registry: (_ for _ in ()).throw(
            RegistryPolicyError("custom release has no SWR mapping")
        ),
    )

    code = cli.main(["doctor", "--registry", "swr"])

    assert code == 2
    assert "no SWR mapping" in capsys.readouterr().err


def test_repair_registry_error_is_sanitized_and_bounded(
    monkeypatch,
    capsys,
) -> None:
    secret = "private-token-value"
    monkeypatch.setattr(
        cli,
        "run_doctor",
        lambda project, manifest, *, registry: (_ for _ in ()).throw(
            RegistryPolicyError(
                f"Authorization: Bearer {secret}\n" + "x" * 10_000
            )
        ),
    )

    code = cli.main(["repair", "/srv/demo"])

    error = capsys.readouterr().err
    assert code == 2
    assert secret not in error
    assert "Authorization: Bearer <redacted>" in error
    assert len(error.encode("utf-8")) <= 4200


def test_noninteractive_repair_requires_yes() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["repair", "/srv/demo", "--non-interactive"]
        )


def passing_report() -> DoctorReport:
    return DoctorReport.create(project=None, diagnostics=(), facts={})


def repairable_report() -> DoctorReport:
    generation = "/srv/demo/.amd-ai/generations/20260710T120000Z-a1b2c3d4"
    return DoctorReport.create(
        project="/srv/demo",
        diagnostics=(
            Diagnostic(
                "IMAGE.PARENT_MISSING",
                DiagnosticDisposition.REPAIRABLE,
                "missing",
                "parent",
                "pull",
            ),
            Diagnostic(
                "IMAGE.PROJECT_CHANGED",
                DiagnosticDisposition.REPAIRABLE,
                "changed",
                "image",
                "rebuild",
            ),
            Diagnostic(
                "TORCH.SHADOWED",
                DiagnosticDisposition.REPAIRABLE,
                "shadow",
                "/workspace/torch.py",
                "repair",
            ),
        ),
        facts={
            "manifest": str(
                Path("tests/fixtures/releases/stable.json").resolve()
            ),
            "release_id": "0.2.0",
            "torch_reference": (
                "ghcr.io/hellcatjack/strix-halo-rocm-pytorch@sha256:"
                + "7" * 64
            ),
            "project_config": "/srv/demo/amd-ai-project.toml",
            "project_image_id": "sha256:" + "8" * 64,
            "current_generation": generation,
            "last_valid_lock": generation + "/overlay.requirements.lock",
        },
    )
