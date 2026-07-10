from __future__ import annotations

import json
from pathlib import Path

from amd_ai import cli
from amd_ai.doctor.models import (
    Diagnostic,
    DiagnosticDisposition,
    DoctorReport,
)


def test_doctor_without_project_checks_platform(monkeypatch, capsys) -> None:
    captured = {}

    def fake_doctor(project, manifest):
        captured.update(project=project, manifest=manifest)
        return repairable_report()

    monkeypatch.setattr(cli, "run_doctor", fake_doctor, raising=False)

    code = cli.main(
        ["doctor", "--manifest", "tests/fixtures/releases/stable.json"]
    )

    assert code == 1
    assert "TORCH.SHADOWED" in capsys.readouterr().out
    assert captured["project"] is None
    assert captured["manifest"] == Path("tests/fixtures/releases/stable.json")


def test_doctor_json_writes_report(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "doctor.json"
    monkeypatch.setattr(
        cli,
        "run_doctor",
        lambda project, manifest: passing_report(),
        raising=False,
    )

    assert cli.main(["doctor", "--json", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"


def test_doctor_blocked_report_returns_two(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "run_doctor",
        lambda project, manifest: DoctorReport.create(
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


def passing_report() -> DoctorReport:
    return DoctorReport.create(project=None, diagnostics=(), facts={})


def repairable_report() -> DoctorReport:
    return DoctorReport.create(
        project="/srv/demo",
        diagnostics=(
            Diagnostic(
                "TORCH.SHADOWED",
                DiagnosticDisposition.REPAIRABLE,
                "shadow",
                "/workspace/torch.py",
                "repair",
            ),
        ),
        facts={"release_id": "0.2.0"},
    )
