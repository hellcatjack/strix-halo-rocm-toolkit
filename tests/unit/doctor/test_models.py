from __future__ import annotations

import pytest

from amd_ai.doctor.models import (
    Diagnostic,
    DiagnosticDisposition,
    DoctorModelError,
    DoctorReport,
    RepairAction,
)


def test_report_status_uses_highest_disposition_and_serializes_exactly() -> None:
    report = DoctorReport.create(
        project="/workspace/demo",
        diagnostics=(
            Diagnostic(
                "OVERLAY.TRANSACTION_INCOMPLETE",
                DiagnosticDisposition.WARNING,
                "stale",
                "tx",
                "inspect",
            ),
            Diagnostic(
                "TORCH.SHADOWED",
                DiagnosticDisposition.REPAIRABLE,
                "shadow",
                "path",
                "repair",
            ),
        ),
        facts={"release_id": "0.2.0"},
    )

    assert report.status == "repairable"
    assert report.to_dict()["diagnostics"][1]["code"] == "TORCH.SHADOWED"
    assert report.project == "/workspace/demo"


def test_report_redacts_url_userinfo_and_secret_environment_values() -> None:
    report = DoctorReport.create(
        project=None,
        diagnostics=(
            Diagnostic(
                "RELEASE.INVALID",
                DiagnosticDisposition.BLOCKED,
                "invalid",
                "https://person:password@example.test/path token-value",
                "fix manifest",
            ),
        ),
        facts={},
        environment={"API_TOKEN": "token-value", "SAFE": "shown"},
    )

    evidence = report.diagnostics[0].evidence
    assert "password" not in evidence
    assert "token-value" not in evidence
    assert "https://<redacted>@example.test/path" in evidence


def test_unknown_diagnostic_code_is_rejected() -> None:
    with pytest.raises(DoctorModelError, match="code"):
        Diagnostic(
            "UNKNOWN.FAILURE",
            DiagnosticDisposition.WARNING,
            "unknown",
            "none",
            "none",
        )


def test_repair_action_rejects_wildcards_or_unknown_kind() -> None:
    with pytest.raises(DoctorModelError):
        RepairAction("prune", "*", "TORCH.SHADOWED")
