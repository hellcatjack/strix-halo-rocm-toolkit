from amd_ai.report import Finding, Report, Severity, Status


def test_report_serializes_stably():
    report = Report(
        command="host-preflight",
        status=Status.CHANGE_REQUIRED,
        generated_at="2026-07-09T12:00:00Z",
        facts={"kernel": "6.17.0-1025-oem"},
        findings=(
            Finding(
                code="HOST.REBOOT",
                severity=Severity.WARNING,
                summary="Reboot required",
                evidence="OEM kernel installed",
                remediation="Reboot and run host-verify",
            ),
        ),
    )

    payload = report.to_dict()

    assert payload["schema_version"] == 1
    assert payload["status"] == "change-required"
    assert payload["findings"][0]["code"] == "HOST.REBOOT"
