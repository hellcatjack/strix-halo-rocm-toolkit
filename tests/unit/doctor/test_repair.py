from __future__ import annotations

from pathlib import Path

from amd_ai.doctor.models import (
    Diagnostic,
    DiagnosticDisposition,
    DoctorReport,
    RepairAction,
)
from amd_ai.doctor.repair import plan_repair


def test_repair_plan_uses_exact_generation_image_id_and_registry_digest() -> None:
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
            (
                "/srv/demo/.amd-ai/generations/20260710T120000Z-a1b2c3d4/"
                "overlay.requirements.lock"
            ),
            "TORCH.SHADOWED",
        ),
        RepairAction(
            "verify-project",
            "/srv/demo/amd-ai-project.toml",
            "TORCH.SHADOWED",
        ),
    )
    assert not plan.blocked
    assert not any("*" in action.exact_target for action in plan.actions)


def test_blocked_report_produces_no_destructive_actions() -> None:
    report = DoctorReport.create(
        project="/srv/demo",
        diagnostics=(
            Diagnostic(
                "GPU.RUNTIME_FAILED",
                DiagnosticDisposition.BLOCKED,
                "gpu failed",
                "reset",
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
        facts=_facts(),
    )

    plan = plan_repair(report)

    assert plan.blocked
    assert plan.actions == ()
    assert plan.blocked_reasons == ("GPU.RUNTIME_FAILED",)


def repairable_project_report() -> DoctorReport:
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
                "build",
            ),
            Diagnostic(
                "TORCH.SHADOWED",
                DiagnosticDisposition.REPAIRABLE,
                "shadow",
                "path",
                "repair",
            ),
        ),
        facts=_facts(),
    )


def _facts() -> dict[str, object]:
    generation = "/srv/demo/.amd-ai/generations/20260710T120000Z-a1b2c3d4"
    return {
        "manifest": str(Path("tests/fixtures/releases/stable.json").resolve()),
        "torch_reference": (
            "ghcr.io/hellcatjack/strix-halo-rocm-pytorch@sha256:" + "7" * 64
        ),
        "torch_config_digest": "sha256:" + "6" * 64,
        "project_config": "/srv/demo/amd-ai-project.toml",
        "project_image_id": "sha256:" + "8" * 64,
        "current_generation": generation,
        "last_valid_lock": generation + "/overlay.requirements.lock",
    }
