from __future__ import annotations

from pathlib import Path

from amd_ai.doctor.models import (
    Diagnostic,
    DiagnosticDisposition,
    DoctorReport,
    RepairAction,
)
import pytest

from amd_ai.doctor.repair import (
    RepairExecutionError,
    execute_repair,
    plan_repair,
)


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


def test_execute_repair_pulls_parent_removes_exact_project_id_and_rebuilds() -> None:
    plan = plan_repair(repairable_project_report())
    executor = FakeRepairExecutor()

    execute_repair(plan, executor=executor)

    assert executor.calls == [
        ("pull-and-verify", plan.release.torch.reference),
        ("remove-image-id", "sha256:" + "8" * 64),
        (
            "build-project",
            plan.project_path,
            plan.release.torch.config_digest,
        ),
        ("repair-overlay", plan.project_path, "TORCH.SHADOWED"),
        ("doctor", plan.project_path),
    ]


@pytest.mark.parametrize("failure", ("pull", "build"))
def test_execute_repair_stops_before_dependent_actions(failure: str) -> None:
    plan = plan_repair(repairable_project_report())
    executor = FakeRepairExecutor(failure=failure)

    with pytest.raises(RepairExecutionError):
        execute_repair(plan, executor=executor)

    if failure == "pull":
        assert executor.calls == [
            ("pull-and-verify", plan.release.torch.reference)
        ]
    else:
        assert not any(call[0] == "repair-overlay" for call in executor.calls)


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


class FakeRepairExecutor:
    def __init__(self, failure: str | None = None) -> None:
        self.failure = failure
        self.calls: list[tuple[object, ...]] = []

    def pull_and_verify(self, release) -> None:
        self.calls.append(("pull-and-verify", release.torch.reference))
        if self.failure == "pull":
            raise RuntimeError("pull failed")

    def remove_image_id(self, image_id: str) -> None:
        self.calls.append(("remove-image-id", image_id))

    def build_project(self, project_path: Path, parent_digest: str) -> None:
        self.calls.append(("build-project", project_path, parent_digest))
        if self.failure == "build":
            raise RuntimeError("build failed")

    def repair_overlay(self, project_path: Path, reason_code: str) -> None:
        self.calls.append(("repair-overlay", project_path, reason_code))

    def doctor(self, project_path: Path) -> DoctorReport:
        self.calls.append(("doctor", project_path))
        return DoctorReport.create(project=project_path, diagnostics=(), facts={})
