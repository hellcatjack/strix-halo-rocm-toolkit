from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

from amd_ai.doctor import repair
from amd_ai.doctor.models import (
    Diagnostic,
    DiagnosticDisposition,
    DoctorReport,
    RepairAction,
)
import pytest

from amd_ai.doctor.repair import (
    RepairExecutionError,
    SystemRepairExecutor,
    execute_repair,
    plan_repair,
)
from amd_ai.installer.registry import registry_candidates
from amd_ai.installer.release import (
    ReleaseAcquisitionError,
    ReleaseIdentityError,
    VerifiedImageIdentity,
    VerifiedReleaseImages,
    load_stable_release,
)


RELEASE_FIXTURE = Path("tests/fixtures/releases/stable.json")


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


def test_repair_plan_accepts_trusted_swr_exact_parent() -> None:
    report = repairable_project_report()
    release = load_stable_release(RELEASE_FIXTURE)
    swr = registry_candidates(release, "swr")[0].release
    report = replace(
        report,
        facts=MappingProxyType(
            {**report.facts, "torch_reference": swr.torch.reference}
        ),
    )

    plan = plan_repair(report)

    assert plan.actions[0] == RepairAction(
        "pull-parent",
        swr.torch.reference,
        "IMAGE.PARENT_MISSING",
    )


def test_repair_plan_rejects_untrusted_swr_like_parent() -> None:
    report = repairable_project_report()
    report = replace(
        report,
        facts=MappingProxyType(
            {
                **report.facts,
                "torch_reference": (
                    "swr.cn-east-3.myhuaweicloud.com/hellcat-home/"
                    "untrusted@sha256:" + "7" * 64
                ),
            }
        ),
    )

    with pytest.raises(
        repair.RepairPlanningError,
        match="trusted release replica",
    ):
        plan_repair(report)


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


@pytest.mark.parametrize(
    ("reason_code", "expected_kinds"),
    (
        ("IMAGE.DIGEST_DRIFT", ("pull-parent",)),
        (
            "TORCH.BASE_CHANGED",
            ("pull-parent", "remove-project-image", "build-project-image"),
        ),
    ),
)
def test_parent_drift_and_base_change_restore_exact_release_parent(
    reason_code: str, expected_kinds: tuple[str, ...]
) -> None:
    report = DoctorReport.create(
        project="/srv/demo",
        diagnostics=(
            Diagnostic(
                reason_code,
                DiagnosticDisposition.REPAIRABLE,
                "parent changed",
                "identity mismatch",
                "restore",
            ),
        ),
        facts=_facts(),
    )

    plan = plan_repair(report)

    assert tuple(action.kind for action in plan.actions) == expected_kinds
    assert plan.actions[0] == RepairAction(
        "pull-parent",
        "ghcr.io/hellcatjack/strix-halo-rocm-pytorch@sha256:" + "7" * 64,
        reason_code,
    )


def test_system_executor_parent_pull_uses_anonymous_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        repair.Docker,
        "detect",
        lambda: SimpleNamespace(prefix=("sudo", "-n", "docker")),
    )
    executor = SystemRepairExecutor(
        manifest_path=RELEASE_FIXTURE,
        registry="ghcr",
    )
    authless_calls: list[str] = []
    monkeypatch.setattr(
        executor.registry,
        "authless_pull",
        lambda reference: authless_calls.append(reference),
    )
    monkeypatch.setattr(
        executor.registry,
        "_completed",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""),
    )

    def verified_pull(release, *, docker):
        docker.pull(release.base.reference)
        docker.pull(release.torch.reference)
        return VerifiedReleaseImages(
            base=VerifiedImageIdentity(
                reference=release.base.reference,
                config_digest=release.base.config_digest,
                repo_digests=(release.base.reference,),
                labels={},
            ),
            torch=VerifiedImageIdentity(
                reference=release.torch.reference,
                config_digest=release.torch.config_digest,
                repo_digests=(release.torch.reference,),
                labels={},
            ),
        )

    monkeypatch.setattr(
        repair,
        "pull_and_verify_release",
        verified_pull,
    )
    tag_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        executor.registry,
        "tag_reference",
        lambda source, target: tag_calls.append((source, target)),
    )

    executor.pull_and_verify(
        executor.release,
        executor.release.torch.reference,
    )

    assert authless_calls == [
        executor.release.base.reference,
        executor.release.torch.reference,
    ]
    assert tag_calls == [
        (executor.release.base.reference, repair.ROCM_PYTHON_TAG),
        (executor.release.torch.reference, repair.STABLE_TORCH_TAG),
    ]


def test_system_executor_falls_back_to_ghcr_on_swr_acquisition_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        repair.Docker,
        "detect",
        lambda: SimpleNamespace(prefix=("sudo", "-n", "docker")),
    )
    executor = SystemRepairExecutor(
        manifest_path=RELEASE_FIXTURE,
        registry="auto",
    )
    swr = registry_candidates(executor.release, "swr")[0].release
    calls: list[str] = []

    def pull(candidate, *, docker):
        del docker
        calls.append(candidate.base.image)
        if candidate.base.image == swr.base.image:
            raise ReleaseAcquisitionError("SWR unavailable")
        return VerifiedReleaseImages(
            base=VerifiedImageIdentity(
                reference=candidate.base.reference,
                config_digest=candidate.base.config_digest,
                repo_digests=(candidate.base.reference,),
                labels={},
            ),
            torch=VerifiedImageIdentity(
                reference=candidate.torch.reference,
                config_digest=candidate.torch.config_digest,
                repo_digests=(candidate.torch.reference,),
                labels={},
            ),
        )

    monkeypatch.setattr(repair, "pull_and_verify_release", pull)
    monkeypatch.setattr(
        executor.registry,
        "tag_reference",
        lambda source, target: None,
    )

    executor.pull_and_verify(
        executor.release,
        swr.torch.reference,
    )

    assert calls == [swr.base.image, executor.release.base.image]


def test_system_executor_prefers_confirmed_ghcr_reference_in_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        repair.Docker,
        "detect",
        lambda: SimpleNamespace(prefix=("sudo", "-n", "docker")),
    )
    executor = SystemRepairExecutor(
        manifest_path=RELEASE_FIXTURE,
        registry="auto",
    )
    calls: list[str] = []

    def pull(candidate, *, docker):
        del docker
        calls.append(candidate.torch.reference)
        return VerifiedReleaseImages(
            base=VerifiedImageIdentity(
                reference=candidate.base.reference,
                config_digest=candidate.base.config_digest,
                repo_digests=(candidate.base.reference,),
                labels={},
            ),
            torch=VerifiedImageIdentity(
                reference=candidate.torch.reference,
                config_digest=candidate.torch.config_digest,
                repo_digests=(candidate.torch.reference,),
                labels={},
            ),
        )

    monkeypatch.setattr(repair, "pull_and_verify_release", pull)
    monkeypatch.setattr(
        executor.registry,
        "tag_reference",
        lambda source, target: None,
    )

    executor.pull_and_verify(
        executor.release,
        executor.release.torch.reference,
    )

    assert calls == [executor.release.torch.reference]


def test_system_executor_tags_verified_swr_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        repair.Docker,
        "detect",
        lambda: SimpleNamespace(prefix=("sudo", "-n", "docker")),
    )
    executor = SystemRepairExecutor(
        manifest_path=RELEASE_FIXTURE,
        registry="auto",
    )
    swr = registry_candidates(executor.release, "swr")[0].release
    monkeypatch.setattr(
        repair,
        "pull_and_verify_release",
        lambda candidate, *, docker: VerifiedReleaseImages(
            base=VerifiedImageIdentity(
                reference=candidate.base.reference,
                config_digest=candidate.base.config_digest,
                repo_digests=(candidate.base.reference,),
                labels={},
            ),
            torch=VerifiedImageIdentity(
                reference=candidate.torch.reference,
                config_digest=candidate.torch.config_digest,
                repo_digests=(candidate.torch.reference,),
                labels={},
            ),
        ),
    )
    tags: list[tuple[str, str]] = []
    monkeypatch.setattr(
        executor.registry,
        "tag_reference",
        lambda source, target: tags.append((source, target)),
    )

    executor.pull_and_verify(
        executor.release,
        swr.torch.reference,
    )

    assert tags == [
        (swr.base.reference, repair.ROCM_PYTHON_TAG),
        (swr.torch.reference, repair.STABLE_TORCH_TAG),
    ]


def test_system_executor_does_not_hide_swr_identity_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        repair.Docker,
        "detect",
        lambda: SimpleNamespace(prefix=("sudo", "-n", "docker")),
    )
    executor = SystemRepairExecutor(
        manifest_path=RELEASE_FIXTURE,
        registry="auto",
    )
    calls: list[str] = []

    def pull(candidate, *, docker):
        del docker
        calls.append(candidate.base.image)
        raise ReleaseIdentityError("config digest changed")

    monkeypatch.setattr(repair, "pull_and_verify_release", pull)

    with pytest.raises(ReleaseIdentityError, match="config digest"):
        swr = registry_candidates(executor.release, "swr")[0].release
        executor.pull_and_verify(
            executor.release,
            swr.torch.reference,
        )

    assert len(calls) == 1
    assert "myhuaweicloud.com" in calls[0]


def test_system_executor_sanitizes_and_bounds_registry_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        repair.Docker,
        "detect",
        lambda: SimpleNamespace(prefix=("sudo", "-n", "docker")),
    )
    executor = SystemRepairExecutor(
        manifest_path=RELEASE_FIXTURE,
        registry="auto",
    )
    secret = "private-token-value"
    monkeypatch.setattr(
        repair,
        "pull_and_verify_release",
        lambda candidate, *, docker: (_ for _ in ()).throw(
            ReleaseAcquisitionError(
                f"Authorization: Bearer {secret}\n" + "x" * 10_000
            )
        ),
    )
    swr = registry_candidates(executor.release, "swr")[0].release

    with pytest.raises(ReleaseAcquisitionError) as caught:
        executor.pull_and_verify(
            executor.release,
            swr.torch.reference,
        )

    rendered = str(caught.value)
    assert secret not in rendered
    assert "<redacted>" in rendered
    assert "swr:" in rendered
    assert "ghcr:" in rendered
    assert len(rendered.encode("utf-8")) <= 4096


def test_execute_repair_rebuilds_before_removing_exact_project_id() -> None:
    plan = plan_repair(repairable_project_report())
    executor = FakeRepairExecutor()

    execute_repair(plan, executor=executor)

    assert executor.calls == [
        (
            "pull-and-verify",
            plan.release.torch.reference,
            plan.actions[0].exact_target,
        ),
        (
            "build-project",
            plan.project_path,
            plan.release.torch.config_digest,
        ),
        ("remove-image-id", "sha256:" + "8" * 64),
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
            (
                "pull-and-verify",
                plan.release.torch.reference,
                plan.actions[0].exact_target,
            )
        ]
    else:
        assert not any(call[0] == "remove-image-id" for call in executor.calls)
        assert not any(call[0] == "repair-overlay" for call in executor.calls)


def test_repair_does_not_remove_image_when_rebuild_has_same_exact_id() -> None:
    plan = plan_repair(repairable_project_report())
    executor = FakeRepairExecutor(built_image_id="sha256:" + "8" * 64)

    execute_repair(plan, executor=executor)

    assert not any(call[0] == "remove-image-id" for call in executor.calls)


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
    def __init__(
        self,
        failure: str | None = None,
        built_image_id: str = "sha256:" + "9" * 64,
    ) -> None:
        self.failure = failure
        self.built_image_id = built_image_id
        self.calls: list[tuple[object, ...]] = []

    def pull_and_verify(self, release, preferred_reference: str) -> None:
        self.calls.append(
            (
                "pull-and-verify",
                release.torch.reference,
                preferred_reference,
            )
        )
        if self.failure == "pull":
            raise RuntimeError("pull failed")

    def remove_image_id(self, image_id: str) -> None:
        self.calls.append(("remove-image-id", image_id))

    def build_project(self, project_path: Path, parent_digest: str) -> str:
        self.calls.append(("build-project", project_path, parent_digest))
        if self.failure == "build":
            raise RuntimeError("build failed")
        return self.built_image_id

    def repair_overlay(self, project_path: Path, reason_code: str) -> None:
        self.calls.append(("repair-overlay", project_path, reason_code))

    def doctor(self, project_path: Path) -> DoctorReport:
        self.calls.append(("doctor", project_path))
        return DoctorReport.create(project=project_path, diagnostics=(), facts={})
