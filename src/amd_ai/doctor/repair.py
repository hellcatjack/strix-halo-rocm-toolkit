from __future__ import annotations

import re
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from amd_ai.doctor.models import (
    DiagnosticDisposition,
    DoctorReport,
    RepairAction,
)
from amd_ai.installer.models import StableRelease
from amd_ai.installer.actions import AnonymousReleaseRegistry
from amd_ai.installer.release import (
    ReleaseError,
    load_stable_release,
    pull_and_verify_release,
)
from amd_ai.image.build import Docker, ROCM_PYTHON_TAG, STABLE_TORCH_TAG
from amd_ai.overlay.models import GENERATION_PATTERN
from amd_ai.project.build import (
    build_or_reuse_project,
    remove_exact_project_image,
)
from amd_ai.project.config import load_project_config
from amd_ai.runner import SubprocessRunner


IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
EXACT_REFERENCE_PATTERN = re.compile(
    r"ghcr\.io/[a-z0-9._-]+/[a-z0-9._-]+@sha256:[0-9a-f]{64}"
)


class RepairPlanningError(ValueError):
    pass


class RepairExecutionError(RuntimeError):
    pass


class RepairExecutor(Protocol):
    def pull_and_verify(self, release: StableRelease) -> None:
        pass

    def remove_image_id(self, image_id: str) -> None:
        pass

    def build_project(self, project_path: Path, parent_digest: str) -> str:
        pass

    def repair_overlay(self, project_path: Path, reason_code: str) -> None:
        pass

    def doctor(self, project_path: Path) -> DoctorReport:
        pass


class SystemRepairExecutor:
    def __init__(self, *, manifest_path: Path) -> None:
        self.manifest_path = manifest_path
        self.release = load_stable_release(manifest_path)
        self.docker = Docker.detect()
        self.registry = AnonymousReleaseRegistry(self.docker.prefix)
        self.runner = SubprocessRunner()

    def pull_and_verify(self, release: StableRelease) -> None:
        if release != self.release:
            raise RepairExecutionError("repair release differs from executor manifest")
        pull_and_verify_release(release, docker=self.registry)
        self.registry.tag_reference(release.base.reference, ROCM_PYTHON_TAG)
        self.registry.tag_reference(release.torch.reference, STABLE_TORCH_TAG)

    def remove_image_id(self, image_id: str) -> None:
        remove_exact_project_image(
            image_id,
            runner=self.runner,
            docker_prefix=self.docker.prefix,
        )

    def build_project(self, project_path: Path, parent_digest: str) -> str:
        config = load_project_config(project_path / "amd-ai-project.toml")
        if config.base_digest != parent_digest:
            raise RepairExecutionError(
                "project config parent digest differs from verified release"
            )
        result = build_or_reuse_project(
            config=config,
            runner=self.runner,
            force=True,
            no_build=False,
            docker_prefix=self.docker.prefix,
        )
        return result.image_id

    def repair_overlay(self, project_path: Path, reason_code: str) -> None:
        config = load_project_config(project_path / "amd-ai-project.toml")
        args = [
            *self.docker.prefix,
            "run",
            "--rm",
            "--read-only",
            "--network",
            "none",
            "--ipc=private",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=1g,mode=1777",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--workdir",
            "/workspace",
            "--entrypoint",
            "/opt/venv/bin/python",
            "--env",
            "PYTHONNOUSERSITE=1",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            f"AMD_AI_PARENT_CONFIG_DIGEST={config.base_digest}",
            "--env",
            "PYTHONPATH=/opt/amd-ai/src",
            "--mount",
            f"type=bind,src={project_path},dst=/workspace",
            config.image,
            "-m",
            "amd_ai.overlay.repair_command",
            reason_code,
        ]
        result = self.runner.run(args, check=False)
        if result.returncode != 0:
            evidence = result.stderr.strip() or result.stdout.strip() or "no output"
            raise RepairExecutionError(f"offline overlay repair failed: {evidence}")

    def doctor(self, project_path: Path) -> DoctorReport:
        from amd_ai.doctor.checks import run_doctor

        return run_doctor(project_path, self.manifest_path)


@dataclass(frozen=True)
class RepairPlan:
    report: DoctorReport
    release: StableRelease | None
    project_path: Path | None
    actions: tuple[RepairAction, ...]
    blocked: bool
    blocked_reasons: tuple[str, ...]


def plan_repair(report: DoctorReport) -> RepairPlan:
    blocked_reasons = tuple(
        sorted(
            {
                diagnostic.code
                for diagnostic in report.diagnostics
                if diagnostic.disposition == DiagnosticDisposition.BLOCKED
            }
        )
    )
    release = _load_report_release(report)
    project_path = (
        None if report.project is None else _normalized_absolute(report.project)
    )
    if blocked_reasons:
        return RepairPlan(
            report,
            release,
            project_path,
            (),
            True,
            blocked_reasons,
        )

    repairable = {
        diagnostic.code
        for diagnostic in report.diagnostics
        if diagnostic.disposition == DiagnosticDisposition.REPAIRABLE
    }
    actions: list[RepairAction] = []
    facts = report.facts

    parent_reason = next(
        (
            code
            for code in (
                "IMAGE.PARENT_MISSING",
                "IMAGE.DIGEST_DRIFT",
                "TORCH.BASE_CHANGED",
            )
            if code in repairable
        ),
        None,
    )
    if parent_reason is not None:
        reference = _fact_string(facts, "torch_reference")
        if EXACT_REFERENCE_PATTERN.fullmatch(reference) is None:
            raise RepairPlanningError("parent repair reference is not immutable")
        if release is not None and reference != release.torch.reference:
            raise RepairPlanningError("parent repair reference differs from release")
        actions.append(
            RepairAction("pull-parent", reference, parent_reason)
        )

    project_reason = next(
        (
            code
            for code in ("IMAGE.PROJECT_CHANGED", "TORCH.BASE_CHANGED")
            if code in repairable
        ),
        None,
    )
    if project_reason is not None:
        config_path = _project_config_path(facts, project_path)
        image_id = facts.get("project_image_id")
        if image_id is not None:
            if not isinstance(image_id, str) or IMAGE_ID_PATTERN.fullmatch(
                image_id
            ) is None:
                raise RepairPlanningError("project image repair target is not exact")
            actions.append(
                RepairAction("remove-project-image", image_id, project_reason)
            )
        actions.append(
            RepairAction("build-project-image", str(config_path), project_reason)
        )

    overlay_reason = next(
        (
            code
            for code in ("TORCH.SHADOWED", "OVERLAY.LOCK_INVALID")
            if code in repairable
        ),
        None,
    )
    if overlay_reason is not None:
        if project_path is None:
            raise RepairPlanningError("overlay repair requires a selected project")
        generation = _generation_path(
            _fact_string(facts, "current_generation"), project_path
        )
        lock_path = _normalized_absolute(
            _fact_string(facts, "last_valid_lock")
        )
        expected_lock = generation / "overlay.requirements.lock"
        if lock_path != expected_lock:
            raise RepairPlanningError(
                "overlay replay lock is not from the selected generation"
            )
        config_path = _project_config_path(facts, project_path)
        actions.extend(
            (
                RepairAction(
                    "quarantine-overlay", str(generation), overlay_reason
                ),
                RepairAction("rebuild-overlay", str(lock_path), overlay_reason),
                RepairAction("verify-project", str(config_path), overlay_reason),
            )
        )

    unique: list[RepairAction] = []
    seen: set[tuple[str, str]] = set()
    for action in actions:
        identity = (action.kind, action.exact_target)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(action)
    return RepairPlan(
        report,
        release,
        project_path,
        tuple(unique),
        False,
        (),
    )


def execute_repair(
    plan: RepairPlan, *, executor: RepairExecutor
) -> DoctorReport:
    if plan.blocked:
        raise RepairExecutionError(
            "repair plan is blocked: " + ", ".join(plan.blocked_reasons)
        )
    if not plan.actions:
        raise RepairExecutionError("repair plan contains no actions")
    if plan.release is None:
        raise RepairExecutionError("repair plan has no verified release")
    if plan.project_path is None:
        raise RepairExecutionError("repair plan has no selected project")

    actions = {action.kind: action for action in plan.actions}
    try:
        parent = actions.get("pull-parent")
        if parent is not None:
            if parent.exact_target != plan.release.torch.reference:
                raise RepairExecutionError(
                    "parent pull action differs from verified release"
                )
            executor.pull_and_verify(plan.release)

        removal = actions.get("remove-project-image")
        build = actions.get("build-project-image")
        rebuilt_image_id: str | None = None
        if build is not None:
            if Path(build.exact_target) != plan.project_path / "amd-ai-project.toml":
                raise RepairExecutionError("project build target escaped project")
            rebuilt_image_id = executor.build_project(
                plan.project_path,
                plan.release.torch.config_digest,
            )
            if IMAGE_ID_PATTERN.fullmatch(rebuilt_image_id) is None:
                raise RepairExecutionError("rebuilt project image ID is not exact")

        if removal is not None:
            if IMAGE_ID_PATTERN.fullmatch(removal.exact_target) is None:
                raise RepairExecutionError("project image removal is not exact")
            if rebuilt_image_id is None:
                raise RepairExecutionError(
                    "project image removal requires a successful rebuild"
                )
            if removal.exact_target != rebuilt_image_id:
                executor.remove_image_id(removal.exact_target)

        quarantine = actions.get("quarantine-overlay")
        replay = actions.get("rebuild-overlay")
        if (quarantine is None) != (replay is None):
            raise RepairExecutionError(
                "overlay quarantine and replay actions must be paired"
            )
        if quarantine is not None and replay is not None:
            generation = Path(quarantine.exact_target)
            if Path(replay.exact_target) != generation / "overlay.requirements.lock":
                raise RepairExecutionError("overlay replay lock differs from quarantine")
            executor.repair_overlay(plan.project_path, quarantine.reason_code)

        report = executor.doctor(plan.project_path)
    except RepairExecutionError:
        raise
    except Exception as error:
        raise RepairExecutionError(f"repair execution failed: {error}") from error
    if report.status != "pass" or any(
        diagnostic.disposition != DiagnosticDisposition.PASS
        for diagnostic in report.diagnostics
    ):
        raise RepairExecutionError(
            f"post-repair doctor did not pass: {report.status}"
        )
    return report


def _load_report_release(report: DoctorReport) -> StableRelease | None:
    value = report.facts.get("manifest")
    if not isinstance(value, str) or not value:
        return None
    try:
        return load_stable_release(Path(value))
    except (OSError, ReleaseError) as error:
        if any(
            diagnostic.code == "RELEASE.INVALID"
            for diagnostic in report.diagnostics
        ):
            return None
        raise RepairPlanningError(
            f"cannot reload doctor release manifest: {error}"
        ) from error


def _project_config_path(
    facts: object, project_path: Path | None
) -> Path:
    if not hasattr(facts, "get"):
        raise RepairPlanningError("doctor facts are invalid")
    path = _normalized_absolute(_fact_string(facts, "project_config"))
    if project_path is None or path != project_path / "amd-ai-project.toml":
        raise RepairPlanningError("project config repair target escaped project")
    return path


def _generation_path(value: str, project_path: Path) -> Path:
    generation = _normalized_absolute(value)
    expected_parent = project_path / ".amd-ai" / "generations"
    if (
        generation.parent != expected_parent
        or GENERATION_PATTERN.fullmatch(generation.name) is None
    ):
        raise RepairPlanningError("overlay generation target escaped project")
    return generation


def _normalized_absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or "\0" in value or str(path) != value:
        raise RepairPlanningError(f"repair target is not normalized: {value}")
    return path


def _fact_string(facts: object, name: str) -> str:
    if not hasattr(facts, "get"):
        raise RepairPlanningError("doctor facts are invalid")
    value = facts.get(name)
    if not isinstance(value, str) or not value:
        raise RepairPlanningError(f"doctor fact is missing: {name}")
    return value
