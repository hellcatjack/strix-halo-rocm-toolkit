from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from amd_ai.doctor.models import (
    DiagnosticDisposition,
    DoctorReport,
    RepairAction,
)
from amd_ai.installer.models import StableRelease
from amd_ai.installer.release import ReleaseError, load_stable_release
from amd_ai.overlay.models import GENERATION_PATTERN


IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
EXACT_REFERENCE_PATTERN = re.compile(
    r"ghcr\.io/[a-z0-9._-]+/[a-z0-9._-]+@sha256:[0-9a-f]{64}"
)


class RepairPlanningError(ValueError):
    pass


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

    if "IMAGE.PARENT_MISSING" in repairable:
        reference = _fact_string(facts, "torch_reference")
        if EXACT_REFERENCE_PATTERN.fullmatch(reference) is None:
            raise RepairPlanningError("parent repair reference is not immutable")
        if release is not None and reference != release.torch.reference:
            raise RepairPlanningError("parent repair reference differs from release")
        actions.append(
            RepairAction("pull-parent", reference, "IMAGE.PARENT_MISSING")
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
