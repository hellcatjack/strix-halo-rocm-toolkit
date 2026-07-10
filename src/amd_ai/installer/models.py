from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
PROJECT_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,62}")
USER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,31}")
STATE_SCHEMA_VERSION = 1


class InstallerModelError(ValueError):
    pass


class InstallMode(StrEnum):
    FULL = "full"
    CONTAINER = "container"
    DOCTOR = "doctor"


class InstallStage(StrEnum):
    BOOTSTRAP = "BOOTSTRAP"
    HOST_PREFLIGHT = "HOST_PREFLIGHT"
    HOST_PLAN = "HOST_PLAN"
    HOST_CONFIRM = "HOST_CONFIRM"
    HOST_APPLY = "HOST_APPLY"
    REBOOT_PENDING = "REBOOT_PENDING"
    HOST_VERIFY = "HOST_VERIFY"
    CONTAINER_HOST_CHECK = "CONTAINER_HOST_CHECK"
    RELEASE_RESOLVE = "RELEASE_RESOLVE"
    IMAGE_PULL_OR_BUILD = "IMAGE_PULL_OR_BUILD"
    IMAGE_VERIFY = "IMAGE_VERIFY"
    PROJECT_INIT = "PROJECT_INIT"
    PROJECT_VERIFY = "PROJECT_VERIFY"
    COMPLETE = "COMPLETE"


FULL_STAGE_ORDER = (
    InstallStage.BOOTSTRAP,
    InstallStage.HOST_PREFLIGHT,
    InstallStage.HOST_PLAN,
    InstallStage.HOST_CONFIRM,
    InstallStage.HOST_APPLY,
    InstallStage.REBOOT_PENDING,
    InstallStage.HOST_VERIFY,
    InstallStage.RELEASE_RESOLVE,
    InstallStage.IMAGE_PULL_OR_BUILD,
    InstallStage.IMAGE_VERIFY,
    InstallStage.PROJECT_INIT,
    InstallStage.PROJECT_VERIFY,
    InstallStage.COMPLETE,
)

CONTAINER_STAGE_ORDER = (
    InstallStage.BOOTSTRAP,
    InstallStage.CONTAINER_HOST_CHECK,
    InstallStage.RELEASE_RESOLVE,
    InstallStage.IMAGE_PULL_OR_BUILD,
    InstallStage.IMAGE_VERIFY,
    InstallStage.PROJECT_INIT,
    InstallStage.PROJECT_VERIFY,
    InstallStage.COMPLETE,
)


def default_state_path() -> Path:
    return (
        Path.home()
        / ".local/state/strix-halo-rocm-toolkit/install-state.json"
    ).resolve(strict=False)


@dataclass(frozen=True)
class InstallOptions:
    mode: InstallMode
    non_interactive: bool = False
    dry_run: bool = False
    project_dir: Path | None = None
    project_name: str = "amd-ai-project"
    image_source: str | None = None
    target_user: str | None = None
    accepted_host_plan_digest: str | None = None
    accept_docker_group: bool = False
    stable_manifest_path: Path | None = None
    source_root: Path | None = None
    state_path: Path = field(default_factory=default_state_path)

    def __post_init__(self) -> None:
        try:
            mode = InstallMode(self.mode)
        except (TypeError, ValueError) as error:
            raise InstallerModelError("install mode is invalid") from error
        object.__setattr__(self, "mode", mode)

        root = _absolute_path(self.source_root or Path.cwd())
        object.__setattr__(self, "source_root", root)
        object.__setattr__(
            self,
            "stable_manifest_path",
            _absolute_path(
                self.stable_manifest_path
                or root / "profiles/releases/stable.json"
            ),
        )
        object.__setattr__(self, "state_path", _absolute_path(self.state_path))
        if self.project_dir is not None:
            object.__setattr__(
                self, "project_dir", _absolute_path(self.project_dir)
            )

    @property
    def manifest_path(self) -> Path:
        """Compatibility name used by the stable release API."""
        assert self.stable_manifest_path is not None
        return self.stable_manifest_path

    def validate(self) -> InstallOptions:
        if sys.version_info[:2] != (3, 12):
            raise InstallerModelError("the installer requires Python 3.12")
        if type(self.non_interactive) is not bool:
            raise InstallerModelError("non_interactive must be a boolean")
        if type(self.dry_run) is not bool:
            raise InstallerModelError("dry_run must be a boolean")
        if type(self.accept_docker_group) is not bool:
            raise InstallerModelError("accept_docker_group must be a boolean")
        if PROJECT_NAME_PATTERN.fullmatch(self.project_name) is None:
            raise InstallerModelError("project name is invalid")
        if self.image_source not in (None, "pull", "build"):
            raise InstallerModelError("image source must be pull or build")
        if self.target_user is not None and (
            USER_PATTERN.fullmatch(self.target_user) is None
        ):
            raise InstallerModelError("target user is invalid")
        if self.accepted_host_plan_digest is not None and (
            SHA256_PATTERN.fullmatch(self.accepted_host_plan_digest) is None
        ):
            raise InstallerModelError("accepted host plan digest is invalid")

        for name in ("source_root", "stable_manifest_path", "state_path"):
            _require_normalized_absolute_path(name, getattr(self, name))
        if self.project_dir is not None:
            _require_normalized_absolute_path("project_dir", self.project_dir)

        if self.non_interactive:
            if self.mode in (InstallMode.FULL, InstallMode.CONTAINER):
                if self.project_dir is None:
                    raise InstallerModelError(
                        "non-interactive install requires a project directory"
                    )
                if self.image_source is None:
                    raise InstallerModelError(
                        "non-interactive install requires an image source"
                    )
            if (
                self.mode is InstallMode.FULL
                and self.accepted_host_plan_digest is None
            ):
                raise InstallerModelError(
                    "non-interactive full install requires a host plan digest"
                )
        return self


@dataclass(frozen=True)
class InstallState:
    schema_version: int
    installer_version: str
    mode: InstallMode
    target_user: str | None
    release_id: str | None
    source_revision: str | None
    base_image_reference: str | None
    base_manifest_digest: str | None
    torch_image_reference: str | None
    torch_manifest_digest: str | None
    project_path: str | None
    current_stage: InstallStage
    completed_stage_input_digests: Mapping[str, str]
    reboot_boot_id: str | None
    created_at: str
    updated_at: str
    installer_source_revision: str
    source_root: str
    host_plan_digest: str | None = None
    last_report_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or (
            self.schema_version != STATE_SCHEMA_VERSION
        ):
            raise InstallerModelError("install state schema version is invalid")
        try:
            mode = InstallMode(self.mode)
            current_stage = InstallStage(self.current_stage)
        except (TypeError, ValueError) as error:
            raise InstallerModelError("install state mode or stage is invalid") from error
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "current_stage", current_stage)

        if not isinstance(self.installer_version, str) or not self.installer_version:
            raise InstallerModelError("installer version is invalid")
        if self.target_user is not None and (
            USER_PATTERN.fullmatch(self.target_user) is None
        ):
            raise InstallerModelError("install state target user is invalid")
        _require_optional_revision("source_revision", self.source_revision)
        _require_optional_revision(
            "installer_source_revision", self.installer_source_revision
        )
        _require_optional_sha256(
            "base_manifest_digest", self.base_manifest_digest, prefixed=True
        )
        _require_optional_sha256(
            "torch_manifest_digest", self.torch_manifest_digest, prefixed=True
        )
        _require_optional_sha256("host_plan_digest", self.host_plan_digest)

        source_root = _absolute_path(Path(self.source_root))
        if str(source_root) != self.source_root:
            raise InstallerModelError("install state source_root is not normalized")
        if self.project_path is not None:
            project_path = _absolute_path(Path(self.project_path))
            if str(project_path) != self.project_path:
                raise InstallerModelError(
                    "install state project_path is not normalized"
                )

        completed: dict[str, str] = {}
        for raw_stage, digest in self.completed_stage_input_digests.items():
            try:
                stage = InstallStage(raw_stage)
            except (TypeError, ValueError) as error:
                raise InstallerModelError(
                    f"unknown completed install stage: {raw_stage}"
                ) from error
            if SHA256_PATTERN.fullmatch(digest) is None:
                raise InstallerModelError(
                    f"completed stage digest is invalid: {stage.value}"
                )
            completed[stage.value] = digest
        object.__setattr__(
            self,
            "completed_stage_input_digests",
            MappingProxyType(completed),
        )

        reports: list[str] = []
        for report in self.last_report_paths:
            path = _absolute_path(Path(report))
            if str(path) != report:
                raise InstallerModelError(
                    "install state report path is not normalized"
                )
            reports.append(report)
        object.__setattr__(self, "last_report_paths", tuple(reports))

        for name in ("created_at", "updated_at"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise InstallerModelError(f"install state {name} is invalid")


@dataclass(frozen=True)
class StageResult:
    facts: Mapping[str, object] = field(default_factory=dict)
    report_paths: tuple[str, ...] = ()
    action_required: bool = False
    blocked: bool = False
    message: str = ""

    def __post_init__(self) -> None:
        if self.action_required and self.blocked:
            raise InstallerModelError(
                "a stage result cannot be action-required and blocked"
            )
        object.__setattr__(self, "facts", MappingProxyType(dict(self.facts)))
        object.__setattr__(self, "report_paths", tuple(self.report_paths))


@dataclass(frozen=True)
class DiskSpaceEstimate:
    location: Path
    payload_bytes: int
    available_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "location", _absolute_path(self.location))
        for name in ("payload_bytes", "available_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InstallerModelError(
                    f"disk estimate {name} must be a nonnegative integer"
                )


def _absolute_path(path: Path) -> Path:
    try:
        return path.expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise InstallerModelError(f"cannot normalize path: {path}") from error


def _require_normalized_absolute_path(name: str, value: object) -> None:
    if not isinstance(value, Path) or not value.is_absolute():
        raise InstallerModelError(f"{name} must be an absolute path")
    if _absolute_path(value) != value:
        raise InstallerModelError(f"{name} must be normalized")


def _require_optional_revision(name: str, value: str | None) -> None:
    if value is not None and REVISION_PATTERN.fullmatch(value) is None:
        raise InstallerModelError(f"{name} is invalid")


def _require_optional_sha256(
    name: str, value: str | None, *, prefixed: bool = False
) -> None:
    if value is None:
        return
    candidate = value.removeprefix("sha256:") if prefixed else value
    if SHA256_PATTERN.fullmatch(candidate) is None or (
        prefixed and not value.startswith("sha256:")
    ):
        raise InstallerModelError(f"{name} is invalid")


@dataclass(frozen=True)
class ReleaseImage:
    image: str
    manifest_digest: str
    config_digest: str
    artifact_digests: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_digests",
            MappingProxyType(dict(self.artifact_digests)),
        )

    @property
    def reference(self) -> str:
        return f"{self.image}@{self.manifest_digest}"


@dataclass(frozen=True)
class StableRelease:
    schema_version: int
    release_id: str
    source_repository: str
    source_revision: str
    qualification_profile_digest: str
    qualification_report_digest: str
    sbom_digest: str
    gpu_arch: str
    supported_host_adapter_ids: tuple[str, ...]
    rocm_version: str
    python_version: str
    torch_version: str
    torch_profile_id: str
    torch_profile_digest: str
    base: ReleaseImage
    torch: ReleaseImage
    published_at: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "supported_host_adapter_ids",
            tuple(self.supported_host_adapter_ids),
        )
