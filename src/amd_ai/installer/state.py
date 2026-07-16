from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import tempfile
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from amd_ai.installer.models import (
    InstallStage,
    InstallState,
    InstallerModelError,
    STATE_SCHEMA_VERSION,
)


BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
STATE_KEYS_V1 = frozenset(
    {
        "schema_version",
        "installer_version",
        "mode",
        "target_user",
        "release_id",
        "source_revision",
        "base_image_reference",
        "base_manifest_digest",
        "torch_image_reference",
        "torch_manifest_digest",
        "project_path",
        "current_stage",
        "completed_stage_input_digests",
        "reboot_boot_id",
        "created_at",
        "updated_at",
        "installer_source_revision",
        "source_root",
        "host_plan_digest",
        "host_adapter_id",
        "docker_group_accepted",
        "base_config_digest",
        "torch_config_digest",
        "last_report_paths",
    }
)
STATE_KEYS_V2 = STATE_KEYS_V1 | frozenset(
    {
        "host_verification_status",
        "host_kernel",
        "host_verification_findings",
    }
)
STATE_KEYS = STATE_KEYS_V2 | frozenset(
    {
        "kernel_plan_digest",
        "kernel_reboot_boot_id",
        "recovery_kernel",
        "display_manager_was_active",
        "kernel_verification_status",
        "kernel_kernel",
        "kernel_verification_findings",
    }
)


class InstallerStateError(RuntimeError):
    pass


class CorruptInstallState(InstallerStateError):
    def __init__(self, message: str, preserved_path: Path) -> None:
        super().__init__(message)
        self.preserved_path = preserved_path


class InstallAlreadyRunning(InstallerStateError):
    pass


class ResumeInputChanged(InstallerStateError):
    def __init__(
        self,
        stage: InstallStage,
        expected_digest: str,
        actual_digest: str,
    ) -> None:
        super().__init__(
            f"completed stage inputs changed for {stage.value}: "
            f"expected {expected_digest}, got {actual_digest}"
        )
        self.stage = stage
        self.expected_digest = expected_digest
        self.actual_digest = actual_digest


@dataclass(frozen=True)
class StatePathSelection:
    path: Path
    source: str


def stage_input_digest(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise InstallerStateError(
            f"stage inputs are not canonical JSON: {error}"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def save_state(path: Path, state: InstallState) -> None:
    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    payload = _state_payload(state)
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def load_state(path: Path) -> InstallState | None:
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as error:
        if isinstance(error, UnicodeError):
            _raise_corrupt(path, f"install state is not UTF-8: {error}")
        raise InstallerStateError(f"cannot read install state: {error}") from error

    try:
        return _decode_state(raw)
    except (KeyError, TypeError, ValueError, InstallerModelError) as error:
        _raise_corrupt(path, f"invalid install state: {error}")


def project_state_path(project_dir: Path, legacy_path: Path) -> Path:
    legacy = Path(legacy_path).resolve(strict=False)
    return (
        legacy.parent
        / "projects"
        / f"{project_identity_key(project_dir)}.json"
    )


def project_identity_key(project_dir: Path) -> str:
    project = Path(project_dir).resolve(strict=False)
    readable = re.sub(r"[^A-Za-z0-9._-]+", "-", project.name)
    readable = readable.strip(".-_")[:48] or "project"
    identity = hashlib.sha256(str(project).encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{identity}"


def select_install_state_path(
    *,
    project_dir: Path,
    requested_path: Path,
    explicit: bool,
) -> StatePathSelection:
    project = Path(project_dir).resolve(strict=False)
    requested = Path(requested_path).resolve(strict=False)
    if explicit:
        return StatePathSelection(requested, "explicit")

    per_project = project_state_path(project, requested)
    if os.path.lexists(per_project):
        return StatePathSelection(per_project, "project")

    legacy_exists, legacy_project = _inspect_state_project_path(requested)
    if not legacy_exists:
        return StatePathSelection(per_project, "project")
    if legacy_project is None or legacy_project == str(project):
        return StatePathSelection(requested, "legacy")
    return StatePathSelection(per_project, "project")


@contextmanager
def install_lock(state_path: Path) -> Iterator[Path]:
    state_path = Path(state_path)
    lock_path = state_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise InstallerStateError(
            f"cannot open installer lock {lock_path}: {error}"
        ) from error
    try:
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in (errno.EACCES, errno.EAGAIN):
                raise InstallAlreadyRunning(
                    f"another installer is using {state_path}"
                ) from error
            raise InstallerStateError(
                f"cannot lock installer state {state_path}: {error}"
            ) from error
        try:
            yield lock_path
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


@contextmanager
def installer_coordination_lock(state_path: Path) -> Iterator[Path]:
    coordination_state = Path(state_path).resolve(strict=False)
    with install_lock(coordination_state) as lock_path:
        yield lock_path


def validate_completed_stage(
    state: InstallState,
    stage: InstallStage,
    current_inputs: object,
) -> bool:
    stage = InstallStage(stage)
    expected = state.completed_stage_input_digests.get(stage.value)
    if expected is None:
        return False
    actual = stage_input_digest(current_inputs)
    if actual != expected:
        raise ResumeInputChanged(stage, expected, actual)
    return True


def read_boot_id(path: Path = BOOT_ID_PATH) -> str:
    try:
        raw = Path(path).read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise InstallerStateError(f"cannot read boot ID: {error}") from error
    return _canonical_boot_id(raw)


def boot_id_changed(
    previous_boot_id: str, *, current_boot_id: str | None = None
) -> bool:
    previous = _canonical_boot_id(previous_boot_id)
    current = (
        read_boot_id()
        if current_boot_id is None
        else _canonical_boot_id(current_boot_id)
    )
    return current != previous


def _state_payload(state: InstallState) -> dict[str, object]:
    return {
        "schema_version": state.schema_version,
        "installer_version": state.installer_version,
        "mode": state.mode.value,
        "target_user": state.target_user,
        "release_id": state.release_id,
        "source_revision": state.source_revision,
        "base_image_reference": state.base_image_reference,
        "base_manifest_digest": state.base_manifest_digest,
        "torch_image_reference": state.torch_image_reference,
        "torch_manifest_digest": state.torch_manifest_digest,
        "project_path": state.project_path,
        "current_stage": state.current_stage.value,
        "completed_stage_input_digests": dict(
            state.completed_stage_input_digests
        ),
        "reboot_boot_id": state.reboot_boot_id,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
        "installer_source_revision": state.installer_source_revision,
        "source_root": state.source_root,
        "host_plan_digest": state.host_plan_digest,
        "host_adapter_id": state.host_adapter_id,
        "docker_group_accepted": state.docker_group_accepted,
        "base_config_digest": state.base_config_digest,
        "torch_config_digest": state.torch_config_digest,
        "last_report_paths": list(state.last_report_paths),
        "host_verification_status": state.host_verification_status,
        "host_kernel": state.host_kernel,
        "host_verification_findings": list(state.host_verification_findings),
        "kernel_plan_digest": state.kernel_plan_digest,
        "kernel_reboot_boot_id": state.kernel_reboot_boot_id,
        "recovery_kernel": state.recovery_kernel,
        "display_manager_was_active": state.display_manager_was_active,
        "kernel_verification_status": state.kernel_verification_status,
        "kernel_kernel": state.kernel_kernel,
        "kernel_verification_findings": list(
            state.kernel_verification_findings
        ),
    }


def _decode_state(raw: str) -> InstallState:
    payload = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(payload, dict):
        raise ValueError("install state must be a JSON object")
    payload = _migrate_payload(payload)
    unknown = sorted(set(payload).difference(STATE_KEYS))
    if unknown:
        raise ValueError("unknown install state keys: " + ", ".join(unknown))
    missing = sorted(STATE_KEYS.difference(payload))
    if missing:
        raise ValueError("missing install state keys: " + ", ".join(missing))
    completed = payload["completed_stage_input_digests"]
    reports = payload["last_report_paths"]
    host_findings = payload["host_verification_findings"]
    kernel_findings = payload["kernel_verification_findings"]
    if not isinstance(completed, dict):
        raise ValueError("completed stage digests must be an object")
    if not isinstance(reports, list):
        raise ValueError("last report paths must be an array")
    if not isinstance(host_findings, list):
        raise ValueError("host verification findings must be an array")
    if not isinstance(kernel_findings, list):
        raise ValueError("kernel verification findings must be an array")
    return InstallState(
        **{
            **payload,
            "completed_stage_input_digests": completed,
            "last_report_paths": tuple(reports),
            "host_verification_findings": tuple(host_findings),
            "kernel_verification_findings": tuple(kernel_findings),
        }
    )


def _inspect_state_project_path(path: Path) -> tuple[bool, str | None]:
    if not os.path.lexists(path):
        return False, None
    try:
        raw = path.read_text(encoding="utf-8")
        state = _decode_state(raw)
    except (KeyError, OSError, TypeError, UnicodeError, ValueError, InstallerModelError):
        return True, None
    return True, state.project_path


def _migrate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    version = payload.get("schema_version")
    if type(version) is not int:
        raise ValueError("install state schema version is invalid")
    if version == 1:
        _require_schema_keys(payload, STATE_KEYS_V1)
        payload = {
            **payload,
            "schema_version": 2,
            "host_verification_status": None,
            "host_kernel": None,
            "host_verification_findings": [],
        }
        version = 2
    if version == 2:
        _require_schema_keys(payload, STATE_KEYS_V2)
        payload = _migrate_schema_two(payload)
        version = STATE_SCHEMA_VERSION
    if version != STATE_SCHEMA_VERSION:
        raise ValueError("install state schema version is invalid")
    return payload


def _require_schema_keys(
    payload: Mapping[str, Any],
    expected: frozenset[str],
) -> None:
    unknown = sorted(set(payload).difference(expected))
    if unknown:
        raise ValueError("unknown install state keys: " + ", ".join(unknown))
    missing = sorted(expected.difference(payload))
    if missing:
        raise ValueError("missing install state keys: " + ", ".join(missing))


def _migrate_schema_two(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = {
        **payload,
        "schema_version": STATE_SCHEMA_VERSION,
        "kernel_plan_digest": None,
        "kernel_reboot_boot_id": None,
        "recovery_kernel": None,
        "display_manager_was_active": False,
        "kernel_verification_status": None,
        "kernel_kernel": None,
        "kernel_verification_findings": [],
    }
    if payload.get("mode") == "full":
        migrated.update(
            {
                "current_stage": InstallStage.BOOTSTRAP.value,
                "completed_stage_input_digests": {},
                "reboot_boot_id": None,
                "host_plan_digest": None,
                "host_adapter_id": None,
                "host_verification_status": None,
                "host_kernel": None,
                "host_verification_findings": [],
            }
        )
    return migrated


def _unique_object(pairs: list[tuple[str, Any]]) -> Mapping[str, Any]:
    values: dict[str, Any] = {}
    for key, value in pairs:
        if key in values:
            raise ValueError(f"duplicate JSON key: {key}")
        values[key] = value
    return values


def _raise_corrupt(path: Path, message: str) -> None:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    stem = path.name.removesuffix(".json")
    preserved = path.with_name(f"{stem}.corrupt.{timestamp}.json")
    try:
        os.replace(path, preserved)
        preserved.chmod(0o600)
        _fsync_directory(path.parent)
    except OSError as error:
        raise InstallerStateError(
            f"{message}; cannot preserve corrupt state: {error}"
        ) from error
    raise CorruptInstallState(message, preserved)


def _canonical_boot_id(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("boot ID is not a canonical UUID") from error
    canonical = str(parsed)
    if value != canonical:
        raise ValueError("boot ID is not a canonical UUID")
    return canonical


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
