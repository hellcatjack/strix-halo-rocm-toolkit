from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from amd_ai.overlay.lock import LockError, parse_lock, validate_lock_artifacts
from amd_ai.overlay.models import OverlayPaths, OverlayState, ProtectedProfile
from amd_ai.overlay.resolver import ProcessRunner
from amd_ai.overlay.transaction import (
    TransactionError,
    build_generation,
    ensure_layout,
    load_generation_state,
    overlay_transaction,
    resolve_current_generation,
)


REASON_PATTERN = re.compile(r"[A-Z][A-Z0-9_.]{2,127}")


class GenerationBuilder(Protocol):
    def build(
        self,
        paths: OverlayPaths,
        *,
        profile: ProtectedProfile,
        input_text: str,
        lock_text: str,
    ) -> OverlayState:
        pass


@dataclass(frozen=True)
class TransactionalGenerationBuilder:
    runner: ProcessRunner
    verifier: Callable[[Path], None]
    base_environment: Mapping[str, str] | None = None

    def build(
        self,
        paths: OverlayPaths,
        *,
        profile: ProtectedProfile,
        input_text: str,
        lock_text: str,
    ) -> OverlayState:
        return build_generation(
            paths,
            profile=profile,
            input_text=input_text,
            lock_text=lock_text,
            runner=self.runner,
            verifier=self.verifier,
            base_environment=self.base_environment,
            acquire_lock=False,
        )


@dataclass(frozen=True)
class OverlayRepairResult:
    quarantine: Path
    new_generation: Path
    state: OverlayState


def repair_overlay(
    paths: OverlayPaths,
    *,
    profile: ProtectedProfile,
    reason_code: str,
    builder: GenerationBuilder,
    doctor_report: Mapping[str, object] | None = None,
) -> OverlayRepairResult:
    if REASON_PATTERN.fullmatch(reason_code) is None:
        raise TransactionError(f"invalid overlay quarantine reason: {reason_code}")
    with overlay_transaction(paths):
        quarantine = _existing_retry(paths, reason_code)
        if quarantine is None:
            quarantine = _quarantine_current(
                paths,
                profile=profile,
                reason_code=reason_code,
                doctor_report=doctor_report,
            )
        generation = quarantine / "generation"
        input_text, lock_text, _ = _validate_quarantined_generation(
            generation,
            paths=paths,
            profile=profile,
        )
        try:
            locked = parse_lock(lock_text)
            validate_lock_artifacts(locked, project=paths.project)
        except LockError as error:
            _update_quarantine_status(
                quarantine, "blocked", f"lock validation failed: {error}"
            )
            raise TransactionError(
                f"quarantined overlay lock is invalid: {error}"
            ) from error
        try:
            state = builder.build(
                paths,
                profile=profile,
                input_text=input_text,
                lock_text=lock_text,
            )
            new_generation = resolve_current_generation(paths)
            if new_generation.name != state.generation_id:
                raise TransactionError(
                    "rebuilt overlay current link differs from returned state"
                )
        except Exception as error:
            _update_quarantine_status(
                quarantine, "rebuild-failed", str(error)
            )
            if isinstance(error, TransactionError):
                raise
            raise TransactionError(f"overlay replay failed: {error}") from error
        _update_quarantine_status(quarantine, "rebuilt", "")
        return OverlayRepairResult(quarantine, new_generation, state)


def _quarantine_current(
    paths: OverlayPaths,
    *,
    profile: ProtectedProfile,
    reason_code: str,
    doctor_report: Mapping[str, object] | None,
) -> Path:
    current = resolve_current_generation(paths)
    input_text, lock_text, state = _validate_quarantined_generation(
        current,
        paths=paths,
        profile=profile,
        validate_artifacts=False,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    quarantine = paths.quarantine / f"{timestamp}-{reason_code}"
    quarantine.mkdir(mode=0o700)
    destination = quarantine / "generation"
    try:
        os.replace(current, destination)
        _fsync_directory(paths.generations)
        _fsync_directory(quarantine)
        _write_json(
            quarantine / "quarantine-state.json",
            {
                "schema_version": 1,
                "reason_code": reason_code,
                "original_generation": state.generation_id,
                "profile_id": profile.profile_id,
                "parent_config_digest": profile.parent_config_digest,
                "input_digest": hashlib.sha256(input_text.encode()).hexdigest(),
                "lock_digest": hashlib.sha256(lock_text.encode()).hexdigest(),
                "status": "quarantined",
                "error": "",
            },
        )
        _write_json(
            quarantine / "doctor-report.json",
            {} if doctor_report is None else dict(doctor_report),
        )
    except Exception as error:
        raise TransactionError(f"cannot quarantine overlay generation: {error}") from error
    return quarantine


def _existing_retry(paths: OverlayPaths, reason_code: str) -> Path | None:
    if not paths.current.is_symlink():
        return None
    try:
        target_name = Path(os.readlink(paths.current)).name
    except OSError as error:
        raise TransactionError(f"cannot inspect dangling current link: {error}") from error
    if paths.current.exists():
        return None
    candidates = []
    for candidate in paths.quarantine.iterdir():
        if (
            candidate.is_dir()
            and candidate.name.endswith("-" + reason_code)
            and (candidate / "generation").is_dir()
        ):
            try:
                state = load_generation_state(candidate / "generation")
            except TransactionError:
                continue
            if state.generation_id == target_name:
                candidates.append(candidate)
    if len(candidates) != 1:
        raise TransactionError(
            "dangling current link does not identify one retryable quarantine"
        )
    return candidates[0]


def _validate_quarantined_generation(
    generation: Path,
    *,
    paths: OverlayPaths,
    profile: ProtectedProfile,
    validate_artifacts: bool = True,
) -> tuple[str, str, OverlayState]:
    state = load_generation_state(generation)
    if (
        (generation.name != "generation" and state.generation_id != generation.name)
        or state.profile_id != profile.profile_id
        or state.parent_config_digest != profile.parent_config_digest
    ):
        raise TransactionError("overlay generation identity differs from profile")
    input_text = _read_regular_text(generation / "overlay.requirements.in")
    lock_text = _read_regular_text(generation / "overlay.requirements.lock")
    if hashlib.sha256(input_text.encode()).hexdigest() != state.input_digest:
        raise TransactionError("overlay generation input digest changed")
    if hashlib.sha256(lock_text.encode()).hexdigest() != state.lock_digest:
        raise TransactionError("overlay generation lock digest changed")
    if validate_artifacts:
        try:
            validate_lock_artifacts(parse_lock(lock_text), project=paths.project)
        except LockError as error:
            raise TransactionError(f"overlay generation lock is invalid: {error}") from error
    return input_text, lock_text, state


def _update_quarantine_status(
    quarantine: Path, status: str, error: str
) -> None:
    path = quarantine / "quarantine-state.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as parse_error:
        raise TransactionError(
            f"cannot read quarantine state: {parse_error}"
        ) from parse_error
    if not isinstance(payload, dict):
        raise TransactionError("quarantine state is invalid")
    payload["status"] = status
    payload["error"] = error
    _write_json(path, payload)


def _read_regular_text(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise TransactionError(f"overlay metadata is not a regular file: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise TransactionError(f"cannot read overlay metadata {path}: {error}") from error


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    content = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
