from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from amd_ai.overlay.models import (
    PROTECTED_DISTRIBUTIONS,
    canonicalize_protected_name,
)
from amd_ai.overlay.packaging_compat import (
    InvalidWheelFilename,
    canonicalize_name,
    parse_wheel_filename,
)
from amd_ai.overlay.resolver import WheelArtifact


HEADER_PATTERN = re.compile(
    r"(?P<name>[a-z0-9][a-z0-9-]*) @ "
    r"(?P<url>file:///workspace/\.amd-ai/artifacts/sha256/"
    r"(?P<digest>[0-9a-f]{64})/(?P<filename>[^\s/]+\.whl)) \\"
)
HASH_PATTERN = re.compile(r"    --hash=sha256:(?P<digest>[0-9a-f]{64})")
CONTAINER_PREFIX = PurePosixPath("/workspace")


class LockError(ValueError):
    pass


@dataclass(frozen=True)
class LockedWheel:
    name: str
    version: str
    sha256: str
    container_path: str


def render_lock(
    artifacts: tuple[WheelArtifact, ...], *, project: Path
) -> str:
    project_root = project.resolve(strict=True)
    lexical_artifact_root = (
        project_root / ".amd-ai" / "artifacts" / "sha256"
    )
    _reject_symlink_components(lexical_artifact_root, project_root)
    artifact_root = lexical_artifact_root.resolve()
    ordered = sorted(artifacts, key=lambda artifact: artifact.name)
    names: set[str] = set()
    lines: list[str] = []
    for artifact in ordered:
        name, version = _wheel_identity(artifact.path.name)
        if name != canonicalize_name(artifact.name) or version != artifact.version:
            raise LockError(
                f"wheel identity differs from artifact metadata: {artifact.path.name}"
            )
        if name in names:
            raise LockError(f"duplicate overlay distribution: {name}")
        if canonicalize_protected_name(name) in PROTECTED_DISTRIBUTIONS:
            raise LockError(f"overlay lock contains protected distribution: {name}")
        path = artifact.path
        if path.is_symlink():
            raise LockError(f"wheel artifact must not be a symlink: {path}")
        resolved = path.resolve(strict=True)
        expected = artifact_root / artifact.sha256 / path.name
        if resolved != expected or not resolved.is_relative_to(artifact_root):
            raise LockError(f"wheel is outside the project artifact store: {path}")
        actual_digest = _hash_file(resolved)
        if actual_digest != artifact.sha256:
            raise LockError(f"wheel artifact hash changed: {path}")
        container_path = (
            CONTAINER_PREFIX
            / ".amd-ai"
            / "artifacts"
            / "sha256"
            / artifact.sha256
            / path.name
        )
        lines.extend(
            (
                f"{name} @ file://{container_path} \\",
                f"    --hash=sha256:{artifact.sha256}",
            )
        )
        names.add(name)
    return "\n".join(lines) + ("\n" if lines else "")


def parse_lock(text: str) -> tuple[LockedWheel, ...]:
    if not text:
        return ()
    lines = text.splitlines()
    if len(lines) % 2:
        raise LockError("overlay lock entries must contain exactly two lines")
    locked: list[LockedWheel] = []
    previous_name = ""
    names: set[str] = set()
    for index in range(0, len(lines), 2):
        header_match = HEADER_PATTERN.fullmatch(lines[index])
        hash_match = HASH_PATTERN.fullmatch(lines[index + 1])
        if header_match is None or hash_match is None:
            raise LockError(
                f"invalid overlay lock entry at line {index + 1}"
            )
        name = header_match.group("name")
        digest = header_match.group("digest")
        if hash_match.group("digest") != digest:
            raise LockError(f"overlay lock hash mismatch for {name}")
        wheel_name, version = _wheel_identity(header_match.group("filename"))
        if wheel_name != name:
            raise LockError(f"overlay lock wheel identity mismatch for {name}")
        if canonicalize_protected_name(name) in PROTECTED_DISTRIBUTIONS:
            raise LockError(f"overlay lock contains protected distribution: {name}")
        if name in names:
            raise LockError(f"duplicate overlay distribution: {name}")
        if previous_name and name < previous_name:
            raise LockError("overlay lock distributions are not sorted")
        path = header_match.group("url").removeprefix("file://")
        locked.append(LockedWheel(name, version, digest, path))
        names.add(name)
        previous_name = name
    return tuple(locked)


def validate_lock_artifacts(
    locked: tuple[LockedWheel, ...], *, project: Path
) -> tuple[Path, ...]:
    project_root = project.resolve(strict=True)
    lexical_artifact_root = (
        project_root / ".amd-ai" / "artifacts" / "sha256"
    )
    _reject_symlink_components(lexical_artifact_root, project_root)
    try:
        artifact_root = lexical_artifact_root.resolve(strict=True)
    except OSError as error:
        raise LockError("project artifact store is missing") from error
    paths: list[Path] = []
    for wheel in locked:
        container_path = PurePosixPath(wheel.container_path)
        try:
            relative = container_path.relative_to(CONTAINER_PREFIX)
        except ValueError as error:
            raise LockError(
                f"lock path is outside /workspace: {wheel.container_path}"
            ) from error
        path = project_root.joinpath(*relative.parts)
        if path.is_symlink():
            raise LockError(f"wheel artifact must not be a symlink: {path}")
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise LockError(f"wheel artifact is missing: {path}") from error
        expected = artifact_root / wheel.sha256 / path.name
        if resolved != expected or not resolved.is_relative_to(artifact_root):
            raise LockError(f"wheel artifact path escaped its store: {path}")
        if _hash_file(resolved) != wheel.sha256:
            raise LockError(f"wheel artifact hash changed: {path}")
        paths.append(resolved)
    return tuple(paths)


def lock_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _wheel_identity(filename: str) -> tuple[str, str]:
    try:
        parsed_name, parsed_version, _, _ = parse_wheel_filename(filename)
    except InvalidWheelFilename as error:
        raise LockError(f"invalid wheel filename: {filename}") from error
    return canonicalize_name(str(parsed_name)), str(parsed_version)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_symlink_components(path: Path, boundary: Path) -> None:
    relative = path.relative_to(boundary)
    current = boundary
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise LockError(f"artifact store contains a symlink: {current}")
