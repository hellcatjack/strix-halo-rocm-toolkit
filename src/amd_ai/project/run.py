from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from amd_ai.project.build import FINGERPRINT_PATTERN
from amd_ai.project.config import (
    ENVIRONMENT_PATTERN,
    IMAGE_ID_PATTERN,
    RESERVED_ENVIRONMENT,
    ProjectConfig,
)
from amd_ai.project.runtime import GpuAccess, mount_argv
from amd_ai.runner import Runner


SENSITIVE_ENVIRONMENT_TOKENS = ("TOKEN", "SECRET", "PASSWORD", "KEY")


class ProjectRunError(RuntimeError):
    pass


class UnverifiedImage(ProjectRunError):
    pass


@dataclass(frozen=True)
class ProjectImageMetadata:
    image_id: str
    profile_id: str
    profile_status: str
    rocm_version: str
    python_version: str
    torch_version: str
    base_digest: str
    fingerprint: str


def require_profile_allowed(
    profile_status: str,
    environment: Mapping[str, str],
) -> None:
    if profile_status == "verified":
        return
    if profile_status != "experimental":
        raise ProjectRunError(f"invalid image profile status: {profile_status!r}")
    if environment.get("ALLOW_UNVERIFIED") != "1":
        raise UnverifiedImage(
            "experimental or unlabeled image requires ALLOW_UNVERIFIED=1"
        )


def build_run_argv(
    *,
    config: ProjectConfig,
    access: GpuAccess,
    uid: int,
    gid: int,
    shm_gib: int,
    environment: Mapping[str, str],
    terminal: bool,
    docker_prefix: Sequence[str] = ("docker",),
) -> tuple[str, ...]:
    _validate_identity(uid, gid)
    if (
        isinstance(shm_gib, bool)
        or not isinstance(shm_gib, int)
        or not 1 <= shm_gib <= 128
    ):
        raise ProjectRunError(
            "shared memory size must be an integer from 1 through 128 GiB"
        )
    if not docker_prefix or any(not value or "\0" in value for value in docker_prefix):
        raise ProjectRunError("Docker command prefix is invalid")
    if not access.devices:
        raise ProjectRunError("at least one GPU device is required")
    if any(group_id < 0 for group_id in access.group_ids):
        raise ProjectRunError("GPU group IDs must be nonnegative")

    project_dir = config.path.parent.resolve()
    if not project_dir.is_dir():
        raise ProjectRunError(f"project directory does not exist: {project_dir}")

    argv: list[str] = [*docker_prefix, "run", "--rm"]
    if terminal:
        argv.extend(("--interactive", "--tty"))
    argv.append("--ipc=private")
    for device in access.devices:
        argv.extend(("--device", str(device)))
    for group_id in access.group_ids:
        argv.extend(("--group-add", str(group_id)))
    argv.extend(
        (
            "--user",
            f"{uid}:{gid}",
            "--shm-size",
            f"{shm_gib}g",
            "--workdir",
            "/workspace",
            "--env",
            "HOME=/workspace/.amd-ai/home",
        )
    )
    for name, value in config.environment:
        _validate_config_environment(name, value)
        argv.extend(("--env", f"{name}={value}"))
    if environment.get("ALLOW_UNVERIFIED") == "1":
        argv.extend(("--env", "ALLOW_UNVERIFIED=1"))
    if config.debug:
        argv.extend(
            (
                "--cap-add",
                "SYS_PTRACE",
                "--security-opt",
                "seccomp=unconfined",
            )
        )
    argv.extend(
        (
            "--mount",
            f"type=bind,src={project_dir},dst=/workspace",
        )
    )
    if any(mount.target == PurePosixPath("/workspace") for mount in config.mounts):
        raise ProjectRunError("user mount target /workspace would replace the project")
    argv.extend(mount_argv(config.mounts))
    argv.append(config.image)
    argv.extend(config.command)
    return tuple(argv)


def redact_run_argv(argv: Sequence[str]) -> tuple[str, ...]:
    redacted = list(argv)
    index = 0
    while index < len(redacted):
        argument = redacted[index]
        if argument == "--env" and index + 1 < len(redacted):
            redacted[index + 1] = _redact_environment_assignment(redacted[index + 1])
            index += 2
            continue
        if argument.startswith("--env="):
            assignment = argument.removeprefix("--env=")
            redacted[index] = "--env=" + _redact_environment_assignment(assignment)
        index += 1
    return tuple(redacted)


def inspect_project_image(
    config: ProjectConfig,
    runner: Runner,
    docker_prefix: Sequence[str] = ("docker",),
) -> ProjectImageMetadata:
    args = [*docker_prefix, "image", "inspect", config.image]
    result = runner.run(args, check=False)
    if result.returncode != 0:
        raise ProjectRunError(
            f"cannot inspect project image {config.image}: {_evidence(result)}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ProjectRunError("cannot parse project image metadata") from error
    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], dict)
    ):
        raise ProjectRunError("unexpected project image metadata")
    record = payload[0]
    image_id = record.get("Id")
    if not isinstance(image_id, str) or IMAGE_ID_PATTERN.fullmatch(image_id) is None:
        raise ProjectRunError("project image has no immutable image ID")
    image_config = record.get("Config")
    if not isinstance(image_config, dict):
        raise ProjectRunError("project image has no config metadata")
    labels = image_config.get("Labels") or {}
    if not isinstance(labels, dict):
        raise ProjectRunError("project image labels are invalid")

    profile_status = str(labels.get("org.amd-ai.profile.status", "experimental"))
    if profile_status not in {"verified", "experimental"}:
        raise ProjectRunError("project image profile status is invalid")
    base_digest = _required_label(labels, "org.amd-ai.base.digest")
    if IMAGE_ID_PATTERN.fullmatch(base_digest) is None:
        raise ProjectRunError("project image base digest label is invalid")
    if base_digest != config.base_digest:
        raise ProjectRunError("project image base digest does not match its config")
    fingerprint = _required_label(labels, "org.amd-ai.project.fingerprint")
    if FINGERPRINT_PATTERN.fullmatch(fingerprint) is None:
        raise ProjectRunError("project image fingerprint label is invalid")

    return ProjectImageMetadata(
        image_id=image_id,
        profile_id=_required_label(labels, "org.amd-ai.profile.id"),
        profile_status=profile_status,
        rocm_version=_required_label(labels, "org.amd-ai.rocm.version"),
        python_version=_required_label(labels, "org.amd-ai.python.version"),
        torch_version=_required_label(labels, "org.amd-ai.torch.version"),
        base_digest=base_digest,
        fingerprint=fingerprint,
    )


def ensure_project_home(project_dir: Path, *, uid: int, gid: int) -> Path:
    _validate_identity(uid, gid)
    project_dir = project_dir.resolve()
    if not project_dir.is_dir():
        raise ProjectRunError(f"project directory does not exist: {project_dir}")
    control_dir = project_dir / ".amd-ai"
    home = control_dir / "home"
    _ensure_private_directory(control_dir, uid=uid, gid=gid)
    _ensure_private_directory(home, uid=uid, gid=gid)
    return home


def _ensure_private_directory(path: Path, *, uid: int, gid: int) -> None:
    if path.is_symlink():
        raise ProjectRunError(f"runtime directory must not be a symbolic link: {path}")
    try:
        path.mkdir(mode=0o700, exist_ok=True)
    except OSError as error:
        raise ProjectRunError(f"cannot create runtime directory {path}: {error}") from error
    if path.is_symlink():
        raise ProjectRunError(f"runtime directory must not be a symbolic link: {path}")
    try:
        metadata = path.stat()
    except OSError as error:
        raise ProjectRunError(f"cannot inspect runtime directory {path}: {error}") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise ProjectRunError(f"runtime path is not a directory: {path}")
    try:
        path.chmod(0o700)
        if (metadata.st_uid, metadata.st_gid) != (uid, gid):
            os.chown(path, uid, gid)
    except OSError as error:
        raise ProjectRunError(
            f"cannot set runtime directory ownership for {path}: {error}"
        ) from error
    metadata = path.stat()
    if (metadata.st_uid, metadata.st_gid) != (uid, gid):
        raise ProjectRunError(f"runtime directory has the wrong owner: {path}")


def _validate_identity(uid: int, gid: int) -> None:
    if (
        isinstance(uid, bool)
        or not isinstance(uid, int)
        or uid < 0
        or isinstance(gid, bool)
        or not isinstance(gid, int)
        or gid < 0
    ):
        raise ProjectRunError("runtime UID and GID must be nonnegative integers")


def _validate_config_environment(name: str, value: str) -> None:
    if ENVIRONMENT_PATTERN.fullmatch(name) is None or name in RESERVED_ENVIRONMENT:
        raise ProjectRunError(f"invalid project environment name: {name!r}")
    if not isinstance(value, str) or not value or "\0" in value:
        raise ProjectRunError(f"invalid project environment value for {name}")


def _redact_environment_assignment(assignment: str) -> str:
    name, _, _ = assignment.partition("=")
    if any(token in name.upper() for token in SENSITIVE_ENVIRONMENT_TOKENS):
        return f"{name}=<redacted>"
    return assignment


def _required_label(labels: Mapping[object, object], name: str) -> str:
    value = labels.get(name)
    if not isinstance(value, str) or not value:
        raise ProjectRunError(f"project image is missing label: {name}")
    return value


def _evidence(result: object) -> str:
    stderr = str(getattr(result, "stderr", "")).strip()
    stdout = str(getattr(result, "stdout", "")).strip()
    return stderr or stdout or "command failed without output"
