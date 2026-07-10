from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,62}")
PROFILE_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
ENVIRONMENT_PATTERN = re.compile(r"[A-Z_][A-Z0-9_]*")
TOP_LEVEL_KEYS = frozenset({"project", "mounts", "environment"})
PROJECT_KEYS = frozenset(
    {
        "name",
        "base_profile",
        "image",
        "base_image",
        "base_digest",
        "command",
        "debug",
        "shm_size_gib",
    }
)
REQUIRED_PROJECT_KEYS = PROJECT_KEYS - {"shm_size_gib"}
MOUNT_KEYS = frozenset({"source", "target", "read_only"})
RESERVED_ENVIRONMENT = frozenset(
    {
        "PATH",
        "PYTHONPATH",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "LD_LIBRARY_PATH",
        "ROCM_PATH",
        "HIP_PATH",
        "HOME",
        "ALLOW_UNVERIFIED",
        "AMD_AI_PROFILE_ID",
        "AMD_AI_PROFILE_STATUS",
        "AMD_AI_OVERLAY",
    }
)
RESERVED_MOUNT_TARGETS = tuple(
    PurePosixPath(value)
    for value in (
        "/opt/venv",
        "/opt/rocm",
        "/usr/local/bin",
        "/opt/amd-ai",
        "/dev",
        "/proc",
        "/sys",
    )
)


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class MountConfig:
    source: Path
    target: PurePosixPath
    read_only: bool


@dataclass(frozen=True)
class ProjectConfig:
    path: Path
    name: str
    base_profile: str
    image: str
    base_image: str
    base_digest: str
    command: tuple[str, ...]
    debug: bool
    shm_size_gib: int | None
    mounts: tuple[MountConfig, ...]
    environment: tuple[tuple[str, str], ...]


def load_project_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).resolve()
    try:
        with config_path.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot read project config {config_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ConfigError("project config must be a TOML table")
    _reject_unknown("top-level", payload, TOP_LEVEL_KEYS)

    project = payload.get("project")
    if not isinstance(project, dict):
        raise ConfigError("[project] table is required")
    _reject_unknown("project", project, PROJECT_KEYS)
    missing = sorted(REQUIRED_PROJECT_KEYS.difference(project))
    if missing:
        raise ConfigError("missing project keys: " + ", ".join(missing))

    name = _matching_string(project, "name", NAME_PATTERN)
    base_profile = _matching_string(project, "base_profile", PROFILE_PATTERN)
    image = _string(project, "image")
    if (
        image.startswith("-")
        or any(character.isspace() for character in image)
        or "\0" in image
    ):
        raise ConfigError(
            "project image must not start with '-' or contain whitespace or NUL"
        )
    base_image = _matching_string(project, "base_image", IMAGE_ID_PATTERN)
    base_digest = _matching_string(project, "base_digest", IMAGE_ID_PATTERN)
    if base_image != base_digest:
        raise ConfigError("base_image and base_digest must match")

    raw_command = project["command"]
    if (
        not isinstance(raw_command, list)
        or not raw_command
        or any(
            not isinstance(value, str) or not value or "\0" in value
            for value in raw_command
        )
    ):
        raise ConfigError("project command must be a nonempty string array")
    debug = project["debug"]
    if not isinstance(debug, bool):
        raise ConfigError("project debug must be a boolean")
    shm_size_gib = project.get("shm_size_gib")
    if shm_size_gib is not None and (
        isinstance(shm_size_gib, bool)
        or not isinstance(shm_size_gib, int)
        or not 1 <= shm_size_gib <= 128
    ):
        raise ConfigError("shm_size_gib must be an integer from 1 through 128")

    mounts = _parse_mounts(payload.get("mounts", []), config_path.parent)
    environment = _parse_environment(payload.get("environment", {}))
    return ProjectConfig(
        path=config_path,
        name=name,
        base_profile=base_profile,
        image=image,
        base_image=base_image,
        base_digest=base_digest,
        command=tuple(raw_command),
        debug=debug,
        shm_size_gib=shm_size_gib,
        mounts=mounts,
        environment=environment,
    )


def _parse_mounts(raw_mounts: object, config_dir: Path) -> tuple[MountConfig, ...]:
    if not isinstance(raw_mounts, list):
        raise ConfigError("mounts must be an array of tables")
    mounts: list[MountConfig] = []
    targets: set[PurePosixPath] = set()
    for index, raw_mount in enumerate(raw_mounts, start=1):
        if not isinstance(raw_mount, dict):
            raise ConfigError(f"mount {index} must be a table")
        _reject_unknown(f"mount {index}", raw_mount, MOUNT_KEYS)
        missing = sorted(MOUNT_KEYS.difference(raw_mount))
        if missing:
            raise ConfigError(f"mount {index} is missing: " + ", ".join(missing))
        source_text = _raw_string(raw_mount["source"], f"mount {index} source")
        if any(token in source_text for token in ("\0", "\n", "\r", ",")):
            raise ConfigError(f"mount {index} source contains a forbidden character")
        source = Path(source_text)
        if not source.is_absolute():
            source = config_dir / source
        source = source.resolve()

        target_text = _raw_string(raw_mount["target"], f"mount {index} target")
        target = PurePosixPath(target_text)
        if (
            not target.is_absolute()
            or str(target) != target_text
            or ".." in target.parts
            or any(token in target_text for token in ("\0", "\n", "\r", ","))
        ):
            raise ConfigError(f"mount {index} target must be absolute and normalized")
        if target in targets:
            raise ConfigError(f"duplicate mount target: {target}")
        if any(
            target == reserved or reserved in target.parents
            for reserved in RESERVED_MOUNT_TARGETS
        ):
            raise ConfigError(f"mount target is reserved: {target}")
        read_only = raw_mount["read_only"]
        if not isinstance(read_only, bool):
            raise ConfigError(f"mount {index} read_only must be a boolean")
        targets.add(target)
        mounts.append(MountConfig(source, target, read_only))
    return tuple(mounts)


def _parse_environment(raw_environment: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw_environment, dict):
        raise ConfigError("[environment] must be a table")
    environment: list[tuple[str, str]] = []
    for name, raw_value in raw_environment.items():
        if ENVIRONMENT_PATTERN.fullmatch(name) is None:
            raise ConfigError(f"invalid environment name: {name}")
        if name in RESERVED_ENVIRONMENT:
            raise ConfigError(f"reserved environment name: {name}")
        value = _raw_string(raw_value, f"environment {name}")
        if "\0" in value:
            raise ConfigError(f"environment {name} contains NUL")
        environment.append((name, value))
    return tuple(sorted(environment))


def _reject_unknown(section: str, values: dict, allowed: frozenset[str]) -> None:
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ConfigError(f"unknown {section} keys: " + ", ".join(unknown))


def _matching_string(values: dict, key: str, pattern: re.Pattern[str]) -> str:
    value = _string(values, key)
    if pattern.fullmatch(value) is None:
        raise ConfigError(f"invalid project {key}: {value!r}")
    return value


def _string(values: dict, key: str) -> str:
    return _raw_string(values[key], f"project {key}")


def _raw_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{label} must be a nonempty string")
    return value
