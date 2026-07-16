from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from amd_ai.project.config import IMAGE_ID_PATTERN, ProjectConfig
from amd_ai.runner import Runner


FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}")
DEFAULT_IGNORES = (
    ".git",
    ".venv",
    ".cache",
    ".amd-ai",
    "models",
    "input",
    "output",
    "reports",
    "__pycache__",
    "*.pyc",
)
REQUIRED_BUILD_FILES = (
    ".dockerignore",
    "Dockerfile",
    "project-entrypoint",
    "amd-ai-project.toml",
    "requirements.in",
    "requirements.lock",
    "torch-constraints.txt",
)


class ProjectBuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class ParentImageMetadata:
    image_id: str
    profile_id: str
    profile_status: str
    rocm_version: str
    python_version: str
    torch_version: str
    layers: tuple[str, ...]


@dataclass(frozen=True)
class ProjectBuildResult:
    image: str
    image_id: str
    fingerprint: str
    built: bool
    parent: ParentImageMetadata


@dataclass(frozen=True)
class _IgnoreRule:
    pattern: str
    negated: bool
    directory_only: bool
    anchored: bool


def remove_exact_project_image(
    image_id: str,
    *,
    runner: Runner,
    docker_prefix: Sequence[str] = ("docker",),
) -> None:
    if IMAGE_ID_PATTERN.fullmatch(image_id) is None:
        raise ProjectBuildError(
            f"project image removal requires an exact image ID: {image_id}"
        )
    if not docker_prefix or any(not value for value in docker_prefix):
        raise ProjectBuildError("Docker command prefix is invalid")
    args = [*docker_prefix, "image", "rm", image_id]
    result = runner.run(args, check=False)
    if result.returncode != 0:
        evidence = result.stderr.strip() or result.stdout.strip() or "no output"
        raise ProjectBuildError(
            f"cannot remove exact project image {image_id}: {evidence}"
        )


def build_context_fingerprint(context: Path) -> str:
    context = context.resolve()
    if not context.is_dir():
        raise ProjectBuildError(f"project context is not a directory: {context}")
    rules = _ignore_rules(context)
    digest = hashlib.sha256()
    for path in _fingerprint_files(context, rules):
        relative = path.relative_to(context).as_posix()
        metadata = path.stat()
        mode = stat.S_IMODE(metadata.st_mode)
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(mode.to_bytes(4, "big"))
        digest.update(metadata.st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def project_parent_alias(parent: str) -> str:
    if IMAGE_ID_PATTERN.fullmatch(parent) is None:
        raise ProjectBuildError(f"invalid immutable project parent: {parent!r}")
    return f"amd-ai-local/project-base:{parent.removeprefix('sha256:')}"


def project_build_argv(
    *,
    context: Path,
    image: str,
    base_image: str,
    base_digest: str,
    profile_id: str,
    profile_status: str,
    fingerprint: str,
    docker_prefix: Sequence[str] = ("docker",),
) -> tuple[str, ...]:
    if base_image != base_digest:
        raise ProjectBuildError("base image and digest must match")
    alias = project_parent_alias(base_image)
    if FINGERPRINT_PATTERN.fullmatch(fingerprint) is None:
        raise ProjectBuildError("project fingerprint must be a lowercase SHA-256")
    if profile_status not in {"verified", "experimental"}:
        raise ProjectBuildError(f"invalid profile status: {profile_status!r}")
    return (
        *docker_prefix,
        "buildx",
        "build",
        "--progress=plain",
        "--load",
        "--platform",
        "linux/amd64",
        "--provenance=false",
        "--sbom=false",
        "--build-arg",
        f"BASE_IMAGE={alias}",
        "--build-arg",
        f"PROFILE_ID={profile_id}",
        "--build-arg",
        f"PROFILE_STATUS={profile_status}",
        "--label",
        f"org.amd-ai.project.fingerprint={fingerprint}",
        "--label",
        f"org.amd-ai.base.digest={base_digest}",
        "--tag",
        image,
        str(context),
    )


def build_or_reuse_project(
    *,
    config: ProjectConfig,
    runner: Runner,
    force: bool,
    no_build: bool,
    docker_prefix: Sequence[str] = ("docker",),
) -> ProjectBuildResult:
    context = config.path.parent
    for name in REQUIRED_BUILD_FILES:
        if not (context / name).is_file():
            raise ProjectBuildError(f"project build file is missing: {name}")
    validate_dockerignore(context)
    fingerprint = build_context_fingerprint(context)
    parent = inspect_parent_image(config, runner, docker_prefix)
    current = _inspect_image(config.image, runner, docker_prefix, required=False)
    if current is not None:
        labels = _labels(current, config.image)
        fresh = (
            labels.get("org.amd-ai.project.fingerprint") == fingerprint
            and labels.get("org.amd-ai.base.digest") == config.base_digest
        )
        if fresh and not force:
            validate_project_image_contract(current, parent)
            _verify_project_torch_manifest(
                config.image,
                runner,
                docker_prefix,
            )
            return ProjectBuildResult(
                image=config.image,
                image_id=_image_id(current, config.image),
                fingerprint=fingerprint,
                built=False,
                parent=parent,
            )
    if no_build:
        state = "stale" if current is not None else "missing"
        raise ProjectBuildError(f"project image is {state} and --no-build was requested")

    _require_buildx(runner, docker_prefix)
    alias = project_parent_alias(config.base_image)
    tag_result = runner.run(
        [*docker_prefix, "tag", config.base_image, alias],
        check=False,
    )
    if tag_result.returncode != 0:
        raise ProjectBuildError(
            f"cannot create immutable project parent alias: {_evidence(tag_result)}"
        )
    alias_record = _inspect_image(alias, runner, docker_prefix, required=True)
    assert alias_record is not None
    if _image_id(alias_record, alias) != config.base_image:
        raise ProjectBuildError("immutable project parent alias resolved to another image")

    argv = project_build_argv(
        context=context,
        image=config.image,
        base_image=config.base_image,
        base_digest=config.base_digest,
        profile_id=parent.profile_id,
        profile_status=parent.profile_status,
        fingerprint=fingerprint,
        docker_prefix=docker_prefix,
    )
    build_result = runner.run(list(argv), check=False)
    if build_result.returncode != 0:
        raise ProjectBuildError(f"project image build failed: {_evidence(build_result)}")
    built = _inspect_image(config.image, runner, docker_prefix, required=True)
    assert built is not None
    labels = _labels(built, config.image)
    if (
        labels.get("org.amd-ai.project.fingerprint") != fingerprint
        or labels.get("org.amd-ai.base.digest") != config.base_digest
    ):
        raise ProjectBuildError("built project image labels do not match its inputs")
    validate_project_image_contract(built, parent)
    _verify_project_torch_manifest(config.image, runner, docker_prefix)
    return ProjectBuildResult(
        image=config.image,
        image_id=_image_id(built, config.image),
        fingerprint=fingerprint,
        built=True,
        parent=parent,
    )


def _require_buildx(runner: Runner, docker_prefix: Sequence[str]) -> str:
    result = runner.run(
        [*docker_prefix, "buildx", "version"],
        check=False,
    )
    version = result.stdout.strip()
    if result.returncode != 0 or not version:
        evidence = result.stderr.strip() or version
        raise ProjectBuildError(
            "Docker Buildx is required to build the project image; "
            "run the toolkit host repair before retrying: "
            + (evidence or "Buildx version unavailable")
        )
    return version


def inspect_parent_image(
    config: ProjectConfig,
    runner: Runner,
    docker_prefix: Sequence[str] = ("docker",),
) -> ParentImageMetadata:
    record = _inspect_image(config.base_image, runner, docker_prefix, required=True)
    assert record is not None
    image_id = _image_id(record, config.base_image)
    if image_id != config.base_digest:
        raise ProjectBuildError("resolved parent image does not match the configured digest")
    labels = _labels(record, config.base_image)
    profile_id = labels.get("org.amd-ai.profile.id", "")
    if not profile_id:
        raise ProjectBuildError("parent image has no profile ID label")
    profile_status = labels.get("org.amd-ai.profile.status", "experimental")
    if profile_status not in {"verified", "experimental"}:
        raise ProjectBuildError("parent image has an invalid profile status label")
    return ParentImageMetadata(
        image_id=image_id,
        profile_id=profile_id,
        profile_status=profile_status,
        rocm_version=_required_label(labels, "org.amd-ai.rocm.version"),
        python_version=_required_label(labels, "org.amd-ai.python.version"),
        torch_version=_required_label(labels, "org.amd-ai.torch.version"),
        layers=_rootfs_layers(record, config.base_image),
    )


def validate_dockerignore(context: Path) -> None:
    path = context / ".dockerignore"
    if not path.is_file():
        raise ProjectBuildError("project .dockerignore is missing")
    try:
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    except OSError as error:
        raise ProjectBuildError(f"cannot read project .dockerignore: {error}") from error
    if any(line.startswith("!") for line in lines):
        raise ProjectBuildError(
            "project .dockerignore negation rules are forbidden by storage policy"
        )
    missing = sorted(set(DEFAULT_IGNORES).difference(lines))
    if missing:
        raise ProjectBuildError(
            "project .dockerignore is missing mandatory rules: " + ", ".join(missing)
        )


def validate_project_image_contract(
    record: Mapping[str, object],
    parent: ParentImageMetadata,
) -> None:
    layers = _rootfs_layers(record, "project image")
    if layers[: len(parent.layers)] != parent.layers:
        raise ProjectBuildError(
            "project image does not inherit the configured parent layer prefix"
        )
    labels = _labels(record, "project image")
    expected_labels = {
        "org.amd-ai.profile.id": parent.profile_id,
        "org.amd-ai.profile.status": parent.profile_status,
        "org.amd-ai.rocm.version": parent.rocm_version,
        "org.amd-ai.python.version": parent.python_version,
        "org.amd-ai.torch.version": parent.torch_version,
    }
    for name, expected in expected_labels.items():
        if labels.get(name) != expected:
            label = name.removeprefix("org.amd-ai.").replace(".", " ")
            raise ProjectBuildError(
                f"project image {label} does not match its parent"
            )
    config = record.get("Config")
    assert isinstance(config, dict)
    environment = _environment(config, "project image")
    for name, expected in {
        "AMD_AI_PROFILE_ID": parent.profile_id,
        "AMD_AI_PROFILE_STATUS": parent.profile_status,
    }.items():
        if environment.get(name) != expected:
            label = name.removeprefix("AMD_AI_").lower().replace("_", " ")
            raise ProjectBuildError(
                f"project image {label} environment does not match its parent"
            )
    if "ALLOW_UNVERIFIED" in environment:
        raise ProjectBuildError("project image must not persist ALLOW_UNVERIFIED")
    if config.get("User") != "1000:1000":
        raise ProjectBuildError("project image must run as user 1000:1000")
    if config.get("WorkingDir") != "/workspace":
        raise ProjectBuildError("project image workdir must be /workspace")
    if config.get("Entrypoint") != ["/usr/local/bin/project-entrypoint"]:
        raise ProjectBuildError("project image entrypoint is not the policy guard")


def project_manifest_argv(
    image: str,
    *,
    docker_prefix: Sequence[str] = ("docker",),
) -> tuple[str, ...]:
    return (
        *docker_prefix,
        "run",
        "--rm",
        "--entrypoint",
        "/opt/venv/bin/python",
        image,
        "/opt/amd-ai/torch-manifest.py",
        "verify",
        "/opt/amd-ai/torch-manifest.json",
    )


def _ignore_rules(context: Path) -> tuple[_IgnoreRule, ...]:
    lines = list(DEFAULT_IGNORES)
    dockerignore = context / ".dockerignore"
    if dockerignore.is_file():
        lines.extend(dockerignore.read_text(encoding="utf-8").splitlines())
    rules = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        if negated:
            line = line[1:]
        anchored = line.startswith("/")
        line = line.removeprefix("/")
        directory_only = line.endswith("/")
        line = line.removesuffix("/")
        if line:
            rules.append(_IgnoreRule(line, negated, directory_only, anchored))
    return tuple(rules)


def _fingerprint_files(
    context: Path,
    rules: tuple[_IgnoreRule, ...],
) -> tuple[Path, ...]:
    files: list[Path] = []
    has_negations = any(rule.negated for rule in rules)
    for root, directories, filenames in os.walk(context, followlinks=False):
        root_path = Path(root)
        if not has_negations:
            directories[:] = [
                name
                for name in directories
                if not _is_ignored(
                    (root_path / name).relative_to(context),
                    is_directory=True,
                    rules=rules,
                )
            ]
        for filename in filenames:
            path = root_path / filename
            relative = path.relative_to(context)
            if relative.as_posix() != ".dockerignore" and _is_ignored(
                relative,
                is_directory=False,
                rules=rules,
            ):
                continue
            metadata = path.lstat()
            if stat.S_ISREG(metadata.st_mode):
                files.append(path)
    return tuple(sorted(files, key=lambda path: path.relative_to(context).as_posix()))


def _is_ignored(
    relative: Path,
    *,
    is_directory: bool,
    rules: tuple[_IgnoreRule, ...],
) -> bool:
    value = relative.as_posix()
    parts = PurePosixPath(value).parts
    ignored = False
    for rule in rules:
        if rule.directory_only and not is_directory:
            matches = any(
                _rule_matches(rule, "/".join(parts[: index + 1]), parts[index])
                for index in range(len(parts) - 1)
            )
        else:
            matches = _rule_matches(rule, value, parts[-1]) or (
                "/" not in rule.pattern
                and any(fnmatch.fnmatchcase(part, rule.pattern) for part in parts)
            )
        if matches:
            ignored = not rule.negated
    return ignored


def _rule_matches(rule: _IgnoreRule, value: str, basename: str) -> bool:
    if rule.anchored or "/" in rule.pattern:
        return fnmatch.fnmatchcase(value, rule.pattern)
    return fnmatch.fnmatchcase(basename, rule.pattern)


def _inspect_image(
    reference: str,
    runner: Runner,
    docker_prefix: Sequence[str],
    *,
    required: bool,
) -> dict[str, object] | None:
    args = [*docker_prefix, "image", "inspect", reference]
    result = runner.run(args, check=False)
    if result.returncode != 0:
        if required:
            raise ProjectBuildError(
                f"cannot inspect image {reference}: {_evidence(result)}"
            )
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ProjectBuildError(f"cannot parse image metadata for {reference}") from error
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise ProjectBuildError(f"unexpected image metadata for {reference}")
    return payload[0]


def _image_id(record: Mapping[str, object], reference: str) -> str:
    image_id = record.get("Id")
    if not isinstance(image_id, str) or IMAGE_ID_PATTERN.fullmatch(image_id) is None:
        raise ProjectBuildError(f"image has no immutable ID: {reference}")
    return image_id


def _rootfs_layers(
    record: Mapping[str, object],
    reference: str,
) -> tuple[str, ...]:
    rootfs = record.get("RootFS")
    if not isinstance(rootfs, dict):
        raise ProjectBuildError(f"image has no RootFS metadata: {reference}")
    layers = rootfs.get("Layers")
    if (
        not isinstance(layers, list)
        or not layers
        or any(not isinstance(layer, str) or not layer for layer in layers)
    ):
        raise ProjectBuildError(f"image RootFS layers are invalid: {reference}")
    return tuple(layers)


def _labels(record: Mapping[str, object], reference: str) -> Mapping[str, str]:
    config = record.get("Config")
    if not isinstance(config, dict):
        raise ProjectBuildError(f"image has no config metadata: {reference}")
    labels = config.get("Labels") or {}
    if not isinstance(labels, dict):
        raise ProjectBuildError(f"image labels are invalid: {reference}")
    return MappingProxyType({str(key): str(value) for key, value in labels.items()})


def _environment(config: Mapping[str, object], reference: str) -> Mapping[str, str]:
    raw_environment = config.get("Env") or []
    if not isinstance(raw_environment, list) or any(
        not isinstance(value, str) or "=" not in value
        for value in raw_environment
    ):
        raise ProjectBuildError(f"image environment is invalid: {reference}")
    values = {}
    for assignment in raw_environment:
        name, _, value = assignment.partition("=")
        values[name] = value
    return MappingProxyType(values)


def _verify_project_torch_manifest(
    image: str,
    runner: Runner,
    docker_prefix: Sequence[str],
) -> None:
    result = runner.run(
        list(project_manifest_argv(image, docker_prefix=docker_prefix)),
        check=False,
    )
    if result.returncode != 0:
        raise ProjectBuildError(
            f"project image changed protected Torch files: {_evidence(result)}"
        )


def _required_label(labels: Mapping[str, str], name: str) -> str:
    value = labels.get(name, "")
    if not value:
        raise ProjectBuildError(f"parent image is missing label: {name}")
    return value


def _evidence(result: object) -> str:
    stderr = str(getattr(result, "stderr", "")).strip()
    stdout = str(getattr(result, "stdout", "")).strip()
    return stderr or stdout or "command failed without output"
