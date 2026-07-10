from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from urllib.parse import unquote, urlsplit

from amd_ai.image.lock import (
    LockError,
    download,
    hash_file,
    parse_package_lock,
    validate_wheelhouse_manifest,
    write_wheelhouse_manifest,
)
from amd_ai.image.profile import ProfileError, TorchProfile, load_profile


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
STABLE_PROFILE = Path("profiles/torch/stable.env")
STABLE_REQUIREMENTS = Path("profiles/torch/stable.requirements.lock")
ROCM_PYTHON_TAG = "rocm-python:7.2.1-py3.12"
STABLE_TORCH_TAG = "rocm-pytorch:7.2.1-py3.12-torch2.9.1"
IMAGE_SOURCE = "https://github.com/hellcatjack/strix-halo-rocm-toolkit"
IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


class BuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocalImage:
    image_id: str
    size: int
    created: datetime
    labels: Mapping[str, str]


class Docker:
    def __init__(self, prefix: Sequence[str]) -> None:
        self.prefix = tuple(prefix)

    @classmethod
    def detect(cls) -> Docker:
        for prefix in (("docker",), ("sudo", "-n", "docker")):
            result = _completed(
                (*prefix, "info", "--format", "{{.ServerVersion}}"),
                check=False,
            )
            if result.returncode == 0:
                docker = cls(prefix)
                docker.capture(("buildx", "version"))
                return docker
        raise BuildError("Docker is unavailable to the current user and sudo -n")

    def capture(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return _completed((*self.prefix, *args), check=check)

    def live(self, args: Sequence[str], *, cwd: Path | None = None) -> None:
        completed = subprocess.run((*self.prefix, *args), check=False, cwd=cwd)
        if completed.returncode != 0:
            raise BuildError(
                f"Docker command failed ({completed.returncode}): {' '.join(args)}"
            )

    def image_id(self, reference: str, *, required: bool = True) -> str | None:
        result = self.capture(
            ("image", "inspect", "--format", "{{.Id}}", reference),
            check=False,
        )
        if result.returncode != 0:
            if required:
                raise BuildError(f"local image does not exist: {reference}")
            return None
        image_id = result.stdout.strip()
        _require_image_id(image_id)
        return image_id

    def supports_attestations(self) -> bool:
        result = self.capture(("info", "--format", "{{json .DriverStatus}}"))
        return driver_supports_attestations(result.stdout)


def immutable_parent_alias(parent: str) -> str:
    _require_image_id(parent)
    return f"amd-ai-local/rocm-python:{parent.removeprefix('sha256:')}"


def driver_supports_attestations(driver_status_json: str) -> bool:
    try:
        status = json.loads(driver_status_json)
    except json.JSONDecodeError:
        return False
    if not isinstance(status, list):
        return False
    return any(
        isinstance(item, list)
        and len(item) == 2
        and item[0] == "driver-type"
        and item[1] == "io.containerd.snapshotter.v1"
        for item in status
    )


def build_torch_argv(
    *,
    profile: TorchProfile,
    parent: str,
    wheelhouse: str | Path,
    revision: str,
    profile_context: str | Path | None = None,
    docker_prefix: Sequence[str] = ("docker",),
    image_source: str = IMAGE_SOURCE,
    tag: str | None = None,
    attestations: bool = True,
    metadata_file: str | Path | None = None,
) -> tuple[str, ...]:
    alias = immutable_parent_alias(parent)
    context = profile_context or Path(".cache/profile-context") / profile.profile_id
    image_tag = tag or torch_image_tag(profile)
    versions = {name: wheel.version for name, wheel in profile.wheels.items()}
    build_args = (
        f"ROCM_PYTHON_BASE={alias}",
        f"PROFILE_ID={profile.profile_id}",
        f"PROFILE_STATUS={profile.status}",
        f"ROCM_VERSION={profile.rocm_version}",
        f"TORCH_VERSION={versions['torch']}",
        f"TORCHVISION_VERSION={versions['torchvision']}",
        f"TORCHAUDIO_VERSION={versions['torchaudio']}",
        f"TRITON_VERSION={versions['triton']}",
        f"VCS_REVISION={revision}",
        f"IMAGE_SOURCE={image_source}",
    )
    argv: list[str] = [
        *docker_prefix,
        "buildx",
        "build",
        "--platform",
        "linux/amd64",
        "--provenance=mode=max" if attestations else "--provenance=false",
        "--sbom=true" if attestations else "--sbom=false",
        "--load",
        "--build-context",
        f"wheels={wheelhouse}",
        "--build-context",
        f"profile-context={context}",
    ]
    if metadata_file is not None:
        argv.extend(("--metadata-file", str(metadata_file)))
    for value in build_args:
        argv.extend(("--build-arg", value))
    argv.extend(
        (
            "--tag",
            image_tag,
            "--file",
            "images/rocm-pytorch/Dockerfile",
            ".",
        )
    )
    return tuple(argv)


def build_rocm_python_argv(
    *,
    ubuntu_base: str,
    uv_image: str,
    revision: str,
    docker_prefix: Sequence[str] = ("docker",),
    image_source: str = IMAGE_SOURCE,
    attestations: bool = True,
    metadata_file: str | Path | None = None,
) -> tuple[str, ...]:
    argv = [
        *docker_prefix,
        "buildx",
        "build",
        "--platform",
        "linux/amd64",
        "--provenance=mode=max" if attestations else "--provenance=false",
        "--sbom=true" if attestations else "--sbom=false",
        "--load",
    ]
    if metadata_file is not None:
        argv.extend(("--metadata-file", str(metadata_file)))
    argv.extend(
        (
            "--build-arg",
            f"UBUNTU_BASE={ubuntu_base}",
            "--build-arg",
            f"UV_IMAGE={uv_image}",
            "--build-arg",
            f"VCS_REVISION={revision}",
            "--build-arg",
            f"IMAGE_SOURCE={image_source}",
            "--tag",
            ROCM_PYTHON_TAG,
            "--file",
            "images/rocm-python/Dockerfile",
            ".",
        )
    )
    return tuple(argv)


def torch_image_tag(profile: TorchProfile) -> str:
    if profile.status == "verified":
        return STABLE_TORCH_TAG
    version = _docker_tag_component(profile.wheels["torch"].version)
    profile_id = _docker_tag_component(profile.profile_id)
    prefix = f"7.2.1-py3.12-torch{version}-"
    available = 128 - len(prefix)
    if len(profile_id) > available:
        digest = hashlib.sha256(profile_id.encode()).hexdigest()[:12]
        profile_id = f"{profile_id[: available - 13]}-{digest}"
    return f"rocm-pytorch:{prefix}{profile_id}"


def materialize_profile_context(
    profile_path: Path,
    requirements_lock: Path,
    destination: Path,
) -> None:
    if not profile_path.is_file():
        raise BuildError(f"profile does not exist: {profile_path}")
    if not requirements_lock.is_file():
        raise BuildError(f"requirements lock does not exist: {requirements_lock}")
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise BuildError(f"profile context is not a directory: {destination}")
        for child in destination.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        destination.mkdir(parents=True)
    _write_text(destination / "profile.env", profile_path.read_text(encoding="utf-8"))
    _write_text(
        destination / "requirements.lock",
        requirements_lock.read_text(encoding="utf-8"),
    )


def project_base_image_ids(project_roots: Iterable[Path]) -> frozenset[str]:
    protected: set[str] = set()
    for root in project_roots:
        if not root.exists():
            continue
        paths = (root,) if root.name == "amd-ai-project.toml" else root.rglob(
            "amd-ai-project.toml"
        )
        for path in paths:
            try:
                with path.open("rb") as stream:
                    payload = tomllib.load(stream)
                project = payload.get("project", {})
            except (OSError, tomllib.TOMLDecodeError) as error:
                raise BuildError(f"cannot parse project reference {path}: {error}") from error
            if not isinstance(project, dict):
                raise BuildError(f"invalid [project] table in {path}")
            for key in ("base_image", "base_digest"):
                value = project.get(key)
                if value is None:
                    continue
                if not isinstance(value, str) or IMAGE_ID_PATTERN.fullmatch(value) is None:
                    raise BuildError(f"invalid {key} in {path}")
                protected.add(value)
    return frozenset(protected)


def select_prunable_images(
    images: Iterable[LocalImage],
    *,
    protected_ids: Iterable[str],
    cutoff: datetime,
) -> tuple[LocalImage, ...]:
    protected = set(protected_ids)
    selected = []
    for image in images:
        managed = (
            "org.amd-ai.profile.id" in image.labels
            or "org.amd-ai.project.fingerprint" in image.labels
        )
        if managed and image.image_id not in protected and image.created <= cutoff:
            selected.append(image)
    return tuple(sorted(selected, key=lambda image: image.image_id))


def build_rocm_python(*, repo_root: Path = REPOSITORY_ROOT) -> tuple[str, str]:
    repo_root = repo_root.resolve()
    locks = _validate_rocm_locks(repo_root)
    docker = Docker.detect()
    revision = _git_revision(repo_root)
    attestations = _attestation_mode(docker)
    metadata = _metadata_path(repo_root, ROCM_PYTHON_TAG)
    argv = build_rocm_python_argv(
        ubuntu_base=locks["UBUNTU_24_04"],
        uv_image=locks["UV_IMAGE"],
        revision=revision,
        docker_prefix=docker.prefix,
        image_source=IMAGE_SOURCE,
        attestations=attestations,
        metadata_file=metadata.relative_to(repo_root),
    )
    _run_live(argv, cwd=repo_root)
    metadata_digest = _validate_build_metadata(metadata)
    image_id = docker.image_id(ROCM_PYTHON_TAG)
    assert image_id is not None
    if metadata_digest != image_id:
        raise BuildError("base image ID does not match its BuildKit metadata")
    check = docker.capture(
        (
            "run",
            "--rm",
            ROCM_PYTHON_TAG,
            "container-check",
            "--mode",
            "rocm",
            "--metadata-only",
            "--json",
            "-",
        )
    )
    print(check.stdout, end="")
    return ROCM_PYTHON_TAG, image_id


def build_rocm_pytorch(
    *,
    profile_path: Path,
    allow_experimental: bool,
    repo_root: Path = REPOSITORY_ROOT,
) -> tuple[str, str]:
    repo_root = repo_root.resolve()
    profile_path = _resolve_input_path(profile_path, repo_root)
    stable = (repo_root / STABLE_PROFILE).resolve()
    allow_verified = profile_path == stable
    try:
        profile = load_profile(profile_path, allow_verified=allow_verified)
    except (OSError, ProfileError) as error:
        raise BuildError(f"invalid Torch profile: {error}") from error
    if profile.status == "experimental" and not allow_experimental:
        raise BuildError("experimental profile requires --allow-experimental")

    wheelhouse, requirements = _prepare_profile_artifacts(
        profile=profile,
        profile_path=profile_path,
        repo_root=repo_root,
        stable=stable,
    )
    _validate_profile_artifacts(profile, wheelhouse, requirements)
    context_key = _profile_cache_key(profile_path, profile)
    profile_context = repo_root / ".cache/profile-context" / context_key
    materialize_profile_context(profile_path, requirements, profile_context)

    _validate_rocm_locks(repo_root)
    docker = Docker.detect()
    parent = docker.image_id(ROCM_PYTHON_TAG)
    assert parent is not None
    alias = immutable_parent_alias(parent)
    docker.live(("tag", parent, alias))
    if docker.image_id(alias) != parent:
        raise BuildError("content-addressed local parent alias changed unexpectedly")

    tag = torch_image_tag(profile)
    attestations = _attestation_mode(docker)
    metadata = _metadata_path(repo_root, tag)
    argv = build_torch_argv(
        profile=profile,
        parent=parent,
        wheelhouse=wheelhouse.relative_to(repo_root),
        profile_context=profile_context.relative_to(repo_root),
        revision=_git_revision(repo_root),
        docker_prefix=docker.prefix,
        image_source=IMAGE_SOURCE,
        tag=tag,
        attestations=attestations,
        metadata_file=metadata.relative_to(repo_root),
    )
    _run_live(argv, cwd=repo_root)
    metadata_digest = _validate_build_metadata(metadata)
    if docker.image_id(alias) != parent:
        raise BuildError("parent alias changed during the image build")
    image_id = docker.image_id(tag)
    assert image_id is not None
    if metadata_digest != image_id:
        raise BuildError("Torch image ID does not match its BuildKit metadata")
    _verify_profile_labels(docker, tag, profile)
    check = docker.capture(
        (
            "run",
            "--rm",
            tag,
            "container-check",
            "--mode",
            "torch",
            "--metadata-only",
            "--json",
            "-",
        )
    )
    print(check.stdout, end="")
    return tag, image_id


def prune_images(
    *,
    apply: bool,
    older_than_hours: int,
    project_roots: Sequence[Path] | None = None,
    repo_root: Path = REPOSITORY_ROOT,
) -> tuple[LocalImage, ...]:
    if older_than_hours <= 0:
        raise BuildError("older-than-hours must be positive")
    repo_root = repo_root.resolve()
    roots = tuple(
        project_roots
        or default_project_roots(repo_root=repo_root, current_dir=Path.cwd())
    )
    protected = set(project_base_image_ids(roots))
    docker = Docker.detect()
    for reference in (ROCM_PYTHON_TAG, STABLE_TORCH_TAG):
        image_id = docker.image_id(reference, required=False)
        if image_id is not None:
            protected.add(image_id)
    protected.update(_running_image_ids(docker))
    images = _local_images(docker)
    cutoff = datetime.now(UTC) - timedelta(hours=older_than_hours)
    selected = select_prunable_images(images, protected_ids=protected, cutoff=cutoff)
    for image in selected:
        print(f"{image.image_id} {image.size} bytes")
    if not selected:
        print("No unreferenced managed images matched the age filter.")
    if apply:
        for image in selected:
            docker.live(("image", "rm", image.image_id))
        docker.live(
            (
                "buildx",
                "prune",
                "--force",
                "--filter",
                f"until={older_than_hours}h",
            )
        )
    else:
        print("Preview only; pass --apply to remove exactly the listed image IDs.")
    return selected


def run_image_check(
    *,
    image: str,
    mode: str,
    metadata_only: bool,
    runtime: bool,
    json_path: str | None,
) -> int:
    if (
        not image
        or image.startswith("-")
        or "\0" in image
        or any(character.isspace() for character in image)
    ):
        raise BuildError("container-check image reference is invalid")
    if metadata_only and runtime:
        raise BuildError("metadata-only and runtime checks are mutually exclusive")
    docker = Docker.detect()
    args: list[str] = ["run", "--rm", "--ipc=private", "--shm-size=1g"]
    if runtime or not metadata_only:
        for device in (Path("/dev/kfd"), Path("/dev/dri")):
            if device.exists():
                args.extend(("--device", str(device)))
        gids = {
            os.stat(device).st_gid
            for device in (Path("/dev/kfd"), Path("/dev/dri/renderD128"))
            if device.exists()
        }
        for gid in sorted(gids):
            args.extend(("--group-add", str(gid)))
    args.extend((image, "container-check", "--mode", mode))
    if metadata_only:
        args.append("--metadata-only")
    if runtime:
        args.append("--runtime")
    args.extend(("--json", "-"))
    result = docker.capture(args, check=False)
    if json_path == "-" or json_path is None:
        print(result.stdout, end="")
    else:
        _write_text(Path(json_path), result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    return result.returncode


def default_project_roots(
    *,
    repo_root: Path,
    current_dir: Path,
) -> tuple[Path, ...]:
    roots = (current_dir.resolve(), (repo_root / "projects").resolve())
    return tuple(dict.fromkeys(roots))


def _prepare_profile_artifacts(
    *,
    profile: TorchProfile,
    profile_path: Path,
    repo_root: Path,
    stable: Path,
) -> tuple[Path, Path]:
    if profile_path == stable:
        return (
            repo_root / ".cache/wheels" / profile.profile_id,
            repo_root / STABLE_REQUIREMENTS,
        )
    return _lock_experimental_profile(profile, profile_path, repo_root)


def _lock_experimental_profile(
    profile: TorchProfile,
    profile_path: Path,
    repo_root: Path,
) -> tuple[Path, Path]:
    cache_key = _profile_cache_key(profile_path, profile)
    wheelhouse = repo_root / ".cache/wheels" / cache_key
    requirements = repo_root / ".cache/locks" / f"{cache_key}.requirements.lock"
    for name, wheel in profile.wheels.items():
        destination = wheelhouse / _wheel_filename(wheel.url)
        print(f"locking experimental {name}: {destination.name}", file=sys.stderr)
        download(wheel.url, destination, wheel.sha256)

    if requirements.is_file():
        errors = validate_wheelhouse_manifest(wheelhouse)
        if not errors:
            try:
                _validate_profile_artifacts(profile, wheelhouse, requirements)
                return wheelhouse, requirements
            except BuildError:
                pass

    lock_input = repo_root / ".cache/locks" / f"{cache_key}.in"
    primary_versions = []
    for name, wheel in profile.wheels.items():
        full_version = _wheel_distribution_version(name, _wheel_filename(wheel.url))
        if wheel.version not in (full_version, full_version.split("+", 1)[0]):
            raise BuildError(
                f"{name} profile version {wheel.version} does not match wheel {full_version}"
            )
        primary_versions.append(f"{name}=={full_version}")
    _write_text(lock_input, "\n".join(primary_versions) + "\n")
    requirements.parent.mkdir(parents=True, exist_ok=True)
    _run_live(
        (
            "uv",
            "pip",
            "compile",
            "--python-version",
            "3.12",
            "--python-platform",
            "x86_64-unknown-linux-gnu",
            "--find-links",
            str(wheelhouse),
            "--generate-hashes",
            "--output-file",
            str(requirements),
            str(lock_input),
        ),
        cwd=repo_root,
    )
    _run_live(
        (
            "python3.12",
            "-m",
            "pip",
            "download",
            "--require-hashes",
            "--dest",
            str(wheelhouse),
            "--requirement",
            str(requirements),
            "--find-links",
            str(wheelhouse),
        ),
        cwd=repo_root,
    )
    write_wheelhouse_manifest(wheelhouse)
    return wheelhouse, requirements


def _validate_profile_artifacts(
    profile: TorchProfile,
    wheelhouse: Path,
    requirements: Path,
) -> None:
    errors = validate_wheelhouse_manifest(wheelhouse)
    if errors:
        raise BuildError("invalid wheelhouse: " + "; ".join(errors))
    try:
        lock_text = requirements.read_text(encoding="utf-8")
    except OSError as error:
        raise BuildError(f"cannot read requirements lock: {error}") from error
    for name, wheel in profile.wheels.items():
        filename = _wheel_filename(wheel.url)
        path = wheelhouse / filename
        if not path.is_file() or hash_file(path) != wheel.sha256:
            raise BuildError(f"profile wheel is missing or changed: {filename}")
        full_version = _wheel_distribution_version(name, filename)
        match = re.search(rf"^{re.escape(name)}==([^\s\\]+)", lock_text, re.MULTILINE)
        if match is None or match.group(1) != full_version:
            resolved = match.group(1) if match else "<missing>"
            raise BuildError(
                f"requirements lock has {name}={resolved}, expected {full_version}"
            )


def _validate_rocm_locks(repo_root: Path) -> Mapping[str, str]:
    base_path = repo_root / "profiles/base-images.lock"
    try:
        values = _parse_key_value_lock(base_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise BuildError(f"cannot read base image lock: {error}") from error
    if set(values) != {"UBUNTU_24_04", "UV_IMAGE", "UV_VERSION"}:
        raise BuildError("base image lock has missing or unknown keys")
    for key in ("UBUNTU_24_04", "UV_IMAGE"):
        if re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", values[key]) is None:
            raise BuildError(f"invalid digest-pinned base image: {key}")
    if values["UV_VERSION"] != "0.11.28":
        raise BuildError("unexpected uv version in base image lock")

    package_path = repo_root / "profiles/rocm/7.2.1-packages.lock"
    try:
        parse_package_lock(package_path.read_text(encoding="utf-8"))
    except (OSError, LockError) as error:
        raise BuildError(f"invalid ROCm package lock: {error}") from error
    keyring = repo_root / "profiles/rocm/rocm.gpg"
    checksum_path = repo_root / "profiles/rocm/rocm.gpg.sha256"
    try:
        checksum = checksum_path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise BuildError(f"cannot read ROCm key checksum: {error}") from error
    match = re.fullmatch(r"([0-9a-f]{64})  rocm\.gpg", checksum)
    if match is None or not keyring.is_file() or hash_file(keyring) != match.group(1):
        raise BuildError("ROCm keyring does not match its SHA-256 lock")
    return MappingProxyType(values)


def _attestation_mode(docker: Docker) -> bool:
    if docker.supports_attestations():
        return True
    print(
        "warning: the Docker classic image store cannot retain provenance/SBOM "
        "attestations; writing max build metadata and loading the runnable image "
        "without attached attestations",
        file=sys.stderr,
    )
    return False


def _metadata_path(repo_root: Path, tag: str) -> Path:
    filename = re.sub(r"[^A-Za-z0-9_.-]", "-", tag) + ".json"
    path = repo_root / ".cache/build-metadata" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    return path


def _validate_build_metadata(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BuildError(f"invalid BuildKit metadata file {path}: {error}") from error
    config_digest = payload.get("containerimage.config.digest")
    if not isinstance(config_digest, str) or IMAGE_ID_PATTERN.fullmatch(
        config_digest
    ) is None:
        raise BuildError("BuildKit metadata has no immutable config digest")
    provenance = payload.get("buildx.build.provenance")
    if not isinstance(provenance, dict) or not {
        "buildType",
        "builder",
        "invocation",
        "materials",
    } <= provenance.keys():
        raise BuildError("BuildKit metadata has no max build provenance record")
    return config_digest


def _verify_profile_labels(docker: Docker, tag: str, profile: TorchProfile) -> None:
    result = docker.capture(
        ("image", "inspect", "--format", "{{json .Config.Labels}}", tag)
    )
    try:
        labels = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise BuildError(f"cannot parse image labels for {tag}") from error
    expected = {
        "org.amd-ai.profile.id": profile.profile_id,
        "org.amd-ai.profile.status": profile.status,
        "org.amd-ai.rocm.version": profile.rocm_version,
        "org.amd-ai.torch.version": profile.wheels["torch"].version,
    }
    for key, value in expected.items():
        if labels.get(key) != value:
            raise BuildError(f"image label mismatch: {key}")


def _local_images(docker: Docker) -> tuple[LocalImage, ...]:
    listed = docker.capture(("image", "ls", "--quiet", "--no-trunc"))
    image_ids = sorted(set(listed.stdout.split()))
    if not image_ids:
        return ()
    inspected = docker.capture(("image", "inspect", *image_ids))
    try:
        payload = json.loads(inspected.stdout)
    except json.JSONDecodeError as error:
        raise BuildError("cannot parse local Docker image metadata") from error
    images = []
    for record in payload:
        image_id = record.get("Id", "")
        _require_image_id(image_id)
        labels = (record.get("Config") or {}).get("Labels") or {}
        if not isinstance(labels, dict):
            labels = {}
        images.append(
            LocalImage(
                image_id=image_id,
                size=int(record.get("Size", 0)),
                created=_parse_docker_datetime(record.get("Created", "")),
                labels=MappingProxyType(
                    {str(key): str(value) for key, value in labels.items()}
                ),
            )
        )
    return tuple(images)


def _running_image_ids(docker: Docker) -> frozenset[str]:
    containers = docker.capture(("ps", "--quiet")).stdout.split()
    image_ids: set[str] = set()
    for container in containers:
        result = docker.capture(("inspect", "--format", "{{.Image}}", container))
        image_id = result.stdout.strip()
        _require_image_id(image_id)
        image_ids.add(image_id)
    return frozenset(image_ids)


def _parse_key_value_lock(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        key, separator, value = line.partition("=")
        if not separator or not key or not value or key in values:
            raise BuildError(f"invalid lock entry on line {line_number}")
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", key) is None or any(
            character.isspace() for character in value
        ):
            raise BuildError(f"invalid lock value on line {line_number}")
        values[key] = value
    return values


def _profile_cache_key(profile_path: Path, profile: TorchProfile) -> str:
    if profile.status == "verified":
        return profile.profile_id
    digest = hashlib.sha256(profile_path.read_bytes()).hexdigest()[:16]
    return f"experimental-{profile.profile_id}-{digest}"


def _resolve_input_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    if path.exists():
        return path.resolve()
    return (repo_root / path).resolve()


def _git_revision(repo_root: Path) -> str:
    result = _completed(
        ("git", "-C", str(repo_root), "rev-parse", "HEAD"),
        check=False,
    )
    revision = result.stdout.strip()
    return revision if re.fullmatch(r"[0-9a-f]{40}", revision) else "unknown"


def _wheel_filename(url: str) -> str:
    filename = unquote(Path(urlsplit(url).path).name)
    if not filename.endswith(".whl") or Path(filename).name != filename:
        raise BuildError(f"URL does not identify a wheel: {url}")
    return filename


def _wheel_distribution_version(name: str, filename: str) -> str:
    fields = filename.removesuffix(".whl").split("-")
    if len(fields) < 5 or fields[0].replace("_", "-").lower() != name.replace(
        "_", "-"
    ).lower():
        raise BuildError(f"wheel filename does not match {name}: {filename}")
    return fields[1]


def _docker_tag_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "-", value).strip(".-")
    if not sanitized:
        raise BuildError(f"cannot use value in Docker tag: {value!r}")
    return sanitized


def _require_image_id(value: str) -> None:
    if IMAGE_ID_PATTERN.fullmatch(value) is None:
        raise BuildError(f"invalid local immutable image ID: {value!r}")


def _parse_docker_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise BuildError(f"invalid Docker image timestamp: {value!r}") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _run_live(argv: Sequence[str], *, cwd: Path) -> None:
    environment = os.environ.copy()
    environment["BUILDX_METADATA_PROVENANCE"] = "max"
    completed = subprocess.run(
        tuple(argv),
        check=False,
        cwd=cwd,
        env=environment,
    )
    if completed.returncode != 0:
        raise BuildError(f"command failed ({completed.returncode}): {' '.join(argv)}")


def _completed(
    argv: Sequence[str],
    *,
    check: bool,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        tuple(argv),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and completed.returncode != 0:
        evidence = completed.stderr.strip() or completed.stdout.strip()
        raise BuildError(
            f"command failed ({completed.returncode}): {' '.join(argv)}: {evidence}"
        )
    return completed


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
