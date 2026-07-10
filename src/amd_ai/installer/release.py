from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

from amd_ai.installer.models import ReleaseImage, StableRelease


DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
SEMVER_PATTERN = re.compile(
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)"
    r"(?:-(?:0|[1-9][0-9]*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)
GHCR_IMAGE_PATTERN = re.compile(
    r"ghcr\.io/"
    r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?/"
    r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?"
)
REPOSITORY_PATTERN = re.compile(
    r"https://github\.com/"
    r"[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?"
)
ADAPTER_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")

ROOT_KEYS = frozenset(
    {
        "schema_version",
        "release_id",
        "source_repository",
        "source_revision",
        "qualification_profile_digest",
        "qualification_report_digest",
        "sbom_digest",
        "gpu_arch",
        "supported_host_adapter_ids",
        "rocm_version",
        "python_version",
        "torch_version",
        "torch_profile_id",
        "torch_profile_digest",
        "base",
        "torch",
        "published_at",
    }
)
IMAGE_KEYS = frozenset(
    {"image", "manifest_digest", "config_digest", "artifact_digests"}
)
BASE_ARTIFACT_KEYS = frozenset({"rocm_keyring", "rocm_packages_lock"})
TORCH_ARTIFACT_KEYS = frozenset(
    {"profile", "requirements_lock", "torch_manifest"}
)
BASE_ARTIFACT_PATHS = {
    "rocm_keyring": "/etc/apt/keyrings/rocm.gpg",
    "rocm_packages_lock": "/opt/amd-ai/locks/rocm-packages.lock",
}
TORCH_ARTIFACT_PATHS = {
    "profile": "/opt/amd-ai/profile.env",
    "requirements_lock": "/opt/amd-ai/profile.requirements.lock",
    "torch_manifest": "/opt/amd-ai/torch-manifest.json",
}


class ReleaseError(RuntimeError):
    pass


class ReleaseAcquisitionError(ReleaseError):
    pass


class ReleaseIdentityError(ReleaseError):
    pass


class ReleaseDocker(Protocol):
    def pull(self, reference: str) -> None:
        pass

    def inspect(self, reference: str) -> Mapping[str, object]:
        pass

    def hash_file(self, reference: str, path: str) -> str:
        pass


@dataclass(frozen=True)
class VerifiedImageIdentity:
    reference: str
    config_digest: str
    repo_digests: tuple[str, ...]
    labels: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_digests", tuple(self.repo_digests))
        object.__setattr__(
            self, "labels", MappingProxyType(dict(self.labels))
        )


@dataclass(frozen=True)
class VerifiedReleaseImages:
    base: VerifiedImageIdentity
    torch: VerifiedImageIdentity


def load_stable_release(path: Path) -> StableRelease:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ReleaseError(f"cannot read stable release manifest: {error}") from error
    try:
        payload = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise ReleaseError(f"cannot parse stable release manifest: {error}") from error
    if not isinstance(payload, dict):
        raise ReleaseError("stable release manifest must be an object")
    _require_keys("release", payload, ROOT_KEYS)

    schema_version = payload["schema_version"]
    if type(schema_version) is not int or schema_version != 1:
        raise ReleaseError("stable release schema_version must be integer 1")
    release_id = _require_string(payload, "release_id")
    if SEMVER_PATTERN.fullmatch(release_id) is None:
        raise ReleaseError("stable release ID is not semantic versioning")
    source_repository = _require_string(payload, "source_repository")
    if REPOSITORY_PATTERN.fullmatch(source_repository) is None:
        raise ReleaseError("stable release source repository is invalid")
    source_revision = _require_string(payload, "source_revision")
    if REVISION_PATTERN.fullmatch(source_revision) is None:
        raise ReleaseError("stable release source revision is invalid")

    qualification_profile_digest = _require_digest(
        payload, "qualification_profile_digest"
    )
    qualification_report_digest = _require_digest(
        payload, "qualification_report_digest"
    )
    sbom_digest = _require_digest(payload, "sbom_digest")
    _require_exact(payload, "gpu_arch", "gfx1151")
    _require_exact(payload, "rocm_version", "7.2.1")
    _require_exact(payload, "python_version", "3.12")
    _require_exact(payload, "torch_version", "2.9.1")
    torch_profile_id = _require_string(payload, "torch_profile_id")
    if torch_profile_id != "rocm-7.2.1-py3.12-torch-2.9.1":
        raise ReleaseError("stable release Torch profile ID is invalid")
    torch_profile_digest = _require_digest(payload, "torch_profile_digest")

    adapters = payload["supported_host_adapter_ids"]
    if (
        not isinstance(adapters, list)
        or not adapters
        or any(
            not isinstance(adapter, str)
            or ADAPTER_PATTERN.fullmatch(adapter) is None
            for adapter in adapters
        )
    ):
        raise ReleaseError("stable release adapter IDs are invalid")
    if len(set(adapters)) != len(adapters):
        raise ReleaseError("stable release adapter IDs contain duplicates")

    published_at = _require_string(payload, "published_at")
    try:
        parsed_timestamp = datetime.strptime(
            published_at, "%Y-%m-%dT%H:%M:%SZ"
        )
    except ValueError as error:
        raise ReleaseError(
            "stable release published_at must be UTC with second precision"
        ) from error
    if parsed_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ") != published_at:
        raise ReleaseError("stable release published_at is not canonical UTC")

    base = _parse_image(
        payload["base"],
        label="base",
        artifact_keys=BASE_ARTIFACT_KEYS,
    )
    torch = _parse_image(
        payload["torch"],
        label="torch",
        artifact_keys=TORCH_ARTIFACT_KEYS,
    )
    if torch_profile_digest != torch.artifact_digests["profile"]:
        raise ReleaseError(
            "Torch profile digest differs from embedded profile artifact"
        )
    return StableRelease(
        schema_version=schema_version,
        release_id=release_id,
        source_repository=source_repository,
        source_revision=source_revision,
        qualification_profile_digest=qualification_profile_digest,
        qualification_report_digest=qualification_report_digest,
        sbom_digest=sbom_digest,
        gpu_arch="gfx1151",
        supported_host_adapter_ids=tuple(adapters),
        rocm_version="7.2.1",
        python_version="3.12",
        torch_version="2.9.1",
        torch_profile_id=torch_profile_id,
        torch_profile_digest=torch_profile_digest,
        base=base,
        torch=torch,
        published_at=published_at,
    )


def verify_release_image(
    release: StableRelease,
    image: ReleaseImage,
    *,
    kind: Literal["base", "torch"],
    docker: ReleaseDocker,
) -> VerifiedImageIdentity:
    if kind not in {"base", "torch"}:
        raise ReleaseIdentityError(f"unsupported release image kind: {kind}")
    expected_image = release.base if kind == "base" else release.torch
    if image != expected_image:
        raise ReleaseIdentityError(
            f"stable release {kind} verifier received the wrong image identity"
        )
    try:
        record = docker.inspect(image.reference)
    except Exception as error:
        raise ReleaseIdentityError(
            f"cannot inspect exact release image {image.reference}: {error}"
        ) from error
    if not isinstance(record, Mapping):
        raise ReleaseIdentityError(
            f"release {kind} image inspect record is invalid"
        )
    config_digest = record.get("Id")
    if config_digest != image.config_digest:
        raise ReleaseIdentityError(
            f"release {kind} config digest does not match: {config_digest}"
        )
    raw_repo_digests = record.get("RepoDigests")
    if (
        not isinstance(raw_repo_digests, list)
        or any(not isinstance(value, str) for value in raw_repo_digests)
        or image.reference not in raw_repo_digests
    ):
        raise ReleaseIdentityError(
            f"release {kind} RepoDigest does not include {image.reference}"
        )
    raw_config = record.get("Config")
    if not isinstance(raw_config, Mapping):
        raise ReleaseIdentityError(f"release {kind} image config is invalid")
    raw_labels = raw_config.get("Labels")
    if not isinstance(raw_labels, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, str)
        for name, value in raw_labels.items()
    ):
        raise ReleaseIdentityError(f"release {kind} image labels are invalid")
    labels = dict(raw_labels)
    expected_labels = {
        "org.opencontainers.image.source": release.source_repository,
        "org.opencontainers.image.revision": release.source_revision,
        "org.amd-ai.rocm.version": release.rocm_version,
        "org.amd-ai.python.version": release.python_version,
    }
    artifact_paths = BASE_ARTIFACT_PATHS
    if kind == "torch":
        expected_labels.update(
            {
                "org.amd-ai.profile.id": release.torch_profile_id,
                "org.amd-ai.profile.status": "verified",
                "org.amd-ai.torch.version": release.torch_version,
            }
        )
        artifact_paths = TORCH_ARTIFACT_PATHS
    for name, expected in expected_labels.items():
        if labels.get(name) != expected:
            raise ReleaseIdentityError(
                f"release {kind} image label {name} does not match"
            )
    for name, path in artifact_paths.items():
        try:
            actual = docker.hash_file(image.reference, path)
        except Exception as error:
            raise ReleaseIdentityError(
                f"cannot hash release {kind} artifact {path}: {error}"
            ) from error
        expected = image.artifact_digests[name]
        if actual != expected:
            raise ReleaseIdentityError(
                f"release {kind} artifact digest does not match: {path}"
            )
    return VerifiedImageIdentity(
        reference=image.reference,
        config_digest=config_digest,
        repo_digests=tuple(raw_repo_digests),
        labels=labels,
    )


def pull_and_verify_release(
    release: StableRelease, *, docker: ReleaseDocker
) -> VerifiedReleaseImages:
    identities: list[VerifiedImageIdentity] = []
    for kind, image in (("base", release.base), ("torch", release.torch)):
        try:
            docker.pull(image.reference)
        except Exception as error:
            raise ReleaseAcquisitionError(
                f"cannot pull exact {kind} release image {image.reference}: {error}"
            ) from error
        identities.append(
            verify_release_image(
                release,
                image,
                kind=kind,
                docker=docker,
            )
        )
    return VerifiedReleaseImages(base=identities[0], torch=identities[1])


def _parse_image(
    raw: object,
    *,
    label: str,
    artifact_keys: frozenset[str],
) -> ReleaseImage:
    if not isinstance(raw, dict):
        raise ReleaseError(f"stable release {label} image must be an object")
    _require_keys(f"{label} image", raw, IMAGE_KEYS)
    image = _require_string(raw, "image")
    if GHCR_IMAGE_PATTERN.fullmatch(image) is None:
        raise ReleaseError(
            f"stable release {label} image must be an untagged GHCR name"
        )
    manifest_digest = _require_digest(raw, "manifest_digest")
    config_digest = _require_digest(raw, "config_digest")
    if manifest_digest == config_digest:
        raise ReleaseError(
            f"stable release {label} manifest and config digests are ambiguous"
        )
    artifacts = raw["artifact_digests"]
    if not isinstance(artifacts, dict):
        raise ReleaseError(
            f"stable release {label} artifact digests must be an object"
        )
    _require_keys(f"{label} artifact digests", artifacts, artifact_keys)
    parsed_artifacts = {
        name: _require_digest(artifacts, name) for name in sorted(artifact_keys)
    }
    return ReleaseImage(
        image=image,
        manifest_digest=manifest_digest,
        config_digest=config_digest,
        artifact_digests=parsed_artifacts,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, value in pairs:
        if name in values:
            raise ReleaseError(f"duplicate JSON key in stable release: {name}")
        values[name] = value
    return values


def _require_keys(
    label: str, values: dict[str, Any], expected: frozenset[str]
) -> None:
    actual = set(values)
    missing = sorted(expected.difference(actual))
    unknown = sorted(actual.difference(expected))
    if missing:
        raise ReleaseError(f"stable {label} is missing keys: {', '.join(missing)}")
    if unknown:
        raise ReleaseError(f"stable {label} has unknown keys: {', '.join(unknown)}")


def _require_string(values: dict[str, Any], name: str) -> str:
    value = values[name]
    if not isinstance(value, str) or not value or "\0" in value:
        raise ReleaseError(f"stable release field {name} must be a string")
    return value


def _require_digest(values: dict[str, Any], name: str) -> str:
    value = _require_string(values, name)
    if DIGEST_PATTERN.fullmatch(value) is None:
        raise ReleaseError(f"stable release digest {name} is invalid")
    return value


def _require_exact(values: dict[str, Any], name: str, expected: str) -> None:
    value = _require_string(values, name)
    if value != expected:
        raise ReleaseError(
            f"stable release {name} must be {expected}, got {value}"
        )
