from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

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


class ReleaseError(RuntimeError):
    pass


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
