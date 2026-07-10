from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from amd_ai.image.build import IMAGE_SOURCE
from amd_ai.installer.release import SEMVER_PATTERN


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
QUALIFICATION_PROFILE = Path("profiles/qualification/stable.toml")
TORCH_PROFILE = Path("profiles/torch/stable.env")
TORCH_REQUIREMENTS_LOCK = Path("profiles/torch/stable.requirements.lock")
ROCM_PACKAGE_LOCK = Path("profiles/rocm/7.2.1-packages.lock")
ROCM_KEYRING = Path("profiles/rocm/rocm.gpg")
STABLE_TORCH_IMAGE = "rocm-pytorch:7.2.1-py3.12-torch2.9.1"
STABLE_TORCH_PROFILE = "rocm-7.2.1-py3.12-torch-2.9.1"
VERIFIED_TAG = STABLE_TORCH_IMAGE + "-gfx1151-verified"
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
REPO_DIGEST_PATTERN = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")
UTC_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")

REQUIRED_QUALIFICATION_CHECKS = frozenset(
    {
        "rocm",
        "torch-fp16",
        "hip",
        "torch-extension",
        "triton",
        "repeated-start",
        "stress",
        "kernel-log",
    }
)
RELEASE_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "generated_at",
        "profile_id",
        "gpu_arch",
        "qualification_digest",
        "profile_digest",
        "design_digest",
        "image_reference",
        "image_id",
        "repo_digest",
        "image_labels",
        "wheel_hashes",
        "rocm_package_lock_digest",
        "torch_profile_digest",
        "sbom_digest",
        "git_revision",
        "verified_tag",
        "qualification_file",
        "sbom_file",
    }
)
QUALIFICATION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "generated_at",
        "profile_id",
        "profile_digest",
        "image",
        "image_id",
        "gpu_arch",
        "required_checks",
        "results",
    }
)


class PublishError(RuntimeError):
    pass


@dataclass(frozen=True)
class PublishCandidate:
    release_id: str
    source_repository: str
    source_revision: str
    base_local_id: str | None
    torch_local_id: str
    gpu_arch: str
    supported_host_adapter_ids: tuple[str, ...]
    qualification_digest: str
    qualification_run_digest: str
    qualification_profile_digest: str
    sbom_digest: str
    torch_profile_id: str
    torch_profile_digest: str
    base_artifact_digests: Mapping[str, str]
    torch_artifact_digests: Mapping[str, str]
    image_labels: Mapping[str, str]
    published_at: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "supported_host_adapter_ids",
            tuple(self.supported_host_adapter_ids),
        )
        for field in (
            "base_artifact_digests",
            "torch_artifact_digests",
            "image_labels",
        ):
            object.__setattr__(
                self,
                field,
                MappingProxyType(dict(getattr(self, field))),
            )


def validate_publish_inputs(
    *,
    release_id: str,
    qualification_path: Path,
    sbom_path: Path,
    current_revision: str,
    torch_image_id: str,
    base_image_id: str | None = None,
    repo_root: Path = REPOSITORY_ROOT,
) -> PublishCandidate:
    if SEMVER_PATTERN.fullmatch(release_id) is None:
        raise PublishError(f"release ID is not semantic versioning: {release_id}")
    if REVISION_PATTERN.fullmatch(current_revision) is None:
        raise PublishError("current source revision is invalid")
    _require_image_id(torch_image_id, "Torch")
    if base_image_id is not None:
        _require_image_id(base_image_id, "base")

    repo_root = repo_root.resolve(strict=True)
    release_bytes, report = _load_json_object(
        qualification_path, "qualified release report"
    )
    _require_keys("qualified release report", report, RELEASE_REPORT_KEYS)
    if type(report["schema_version"]) is not int or report["schema_version"] != 1:
        raise PublishError("qualified release report schema is invalid")
    _require_equal(report, "status", "verified")
    _require_equal(report, "profile_id", "stable-gfx1151")
    _require_equal(report, "gpu_arch", "gfx1151")
    _require_equal(report, "image_reference", STABLE_TORCH_IMAGE)
    _require_equal(report, "verified_tag", VERIFIED_TAG)
    _require_equal(report, "git_revision", current_revision)
    _require_equal(report, "image_id", torch_image_id)
    generated_at = _require_string(report, "generated_at")
    if UTC_PATTERN.fullmatch(generated_at) is None:
        raise PublishError("qualified release timestamp is not canonical UTC")

    expected_profile_digest = _hash_file(repo_root / QUALIFICATION_PROFILE)
    expected_torch_profile_digest = _hash_file(repo_root / TORCH_PROFILE)
    expected_rocm_lock_digest = _hash_file(repo_root / ROCM_PACKAGE_LOCK)
    _require_equal(report, "profile_digest", expected_profile_digest)
    _require_equal(
        report, "torch_profile_digest", expected_torch_profile_digest
    )
    _require_equal(
        report, "rocm_package_lock_digest", expected_rocm_lock_digest
    )
    _require_hex_digest(report, "design_digest")

    repo_digest = report["repo_digest"]
    if repo_digest is not None and (
        not isinstance(repo_digest, str)
        or REPO_DIGEST_PATTERN.fullmatch(repo_digest) is None
    ):
        raise PublishError("qualified release RepoDigest is invalid")
    labels = _require_string_mapping(report, "image_labels")
    for name, expected in {
        "org.opencontainers.image.source": IMAGE_SOURCE,
        "org.opencontainers.image.revision": current_revision,
        "org.amd-ai.profile.id": STABLE_TORCH_PROFILE,
        "org.amd-ai.profile.status": "verified",
        "org.amd-ai.rocm.version": "7.2.1",
        "org.amd-ai.python.version": "3.12",
        "org.amd-ai.torch.version": "2.9.1",
    }.items():
        if labels.get(name) != expected:
            raise PublishError(f"qualified image label does not match: {name}")
    wheel_hashes = _require_string_mapping(report, "wheel_hashes")
    if set(wheel_hashes) != {"torch", "torchvision", "torchaudio", "triton"}:
        raise PublishError("qualified release must bind four primary wheels")
    for name, digest in wheel_hashes.items():
        if DIGEST_PATTERN.fullmatch(digest) is None:
            raise PublishError(f"qualified wheel digest is invalid: {name}")

    qualification_file = _resolve_evidence_path(
        _require_string(report, "qualification_file"),
        report_path=qualification_path,
        repo_root=repo_root,
    )
    qualification_bytes, qualification = _load_json_object(
        qualification_file, "qualification report"
    )
    qualification_run_digest = hashlib.sha256(qualification_bytes).hexdigest()
    if _require_hex_digest(report, "qualification_digest") != qualification_run_digest:
        raise PublishError("qualification report digest does not match evidence")
    _validate_qualification(
        qualification,
        expected_profile_digest=expected_profile_digest,
        expected_image_id=torch_image_id,
    )

    sbom_bytes, sbom = _load_json_object(sbom_path, "SPDX SBOM")
    if sbom.get("spdxVersion") != "SPDX-2.3":
        raise PublishError("release SBOM must be SPDX 2.3")
    sbom_digest = hashlib.sha256(sbom_bytes).hexdigest()
    if _require_hex_digest(report, "sbom_digest") != sbom_digest:
        raise PublishError("SPDX SBOM digest does not match release report")
    if Path(_require_string(report, "sbom_file")).name != sbom_path.name:
        raise PublishError("SPDX SBOM filename does not match release report")

    return PublishCandidate(
        release_id=release_id,
        source_repository=IMAGE_SOURCE,
        source_revision=current_revision,
        base_local_id=base_image_id,
        torch_local_id=torch_image_id,
        gpu_arch="gfx1151",
        supported_host_adapter_ids=("ubuntu-24.04",),
        qualification_digest=_prefixed_hash(release_bytes),
        qualification_run_digest="sha256:" + qualification_run_digest,
        qualification_profile_digest=(
            "sha256:" + expected_profile_digest
        ),
        sbom_digest="sha256:" + sbom_digest,
        torch_profile_id=STABLE_TORCH_PROFILE,
        torch_profile_digest="sha256:" + expected_torch_profile_digest,
        base_artifact_digests={
            "rocm_keyring": "sha256:" + _hash_file(repo_root / ROCM_KEYRING),
            "rocm_packages_lock": "sha256:" + expected_rocm_lock_digest,
        },
        torch_artifact_digests={
            "profile": "sha256:" + expected_torch_profile_digest,
            "requirements_lock": "sha256:"
            + _hash_file(repo_root / TORCH_REQUIREMENTS_LOCK),
        },
        image_labels=labels,
        published_at=generated_at,
    )


def _validate_qualification(
    report: dict[str, Any],
    *,
    expected_profile_digest: str,
    expected_image_id: str,
) -> None:
    _require_keys("qualification report", report, QUALIFICATION_KEYS)
    if type(report["schema_version"]) is not int or report["schema_version"] != 1:
        raise PublishError("qualification report schema is invalid")
    for name, expected in {
        "status": "pass",
        "profile_id": "stable-gfx1151",
        "profile_digest": expected_profile_digest,
        "image": STABLE_TORCH_IMAGE,
        "image_id": expected_image_id,
        "gpu_arch": "gfx1151",
    }.items():
        _require_equal(report, name, expected)
    required = report["required_checks"]
    if (
        not isinstance(required, list)
        or len(required) != len(REQUIRED_QUALIFICATION_CHECKS)
        or set(required) != REQUIRED_QUALIFICATION_CHECKS
    ):
        raise PublishError("qualification required-check set is incomplete")
    results = report["results"]
    if not isinstance(results, list):
        raise PublishError("qualification results are invalid")
    names = [
        item.get("name") for item in results if isinstance(item, dict)
    ]
    counts = Counter(names)
    for name in REQUIRED_QUALIFICATION_CHECKS:
        matches = [
            item
            for item in results
            if isinstance(item, dict) and item.get("name") == name
        ]
        if counts[name] != 1 or matches[0].get("passed") is not True:
            raise PublishError(f"qualification check did not pass once: {name}")


def _load_json_object(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise PublishError(f"cannot read {label}: {error}") from error
    try:
        payload = json.loads(content, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PublishError(f"cannot parse {label}: {error}") from error
    if not isinstance(payload, dict):
        raise PublishError(f"{label} must be a JSON object")
    return content, payload


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, value in pairs:
        if name in values:
            raise PublishError(f"duplicate JSON key in publication evidence: {name}")
        values[name] = value
    return values


def _require_keys(
    label: str, values: dict[str, Any], expected: frozenset[str]
) -> None:
    missing = sorted(expected.difference(values))
    unknown = sorted(set(values).difference(expected))
    if missing:
        raise PublishError(f"{label} is missing keys: {', '.join(missing)}")
    if unknown:
        raise PublishError(f"{label} has unknown keys: {', '.join(unknown)}")


def _require_string(values: dict[str, Any], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value or "\0" in value:
        raise PublishError(f"publication field {name} must be a string")
    return value


def _require_equal(values: dict[str, Any], name: str, expected: object) -> None:
    if values.get(name) != expected:
        raise PublishError(
            f"publication field {name} does not match expected evidence"
        )


def _require_hex_digest(values: dict[str, Any], name: str) -> str:
    value = _require_string(values, name)
    if DIGEST_PATTERN.fullmatch(value) is None:
        raise PublishError(f"publication digest {name} is invalid")
    return value


def _require_string_mapping(
    values: dict[str, Any], name: str
) -> dict[str, str]:
    raw = values.get(name)
    if not isinstance(raw, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in raw.items()
    ):
        raise PublishError(f"publication field {name} is invalid")
    return dict(raw)


def _require_image_id(value: str, label: str) -> None:
    if IMAGE_ID_PATTERN.fullmatch(value) is None:
        raise PublishError(f"{label} local image ID is invalid")


def _resolve_evidence_path(
    value: str, *, report_path: Path, repo_root: Path
) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    repository_candidate = repo_root / candidate
    if repository_candidate.is_file():
        return repository_candidate
    return report_path.parent / candidate


def _hash_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise PublishError(f"cannot hash publication evidence {path}: {error}") from error


def _prefixed_hash(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()
