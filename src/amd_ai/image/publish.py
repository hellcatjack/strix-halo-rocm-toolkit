from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from amd_ai.image.build import IMAGE_SOURCE
from amd_ai.installer.models import ReleaseImage, StableRelease
from amd_ai.installer.release import (
    BASE_ARTIFACT_PATHS,
    SEMVER_PATTERN,
    TORCH_ARTIFACT_PATHS,
    load_stable_release,
    verify_release_image,
)
from amd_ai.runner import CommandResult, Runner, SubprocessRunner


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
QUALIFICATION_PROFILE = Path("profiles/qualification/stable.toml")
TORCH_PROFILE = Path("profiles/torch/stable.env")
TORCH_REQUIREMENTS_LOCK = Path("profiles/torch/stable.requirements.lock")
ROCM_PACKAGE_LOCK = Path("profiles/rocm/7.2.1-packages.lock")
ROCM_KEYRING = Path("profiles/rocm/rocm.gpg")
STABLE_TORCH_IMAGE = "rocm-pytorch:7.2.1-py3.12-torch2.9.1"
STABLE_TORCH_PROFILE = "rocm-7.2.1-py3.12-torch-2.9.1"
VERIFIED_TAG = STABLE_TORCH_IMAGE + "-gfx1151-verified"
BASE_PACKAGE = "ghcr.io/hellcatjack/strix-halo-rocm-python"
TORCH_PACKAGE = "ghcr.io/hellcatjack/strix-halo-rocm-pytorch"
MAX_GHCR_LAYER_BYTES = 10_000_000_000
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


@dataclass(frozen=True)
class RegistryImageObservation:
    image: str
    manifest_digest: str
    config_digest: str
    artifact_digests: Mapping[str, str]
    raw_manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_digests",
            MappingProxyType(dict(self.artifact_digests)),
        )
        object.__setattr__(
            self,
            "raw_manifest",
            MappingProxyType(dict(self.raw_manifest)),
        )


class PublishRegistry(Protocol):
    def tag(self, image_id: str, target: str) -> None:
        pass

    def push(self, target: str) -> None:
        pass

    def observe(self, target: str) -> RegistryImageObservation:
        pass

    def authless_pull(self, reference: str) -> None:
        pass

    def authless_manifest_config_digest(self, reference: str) -> str:
        pass

    def inspect(self, reference: str) -> Mapping[str, object]:
        pass

    def hash_file(self, reference: str, path: str) -> str:
        pass


class DockerPublishRegistry:
    def __init__(
        self,
        docker_prefix: tuple[str, ...] = ("docker",),
        *,
        runner: Runner | None = None,
    ) -> None:
        if not docker_prefix or any(not value for value in docker_prefix):
            raise PublishError("Docker command prefix is invalid")
        self.docker_prefix = tuple(docker_prefix)
        self.runner = runner if runner is not None else SubprocessRunner()

    def tag(self, image_id: str, target: str) -> None:
        _require_image_id(image_id, "publication")
        self._completed(("tag", image_id, target))

    def tag_reference(self, source: str, target: str) -> None:
        if REPO_DIGEST_PATTERN.fullmatch(source) is None:
            raise PublishError("tag source must be an exact image reference")
        if (
            not target
            or target.startswith("-")
            or "\0" in target
            or any(character.isspace() for character in target)
        ):
            raise PublishError("tag target is invalid")
        self._completed(("tag", source, target))

    def push(self, target: str) -> None:
        self._completed(("push", target))

    def pull(self, reference: str) -> None:
        self._completed(("pull", reference))

    def authless_pull(self, reference: str) -> None:
        self._authless_completed(("pull", reference))

    def manifest_config_digest(self, reference: str) -> str:
        result = self._completed(
            ("manifest", "inspect", "--verbose", reference)
        )
        return _manifest_config_digest(result.stdout)

    def authless_manifest_config_digest(self, reference: str) -> str:
        result = self._authless_completed(
            ("manifest", "inspect", "--verbose", reference)
        )
        return _manifest_config_digest(result.stdout)

    def _authless_completed(
        self,
        args: tuple[str, ...],
    ) -> CommandResult:
        with tempfile.TemporaryDirectory(prefix="amd-ai-authless-") as directory:
            config = Path(directory)
            config.chmod(0o700)
            if any(config.iterdir()):
                raise PublishError("temporary authless Docker config is not empty")
            return self._completed(("--config", str(config), *args))

    def inspect(self, reference: str) -> Mapping[str, object]:
        result = self._completed(("image", "inspect", reference))
        try:
            payload = json.loads(result.stdout, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise PublishError("cannot parse Docker image inspection") from error
        if (
            not isinstance(payload, list)
            or len(payload) != 1
            or not isinstance(payload[0], dict)
        ):
            raise PublishError("Docker image inspection is not one object")
        return payload[0]

    def hash_file(self, reference: str, path: str) -> str:
        result = self._completed(
            (
                "run",
                "--rm",
                "--entrypoint",
                "/usr/bin/sha256sum",
                reference,
                path,
            )
        )
        digest = result.stdout.strip().partition(" ")[0]
        if DIGEST_PATTERN.fullmatch(digest) is None:
            raise PublishError(f"cannot parse embedded artifact digest: {path}")
        return "sha256:" + digest

    def observe(self, target: str) -> RegistryImageObservation:
        package = _package_from_tag(target)
        self.pull(target)
        record = self.inspect(target)
        local_image_id = record.get("Id")
        if not isinstance(local_image_id, str) or IMAGE_ID_PATTERN.fullmatch(
            local_image_id
        ) is None:
            raise PublishError("pushed image has no valid local ID")
        raw_repo_digests = record.get("RepoDigests")
        if not isinstance(raw_repo_digests, list) or any(
            not isinstance(value, str) for value in raw_repo_digests
        ):
            raise PublishError("pushed image has no RepoDigests")
        matches = sorted(
            {
                value
                for value in raw_repo_digests
                if value.startswith(package + "@sha256:")
            }
        )
        if len(matches) != 1:
            raise PublishError(
                "pushed image must expose exactly one matching RepoDigest"
            )
        reference = matches[0]
        manifest_digest = reference.removeprefix(package + "@")
        raw_result = self._completed(
            ("buildx", "imagetools", "inspect", "--raw", reference)
        )
        try:
            raw_manifest = json.loads(
                raw_result.stdout, object_pairs_hook=_unique_object
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise PublishError("cannot parse raw registry manifest") from error
        if not isinstance(raw_manifest, dict):
            raise PublishError("raw registry manifest must be an object")
        config_digest = _manifest_config_digest_from_record(raw_manifest)
        if local_image_id not in {config_digest, manifest_digest}:
            raise PublishError(
                "pushed image local ID differs from its manifest and config"
            )
        paths = (
            BASE_ARTIFACT_PATHS
            if package == BASE_PACKAGE
            else TORCH_ARTIFACT_PATHS
        )
        artifacts = {
            name: self.hash_file(reference, path)
            for name, path in paths.items()
        }
        return RegistryImageObservation(
            image=package,
            manifest_digest=manifest_digest,
            config_digest=config_digest,
            artifact_digests=artifacts,
            raw_manifest=raw_manifest,
        )

    def _completed(
        self,
        args: tuple[str, ...],
    ) -> CommandResult:
        result = self.runner.run(
            [*self.docker_prefix, *args], check=False
        )
        if result.returncode != 0:
            evidence = result.stderr.strip() or result.stdout.strip() or "no output"
            raise PublishError(
                f"Docker publication command failed ({args[0]}): {evidence}"
            )
        return result


class AnonymousDockerRegistry(DockerPublishRegistry):
    def pull(self, reference: str) -> None:
        self.authless_pull(reference)

    def manifest_config_digest(self, reference: str) -> str:
        return self.authless_manifest_config_digest(reference)


class _AnonymousVerificationRegistry:
    def __init__(self, registry: PublishRegistry) -> None:
        self.registry = registry

    def inspect(self, reference: str) -> Mapping[str, object]:
        return self.registry.inspect(reference)

    def manifest_config_digest(self, reference: str) -> str:
        return self.registry.authless_manifest_config_digest(reference)

    def hash_file(self, reference: str, path: str) -> str:
        return self.registry.hash_file(reference, path)


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


def publish_images(
    candidate: PublishCandidate, *, registry: PublishRegistry
) -> StableRelease:
    candidate = verify_publish_candidate_local_images(
        candidate, registry=registry
    )
    if candidate.base_local_id is None:
        raise PublishError("base local image ID is required for publication")
    _require_image_id(candidate.base_local_id, "base")
    _require_image_id(candidate.torch_local_id, "Torch")
    observations: list[RegistryImageObservation] = []
    for kind, local_id, package in (
        ("base", candidate.base_local_id, BASE_PACKAGE),
        ("torch", candidate.torch_local_id, TORCH_PACKAGE),
    ):
        target = f"{package}:{candidate.release_id}"
        try:
            registry.tag(local_id, target)
            registry.push(target)
            observation = registry.observe(target)
        except PublishError:
            raise
        except Exception as error:
            raise PublishError(
                f"cannot publish or observe {kind} image {target}: {error}"
            ) from error
        _validate_registry_observation(
            observation,
            kind=kind,
            package=package,
            local_id=local_id,
            candidate=candidate,
        )
        observations.append(observation)

    base = _release_image(observations[0])
    torch = _release_image(observations[1])
    return StableRelease(
        schema_version=1,
        release_id=candidate.release_id,
        source_repository=candidate.source_repository,
        source_revision=candidate.source_revision,
        qualification_profile_digest=candidate.qualification_profile_digest,
        qualification_report_digest=candidate.qualification_digest,
        sbom_digest=candidate.sbom_digest,
        gpu_arch=candidate.gpu_arch,
        supported_host_adapter_ids=candidate.supported_host_adapter_ids,
        rocm_version="7.2.1",
        python_version="3.12",
        torch_version="2.9.1",
        torch_profile_id=candidate.torch_profile_id,
        torch_profile_digest=candidate.torch_profile_digest,
        base=base,
        torch=torch,
        published_at=candidate.published_at,
    )


def verify_publish_candidate_local_images(
    candidate: PublishCandidate, *, registry: PublishRegistry
) -> PublishCandidate:
    if candidate.base_local_id is None:
        raise PublishError("base local image ID is required for publication")
    artifacts_by_kind: dict[str, dict[str, str]] = {}
    for kind, image_id, paths in (
        ("base", candidate.base_local_id, BASE_ARTIFACT_PATHS),
        ("torch", candidate.torch_local_id, TORCH_ARTIFACT_PATHS),
    ):
        _require_image_id(image_id, kind)
        try:
            record = registry.inspect(image_id)
        except PublishError:
            raise
        except Exception as error:
            raise PublishError(f"cannot inspect local {kind} image: {error}") from error
        if not isinstance(record, Mapping) or record.get("Id") != image_id:
            raise PublishError(f"local {kind} image config ID does not match")
        config = record.get("Config")
        labels = config.get("Labels") if isinstance(config, Mapping) else None
        if not isinstance(labels, Mapping):
            raise PublishError(f"local {kind} image labels are invalid")
        expected_labels = {
            "org.opencontainers.image.source": candidate.source_repository,
            "org.opencontainers.image.revision": candidate.source_revision,
            "org.amd-ai.rocm.version": "7.2.1",
            "org.amd-ai.python.version": "3.12",
        }
        if kind == "torch":
            expected_labels.update(
                {
                    "org.amd-ai.profile.id": candidate.torch_profile_id,
                    "org.amd-ai.profile.status": "verified",
                    "org.amd-ai.torch.version": "2.9.1",
                }
            )
        for name, expected in expected_labels.items():
            if labels.get(name) != expected:
                raise PublishError(
                    f"local {kind} image label does not match: {name}"
                )
        observed_artifacts: dict[str, str] = {}
        for name, path in paths.items():
            try:
                digest = registry.hash_file(image_id, path)
            except PublishError:
                raise
            except Exception as error:
                raise PublishError(
                    f"cannot hash local {kind} artifact {path}: {error}"
                ) from error
            if IMAGE_ID_PATTERN.fullmatch(digest) is None:
                raise PublishError(f"local {kind} artifact digest is invalid: {name}")
            observed_artifacts[name] = digest
        expected_artifacts = (
            candidate.base_artifact_digests
            if kind == "base"
            else candidate.torch_artifact_digests
        )
        for name, expected in expected_artifacts.items():
            if observed_artifacts.get(name) != expected:
                raise PublishError(
                    f"local {kind} artifact differs from source evidence: {name}"
                )
        artifacts_by_kind[kind] = observed_artifacts
    return replace(
        candidate,
        base_artifact_digests=artifacts_by_kind["base"],
        torch_artifact_digests=artifacts_by_kind["torch"],
    )


def publish_stable_release(
    candidate: PublishCandidate,
    *,
    registry: PublishRegistry,
    output: Path,
    observed: StableRelease | None = None,
) -> StableRelease:
    release = observed or publish_images(candidate, registry=registry)
    _require_release_matches_candidate(release, candidate)
    anonymous_registry = _AnonymousVerificationRegistry(registry)
    for image, kind in ((release.base, "base"), (release.torch, "torch")):
        try:
            registry.authless_pull(image.reference)
        except PublishError:
            raise
        except Exception as error:
            raise PublishError(
                f"anonymous pull failed for {image.reference}: {error}"
            ) from error
        try:
            verify_release_image(
                release,
                image,
                kind=kind,
                docker=anonymous_registry,
            )
        except Exception as error:
            if isinstance(error, PublishError):
                raise
            raise PublishError(
                f"anonymous image verification failed for {image.reference}: {error}"
            ) from error
    content = (
        json.dumps(_stable_release_dict(release), indent=2, sort_keys=True)
        + "\n"
    )
    _atomic_write(output, content, mode=0o644)
    return release


def write_observed_release(path: Path, release: StableRelease) -> None:
    content = (
        json.dumps(_stable_release_dict(release), indent=2, sort_keys=True)
        + "\n"
    )
    _atomic_write(path, content, mode=0o600)


def observe_pushed_release(path: Path) -> StableRelease:
    return load_stable_release(path)


def _require_release_matches_candidate(
    release: StableRelease, candidate: PublishCandidate
) -> None:
    expected = {
        "release_id": candidate.release_id,
        "source_repository": candidate.source_repository,
        "source_revision": candidate.source_revision,
        "qualification_profile_digest": candidate.qualification_profile_digest,
        "qualification_report_digest": candidate.qualification_digest,
        "sbom_digest": candidate.sbom_digest,
        "gpu_arch": candidate.gpu_arch,
        "supported_host_adapter_ids": candidate.supported_host_adapter_ids,
        "torch_profile_id": candidate.torch_profile_id,
        "torch_profile_digest": candidate.torch_profile_digest,
        "published_at": candidate.published_at,
    }
    for name, value in expected.items():
        if getattr(release, name) != value:
            raise PublishError(
                f"observed publication differs from current evidence: {name}"
            )
    if candidate.base_local_id is None:
        raise PublishError("base local image ID is required for stable publication")
    if candidate.base_local_id not in {
        release.base.config_digest,
        release.base.manifest_digest,
    }:
        raise PublishError("observed base identity differs from current local image")
    if candidate.torch_local_id not in {
        release.torch.config_digest,
        release.torch.manifest_digest,
    }:
        raise PublishError("observed Torch identity differs from current local image")
    if (release.base.image, release.torch.image) != (
        BASE_PACKAGE,
        TORCH_PACKAGE,
    ):
        raise PublishError("observed publication uses an unexpected package")
    for name, digest in candidate.base_artifact_digests.items():
        if release.base.artifact_digests.get(name) != digest:
            raise PublishError(
                f"observed base artifact differs from current evidence: {name}"
            )
    for name, digest in candidate.torch_artifact_digests.items():
        if release.torch.artifact_digests.get(name) != digest:
            raise PublishError(
                f"observed Torch artifact differs from current evidence: {name}"
            )


def _validate_registry_observation(
    observation: RegistryImageObservation,
    *,
    kind: str,
    package: str,
    local_id: str,
    candidate: PublishCandidate,
) -> None:
    if observation.image != package:
        raise PublishError(f"observed {kind} package name does not match")
    if (
        not isinstance(observation.manifest_digest, str)
        or not observation.manifest_digest.startswith("sha256:")
        or IMAGE_ID_PATTERN.fullmatch(observation.manifest_digest) is None
    ):
        raise PublishError(f"observed {kind} registry manifest digest is invalid")
    if local_id not in {
        observation.config_digest,
        observation.manifest_digest,
    }:
        raise PublishError(f"observed {kind} identity differs from local image")
    if observation.manifest_digest == observation.config_digest:
        raise PublishError(f"observed {kind} manifest and config IDs are ambiguous")

    required = (
        {"rocm_keyring", "rocm_packages_lock"}
        if kind == "base"
        else {"profile", "requirements_lock", "torch_manifest"}
    )
    artifacts = dict(observation.artifact_digests)
    if set(artifacts) != required:
        raise PublishError(f"observed {kind} artifact set is incomplete")
    for name, digest in artifacts.items():
        if IMAGE_ID_PATTERN.fullmatch(digest) is None:
            raise PublishError(f"observed {kind} artifact digest is invalid: {name}")
    expected_sources = (
        candidate.base_artifact_digests
        if kind == "base"
        else candidate.torch_artifact_digests
    )
    for name, expected in expected_sources.items():
        if artifacts.get(name) != expected:
            raise PublishError(
                f"observed {kind} artifact differs from source evidence: {name}"
            )
    _validate_raw_manifest(
        observation.raw_manifest,
        config_digest=observation.config_digest,
    )


def _validate_raw_manifest(
    manifest: Mapping[str, object], *, config_digest: str
) -> None:
    if manifest.get("schemaVersion") != 2:
        raise PublishError("registry manifest schema is invalid")
    if "manifests" in manifest:
        descriptors = manifest.get("manifests")
        if not isinstance(descriptors, list) or len(descriptors) != 1:
            raise PublishError("registry manifest must contain exactly one platform")
        descriptor = descriptors[0]
        if not isinstance(descriptor, Mapping):
            raise PublishError("registry platform descriptor is invalid")
        platform = descriptor.get("platform")
        if not isinstance(platform, Mapping) or (
            platform.get("os"), platform.get("architecture")
        ) != ("linux", "amd64"):
            raise PublishError("registry platform must be linux/amd64")
        _validate_descriptor(descriptor, label="platform manifest")
        return

    config = manifest.get("config")
    if not isinstance(config, Mapping) or config.get("digest") != config_digest:
        raise PublishError("registry image config descriptor does not match")
    _validate_descriptor(config, label="config")
    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise PublishError("registry image manifest has no layers")
    for layer in layers:
        if not isinstance(layer, Mapping):
            raise PublishError("registry layer descriptor is invalid")
        _validate_descriptor(layer, label="layer")
        size = layer.get("size")
        if type(size) is not int or size > MAX_GHCR_LAYER_BYTES:
            raise PublishError("registry layer exceeds GHCR size limit")


def _validate_descriptor(
    descriptor: Mapping[str, object], *, label: str
) -> None:
    digest = descriptor.get("digest")
    size = descriptor.get("size")
    if (
        not isinstance(digest, str)
        or IMAGE_ID_PATTERN.fullmatch(digest) is None
        or type(size) is not int
        or size < 0
    ):
        raise PublishError(f"registry {label} descriptor is invalid")


def _release_image(observation: RegistryImageObservation) -> ReleaseImage:
    return ReleaseImage(
        image=observation.image,
        manifest_digest=observation.manifest_digest,
        config_digest=observation.config_digest,
        artifact_digests=observation.artifact_digests,
    )


def _stable_release_dict(release: StableRelease) -> dict[str, object]:
    def image_dict(image: ReleaseImage) -> dict[str, object]:
        return {
            "image": image.image,
            "manifest_digest": image.manifest_digest,
            "config_digest": image.config_digest,
            "artifact_digests": dict(image.artifact_digests),
        }

    return {
        "schema_version": release.schema_version,
        "release_id": release.release_id,
        "source_repository": release.source_repository,
        "source_revision": release.source_revision,
        "qualification_profile_digest": release.qualification_profile_digest,
        "qualification_report_digest": release.qualification_report_digest,
        "sbom_digest": release.sbom_digest,
        "gpu_arch": release.gpu_arch,
        "supported_host_adapter_ids": list(release.supported_host_adapter_ids),
        "rocm_version": release.rocm_version,
        "python_version": release.python_version,
        "torch_version": release.torch_version,
        "torch_profile_id": release.torch_profile_id,
        "torch_profile_digest": release.torch_profile_digest,
        "base": image_dict(release.base),
        "torch": image_dict(release.torch),
        "published_at": release.published_at,
    }


def _atomic_write(path: Path, content: str, *, mode: int) -> None:
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            mode,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


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


def _manifest_config_digest(raw: str) -> str:
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PublishError("cannot parse Docker manifest inspection") from error
    if not isinstance(payload, dict):
        raise PublishError("Docker manifest inspection is not an object")
    return _manifest_config_digest_from_record(payload)


def _manifest_config_digest_from_record(payload: Mapping[str, object]) -> str:
    wrappers = [
        payload[name]
        for name in ("SchemaV2Manifest", "OCIManifest")
        if name in payload
    ]
    if len(wrappers) > 1:
        raise PublishError("Docker manifest record is ambiguous")
    manifest = wrappers[0] if wrappers else payload
    if not isinstance(manifest, Mapping):
        raise PublishError("Docker manifest record is invalid")
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise PublishError("Docker manifest config descriptor is missing")
    digest = config.get("digest")
    if not isinstance(digest, str) or IMAGE_ID_PATTERN.fullmatch(digest) is None:
        raise PublishError("Docker manifest config digest is invalid")
    return digest


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


def _package_from_tag(target: str) -> str:
    for package in (BASE_PACKAGE, TORCH_PACKAGE):
        if target.startswith(package + ":"):
            tag = target.removeprefix(package + ":")
            if ":" not in tag and SEMVER_PATTERN.fullmatch(tag) is not None:
                return package
    raise PublishError(f"publication target is not an approved release tag: {target}")


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
