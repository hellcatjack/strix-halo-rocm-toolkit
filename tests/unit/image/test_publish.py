from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from amd_ai.image.build import IMAGE_SOURCE
from amd_ai.image.publish import PublishError, validate_publish_inputs
from amd_ai.image.publish import (
    MAX_GHCR_LAYER_BYTES,
    RegistryImageObservation,
    observe_pushed_release,
    publish_images,
    write_observed_release,
)
from amd_ai.qualification.models import REQUIRED_CHECKS


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class FakeRegistry:
    def __init__(self, candidate) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.push_error: Exception | None = None
        self.observations = {
            "ghcr.io/hellcatjack/strix-halo-rocm-python:0.2.0": (
                RegistryImageObservation(
                    image="ghcr.io/hellcatjack/strix-halo-rocm-python",
                    manifest_digest="sha256:" + "d" * 64,
                    config_digest=candidate.base_local_id,
                    artifact_digests=candidate.base_artifact_digests,
                    raw_manifest=_raw_manifest(candidate.base_local_id),
                )
            ),
            "ghcr.io/hellcatjack/strix-halo-rocm-pytorch:0.2.0": (
                RegistryImageObservation(
                    image="ghcr.io/hellcatjack/strix-halo-rocm-pytorch",
                    manifest_digest="sha256:" + "e" * 64,
                    config_digest=candidate.torch_local_id,
                    artifact_digests={
                        **candidate.torch_artifact_digests,
                        "torch_manifest": "sha256:" + "f" * 64,
                    },
                    raw_manifest=_raw_manifest(candidate.torch_local_id),
                )
            ),
        }

    @classmethod
    def for_candidate(cls, candidate):
        return cls(candidate)

    def tag(self, image_id: str, target: str) -> None:
        self.calls.append(("tag", image_id, target))

    def push(self, target: str) -> None:
        self.calls.append(("push", target))
        if self.push_error is not None:
            raise self.push_error

    def observe(self, target: str) -> RegistryImageObservation:
        self.calls.append(("observe", target))
        return self.observations[target]


@pytest.fixture
def candidate(tmp_path: Path):
    qualification, sbom = write_publish_evidence(
        tmp_path,
        revision="a" * 40,
        image_id="sha256:" + "b" * 64,
    )
    return validate_publish_inputs(
        release_id="0.2.0",
        qualification_path=qualification,
        sbom_path=sbom,
        current_revision="a" * 40,
        base_image_id="sha256:" + "c" * 64,
        torch_image_id="sha256:" + "b" * 64,
    )


def test_candidate_requires_matching_revision_image_and_evidence(
    tmp_path: Path,
) -> None:
    qualification, sbom = write_publish_evidence(
        tmp_path,
        revision="a" * 40,
        image_id="sha256:" + "b" * 64,
    )

    candidate = validate_publish_inputs(
        release_id="0.2.0",
        qualification_path=qualification,
        sbom_path=sbom,
        current_revision="a" * 40,
        torch_image_id="sha256:" + "b" * 64,
    )

    assert candidate.gpu_arch == "gfx1151"
    assert candidate.qualification_digest.startswith("sha256:")
    assert candidate.sbom_digest.startswith("sha256:")
    assert candidate.torch_local_id == "sha256:" + "b" * 64


@pytest.mark.parametrize(
    "damage",
    (
        "revision",
        "image",
        "status",
        "architecture",
        "checks",
        "qualification-digest",
        "sbom-digest",
        "source-label",
    ),
)
def test_candidate_rejects_stale_or_incomplete_evidence(
    tmp_path: Path, damage: str
) -> None:
    qualification, sbom = write_publish_evidence(
        tmp_path,
        revision="a" * 40,
        image_id="sha256:" + "b" * 64,
        damage=damage,
    )
    current_revision = "c" * 40 if damage == "revision" else "a" * 40
    image_id = (
        "sha256:" + "c" * 64
        if damage == "image"
        else "sha256:" + "b" * 64
    )

    with pytest.raises(PublishError):
        validate_publish_inputs(
            release_id="0.2.0",
            qualification_path=qualification,
            sbom_path=sbom,
            current_revision=current_revision,
            torch_image_id=image_id,
        )


def test_publish_tags_pushes_and_observes_each_registry_digest(
    candidate, tmp_path: Path
) -> None:
    registry = FakeRegistry.for_candidate(candidate)

    observed = publish_images(candidate, registry=registry)

    assert registry.calls == [
        (
            "tag",
            candidate.base_local_id,
            "ghcr.io/hellcatjack/strix-halo-rocm-python:0.2.0",
        ),
        ("push", "ghcr.io/hellcatjack/strix-halo-rocm-python:0.2.0"),
        ("observe", "ghcr.io/hellcatjack/strix-halo-rocm-python:0.2.0"),
        (
            "tag",
            candidate.torch_local_id,
            "ghcr.io/hellcatjack/strix-halo-rocm-pytorch:0.2.0",
        ),
        ("push", "ghcr.io/hellcatjack/strix-halo-rocm-pytorch:0.2.0"),
        ("observe", "ghcr.io/hellcatjack/strix-halo-rocm-pytorch:0.2.0"),
    ]
    assert observed.torch.manifest_digest.startswith("sha256:")

    report = tmp_path / "publish-candidate.json"
    write_observed_release(report, observed)
    assert observe_pushed_release(report) == observed


@pytest.mark.parametrize(
    "damage", ("push", "manifest", "config", "layer", "platform")
)
def test_publish_rejects_incomplete_registry_identity(
    candidate, damage: str
) -> None:
    registry = FakeRegistry.for_candidate(candidate)
    base_tag = "ghcr.io/hellcatjack/strix-halo-rocm-python:0.2.0"
    observation = registry.observations[base_tag]
    if damage == "push":
        registry.push_error = PublishError("push stopped")
    elif damage == "manifest":
        registry.observations[base_tag] = RegistryImageObservation(
            observation.image,
            "missing",
            observation.config_digest,
            observation.artifact_digests,
            observation.raw_manifest,
        )
    elif damage == "config":
        registry.observations[base_tag] = RegistryImageObservation(
            observation.image,
            observation.manifest_digest,
            "sha256:" + "9" * 64,
            observation.artifact_digests,
            _raw_manifest("sha256:" + "9" * 64),
        )
    elif damage == "layer":
        raw = _raw_manifest(observation.config_digest)
        raw["layers"][0]["size"] = MAX_GHCR_LAYER_BYTES + 1
        registry.observations[base_tag] = RegistryImageObservation(
            observation.image,
            observation.manifest_digest,
            observation.config_digest,
            observation.artifact_digests,
            raw,
        )
    else:
        registry.observations[base_tag] = RegistryImageObservation(
            observation.image,
            observation.manifest_digest,
            observation.config_digest,
            observation.artifact_digests,
            {
                "schemaVersion": 2,
                "manifests": [
                    {
                        "digest": "sha256:" + "8" * 64,
                        "size": 123,
                        "platform": {"os": "linux", "architecture": "arm64"},
                    }
                ],
            },
        )

    with pytest.raises(PublishError):
        publish_images(candidate, registry=registry)


def write_publish_evidence(
    root: Path,
    *,
    revision: str,
    image_id: str,
    damage: str | None = None,
) -> tuple[Path, Path]:
    profile_digest = _hash(REPOSITORY_ROOT / "profiles/qualification/stable.toml")
    torch_profile_digest = _hash(
        REPOSITORY_ROOT / "profiles/torch/stable.env"
    )
    rocm_lock_digest = _hash(
        REPOSITORY_ROOT / "profiles/rocm/7.2.1-packages.lock"
    )
    checks = list(REQUIRED_CHECKS)
    if damage == "checks":
        checks.pop()
    qualification_payload = {
        "generated_at": "2026-07-10T12:00:00Z",
        "gpu_arch": "gfx1151",
        "image": "rocm-pytorch:7.2.1-py3.12-torch2.9.1",
        "image_id": image_id,
        "profile_digest": profile_digest,
        "profile_id": "stable-gfx1151",
        "required_checks": checks,
        "results": [
            {"name": name, "passed": True} for name in checks
        ],
        "schema_version": 1,
        "status": "pass",
    }
    qualification_run = root / "qualification.json"
    qualification_run.write_text(
        json.dumps(qualification_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sbom = root / "release.spdx.json"
    sbom.write_text(
        json.dumps({"spdxVersion": "SPDX-2.3"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    qualification_digest = _hash(qualification_run)
    sbom_digest = _hash(sbom)
    labels = {
        "org.amd-ai.profile.id": "rocm-7.2.1-py3.12-torch-2.9.1",
        "org.amd-ai.profile.status": "verified",
        "org.amd-ai.python.version": "3.12",
        "org.amd-ai.rocm.version": "7.2.1",
        "org.amd-ai.torch.version": "2.9.1",
        "org.opencontainers.image.revision": revision,
        "org.opencontainers.image.source": IMAGE_SOURCE,
    }
    if damage == "source-label":
        labels["org.opencontainers.image.source"] = "local"
    release_payload = {
        "design_digest": "d" * 64,
        "generated_at": "2026-07-10T12:00:00Z",
        "git_revision": revision,
        "gpu_arch": "gfx1100" if damage == "architecture" else "gfx1151",
        "image_id": image_id,
        "image_labels": labels,
        "image_reference": "rocm-pytorch:7.2.1-py3.12-torch2.9.1",
        "profile_digest": profile_digest,
        "profile_id": "stable-gfx1151",
        "qualification_digest": (
            "0" * 64 if damage == "qualification-digest" else qualification_digest
        ),
        "qualification_file": str(qualification_run),
        "repo_digest": None,
        "rocm_package_lock_digest": rocm_lock_digest,
        "sbom_digest": "0" * 64 if damage == "sbom-digest" else sbom_digest,
        "sbom_file": sbom.name,
        "schema_version": 1,
        "status": "experimental" if damage == "status" else "verified",
        "torch_profile_digest": torch_profile_digest,
        "verified_tag": (
            "rocm-pytorch:7.2.1-py3.12-torch2.9.1-gfx1151-verified"
        ),
        "wheel_hashes": {
            "torch": "1" * 64,
            "torchaudio": "2" * 64,
            "torchvision": "3" * 64,
            "triton": "4" * 64,
        },
    }
    release_report = root / "release.json"
    release_report.write_text(
        json.dumps(release_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return release_report, sbom


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_manifest(config_digest: str) -> dict[str, object]:
    return {
        "schemaVersion": 2,
        "config": {"digest": config_digest, "size": 1024},
        "layers": [
            {"digest": "sha256:" + "7" * 64, "size": 1024}
        ],
    }
