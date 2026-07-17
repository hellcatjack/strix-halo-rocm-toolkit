from __future__ import annotations

import json
from pathlib import Path

import pytest

from amd_ai.installer.release import (
    ReleaseAcquisitionError,
    ReleaseError,
    ReleaseIdentityError,
    load_stable_release,
    pull_and_verify_release,
    verify_release_image,
)
from tests.unit.installer.fakes import FakeReleaseDocker


FIXTURE = Path("tests/fixtures/releases/stable.json")


@pytest.fixture
def release():
    return load_stable_release(FIXTURE)


def test_valid_release_distinguishes_manifest_and_config_digest() -> None:
    release = load_stable_release(FIXTURE)

    assert release.torch.reference.startswith(
        "ghcr.io/hellcatjack/strix-halo-rocm-pytorch@sha256:"
    )
    assert release.torch.manifest_digest != release.torch.config_digest
    assert release.supported_host_adapter_ids == ("ubuntu-24.04",)


@pytest.mark.parametrize(
    "mutation", ("missing", "unknown", "bad-digest", "mutable-image")
)
def test_release_schema_rejects_ambiguous_payload(
    tmp_path: Path, mutation: str
) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if mutation == "missing":
        del payload["source_revision"]
    elif mutation == "unknown":
        payload["latest"] = True
    elif mutation == "bad-digest":
        payload["torch"]["manifest_digest"] = payload["torch"][
            "config_digest"
        ]
    else:
        payload["torch"]["image"] += ":latest"
    path = tmp_path / "release.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReleaseError):
        load_stable_release(path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", True),
        ("release_id", "latest"),
        ("gpu_arch", "gfx1100"),
        ("rocm_version", "7.2.4"),
        ("python_version", "3.13"),
        ("torch_version", "2.10.0"),
        ("published_at", "2026-07-10T12:00:00+00:00"),
    ),
)
def test_release_schema_rejects_wrong_fixed_identity(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload[field] = value
    path = tmp_path / "release.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReleaseError):
        load_stable_release(path)


def test_release_schema_rejects_duplicate_json_key(tmp_path: Path) -> None:
    text = FIXTURE.read_text(encoding="utf-8")
    path = tmp_path / "release.json"
    path.write_text(
        text.replace(
            '"schema_version": 1,',
            '"schema_version": 1,\n  "schema_version": 1,',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseError, match="duplicate"):
        load_stable_release(path)


def test_release_schema_rejects_duplicate_adapter(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["supported_host_adapter_ids"] = [
        "ubuntu-24.04",
        "ubuntu-24.04",
    ]
    path = tmp_path / "release.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReleaseError, match="adapter"):
        load_stable_release(path)


def test_verify_release_image_requires_repo_digest_config_labels_and_artifacts(
    release,
) -> None:
    docker = FakeReleaseDocker.for_release(release)

    identity = verify_release_image(
        release, release.torch, kind="torch", docker=docker
    )

    assert identity.config_digest == release.torch.config_digest
    assert identity.repo_digests == (release.torch.reference,)
    assert docker.hash_calls == [
        (release.torch.reference, "/opt/amd-ai/profile.env"),
        (release.torch.reference, "/opt/amd-ai/profile.requirements.lock"),
        (release.torch.reference, "/opt/amd-ai/torch-manifest.json"),
    ]


def test_verify_release_image_rejects_friendly_tag_drift(release) -> None:
    docker = FakeReleaseDocker.for_release(release)
    docker.records[release.torch.reference]["RepoDigests"] = [
        release.torch.image + "@sha256:" + "9" * 64
    ]

    with pytest.raises(ReleaseError, match="RepoDigest"):
        verify_release_image(
            release, release.torch, kind="torch", docker=docker
        )


def test_pull_failure_is_distinct_from_pulled_identity_failure(release) -> None:
    unavailable = FakeReleaseDocker.for_release(release)
    unavailable.pull_error = OSError("network unavailable")

    with pytest.raises(ReleaseAcquisitionError):
        pull_and_verify_release(release, docker=unavailable)

    drifted = FakeReleaseDocker.for_release(release)
    drifted.records[release.base.reference]["Id"] = "sha256:" + "8" * 64
    with pytest.raises(ReleaseIdentityError):
        pull_and_verify_release(release, docker=drifted)


def test_containerd_store_manifest_id_uses_manifest_config_descriptor(
    release,
) -> None:
    docker = FakeReleaseDocker.for_release(release)
    for image in (release.base, release.torch):
        docker.records[image.reference]["Id"] = image.manifest_digest

    result = pull_and_verify_release(release, docker=docker)

    assert result.base.config_digest == release.base.config_digest
    assert result.torch.config_digest == release.torch.config_digest
    assert docker.manifest_config_calls == [
        release.base.reference,
        release.torch.reference,
    ]


def test_containerd_store_rejects_manifest_config_descriptor_drift(release) -> None:
    docker = FakeReleaseDocker.for_release(release)
    docker.records[release.base.reference]["Id"] = release.base.manifest_digest
    docker.manifest_config_digests[release.base.reference] = "sha256:" + "8" * 64

    with pytest.raises(ReleaseIdentityError, match="config digest"):
        pull_and_verify_release(release, docker=docker)


@pytest.mark.parametrize("damage", ("config", "label", "artifact", "kind"))
def test_verify_release_image_rejects_identity_damage(
    release, damage: str
) -> None:
    docker = FakeReleaseDocker.for_release(release)
    kind = "torch"
    if damage == "config":
        docker.records[release.torch.reference]["Id"] = "sha256:" + "8" * 64
    elif damage == "label":
        config = docker.records[release.torch.reference]["Config"]
        config["Labels"]["org.amd-ai.profile.status"] = "experimental"
    elif damage == "artifact":
        docker.hashes[
            (release.torch.reference, "/opt/amd-ai/torch-manifest.json")
        ] = "sha256:" + "8" * 64
    else:
        kind = "base"

    with pytest.raises(ReleaseError):
        verify_release_image(
            release, release.torch, kind=kind, docker=docker
        )


def test_pull_release_uses_only_manifest_digest_references(release) -> None:
    docker = FakeReleaseDocker.for_release(release)

    result = pull_and_verify_release(release, docker=docker)

    assert docker.pull_calls == [release.base.reference, release.torch.reference]
    assert result.base.config_digest == release.base.config_digest
    assert result.torch.config_digest == release.torch.config_digest


def test_pull_failure_does_not_fall_back_to_mutable_tag(release) -> None:
    docker = FakeReleaseDocker.for_release(release)
    docker.pull_error = ReleaseError("network stopped")

    with pytest.raises(ReleaseError, match="network stopped"):
        pull_and_verify_release(release, docker=docker)

    assert all("@sha256:" in value for value in docker.pull_calls)
    assert docker.inspect_calls == []
