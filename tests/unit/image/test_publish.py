from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from amd_ai.image import publish
from amd_ai.image.build import IMAGE_SOURCE
from amd_ai.image.publish import (
    AnonymousDockerRegistry,
    DockerPublishRegistry,
    PublishError,
    validate_publish_inputs,
)
from amd_ai.image.publish import (
    MAX_GHCR_LAYER_BYTES,
    RegistryImageObservation,
    observe_pushed_release,
    publish_images,
    publish_stable_release,
    write_observed_release,
)
from amd_ai.installer.release import load_stable_release
from amd_ai.qualification.models import REQUIRED_CHECKS
from amd_ai.runner import CommandResult


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class FakeRegistry:
    def __init__(self, candidate) -> None:
        self.candidate = candidate
        self.calls: list[tuple[str, ...]] = []
        self.push_error: Exception | None = None
        self.authless_error: Exception | None = None
        self.authless_pull_calls: list[str] = []
        self.manifest_calls: list[str] = []
        self.authless_manifest_calls: list[str] = []
        self.exact_inspect_uses_manifest = False
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

    def authless_pull(self, reference: str) -> None:
        self.authless_pull_calls.append(reference)
        if self.authless_error is not None:
            raise self.authless_error

    def inspect(self, reference: str):
        observation = self._by_reference(reference)
        labels = {
            "org.opencontainers.image.source": self.candidate.source_repository,
            "org.opencontainers.image.revision": self.candidate.source_revision,
            "org.amd-ai.rocm.version": "7.2.1",
            "org.amd-ai.python.version": "3.12",
        }
        if observation.image.endswith("-pytorch"):
            labels.update(
                {
                    "org.amd-ai.profile.id": self.candidate.torch_profile_id,
                    "org.amd-ai.profile.status": "verified",
                    "org.amd-ai.torch.version": "2.9.1",
                }
            )
        image_id = observation.config_digest
        if (
            reference == observation.manifest_digest
            or self.exact_inspect_uses_manifest
            and "@sha256:" in reference
        ):
            image_id = observation.manifest_digest
        return {
            "Id": image_id,
            "RepoDigests": [reference],
            "Config": {"Labels": labels},
        }

    def manifest_config_digest(self, reference: str) -> str:
        self.manifest_calls.append(reference)
        return self._by_reference(reference).config_digest

    def authless_manifest_config_digest(self, reference: str) -> str:
        self.authless_manifest_calls.append(reference)
        return self._by_reference(reference).config_digest

    def hash_file(self, reference: str, path: str) -> str:
        observation = self._by_reference(reference)
        names = {
            "/etc/apt/keyrings/rocm.gpg": "rocm_keyring",
            "/opt/amd-ai/locks/rocm-packages.lock": "rocm_packages_lock",
            "/opt/amd-ai/profile.env": "profile",
            "/opt/amd-ai/profile.requirements.lock": "requirements_lock",
            "/opt/amd-ai/torch-manifest.json": "torch_manifest",
        }
        return observation.artifact_digests[names[path]]

    def _by_reference(self, reference: str) -> RegistryImageObservation:
        for observation in self.observations.values():
            if (
                f"{observation.image}@{observation.manifest_digest}"
                == reference
                or observation.config_digest == reference
                or observation.manifest_digest == reference
            ):
                return observation
        raise KeyError(reference)


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


def test_publish_accepts_containerd_local_manifest_ids(
    candidate, tmp_path: Path
) -> None:
    registry = FakeRegistry.for_candidate(candidate)
    containerd_candidate = replace(
        candidate,
        base_local_id=registry.observations[
            publish.BASE_PACKAGE + ":0.2.0"
        ].manifest_digest,
        torch_local_id=registry.observations[
            publish.TORCH_PACKAGE + ":0.2.0"
        ].manifest_digest,
    )
    registry.candidate = containerd_candidate

    release = publish_images(containerd_candidate, registry=registry)
    published = publish_stable_release(
        containerd_candidate,
        registry=registry,
        output=tmp_path / "stable.json",
        observed=release,
    )

    assert release.base.config_digest == candidate.base_local_id
    assert release.torch.config_digest == candidate.torch_local_id
    assert published == release


def test_manifest_is_written_only_after_two_authless_pulls(
    candidate, tmp_path: Path
) -> None:
    registry = FakeRegistry.for_candidate(candidate)
    output = tmp_path / "stable.json"

    release = publish_stable_release(
        candidate, registry=registry, output=output
    )

    assert registry.authless_pull_calls == [
        release.base.reference,
        release.torch.reference,
    ]
    assert output.is_file()
    assert load_stable_release(output) == release


def test_publish_final_manifest_descriptor_check_is_anonymous(
    candidate, tmp_path: Path
) -> None:
    registry = FakeRegistry.for_candidate(candidate)
    registry.exact_inspect_uses_manifest = True

    release = publish_stable_release(
        candidate,
        registry=registry,
        output=tmp_path / "stable.json",
    )

    assert registry.authless_manifest_calls == [
        release.base.reference,
        release.torch.reference,
    ]
    assert registry.manifest_calls == []


def test_failed_authless_pull_leaves_existing_manifest_unchanged(
    candidate, tmp_path: Path
) -> None:
    output = tmp_path / "stable.json"
    output.write_text("known-good\n", encoding="utf-8")
    registry = FakeRegistry.for_candidate(candidate)
    registry.authless_error = PublishError("denied")

    with pytest.raises(PublishError, match="denied"):
        publish_stable_release(candidate, registry=registry, output=output)

    assert output.read_text(encoding="utf-8") == "known-good\n"


def test_authless_pull_passes_empty_config_to_sudo_docker(
) -> None:
    class RecordingRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []
            self.config_was_empty = False

        def run(self, args, *, check=True, input_text=None):
            del check, input_text
            command = tuple(args)
            config = Path(command[command.index("--config") + 1])
            self.config_was_empty = config.is_dir() and not tuple(
                config.iterdir()
            )
            self.calls.append(command)
            return CommandResult(command, 0, "", "")

    runner = RecordingRunner()
    reference = "ghcr.io/hellcatjack/example@sha256:" + "a" * 64

    DockerPublishRegistry(
        ("sudo", "-n", "docker"), runner=runner
    ).authless_pull(reference)

    command = runner.calls[0]
    assert command[:4] == ("sudo", "-n", "docker", "--config")
    assert command[-2:] == ("pull", reference)
    assert runner.config_was_empty is True


def test_authless_manifest_inspect_returns_config_descriptor_digest() -> None:
    config_digest = "sha256:" + "b" * 64

    class RecordingRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []
            self.config_was_empty = False

        def run(self, args, *, check=True, input_text=None):
            del check, input_text
            command = tuple(args)
            config = Path(command[command.index("--config") + 1])
            self.config_was_empty = config.is_dir() and not tuple(
                config.iterdir()
            )
            self.calls.append(command)
            payload = {
                "Ref": command[-1],
                "SchemaV2Manifest": {
                    "schemaVersion": 2,
                    "config": {"digest": config_digest},
                },
            }
            return CommandResult(command, 0, json.dumps(payload), "")

    runner = RecordingRunner()
    reference = "ghcr.io/hellcatjack/example@sha256:" + "a" * 64
    registry = DockerPublishRegistry(
        ("sudo", "-n", "docker"), runner=runner
    )

    observed = registry.authless_manifest_config_digest(reference)

    assert observed == config_digest
    command = runner.calls[0]
    assert command[:4] == ("sudo", "-n", "docker", "--config")
    assert command[-4:] == ("manifest", "inspect", "--verbose", reference)
    assert runner.config_was_empty is True


def test_anonymous_registry_routes_pull_and_manifest_through_empty_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = AnonymousDockerRegistry()
    reference = "ghcr.io/example/image@sha256:" + "a" * 64
    expected = "sha256:" + "b" * 64
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        registry,
        "authless_pull",
        lambda value: calls.append(("pull", value)),
    )
    monkeypatch.setattr(
        registry,
        "authless_manifest_config_digest",
        lambda value: calls.append(("manifest", value)) or expected,
    )

    registry.pull(reference)
    observed = registry.manifest_config_digest(reference)

    assert observed == expected
    assert calls == [("pull", reference), ("manifest", reference)]


def test_manifest_parser_accepts_oci_verbose_record() -> None:
    config_digest = "sha256:" + "b" * 64
    payload = {
        "Ref": "ghcr.io/example/image@sha256:" + "a" * 64,
        "OCIManifest": {
            "schemaVersion": 2,
            "config": {"digest": config_digest},
        },
    }

    assert publish._manifest_config_digest(json.dumps(payload)) == config_digest


def test_registry_pull_and_inspect_use_injected_runner() -> None:
    record = {
        "Id": "sha256:" + "b" * 64,
        "RepoDigests": ["ghcr.io/example/image@sha256:" + "a" * 64],
    }

    class RecordingRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def run(self, args, *, check=True, input_text=None):
            del check, input_text
            command = tuple(args)
            self.calls.append(command)
            stdout = json.dumps([record]) if "inspect" in command else ""
            return CommandResult(command, 0, stdout, "")

    runner = RecordingRunner()
    registry = DockerPublishRegistry(runner=runner)
    reference = "ghcr.io/example/image@sha256:" + "a" * 64

    registry.pull(reference)
    inspected = registry.inspect(reference)

    assert inspected == record
    assert runner.calls == [
        ("docker", "pull", reference),
        ("docker", "image", "inspect", reference),
    ]


def test_registry_tags_verified_exact_reference_with_injected_runner() -> None:
    source = "ghcr.io/example/image@sha256:" + "a" * 64
    target = "image:stable"

    class RecordingRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def run(self, args, *, check=True, input_text=None):
            del check, input_text
            command = tuple(args)
            self.calls.append(command)
            return CommandResult(command, 0, "", "")

    runner = RecordingRunner()
    registry = DockerPublishRegistry(runner=runner)

    registry.tag_reference(source, target)

    assert runner.calls == [("docker", "tag", source, target)]
    with pytest.raises(PublishError, match="exact"):
        registry.tag_reference("ghcr.io/example/image:latest", target)


def test_registry_observation_uses_manifest_config_on_containerd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = DockerPublishRegistry()
    target = publish.BASE_PACKAGE + ":0.3.2"
    manifest_digest = "sha256:" + "a" * 64
    config_digest = "sha256:" + "b" * 64
    exact = publish.BASE_PACKAGE + "@" + manifest_digest
    monkeypatch.setattr(registry, "pull", lambda value: None)
    monkeypatch.setattr(
        registry,
        "inspect",
        lambda value: {"Id": manifest_digest, "RepoDigests": [exact]},
    )
    monkeypatch.setattr(
        registry,
        "_completed",
        lambda args: CommandResult(
            tuple(args), 0, json.dumps(_raw_manifest(config_digest)), ""
        ),
    )
    monkeypatch.setattr(
        registry,
        "hash_file",
        lambda reference, path: "sha256:" + "c" * 64,
    )

    observation = registry.observe(target)

    assert observation.manifest_digest == manifest_digest
    assert observation.config_digest == config_digest


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
