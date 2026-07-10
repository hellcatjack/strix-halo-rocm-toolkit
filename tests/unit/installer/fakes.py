from __future__ import annotations

from collections.abc import Mapping

from amd_ai.installer.models import ReleaseImage, StableRelease


class FakeReleaseDocker:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}
        self.hashes: dict[tuple[str, str], str] = {}
        self.pull_calls: list[str] = []
        self.inspect_calls: list[str] = []
        self.hash_calls: list[tuple[str, str]] = []
        self.pull_error: Exception | None = None

    @classmethod
    def for_release(cls, release: StableRelease) -> FakeReleaseDocker:
        fake = cls()
        fake._add_image(release, release.base, kind="base")
        fake._add_image(release, release.torch, kind="torch")
        return fake

    def pull(self, reference: str) -> None:
        self.pull_calls.append(reference)
        if self.pull_error is not None:
            raise self.pull_error

    def inspect(self, reference: str) -> Mapping[str, object]:
        self.inspect_calls.append(reference)
        return self.records[reference]

    def hash_file(self, reference: str, path: str) -> str:
        self.hash_calls.append((reference, path))
        return self.hashes[(reference, path)]

    def _add_image(
        self,
        release: StableRelease,
        image: ReleaseImage,
        *,
        kind: str,
    ) -> None:
        labels = {
            "org.opencontainers.image.source": release.source_repository,
            "org.opencontainers.image.revision": release.source_revision,
            "org.amd-ai.rocm.version": release.rocm_version,
            "org.amd-ai.python.version": release.python_version,
        }
        paths = {
            "rocm_keyring": "/etc/apt/keyrings/rocm.gpg",
            "rocm_packages_lock": "/opt/amd-ai/locks/rocm-packages.lock",
        }
        if kind == "torch":
            labels.update(
                {
                    "org.amd-ai.profile.id": release.torch_profile_id,
                    "org.amd-ai.profile.status": "verified",
                    "org.amd-ai.torch.version": release.torch_version,
                }
            )
            paths = {
                "profile": "/opt/amd-ai/profile.env",
                "requirements_lock": "/opt/amd-ai/profile.requirements.lock",
                "torch_manifest": "/opt/amd-ai/torch-manifest.json",
            }
        self.records[image.reference] = {
            "Id": image.config_digest,
            "RepoDigests": [image.reference],
            "Config": {"Labels": labels},
        }
        for name, path in paths.items():
            self.hashes[(image.reference, path)] = image.artifact_digests[name]
