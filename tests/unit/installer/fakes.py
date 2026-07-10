from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from amd_ai.installer.models import (
    DiskSpaceEstimate,
    InstallOptions,
    InstallStage,
    ReleaseImage,
    StageResult,
    StableRelease,
)
from amd_ai.installer.release import load_stable_release


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


class FakeInstallerActions:
    def __init__(self, release: StableRelease) -> None:
        self.release = release
        self.calls: list[str] = []
        self.image_calls: list[tuple[str, str]] = []
        self.input_facts: dict[InstallStage, object] = {}
        self.stop_stage: InstallStage | None = None
        self.failures: dict[InstallStage, BaseException] = {}
        self.image_estimate = DiskSpaceEstimate(
            location=Path("/var/lib/docker"),
            payload_bytes=10 * 1024**3,
            available_bytes=100 * 1024**3,
        )
        self.project_estimate = DiskSpaceEstimate(
            location=Path("/tmp"),
            payload_bytes=1024**3,
            available_bytes=100 * 1024**3,
        )

    @classmethod
    def healthy(cls) -> FakeInstallerActions:
        return cls(load_stable_release(Path("tests/fixtures/releases/stable.json")))

    @classmethod
    def stop_after(cls, stage: InstallStage) -> FakeInstallerActions:
        fake = cls.healthy()
        fake.stop_stage = stage
        return fake

    def stage_inputs(
        self,
        stage: InstallStage,
        options: InstallOptions,
        state: object,
    ) -> object:
        del options, state
        return self.input_facts.get(stage, {})

    def bootstrap(self, **kwargs: object) -> StageResult:
        del kwargs
        return self._record(InstallStage.BOOTSTRAP, "bootstrap")

    def container_host_check(self) -> StageResult:
        return self._record(
            InstallStage.CONTAINER_HOST_CHECK, "container_host_check"
        )

    def resolve_release(self, manifest_path: Path) -> StableRelease:
        del manifest_path
        self._raise_if_needed(InstallStage.RELEASE_RESOLVE)
        self.calls.append("resolve_release")
        return self.release

    def pull_release(self, release: StableRelease) -> object:
        self._raise_if_needed(InstallStage.IMAGE_PULL_OR_BUILD)
        self.calls.append("pull_release")
        self.image_calls.extend(
            (("pull", release.base.reference), ("pull", release.torch.reference))
        )
        return object()

    def verify_torch_image(self, image: str) -> StageResult:
        del image
        return self._record(InstallStage.IMAGE_VERIFY, "verify_torch_image")

    def initialize_project(self, **kwargs: object) -> StageResult:
        project_dir = kwargs["project_dir"]
        assert isinstance(project_dir, Path)
        project_dir.mkdir(parents=True, exist_ok=True)
        return self._record(InstallStage.PROJECT_INIT, "initialize_project")

    def verify_project(self, **kwargs: object) -> StageResult:
        del kwargs
        return self._record(InstallStage.PROJECT_VERIFY, "verify_project")

    def image_disk_estimate(self, **kwargs: object) -> DiskSpaceEstimate:
        del kwargs
        return self.image_estimate

    def project_disk_estimate(self, **kwargs: object) -> DiskSpaceEstimate:
        del kwargs
        return self.project_estimate

    def stage_result(
        self, stage: InstallStage, output: object
    ) -> StageResult:
        del output
        return StageResult(
            action_required=self.stop_stage is stage,
            message="simulated stop" if self.stop_stage is stage else "",
        )

    def _record(self, stage: InstallStage, name: str) -> StageResult:
        self._raise_if_needed(stage)
        self.calls.append(name)
        return StageResult(
            action_required=self.stop_stage is stage,
            message="simulated stop" if self.stop_stage is stage else "",
        )

    def _raise_if_needed(self, stage: InstallStage) -> None:
        error = self.failures.get(stage)
        if error is not None:
            raise error
