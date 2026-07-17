from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from amd_ai.host.models import (
    HostPlanPhase,
    HostSnapshot,
    PlannedAction,
    PreparePlan,
)
from amd_ai.installer.actions import (
    HostPlanResult,
    LocalBuildResult,
    prepare_plan_payload,
)
from amd_ai.installer.models import (
    DiskSpaceEstimate,
    InstallOptions,
    InstallStage,
    ReleaseImage,
    StageResult,
    StableRelease,
)
from amd_ai.installer.release import (
    VerifiedImageIdentity,
    VerifiedReleaseImages,
    load_stable_release,
)
from amd_ai.installer.state import stage_input_digest
from amd_ai.report import Report, Status
from tests.unit.host.fakes import healthy_snapshot


class FakeReleaseDocker:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}
        self.hashes: dict[tuple[str, str], str] = {}
        self.pull_calls: list[str] = []
        self.inspect_calls: list[str] = []
        self.manifest_config_calls: list[str] = []
        self.manifest_config_digests: dict[str, str] = {}
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

    def manifest_config_digest(self, reference: str) -> str:
        self.manifest_config_calls.append(reference)
        return self.manifest_config_digests[reference]

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
        self.manifest_config_digests[image.reference] = image.config_digest
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
        self.build_image_estimate = self.image_estimate
        self.project_estimate = DiskSpaceEstimate(
            location=Path("/tmp"),
            payload_bytes=1024**3,
            available_bytes=100 * 1024**3,
        )
        self.image_estimate_kwargs: list[dict[str, object]] = []
        self.snapshot = healthy_snapshot()
        self.host_apply_include_docker_group: bool | None = None
        self.kernel_plan_result = self._make_host_plan(
            phase=HostPlanPhase.KERNEL,
            reboot_required=False,
        )
        self.host_plan_result = self._make_host_plan(
            phase=HostPlanPhase.TUNING,
            reboot_required=False,
        )
        self.pull_error: BaseException | None = None
        self.pull_errors: dict[str, BaseException] = {}
        self.project_init_kwargs: dict[str, object] = {}

    @classmethod
    def healthy(cls) -> FakeInstallerActions:
        return cls(load_stable_release(Path("tests/fixtures/releases/stable.json")))

    @classmethod
    def stop_after(cls, stage: InstallStage) -> FakeInstallerActions:
        fake = cls.healthy()
        fake.stop_stage = stage
        return fake

    @classmethod
    def host_change_requires_reboot(cls) -> FakeInstallerActions:
        return cls.healthy()

    @classmethod
    def host_change_requires_kernel_reboot(cls) -> FakeInstallerActions:
        fake = cls.healthy()
        fake.kernel_plan_result = fake._make_host_plan(
            phase=HostPlanPhase.KERNEL,
            reboot_required=True,
            running_kernel="6.14.0-1020-oem",
        )
        return fake

    @classmethod
    def full_no_reboot(cls) -> FakeInstallerActions:
        fake = cls.healthy()
        fake.host_plan_result = fake._make_host_plan(
            phase=HostPlanPhase.TUNING,
            reboot_required=False,
        )
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

    def host_preflight(self, **kwargs: object) -> StageResult:
        del kwargs
        return self._record(InstallStage.HOST_PREFLIGHT, "host_preflight")

    def host_plan(self, **kwargs: object) -> HostPlanResult:
        phase = HostPlanPhase(kwargs["phase"])
        stage = (
            InstallStage.KERNEL_PLAN
            if phase is HostPlanPhase.KERNEL
            else InstallStage.HOST_PLAN
        )
        name = "kernel_plan" if phase is HostPlanPhase.KERNEL else "host_plan"
        self._raise_if_needed(stage)
        self.calls.append(name)
        return (
            self.kernel_plan_result
            if phase is HostPlanPhase.KERNEL
            else self.host_plan_result
        )

    def host_apply(
        self,
        host_plan: HostPlanResult,
        *,
        include_docker_group: bool,
    ) -> StageResult:
        if host_plan.plan.phase is HostPlanPhase.KERNEL:
            assert host_plan.plan_digest == self.kernel_plan_result.plan_digest
            assert include_docker_group is False
            return self._record(InstallStage.KERNEL_APPLY, "kernel_apply")
        assert host_plan.plan_digest == self.host_plan_result.plan_digest
        self.host_apply_include_docker_group = include_docker_group
        return self._record(InstallStage.HOST_APPLY, "host_apply")

    def kernel_verify(self, **kwargs: object) -> Report:
        del kwargs
        self._raise_if_needed(InstallStage.KERNEL_VERIFY)
        self.calls.append("kernel_verify")
        return Report(
            command="host-kernel-verify",
            status=Status.PASS,
            generated_at="2026-07-16T12:00:00Z",
            facts={"kernel": "6.17.0-1028-oem"},
            findings=(),
        )

    def host_verify(self, **kwargs: object) -> StageResult:
        del kwargs
        return self._record(InstallStage.HOST_VERIFY, "host_verify")

    def resolve_release(self, manifest_path: Path) -> StableRelease:
        del manifest_path
        self._raise_if_needed(InstallStage.RELEASE_RESOLVE)
        self.calls.append("resolve_release")
        return self.release

    def pull_release(self, release: StableRelease) -> VerifiedReleaseImages:
        self._raise_if_needed(InstallStage.IMAGE_PULL_OR_BUILD)
        self.calls.append("pull_release")
        self.image_calls.extend(
            (("pull", release.base.reference), ("pull", release.torch.reference))
        )
        error = self.pull_errors.get(release.base.image) or self.pull_error
        if error is not None:
            raise error
        return VerifiedReleaseImages(
            base=VerifiedImageIdentity(
                reference=release.base.reference,
                config_digest=release.base.config_digest,
                repo_digests=(release.base.reference,),
                labels={},
            ),
            torch=VerifiedImageIdentity(
                reference=release.torch.reference,
                config_digest=release.torch.config_digest,
                repo_digests=(release.torch.reference,),
                labels={},
            ),
        )

    def build_local_images(self, **kwargs: object) -> LocalBuildResult:
        del kwargs
        self._raise_if_needed(InstallStage.IMAGE_PULL_OR_BUILD)
        self.calls.append("build_local_images")
        self.image_calls.extend(
            (("build", "rocm-python"), ("build", "rocm-pytorch"))
        )
        return LocalBuildResult(
            base_reference="sha256:" + "6" * 64,
            base_config_digest="sha256:" + "8" * 64,
            torch_reference="sha256:" + "7" * 64,
            torch_config_digest="sha256:" + "9" * 64,
            source_revision="d" * 40,
        )

    def verify_torch_image(self, image: str) -> StageResult:
        del image
        return self._record(InstallStage.IMAGE_VERIFY, "verify_torch_image")

    def initialize_project(self, **kwargs: object) -> StageResult:
        self.project_init_kwargs = dict(kwargs)
        project_dir = kwargs["project_dir"]
        assert isinstance(project_dir, Path)
        project_dir.mkdir(parents=True, exist_ok=True)
        return self._record(InstallStage.PROJECT_INIT, "initialize_project")

    def verify_project(self, **kwargs: object) -> StageResult:
        del kwargs
        return self._record(InstallStage.PROJECT_VERIFY, "verify_project")

    def image_disk_estimate(self, **kwargs: object) -> DiskSpaceEstimate:
        self.image_estimate_kwargs.append(dict(kwargs))
        source = kwargs.get("image_source")
        return (
            self.build_image_estimate
            if source == "build"
            else self.image_estimate
        )

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

    def _make_host_plan(
        self,
        *,
        phase: HostPlanPhase,
        reboot_required: bool,
        running_kernel: str | None = None,
    ) -> HostPlanResult:
        plan = PreparePlan(
            phase=phase,
            supported=True,
            target_user="developer",
            actions=(
                PlannedAction(
                    code=(
                        "KERNEL.CHANGE"
                        if phase is HostPlanPhase.KERNEL
                        else "HOST.CHANGE"
                    ),
                    summary="Apply reviewed host change",
                    argv=("true",),
                    privileged=True,
                ),
            ),
            reboot_required=reboot_required,
        )
        return HostPlanResult(
            snapshot=self.snapshot,
            plan=plan,
            plan_digest=stage_input_digest(prepare_plan_payload(plan)),
            adapter_id="ubuntu-24.04",
            running_kernel=running_kernel or self.snapshot.kernel,
            display_manager_loaded=self.snapshot.display_manager_loaded,
            display_manager_active=self.snapshot.display_manager_active,
        )


class FakePrompts:
    def __init__(
        self,
        *,
        exact: Mapping[str, bool] | None = None,
        yes_no: Mapping[str, bool] | None = None,
        image_fallback: str | None = None,
    ) -> None:
        self.exact = dict(exact or {})
        self.yes_no = dict(yes_no or {})
        self.image_fallback = image_fallback
        self.statuses: list[tuple[str, str]] = []

    def confirm_exact(self, word: str) -> bool:
        return self.exact.get(word, False)

    def confirm_yes_no(self, question: str) -> bool:
        return self.yes_no.get(question, False)

    def choose_image_fallback(self) -> str | None:
        return self.image_fallback

    def ask_project_dir(self) -> Path:
        raise AssertionError("project directory should be explicit in tests")

    def status(self, prefix: str, message: str) -> None:
        self.statuses.append((prefix, message))
