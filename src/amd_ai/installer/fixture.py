from __future__ import annotations

import json
from pathlib import Path

from amd_ai.host.models import HostSnapshot, PlannedAction, PreparePlan
from amd_ai.host.parsers import GpuPciInfo
from amd_ai.installer.actions import (
    HostPlanResult,
    LocalBuildResult,
    prepare_plan_payload,
)
from amd_ai.installer.models import (
    DiskSpaceEstimate,
    InstallOptions,
    InstallStage,
    StageResult,
    StableRelease,
)
from amd_ai.installer.release import load_stable_release
from amd_ai.installer.state import read_boot_id, stage_input_digest
from amd_ai.project.config import load_project_config


class FixtureBackendError(RuntimeError):
    pass


class FixtureInstallerActions:
    def __init__(self, root: Path) -> None:
        raw = Path(root).expanduser()
        if raw.is_symlink():
            raise FixtureBackendError("fixture root must not be a symlink")
        try:
            self.root = raw.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise FixtureBackendError(f"cannot resolve fixture root: {error}") from error
        scenario_path = self.root / "scenario.json"
        try:
            payload = json.loads(scenario_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FixtureBackendError(f"cannot load installer fixture: {error}") from error
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise FixtureBackendError("installer fixture schema is invalid")
        self.scenario = payload

    def stage_inputs(
        self,
        stage: InstallStage,
        options: InstallOptions,
        state: object,
    ) -> object:
        del options, state
        return {
            "fact_revision": self.scenario.get("fact_revision", "fixture-v1"),
            "stage": stage.value,
        }

    def bootstrap(self, **kwargs: object) -> StageResult:
        del kwargs
        return self._result(InstallStage.BOOTSTRAP, "bootstrap")

    def container_host_check(self) -> StageResult:
        if self.scenario.get("docker_version") is None:
            return StageResult(
                blocked=True,
                message="fixture Docker daemon is unavailable",
            )
        devices = self.scenario.get("device_gids")
        if not isinstance(devices, dict) or not {
            "/dev/kfd",
            "/dev/dri/renderD128",
        }.issubset(devices):
            return StageResult(
                blocked=True,
                message="fixture GPU devices are unavailable",
            )
        return self._result(
            InstallStage.CONTAINER_HOST_CHECK, "container_host_check"
        )

    def host_preflight(self, **kwargs: object) -> StageResult:
        del kwargs
        return self._result(InstallStage.HOST_PREFLIGHT, "host_preflight")

    def host_plan(self, *, target_user: str, **kwargs: object) -> HostPlanResult:
        del kwargs
        result = self._result(InstallStage.HOST_PLAN, "host_plan")
        if result.blocked:
            raise FixtureBackendError(result.message)
        plan = fixture_prepare_plan(
            target_user,
            reboot_required=bool(self.scenario.get("reboot_required", True)),
        )
        return HostPlanResult(
            snapshot=_fixture_snapshot(),
            plan=plan,
            plan_digest=stage_input_digest(prepare_plan_payload(plan)),
            adapter_id=str(self.scenario.get("adapter_id", "ubuntu-24.04")),
        )

    def host_apply(
        self,
        host_plan: HostPlanResult,
        *,
        include_docker_group: bool,
    ) -> StageResult:
        del host_plan, include_docker_group
        return self._result(InstallStage.HOST_APPLY, "host_apply")

    def host_verify(self, **kwargs: object) -> StageResult:
        del kwargs
        return self._result(InstallStage.HOST_VERIFY, "host_verify")

    def resolve_release(self, manifest_path: Path) -> StableRelease:
        blocked = self._result(InstallStage.RELEASE_RESOLVE, "resolve_release")
        if blocked.blocked:
            raise FixtureBackendError(blocked.message)
        return load_stable_release(manifest_path)

    def pull_release(self, release: StableRelease) -> object:
        del release
        blocked = self._result(
            InstallStage.IMAGE_PULL_OR_BUILD, "pull_release"
        )
        if blocked.blocked:
            raise FixtureBackendError(blocked.message)
        return object()

    def build_local_images(self, **kwargs: object) -> LocalBuildResult:
        del kwargs
        self._log("build_local_images")
        return LocalBuildResult(
            base_reference="rocm-python:fixture",
            base_config_digest="sha256:" + "8" * 64,
            torch_reference="rocm-pytorch:fixture",
            torch_config_digest="sha256:" + "9" * 64,
            source_revision="d" * 40,
        )

    def verify_torch_image(self, image: str) -> StageResult:
        del image
        return self._result(InstallStage.IMAGE_VERIFY, "verify_torch_image")

    def initialize_project(
        self,
        *,
        project_dir: Path,
        project_name: str,
        base_config_digest: str,
        **kwargs: object,
    ) -> StageResult:
        del kwargs
        result = self._result(InstallStage.PROJECT_INIT, "initialize_project")
        if result.blocked:
            return result
        project = Path(project_dir).resolve()
        project.mkdir(parents=True, exist_ok=True)
        config = project / "amd-ai-project.toml"
        if not config.exists():
            config.write_text(
                "[project]\n"
                f"name = {json.dumps(project_name)}\n"
                'base_profile = "stable"\n'
                f"image = {json.dumps(project_name + ':runtime')}\n"
                f"base_image = {json.dumps(base_config_digest)}\n"
                f"base_digest = {json.dumps(base_config_digest)}\n"
                'command = ["bash"]\n'
                "debug = false\n",
                encoding="utf-8",
            )
        parsed = load_project_config(config)
        if parsed.base_digest != base_config_digest:
            return StageResult(
                blocked=True,
                message="fixture project parent digest changed",
            )
        return result

    def verify_project(
        self, *, project_dir: Path, **kwargs: object
    ) -> StageResult:
        del kwargs
        load_project_config(Path(project_dir) / "amd-ai-project.toml")
        return self._result(InstallStage.PROJECT_VERIFY, "verify_project")

    def image_disk_estimate(self, **kwargs: object) -> DiskSpaceEstimate:
        del kwargs
        return DiskSpaceEstimate(
            location=self.root,
            payload_bytes=1024,
            available_bytes=100 * 1024**3,
        )

    def project_disk_estimate(self, **kwargs: object) -> DiskSpaceEstimate:
        del kwargs
        return DiskSpaceEstimate(
            location=self.root,
            payload_bytes=0,
            available_bytes=100 * 1024**3,
        )

    def read_boot_id(self) -> str:
        return read_boot_id(self.root / "boot_id")

    def _result(self, stage: InstallStage, call: str) -> StageResult:
        self._log(call)
        blocked = self.scenario.get("blocked_stage") == stage.value
        return StageResult(
            blocked=blocked,
            message=f"fixture blocked {stage.value}" if blocked else "",
        )

    def _log(self, value: str) -> None:
        with (self.root / "calls.log").open("a", encoding="utf-8") as stream:
            stream.write(value + "\n")


def fixture_prepare_plan(
    target_user: str, *, reboot_required: bool = True
) -> PreparePlan:
    return PreparePlan(
        supported=True,
        target_user=target_user,
        actions=(
            PlannedAction(
                code="HOST.FIXTURE_CHANGE",
                summary="Apply fixture host change",
                argv=("true",),
                privileged=True,
            ),
        ),
        reboot_required=reboot_required,
    )


def fixture_host_plan_digest(target_user: str) -> str:
    return stage_input_digest(
        prepare_plan_payload(fixture_prepare_plan(target_user))
    )


def _fixture_snapshot() -> HostSnapshot:
    return HostSnapshot(
        os_id="ubuntu",
        os_version="24.04",
        architecture="x86_64",
        kernel="6.14.0-1018-oem",
        gpu=GpuPciInfo("1002:1586", "amdgpu", "fixture"),
        mem_total_kib=128 * 1024**2,
        swap_total_kib=8 * 1024**2,
        page_size=4096,
        kernel_args={},
        ttm_pages_limit=128 * 1024**3 // 4096,
        dmi_memory_bytes=128 * 1024**3,
        device_gids={"/dev/kfd": 109, "/dev/dri/renderD128": 110},
        current_group_ids=(109, 110),
        packages=(),
        apt_sources=(),
        dkms_status="",
        docker_version="fixture",
        dmesg="",
        dmesg_available=True,
        dedicated_vram_mib=512,
    )
