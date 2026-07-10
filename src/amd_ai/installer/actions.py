from __future__ import annotations

import json
import os
import pwd
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from amd_ai.doctor.checks import run_doctor
from amd_ai.doctor.models import DoctorReport
from amd_ai.host.adapters.base import select_adapter
from amd_ai.host.apply import execute_plan
from amd_ai.host.models import HostSnapshot, PreparePlan
from amd_ai.host.policy import evaluate_preflight
from amd_ai.host.prepare import create_prepare_plan, with_docker_group_action
from amd_ai.host.probe import HostProbe
from amd_ai.host.verify import evaluate_post_reboot
from amd_ai.image.build import (
    ROCM_PYTHON_TAG,
    STABLE_TORCH_TAG,
    build_rocm_python,
    build_rocm_pytorch,
    run_image_check,
)
from amd_ai.image.publish import DockerPublishRegistry
from amd_ai.installer.models import DiskSpaceEstimate, StageResult, StableRelease
from amd_ai.installer.release import (
    ReleaseDocker,
    VerifiedReleaseImages,
    load_stable_release,
    pull_and_verify_release,
)
from amd_ai.installer.state import stage_input_digest
from amd_ai.overlay.models import OverlayPaths
from amd_ai.overlay.transaction import initialize_overlay
from amd_ai.project.build import ProjectBuildResult, build_or_reuse_project
from amd_ai.project.config import ProjectConfig, load_project_config
from amd_ai.project.init import initialize_project as create_project
from amd_ai.project.run import (
    build_run_argv,
    ensure_project_home,
    inspect_project_image,
    load_project_protected_profile,
)
from amd_ai.project.runtime import (
    compute_shm_gib,
    discover_gpu_access,
    read_mem_total_kib,
)
from amd_ai.report import Report, Status
from amd_ai.runner import Runner, SubprocessRunner


REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
ARTIFACT_NAME_PATTERN = re.compile(r"[0-9a-f]{64}")
LOCAL_BUILD_ESTIMATE_BYTES = 32 * 1024**3
REQUIRED_LOCAL_BUILD_PATHS = (
    "images/common/container-check",
    "images/common/torch-manifest.py",
    "images/rocm-python/Dockerfile",
    "images/rocm-pytorch/Dockerfile",
    "profiles/base-images.lock",
    "profiles/rocm/7.2.1-packages.lock",
    "profiles/rocm/rocm.gpg",
    "profiles/torch/stable.env",
    "profiles/torch/stable.requirements.lock",
    "profiles/torch/stable.sources.env",
    "pyproject.toml",
)


class ActionError(RuntimeError):
    pass


@dataclass(frozen=True)
class HostPlanResult:
    snapshot: HostSnapshot
    plan: PreparePlan
    plan_digest: str
    adapter_id: str


@dataclass(frozen=True)
class LocalBuildResult:
    base_reference: str
    base_config_digest: str
    torch_reference: str
    torch_config_digest: str
    source_revision: str
    qualified: bool = False


@dataclass(frozen=True)
class ProjectInstallResult:
    config: ProjectConfig
    build: ProjectBuildResult


class AnonymousReleaseRegistry(DockerPublishRegistry):
    def pull(self, reference: str) -> None:
        self.authless_pull(reference)


class ProductionInstallerActions:
    def __init__(
        self,
        *,
        runner: Runner | None = None,
        root: Path = Path("/"),
        docker_prefix: Sequence[str] = ("docker",),
        release_docker: ReleaseDocker | None = None,
        effective_uid: int | None = None,
    ) -> None:
        if not docker_prefix or any(
            not isinstance(value, str) or not value or "\0" in value
            for value in docker_prefix
        ):
            raise ActionError("Docker command prefix is invalid")
        self.runner = runner or SubprocessRunner()
        self.root = Path(root)
        self.docker_prefix = tuple(docker_prefix)
        self.release_docker = release_docker or AnonymousReleaseRegistry(
            self.docker_prefix
        )
        self.effective_uid = (
            os.geteuid() if effective_uid is None else effective_uid
        )

    def bootstrap(self, *, source_root: Path, revision: str) -> StageResult:
        return StageResult(
            facts={
                "source_root": str(source_root.resolve()),
                "installer_source_revision": revision,
            }
        )

    def host_preflight(self, *, target_user: str | None = None) -> Report:
        return evaluate_preflight(self._snapshot(target_user=target_user))

    def host_plan(
        self,
        *,
        target_user: str,
        memory_gib: int | None = None,
    ) -> HostPlanResult:
        snapshot = self._snapshot(target_user=target_user)
        adapter = select_adapter(snapshot)
        if adapter is None:
            raise ActionError("no formal host write adapter is available")
        plan = create_prepare_plan(
            snapshot,
            target_user=target_user,
            memory_gib=memory_gib,
        )
        return HostPlanResult(
            snapshot=snapshot,
            plan=plan,
            plan_digest=stage_input_digest(prepare_plan_payload(plan)),
            adapter_id=adapter.adapter_id,
        )

    def host_apply(
        self,
        host_plan: HostPlanResult,
        *,
        include_docker_group: bool,
    ) -> StageResult:
        plan = (
            with_docker_group_action(host_plan.plan)
            if include_docker_group
            else host_plan.plan
        )
        result = execute_plan(
            plan,
            self.runner,
            effective_uid=self.effective_uid,
            confirmed=True,
            reboot=False,
            snapshot=host_plan.snapshot,
            root=self.root,
        )
        return StageResult(
            facts={
                "backup_path": str(result.backup_path),
                "executed_codes": list(result.executed_codes),
                "skipped_codes": list(result.skipped_codes),
                "reboot_required": plan.reboot_required,
                "docker_group_added": include_docker_group,
            }
        )

    def host_verify(self, *, image: str = ROCM_PYTHON_TAG) -> Report:
        del image
        return evaluate_post_reboot(self._snapshot())

    def container_host_check(self) -> StageResult:
        snapshot = self._snapshot()
        report = evaluate_preflight(snapshot)
        adapter = select_adapter(snapshot)
        reasons: list[str] = []
        if snapshot.docker_version is None:
            reasons.append("Docker daemon is unavailable")
        if report.status not in (Status.PASS, Status.UNVERIFIED):
            reasons.append(
                "host policy: "
                + ", ".join(finding.code for finding in report.findings)
            )
        has_kfd = "/dev/kfd" in snapshot.device_gids
        has_render = any(
            path.startswith("/dev/dri/render")
            for path in snapshot.device_gids
        )
        if not has_kfd or not has_render:
            reasons.append(
                f"GPU device mapping is incomplete (kfd={has_kfd}, render={has_render})"
            )
        missing_gids = sorted(
            set(snapshot.device_gids.values()).difference(
                snapshot.current_group_ids
            )
        )
        if missing_gids:
            reasons.append(
                "current user lacks GPU GIDs: "
                + ", ".join(str(value) for value in missing_gids)
            )
        return StageResult(
            facts={
                "adapter_id": None if adapter is None else adapter.adapter_id,
                "host_report": report.to_dict(),
                "docker_version": snapshot.docker_version,
                "device_gids": dict(snapshot.device_gids),
            },
            blocked=bool(reasons),
            message="; ".join(reasons),
        )

    def resolve_release(self, manifest_path: Path) -> StableRelease:
        return load_stable_release(manifest_path)

    def pull_release(self, release: StableRelease) -> VerifiedReleaseImages:
        return pull_and_verify_release(release, docker=self.release_docker)

    def build_local_images(
        self,
        *,
        source_root: Path,
        installer_source_revision: str,
    ) -> LocalBuildResult:
        source = validate_local_build_source(
            source_root,
            expected_revision=installer_source_revision,
        )
        base_reference, base_digest = build_rocm_python(repo_root=source)
        torch_reference, torch_digest = build_rocm_pytorch(
            profile_path=source / "profiles/torch/stable.env",
            allow_experimental=False,
            repo_root=source,
        )
        validate_local_build_source(
            source,
            expected_revision=installer_source_revision,
        )
        return LocalBuildResult(
            base_reference=base_reference,
            base_config_digest=base_digest,
            torch_reference=torch_reference,
            torch_config_digest=torch_digest,
            source_revision=installer_source_revision,
        )

    def verify_torch_image(self, image: str) -> StageResult:
        returncode = run_image_check(
            image=image,
            mode="torch",
            metadata_only=False,
            runtime=True,
            json_path="-",
        )
        return StageResult(
            facts={"image": image, "runtime_returncode": returncode},
            blocked=returncode != 0,
            message=(
                "gfx1151 Torch runtime verification failed"
                if returncode != 0
                else ""
            ),
        )

    def image_disk_estimate(
        self, *, release: StableRelease, image_source: str
    ) -> DiskSpaceEstimate:
        location, available = _docker_root_and_available(
            self.runner, self.docker_prefix
        )
        if image_source == "pull":
            payload = _missing_release_layer_bytes(
                release, self.runner, self.docker_prefix
            )
        elif image_source == "build":
            payload = LOCAL_BUILD_ESTIMATE_BYTES
        else:
            raise ActionError("image source must be pull or build")
        return DiskSpaceEstimate(
            location=location,
            payload_bytes=payload,
            available_bytes=available,
        )

    def project_disk_estimate(
        self, *, project_dir: Path
    ) -> DiskSpaceEstimate:
        project = Path(project_dir).expanduser().resolve(strict=False)
        artifacts = project / ".amd-ai/artifacts/sha256"
        payload = 0
        if artifacts.is_dir() and not artifacts.is_symlink():
            for artifact in artifacts.iterdir():
                if (
                    ARTIFACT_NAME_PATTERN.fullmatch(artifact.name) is not None
                    and artifact.is_file()
                    and not artifact.is_symlink()
                ):
                    payload += artifact.stat().st_size
        probe = _nearest_existing_path(project)
        try:
            available = shutil.disk_usage(probe).free
        except OSError as error:
            raise ActionError(
                f"cannot inspect project filesystem space at {probe}: {error}"
            ) from error
        return DiskSpaceEstimate(
            location=project,
            payload_bytes=payload,
            available_bytes=available,
        )

    def initialize_project(
        self,
        *,
        project_dir: Path,
        project_name: str,
        base_profile: str = "stable",
        base_image_reference: str | None = None,
        base_config_digest: str | None = None,
        target_user: str | None = None,
        owner_uid: int | None = None,
        owner_gid: int | None = None,
    ) -> ProjectInstallResult:
        if base_profile != "stable":
            raise ActionError("the installer only initializes the stable profile")
        if base_image_reference is None or base_config_digest is None:
            raise ActionError("selected parent image identity is unavailable")
        bind_selected_parent(
            reference=base_image_reference,
            config_digest=base_config_digest,
            runner=self.runner,
            docker_prefix=self.docker_prefix,
        )
        if (owner_uid is None) != (owner_gid is None):
            raise ActionError("project owner UID and GID must be supplied together")
        if owner_uid is None or owner_gid is None:
            owner_uid, owner_gid = _user_identity(target_user)
        config_path = project_dir / "amd-ai-project.toml"
        if config_path.exists():
            config = load_project_config(config_path)
            if config.base_digest != base_config_digest:
                raise ActionError(
                    "existing project parent config digest differs from selection"
                )
        else:
            create_project(
                name=project_name,
                destination=project_dir,
                base_profile=base_profile,
                runner=self.runner,
                docker_prefix=self.docker_prefix,
                owner_uid=owner_uid,
                owner_gid=owner_gid,
            )
            config = load_project_config(config_path)
        build = build_or_reuse_project(
            config=config,
            runner=self.runner,
            force=False,
            no_build=False,
            docker_prefix=self.docker_prefix,
        )
        metadata = inspect_project_image(
            config, self.runner, self.docker_prefix
        )
        if metadata.profile_status != "verified":
            raise ActionError(
                "installer project parent must use the verified Torch profile"
            )
        ensure_project_home(
            config.path.parent, uid=owner_uid, gid=owner_gid
        )
        profile = load_project_protected_profile(
            config=config,
            metadata=metadata,
            runner=self.runner,
            docker_prefix=self.docker_prefix,
        )
        initialize_overlay(
            OverlayPaths.for_project(config.path.parent), profile=profile
        )
        return ProjectInstallResult(config=config, build=build)

    def verify_project(
        self,
        *,
        project_dir: Path,
        manifest_path: Path,
        qualified: bool = True,
        target_user: str | None = None,
    ) -> StageResult:
        config = load_project_config(project_dir / "amd-ai-project.toml")
        metadata = inspect_project_image(
            config, self.runner, self.docker_prefix
        )
        if metadata.profile_status != "verified":
            return StageResult(
                blocked=True,
                message="project image does not use the verified Torch profile",
            )
        uid, gid = _user_identity(target_user)
        ensure_project_home(config.path.parent, uid=uid, gid=gid)
        access = discover_gpu_access()
        shm_gib = config.shm_size_gib or compute_shm_gib(
            mem_total_kib=read_mem_total_kib()
        )
        probe_config = replace(config, command=("/bin/true",))
        argv = build_run_argv(
            config=probe_config,
            access=access,
            uid=uid,
            gid=gid,
            shm_gib=shm_gib,
            environment=os.environ,
            terminal=False,
            docker_prefix=self.docker_prefix,
        )
        result = self.runner.run(list(argv), check=False)
        if result.returncode != 0:
            evidence = result.stderr.strip() or result.stdout.strip() or "no output"
            return StageResult(
                facts={"managed_startup_returncode": result.returncode},
                blocked=True,
                message=f"managed project startup verification failed: {evidence}",
            )
        if not qualified:
            return StageResult(
                facts={
                    "managed_startup_returncode": 0,
                    "release_status": "local-unqualified",
                }
            )
        report = run_doctor(project_dir, manifest_path)
        blocked = report.status not in {"pass", "warning"}
        return StageResult(
            facts={
                "managed_startup_returncode": 0,
                "doctor": report.to_dict(),
            },
            blocked=blocked,
            message="project verification failed" if blocked else "",
        )

    def doctor(
        self, *, project: Path | None, manifest_path: Path
    ) -> DoctorReport:
        return run_doctor(project, manifest_path)

    def _snapshot(self, *, target_user: str | None = None) -> HostSnapshot:
        group_ids = (
            None
            if target_user is None
            else _target_user_group_ids(target_user)
        )
        return HostProbe(
            root=self.root,
            runner=self.runner,
            current_group_ids=group_ids,
            docker_prefix=self.docker_prefix,
            dmesg_fallback=("sudo", "-n", "dmesg", "--color=never"),
        ).collect()


def prepare_plan_payload(plan: PreparePlan) -> dict[str, object]:
    return {
        "supported": plan.supported,
        "target_user": plan.target_user,
        "actions": [asdict(action) for action in plan.actions],
        "reboot_required": plan.reboot_required,
    }


def validate_local_build_source(
    source_root: Path,
    *,
    expected_revision: str,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    raw = Path(source_root).expanduser()
    if raw.is_symlink():
        raise ActionError("local build source root must not be a symlink")
    try:
        source = raw.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ActionError(f"cannot resolve local build source: {error}") from error
    if not source.is_dir():
        raise ActionError("local build source is not a directory")
    if REVISION_PATTERN.fullmatch(expected_revision) is None:
        raise ActionError("expected installer source revision is invalid")
    missing = [
        relative
        for relative in REQUIRED_LOCAL_BUILD_PATHS
        if not (source / relative).is_file()
        or (source / relative).is_symlink()
    ]
    if missing:
        raise ActionError(
            "local build source is incomplete: " + ", ".join(missing)
        )

    revision = _run_git(
        run,
        ("git", "-C", str(source), "rev-parse", "HEAD"),
        "read local build source revision",
    ).stdout.strip()
    if revision != expected_revision:
        raise ActionError(
            "local build source revision differs from installer state: "
            f"expected {expected_revision}, got {revision or '<unknown>'}"
        )
    status = _run_git(
        run,
        ("git", "-C", str(source), "status", "--porcelain"),
        "inspect local build source status",
    )
    if status.stdout.strip():
        raise ActionError("local build source checkout is not clean")
    return source


def _run_git(
    run: Callable[..., subprocess.CompletedProcess[str]],
    argv: tuple[str, ...],
    operation: str,
) -> subprocess.CompletedProcess[str]:
    try:
        result = run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ActionError(f"cannot {operation}: {error}") from error
    if result.returncode != 0:
        evidence = result.stderr.strip() or result.stdout.strip() or "no output"
        raise ActionError(f"cannot {operation}: {evidence}")
    return result


def _target_user_group_ids(target_user: str) -> tuple[int, ...]:
    try:
        record = pwd.getpwnam(target_user)
        return tuple(
            sorted(set(os.getgrouplist(target_user, record.pw_gid)))
        )
    except (KeyError, OSError) as error:
        raise ActionError(
            f"cannot resolve groups for target user {target_user!r}"
        ) from error


def bind_selected_parent(
    *,
    reference: str,
    config_digest: str,
    runner: Runner,
    docker_prefix: Sequence[str] = ("docker",),
) -> None:
    if (
        not reference
        or reference.startswith("-")
        or "\0" in reference
        or any(character.isspace() for character in reference)
    ):
        raise ActionError("selected parent image reference is invalid")
    if DIGEST_PATTERN.fullmatch(config_digest) is None:
        raise ActionError("selected parent config digest is invalid")
    inspected = runner.run(
        [
            *docker_prefix,
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            reference,
        ],
        check=False,
    )
    if inspected.returncode != 0 or inspected.stdout.strip() != config_digest:
        evidence = (
            inspected.stderr.strip()
            or inspected.stdout.strip()
            or "image is missing"
        )
        raise ActionError(
            "selected parent config digest does not match exact reference: "
            + evidence
        )
    tagged = runner.run(
        [*docker_prefix, "tag", reference, STABLE_TORCH_TAG],
        check=False,
    )
    if tagged.returncode != 0:
        evidence = tagged.stderr.strip() or tagged.stdout.strip() or "no output"
        raise ActionError(f"cannot bind selected parent alias: {evidence}")
    alias = runner.run(
        [
            *docker_prefix,
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            STABLE_TORCH_TAG,
        ],
        check=False,
    )
    if alias.returncode != 0 or alias.stdout.strip() != config_digest:
        raise ActionError("selected parent alias does not preserve config digest")


def _user_identity(target_user: str | None) -> tuple[int, int]:
    if target_user is None:
        return os.getuid(), os.getgid()
    try:
        record = pwd.getpwnam(target_user)
    except KeyError as error:
        raise ActionError(f"cannot resolve target user {target_user!r}") from error
    return record.pw_uid, record.pw_gid


def _docker_root_and_available(
    runner: Runner, docker_prefix: Sequence[str]
) -> tuple[Path, int]:
    try:
        result = runner.run(
            [
                *docker_prefix,
                "info",
                "--format",
                "{{.DockerRootDir}}",
            ],
            check=False,
        )
    except OSError as error:
        raise ActionError(f"cannot inspect Docker data root: {error}") from error
    raw = result.stdout.strip()
    if (
        result.returncode != 0
        or not raw
        or "\0" in raw
        or not Path(raw).is_absolute()
    ):
        evidence = result.stderr.strip() or raw or "no output"
        raise ActionError(f"cannot inspect Docker data root: {evidence}")
    root = Path(raw).resolve(strict=False)
    probe = _nearest_existing_path(root)
    try:
        available = shutil.disk_usage(probe).free
    except OSError as error:
        raise ActionError(
            f"cannot inspect Docker filesystem space at {probe}: {error}"
        ) from error
    return root, available


def _missing_release_layer_bytes(
    release: StableRelease,
    runner: Runner,
    docker_prefix: Sequence[str],
) -> int:
    layers: dict[str, int] = {}
    for image in (release.base, release.torch):
        local = runner.run(
            [
                *docker_prefix,
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                image.reference,
            ],
            check=False,
        )
        if local.returncode == 0 and local.stdout.strip() == image.config_digest:
            continue
        manifest = runner.run(
            [*docker_prefix, "manifest", "inspect", "--verbose", image.reference],
            check=False,
        )
        if manifest.returncode != 0:
            evidence = (
                manifest.stderr.strip()
                or manifest.stdout.strip()
                or "no output"
            )
            raise ActionError(
                f"cannot estimate remote image layers for {image.reference}: {evidence}"
            )
        try:
            payload = json.loads(manifest.stdout)
        except json.JSONDecodeError as error:
            raise ActionError(
                f"cannot parse remote image layers for {image.reference}"
            ) from error
        for digest, size in _manifest_layers(payload):
            previous = layers.get(digest)
            if previous is not None and previous != size:
                raise ActionError(
                    f"remote layer size is ambiguous for {digest}"
                )
            layers[digest] = size
    return sum(layers.values())


def _manifest_layers(payload: object) -> tuple[tuple[str, int], ...]:
    if not isinstance(payload, dict):
        raise ActionError("remote image manifest is not an object")
    manifest = payload.get("SchemaV2Manifest", payload)
    if not isinstance(manifest, dict):
        raise ActionError("remote image manifest body is invalid")
    raw_layers = manifest.get("layers")
    if not isinstance(raw_layers, list):
        raise ActionError("remote image manifest has no layer list")
    layers: list[tuple[str, int]] = []
    for raw_layer in raw_layers:
        if not isinstance(raw_layer, dict):
            raise ActionError("remote image layer record is invalid")
        digest = raw_layer.get("digest")
        size = raw_layer.get("size")
        if (
            not isinstance(digest, str)
            or DIGEST_PATTERN.fullmatch(digest) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ActionError("remote image layer identity is invalid")
        layers.append((digest, size))
    return tuple(layers)


def _nearest_existing_path(path: Path) -> Path:
    probe = path
    while not probe.exists():
        if probe.parent == probe:
            raise ActionError(f"no existing filesystem ancestor for {path}")
        probe = probe.parent
    return probe
