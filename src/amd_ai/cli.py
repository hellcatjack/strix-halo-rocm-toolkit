from __future__ import annotations

import argparse
import getpass
import json
import os
import pwd
import re
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from amd_ai import __version__
from amd_ai.doctor.checks import run_doctor
from amd_ai.doctor.models import DiagnosticDisposition, DoctorModelError
from amd_ai.doctor.repair import (
    RepairExecutionError,
    RepairPlanningError,
    SystemRepairExecutor,
    execute_repair,
    plan_repair,
)
from amd_ai.host.apply import ApplyError, ApplyRefused, execute_plan
from amd_ai.host.models import HostSnapshot, PlannedAction, PreparePlan
from amd_ai.host.policy import evaluate_preflight
from amd_ai.host.prepare import (
    HostPlanningError,
    UnsupportedHostError,
    create_prepare_plan,
    with_docker_group_action,
)
from amd_ai.host.probe import FixtureRunner, HostProbe, load_fixture_device_gids
from amd_ai.host.verify import verify_host
from amd_ai.image.build import (
    BuildError,
    Docker,
    build_rocm_python,
    build_rocm_pytorch,
    prune_images,
    run_image_check,
    ROCM_PYTHON_TAG,
    STABLE_TORCH_TAG,
)
from amd_ai.image.publish import (
    DockerPublishRegistry,
    PublishError,
    observe_pushed_release,
    publish_images,
    publish_stable_release,
    validate_publish_inputs,
    verify_publish_candidate_local_images,
    write_observed_release,
)
from amd_ai.installer.actions import ProductionInstallerActions
from amd_ai.installer.models import (
    InstallMode,
    InstallOptions,
    InstallerModelError,
    default_state_path,
)
from amd_ai.installer.prompts import (
    NonInteractivePrompts,
    PromptError,
    TerminalPrompts,
)
from amd_ai.installer.release import (
    ReleaseError,
    load_stable_release,
    pull_and_verify_release,
)
from amd_ai.installer.workflow import InstallerWorkflow
from amd_ai.overlay.models import (
    PROTECTED_DISTRIBUTIONS,
    OverlayError,
    OverlayPaths,
    ProtectedComponent,
    ProtectedProfile,
)
from amd_ai.overlay.transaction import TransactionError, initialize_overlay
from amd_ai.project.build import ProjectBuildError, build_or_reuse_project
from amd_ai.project.config import ConfigError, load_project_config
from amd_ai.project.dependencies import (
    DependencyError,
    lock_project_dependencies_in_container,
)
from amd_ai.project.init import ProjectInitError, initialize_project
from amd_ai.project.run import (
    ProjectRunError,
    build_run_argv,
    ensure_project_home,
    inspect_project_image,
    load_project_protected_profile,
    redact_run_argv,
    require_profile_allowed,
)
from amd_ai.project.runtime import (
    RuntimePolicyError,
    compute_shm_gib,
    discover_gpu_access,
    read_mem_total_kib,
)
from amd_ai.qualification.models import ProfileError as QualificationProfileError
from amd_ai.qualification.run import (
    QualificationError,
    run_profile as run_qualification_profile,
)
from amd_ai.qualification.release import main as qualification_release_main
from amd_ai.report import Report, Status
from amd_ai.runner import Runner, SubprocessRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="amd-ai")
    parser.add_argument(
        "--version",
        action="version",
        version=f"amd-ai {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    install = subparsers.add_parser("install")
    install.add_argument("--mode", choices=("full", "container"))
    install.add_argument("--non-interactive", action="store_true")
    install.add_argument("--dry-run", action="store_true")
    install.add_argument("--project-dir", type=Path)
    install.add_argument("--project-name", default="amd-ai-project")
    install.add_argument("--image-source", choices=("pull", "build"))
    install.add_argument("--target-user")
    install.add_argument("--accept-host-plan-digest")
    install.add_argument("--accept-docker-group", action="store_true")
    install.add_argument("--manifest", type=Path)
    install.add_argument("--source-root", type=Path)
    install.add_argument("--state-path", type=Path)
    preflight = subparsers.add_parser("host-preflight")
    preflight.add_argument("--json", type=Path, dest="json_path")
    preflight.add_argument(
        "--fixture-root",
        type=Path,
        help=argparse.SUPPRESS,
    )
    prepare = subparsers.add_parser("host-prepare")
    prepare_modes = prepare.add_subparsers(dest="prepare_mode", required=True)
    for mode in ("plan", "apply"):
        mode_parser = prepare_modes.add_parser(mode)
        mode_parser.add_argument("--target-user")
        mode_parser.add_argument("--memory-gib", type=_positive_int)
        mode_parser.add_argument("--json", type=Path, dest="json_path")
        mode_parser.add_argument(
            "--fixture-root",
            type=Path,
            help=argparse.SUPPRESS,
        )
        if mode == "apply":
            mode_parser.add_argument("--yes", action="store_true")
            mode_parser.add_argument("--reboot", action="store_true")
    verify = subparsers.add_parser("host-verify")
    verify.add_argument(
        "--probe-image",
        default="rocm-python:7.2.1-py3.12",
    )
    verify.add_argument("--json", type=Path, dest="json_path")
    verify.add_argument(
        "--fixture-root",
        type=Path,
        help=argparse.SUPPRESS,
    )
    image_build = subparsers.add_parser("image-build")
    image_modes = image_build.add_subparsers(dest="image_mode", required=True)
    image_modes.add_parser("rocm-python")
    pytorch = image_modes.add_parser("rocm-pytorch")
    pytorch.add_argument(
        "--profile",
        type=Path,
        default=Path("profiles/torch/stable.env"),
    )
    pytorch.add_argument("--allow-experimental", action="store_true")
    prune = image_modes.add_parser("prune")
    prune.add_argument("--apply", action="store_true")
    prune.add_argument("--older-than-hours", type=_positive_int, default=168)
    prune.add_argument("--project-root", type=Path, action="append")

    container_check = subparsers.add_parser("container-check")
    container_check.add_argument(
        "--image",
        default="rocm-pytorch:7.2.1-py3.12-torch2.9.1",
    )
    container_check.add_argument("--mode", choices=("rocm", "torch"), default="torch")
    check_kind = container_check.add_mutually_exclusive_group()
    check_kind.add_argument("--metadata-only", action="store_true")
    check_kind.add_argument("--runtime", action="store_true")
    container_check.add_argument("--json", dest="json_path")
    container_check.add_argument("--suite", choices=("stable",))
    container_check.add_argument(
        "--profile",
        type=Path,
        default=Path("profiles/qualification/stable.toml"),
    )

    project_init = subparsers.add_parser("project-init")
    _add_project_init_arguments(project_init)

    project_lock = subparsers.add_parser("project-lock")
    _add_project_lock_arguments(project_lock)

    project_run = subparsers.add_parser("project-run")
    _add_project_run_arguments(project_run)

    project = subparsers.add_parser("project")
    project_modes = project.add_subparsers(
        dest="project_command", required=True
    )
    _add_project_init_arguments(project_modes.add_parser("init"))
    _add_project_lock_arguments(project_modes.add_parser("lock"))
    _add_project_run_arguments(project_modes.add_parser("run"))

    gpu_release = subparsers.add_parser("gpu-release")
    gpu_release.add_argument("--qualification", type=Path, required=True)
    gpu_release.add_argument("--image", required=True)
    gpu_release.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/releases"),
    )

    release = subparsers.add_parser("release")
    release_modes = release.add_subparsers(dest="release_mode", required=True)
    release_verify = release_modes.add_parser("verify")
    release_verify.add_argument(
        "--manifest",
        dest="manifest_path",
        type=Path,
        default=Path("profiles/releases/stable.json"),
    )
    release_publish = release_modes.add_parser("publish")
    release_publish.add_argument("--release-id", required=True)
    release_publish.add_argument(
        "--qualification", type=Path, required=True
    )
    release_publish.add_argument("--sbom", type=Path, required=True)
    release_publish.add_argument(
        "--output",
        type=Path,
        default=Path("profiles/releases/stable.json"),
    )
    release_publish.add_argument(
        "--publish-report",
        type=Path,
        default=Path("reports/publish-candidate.json"),
    )
    publication_stage = release_publish.add_mutually_exclusive_group()
    publication_stage.add_argument("--dry-run", action="store_true")
    publication_stage.add_argument("--push-only", action="store_true")

    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("project", nargs="?", type=Path)
    doctor.add_argument(
        "--manifest",
        type=Path,
        default=Path("profiles/releases/stable.json"),
    )
    doctor.add_argument("--json", dest="json_path", type=Path)

    repair = subparsers.add_parser("repair")
    repair.add_argument("project", type=Path)
    repair.add_argument(
        "--manifest",
        type=Path,
        default=Path("profiles/releases/stable.json"),
    )
    repair.add_argument("--yes", action="store_true")
    repair.add_argument("--json", dest="json_path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "install":
        try:
            return _install_command(args)
        except (
            InstallerModelError,
            OSError,
            PromptError,
            ValueError,
        ) as error:
            print(f"install: {error}", file=sys.stderr)
            return 2
    if args.command == "host-preflight":
        return _host_preflight(args.fixture_root, args.json_path)
    if args.command == "host-prepare":
        return _host_prepare(args)
    if args.command == "host-verify":
        return _host_verify(args.fixture_root, args.probe_image, args.json_path)
    if args.command == "image-build":
        try:
            return _image_build(args)
        except BuildError as error:
            print(f"image-build: {error}", file=sys.stderr)
            return 2
    if args.command == "container-check":
        try:
            if args.suite == "stable":
                return _qualification_suite(args)
            return run_image_check(
                image=args.image,
                mode=args.mode,
                metadata_only=args.metadata_only,
                runtime=args.runtime,
                json_path=args.json_path,
            )
        except (
            BuildError,
            ProjectRunError,
            QualificationError,
            QualificationProfileError,
            RuntimePolicyError,
        ) as error:
            print(f"container-check: {error}", file=sys.stderr)
            return 2
    if args.command in {"project", "project-init", "project-lock", "project-run"}:
        try:
            project_command = (
                args.project_command
                if args.command == "project"
                else args.command.removeprefix("project-")
            )
            if project_command == "init":
                return _project_init(args)
            if project_command == "lock":
                return _project_lock(args)
            return _project_run(args)
        except (
            BuildError,
            ConfigError,
            DependencyError,
            ProjectBuildError,
            ProjectInitError,
            ProjectRunError,
            RuntimePolicyError,
            OverlayError,
            TransactionError,
        ) as error:
            print(f"{args.command}: {error}", file=sys.stderr)
            return 2
    if args.command == "gpu-release":
        return _gpu_release(args)
    if args.command == "doctor":
        try:
            report = run_doctor(args.project, args.manifest)
        except (
            BuildError,
            ConfigError,
            DoctorModelError,
            OSError,
            PublishError,
            ReleaseError,
            RuntimePolicyError,
            TransactionError,
        ) as error:
            print(f"doctor: {error}", file=sys.stderr)
            return 2
        for diagnostic in report.diagnostics:
            if diagnostic.disposition != DiagnosticDisposition.PASS:
                print(
                    f"{diagnostic.code} [{diagnostic.disposition.value}]: "
                    f"{diagnostic.summary}"
                )
        if args.json_path is not None:
            args.json_path.parent.mkdir(parents=True, exist_ok=True)
            args.json_path.write_text(
                json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if report.status in {"pass", "warning"}:
            return 0
        return 1 if report.status == "repairable" else 2
    if args.command == "repair":
        try:
            return _repair_command(args)
        except (
            BuildError,
            ConfigError,
            DoctorModelError,
            OSError,
            PublishError,
            ReleaseError,
            RepairExecutionError,
            RepairPlanningError,
            RuntimePolicyError,
            TransactionError,
        ) as error:
            print(f"repair: {error}", file=sys.stderr)
            return 2
    if args.command == "release":
        try:
            if args.release_mode == "verify":
                return verify_release_command(
                    manifest_path=args.manifest_path
                )
            return publish_release_command(
                release_id=args.release_id,
                qualification_path=args.qualification,
                sbom_path=args.sbom,
                output=args.output,
                publish_report=args.publish_report,
                dry_run=args.dry_run,
                push_only=args.push_only,
            )
        except (BuildError, PublishError, ReleaseError, OSError) as error:
            print(
                "release: " + _redact_release_error(str(error)),
                file=sys.stderr,
            )
            return 2
    raise AssertionError(f"unhandled command: {args.command}")


def _add_project_init_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("name")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--base-profile", default="stable")


def _add_project_lock_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("project", type=Path)


def _add_project_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("project", type=Path)
    build_mode = parser.add_mutually_exclusive_group()
    build_mode.add_argument("--build", action="store_true")
    build_mode.add_argument("--no-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--shm-size-gib", type=_shm_gib)


def _install_command(args: argparse.Namespace) -> int:
    prompts = (
        NonInteractivePrompts()
        if args.non_interactive
        else TerminalPrompts()
    )
    mode = InstallMode(args.mode) if args.mode is not None else None
    if mode is None:
        if args.non_interactive:
            raise InstallerModelError(
                "--mode is required with --non-interactive"
            )
        selected = prompts.choose_mode()
        if selected is None:
            return 0
        if selected is InstallMode.DOCTOR:
            return _interactive_doctor(args, prompts)
        mode = selected

    source_root = (
        args.source_root
        or Path(os.environ.get("AMD_AI_TOOLKIT_ROOT", ""))
        or Path(__file__).resolve().parents[2]
    )
    if str(source_root) == ".":
        source_root = Path(__file__).resolve().parents[2]
    manifest = args.manifest or source_root / "profiles/releases/stable.json"
    state_path = args.state_path or default_state_path()
    options = InstallOptions(
        mode=mode,
        non_interactive=args.non_interactive,
        dry_run=args.dry_run,
        project_dir=args.project_dir,
        project_name=args.project_name,
        image_source=args.image_source,
        target_user=args.target_user,
        accepted_host_plan_digest=args.accept_host_plan_digest,
        accept_docker_group=args.accept_docker_group,
        stable_manifest_path=manifest,
        source_root=source_root,
        state_path=state_path,
        state_path_explicit=args.state_path is not None,
    )
    revision = _installer_source_revision(source_root)
    fixture_root = os.environ.get("AMD_AI_INSTALLER_FIXTURE_ROOT")
    workflow_arguments: dict[str, object] = {}
    if fixture_root:
        if os.environ.get("AMD_AI_INSTALLER_ENABLE_FIXTURES") != "1":
            raise InstallerModelError(
                "installer fixture backend was not explicitly enabled"
            )
        from amd_ai.installer.fixture import FixtureInstallerActions

        actions = FixtureInstallerActions(Path(fixture_root))
        workflow_arguments["boot_id_reader"] = actions.read_boot_id
    else:
        try:
            docker_prefix = Docker.detect().prefix
        except BuildError:
            docker_prefix = ("docker",)
        actions = ProductionInstallerActions(
            non_interactive=args.non_interactive,
            docker_prefix=docker_prefix,
        )
    workflow = InstallerWorkflow(
        options=options,
        actions=actions,
        installer_version=__version__,
        installer_source_revision=revision,
        prompts=prompts,
        **workflow_arguments,
    )
    result = workflow.run()
    if result.message:
        stream = sys.stderr if result.exit_code == 2 else sys.stdout
        print(result.message, file=stream)
    return result.exit_code


def _interactive_doctor(
    args: argparse.Namespace, prompts: TerminalPrompts
) -> int:
    source_root = args.source_root or Path(__file__).resolve().parents[2]
    manifest = args.manifest or source_root / "profiles/releases/stable.json"
    report = run_doctor(args.project_dir, manifest)
    for diagnostic in report.diagnostics:
        if diagnostic.disposition != DiagnosticDisposition.PASS:
            print(
                f"{diagnostic.code} [{diagnostic.disposition.value}]: "
                f"{diagnostic.summary}"
            )
    if (
        report.status == "repairable"
        and args.project_dir is not None
        and prompts.confirm_exact("REPAIR")
    ):
        return _repair_command(
            argparse.Namespace(
                project=args.project_dir,
                manifest=manifest,
                yes=True,
                json_path=None,
            )
        )
    if report.status in {"pass", "warning"}:
        return 0
    return 1 if report.status == "repairable" else 2


def _installer_source_revision(source_root: Path) -> str:
    environment_value = os.environ.get("AMD_AI_INSTALLER_SOURCE_REVISION", "")
    if re.fullmatch(r"[0-9a-f]{40}", environment_value):
        return environment_value
    runtime_root = Path(os.environ.get("AMD_AI_TOOLKIT_ROOT", source_root))
    metadata = runtime_root / ".installer-runtime.json"
    if metadata.is_file() and not metadata.is_symlink():
        try:
            payload = json.loads(metadata.read_text(encoding="ascii"))
        except (OSError, json.JSONDecodeError) as error:
            raise InstallerModelError(
                f"cannot read installer runtime identity: {error}"
            ) from error
        revision = payload.get("installer_source_revision")
        if isinstance(revision, str) and re.fullmatch(
            r"[0-9a-f]{40}", revision
        ):
            return revision
        raise InstallerModelError("installer runtime identity is invalid")
    completed = subprocess.run(
        ("git", "-C", str(source_root), "rev-parse", "HEAD"),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    revision = completed.stdout.strip()
    if completed.returncode != 0 or re.fullmatch(
        r"[0-9a-f]{40}", revision
    ) is None:
        raise InstallerModelError(
            "cannot determine installer source revision"
        )
    return revision


def _repair_command(args: argparse.Namespace) -> int:
    report = run_doctor(args.project, args.manifest)
    if args.json_path is not None:
        pre_path, _ = _repair_report_paths(args.json_path)
        _write_doctor_json(pre_path, report.to_dict())
    plan = plan_repair(report)
    for action in plan.actions:
        print(
            f"{action.kind}: {action.exact_target} "
            f"[{action.reason_code}]"
        )
    if plan.blocked or not plan.actions:
        reasons = ", ".join(plan.blocked_reasons) or "no repairable actions"
        print(f"repair blocked: {reasons}", file=sys.stderr)
        return 2
    if not args.yes:
        try:
            confirmation = input("Type REPAIR to execute these exact actions: ")
        except EOFError:
            confirmation = ""
        if confirmation != "REPAIR":
            print("repair cancelled", file=sys.stderr)
            return 2
    executor = SystemRepairExecutor(manifest_path=args.manifest)
    post_report = execute_repair(plan, executor=executor)
    if args.json_path is not None:
        _, post_path = _repair_report_paths(args.json_path)
        _write_doctor_json(post_path, post_report.to_dict())
    return 0


def _repair_report_paths(path: Path) -> tuple[Path, Path]:
    suffix = path.suffix or ".json"
    stem = path.name[: -len(path.suffix)] if path.suffix else path.name
    return (
        path.with_name(f"{stem}.pre{suffix}"),
        path.with_name(f"{stem}.post{suffix}"),
    )


def _write_doctor_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RepairExecutionError(
            f"repair report path already exists: {path}"
        )
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _image_build(args: argparse.Namespace) -> int:
    if args.image_mode == "rocm-python":
        tag, image_id = build_rocm_python()
        print(f"built {tag} {image_id}")
        return 0
    if args.image_mode == "rocm-pytorch":
        tag, image_id = build_rocm_pytorch(
            profile_path=args.profile,
            allow_experimental=args.allow_experimental,
        )
        print(f"built {tag} {image_id}")
        return 0
    if args.image_mode == "prune":
        prune_images(
            apply=args.apply,
            older_than_hours=args.older_than_hours,
            project_roots=args.project_root,
        )
        return 0
    raise AssertionError(f"unhandled image mode: {args.image_mode}")


def _project_init(args: argparse.Namespace) -> int:
    docker = Docker.detect()
    destination = args.directory or Path(args.name)
    project_dir = initialize_project(
        name=args.name,
        destination=destination,
        base_profile=args.base_profile,
        runner=SubprocessRunner(),
        docker_prefix=docker.prefix,
    )
    print(f"initialized {project_dir}")
    return 0


def _qualification_suite(args: argparse.Namespace) -> int:
    if args.metadata_only or args.runtime:
        raise QualificationError(
            "--suite cannot be combined with --metadata-only or --runtime"
        )
    docker = Docker.detect()
    runner = SubprocessRunner()
    access = discover_gpu_access()
    uid, gid = _runtime_identity()
    output_path = (
        None
        if args.json_path in {None, "-"}
        else Path(args.json_path)
    )
    report = run_qualification_profile(
        profile_path=args.profile,
        output_path=output_path,
        runner=runner,
        docker_prefix=docker.prefix,
        gids=access.group_ids,
        uid=uid,
        gid=gid,
    )
    if output_path is None:
        print(json.dumps(report.to_dict(), sort_keys=True))
    return 0 if report.status == "pass" else 2


def _gpu_release(args: argparse.Namespace) -> int:
    return qualification_release_main(
        [
            "--qualification",
            str(args.qualification),
            "--image",
            args.image,
            "--output-dir",
            str(args.output_dir),
        ]
    )


def verify_release_command(*, manifest_path: Path) -> int:
    release = load_stable_release(manifest_path)
    docker = Docker.detect()
    registry = DockerPublishRegistry(docker.prefix)
    pull_and_verify_release(release, docker=registry)
    print(f"verified {release.base.reference}")
    print(f"verified {release.torch.reference}")
    return 0


def publish_release_command(
    *,
    release_id: str,
    qualification_path: Path,
    sbom_path: Path,
    output: Path,
    publish_report: Path,
    dry_run: bool,
    push_only: bool,
) -> int:
    revision = _clean_release_revision()
    docker = Docker.detect()
    base_image_id = docker.image_id(ROCM_PYTHON_TAG)
    torch_image_id = docker.image_id(STABLE_TORCH_TAG)
    assert base_image_id is not None and torch_image_id is not None
    registry = DockerPublishRegistry(docker.prefix)
    candidate = validate_publish_inputs(
        release_id=release_id,
        qualification_path=qualification_path,
        sbom_path=sbom_path,
        current_revision=revision,
        base_image_id=base_image_id,
        torch_image_id=torch_image_id,
    )
    candidate = verify_publish_candidate_local_images(
        candidate, registry=registry
    )
    if dry_run:
        print(
            f"release evidence valid for {candidate.source_revision} "
            f"({candidate.torch_local_id})"
        )
        return 0
    if push_only:
        observed = publish_images(candidate, registry=registry)
        write_observed_release(publish_report, observed)
        print(f"published {observed.base.reference}")
        print(f"published {observed.torch.reference}")
        print(f"publication report: {publish_report}")
        return 0

    observed = observe_pushed_release(publish_report)
    release = publish_stable_release(
        candidate,
        registry=registry,
        output=output,
        observed=observed,
    )
    print(f"anonymous pull verified {release.base.reference}")
    print(f"anonymous pull verified {release.torch.reference}")
    print(f"stable release manifest: {output}")
    return 0


def _clean_release_revision() -> str:
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if status.returncode != 0:
        raise PublishError(
            "cannot inspect source worktree: "
            + (status.stderr.strip() or "git status failed")
        )
    if status.stdout.strip():
        raise PublishError("tracked source files are modified")
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    value = revision.stdout.strip()
    if revision.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise PublishError("cannot determine exact source revision")
    return value


def _redact_release_error(value: str) -> str:
    return re.sub(
        r"(https?://)[^\s/@]+@",
        r"\1<redacted>@",
        value,
    )


def _project_lock(args: argparse.Namespace) -> int:
    config = load_project_config(_project_config_path(args.project))
    docker = Docker.detect()
    uid, gid = _runtime_identity()
    lock_path = lock_project_dependencies_in_container(
        project_dir=config.path.parent,
        base_image=config.base_image,
        uid=uid,
        gid=gid,
        runner=SubprocessRunner(),
        docker_prefix=docker.prefix,
    )
    print(f"locked {lock_path}")
    return 0


def _project_run(args: argparse.Namespace) -> int:
    config = load_project_config(_project_config_path(args.project))
    if args.debug and not config.debug:
        config = replace(config, debug=True)

    docker = Docker.detect()
    runner = SubprocessRunner()
    access = discover_gpu_access()
    shm_gib = (
        args.shm_size_gib
        or config.shm_size_gib
        or compute_shm_gib(mem_total_kib=read_mem_total_kib())
    )
    uid, gid = _runtime_identity()

    build_or_reuse_project(
        config=config,
        runner=runner,
        force=args.build,
        no_build=args.no_build,
        docker_prefix=docker.prefix,
    )
    metadata = inspect_project_image(config, runner, docker.prefix)
    require_profile_allowed(metadata.profile_status, os.environ)
    ensure_project_home(config.path.parent, uid=uid, gid=gid)
    profile = _project_protected_profile(
        config=config,
        metadata=metadata,
        runner=runner,
        docker_prefix=docker.prefix,
    )
    initialize_overlay(
        OverlayPaths.for_project(config.path.parent),
        profile=profile,
    )
    terminal = sys.stdin.isatty() and sys.stdout.isatty()
    argv = build_run_argv(
        config=config,
        access=access,
        uid=uid,
        gid=gid,
        shm_gib=shm_gib,
        environment=os.environ,
        terminal=terminal,
        docker_prefix=docker.prefix,
    )
    if args.dry_run:
        print(shlex.join(redact_run_argv(argv)))
        return 0
    return _run_live(argv)


def _project_protected_profile(
    *,
    config,
    metadata,
    runner: Runner,
    docker_prefix: Sequence[str],
) -> ProtectedProfile:
    return load_project_protected_profile(
        config=config,
        metadata=metadata,
        runner=runner,
        docker_prefix=docker_prefix,
    )


def _project_config_path(project: Path) -> Path:
    project = project.expanduser()
    if project.is_dir() or (not project.exists() and project.suffix != ".toml"):
        return project / "amd-ai-project.toml"
    return project


def _runtime_identity() -> tuple[int, int]:
    if os.geteuid() != 0:
        return os.getuid(), os.getgid()
    sudo_uid = os.environ.get("SUDO_UID")
    sudo_gid = os.environ.get("SUDO_GID")
    if sudo_uid is None and sudo_gid is None:
        return os.getuid(), os.getgid()
    if sudo_uid is None or sudo_gid is None:
        raise ProjectRunError("SUDO_UID and SUDO_GID must either both be set or absent")
    try:
        uid = int(sudo_uid)
        gid = int(sudo_gid)
    except ValueError as error:
        raise ProjectRunError("SUDO_UID and SUDO_GID must be integers") from error
    if uid < 0 or gid < 0:
        raise ProjectRunError("SUDO_UID and SUDO_GID must be nonnegative")
    return uid, gid


def _run_live(argv: Sequence[str]) -> int:
    try:
        completed = subprocess.run(argv, check=False)
    except OSError as error:
        raise ProjectRunError(f"cannot start project container: {error}") from error
    return completed.returncode


def _host_preflight(fixture_root: Path | None, json_path: Path | None) -> int:
    snapshot, _, _, _ = _collect_host(fixture_root)
    report = evaluate_preflight(snapshot)
    for finding in report.findings:
        print(f"[{finding.severity.value}] {finding.code}: {finding.summary}")
    if json_path is not None:
        report.write_json(json_path)
    return {
        Status.BLOCKED: 2,
        Status.CHANGE_REQUIRED: 1,
        Status.REBOOT_REQUIRED: 1,
        Status.PASS: 0,
        Status.UNVERIFIED: 0,
    }[report.status]


def _host_prepare(args: argparse.Namespace) -> int:
    snapshot, runner, root, _ = _collect_host(args.fixture_root)
    target_user = args.target_user or os.environ.get("SUDO_USER") or getpass.getuser()
    try:
        snapshot = replace(
            snapshot,
            current_group_ids=_target_user_group_ids(
                target_user,
                fixture_root=args.fixture_root,
                fixture_group_ids=snapshot.current_group_ids,
            ),
        )
        plan = create_prepare_plan(
            snapshot,
            target_user=target_user,
            memory_gib=args.memory_gib,
        )
    except (HostPlanningError, UnsupportedHostError, ValueError) as error:
        print(f"host-prepare: {error}", file=sys.stderr)
        return 2

    _print_plan(plan)
    report = _prepare_report(plan, mode=args.prepare_mode, applied=False)
    if args.json_path is not None:
        report.write_json(args.json_path)
    if args.prepare_mode == "plan":
        return 0

    if not args.yes:
        try:
            confirmed = input("Type APPLY to execute this plan: ") == "APPLY"
        except EOFError:
            confirmed = False
        if not confirmed:
            print(
                "host-prepare: confirmation refused; exact input APPLY is required",
                file=sys.stderr,
            )
            return 2

    plan = _offer_docker_group(plan)
    try:
        execute_plan(
            plan,
            runner,
            effective_uid=os.geteuid(),
            confirmed=True,
            reboot=args.reboot,
            snapshot=snapshot,
            root=root,
        )
    except (ApplyError, ApplyRefused) as error:
        print(f"host-prepare: {error}", file=sys.stderr)
        return 2

    if args.json_path is not None:
        _prepare_report(plan, mode=args.prepare_mode, applied=True).write_json(
            args.json_path
        )
    return 0


def _host_verify(
    fixture_root: Path | None,
    probe_image: str,
    json_path: Path | None,
) -> int:
    snapshot, runner, _, docker_prefix = _collect_host(fixture_root)
    report = verify_host(
        snapshot,
        image=probe_image,
        runner=runner,
        docker_prefix=docker_prefix,
    )
    for finding in report.findings:
        print(f"[{finding.severity.value}] {finding.code}: {finding.summary}")
    if json_path is not None:
        report.write_json(json_path)
    if report.status == Status.PASS:
        return 0
    return 2 if report.status == Status.BLOCKED else 1


def _collect_host(
    fixture_root: Path | None,
) -> tuple[HostSnapshot, Runner, Path, tuple[str, ...]]:
    if fixture_root is None:
        root = Path("/")
        runner = SubprocessRunner()
        device_gids = None
        current_group_ids = None
        docker_prefix = _detect_docker_prefix(runner)
        dmesg_fallback = ("sudo", "-n", "dmesg", "--color=never")
    else:
        root = fixture_root
        runner = FixtureRunner.from_root(root)
        device_gids = load_fixture_device_gids(root)
        current_group_ids = tuple(sorted(set(device_gids.values())))
        docker_prefix = ("docker",)
        dmesg_fallback = None

    snapshot = HostProbe(
        root=root,
        runner=runner,
        device_gids=device_gids,
        current_group_ids=current_group_ids,
        docker_prefix=docker_prefix,
        dmesg_fallback=dmesg_fallback,
    ).collect()
    return snapshot, runner, root, docker_prefix


def _detect_docker_prefix(runner: Runner) -> tuple[str, ...]:
    for prefix in (("docker",), ("sudo", "-n", "docker")):
        try:
            result = runner.run(
                [*prefix, "version", "--format", "{{.Server.Version}}"],
                check=False,
            )
        except OSError:
            continue
        if result.returncode == 0:
            return prefix
    return ("docker",)


def _target_user_group_ids(
    target_user: str,
    *,
    fixture_root: Path | None,
    fixture_group_ids: tuple[int, ...],
) -> tuple[int, ...]:
    if fixture_root is not None:
        return fixture_group_ids
    try:
        user = pwd.getpwnam(target_user)
        group_ids = os.getgrouplist(target_user, user.pw_gid)
    except (KeyError, OSError) as error:
        raise UnsupportedHostError(
            f"cannot resolve groups for target user {target_user!r}"
        ) from error
    return tuple(sorted(set(group_ids)))


def _print_plan(plan: PreparePlan) -> None:
    for action in plan.actions:
        print(f"{action.code}: {action.summary}")


def _offer_docker_group(plan: PreparePlan) -> PreparePlan:
    if plan.target_user == "root":
        return plan
    print("docker group grants root-equivalent daemon control")
    try:
        response = input(
            "Type ADD_DOCKER_GROUP to grant access, or press Enter to retain sudo docker: "
        )
    except EOFError:
        response = ""
    if response != "ADD_DOCKER_GROUP":
        return plan
    return with_docker_group_action(plan)


def _prepare_report(plan: PreparePlan, *, mode: str, applied: bool) -> Report:
    return Report(
        command="host-prepare",
        status=Status.REBOOT_REQUIRED if applied else Status.CHANGE_REQUIRED,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        facts={
            "mode": mode,
            "target_user": plan.target_user,
            "reboot_required": plan.reboot_required,
            "actions": [asdict(action) for action in plan.actions],
        },
        findings=(),
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _shm_gib(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > 128:
        raise argparse.ArgumentTypeError("must not exceed 128 GiB")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
