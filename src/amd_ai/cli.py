from __future__ import annotations

import argparse
import getpass
import json
import os
import pwd
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from amd_ai import __version__
from amd_ai.host.apply import ApplyError, ApplyRefused, execute_plan
from amd_ai.host.models import HostSnapshot, PlannedAction, PreparePlan
from amd_ai.host.policy import evaluate_preflight
from amd_ai.host.prepare import (
    HostPlanningError,
    UnsupportedHostError,
    create_prepare_plan,
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
)
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
    project_init.add_argument("name")
    project_init.add_argument("--directory", type=Path)
    project_init.add_argument("--base-profile", default="stable")

    project_lock = subparsers.add_parser("project-lock")
    project_lock.add_argument("project", type=Path)

    project_run = subparsers.add_parser("project-run")
    project_run.add_argument("project", type=Path)
    build_mode = project_run.add_mutually_exclusive_group()
    build_mode.add_argument("--build", action="store_true")
    build_mode.add_argument("--no-build", action="store_true")
    project_run.add_argument("--dry-run", action="store_true")
    project_run.add_argument("--debug", action="store_true")
    project_run.add_argument("--shm-size-gib", type=_shm_gib)

    gpu_release = subparsers.add_parser("gpu-release")
    gpu_release.add_argument("--qualification", type=Path, required=True)
    gpu_release.add_argument("--image", required=True)
    gpu_release.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/releases"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
    if args.command in {"project-init", "project-lock", "project-run"}:
        try:
            if args.command == "project-init":
                return _project_init(args)
            if args.command == "project-lock":
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
        ) as error:
            print(f"{args.command}: {error}", file=sys.stderr)
            return 2
    if args.command == "gpu-release":
        return _gpu_release(args)
    raise AssertionError(f"unhandled command: {args.command}")


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

    action = PlannedAction(
        code="DOCKER.ADD_USER_TO_GROUP",
        summary="Grant the target user access to the Docker daemon",
        argv=("usermod", "-a", "-G", "docker", plan.target_user),
        privileged=True,
    )
    actions = list(plan.actions)
    insertion = next(
        (
            index + 1
            for index, existing in enumerate(actions)
            if existing.code == "DOCKER.INSTALL_IF_MISSING"
        ),
        next(
            (
                index + 1
                for index, existing in enumerate(actions)
                if existing.code == "APT.INSTALL_HOST_TOOLS"
            ),
            1,
        ),
    )
    actions.insert(insertion, action)
    return replace(plan, actions=tuple(actions))


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
