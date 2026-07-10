from __future__ import annotations

import argparse
import getpass
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from amd_ai import __version__
from amd_ai.host.apply import ApplyError, ApplyRefused, execute_plan
from amd_ai.host.models import HostSnapshot, PlannedAction, PreparePlan
from amd_ai.host.policy import evaluate_preflight
from amd_ai.host.prepare import UnsupportedHostError, create_prepare_plan
from amd_ai.host.probe import FixtureRunner, HostProbe, load_fixture_device_gids
from amd_ai.host.verify import verify_host
from amd_ai.image.build import (
    BuildError,
    build_rocm_python,
    build_rocm_pytorch,
    prune_images,
    run_image_check,
)
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
            return run_image_check(
                image=args.image,
                mode=args.mode,
                metadata_only=args.metadata_only,
                runtime=args.runtime,
                json_path=args.json_path,
            )
        except BuildError as error:
            print(f"container-check: {error}", file=sys.stderr)
            return 2
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


def _host_preflight(fixture_root: Path | None, json_path: Path | None) -> int:
    snapshot, _, _ = _collect_host(fixture_root)
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
    snapshot, runner, root = _collect_host(args.fixture_root)
    target_user = args.target_user or os.environ.get("SUDO_USER") or getpass.getuser()
    try:
        plan = create_prepare_plan(
            snapshot,
            target_user=target_user,
            memory_gib=args.memory_gib,
        )
    except (UnsupportedHostError, ValueError) as error:
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
    snapshot, runner, _ = _collect_host(fixture_root)
    report = verify_host(snapshot, image=probe_image, runner=runner)
    for finding in report.findings:
        print(f"[{finding.severity.value}] {finding.code}: {finding.summary}")
    if json_path is not None:
        report.write_json(json_path)
    if report.status == Status.PASS:
        return 0
    return 2 if report.status == Status.BLOCKED else 1


def _collect_host(
    fixture_root: Path | None,
) -> tuple[HostSnapshot, Runner, Path]:
    if fixture_root is None:
        root = Path("/")
        runner = SubprocessRunner()
        device_gids = None
        current_group_ids = None
    else:
        root = fixture_root
        runner = FixtureRunner.from_root(root)
        device_gids = load_fixture_device_gids(root)
        current_group_ids = tuple(sorted(set(device_gids.values())))

    snapshot = HostProbe(
        root=root,
        runner=runner,
        device_gids=device_gids,
        current_group_ids=current_group_ids,
    ).collect()
    return snapshot, runner, root


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


if __name__ == "__main__":
    raise SystemExit(main())
