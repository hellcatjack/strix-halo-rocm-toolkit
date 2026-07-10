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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "host-preflight":
        return _host_preflight(args.fixture_root, args.json_path)
    if args.command == "host-prepare":
        return _host_prepare(args)
    raise AssertionError(f"unhandled command: {args.command}")


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
