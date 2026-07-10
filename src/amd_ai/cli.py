from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from amd_ai import __version__
from amd_ai.host.policy import evaluate_preflight
from amd_ai.host.probe import FixtureRunner, HostProbe, load_fixture_device_gids
from amd_ai.report import Status
from amd_ai.runner import SubprocessRunner


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "host-preflight":
        return _host_preflight(args.fixture_root, args.json_path)
    raise AssertionError(f"unhandled command: {args.command}")


def _host_preflight(fixture_root: Path | None, json_path: Path | None) -> int:
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


if __name__ == "__main__":
    raise SystemExit(main())
