from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Sequence

from amd_ai.installer.actions import (
    ActionError,
    ProductionInstallerActions,
    prepare_plan_payload,
)
from amd_ai.host.models import HostPlanPhase
from amd_ai.installer.progress import (
    ProgressMode,
    StderrCommandObserver,
    sanitize_output,
)
from amd_ai.installer.state import stage_input_digest
from amd_ai.runner import SubprocessRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="strix-halo-rocm-host-helper")
    parser.add_argument("--target-user")
    parser.add_argument("--expected-plan-digest")
    parser.add_argument("--include-docker-group", action="store_true")
    parser.add_argument(
        "--phase",
        choices=tuple(phase.value for phase in HostPlanPhase),
    )
    parser.add_argument(
        "--display-manager-was-active",
        action="store_true",
    )
    parser.add_argument(
        "--progress-mode",
        choices=tuple(mode.value for mode in ProgressMode),
        default=ProgressMode.QUIET.value,
    )
    parser.add_argument(
        "operation",
        choices=("plan", "apply", "verify", "verify-kernel"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if os.geteuid() != 0:
        print("privileged-host: root is required", file=sys.stderr)
        return 2
    try:
        progress_mode = ProgressMode(args.progress_mode)
        observer = StderrCommandObserver(
            mode=progress_mode,
            stderr=sys.stderr,
        )
        actions = ProductionInstallerActions(
            effective_uid=0,
            runner=SubprocessRunner(observer=observer),
            command_observer=observer,
            progress_mode=progress_mode,
        )
        if args.operation in {"verify", "verify-kernel"}:
            if not args.target_user:
                raise ActionError("verify mode requires --target-user")
            if any(
                (
                    args.expected_plan_digest is not None,
                    args.include_docker_group,
                    args.phase is not None,
                )
            ):
                raise ActionError("verify mode received plan/apply arguments")
            if args.operation == "verify" and args.display_manager_was_active:
                raise ActionError("final verify received kernel-only arguments")
            report = (
                actions.kernel_verify(
                    target_user=args.target_user,
                    display_manager_was_active=(
                        args.display_manager_was_active
                    ),
                )
                if args.operation == "verify-kernel"
                else actions.host_verify(target_user=args.target_user)
            )
            payload = {
                "report": report.to_dict(),
                "schema_version": 1,
            }
        else:
            if not args.target_user:
                raise ActionError("plan/apply mode requires --target-user")
            if args.phase is None:
                raise ActionError("plan/apply mode requires --phase")
            if args.display_manager_was_active:
                raise ActionError("plan/apply mode received verify-only arguments")
            host_plan = actions.host_plan(
                target_user=args.target_user,
                phase=HostPlanPhase(args.phase),
            )
        if args.operation == "plan":
            if args.expected_plan_digest is not None or args.include_docker_group:
                raise ActionError("plan mode received apply-only arguments")
            payload = {
                "adapter_id": host_plan.adapter_id,
                "plan": prepare_plan_payload(host_plan.plan),
                "plan_digest": host_plan.plan_digest,
                "running_kernel": host_plan.running_kernel,
                "display_manager_active": host_plan.display_manager_active,
                "schema_version": 1,
            }
        elif args.operation == "apply":
            expected = args.expected_plan_digest
            if (
                not isinstance(expected, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected) is None
            ):
                raise ActionError(
                    "apply mode requires an exact host plan digest"
                )
            actual = stage_input_digest(prepare_plan_payload(host_plan.plan))
            if actual != expected or host_plan.plan_digest != expected:
                raise ActionError(
                    "host plan changed before privileged apply; replan is required"
                )
            result = actions.host_apply(
                host_plan,
                include_docker_group=args.include_docker_group,
            )
            payload = {"facts": dict(result.facts), "schema_version": 1}
        print(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        )
        return 0
    except Exception as error:
        print(
            "privileged-host: " + sanitize_output(str(error)),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
