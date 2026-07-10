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
from amd_ai.installer.state import stage_input_digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="strix-halo-rocm-host-helper")
    parser.add_argument("--target-user")
    parser.add_argument("--memory-gib", type=_positive_int)
    parser.add_argument("--expected-plan-digest")
    parser.add_argument("--include-docker-group", action="store_true")
    parser.add_argument("operation", choices=("plan", "apply", "verify"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if os.geteuid() != 0:
        print("privileged-host: root is required", file=sys.stderr)
        return 2
    try:
        actions = ProductionInstallerActions(effective_uid=0)
        if args.operation == "verify":
            if not args.target_user:
                raise ActionError("verify mode requires --target-user")
            if any(
                (
                    args.memory_gib is not None,
                    args.expected_plan_digest is not None,
                    args.include_docker_group,
                )
            ):
                raise ActionError("verify mode received plan/apply arguments")
            payload = {
                "report": actions.host_verify(
                    target_user=args.target_user
                ).to_dict(),
                "schema_version": 1,
            }
        else:
            if not args.target_user:
                raise ActionError("plan/apply mode requires --target-user")
            host_plan = actions.host_plan(
                target_user=args.target_user,
                memory_gib=args.memory_gib,
            )
        if args.operation == "plan":
            if args.expected_plan_digest is not None or args.include_docker_group:
                raise ActionError("plan mode received apply-only arguments")
            payload = {
                "adapter_id": host_plan.adapter_id,
                "plan": prepare_plan_payload(host_plan.plan),
                "plan_digest": host_plan.plan_digest,
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
        print(f"privileged-host: {error}", file=sys.stderr)
        return 2


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
