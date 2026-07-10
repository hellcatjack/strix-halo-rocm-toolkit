from __future__ import annotations

import argparse
import sys
from pathlib import Path

from amd_ai.overlay.models import OverlayPaths
from amd_ai.overlay.repair import TransactionalGenerationBuilder, repair_overlay
from amd_ai.overlay.resolver import SubprocessProcessRunner
from amd_ai.overlay.transaction import TransactionError
from amd_ai.overlay.verify import (
    OverlayVerificationError,
    load_protected_profile,
    verify_base_manifest,
    verify_candidate_overlay,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="amd-ai-overlay-repair")
    parser.add_argument("reason_code")
    args = parser.parse_args(argv)
    runner = SubprocessProcessRunner()
    try:
        paths = OverlayPaths.for_project(Path("/workspace"))
        profile = load_protected_profile()
        verify_base_manifest(runner=runner)
        builder = TransactionalGenerationBuilder(
            runner=runner,
            verifier=lambda site_packages: verify_candidate_overlay(
                site_packages,
                runner=runner,
                profile=profile,
            ),
        )
        result = repair_overlay(
            paths,
            profile=profile,
            reason_code=args.reason_code,
            builder=builder,
        )
    except (OSError, TransactionError, OverlayVerificationError) as error:
        print(f"overlay-repair: {error}", file=sys.stderr)
        return 2
    print(result.new_generation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
