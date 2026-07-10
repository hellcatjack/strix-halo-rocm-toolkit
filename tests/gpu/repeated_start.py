from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Sequence


CHILD_SCRIPT = r"""
import json
import time
import torch

started = time.monotonic()
assert torch.cuda.is_available(), "GPU unavailable; CPU fallback is forbidden"
properties = torch.cuda.get_device_properties(0)
architecture = str(getattr(properties, "gcnArchName", ""))
if not architecture:
    architecture = next(
        (value for value in torch.cuda.get_arch_list() if value.startswith("gfx1151")),
        "",
    )
assert architecture.startswith("gfx1151"), architecture
torch.manual_seed(11)
tensor = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
output = tensor @ tensor
torch.cuda.synchronize()
assert torch.isfinite(output).all().item(), "nonfinite repeated-start output"
print(json.dumps({
    "arch": architecture,
    "device": torch.cuda.get_device_name(0),
    "child_seconds": time.monotonic() - started,
}, sort_keys=True))
"""


def run(count: int) -> dict[str, object]:
    if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= 20:
        raise ValueError("repeated start count must be from 2 through 20")
    started = time.monotonic()
    starts: list[dict[str, object]] = []
    for index in range(count):
        child_started = time.monotonic()
        completed = subprocess.run(
            (sys.executable, "-c", CHILD_SCRIPT),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        process_seconds = time.monotonic() - child_started
        if completed.returncode != 0:
            evidence = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"GPU child {index + 1} failed ({completed.returncode}): {evidence}"
            )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError(f"GPU child {index + 1} produced no JSON result")
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"GPU child {index + 1} produced invalid JSON"
            ) from error
        architecture = str(payload.get("arch", ""))
        if not architecture.startswith("gfx1151"):
            raise RuntimeError(
                f"GPU child {index + 1} reported architecture {architecture!r}"
            )
        starts.append(
            {
                "attempt": index + 1,
                "process_seconds": process_seconds,
                **payload,
            }
        )
    return {
        "arch": str(starts[-1]["arch"]),
        "count": count,
        "starts": starts,
        "seconds": time.monotonic() - started,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.count), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
