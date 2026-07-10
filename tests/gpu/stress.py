from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence


def run(seconds: int) -> dict[str, object]:
    if isinstance(seconds, bool) or not isinstance(seconds, int) or not 1 <= seconds <= 3600:
        raise ValueError("stress duration must be from 1 through 3600 seconds")

    import torch
    import torch.nn.functional as functional

    assert torch.cuda.is_available(), "GPU unavailable; CPU fallback is forbidden"
    properties = torch.cuda.get_device_properties(0)
    architecture = str(getattr(properties, "gcnArchName", ""))
    if not architecture:
        architecture = next(
            (
                value
                for value in torch.cuda.get_arch_list()
                if value.startswith("gfx1151")
            ),
            "",
        )
    assert architecture.startswith("gfx1151"), architecture

    torch.manual_seed(19)
    torch.cuda.reset_peak_memory_stats()
    left = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
    right = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
    image = torch.randn(
        (16, 16, 256, 256),
        device="cuda",
        dtype=torch.float16,
    )
    kernel = torch.randn((16, 16, 3, 3), device="cuda", dtype=torch.float16)
    started = time.monotonic()
    iterations = 0
    output = None
    while time.monotonic() - started < seconds:
        if iterations % 2 == 0:
            output = left @ right
        else:
            output = functional.conv2d(image, kernel, padding=1)
        iterations += 1
        if iterations % 10 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    wall_seconds = time.monotonic() - started
    assert output is not None and torch.isfinite(output).all().item(), (
        "nonfinite stress output"
    )
    return {
        "arch": architecture,
        "iterations": iterations,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "seconds": wall_seconds,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=300)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.seconds), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
