from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    left,
    right,
    output,
    count: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < count
    left_values = tl.load(left + offsets, mask=mask)
    right_values = tl.load(right + offsets, mask=mask)
    tl.store(output + offsets, left_values + right_values, mask=mask)


def run() -> dict[str, object]:
    assert torch.cuda.is_available(), "GPU unavailable; CPU fallback is forbidden"
    cache_root_value = os.environ.get("AMD_AI_QUALIFICATION_CACHE")
    cache_root = Path(cache_root_value) if cache_root_value else None
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="triton-",
        dir=cache_root,
    ) as temporary:
        cache_dir = Path(temporary)
        previous_cache = os.environ.get("TRITON_CACHE_DIR")
        os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
        try:
            count = 1 << 20
            left = torch.randn(count, device="cuda", dtype=torch.float32)
            right = torch.randn(count, device="cuda", dtype=torch.float32)
            output = torch.empty_like(left)
            started = time.monotonic()
            add_kernel[(triton.cdiv(count, 256),)](
                left,
                right,
                output,
                count=count,
                BLOCK=256,
            )
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            torch.testing.assert_close(output, left + right, rtol=0, atol=0)
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
            return {
                "arch": architecture,
                "count": count,
                "seconds": seconds,
                "cache_bytes": _directory_size(cache_dir),
            }
        finally:
            if previous_cache is None:
                os.environ.pop("TRITON_CACHE_DIR", None)
            else:
                os.environ["TRITON_CACHE_DIR"] = previous_cache


def _directory_size(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
