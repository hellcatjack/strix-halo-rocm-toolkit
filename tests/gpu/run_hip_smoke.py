from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path


def compile_argv(source: Path, output: Path) -> tuple[str, ...]:
    return (
        "/opt/rocm/bin/hipcc",
        "-O2",
        "-std=c++17",
        "--offload-arch=gfx1151",
        str(source),
        "-o",
        str(output),
    )


def run(source: Path | None = None) -> dict[str, object]:
    source = source or Path(__file__).with_name("hip_vector_add.cpp")
    with tempfile.TemporaryDirectory(prefix="amd-ai-hip-") as temporary:
        binary = Path(temporary) / "hip-vector-add"
        compile_started = time.monotonic()
        compiled = subprocess.run(
            compile_argv(source, binary),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        compile_seconds = time.monotonic() - compile_started
        if compiled.returncode != 0:
            evidence = compiled.stderr.strip() or compiled.stdout.strip()
            raise RuntimeError(
                f"hipcc failed ({compiled.returncode}): {evidence}"
            )
        run_started = time.monotonic()
        executed = subprocess.run(
            (str(binary),),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        run_seconds = time.monotonic() - run_started
        evidence = executed.stderr.strip() or executed.stdout.strip()
        if executed.returncode != 0:
            raise RuntimeError(
                f"HIP program failed ({executed.returncode}): {evidence}"
            )
        if executed.stdout.strip() != "HIP vector add PASS":
            raise RuntimeError(f"unexpected HIP program output: {evidence}")
        return {
            "compile_seconds": compile_seconds,
            "run_seconds": run_seconds,
            "binary_bytes": binary.stat().st_size,
            "output": executed.stdout.strip(),
        }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
