from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from amd_ai.report import Finding, Severity


FAILURE_PATTERNS = (
    (
        re.compile(
            r"(?:Fatal error during GPU init|amdgpu:\s*probe of .* failed with error)",
            re.IGNORECASE,
        ),
        "GPU.INIT_FATAL",
        "AMD GPU initialization failed",
    ),
    (
        re.compile(r"\bMES\b.*(?:timeout|failed to respond)", re.IGNORECASE),
        "GPU.MES_TIMEOUT",
        "AMD GPU MES timeout or response failure",
    ),
    (
        re.compile(r"GPU reset begin", re.IGNORECASE),
        "GPU.RESET",
        "AMD GPU reset began during qualification",
    ),
    (
        re.compile(r"amdgpu.*page fault", re.IGNORECASE),
        "GPU.PAGE_FAULT",
        "AMD GPU page fault occurred during qualification",
    ),
    (
        re.compile(r"ring .* timeout", re.IGNORECASE),
        "GPU.RING_TIMEOUT",
        "AMD GPU ring timeout occurred during qualification",
    ),
    (
        re.compile(r"failed to load firmware", re.IGNORECASE),
        "GPU.FIRMWARE",
        "GPU firmware failed to load during qualification",
    ),
)
GPU_EVIDENCE_PATTERN = re.compile(
    r"\b(?:amdgpu|drm|kfd|ttm|firmware)\b",
    re.IGNORECASE,
)


class KernelLogDiscontinuity(RuntimeError):
    pass


def new_kernel_lines(before: str, after: str) -> tuple[str, ...]:
    before_lines = tuple(before.splitlines())
    after_lines = tuple(after.splitlines())
    if before_lines and not after_lines:
        raise KernelLogDiscontinuity("post-run kernel log is empty")
    if after_lines[: len(before_lines)] == before_lines:
        return after_lines[len(before_lines) :]

    remaining = Counter(before_lines)
    overlap = sum((Counter(before_lines) & Counter(after_lines)).values())
    if before_lines and overlap == 0:
        raise KernelLogDiscontinuity(
            "pre-run and post-run kernel logs have no common lines"
        )
    new_lines: list[str] = []
    for line in after_lines:
        if remaining[line] > 0:
            remaining[line] -= 1
        else:
            new_lines.append(line)
    return tuple(new_lines)


def relevant_gpu_lines(lines: Sequence[str]) -> tuple[str, ...]:
    return tuple(line for line in lines if GPU_EVIDENCE_PATTERN.search(line))


def classify_new_lines(lines: Sequence[str]) -> tuple[Finding, ...]:
    findings: list[Finding] = []
    for line in lines:
        for pattern, code, summary in FAILURE_PATTERNS:
            if pattern.search(line):
                findings.append(
                    Finding(
                        code=code,
                        severity=Severity.ERROR,
                        summary=summary,
                        evidence=line,
                        remediation=(
                            "Preserve the report bundle and diagnose the host driver before "
                            "rerunning qualification."
                        ),
                    )
                )
                break
    return tuple(findings)
