from __future__ import annotations

from amd_ai.qualification.kernel_log import (
    KernelLogDiscontinuity,
    classify_new_lines,
    new_kernel_lines,
    relevant_gpu_lines,
)
import pytest


def test_known_gpu_failures_are_blocking():
    findings = classify_new_lines(
        [
            "amdgpu 0000:c5:00.0: MES failed to respond to msg=REMOVE_QUEUE",
            "amdgpu: GPU reset begin!",
            "amdgpu: page fault (src_id:0 ring:24 vmid:3)",
            "amdgpu: ring gfx_0.0.0 timeout",
            "amdgpu: failed to load firmware file",
            "amdgpu: probe of 0000:c5:00.0 failed with error -22",
        ]
    )

    assert {finding.code for finding in findings} == {
        "GPU.MES_TIMEOUT",
        "GPU.RESET",
        "GPU.PAGE_FAULT",
        "GPU.RING_TIMEOUT",
        "GPU.FIRMWARE",
        "GPU.INIT_FATAL",
    }


def test_unrelated_warning_is_retained_nowhere_as_a_blocker():
    findings = classify_new_lines(["usb 1-1: reset high-speed USB device"])

    assert findings == ()


def test_log_diff_preserves_duplicate_new_messages():
    before = "line one\namdgpu: old warning\namdgpu: repeated\n"
    after = (
        "line one\namdgpu: old warning\namdgpu: repeated\n"
        "amdgpu: repeated\namdgpu: GPU reset begin!\n"
    )

    assert new_kernel_lines(before, after) == (
        "amdgpu: repeated",
        "amdgpu: GPU reset begin!",
    )


def test_relevant_evidence_keeps_nonblocking_gpu_subsystem_lines():
    lines = (
        "usb: unrelated",
        "[drm] initialized amdgpu",
        "kfd kfd: added device",
        "firmware: loading completed",
    )

    assert relevant_gpu_lines(lines) == lines[1:]


def test_cleared_post_run_log_is_a_blocking_discontinuity():
    with pytest.raises(KernelLogDiscontinuity):
        new_kernel_lines("amdgpu: initialized\n", "")
