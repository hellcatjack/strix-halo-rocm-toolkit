from __future__ import annotations

import json
from pathlib import Path

import pytest

from amd_ai.qualification.models import (
    CheckResult,
    ProfileError,
    QualificationReport,
    load_profile,
)


PROFILE = Path("profiles/qualification/stable.toml")


def test_stable_profile_locks_target_and_required_checks():
    profile = load_profile(PROFILE)

    assert profile.profile_id == "stable-gfx1151"
    assert profile.image == "rocm-pytorch:7.2.1-py3.12-torch2.9.1"
    assert profile.rocm_version == "7.2.1"
    assert profile.torch_version == "2.9.1"
    assert profile.gpu_arch == "gfx1151"
    assert profile.stress_seconds == 300
    assert profile.repeated_starts == 5
    assert profile.required_checks == (
        "rocm",
        "torch-fp16",
        "hip",
        "torch-extension",
        "triton",
        "repeated-start",
        "stress",
        "kernel-log",
    )


def test_required_failure_or_missing_check_blocks_report():
    failed = QualificationReport.from_results(
        profile_id="stable-gfx1151",
        results=(
            CheckResult("rocm", True, 0.2, {"arch": "gfx1151"}, ""),
            CheckResult("triton", False, 1.0, {}, "compile failed"),
        ),
        required_checks=("rocm", "triton"),
    )
    missing = QualificationReport.from_results(
        profile_id="stable-gfx1151",
        results=(CheckResult("rocm", True, 0.2, {}, ""),),
        required_checks=("rocm", "triton"),
    )

    assert failed.status == "blocked"
    assert missing.status == "blocked"


def test_complete_successful_report_passes_and_serializes_stably(tmp_path):
    report = QualificationReport.from_results(
        profile_id="stable-gfx1151",
        results=(
            CheckResult("rocm", True, 0.2, {"arch": "gfx1151"}, ""),
            CheckResult("triton", True, 1.0, {"count": 1024}, ""),
        ),
        required_checks=("rocm", "triton"),
    )
    output = tmp_path / "qualification.json"

    report.write_json(output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert report.status == "pass"
    assert payload["schema_version"] == 1
    assert payload["results"][0]["details"] == {"arch": "gfx1151"}
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_profile_rejects_unknown_keys_and_invalid_bounds(tmp_path):
    text = PROFILE.read_text(encoding="utf-8")
    unknown = tmp_path / "unknown.toml"
    unknown.write_text(text + '\nextra = "forbidden"\n', encoding="utf-8")
    short = tmp_path / "short.toml"
    short.write_text(text.replace("stress_seconds = 300", "stress_seconds = 59"))

    with pytest.raises(ProfileError, match="unknown"):
        load_profile(unknown)
    with pytest.raises(ProfileError, match="stress_seconds"):
        load_profile(short)
