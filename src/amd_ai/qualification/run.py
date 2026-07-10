from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from amd_ai.qualification.kernel_log import (
    classify_new_lines,
    new_kernel_lines,
    relevant_gpu_lines,
)
from amd_ai.qualification.models import (
    CheckResult,
    QualificationProfile,
    QualificationReport,
    load_profile,
)
from amd_ai.runner import CommandResult, Runner


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEST_ROOT = REPOSITORY_ROOT / "tests/gpu"
DEFAULT_CACHE_DIR = REPOSITORY_ROOT / "reports/qualification-cache"
DEFAULT_DMESG_ARGV = ("sudo", "-n", "dmesg", "--color=never")


class QualificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SuiteCommand:
    name: str
    argv: tuple[str, ...]


def build_suite_commands(
    *,
    image: str,
    gids: Sequence[int],
    stress_seconds: int,
    repeated_starts: int,
    uid: int = 1000,
    gid: int = 1000,
    docker_prefix: Sequence[str] = ("docker",),
    test_root: Path = DEFAULT_TEST_ROOT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> tuple[SuiteCommand, ...]:
    if not image or image.startswith("-") or any(value.isspace() for value in image):
        raise QualificationError("qualification image is invalid")
    if (
        isinstance(stress_seconds, bool)
        or not isinstance(stress_seconds, int)
        or not 1 <= stress_seconds <= 3600
    ):
        raise QualificationError("stress duration must be from 1 through 3600 seconds")
    if (
        isinstance(repeated_starts, bool)
        or not isinstance(repeated_starts, int)
        or not 2 <= repeated_starts <= 20
    ):
        raise QualificationError("repeated starts must be from 2 through 20")
    if uid < 0 or gid < 0:
        raise QualificationError("qualification UID and GID must be nonnegative")
    group_ids = tuple(sorted(set(gids)))
    if not group_ids or any(value < 0 for value in group_ids):
        raise QualificationError("qualification requires nonnegative GPU group IDs")
    test_root = test_root.resolve()
    cache_dir = cache_dir.resolve()
    for path in (test_root, cache_dir):
        if any(character in str(path) for character in (",", "\n", "\r", "\0")):
            raise QualificationError(f"path cannot be represented as a Docker mount: {path}")

    common: list[str] = [
        *docker_prefix,
        "run",
        "--rm",
        "--ipc=private",
        "--shm-size",
        "16g",
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri",
    ]
    for group_id in group_ids:
        common.extend(("--group-add", str(group_id)))
    common.extend(
        (
            "--user",
            f"{uid}:{gid}",
            "--env",
            "HOME=/opt/amd-ai/cache/home",
            "--env",
            "AMD_AI_QUALIFICATION_CACHE=/opt/amd-ai/cache",
            "--mount",
            f"type=bind,src={test_root},dst=/opt/amd-ai/tests,readonly",
            "--mount",
            f"type=bind,src={cache_dir},dst=/opt/amd-ai/cache",
        )
    )

    def python_command(name: str, script: str, *arguments: str) -> SuiteCommand:
        return SuiteCommand(
            name,
            (
                *common,
                "--entrypoint",
                "/opt/venv/bin/python",
                image,
                f"/opt/amd-ai/tests/{script}",
                *arguments,
            ),
        )

    return (
        SuiteCommand(
            "rocm",
            (
                *common,
                image,
                "container-check",
                "--mode",
                "rocm",
                "--runtime",
                "--json",
                "-",
            ),
        ),
        python_command("torch-fp16", "torch_smoke.py"),
        python_command("hip", "run_hip_smoke.py"),
        python_command("torch-extension", "torch_extension_smoke.py"),
        python_command("triton", "triton_smoke.py"),
        python_command(
            "repeated-start",
            "repeated_start.py",
            "--count",
            str(repeated_starts),
        ),
        python_command("stress", "stress.py", "--seconds", str(stress_seconds)),
    )


def execute_suite(
    *,
    profile: QualificationProfile,
    profile_digest: str,
    commands: Sequence[SuiteCommand],
    runner: Runner,
    dmesg_argv: Sequence[str] = DEFAULT_DMESG_ARGV,
) -> QualificationReport:
    generated_at = datetime.now(UTC).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    results: list[CheckResult] = []
    before, before_seconds = _capture_dmesg(runner, dmesg_argv)
    if before.returncode != 0:
        results.append(_dmesg_unavailable(before, before_seconds))
        return _report(profile, profile_digest, generated_at, results)

    for command in commands:
        result = _run_command(command, profile.gpu_arch, runner)
        results.append(result)
        if not result.passed:
            break

    after, after_seconds = _capture_dmesg(runner, dmesg_argv)
    if after.returncode != 0:
        results.append(_dmesg_unavailable(after, after_seconds))
    else:
        lines = new_kernel_lines(before.stdout, after.stdout)
        relevant = relevant_gpu_lines(lines)
        findings = classify_new_lines(lines)
        results.append(
            CheckResult(
                "kernel-log",
                not findings,
                before_seconds + after_seconds,
                {
                    "new_relevant_lines": list(relevant),
                    "blocking_codes": [finding.code for finding in findings],
                },
                "\n".join(relevant),
            )
        )
    return _report(profile, profile_digest, generated_at, results)


def run_profile(
    *,
    profile_path: Path,
    output_path: Path | None,
    runner: Runner,
    docker_prefix: Sequence[str],
    gids: Sequence[int],
    uid: int,
    gid: int,
    test_root: Path = DEFAULT_TEST_ROOT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> QualificationReport:
    profile = load_profile(profile_path)
    if not test_root.is_dir():
        raise QualificationError(f"qualification test directory is missing: {test_root}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "home").mkdir(mode=0o700, exist_ok=True)
    profile_digest = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    commands = build_suite_commands(
        image=profile.image,
        gids=gids,
        stress_seconds=profile.stress_seconds,
        repeated_starts=profile.repeated_starts,
        uid=uid,
        gid=gid,
        docker_prefix=docker_prefix,
        test_root=test_root,
        cache_dir=cache_dir,
    )
    report = execute_suite(
        profile=profile,
        profile_digest=profile_digest,
        commands=commands,
        runner=runner,
    )
    if output_path is not None:
        report.write_json(output_path)
    return report


def _run_command(
    command: SuiteCommand,
    expected_arch: str,
    runner: Runner,
) -> CheckResult:
    started = time.monotonic()
    try:
        completed = runner.run(list(command.argv), check=False)
    except OSError as error:
        completed = CommandResult(command.argv, 127, "", str(error))
    duration = time.monotonic() - started
    payload = _last_json_object(completed.stdout)
    details = dict(payload) if payload is not None else {}
    details["returncode"] = completed.returncode
    passed = completed.returncode == 0 and payload is not None
    if details.get("status") == "blocked":
        passed = False
    reported_arch = _reported_arch(command.name, details)
    if reported_arch is not None and not reported_arch.startswith(expected_arch):
        passed = False
        details["architecture_error"] = (
            f"expected {expected_arch}, got {reported_arch or '<missing>'}"
        )
    evidence = _command_evidence(completed)
    if payload is None:
        evidence = (evidence + "\nmissing final JSON object").strip()
    return CheckResult(command.name, passed, duration, details, evidence)


def _capture_dmesg(
    runner: Runner,
    argv: Sequence[str],
) -> tuple[CommandResult, float]:
    started = time.monotonic()
    try:
        result = runner.run(list(argv), check=False)
    except OSError as error:
        result = CommandResult(tuple(argv), 127, "", str(error))
    return result, time.monotonic() - started


def _dmesg_unavailable(result: CommandResult, duration: float) -> CheckResult:
    return CheckResult(
        "kernel-log",
        False,
        duration,
        {"blocking_codes": ["HOST.DMESG_UNAVAILABLE"]},
        _command_evidence(result) or "dmesg unavailable",
    )


def _report(
    profile: QualificationProfile,
    profile_digest: str,
    generated_at: str,
    results: Sequence[CheckResult],
) -> QualificationReport:
    return QualificationReport.from_results(
        profile_id=profile.profile_id,
        results=results,
        required_checks=profile.required_checks,
        generated_at=generated_at,
        profile_digest=profile_digest,
        image=profile.image,
        gpu_arch=profile.gpu_arch,
    )


def _last_json_object(output: str) -> dict[str, object] | None:
    for line in reversed(output.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _reported_arch(name: str, details: dict[str, object]) -> str | None:
    architecture = details.get("arch")
    if isinstance(architecture, str):
        return architecture
    if name == "rocm":
        facts = details.get("facts")
        if isinstance(facts, dict):
            architectures = facts.get("rocminfo_architectures")
            if isinstance(architectures, list):
                return next(
                    (value for value in architectures if isinstance(value, str)),
                    "",
                )
        return ""
    if name in {
        "torch-fp16",
        "torch-extension",
        "triton",
        "repeated-start",
        "stress",
    }:
        return ""
    return None


def _command_evidence(result: CommandResult) -> str:
    values = []
    if result.stdout.strip():
        values.append("stdout:\n" + result.stdout.strip())
    if result.stderr.strip():
        values.append("stderr:\n" + result.stderr.strip())
    return "\n".join(values)
