from __future__ import annotations

import json
from pathlib import Path

from amd_ai.qualification.models import load_profile
from amd_ai.qualification.run import build_suite_commands, execute_suite
from amd_ai.runner import CommandResult


def test_suite_orders_cheap_checks_before_stress_and_uses_safe_devices(tmp_path):
    commands = build_suite_commands(
        image="rocm-pytorch:7.2.1-py3.12-torch2.9.1",
        gids=(109, 110),
        stress_seconds=300,
        repeated_starts=5,
        test_root=tmp_path / "tests",
        cache_dir=tmp_path / "cache",
    )

    assert [command.name for command in commands] == [
        "rocm",
        "torch-fp16",
        "hip",
        "torch-extension",
        "triton",
        "repeated-start",
        "stress",
    ]
    for command in commands:
        assert "--privileged" not in command.argv
        assert "--ipc=host" not in command.argv
        assert "--ipc=private" in command.argv
        assert "/dev/kfd" in command.argv and "/dev/dri" in command.argv
        assert "109" in command.argv and "110" in command.argv
        assert not any("HF_HOME" in value or "models" in value for value in command.argv)


def test_suite_stops_after_first_container_failure_but_still_checks_dmesg(tmp_path):
    profile = load_profile(Path("profiles/qualification/stable.toml"))
    commands = build_suite_commands(
        image=profile.image,
        gids=(109, 110),
        stress_seconds=profile.stress_seconds,
        repeated_starts=profile.repeated_starts,
        test_root=tmp_path / "tests",
        cache_dir=tmp_path / "cache",
    )
    runner = SequenceRunner(
        container_results=[
            (0, {"status": "pass", "facts": {"rocminfo_architectures": ["gfx1151"]}}, ""),
            (2, {}, "FP16 failed"),
        ],
        dmesg_results=["amdgpu: initialized\n", "amdgpu: initialized\n"],
    )

    report = execute_suite(
        profile=profile,
        profile_digest="a" * 64,
        commands=commands,
        runner=runner,
    )

    assert [result.name for result in report.results] == [
        "rocm",
        "torch-fp16",
        "kernel-log",
    ]
    assert report.results[1].passed is False
    assert report.results[-1].passed is True
    assert report.status == "blocked"
    assert runner.container_calls == 2
    assert runner.dmesg_calls == 2


def test_new_gpu_reset_blocks_kernel_log_result(tmp_path):
    profile = load_profile(Path("profiles/qualification/stable.toml"))
    runner = SequenceRunner(
        container_results=[],
        dmesg_results=[
            "amdgpu: initialized\n",
            "amdgpu: initialized\namdgpu: GPU reset begin!\n",
        ],
    )

    report = execute_suite(
        profile=profile,
        profile_digest="b" * 64,
        commands=(),
        runner=runner,
    )

    kernel = report.results[-1]
    assert kernel.name == "kernel-log"
    assert kernel.passed is False
    assert kernel.details["blocking_codes"] == ["GPU.RESET"]


class SequenceRunner:
    def __init__(self, *, container_results, dmesg_results):
        self.container_results = list(container_results)
        self.dmesg_results = list(dmesg_results)
        self.container_calls = 0
        self.dmesg_calls = 0

    def run(self, args, *, check=True, input_text=None):
        argv = tuple(args)
        if "dmesg" in argv:
            output = self.dmesg_results[self.dmesg_calls]
            self.dmesg_calls += 1
            return CommandResult(argv, 0, output, "")
        returncode, payload, stderr = self.container_results[self.container_calls]
        self.container_calls += 1
        stdout = json.dumps(payload) + "\n" if payload else ""
        return CommandResult(argv, returncode, stdout, stderr)
