from __future__ import annotations

import io
import stat
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from amd_ai.installer import progress as progress_module
from amd_ai.installer.progress import (
    HeartbeatSchedule,
    InstallerProgress,
    ProgressError,
    ProgressMode,
    ProgressOutcome,
    SessionPlan,
    SessionLog,
    StagePosition,
    sanitize_output,
)
from amd_ai.installer.models import (
    CONTAINER_STAGE_ORDER,
    InstallMode,
    InstallStage,
)
from amd_ai.runner import CommandResult, CommandStream


@dataclass
class FakeClock:
    wall: datetime
    monotonic: float

    def wall_now(self) -> datetime:
        return self.wall

    def monotonic_now(self) -> float:
        return self.monotonic

    def advance(self, seconds: float) -> None:
        self.wall += timedelta(seconds=seconds)
        self.monotonic += seconds


def container_plan(tmp_path: Path) -> SessionPlan:
    return SessionPlan(
        mode=InstallMode.CONTAINER,
        project_dir=tmp_path / "video-lab",
        project_name="video-lab",
        state_path=tmp_path / "state.json",
        state_source="project",
        image_source="pull",
        release_id="0.2.0",
        stages=CONTAINER_STAGE_ORDER,
        first_incomplete=InstallStage.IMAGE_PULL_OR_BUILD,
    )


def opened_reporter(
    tmp_path: Path,
    mode: ProgressMode,
) -> tuple[InstallerProgress, io.StringIO, io.StringIO]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    reporter = InstallerProgress(
        mode=mode,
        stdout=stdout,
        stderr=stderr,
        log_root=tmp_path / "logs",
        process_id=11,
    )
    reporter.open_session(tmp_path / "project")
    return reporter, stdout, stderr


def test_sanitize_output_removes_controls_and_redacts_credentials() -> None:
    raw = (
        "\x1b]0;title\x07\x1b[31mred\x1b[0m\rnext\n"
        "https://alice:ghp_example@github.com/org/repo "
        "HF_TOKEN=hf_example Authorization: Bearer bearer_example "
        "--password secret-value"
    )

    rendered = sanitize_output(raw)

    assert "\x1b" not in rendered
    assert "\r" not in rendered
    assert "red\nnext" in rendered
    for secret in (
        "alice:ghp_example",
        "hf_example",
        "bearer_example",
        "secret-value",
    ):
        assert secret not in rendered
    assert rendered.count("<redacted>") >= 4


def test_session_log_is_private_and_rejects_collisions(
    tmp_path: Path,
) -> None:
    project = tmp_path / "video-lab"

    def wall_clock() -> datetime:
        return datetime(2026, 7, 10, 14, 25, 33, tzinfo=UTC)

    log = SessionLog.open(
        project_dir=project,
        log_root=tmp_path / "state" / "logs",
        wall_clock=wall_clock,
        process_id=18421,
    )
    log.write("stdout", "PLAN", "token=visible-name")
    path = log.path
    log.close()

    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert "2026-07-10T14:25:33" in path.read_text(encoding="utf-8")
    with pytest.raises(ProgressError, match="already exists"):
        SessionLog.open(
            project_dir=project,
            log_root=tmp_path / "state" / "logs",
            wall_clock=wall_clock,
            process_id=18421,
        )


def test_session_log_rejects_symlinked_control_directory(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    root = tmp_path / "logs"
    root.symlink_to(target, target_is_directory=True)

    with pytest.raises(ProgressError, match="symlink"):
        SessionLog.open(
            project_dir=tmp_path / "project",
            log_root=root,
            wall_clock=lambda: datetime(2026, 7, 10, tzinfo=UTC),
            process_id=1,
        )


def test_default_reporter_prints_plan_stage_and_elapsed_time(
    tmp_path: Path,
) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    clock = FakeClock(
        wall=datetime(2026, 7, 10, tzinfo=UTC), monotonic=10.0
    )
    reporter = InstallerProgress(
        mode=ProgressMode.DEFAULT,
        stdout=stdout,
        stderr=stderr,
        log_root=tmp_path / "logs",
        wall_clock=clock.wall_now,
        monotonic=clock.monotonic_now,
        process_id=7,
    )
    reporter.open_session(tmp_path / "video-lab")
    reporter.session_plan(container_plan(tmp_path))
    position = StagePosition(InstallStage.IMAGE_PULL_OR_BUILD, 4, 8)
    reporter.stage_started(position)
    clock.advance(134.0)
    reporter.stage_passed(position)
    reporter.installation_finished(
        outcome=ProgressOutcome.SUCCESS,
        exit_code=0,
        message="installation complete",
        state_path=tmp_path / "state.json",
        project_dir=tmp_path / "video-lab",
        position=position,
    )
    reporter.close()

    output = stdout.getvalue()
    assert "PLAN     模式=container" in output
    assert f"项目={tmp_path / 'video-lab'}" in output
    assert "名称=video-lab" in output
    assert f"状态={tmp_path / 'state.json'}（per-project）" in output
    assert "镜像来源=pull，stable release=0.2.0" in output
    assert "共 8 个阶段，从 IMAGE_PULL_OR_BUILD 继续" in output
    assert "START    [4/8] 获取或构建 stable 镜像" in output
    assert "PASS     [4/8] 获取或构建 stable 镜像，用时 02:14" in output
    assert "SUMMARY" in output
    assert stderr.getvalue() == ""


@pytest.mark.parametrize(
    ("mode", "present", "absent"),
    [
        (ProgressMode.DEFAULT, ("START", "PASS"), ("COMMAND", "DEBUG")),
        (
            ProgressMode.VERBOSE,
            ("START", "PASS", "COMMAND", "DEBUG"),
            (),
        ),
        (
            ProgressMode.QUIET,
            ("SUMMARY",),
            ("PLAN", "START", "WAIT", "PASS"),
        ),
    ],
)
def test_progress_mode_filters_terminal_but_not_log(
    tmp_path: Path,
    mode: ProgressMode,
    present: tuple[str, ...],
    absent: tuple[str, ...],
) -> None:
    reporter, stdout, stderr = opened_reporter(tmp_path, mode)
    position = StagePosition(InstallStage.PROJECT_INIT, 6, 8)
    reporter.session_plan(container_plan(tmp_path))
    reporter.stage_started(position)
    reporter.command_started(("uv", "pip", "compile"), live=True)
    reporter.debug("resolver probe")
    reporter.stage_passed(position)
    reporter.installation_finished(
        outcome=ProgressOutcome.SUCCESS,
        exit_code=0,
        message="installation complete",
        state_path=tmp_path / "state.json",
        project_dir=tmp_path / "project",
        position=position,
    )
    log_path = reporter.log_path
    reporter.close()

    assert log_path is not None
    terminal = stdout.getvalue() + stderr.getvalue()
    for token in present:
        assert token in terminal
    for token in absent:
        assert token not in terminal
    log = log_path.read_text(encoding="utf-8")
    for token in (
        "PLAN",
        "START",
        "COMMAND",
        "DEBUG",
        "PASS",
        "SUMMARY",
    ):
        assert token in log


def test_heartbeat_schedule_repeats_only_after_fifteen_seconds() -> None:
    schedule = HeartbeatSchedule(interval=15.0, last_activity=100.0)
    assert schedule.consume_if_due(114.999) is False
    assert schedule.consume_if_due(115.0) is True
    assert schedule.consume_if_due(129.999) is False
    assert schedule.consume_if_due(130.0) is True
    schedule.activity(131.0)
    assert schedule.consume_if_due(145.999) is False
    assert schedule.consume_if_due(146.0) is True
    schedule.pause()
    assert schedule.consume_if_due(200.0) is False
    schedule.resume(200.0)
    assert schedule.consume_if_due(214.999) is False
    assert schedule.consume_if_due(215.0) is True


def test_failure_summary_uses_bounded_tail_and_recovery_paths(
    tmp_path: Path,
) -> None:
    reporter, _, stderr = opened_reporter(tmp_path, ProgressMode.DEFAULT)
    position = StagePosition(InstallStage.PROJECT_INIT, 6, 8)
    reporter.stage_started(position)
    for number in range(80):
        reporter.command_output(CommandStream.STDERR, f"line-{number}\n")
    reporter.installation_finished(
        outcome=ProgressOutcome.FAILURE,
        exit_code=2,
        message="PROJECT_INIT failed: uv download failed",
        state_path=tmp_path / "state.json",
        project_dir=tmp_path / "project",
        position=position,
    )
    reporter.close()

    output = stderr.getvalue()
    assert "FAIL     [6/8]" in output
    failure = output[output.index("FAIL") :]
    assert "CAUSE    PROJECT_INIT failed: uv download failed" in failure
    assert "line-0" not in failure
    assert "line-79" in failure
    cause = failure.partition("CAUSE")[2].partition("STATE")[0]
    assert cause.count("line-") <= 40
    assert len(cause.encode("utf-8")) <= 16 * 1024
    assert "STATE" in failure and "RESUME" in failure and "LOG" in failure


def test_failure_summary_caps_an_embedded_command_tail(tmp_path: Path) -> None:
    reporter, _, stderr = opened_reporter(tmp_path, ProgressMode.DEFAULT)
    position = StagePosition(InstallStage.PROJECT_INIT, 6, 8)
    reporter.stage_started(position)
    reporter.installation_finished(
        outcome=ProgressOutcome.FAILURE,
        exit_code=2,
        message="failure: " + "x" * (300 * 1024),
        state_path=tmp_path / "state.json",
        project_dir=tmp_path / "project",
        position=position,
    )
    reporter.close()

    cause = stderr.getvalue().partition("CAUSE")[2].partition("STATE")[0]
    assert "failure:" in cause
    assert len(cause.encode("utf-8")) <= 16 * 1024


def test_pre_stage_failure_has_no_fabricated_position(tmp_path: Path) -> None:
    reporter, _, stderr = opened_reporter(tmp_path, ProgressMode.DEFAULT)
    reporter.installation_finished(
        outcome=ProgressOutcome.FAILURE,
        exit_code=2,
        message="invalid transition",
        state_path=tmp_path / "state.json",
        project_dir=tmp_path / "project",
        position=None,
    )
    reporter.close()

    output = stderr.getvalue()
    assert output.startswith("FAIL     installer")
    assert "[" not in output.partition("\n")[0]
    assert all(token in output for token in ("CAUSE", "STATE", "RESUME", "LOG"))


def test_reporter_redacts_environment_and_url_secrets_from_every_sink(
    tmp_path: Path,
) -> None:
    reporter, stdout, stderr = opened_reporter(
        tmp_path, ProgressMode.VERBOSE
    )
    position = StagePosition(InstallStage.PROJECT_INIT, 6, 8)
    reporter.stage_started(position)
    reporter.command_started(
        (
            "uv",
            "pip",
            "compile",
            "--index-url=https://alice:ghp_private@packages.example/simple",
        ),
        live=True,
        environment={"HF_TOKEN": "hf_private_value", "PATH": "/usr/bin"},
    )
    reporter.command_output(
        CommandStream.STDOUT, "token=hf_private_value\n"
    )
    reporter.command_output(
        CommandStream.STDERR,
        "https://alice:ghp_private@packages.example/simple failed\n",
    )
    log_path = reporter.log_path
    reporter.installation_finished(
        outcome=ProgressOutcome.FAILURE,
        exit_code=2,
        message="dependency failure",
        state_path=tmp_path / "state.json",
        project_dir=tmp_path / "project",
        position=position,
    )
    reporter.close()

    assert log_path is not None
    rendered = (
        stdout.getvalue()
        + stderr.getvalue()
        + log_path.read_text(encoding="utf-8")
    )
    assert "<redacted>" in rendered
    assert "hf_private_value" not in rendered
    assert "ghp_private" not in rendered


@pytest.mark.parametrize(
    ("mode", "command_visible", "output_visible"),
    [
        (ProgressMode.DEFAULT, False, True),
        (ProgressMode.VERBOSE, True, True),
        (ProgressMode.QUIET, False, False),
    ],
)
def test_stderr_command_observer_preserves_protocol_channel_and_modes(
    mode: ProgressMode,
    command_visible: bool,
    output_visible: bool,
) -> None:
    stderr = io.StringIO()
    observer = progress_module.StderrCommandObserver(
        mode=mode, stderr=stderr
    )
    command = (
        "apt-get",
        "install",
        "HF_TOKEN=hf_private",
    )
    observer.command_started(
        command,
        live=True,
        environment={"HF_TOKEN": "hf_private"},
    )
    first = observer.command_output(
        CommandStream.STDOUT, "installing hf_private\n"
    )
    second = observer.command_output(
        CommandStream.STDERR, "warning hf_private\n"
    )
    observer.command_finished(
        CommandResult(command, 0, first, second), live=True
    )

    rendered = stderr.getvalue()
    assert ("COMMAND" in rendered) is command_visible
    assert ("installing" in rendered) is output_visible
    assert ("warning" in rendered) is output_visible
    assert "hf_private" not in rendered
    assert "hf_private" not in first
    assert "hf_private" not in second


def test_heartbeat_worker_stops_before_pass(tmp_path: Path) -> None:
    stdout = io.StringIO()
    reporter = InstallerProgress(
        mode=ProgressMode.DEFAULT,
        stdout=stdout,
        stderr=io.StringIO(),
        log_root=tmp_path / "logs",
        heartbeat_interval=0.02,
        process_id=31,
    )
    reporter.open_session(tmp_path / "project")
    position = StagePosition(InstallStage.PROJECT_INIT, 6, 8)
    reporter.stage_started(position)
    deadline = time.monotonic() + 1.0
    while "WAIT" not in stdout.getvalue() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert "WAIT" in stdout.getvalue()
    reporter.stage_passed(position)
    after_pass = stdout.getvalue()
    time.sleep(0.05)
    reporter.close()
    assert stdout.getvalue() == after_pass
