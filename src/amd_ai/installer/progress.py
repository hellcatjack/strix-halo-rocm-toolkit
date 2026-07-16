from __future__ import annotations

import os
import re
import shlex
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, TextIO

from amd_ai.installer.models import InstallMode, InstallStage
from amd_ai.installer.state import project_identity_key
from amd_ai.runner import CommandResult, CommandStream


CSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
OSC_PATTERN = re.compile(r"\x1b\].*?(?:\x07|\x1b\\)", re.DOTALL)
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
URL_USERINFO_PATTERN = re.compile(
    r"(?P<scheme>\b[A-Za-z][A-Za-z0-9+.-]{0,31}://)[^\s/@]+@"
)
SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r"(?P<name>\b[A-Za-z_][A-Za-z0-9_]*"
    r"(?:TOKEN|PASSWORD|PASS|SECRET|KEY|CREDENTIAL|AUTH))=[^\s]+",
    re.IGNORECASE,
)
AUTHORIZATION_PATTERN = re.compile(
    r"(?P<label>\bAuthorization\s*:\s*(?:Bearer|Basic)\s+)[^\s]+",
    re.IGNORECASE,
)
CREDENTIAL_FLAG_PATTERN = re.compile(
    r"(?P<flag>--(?:token|password|secret|key|credential|auth|index-url)"
    r"(?:=|\s+))[^\s]+",
    re.IGNORECASE,
)
URL_USERINFO_VALUE_PATTERN = re.compile(
    r"\b[A-Za-z][A-Za-z0-9+.-]{0,31}://(?P<userinfo>[^\s/@]+)@"
)
SENSITIVE_ENVIRONMENT_PATTERN = re.compile(
    r"(?:TOKEN|PASSWORD|PASS|SECRET|KEY|CREDENTIAL|AUTH)$",
    re.IGNORECASE,
)
EVENT_KINDS = frozenset(
    {
        "PLAN",
        "LOG",
        "SKIP",
        "START",
        "DETAIL",
        "WAIT",
        "PASS",
        "INFO",
        "WARN",
        "ACTION",
        "BLOCKED",
        "FAIL",
        "CAUSE",
        "STATE",
        "RESUME",
        "SUMMARY",
        "COMMAND",
        "DEBUG",
    }
)
VERBOSE_ONLY = frozenset({"COMMAND", "DEBUG"})
QUIET_VISIBLE = frozenset(
    {
        "WARN",
        "ACTION",
        "BLOCKED",
        "FAIL",
        "CAUSE",
        "STATE",
        "RESUME",
        "LOG",
        "SUMMARY",
    }
)
STDERR_EVENTS = frozenset(
    {"WARN", "ACTION", "BLOCKED", "FAIL", "CAUSE", "STATE", "RESUME"}
)
FAILURE_CONTENT_BYTES = 15 * 1024

STAGE_LABELS: Mapping[InstallStage, str] = MappingProxyType(
    {
        InstallStage.BOOTSTRAP: "安装用户运行时",
        InstallStage.HOST_PREFLIGHT: "检查宿主",
        InstallStage.KERNEL_PLAN: "生成 OEM 6.17 内核计划",
        InstallStage.KERNEL_CONFIRM: "确认内核变更",
        InstallStage.KERNEL_APPLY: "安装 OEM 6.17 内核",
        InstallStage.KERNEL_REBOOT_PENDING: "等待内核重启",
        InstallStage.KERNEL_VERIFY: "验证桌面与 GPU 内核",
        InstallStage.HOST_PLAN: "生成主机平台计划",
        InstallStage.HOST_CONFIRM: "确认主机平台准备",
        InstallStage.HOST_APPLY: "应用 Docker 与设备权限准备",
        InstallStage.HOST_VERIFY: "验证宿主平台",
        InstallStage.CONTAINER_HOST_CHECK: "验证容器宿主",
        InstallStage.RELEASE_RESOLVE: "解析 stable release",
        InstallStage.IMAGE_PULL_OR_BUILD: "获取或构建 stable 镜像",
        InstallStage.IMAGE_VERIFY: "验证 gfx1151 PyTorch GPU runtime",
        InstallStage.PROJECT_INIT: "创建项目镜像与 Python 依赖",
        InstallStage.PROJECT_VERIFY: "验证项目",
        InstallStage.COMPLETE: "完成安装",
    }
)


class ProgressError(RuntimeError):
    pass


class ProgressMode(StrEnum):
    DEFAULT = "default"
    VERBOSE = "verbose"
    QUIET = "quiet"


class ProgressOutcome(StrEnum):
    SUCCESS = "success"
    ACTION = "action"
    BLOCKED = "blocked"
    FAILURE = "failure"


@dataclass(frozen=True)
class StagePosition:
    stage: InstallStage
    index: int
    total: int


@dataclass(frozen=True)
class SessionPlan:
    mode: InstallMode
    project_dir: Path
    project_name: str
    state_path: Path
    state_source: str
    image_source: str
    release_id: str | None
    stages: tuple[InstallStage, ...]
    first_incomplete: InstallStage | None


@dataclass
class HeartbeatSchedule:
    interval: float
    last_activity: float
    paused: bool = False

    def activity(self, now: float) -> None:
        self.last_activity = now

    def consume_if_due(self, now: float) -> bool:
        if self.paused or now - self.last_activity < self.interval:
            return False
        self.last_activity = now
        return True

    def remaining(self, now: float) -> float:
        if self.paused:
            return self.interval
        return max(0.0, self.interval - (now - self.last_activity))

    def pause(self) -> None:
        self.paused = True

    def resume(self, now: float) -> None:
        self.paused = False
        self.activity(now)


class ProgressReporter(Protocol):
    @property
    def log_path(self) -> Path | None: ...

    def open_session(self, project_dir: Path) -> None: ...

    def session_plan(self, plan: SessionPlan) -> None: ...

    def stage_candidate(self, position: StagePosition) -> None: ...

    def stage_skipped(self, position: StagePosition) -> None: ...

    def stage_started(self, position: StagePosition) -> None: ...

    def detail(self, message: str) -> None: ...

    def stage_passed(self, position: StagePosition) -> None: ...

    def status(self, prefix: str, message: str) -> None: ...

    def debug(self, message: str) -> None: ...

    def pause_heartbeat(self) -> None: ...

    def resume_heartbeat(self) -> None: ...

    def installation_finished(
        self,
        *,
        outcome: ProgressOutcome,
        exit_code: int,
        message: str,
        state_path: Path,
        project_dir: Path | None,
        position: StagePosition | None,
    ) -> None: ...

    def fallback_failure(self, message: str) -> None: ...

    def close(self) -> None: ...


def default_log_root() -> Path:
    return Path.home() / ".local/state/strix-halo-rocm-toolkit/logs"


def sanitize_output(
    value: str, *, secret_values: Iterable[str] = ()
) -> str:
    rendered = value.replace("\r\n", "\n").replace("\r", "\n")
    rendered = OSC_PATTERN.sub("", rendered)
    rendered = CSI_PATTERN.sub("", rendered)
    rendered = CONTROL_PATTERN.sub("", rendered)
    secrets = {item for item in secret_values if item}
    for secret in sorted(secrets, key=len, reverse=True):
        rendered = rendered.replace(secret, "<redacted>")
    rendered = URL_USERINFO_PATTERN.sub(
        r"\g<scheme><redacted>@", rendered
    )
    rendered = SENSITIVE_ASSIGNMENT_PATTERN.sub(
        r"\g<name>=<redacted>", rendered
    )
    rendered = AUTHORIZATION_PATTERN.sub(
        r"\g<label><redacted>", rendered
    )
    return CREDENTIAL_FLAG_PATTERN.sub(r"\g<flag><redacted>", rendered)


class SessionLog:
    def __init__(
        self,
        *,
        path: Path,
        stream: TextIO,
        wall_clock: Callable[[], datetime],
    ) -> None:
        self.path = path
        self._stream = stream
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._closed = False

    @classmethod
    def open(
        cls,
        *,
        project_dir: Path,
        log_root: Path,
        wall_clock: Callable[[], datetime],
        process_id: int,
    ) -> SessionLog:
        root = Path(log_root).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        _ensure_private_directory(root.parent, parents=True)
        _ensure_private_directory(root)
        project_root = root / project_identity_key(project_dir)
        _ensure_private_directory(project_root)

        now = _utc_now(wall_clock)
        timestamp = now.strftime("%Y%m%dT%H%M%SZ")
        path = project_root / f"install-{timestamp}-{process_id}.log"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError as error:
            raise ProgressError(
                f"installer log already exists: {path}"
            ) from error
        except OSError as error:
            raise ProgressError(
                f"cannot create installer log {path}: {error}"
            ) from error
        try:
            os.fchmod(descriptor, 0o600)
            stream = os.fdopen(
                descriptor,
                "w",
                encoding="utf-8",
                newline="\n",
            )
        except Exception:
            os.close(descriptor)
            path.unlink(missing_ok=True)
            raise
        return cls(path=path, stream=stream, wall_clock=wall_clock)

    def write(self, stream: str, kind: str, text: str) -> None:
        rendered = sanitize_output(text)
        with self._lock:
            if self._closed:
                raise ProgressError("installer log is closed")
            timestamp = _utc_now(self._wall_clock).isoformat(
                timespec="milliseconds"
            ).replace("+00:00", "Z")
            lines = rendered.split("\n")
            if lines and lines[-1] == "":
                lines.pop()
            if not lines:
                lines = [""]
            try:
                for line in lines:
                    self._stream.write(
                        f"{timestamp} {stream} {kind} {line}\n"
                    )
                self._stream.flush()
            except OSError as error:
                raise ProgressError(
                    f"cannot write installer log {self.path}: {error}"
                ) from error

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._stream.flush()
                os.fsync(self._stream.fileno())
            except OSError as error:
                raise ProgressError(
                    f"cannot flush installer log {self.path}: {error}"
                ) from error
            finally:
                self._stream.close()
                self._closed = True


class InstallerProgress:
    def __init__(
        self,
        *,
        mode: ProgressMode,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        log_root: Path | None = None,
        wall_clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
        heartbeat_interval: float = 15.0,
        process_id: int | None = None,
    ) -> None:
        self.mode = ProgressMode(mode)
        self._stdout = stdout if stdout is not None else sys.stdout
        self._stderr = stderr if stderr is not None else sys.stderr
        self._log_root = log_root
        self._wall_clock = wall_clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic or time.monotonic
        if heartbeat_interval <= 0:
            raise ProgressError("heartbeat interval must be positive")
        self._heartbeat_interval = heartbeat_interval
        self._process_id = os.getpid() if process_id is None else process_id
        self._lock = threading.RLock()
        self._log: SessionLog | None = None
        self._session_started_at = self._monotonic()
        self._current_position: StagePosition | None = None
        self._stage_started_at: float | None = None
        self._tail: deque[str] = deque(maxlen=40)
        self._command_secrets: set[str] = set()
        self._heartbeat_schedule: HeartbeatSchedule | None = None
        self._heartbeat_stop: threading.Event | None = None
        self._heartbeat_wake: threading.Event | None = None
        self._heartbeat_thread: threading.Thread | None = None

    @property
    def log_path(self) -> Path | None:
        with self._lock:
            return None if self._log is None else self._log.path

    def open_session(self, project_dir: Path) -> None:
        with self._lock:
            if self._log is not None:
                raise ProgressError("installer progress session is already open")
            root = self._log_root or default_log_root()
            self._log = SessionLog.open(
                project_dir=project_dir,
                log_root=root,
                wall_clock=self._wall_clock,
                process_id=self._process_id,
            )

    def session_plan(self, plan: SessionPlan) -> None:
        source = {
            "project": "per-project",
            "legacy": "legacy",
            "explicit": "explicit",
        }.get(plan.state_source, plan.state_source)
        self._emit(
            "PLAN",
            f"模式={plan.mode.value}，项目={plan.project_dir}，"
            f"名称={plan.project_name}",
        )
        self._emit("PLAN", f"状态={plan.state_path}（{source}）")
        release_id = plan.release_id or "待解析"
        self._emit(
            "PLAN",
            f"镜像来源={plan.image_source}，stable release={release_id}",
        )
        if plan.first_incomplete is None:
            summary = f"共 {len(plan.stages)} 个阶段，已经全部完成"
        else:
            summary = (
                f"共 {len(plan.stages)} 个阶段，从 "
                f"{plan.first_incomplete.value} 继续"
            )
        self._emit("PLAN", summary)
        path = self.log_path
        if path is not None:
            self._emit("LOG", str(path), quiet_visible=False)

    def stage_candidate(self, position: StagePosition) -> None:
        with self._lock:
            self._current_position = position
            self._stage_started_at = None

    def stage_skipped(self, position: StagePosition) -> None:
        self.stage_candidate(position)
        self._emit(
            "SKIP",
            f"{_position_prefix(position)} {_stage_label(position.stage)}："
            "已有可信检查点",
        )

    def stage_started(self, position: StagePosition) -> None:
        self._stop_heartbeat()
        now = self._monotonic()
        with self._lock:
            self._current_position = position
            self._stage_started_at = now
            self._tail.clear()
            self._heartbeat_schedule = HeartbeatSchedule(
                interval=self._heartbeat_interval,
                last_activity=now,
            )
        suffix = (
            f"（{position.stage.value}）"
            if self.mode is ProgressMode.VERBOSE
            else ""
        )
        self._emit(
            "START",
            f"{_position_prefix(position)} {_stage_label(position.stage)}"
            f"{suffix}",
        )
        self._start_heartbeat()

    def detail(self, message: str) -> None:
        self._emit("DETAIL", message)

    def stage_passed(self, position: StagePosition) -> None:
        self._stop_heartbeat()
        suffix = (
            f"（{position.stage.value}）"
            if self.mode is ProgressMode.VERBOSE
            else ""
        )
        self._emit(
            "PASS",
            f"{_position_prefix(position)} {_stage_label(position.stage)}"
            f"{suffix}，用时 {self._stage_duration()}",
        )

    def status(self, prefix: str, message: str) -> None:
        self._emit(prefix, message)

    def debug(self, message: str) -> None:
        self._emit("DEBUG", message)

    def pause_heartbeat(self) -> None:
        with self._lock:
            if self._heartbeat_schedule is not None:
                self._heartbeat_schedule.pause()
                self._wake_heartbeat_locked()

    def resume_heartbeat(self) -> None:
        with self._lock:
            if self._heartbeat_schedule is not None:
                self._heartbeat_schedule.resume(self._monotonic())
                self._wake_heartbeat_locked()

    def command_started(
        self,
        args: tuple[str, ...],
        *,
        live: bool,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        secrets = _command_secret_values(args, environment)
        with self._lock:
            self._command_secrets = secrets
        rendered = sanitize_output(
            shlex.join(args), secret_values=secrets
        )
        disposition = "live" if live else "captured"
        self._emit("COMMAND", f"[{disposition}] {rendered}")

    def command_output(self, stream: CommandStream, text: str) -> str:
        with self._lock:
            rendered = sanitize_output(
                text, secret_values=self._command_secrets
            )
            for line in rendered.splitlines():
                self._tail.append(line)
            if self._log is not None:
                self._log.write(stream.value, "CHILD", rendered)
            if self.mode is not ProgressMode.QUIET and rendered:
                target = (
                    self._stdout
                    if stream is CommandStream.STDOUT
                    else self._stderr
                )
                target.write(rendered)
                if not rendered.endswith("\n"):
                    target.write("\n")
                target.flush()
            self._activity_locked()
            return rendered

    def command_finished(
        self, result: CommandResult, *, live: bool
    ) -> None:
        disposition = "live" if live else "captured"
        self._emit(
            "DEBUG",
            f"[{disposition}] returncode={result.returncode} "
            f"stdout_bytes={len(result.stdout.encode('utf-8'))} "
            f"stderr_bytes={len(result.stderr.encode('utf-8'))}",
        )
        with self._lock:
            self._command_secrets.clear()

    def installation_finished(
        self,
        *,
        outcome: ProgressOutcome,
        exit_code: int,
        message: str,
        state_path: Path,
        project_dir: Path | None,
        position: StagePosition | None,
    ) -> None:
        del exit_code
        self._stop_heartbeat()
        active = position or self._current_position
        events: list[tuple[str, str, bool]] = []
        if outcome is ProgressOutcome.SUCCESS:
            project = str(project_dir) if project_dir is not None else "未知"
            result_detail = (
                f"，结果={message}"
                if message and message != "installation complete"
                else ""
            )
            events.append(
                (
                    "SUMMARY",
                    f"安装完成，用时 {self._session_duration()}，"
                    f"项目={project}，状态={state_path}{result_detail}",
                    False,
                )
            )
        elif outcome is ProgressOutcome.ACTION:
            subject = self._failure_subject(active, include_duration=False)
            events.append(
                (
                    "ACTION",
                    f"{subject}：{message or '需要操作员处理'}",
                    True,
                )
            )
            events.append(("STATE", str(state_path), True))
            if "manual reboot" in message.casefold():
                resume = "sudo reboot；重启后重新执行同一条 install 命令"
            else:
                resume = "完成上述操作后重新执行同一条 install 命令"
            events.append(("RESUME", resume, True))
        else:
            prefix = (
                "BLOCKED"
                if outcome is ProgressOutcome.BLOCKED
                else "FAIL"
            )
            events.append(
                (
                    prefix,
                    self._failure_subject(active, include_duration=True),
                    True,
                )
            )
            events.extend(
                ("CAUSE", line, True)
                for line in self._failure_lines(message)
            )
            events.append(("STATE", str(state_path), True))
            events.append(
                (
                    "RESUME",
                    "修复问题后重新执行同一条 install 命令；"
                    "已完成阶段不会重放",
                    True,
                )
            )
        path = self.log_path
        if path is not None:
            events.append(("LOG", str(path), outcome is not ProgressOutcome.SUCCESS))
        self._emit_final_events(events)

    def fallback_failure(self, message: str) -> None:
        self._stop_heartbeat()
        rendered = _truncate_utf8(
            sanitize_output(message), FAILURE_CONTENT_BYTES
        )
        with self._lock:
            self._stderr.write(
                "FAIL     installer progress reporting failed\n"
            )
            for line in rendered.splitlines() or ["unknown progress error"]:
                self._stderr.write(f"CAUSE    {line}\n")
            self._stderr.flush()

    def close(self) -> None:
        self._stop_heartbeat()
        with self._lock:
            log = self._log
        if log is not None:
            log.close()

    def _failure_subject(
        self,
        position: StagePosition | None,
        *,
        include_duration: bool,
    ) -> str:
        if position is None:
            return "installer"
        subject = (
            f"{_position_prefix(position)} {_stage_label(position.stage)}"
            f"（{position.stage.value}）"
        )
        if include_duration:
            subject += f"，用时 {self._stage_duration()}"
        return subject

    def _failure_lines(self, message: str) -> tuple[str, ...]:
        with self._lock:
            rendered = sanitize_output(
                message, secret_values=self._command_secrets
            )
            first = next(
                (line for line in rendered.splitlines() if line),
                "installer failed",
            )
            tail = tuple(self._tail)
        first = _truncate_utf8(first, FAILURE_CONTENT_BYTES)
        selected: list[str] = []
        used = len(first.encode("utf-8")) + 1
        for line in reversed(tail):
            encoded = len(line.encode("utf-8")) + 1
            if used + encoded > FAILURE_CONTENT_BYTES:
                continue
            selected.append(line)
            used += encoded
        selected.reverse()
        return (first, *selected)

    def _emit_final_events(
        self, events: Sequence[tuple[str, str, bool]]
    ) -> None:
        for kind, message, is_error in events:
            self._emit(
                kind,
                message,
                terminal_enabled=False,
                error_stream=is_error,
            )
        self.close()
        for kind, message, is_error in events:
            self._emit_terminal_event(
                kind,
                message,
                error_stream=is_error,
            )

    def _emit_terminal_event(
        self,
        kind: str,
        message: str,
        *,
        error_stream: bool,
    ) -> None:
        with self._lock:
            rendered = sanitize_output(
                message, secret_values=self._command_secrets
            )
            target = self._stderr if error_stream else self._stdout
            for line in rendered.splitlines() or [""]:
                target.write(f"{kind:<8} {line}\n")
            target.flush()

    def _emit(
        self,
        kind: str,
        message: str,
        *,
        force_terminal: bool = False,
        quiet_visible: bool | None = None,
        error_stream: bool | None = None,
        terminal_enabled: bool = True,
    ) -> None:
        if kind not in EVENT_KINDS:
            raise ProgressError(f"unsupported progress event: {kind}")
        with self._lock:
            rendered = sanitize_output(
                message, secret_values=self._command_secrets
            )
            is_error = (
                kind in STDERR_EVENTS
                if error_stream is None
                else error_stream
            )
            stream_name = "stderr" if is_error else "stdout"
            if self._log is not None:
                self._log.write(stream_name, kind, rendered)
            visible = terminal_enabled and self._terminal_visible(
                kind,
                force_terminal=force_terminal,
                quiet_visible=quiet_visible,
            )
            if visible:
                target = self._stderr if is_error else self._stdout
                lines = rendered.splitlines() or [""]
                for line in lines:
                    target.write(f"{kind:<8} {line}\n")
                target.flush()
            self._activity_locked()

    def _terminal_visible(
        self,
        kind: str,
        *,
        force_terminal: bool,
        quiet_visible: bool | None,
    ) -> bool:
        if force_terminal:
            return True
        if kind in VERBOSE_ONLY:
            return self.mode is ProgressMode.VERBOSE
        if self.mode is not ProgressMode.QUIET:
            return True
        if quiet_visible is not None:
            return quiet_visible
        return kind in QUIET_VISIBLE

    def _activity_locked(self) -> None:
        if self._heartbeat_schedule is not None:
            self._heartbeat_schedule.activity(self._monotonic())
            self._wake_heartbeat_locked()

    def _wake_heartbeat_locked(self) -> None:
        if self._heartbeat_wake is not None:
            self._heartbeat_wake.set()

    def _start_heartbeat(self) -> None:
        if self.mode is ProgressMode.QUIET:
            return
        with self._lock:
            stop = threading.Event()
            wake = threading.Event()
            self._heartbeat_stop = stop
            self._heartbeat_wake = wake
            thread = threading.Thread(
                target=self._heartbeat_worker,
                args=(stop, wake),
                name="installer-progress-heartbeat",
                daemon=True,
            )
            self._heartbeat_thread = thread
        thread.start()

    def _heartbeat_worker(
        self, stop: threading.Event, wake: threading.Event
    ) -> None:
        while not stop.is_set():
            with self._lock:
                schedule = self._heartbeat_schedule
                remaining = (
                    self._heartbeat_interval
                    if schedule is None
                    else schedule.remaining(self._monotonic())
                )
            signaled = wake.wait(timeout=max(remaining, 0.001))
            wake.clear()
            if stop.is_set():
                return
            if signaled:
                continue
            with self._lock:
                schedule = self._heartbeat_schedule
                position = self._current_position
                due = (
                    schedule is not None
                    and schedule.consume_if_due(self._monotonic())
                )
            if due and position is not None:
                self._emit(
                    "WAIT",
                    f"{_position_prefix(position)} 已运行 "
                    f"{self._stage_duration()}，15 秒内没有新输出，"
                    f"仍在{_stage_label(position.stage)}",
                )

    def _stop_heartbeat(self) -> None:
        with self._lock:
            stop = self._heartbeat_stop
            wake = self._heartbeat_wake
            thread = self._heartbeat_thread
            if stop is not None:
                stop.set()
            if wake is not None:
                wake.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        with self._lock:
            if self._heartbeat_thread is thread:
                self._heartbeat_stop = None
                self._heartbeat_wake = None
                self._heartbeat_thread = None

    def _stage_duration(self) -> str:
        with self._lock:
            started = self._stage_started_at
        elapsed = 0.0 if started is None else self._monotonic() - started
        return _format_duration(elapsed)

    def _session_duration(self) -> str:
        return _format_duration(self._monotonic() - self._session_started_at)


class PromptProgressAdapter:
    def __init__(self, status_fn: Callable[[str, str], None]) -> None:
        self._status_fn = status_fn
        self._position: StagePosition | None = None

    @property
    def log_path(self) -> Path | None:
        return None

    def open_session(self, project_dir: Path) -> None:
        del project_dir

    def session_plan(self, plan: SessionPlan) -> None:
        self._status_fn(
            "PLAN", f"模式={plan.mode.value}，项目={plan.project_dir}"
        )

    def stage_candidate(self, position: StagePosition) -> None:
        self._position = position

    def stage_skipped(self, position: StagePosition) -> None:
        self.stage_candidate(position)
        self._status_fn(
            "SKIP",
            f"{_position_prefix(position)} {_stage_label(position.stage)}："
            "已有可信检查点",
        )

    def stage_started(self, position: StagePosition) -> None:
        self.stage_candidate(position)
        self._status_fn(
            "START",
            f"{_position_prefix(position)} {_stage_label(position.stage)}",
        )

    def detail(self, message: str) -> None:
        self._status_fn("DETAIL", message)

    def stage_passed(self, position: StagePosition) -> None:
        self._status_fn(
            "PASS",
            f"{_position_prefix(position)} {_stage_label(position.stage)}",
        )

    def status(self, prefix: str, message: str) -> None:
        self._status_fn(prefix, message)

    def debug(self, message: str) -> None:
        del message

    def pause_heartbeat(self) -> None:
        pass

    def resume_heartbeat(self) -> None:
        pass

    def installation_finished(
        self,
        *,
        outcome: ProgressOutcome,
        exit_code: int,
        message: str,
        state_path: Path,
        project_dir: Path | None,
        position: StagePosition | None,
    ) -> None:
        del exit_code, project_dir
        if outcome is ProgressOutcome.SUCCESS:
            self._status_fn("SUMMARY", message or "installation complete")
            return
        prefix = {
            ProgressOutcome.ACTION: "ACTION",
            ProgressOutcome.BLOCKED: "BLOCKED",
            ProgressOutcome.FAILURE: "FAIL",
        }[outcome]
        subject = (
            "installer"
            if position is None
            else f"{_position_prefix(position)} {_stage_label(position.stage)}"
        )
        self._status_fn(prefix, subject)
        if message:
            self._status_fn("CAUSE", message)
        self._status_fn("STATE", str(state_path))

    def fallback_failure(self, message: str) -> None:
        self._status_fn("FAIL", "installer progress reporting failed")
        self._status_fn("CAUSE", message)

    def close(self) -> None:
        pass


class StderrCommandObserver:
    def __init__(
        self,
        *,
        mode: ProgressMode,
        stderr: TextIO,
    ) -> None:
        self.mode = ProgressMode(mode)
        self._stderr = stderr
        self._lock = threading.Lock()
        self._command_secrets: set[str] = set()

    def command_started(
        self,
        args: tuple[str, ...],
        *,
        live: bool,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        secrets = _command_secret_values(args, environment)
        with self._lock:
            self._command_secrets = secrets
            if self.mode is not ProgressMode.VERBOSE:
                return
            command = sanitize_output(
                shlex.join(args), secret_values=secrets
            )
            disposition = "live" if live else "captured"
            self._write_locked(f"COMMAND  [{disposition}] {command}\n")

    def command_output(self, stream: CommandStream, text: str) -> str:
        del stream
        with self._lock:
            rendered = sanitize_output(
                text, secret_values=self._command_secrets
            )
            if self.mode is not ProgressMode.QUIET and rendered:
                self._write_locked(rendered)
                if not rendered.endswith("\n"):
                    self._write_locked("\n")
            return rendered

    def command_finished(
        self, result: CommandResult, *, live: bool
    ) -> None:
        del result, live
        with self._lock:
            self._command_secrets.clear()

    def _write_locked(self, value: str) -> None:
        try:
            self._stderr.write(value)
            self._stderr.flush()
        except OSError as error:
            raise ProgressError(
                f"cannot write privileged progress: {error}"
            ) from error


def _command_secret_values(
    args: Sequence[str], environment: Mapping[str, str] | None
) -> set[str]:
    secrets = {
        value
        for name, value in (environment or {}).items()
        if value and SENSITIVE_ENVIRONMENT_PATTERN.search(name)
    }
    credential_flags = {
        "--token",
        "--password",
        "--secret",
        "--key",
        "--credential",
        "--auth",
        "--index-url",
    }
    index = 0
    while index < len(args):
        argument = args[index]
        name, separator, value = argument.partition("=")
        if separator and SENSITIVE_ENVIRONMENT_PATTERN.search(name):
            secrets.add(value)
        if separator and name.casefold() in credential_flags:
            secrets.add(value)
        elif argument.casefold() in credential_flags and index + 1 < len(args):
            secrets.add(args[index + 1])
            index += 1
        for match in URL_USERINFO_VALUE_PATTERN.finditer(argument):
            secrets.add(match.group("userinfo"))
        index += 1
    return {value for value in secrets if value}


def _position_prefix(position: StagePosition) -> str:
    return f"[{position.index}/{position.total}]"


def _stage_label(stage: InstallStage) -> str:
    return STAGE_LABELS[InstallStage(stage)]


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, remainder = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remainder:02d}"
    return f"{minutes:02d}:{remainder:02d}"


def _truncate_utf8(value: str, maximum: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum:
        return value
    return encoded[:maximum].decode("utf-8", errors="ignore")


def _ensure_private_directory(path: Path, *, parents: bool = False) -> None:
    if os.path.lexists(path):
        if path.is_symlink():
            raise ProgressError(
                f"installer log directory is a symlink: {path}"
            )
        if not path.is_dir():
            raise ProgressError(
                f"installer log control path is not a directory: {path}"
            )
    else:
        try:
            path.mkdir(parents=parents, mode=0o700)
        except OSError as error:
            raise ProgressError(
                f"cannot create installer log directory {path}: {error}"
            ) from error
    try:
        metadata = path.stat(follow_symlinks=False)
        if metadata.st_uid != os.geteuid():
            raise ProgressError(
                f"installer log directory is not owned by current user: {path}"
            )
        path.chmod(0o700, follow_symlinks=False)
    except OSError as error:
        raise ProgressError(
            f"cannot secure installer log directory {path}: {error}"
        ) from error


def _utc_now(wall_clock: Callable[[], datetime]) -> datetime:
    value = wall_clock()
    if value.tzinfo is None:
        raise ProgressError("installer log clock must be timezone-aware")
    return value.astimezone(UTC)
