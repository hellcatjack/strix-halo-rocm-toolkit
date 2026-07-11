# Installer Real-time Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every installer run show structured stage progress, live sanitized output for long Docker/package operations, private session logs, and exact recovery guidance without changing stable images or trusted checkpoints.

**Architecture:** Add one installer progress reporter that owns event formatting, output modes, heartbeat timing, redaction, and secure logs. Extend the existing subprocess runner with an observed tee path while preserving the `Runner.run` signature, then inject the same reporter through the workflow, Docker release acquisition, project builds, and the privileged helper's stderr-only progress channel. Keep progress data outside install state and migrate only the already-versioned `BOOTSTRAP` digest when a compatible `v0.2.2` container state resumes under `v0.2.3`.

**Tech Stack:** Python 3.12, standard-library `argparse`, `dataclasses`, `enum`, `hashlib`, `pathlib`, `subprocess`, `threading`, pytest 8.4, Docker/Buildx, Bash launcher, Markdown documentation.

---

## File map

- Create `src/amd_ai/installer/progress.py`: progress modes, stage labels, event rendering, heartbeat, redaction, private session logs, command-observer implementation, and the prompt-only compatibility adapter.
- Modify `src/amd_ai/runner.py`: command-observer protocol, live-command classification, concurrent stdout/stderr tee, bounded stream tails, and protocol-command execution.
- Modify `src/amd_ai/installer/state.py`: extract one reusable project identity key for both state and log paths.
- Modify `src/amd_ai/installer/workflow.py`: plan/start/skip/detail/pass/failure/final events, disk detail, final recovery output, and compatible container-state adoption.
- Modify `src/amd_ai/installer/prompts.py`: accept the complete structured event vocabulary while retaining prompt input behavior.
- Modify `src/amd_ai/installer/actions.py`: share the observed runner with release pulls and builds, and isolate privileged stdout JSON from streamed stderr.
- Modify `src/amd_ai/installer/privileged.py`: accept an explicit progress mode and route root command output only to stderr.
- Modify `src/amd_ai/image/publish.py`: execute pulls through an injected runner while keeping inspect and JSON commands captured.
- Modify `src/amd_ai/image/lock.py`: report bounded progress for direct locked-wheel downloads used by local image fallback builds.
- Modify `src/amd_ai/image/build.py`: send local Buildx output through the observed command path and request plain progress.
- Modify `src/amd_ai/project/build.py`: request plain BuildKit output; its existing runner call becomes live through classification.
- Modify `src/amd_ai/cli.py`: add `--verbose`/`--quiet`, construct one reporter and runner, and remove duplicate result printing.
- Modify `src/amd_ai/__init__.py`: bump only the toolkit version to `0.2.3`.
- Create `tests/unit/installer/test_progress.py`: security, modes, formatting, stage timing, heartbeat, failure evidence, and log-permission tests.
- Modify focused tests under `tests/unit`, `tests/cli`, and `tests/test_version.py` for each integration boundary.
- Modify `README.md` and `docs/install.md`; create `docs/releases/v0.2.3.md` for the user workflow and release invariants.

## Task 1: Secure output and log primitives

**Files:**

- Create: `src/amd_ai/installer/progress.py`
- Modify: `src/amd_ai/installer/state.py:148`
- Modify: `src/amd_ai/runner.py:8-15`
- Create: `tests/unit/installer/test_progress.py`
- Modify: `tests/unit/installer/test_state.py`
- Modify: `tests/unit/test_runner.py`

- [x] **Step 1: Write failing project-key, redaction, and private-log tests**

Add these tests with the exact public names and assertions:

```python
def test_project_identity_key_is_shared_by_state_and_log_names(
    tmp_path: Path,
) -> None:
    project = tmp_path / "unsafe name" / "video lab"
    key = project_identity_key(project)

    assert re.fullmatch(r"[A-Za-z0-9._-]+-[0-9a-f]{12}", key)
    assert project_state_path(project, tmp_path / "install-state.json").stem == key


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


def test_session_log_is_private_and_rejects_collisions(tmp_path: Path) -> None:
    project = tmp_path / "video-lab"
    log = SessionLog.open(
        project_dir=project,
        log_root=tmp_path / "state" / "logs",
        wall_clock=lambda: datetime(
            2026, 7, 10, 14, 25, 33, tzinfo=UTC
        ),
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
            wall_clock=lambda: datetime(
                2026, 7, 10, 14, 25, 33, tzinfo=UTC
            ),
            process_id=18421,
        )


def test_session_log_rejects_symlinked_control_directory(tmp_path: Path) -> None:
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
```

Extend `tests/unit/test_runner.py` to prove existing four-positional-argument
`CommandResult` construction still works after adding truncation metadata:

```python
def test_command_result_truncation_defaults_are_backward_compatible() -> None:
    result = CommandResult(("true",), 0, "", "")
    assert result.stdout_truncated is False
    assert result.stderr_truncated is False
```

- [x] **Step 2: Run the new tests and verify RED**

Run:

```bash
PYTHONPATH=src /app/imgMaker/.venv/bin/python -m pytest \
  tests/unit/installer/test_progress.py \
  tests/unit/installer/test_state.py \
  tests/unit/test_runner.py -q
```

Expected: collection fails because `progress.py`, `project_identity_key`, and
the `CommandResult` truncation fields do not exist.

- [x] **Step 3: Extract the reusable project identity key**

In `src/amd_ai/installer/state.py`, add and reuse this function without changing
the generated state filename:

```python
def project_identity_key(project_dir: Path) -> str:
    project = Path(project_dir).resolve(strict=False)
    readable = re.sub(r"[^A-Za-z0-9._-]+", "-", project.name)
    readable = readable.strip(".-_")[:48] or "project"
    identity = hashlib.sha256(str(project).encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{identity}"


def project_state_path(project_dir: Path, legacy_path: Path) -> Path:
    legacy = Path(legacy_path).resolve(strict=False)
    return legacy.parent / "projects" / f"{project_identity_key(project_dir)}.json"
```

- [x] **Step 4: Add command output types without changing `Runner.run`**

In `src/amd_ai/runner.py`, keep the `Runner.run(args, *, check=True,
input_text=None)` protocol unchanged. Extend the result and declare the observer
boundary used by later tasks:

```python
class CommandStream(StrEnum):
    STDOUT = "stdout"
    STDERR = "stderr"


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class CommandObserver(Protocol):
    def command_started(
        self,
        args: tuple[str, ...],
        *,
        live: bool,
        environment: Mapping[str, str] | None = None,
    ) -> None: ...

    def command_output(self, stream: CommandStream, text: str) -> str: ...

    def command_finished(self, result: CommandResult, *, live: bool) -> None: ...
```

The observer returns sanitized text from `command_output`; the runner must use
that returned value for its bounded in-memory tail.

- [x] **Step 5: Implement sanitization and atomic private logs**

Create `src/amd_ai/installer/progress.py` with `ProgressError`,
`ProgressMode(DEFAULT, VERBOSE, QUIET)`, `sanitize_output`, and `SessionLog`.
Use compiled CSI/OSC/control, URL-userinfo, sensitive-assignment,
authorization, and credential-flag patterns. Apply transformations in this
order:

```python
def sanitize_output(value: str, *, secret_values: Iterable[str] = ()) -> str:
    rendered = value.replace("\r\n", "\n").replace("\r", "\n")
    rendered = OSC_PATTERN.sub("", rendered)
    rendered = CSI_PATTERN.sub("", rendered)
    rendered = CONTROL_PATTERN.sub("", rendered)
    for secret in sorted({item for item in secret_values if item}, key=len, reverse=True):
        rendered = rendered.replace(secret, "<redacted>")
    rendered = URL_USERINFO_PATTERN.sub(r"\g<scheme><redacted>@", rendered)
    rendered = SENSITIVE_ASSIGNMENT_PATTERN.sub(r"\g<name>=<redacted>", rendered)
    rendered = AUTHORIZATION_PATTERN.sub(r"\g<label><redacted>", rendered)
    return CREDENTIAL_FLAG_PATTERN.sub(r"\g<flag><redacted>", rendered)
```

Define the production root lazily so fixture HOME values are honored:

```python
def default_log_root() -> Path:
    return (
        Path.home()
        / ".local/state/strix-halo-rocm-toolkit/logs"
    )
```

`SessionLog.open` accepts `wall_clock: Callable[[], datetime]` and must create
`<log_root>/<project_identity_key>/install-<UTC>-<pid>.log`, reject symlinked
controlled directories, tighten owned controlled directories to `0700`, and
open the file with `O_CREAT|O_EXCL|O_WRONLY|O_NOFOLLOW` and mode `0600`.
Normalize relative injected roots against `Path.cwd()` without calling
`Path.resolve()` on controlled log components; resolving first would hide a
symlink that must be rejected.
`write` must hold a lock, sanitize before encoding, write one UTC timestamped
line per logical line, and never retain raw text. `close` flushes, `fsync`s, and
closes exactly once.

- [x] **Step 6: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass and the existing
state filenames are byte-for-byte compatible with their previous algorithm.

- [x] **Step 7: Commit secure output primitives**

```bash
git add src/amd_ai/installer/progress.py src/amd_ai/installer/state.py \
  src/amd_ai/runner.py tests/unit/installer/test_progress.py \
  tests/unit/installer/test_state.py tests/unit/test_runner.py
git commit -m "feat: add secure installer progress logs"
```

## Task 2: Structured reporter, output modes, and heartbeat

**Files:**

- Modify: `src/amd_ai/installer/progress.py`
- Modify: `src/amd_ai/installer/prompts.py:17-20`
- Modify: `tests/unit/installer/test_progress.py`
- Modify: `tests/unit/installer/test_prompts.py`

- [x] **Step 1: Write failing event-mode and heartbeat tests**

Add `StagePosition`/`SessionPlan` imports and these complete test helpers before
the event tests:

```python
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
```

Then add these tests:

```python
def test_default_reporter_prints_plan_stage_and_elapsed_time(tmp_path: Path) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    clock = FakeClock(wall=datetime(2026, 7, 10, tzinfo=UTC), monotonic=10.0)
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
        (ProgressMode.VERBOSE, ("START", "PASS", "COMMAND", "DEBUG"), ()),
        (ProgressMode.QUIET, ("SUMMARY",), ("PLAN", "START", "WAIT", "PASS")),
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

    terminal = stdout.getvalue() + stderr.getvalue()
    for token in present:
        assert token in terminal
    for token in absent:
        assert token not in terminal
    log = log_path.read_text(encoding="utf-8")
    for token in ("PLAN", "START", "COMMAND", "DEBUG", "PASS", "SUMMARY"):
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


def test_failure_summary_uses_bounded_tail_and_recovery_paths(tmp_path: Path) -> None:
    reporter, stdout, stderr = opened_reporter(tmp_path, ProgressMode.DEFAULT)
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

    failure = stderr.getvalue()
    assert "FAIL     [6/8]" in failure
    assert "CAUSE    PROJECT_INIT failed: uv download failed" in failure
    assert "line-0" not in failure
    assert "line-79" in failure
    assert failure.count("line-") <= 40
    assert "STATE" in failure and "RESUME" in failure and "LOG" in failure
```

Add a companion test using `message="failure: " + "x" * (300 * 1024)` and
assert the UTF-8 bytes between the `CAUSE` line and the following `STATE` line
are at most 16 KiB while the `failure:` prefix remains present.

Call `installation_finished` with `ProgressOutcome.FAILURE` and `position=None`
in a separate test. Assert stderr starts with `FAIL     installer`, omits a
fabricated `[i/N]`, and still contains `CAUSE`, `STATE`, `RESUME`, and the real
`LOG` path.

Add this end-to-end reporter redaction test:

```python
def test_reporter_redacts_environment_and_url_secrets_from_every_sink(
    tmp_path: Path,
) -> None:
    reporter, stdout, stderr = opened_reporter(tmp_path, ProgressMode.VERBOSE)
    position = StagePosition(InstallStage.PROJECT_INIT, 6, 8)
    reporter.stage_started(position)
    reporter.command_started(
        (
            "uv", "pip", "compile",
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
    rendered = (
        stdout.getvalue()
        + stderr.getvalue()
        + log_path.read_text(encoding="utf-8")
    )
    assert "<redacted>" in rendered
    assert "hf_private_value" not in rendered
    assert "ghp_private" not in rendered
```

Add this short real-thread lifecycle test:

```python
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
```

- [x] **Step 2: Run progress and prompt tests and verify RED**

```bash
PYTHONPATH=src /app/imgMaker/.venv/bin/python -m pytest \
  tests/unit/installer/test_progress.py \
  tests/unit/installer/test_prompts.py -q
```

Expected: failures report missing reporter, plan, heartbeat, and event-prefix
APIs.

- [x] **Step 3: Implement the reporter contract and stable labels**

Add these immutable inputs and protocol to `progress.py`:

```python
STAGE_LABELS: Mapping[InstallStage, str] = MappingProxyType({
    InstallStage.BOOTSTRAP: "安装用户运行时",
    InstallStage.HOST_PREFLIGHT: "检查宿主",
    InstallStage.HOST_PLAN: "生成宿主变更计划",
    InstallStage.HOST_CONFIRM: "确认宿主变更",
    InstallStage.HOST_APPLY: "应用宿主变更",
    InstallStage.REBOOT_PENDING: "检查重启状态",
    InstallStage.HOST_VERIFY: "验证重启后的宿主",
    InstallStage.CONTAINER_HOST_CHECK: "验证容器宿主",
    InstallStage.RELEASE_RESOLVE: "解析 stable release",
    InstallStage.IMAGE_PULL_OR_BUILD: "获取或构建 stable 镜像",
    InstallStage.IMAGE_VERIFY: "验证 gfx1151 PyTorch GPU runtime",
    InstallStage.PROJECT_INIT: "创建项目镜像与 Python 依赖",
    InstallStage.PROJECT_VERIFY: "验证项目",
    InstallStage.COMPLETE: "完成安装",
})


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


class ProgressOutcome(StrEnum):
    SUCCESS = "success"
    ACTION = "action"
    BLOCKED = "blocked"
    FAILURE = "failure"


class ProgressReporter(Protocol):
    log_path: Path | None
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
        self, *, outcome: ProgressOutcome, exit_code: int, message: str,
        state_path: Path, project_dir: Path | None,
        position: StagePosition | None,
    ) -> None: ...
    def close(self) -> None: ...
```

`InstallerProgress` implements this protocol plus `CommandObserver`. It must
format durations from a monotonic clock, reset the stage tail at `START`, retain
at most 40 lines and 16 KiB for failure display, and retain no unsanitized
value. Default `START`/`PASS` shows the Chinese label; verbose mode appends the
enum in parentheses; failure output always appends the enum.

Build failure evidence from the sanitized first exception line plus the newest
child-tail lines. Apply the 40-child-line limit, then cap the complete rendered
cause at 16 KiB so an action exception that already embeds a 256-KiB
`CommandResult` tail cannot bypass terminal bounds. Keep the exception prefix
and newest evidence when trimming.

At `command_started`, extract nonempty values from sensitive keys in the
effective process environment, sensitive argv assignments, URL user
information, authorization arguments, and values following credential flags
into a per-command secret set. Render the verbose command with `shlex.join`,
then call `sanitize_output(..., secret_values=secret_set)`. Every subsequent
`command_output` uses the same set and returns only the sanitized text;
`command_finished` clears it after recording redacted return-code and byte count
diagnostics. Never render or persist the complete environment.

When its `log_root` constructor argument is omitted, `InstallerProgress` calls
`default_log_root()` when `open_session` runs. It must not resolve HOME or create
directories at module import time.

`ProgressOutcome.SUCCESS` emits `SUMMARY`; `ACTION` emits `ACTION` plus recovery
paths; `BLOCKED` emits `BLOCKED` plus cause and recovery paths; `FAILURE` emits
`FAIL` plus cause and recovery paths. Exit code alone must not be used to guess
whether a stage was blocked or failed.

When rendering the plan, map internal state source `project` to user-facing
`per-project`; retain `legacy` and `explicit` unchanged. Do not change the
persisted selection source or state path algorithm.

- [x] **Step 4: Implement heartbeat lifecycle and output filtering**

Add `HeartbeatSchedule` and a daemon worker owned by `InstallerProgress`.
`stage_started` starts one worker; every structured or child line calls
`activity`; `stage_passed`, `installation_finished`, and `close` set the stop
event and join the worker before emitting the terminal event. Quiet mode does
not start the worker. `pause_heartbeat` suppresses due events without stopping
the worker; `resume_heartbeat` resets activity to the current monotonic time.
The prompt compatibility adapter implements both as no-ops.

Use these terminal visibility rules exactly:

```python
VERBOSE_ONLY = frozenset({"COMMAND", "DEBUG"})
QUIET_VISIBLE = frozenset({
    "WARN", "ACTION", "BLOCKED", "FAIL", "CAUSE", "STATE", "RESUME",
    "LOG", "SUMMARY",
})
STDERR_EVENTS = frozenset({
    "WARN", "ACTION", "BLOCKED", "FAIL", "CAUSE", "STATE", "RESUME",
})
```

The initial `LOG` belongs to the plan and is hidden in quiet mode; the final or
failure `LOG` is emitted with `force_terminal=True`. Every event is still
written to the private log.

`command_output` writes its sanitized return value to the session log as
`CHILD`, sends stdout records to installer stdout and stderr records to installer
stderr in default/verbose modes, and suppresses both terminal writes in quiet
mode. It emits complete newline-delimited records only and never writes cursor
movement or carriage-return updates.

Use one reporter lock to serialize terminal writes, log writes, stage-tail
updates, secret-set access, and heartbeat activity across both reader threads
and the heartbeat worker. Do not hold that lock while joining the worker or
child reader threads.

- [x] **Step 5: Add the prompt compatibility adapter and prefixes**

Implement `PromptProgressAdapter(status_fn)` with the same workflow-facing
methods, no log, no worker, and no command output. It formats stage events and
delegates them to the existing `status_fn`, preserving direct workflow tests
and non-CLI callers.

Change `STATUS_PREFIXES` in `prompts.py` to the exact event vocabulary:

```python
STATUS_PREFIXES = frozenset({
    "PLAN", "LOG", "SKIP", "START", "DETAIL", "WAIT", "PASS", "INFO",
    "WARN", "ACTION", "BLOCKED", "FAIL", "CAUSE", "STATE", "RESUME",
    "SUMMARY", "COMMAND", "DEBUG",
})
```

Update `test_status_renderer_accepts_only_approved_prefixes` to accept `START`,
`WAIT`, `FAIL`, and `SUMMARY`, while still rejecting `UNKNOWN`.

- [x] **Step 6: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all progress and prompt tests pass,
including no late heartbeat and complete quiet-mode logs.

- [x] **Step 7: Commit structured progress reporting**

```bash
git add src/amd_ai/installer/progress.py src/amd_ai/installer/prompts.py \
  tests/unit/installer/test_progress.py tests/unit/installer/test_prompts.py
git commit -m "feat: report installer stages and heartbeats"
```

## Task 3: Live subprocess tee with bounded evidence

**Files:**

- Modify: `src/amd_ai/runner.py`
- Modify: `tests/unit/test_runner.py`
- Modify: `tests/unit/installer/test_progress.py`

- [x] **Step 1: Write failing classification and incremental-output tests**

Add exact classifier expectations:

```python
@pytest.mark.parametrize(
    "argv",
    [
        ["docker", "pull", "ghcr.io/example/image@sha256:" + "a" * 64],
        ["sudo", "-n", "docker", "--config", "/tmp/empty", "pull", "image"],
        ["docker", "buildx", "build", "--progress=plain", "."],
        ["apt-get", "update"],
        ["apt", "install", "docker-ce"],
        ["dpkg", "--install", "package.deb"],
        ["uv", "pip", "compile", "requirements.in"],
        ["python3.12", "-m", "pip", "install", "package"],
        [
            "docker", "run", "--entrypoint", "/usr/local/bin/uv", "base",
            "pip", "compile", "requirements.in",
        ],
    ],
)
def test_live_command_policy_accepts_only_long_mutations(argv: list[str]) -> None:
    assert is_live_command(argv) is True


@pytest.mark.parametrize(
    "argv",
    [
        ["docker", "image", "inspect", "image"],
        ["docker", "buildx", "imagetools", "inspect", "--raw", "image"],
        ["docker", "run", "--rm", "image", "container-check", "--json", "-"],
        ["pip", "list", "--format=json"],
        ["sha256sum", "artifact"],
    ],
)
def test_live_command_policy_keeps_protocol_and_probe_commands_captured(
    argv: list[str],
) -> None:
    assert is_live_command(argv) is False
```

Create a recording observer whose `command_output` sets an event and returns
`sanitize_output(text)`:

```python
class RecordingObserver:
    def __init__(self) -> None:
        self.lines: list[tuple[CommandStream, str]] = []
        self.first_output = threading.Event()
        self.started: list[tuple[tuple[str, ...], bool]] = []
        self.finished: list[tuple[CommandResult, bool]] = []

    def command_started(
        self,
        args: tuple[str, ...],
        *,
        live: bool,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        del environment
        self.started.append((args, live))

    def command_output(self, stream: CommandStream, text: str) -> str:
        rendered = sanitize_output(text)
        self.lines.append((stream, rendered))
        self.first_output.set()
        return rendered

    def command_finished(self, result: CommandResult, *, live: bool) -> None:
        self.finished.append((result, live))
```

Run this subprocess in a background thread:

```python
[
    "python3.12", "-c",
    "import sys,time; print('first', flush=True); "
    "print('problem', file=sys.stderr, flush=True); time.sleep(0.2); "
    "print('last', flush=True)",
]
```

Assert the observer receives `first` before the runner thread completes,
stdout and stderr remain ordered within their own streams, and the result
contains the same sanitized lines. Add a 400-KiB command-output test asserting
both result text lengths are at most `256 * 1024` and both truncation flags are
true. Add a 2-MiB output with no newline and assert observer fragments never
exceed 64 KiB. Add an unterminated-final-fragment test.

Add an observer that raises `OSError("log full")` on its first line while the
child writes 1 MiB to both streams. Assert the runner drains and exits without
deadlock, then raises `CommandObservationError` chained from `OSError`; no raw
post-failure output is retained.

- [x] **Step 2: Run runner tests and verify RED**

```bash
PYTHONPATH=src /app/imgMaker/.venv/bin/python -m pytest \
  tests/unit/test_runner.py \
  tests/unit/installer/test_progress.py -q
```

Expected: failures report missing `is_live_command`, observed execution, and
bounded-tail behavior.

- [x] **Step 3: Implement explicit long-command classification**

Add `is_live_command(args: Sequence[str]) -> bool`. Strip only audited wrappers
(`sudo` options, `--`, and `env` assignments), identify Docker global options
that consume a value, and classify only the command families listed in the
approved design. For `docker run`, treat an exact uv/pip entrypoint or an exact
post-image `uv`, `pip`, `pip3`, or `python -m pip` command as live. Do not use
substring matching on arbitrary paths or arguments.

- [x] **Step 4: Implement concurrent observed execution**

Add `CommandObservationError(RuntimeError)` and this reusable entry point while
leaving `Runner.run` unchanged:

```python
def run_observed_command(
    args: Sequence[str],
    *,
    observer: CommandObserver | None,
    check: bool = True,
    input_text: str | None = None,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
    observe_stdout: bool = True,
    observe_stderr: bool = True,
    stdout_limit: int = 256 * 1024,
    stderr_limit: int = 256 * 1024,
) -> CommandResult:
    return _execute_observed_command(
        tuple(args),
        observer=observer,
        check=check,
        input_text=input_text,
        cwd=cwd,
        environment=environment,
        observe_stdout=observe_stdout,
        observe_stderr=observe_stderr,
        stdout_limit=stdout_limit,
        stderr_limit=stderr_limit,
    )
```

Implement `_execute_observed_command` in the same module as the private process
engine; it has the same keyword parameters after the normalized command tuple.
When `observer is None` and both observe flags are true, retain inherited
stdout/stderr behavior for callers that explicitly use this function for a live
build. If either observe flag is false, still use pipes so protocol stdout can
be captured even without an observer. Start `Popen` with separate pipes, drain
each stream concurrently, split both newline and carriage-return records, emit
a final fragment at EOF, and append observer-returned sanitized text to bounded
buffers. A stream with no observer is captured but never written to a terminal.
Use one incremental UTF-8 decoder per stream with `errors="replace"` so byte
boundaries and invalid child bytes cannot crash reader threads.
If an unfinished record reaches 64 KiB without `\n` or `\r`, emit that chunk
immediately and retain only the remainder, preventing an unbounded partial-line
buffer. Write `input_text`, close stdin, wait for both readers, then create
`CommandResult`.
An unobserved protocol stdout stream bypasses line normalization and is decoded
and buffered exactly, so strict JSON parsing sees the child's original bytes;
its 1-MiB bound and truncation flag still apply.
For redaction, `effective_environment` is a read-only copy of the supplied
mapping or `os.environ` when no mapping was supplied; pass only that mapping to
`command_started` and never include it in diagnostics.
Call `observer.command_started(command, live=True,
environment=effective_environment)` before process creation and
`observer.command_finished(result, live=True)` after readers join and before any
`CommandError` is raised. This makes direct local-image and privileged protocol
calls produce the same command lifecycle as `SubprocessRunner`.
If an observer callback fails, record only the first exception, continue
draining and discard subsequent raw output to prevent pipe deadlock, join the
process/readers, then raise `CommandObservationError` from that exception.
On `KeyboardInterrupt`, terminate the child, drain and join readers, then
re-raise. On `check=True` and nonzero status, raise `CommandError(result)`.

- [x] **Step 5: Route classified `SubprocessRunner` commands through the tee**

Give `SubprocessRunner` this constructor only; do not alter its `run` signature:

```python
def __init__(
    self,
    *,
    observer: CommandObserver | None = None,
    live_policy: Callable[[Sequence[str]], bool] = is_live_command,
) -> None:
    self.observer = observer
    self.live_policy = live_policy
```

A classified command with an observer delegates the complete lifecycle to
`run_observed_command`; an unclassified command keeps the current captured
`subprocess.run` path and calls `command_started(..., live=False,
environment=os.environ)` and
`command_finished(..., live=False)` itself. Captured completion exposes only
return code and byte-count diagnostics; it does not mirror parseable
stdout/stderr. A runner without an observer keeps all current capture behavior.

- [x] **Step 6: Run runner tests and verify GREEN**

Run the command from Step 2. Expected: all tests pass, output arrives before
process completion, and no protocol command is streamed.

- [x] **Step 7: Commit the observed runner**

```bash
git add src/amd_ai/runner.py tests/unit/test_runner.py \
  tests/unit/installer/test_progress.py
git commit -m "feat: stream long installer commands"
```

## Task 4: Integrate progress with resumable workflow semantics

**Files:**

- Modify: `src/amd_ai/installer/workflow.py:82-295,662-836`
- Modify: `tests/unit/installer/fakes.py`
- Modify: `tests/unit/installer/test_workflow.py`

- [x] **Step 1: Write failing workflow event-order tests**

Extend `installer_workflow` with an optional `progress` argument. Add a helper
that creates `InstallerProgress` with `tmp_path / "logs"` and `StringIO` sinks;
the reporter unit tests already cover exact elapsed time with a fake clock:

```python
def workflow_progress(
    tmp_path: Path,
    *,
    mode: ProgressMode = ProgressMode.DEFAULT,
) -> tuple[InstallerProgress, io.StringIO, io.StringIO]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    progress = InstallerProgress(
        mode=mode,
        stdout=stdout,
        stderr=stderr,
        log_root=tmp_path / "logs",
        process_id=23,
    )
    return progress, stdout, stderr
```

Add tests that assert:

```python
def test_resume_reports_plan_skips_start_disk_detail_and_checkpointed_pass(
    tmp_path: Path,
) -> None:
    first_actions = FakeInstallerActions.stop_after(InstallStage.RELEASE_RESOLVE)
    assert installer_workflow(tmp_path, actions=first_actions).run().exit_code == 1
    progress, stdout, _ = workflow_progress(tmp_path)

    result = installer_workflow(
        tmp_path,
        actions=FakeInstallerActions.healthy(),
        progress=progress,
    ).run()

    assert result.exit_code == 0
    output = stdout.getvalue()
    assert "PLAN     共 8 个阶段，从 IMAGE_PULL_OR_BUILD 继续" in output
    assert "SKIP     [1/8] 安装用户运行时：已有可信检查点" in output
    assert output.index("START    [4/8]") < output.index("DETAIL   缺失层=10.0 GiB")
    assert output.index("DETAIL   缺失层=10.0 GiB") < output.index("PASS     [4/8]")
    persisted = load_state(tmp_path / "install-state.json")
    assert persisted is not None
    assert InstallStage.IMAGE_PULL_OR_BUILD.value in persisted.completed_stage_input_digests


def test_failed_stage_reports_state_resume_and_log_without_checkpoint(
    tmp_path: Path,
) -> None:
    actions = FakeInstallerActions.healthy()
    actions.failures[InstallStage.PROJECT_INIT] = RuntimeError("uv download failed")
    progress, _, stderr = workflow_progress(tmp_path)

    result = installer_workflow(tmp_path, actions=actions, progress=progress).run()

    assert result.exit_code == 2
    assert "FAIL     [6/8]" in stderr.getvalue()
    assert "CAUSE    PROJECT_INIT failed: uv download failed" in stderr.getvalue()
    assert "STATE" in stderr.getvalue()
    assert "RESUME" in stderr.getvalue()
    state = load_state(tmp_path / "install-state.json")
    assert state is not None
    assert InstallStage.PROJECT_INIT.value not in state.completed_stage_input_digests


def test_complete_rerun_reports_all_skips_and_summary(tmp_path: Path) -> None:
    assert installer_workflow(
        tmp_path, actions=FakeInstallerActions.healthy()
    ).run().exit_code == 0
    progress, stdout, _ = workflow_progress(tmp_path)

    result = installer_workflow(
        tmp_path, actions=FakeInstallerActions.healthy(), progress=progress
    ).run()

    assert result.exit_code == 0
    assert stdout.getvalue().count("SKIP") == len(CONTAINER_STAGE_ORDER)
    assert "SUMMARY" in stdout.getvalue()
```

Add secure-log failure coverage:

```python
def test_log_creation_failure_runs_no_stage(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    log_root = tmp_path / "unsafe-logs"
    log_root.symlink_to(target, target_is_directory=True)
    progress = InstallerProgress(
        mode=ProgressMode.DEFAULT,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        log_root=log_root,
    )
    actions = FakeInstallerActions.healthy()

    result = installer_workflow(
        tmp_path, actions=actions, progress=progress
    ).run()

    assert result.exit_code == 2
    assert "log" in result.message.lower()
    assert actions.calls == []
```

For a new state, assert the session plan says `stable release=待解析`. After
`RELEASE_RESOLVE` applies the trusted manifest output, emit and test a `DETAIL`
line containing release ID `0.2.0` before that stage's checkpointed `PASS`.
Verbose `DEBUG` lines include both exact image references and their manifest
digests.

Add disk-shortage coverage asserting `START`, `DETAIL`, then `FAIL`, with no
action call. Add action-required coverage asserting checkpointed `PASS` followed
by `ACTION`, `STATE`, `RESUME`, and `LOG`. Preserve every existing strict
digest, mode, project, and reboot test. Run equivalent workflows with default
and quiet reporters using separate state files but the same project/options,
then assert their `completed_stage_input_digests` mappings are identical; no
progress mode, log path, timestamp, or elapsed duration may enter stage inputs
or state schema.

Add a container `dry_run=True` test asserting every existing dry-run `PASS` or
`ACTION` line now includes its `[i/8]` position, no `START` heartbeat worker is
created, final `SUMMARY` says no stages were persisted, and the state file is
still absent. Keep the current dry-run mutating-stage set unchanged.

For the same-boot `REBOOT_PENDING` path, change the action message to name the
exact manual command without executing it. Assert output contains
`RESUME   sudo reboot；重启后重新执行同一条 install 命令` and host apply is not
replayed. Other action-required results use their existing message plus the
generic same-command resume instruction.

Inject progress into one no-reboot full-mode fixture and assert its plan says
13 stages, `HOST_PREFLIGHT` is `[2/13]`, `HOST_VERIFY` is `[7/13]`, and
`COMPLETE` is `[13/13]`. This guards against accidentally numbering stages by
global enum position instead of the selected workflow order.

Add this focused fake constructor rather than mutating private test state at the
call site:

```python
@classmethod
def full_no_reboot(cls) -> FakeInstallerActions:
    fake = cls.healthy()
    fake.host_plan_result = fake._make_host_plan(reboot_required=False)
    return fake
```

```python
def test_progress_mode_does_not_change_checkpoint_digests(tmp_path: Path) -> None:
    project = tmp_path / "project"
    base = workflow_options(tmp_path, project_dir=project)
    default_progress, _, _ = workflow_progress(tmp_path / "default")
    quiet_progress, _, _ = workflow_progress(
        tmp_path / "quiet", mode=ProgressMode.QUIET
    )
    default_result = installer_workflow(
        tmp_path,
        actions=FakeInstallerActions.healthy(),
        options=replace(base, state_path=tmp_path / "default-state.json"),
        progress=default_progress,
    ).run()
    quiet_result = installer_workflow(
        tmp_path,
        actions=FakeInstallerActions.healthy(),
        options=replace(base, state_path=tmp_path / "quiet-state.json"),
        progress=quiet_progress,
    ).run()

    assert default_result.state is not None
    assert quiet_result.state is not None
    assert (
        default_result.state.completed_stage_input_digests
        == quiet_result.state.completed_stage_input_digests
    )
```

- [x] **Step 2: Write the failing `v0.2.2` container adoption test**

Complete a container workflow with installer version `0.2.2` and revision
`d * 40`, then rerun the same state as version `0.2.3` and revision `e * 40`.
Assert no actions replay, only the `BOOTSTRAP` digest changes, all other stage
digests remain identical, and state schema stays 2. Add a rejection case for a
container state whose old `BOOTSTRAP` digest has been altered.

```python
def test_compatible_patch_installer_adopts_container_state(tmp_path: Path) -> None:
    first = installer_workflow(
        tmp_path,
        actions=FakeInstallerActions.healthy(),
        installer_version="0.2.2",
        installer_source_revision="d" * 40,
    ).run()
    assert first.state is not None
    old_digests = dict(first.state.completed_stage_input_digests)
    resumed_actions = FakeInstallerActions.healthy()

    resumed = installer_workflow(
        tmp_path,
        actions=resumed_actions,
        installer_version="0.2.3",
        installer_source_revision="e" * 40,
    ).run()

    assert resumed.exit_code == 0
    assert resumed_actions.calls == []
    assert resumed.state is not None and resumed.state.schema_version == 2
    changed = {
        name
        for name, digest in resumed.state.completed_stage_input_digests.items()
        if old_digests[name] != digest
    }
    assert changed == {InstallStage.BOOTSTRAP.value}
```

For rejection, use a separate test with:

```python
completed = dict(first.state.completed_stage_input_digests)
completed[InstallStage.BOOTSTRAP.value] = "0" * 64
save_state(
    tmp_path / "install-state.json",
    replace(first.state, completed_stage_input_digests=completed),
)
actions = FakeInstallerActions.healthy()
rejected = installer_workflow(
    tmp_path,
    actions=actions,
    installer_version="0.2.3",
    installer_source_revision="e" * 40,
).run()
assert rejected.exit_code == 2
assert "inputs changed" in rejected.message
assert actions.calls == []
```

- [x] **Step 3: Run workflow tests and verify RED**

```bash
PYTHONPATH=src /app/imgMaker/.venv/bin/python -m pytest \
  tests/unit/installer/test_workflow.py -q
```

Expected: failures show no injected progress contract, no structured stage
events, and no compatible container adoption.

- [x] **Step 4: Open the session and emit the plan before stage execution**

Add `progress: ProgressReporter | None = None` to `InstallerWorkflow.__init__`.
Default to `PromptProgressAdapter(prompts.status)` for direct callers. After
project normalization, call `open_session` before any persisted or host
mutation. Record the selected state source in `_select_state_path`; after state
load, compatible adoption, and transition validation, emit `SessionPlan` using
the selected workflow order and first incomplete checkpoint. Compute
`first_incomplete` with `next((stage for stage in order if stage.value not in
state.completed_stage_input_digests), None)` so an already-complete state is
reported as complete instead of appearing to resume at `COMPLETE`.

Catch `ProgressError` as an installer failure. If secure log creation fails,
emit `FAIL installer` to stderr without a `LOG` line, execute no stage, and
return exit code 2. Both concrete reporters must make `close` and final rendering
safe when `open_session` did not complete.

Wrap the existing result construction so every normal return and caught error
passes once through:

```python
self.progress.installation_finished(
    outcome=result.progress_outcome,
    exit_code=result.exit_code,
    message=result.message,
    state_path=self.options.state_path,
    project_dir=self.options.project_dir,
    position=self._current_position,
)
```

Always close the reporter in `finally`. Remove no error class and preserve exit
codes. Extend `WorkflowResult` with
`progress_outcome: ProgressOutcome = ProgressOutcome.FAILURE`; set it explicitly
to `SUCCESS` for exit 0, `ACTION` for interruption/reboot/operator action,
`BLOCKED` for a blocked `StageResult`, and `FAILURE` for exceptions, invalid
state, transition errors, and disk shortage. The default keeps existing test
and external construction source-compatible.

- [x] **Step 5: Bracket stages without changing checkpoint authority**

In `_run_stages`, construct `StagePosition(stage, index + 1, len(order))` and
call `stage_candidate` before input validation. On a trusted checkpoint call
`stage_skipped` and continue. For an incomplete stage call `stage_started`,
perform disk checks, dispatch and apply output, save the checkpoint, then call
`stage_passed`. Do not emit `PASS` before `save_state` returns.

In `_run_dry`, construct the same `StagePosition` values and include them in the
existing `PASS`/`ACTION` dry-run statuses without calling `stage_started` or
persisting checkpoints.

Call `pause_heartbeat` immediately before interactive `confirm_exact`,
`confirm_yes_no`, and `choose_image_fallback` prompts; call `resume_heartbeat`
in `finally`. This prevents background `WAIT` lines from overwriting an active
input prompt while preserving heartbeats for sudo password waits and commands.

Replace `_disk_shortage` with an internal frozen `DiskRequirement` containing
`operation`, `source`, `estimate`, and `required_bytes`. Emit one `DETAIL` with
payload, required, available, location, and source before checking shortage.
Use one-decimal binary GiB in default mode and append exact bytes through
`progress.debug` in verbose mode. Emit a second detail when interactive pull
fallback switches to a local build estimate.

Render image source `pull` as `公开 GHCR`, image source `build` as `本地源码构建`,
and project initialization as `项目文件系统`. For pulls label payload
`缺失层`; for local image builds label it `构建估算`; for project initialization
label it `项目数据`.

- [x] **Step 6: Extend only compatible container installer adoption**

Keep the existing full-mode boundary. Change `_can_adopt_installer_update` to:

```python
if state.mode is InstallMode.FULL:
    start = FULL_STAGE_ORDER.index(InstallStage.HOST_VERIFY)
    return state.current_stage in FULL_STAGE_ORDER[start:]
if state.mode is InstallMode.CONTAINER:
    return (
        InstallStage.BOOTSTRAP.value in state.completed_stage_input_digests
        and state.current_stage in CONTAINER_STAGE_ORDER[1:]
    )
return False
```

Retain same-series validation, old-input reconstruction, and exact old digest
verification. Rewrite only installer metadata and the `BOOTSTRAP` digest.

- [x] **Step 7: Run workflow tests and verify GREEN**

Run the command from Step 3. Expected: all old and new workflow tests pass;
failure/action output is emitted once and trusted checkpoints remain strict.

- [x] **Step 8: Commit workflow progress semantics**

```bash
git add src/amd_ai/installer/workflow.py tests/unit/installer/fakes.py \
  tests/unit/installer/test_workflow.py
git commit -m "feat: expose resumable installer progress"
```

## Task 5: Stream Docker pulls, BuildKit, uv, and pip

**Files:**

- Modify: `src/amd_ai/image/publish.py:177-309`
- Modify: `src/amd_ai/image/lock.py:38-73`
- Modify: `src/amd_ai/image/build.py:58-76,300-410,843-856`
- Modify: `src/amd_ai/project/build.py:102-139,229`
- Modify: `src/amd_ai/installer/actions.py:116-155,430-590`
- Modify: `tests/unit/image/test_publish.py`
- Modify: `tests/unit/image/test_lock.py`
- Modify: `tests/unit/image/test_build.py`
- Modify: `tests/unit/project/test_build.py`
- Modify: `tests/unit/project/test_dependencies.py`
- Modify: `tests/unit/installer/test_actions.py`

- [x] **Step 1: Write failing registry and BuildKit progress tests**

Replace the authless-pull subprocess monkeypatch with a recording `Runner` and
assert the exact command still contains the empty temporary Docker config.
Add a test where `DockerPublishRegistry(runner=recording_runner)` calls `pull`
through that runner while `inspect` returns captured JSON.

In image and project build tests assert every Buildx build argv contains exactly
one `--progress=plain`. Add a fake `CommandObserver` test for `_run_live` that
runs a harmless Python command in a temporary cwd and receives output through
`command_output`.

In `tests/unit/image/test_lock.py`, use a fake response with `Content-Length`
and deterministic small chunks. Monkeypatch
`PROGRESS_INTERVAL_BYTES` from its production 64 MiB to 4 bytes, pass a callback,
and assert it receives `(0, total)`, each threshold crossed, and the exact final
byte count. Add an unknown-length response and assert the callback receives
`total=None`. A cache hit with a matching digest emits no download callback.

In dependency tests, construct the generated Docker uv command and assert
`is_live_command(argv)` is true; retain the existing temporary lock-file and
protected-Torch validation assertions.

- [x] **Step 2: Run focused image/project/action tests and verify RED**

```bash
PYTHONPATH=src /app/imgMaker/.venv/bin/python -m pytest \
  tests/unit/image/test_publish.py \
  tests/unit/image/test_lock.py \
  tests/unit/image/test_build.py \
  tests/unit/project/test_build.py \
  tests/unit/project/test_dependencies.py \
  tests/unit/installer/test_actions.py -q
```

Expected: tests fail because registry runner injection, plain BuildKit progress,
and observed local builds do not exist.

- [x] **Step 3: Inject the runner into release registry commands**

Change `DockerPublishRegistry.__init__` to accept `runner: Runner | None = None`
and default to `SubprocessRunner()` so publication callers preserve captured
behavior. Replace `_completed` with
`runner.run([*self.docker_prefix, *args], check=False)` and return
`CommandResult`; keep the current `PublishError` evidence and JSON parsing.
`authless_pull` must continue using `--config <empty-0700-directory>` and no
credential-bearing environment.

Construct the default installer registry with the already-injected action
runner:

```python
self.runner = runner or SubprocessRunner()
self.release_docker = release_docker or AnonymousReleaseRegistry(
    self.docker_prefix,
    runner=self.runner,
)
```

- [x] **Step 4: Route local and project builds through observed plain output**

Add `--progress=plain` to `build_rocm_python_argv`, `build_torch_argv`, and
`project_build_argv`. Add optional `observer: CommandObserver | None = None` to
`build_rocm_python` and `build_rocm_pytorch`; both pass it to `_run_live`, while
the PyTorch build also passes it to `_prepare_profile_artifacts`. Replace the
inherited subprocess call with:

```python
result = run_observed_command(
    argv,
    observer=observer,
    check=False,
    cwd=cwd,
    environment=environment,
)
if result.returncode != 0:
    raise BuildError(
        f"command failed ({result.returncode}): {' '.join(argv)}: "
        f"{result.stderr.strip() or result.stdout.strip() or 'no output'}"
    )
```

Give `ProductionInstallerActions` an optional
`command_observer: CommandObserver | None` and pass it into both local image
build functions. Project Buildx and Docker-contained uv/pip already use
`self.runner`; the Task 3 classifier makes them live automatically.

Extend `image.lock.download` with an optional
`progress: Callable[[int, int | None], None] = None`. Read a valid nonnegative
`Content-Length`, emit `(0, total)` when network transfer starts, then emit after
every additional `PROGRESS_INTERVAL_BYTES = 64 * 1024**2` and once at completion.
Do not emit for an already verified cache hit. In `_prepare_profile_artifacts`,
pass a closure that renders
`下载 <wheel filename>: <downloaded GiB>/<total GiB>` (or `总大小未知`) through
`command_observer.command_output(CommandStream.STDOUT, line)`. Never include an
unredacted wheel URL in the line. This direct downloader output follows the same
quiet/log/redaction behavior and resets the stage heartbeat.

- [x] **Step 5: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass; pulls and builds
use the observed path, while inspect/hash/JSON calls remain captured.

- [x] **Step 6: Commit long Docker and package output integration**

```bash
git add src/amd_ai/image/publish.py src/amd_ai/image/build.py \
  src/amd_ai/image/lock.py src/amd_ai/project/build.py \
  src/amd_ai/installer/actions.py tests/unit/image/test_publish.py \
  tests/unit/image/test_lock.py tests/unit/image/test_build.py \
  tests/unit/project/test_build.py tests/unit/project/test_dependencies.py \
  tests/unit/installer/test_actions.py
git commit -m "feat: stream Docker and package progress"
```

## Task 6: Preserve privileged stdout JSON while streaming stderr

**Files:**

- Modify: `src/amd_ai/runner.py`
- Modify: `src/amd_ai/installer/progress.py`
- Modify: `src/amd_ai/installer/actions.py:247-387`
- Modify: `src/amd_ai/installer/privileged.py`
- Modify: `tests/unit/test_runner.py`
- Modify: `tests/unit/installer/test_progress.py`
- Modify: `tests/unit/installer/test_actions.py`
- Modify: `tests/unit/installer/test_privileged.py`

- [x] **Step 1: Write failing protocol-channel tests**

Add a runner test that invokes Python writing `progress-one` and
`progress-two` to stderr around one JSON object on stdout. Call
`run_protocol_command` with a recording observer and assert:

```python
assert json.loads(result.stdout) == {"schema_version": 1}
assert "progress-one" not in result.stdout
assert observer.stderr_lines == ["progress-one\n", "progress-two\n"]
assert result.stdout_truncated is False
```

Add an output-limit test that writes more than 1 MiB to stdout and asserts
`stdout_truncated is True`.

Update action tests to inject a `privileged_run(argv, *, observer)` returning a
`CommandResult`. Assert the helper command includes
`--progress-mode default`, stdout JSON parses, and observer receives stderr.
Add malformed cases for JSON plus extra stdout and truncated stdout; all must
raise `ActionError` and include bounded stderr evidence, never raw stdout in the
terminal sink.

In helper tests, monkeypatch root actions so apply emits simulated command
output. Assert helper stdout is exactly one parseable JSON line and all progress
is on stderr. Assert direct invocation without `--progress-mode` is quiet.

- [x] **Step 2: Run protocol tests and verify RED**

```bash
PYTHONPATH=src /app/imgMaker/.venv/bin/python -m pytest \
  tests/unit/test_runner.py \
  tests/unit/installer/test_progress.py \
  tests/unit/installer/test_actions.py \
  tests/unit/installer/test_privileged.py -q
```

Expected: failures show the privileged helper still captures both streams with
`subprocess.run` and has no progress-mode channel contract.

- [x] **Step 3: Add stderr-only protocol command execution**

Implement this wrapper in `runner.py` by reusing the concurrent pipe engine:

```python
def run_protocol_command(
    args: Sequence[str],
    *,
    observer: CommandObserver | None,
) -> CommandResult:
    return run_observed_command(
        args,
        observer=observer,
        check=False,
        observe_stdout=False,
        observe_stderr=True,
        stdout_limit=1024 * 1024,
        stderr_limit=256 * 1024,
    )
```

Unobserved stdout must be captured exactly for JSON parsing. Observed stderr is
sanitized before terminal/log/tail storage. Set the truncation flag instead of
silently accepting oversized protocol output.

- [x] **Step 4: Add a helper stderr command observer**

Implement `StderrCommandObserver(mode, stderr)` in `progress.py`. It applies the
same sanitizer and secret tracking as `InstallerProgress`, maps both child
streams to helper stderr, suppresses live lines in quiet mode, and emits
redacted `COMMAND` lines only in verbose mode. It never opens a log; the parent
reporter logs the helper's stderr.

- [x] **Step 5: Replace privileged `subprocess.run` with the protocol runner**

Replace the `sudo_run` constructor argument with:

```python
privileged_run: Callable[..., CommandResult] = run_protocol_command,
command_observer: CommandObserver | None = None,
progress_mode: ProgressMode = ProgressMode.DEFAULT,
```

Append `--progress-mode <value>` in `_sudo_helper_command`. In
`_run_sudo_helper`, call `privileged_run(command, observer=observer)` where
`observer` is `self.command_observer` or a quiet `StderrCommandObserver` when no
installer reporter was injected. This guarantees stderr is sanitized even for
direct action callers while remaining terminal-silent. Reject nonzero status,
truncated stdout, and any `json.loads` failure. `json.loads` remains
intentionally strict about extra non-whitespace before or after the object.

- [x] **Step 6: Configure root helper commands to stderr only**

Add `--progress-mode` choices to the helper parser with default `quiet`. Create
`StderrCommandObserver`, then construct:

```python
observer = StderrCommandObserver(
    mode=ProgressMode(args.progress_mode),
    stderr=sys.stderr,
)
actions = ProductionInstallerActions(
    effective_uid=0,
    runner=SubprocessRunner(observer=observer),
    command_observer=observer,
    progress_mode=ProgressMode(args.progress_mode),
)
```

Keep the final `json.dumps(...)` as the only stdout write. Errors remain one
sanitized stderr line and exit code 2.

- [x] **Step 7: Run protocol tests and verify GREEN**

Run the command from Step 2. Expected: all tests pass; helper stdout is strict
JSON and long root command output reaches only the parent's stderr observer.

- [x] **Step 8: Commit privileged channel separation**

```bash
git add src/amd_ai/runner.py src/amd_ai/installer/progress.py \
  src/amd_ai/installer/actions.py src/amd_ai/installer/privileged.py \
  tests/unit/test_runner.py tests/unit/installer/test_progress.py \
  tests/unit/installer/test_actions.py tests/unit/installer/test_privileged.py
git commit -m "feat: stream privileged progress safely"
```

## Task 7: Wire CLI modes and fixture-level progress

**Files:**

- Modify: `src/amd_ai/cli.py:121-136,445-519`
- Modify: `tests/cli/test_installer_commands.py`
- Modify: `tests/cli/test_installer_resume.py`

- [x] **Step 1: Write failing CLI flag and wiring tests**

Add parser tests:

```python
def test_install_progress_modes_are_mutually_exclusive() -> None:
    verbose = cli.build_parser().parse_args(["install", "--verbose"])
    quiet = cli.build_parser().parse_args(["install", "--quiet"])
    assert verbose.verbose is True and verbose.quiet is False
    assert quiet.quiet is True and quiet.verbose is False
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["install", "--verbose", "--quiet"])
```

Extend workflow-construction tests to capture that the same reporter instance
is passed to `InstallerWorkflow`, `SubprocessRunner(observer=...)`, and
`ProductionInstallerActions(command_observer=...)`, and that quiet/verbose map
to the exact `ProgressMode` enum.

- [x] **Step 2: Add fixture CLI progress and private-log regressions**

Extend `run_install` with progress flags. For a healthy container fixture,
assert default stdout contains `PLAN`, `[1/8]`, `START`, `PASS`, `SUMMARY`, and
an initial and final `LOG` line containing the same path. Resolve that path and
assert it is under the fixture HOME, exists with mode `0600`, and contains all
eight stage events.

Run a second completed invocation and assert all eight stages are `SKIP` and
the fixture action log is unchanged. Run with `--quiet` and assert terminal
output omits `PLAN`, `START`, `WAIT`, and `PASS`, contains `SUMMARY` and the
final log path, while the log itself contains suppressed events. Run with
`--verbose` and assert stage enum names and `DEBUG` diagnostics appear.

Update the implicit-state assertion from the old informational line to:

```python
assert f"状态={selected}（per-project）" in second.stdout
```

- [x] **Step 3: Run CLI tests and verify RED**

```bash
PYTHONPATH=src /app/imgMaker/.venv/bin/python -m pytest \
  tests/cli/test_installer_commands.py \
  tests/cli/test_installer_resume.py -q
```

Expected: parser and integration tests fail because CLI reporter construction
and output flags are not wired.

- [x] **Step 4: Add CLI flags and construct one dependency graph**

Create a mutually exclusive argument group on the `install` parser. In
`_install_command`, map flags to `ProgressMode`, then construct dependencies in
this order:

```python
progress = InstallerProgress(mode=progress_mode)
runner = SubprocessRunner(observer=progress)
actions = ProductionInstallerActions(
    runner=runner,
    command_observer=progress,
    progress_mode=progress_mode,
    non_interactive=args.non_interactive,
    docker_prefix=docker_prefix,
)
workflow = InstallerWorkflow(
    options=options,
    actions=actions,
    installer_version=__version__,
    installer_source_revision=revision,
    prompts=prompts,
    progress=progress,
    **workflow_arguments,
)
```

Fixture actions still receive the real reporter through the workflow but do
not construct production runners. Remove the post-workflow raw
`print(result.message, ...)`; `installation_finished` now owns the one final
rendering. Preserve `WorkflowResult.message` and exit codes for Python callers.
`install.sh` already forwards unknown arguments unchanged; keep and test that
behavior rather than adding shell parsing.

- [x] **Step 5: Run CLI tests and verify GREEN**

Run the command from Step 3. Expected: all CLI tests pass with deterministic
line-oriented events and private logs under the fixture HOME.

- [x] **Step 6: Commit CLI progress integration**

```bash
git add src/amd_ai/cli.py tests/cli/test_installer_commands.py \
  tests/cli/test_installer_resume.py
git commit -m "feat: add installer progress modes"
```

## Task 8: Version, user guide, and immutable-image release note

**Files:**

- Modify: `src/amd_ai/__init__.py`
- Modify: `tests/test_version.py`
- Modify: `README.md`
- Modify: `docs/install.md`
- Create: `docs/releases/v0.2.3.md`

- [x] **Step 1: Write failing version and stable-baseline assertions**

Change both version assertions in `tests/test_version.py` to `0.2.3`. Add:

```python
def test_installer_only_release_keeps_stable_image_baseline() -> None:
    path = Path("profiles/releases/stable.json")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "4226d04bf995c9c253c6a978f08bdbb9466ccd47119f967ebd39f0c08b7bfe2d"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["release_id"] == "0.2.0"
    assert payload["base"]["manifest_digest"] == (
        "sha256:e9991f97f578156c8620fbb587d2d34504eb632f165cc5597deaadaa3e692a12"
    )
    assert payload["torch"]["manifest_digest"] == (
        "sha256:dc0bb217474cfd4f602423bd3bf4fe8714b03e900cf3c6b4417b99e622ebcf8b"
    )
```

- [x] **Step 2: Run version tests and verify RED**

```bash
PYTHONPATH=src /app/imgMaker/.venv/bin/python -m pytest tests/test_version.py -q
```

Expected: version assertions fail with current `0.2.2`; stable-image
assertions already pass.

- [x] **Step 3: Bump only toolkit version and document the complete workflow**

Set `__version__ = "0.2.3"`. Update every current clone/tag/version example in
README to `v0.2.3`, but leave stable release ID, ROCm 7.2.1, Python 3.12,
PyTorch 2.9.1, and all image digests unchanged.

Add a README section named `安装进度与私有日志` covering:

- the default `PLAN/SKIP/START/DETAIL/WAIT/PASS/SUMMARY` sequence;
- `--verbose` and `--quiet`, including mutual exclusion and CI behavior;
- the per-project log path and `0700`/`0600` permissions;
- logs are not rotated or deleted automatically in `v0.2.3`;
- the 15-second heartbeat and line-oriented non-TTY behavior;
- Docker pull, BuildKit, uv/pip, locked-wheel downloads, and sudo host
  operations that stream;
- failure `CAUSE/STATE/RESUME/LOG` interpretation;
- review of sanitized logs for project-specific non-secret data before sharing.

Update `docs/install.md` with complete new, resumed, already-complete, reboot,
and failed examples. Document that a `v0.2.2` container state after `BOOTSTRAP`
can be adopted by `v0.2.3`, only its verified `BOOTSTRAP` digest changes, and
state deletion remains forbidden remediation.

Create `docs/releases/v0.2.3.md` stating that this is an installer
observability release, no ComfyUI/model-cache policy changes are included, and
the two exact stable manifests are reused without image publication.

- [x] **Step 4: Verify version, documentation, and immutable release data**

```bash
PYTHONPATH=src /app/imgMaker/.venv/bin/python -m pytest tests/test_version.py -q
npx --yes markdownlint-cli@0.45.0 README.md docs/install.md \
  docs/releases/v0.2.3.md \
  docs/superpowers/specs/2026-07-10-installer-progress-design.md \
  docs/superpowers/plans/2026-07-10-installer-progress.md \
  --disable MD013
sha256sum profiles/releases/stable.json
git diff --check
```

Expected: tests and Markdown pass; SHA-256 is exactly
`4226d04bf995c9c253c6a978f08bdbb9466ccd47119f967ebd39f0c08b7bfe2d`.

- [x] **Step 5: Commit release-facing changes**

```bash
git add src/amd_ai/__init__.py tests/test_version.py README.md \
  docs/install.md docs/releases/v0.2.3.md
git commit -m "docs: prepare installer progress release"
```

## Task 9: Full verification, production-state regression, and publication

**Files:**

- Verify only; no planned source edits

- [ ] **Step 1: Run strict checks on every changed Python file**

```bash
uvx --from ruff==0.12.3 ruff check \
  src/amd_ai/__init__.py src/amd_ai/cli.py src/amd_ai/runner.py \
  src/amd_ai/installer/actions.py src/amd_ai/installer/privileged.py \
  src/amd_ai/installer/progress.py src/amd_ai/installer/prompts.py \
  src/amd_ai/installer/state.py src/amd_ai/installer/workflow.py \
  src/amd_ai/image/build.py src/amd_ai/image/lock.py \
  src/amd_ai/image/publish.py \
  src/amd_ai/project/build.py tests/test_version.py \
  tests/unit/test_runner.py tests/unit/installer/test_actions.py \
  tests/unit/installer/test_privileged.py tests/unit/installer/test_progress.py \
  tests/unit/installer/test_prompts.py tests/unit/installer/test_state.py \
  tests/unit/installer/test_workflow.py tests/unit/image/test_build.py \
  tests/unit/image/test_lock.py tests/unit/image/test_publish.py \
  tests/unit/project/test_build.py \
  tests/unit/project/test_dependencies.py tests/cli/test_installer_commands.py \
  tests/cli/test_installer_resume.py
```

Expected: zero Ruff findings in changed files. Do not apply repository-wide
cleanup for pre-existing F401, F541, or F811 findings outside this list.

- [ ] **Step 2: Run the complete non-hardware suite and documentation checks**

```bash
PYTHONPATH=src /app/imgMaker/.venv/bin/python -m pytest -m 'not hardware' -q
npx --yes markdownlint-cli@0.45.0 README.md docs/install.md \
  docs/releases/v0.2.3.md \
  docs/superpowers/specs/2026-07-10-installer-progress-design.md \
  docs/superpowers/plans/2026-07-10-installer-progress.md \
  --disable MD013
git diff --check
git status --short
```

Expected: all non-hardware tests pass, Markdown and whitespace checks exit zero,
and only intentional implementation-plan progress edits remain, if the plan is
being checked off during execution.

- [ ] **Step 3: Verify release identity without rebuilding images**

```bash
PYTHONPATH=src bin/strix-halo-rocm release verify \
  --manifest profiles/releases/stable.json
sha256sum profiles/releases/stable.json
```

Expected exact stable manifests:

```text
sha256:e9991f97f578156c8620fbb587d2d34504eb632f165cc5597deaadaa3e692a12
sha256:dc0bb217474cfd4f602423bd3bf4fe8714b03e900cf3c6b4417b99e622ebcf8b
```

The manifest file SHA-256 remains
`4226d04bf995c9c253c6a978f08bdbb9466ccd47119f967ebd39f0c08b7bfe2d`.
Do not run image build, push, or release publish commands.

- [ ] **Step 4: Exercise patch adoption and real local Docker safely**

Record SHA-256 for the user's original
`~/.local/state/strix-halo-rocm-toolkit/projects/video-lab-b9bb64878f63.json`
and `/app/test/video-lab/amd-ai-project.toml`, but do not use either as a fixture
input before publication. Their digests bind absolute manifest/source/project
paths and cannot be relocated without correctly triggering strict input-change
protection.

Instead, use the detached public `v0.2.2` checkout and the existing healthy
installer fixture to create three completed states under a temporary HOME. Use
the same absolute project and manifest paths for each initial run. Resume copies
with the feature worktree's `v0.2.3` installer in default, quiet, and verbose
modes and verify:

- only each verified `BOOTSTRAP` digest changes during adoption;
- all completed actions remain skipped;
- every run creates a private log with the documented terminal/log filtering;
- a fourth state with a damaged old `BOOTSTRAP` digest prevents every action;
- the recorded real user files remain byte-for-byte unchanged.

Then use a harmless local Python observed-command probe and captured
`docker info --format '{{.ServerVersion}}'` metadata probe to confirm tee and
capture classification. Do not pull or rebuild the multi-gigabyte stable images
for this check.

- [ ] **Step 5: Review and publish toolkit `v0.2.3`**

Review `git diff v0.2.2...HEAD`, confirm no stable manifest, profile lock,
Dockerfile baseline, or image digest changed, then use the finishing-branch
workflow to fast-forward `main`. Create annotated tag `v0.2.3` and atomically
push `main` plus the tag. Do not publish a GHCR image for this toolkit-only
release.

```bash
git tag -a v0.2.3 -m "Strix Halo ROCm Toolkit v0.2.3"
git push --atomic origin main v0.2.3
```

Verify the public tag and raw version file:

```bash
git ls-remote --refs origin refs/heads/main refs/tags/v0.2.3
```

Open the public raw `src/amd_ai/__init__.py` at tag `v0.2.3` and verify it says
`0.2.3`. Confirm anonymous pulls still resolve the two existing stable digest
references; do not pull layers that are already present solely to generate
progress output.

- [ ] **Step 6: Run the user's completed project against the public tag**

Update `/app/test/strix-halo-rocm-toolkit` to detached public tag `v0.2.3` and
run:

```bash
git fetch --tags origin
git switch --detach v0.2.3
./install.sh --mode container \
  --project-dir /app/test/video-lab \
  --project-name video-lab \
  --image-source pull
```

Expected: plan and private log path are shown, all eight trusted stages report
`SKIP`, final `SUMMARY` succeeds, no Docker layer is redownloaded, the old
`/app/test/rocmToolkit` state remains unchanged, and the project continues to
use the exact PyTorch 2.9.1 stable parent.
