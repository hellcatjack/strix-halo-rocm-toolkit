# Installer real-time progress design

## Problem

The installer can spend minutes downloading or building tens of gigabytes, but
its current stage loop prints a `PASS` line only after each stage completes.
Several production paths also capture child-process output in pipes and expose
it only after failure:

- Docker release pulls are captured by the release registry;
- Docker Buildx project builds, including uv and pip work in the build, are
  captured by `SubprocessRunner`;
- host package operations executed through the privileged helper are captured;
- completed checkpoints are silently skipped, so a resumed run does not explain
  where it resumed or why earlier work did not run again.

As a result, a healthy large download looks stalled, while a failure lacks the
stage position, elapsed time, state path, log path, and exact resume action that
an operator needs.

## Decision

Add a structured progress subsystem to the installer and stream sanitized output
for known long-running commands. Every installation will show its plan, stage
position, completed checkpoints, current activity, elapsed time, and recovery
instructions. A private persistent log will contain the same structured events
and all sanitized child output.

This is toolkit release `v0.2.3`. Stable image release ID `0.2.0`, the ROCm
7.2.1 and PyTorch 2.9.1 baseline, immutable image references, release manifest,
and image digests remain unchanged. Publishing `v0.2.3` does not rebuild or
republish either stable image.

## Goals

- Make long downloads, builds, and package installation visibly active in both
  interactive terminals and CI logs.
- Identify every stage as `[current/total]` and report its elapsed time.
- Explain checkpoint reuse with explicit `SKIP` events.
- Emit a `WAIT` heartbeat after 15 seconds without visible child output.
- Preserve enough sanitized output to diagnose a failed command without
  retaining unbounded data in memory.
- Produce a mode-`0600` session log and show its path on success and failure.
- Keep the privileged helper's stdout as a strict single-JSON-object protocol.
- Preserve current checkpoint, state digest, image identity, and PyTorch guard
  behavior.

## Non-goals

- No TTY spinner, progress bar, cursor movement, color requirement, or terminal
  dashboard is introduced.
- Package-manager output is not parsed into synthetic byte or percentage
  counters; the package manager remains the source of those details.
- Fast metadata, hash, GPU, and Docker inspect probes are not streamed by
  default.
- ComfyUI is not installed, and no ComfyUI or Hugging Face cache-sharing policy
  is added.
- This change does not alter project container isolation, protected PyTorch
  repair, user-selected PyTorch support, or host configuration policy.

## User-visible contract

### Session plan

After project input and state-path selection, but before dispatching a stage,
the installer records a plan containing:

- install mode;
- normalized project directory and project name;
- selected state path and its selection source;
- requested image source and stable release ID when resolved;
- total stage count;
- first incomplete stage, or `already complete`;
- private session-log path.

Default and verbose modes display the plan; quiet mode writes it only to the
session log. The plan uses one line per field so it remains readable when
redirected. A container-mode resume can look like this:

```text
PLAN     模式=container，项目=/app/test/video-lab，名称=video-lab
PLAN     状态=/home/hellcat/.local/state/strix-halo-rocm-toolkit/projects/video-lab-b9bb64878f63.json（per-project）
PLAN     镜像来源=pull，stable release=0.2.0
PLAN     共 8 个阶段，从 IMAGE_PULL_OR_BUILD 继续
LOG      /home/hellcat/.local/state/strix-halo-rocm-toolkit/logs/video-lab-b9bb64878f63/install-20260710T142533Z-18421.log
```

`RELEASE_RESOLVE` may be the stage that first makes the release ID available.
If it was not persisted before the plan is printed, the plan says
`stable release=待解析`, and a `DETAIL` event reports the resolved ID later.

### Stage events

The structured event vocabulary is `PLAN`, `LOG`, `SKIP`, `START`, `DETAIL`,
`WAIT`, `PASS`, `INFO`, `WARN`, `ACTION`, `BLOCKED`, `FAIL`, `CAUSE`, `STATE`,
`RESUME`, and `SUMMARY`. Verbose-only command and diagnostic events use
`COMMAND` and `DEBUG`. Prefixes keep the existing eight-character padded
column; raw child lines are not given a misleading structured prefix.

Each stage has a stable display label. The enum value is retained in verbose
output and all failure summaries so documentation, state, and terminal output
can be correlated.

| Stage | Display label |
| --- | --- |
| `BOOTSTRAP` | 安装用户运行时 |
| `HOST_PREFLIGHT` | 检查宿主 |
| `HOST_PLAN` | 生成宿主变更计划 |
| `HOST_CONFIRM` | 确认宿主变更 |
| `HOST_APPLY` | 应用宿主变更 |
| `REBOOT_PENDING` | 检查重启状态 |
| `HOST_VERIFY` | 验证重启后的宿主 |
| `CONTAINER_HOST_CHECK` | 验证容器宿主 |
| `RELEASE_RESOLVE` | 解析 stable release |
| `IMAGE_PULL_OR_BUILD` | 获取或构建 stable 镜像 |
| `IMAGE_VERIFY` | 验证 gfx1151 PyTorch GPU runtime |
| `PROJECT_INIT` | 创建项目镜像与 Python 依赖 |
| `PROJECT_VERIFY` | 验证项目 |
| `COMPLETE` | 完成安装 |

The full workflow has 13 stages and the container workflow has 8 stages, using
the existing `FULL_STAGE_ORDER` and `CONTAINER_STAGE_ORDER` respectively. Stage
numbers are positions in the selected workflow, not global enum positions.

For each completed checkpoint whose inputs still validate, emit `SKIP`. Emit
`START` before the incomplete stage's pre-dispatch checks and `PASS` only after
its output is applied and its checkpoint is saved. This ordering makes slow disk
and registry estimates visible as part of the stage and ensures that `PASS`
means the stage is resumable after process loss.

```text
SKIP     [1/8] 安装用户运行时：已有可信检查点
START    [4/8] 获取或构建 stable 镜像（IMAGE_PULL_OR_BUILD）
DETAIL   缺失层=18.7 GiB，可用空间=126.4 GiB，位置=/var/lib/docker，来源=公开 GHCR
         <Docker pull 的实时分层下载输出>
WAIT     [4/8] 已运行 00:30，15 秒内没有新输出，仍在获取或构建 stable 镜像
PASS     [4/8] 获取或构建 stable 镜像，用时 02:14
START    [6/8] 创建项目镜像与 Python 依赖（PROJECT_INIT）
         <BuildKit、uv 或 pip 的实时解析、下载和安装输出>
PASS     [6/8] 创建项目镜像与 Python 依赖，用时 04:38
```

Before `IMAGE_PULL_OR_BUILD`, emit payload bytes, required bytes including the
existing safety margin, available bytes, Docker storage location, and source.
For a pull, payload bytes mean missing compressed release layers; for a local
build, they remain the existing conservative build estimate. Before
`PROJECT_INIT`, emit the existing project payload, required bytes, available
bytes, and project filesystem location. Values are rendered in binary GiB with
one decimal place and exact bytes are included only in verbose mode.

### Heartbeats

A stage timer uses a monotonic clock. If a running stage produces no structured
event or visible child output for 15 seconds, emit `WAIT`; continue emitting one
`WAIT` every 15 seconds until activity resumes or the stage ends. Any emitted
line resets the inactivity timer. Heartbeats are disabled in quiet mode and are
cancelled and joined before the terminal stage event, preventing a late `WAIT`
after `PASS` or `FAIL`.

### Completion and failure

Successful completion reports total elapsed time, project directory, state
path, and log path. A run whose state was already complete still emits the
plan, all validated `SKIP` events, and the completion summary in default mode.

Stage failure output is bounded and actionable:

```text
FAIL     [6/8] 创建项目镜像与 Python 依赖（PROJECT_INIT），用时 03:17
CAUSE    uv 无法下载 package ...
STATE    /home/hellcat/.local/state/strix-halo-rocm-toolkit/projects/video-lab-b9bb64878f63.json
RESUME   修复网络或依赖后重新执行同一条 install 命令；前 5 个阶段不会重放
LOG      /home/hellcat/.local/state/strix-halo-rocm-toolkit/logs/video-lab-b9bb64878f63/install-20260710T142533Z-18421.log
```

The cause contains the exception plus at most the final 40 sanitized child
lines, capped at 16 KiB. The full sanitized stream remains in the session log.
Pre-stage errors use `FAIL installer` without a stage position but still show
the selected state and log paths when they are known. Exit codes remain `0` for
success, `1` for interruption or required operator action, and `2` for blocked
or invalid execution, matching current behavior.

For reboot-required or other operator-action outcomes, use `ACTION` rather than
`FAIL`, then print `STATE`, the exact next command or action, and `LOG`. Existing
strict state mismatch messages remain errors and never recommend deleting a
state file.

## Output modes and streams

The `install` command gains a mutually exclusive argument group:

- default: structured events plus live output for classified long commands;
- `--verbose`: default output plus redacted command lines, exact byte counts,
  probe details, and captured diagnostic output;
- `--quiet`: prompts, required actions, warnings, failures, and the final
  summary only; successful `SKIP`, `START`, `DETAIL`, `WAIT`, `PASS`, and live
  child lines are suppressed from the terminal but still written to the log.

`--non-interactive` does not imply `--quiet`. Supplying both `--verbose` and
`--quiet` is a parser error before workflow execution. The top-level
`install.sh` forwards both options without special handling.

Normal structured events and normal child stdout are written to installer
stdout. Warnings, failures, required actions, and child stderr are written to
installer stderr. Prompts continue using the controlling terminal streams.
When stdout or stderr is not a TTY, output remains newline-delimited and never
uses cursor control, carriage-return rewriting, or a spinner. The same
line-oriented format is valid in a TTY so logs and screenshots are stable.

The private session log records both streams with UTC timestamp, event type or
child stream, and sanitized text. Cross-stream ordering is best effort, while
ordering within each child stream is preserved.

## Progress architecture

### Reporter

Add `src/amd_ai/installer/progress.py` with a small `ProgressReporter` protocol
and one production implementation. It owns:

- output mode filtering;
- stage metadata and `[current/total]` formatting;
- session and stage monotonic timers;
- the heartbeat worker;
- stdout and stderr terminal sinks;
- serialized, sanitized log writes;
- a bounded tail of child output for failure evidence.

The reporter accepts injected wall and monotonic clocks for deterministic
tests. Calls are thread-safe because stdout and stderr readers and the heartbeat
worker may report concurrently. The reporter does not mutate install state and
cannot decide whether a stage succeeds.

The CLI creates the reporter and injects it into the workflow, production
actions, release registry, and subprocess runner. The workflow opens the log
after project input is normalized, emits the session plan after state selection
and transition validation, and brackets each stage with reporter calls. Direct
workflow construction remains supported through a compatibility reporter that
delegates legacy status events to the supplied prompts.

Existing `prompts.status` calls are routed through the reporter in production
so one event is printed and logged exactly once. Prompt classes continue to own
interactive input only.

### Log identity and permissions

Logs live at:

```text
~/.local/state/strix-halo-rocm-toolkit/logs/
  <safe-project-basename>-<normalized-path-sha256-prefix>/
  install-<UTC-YYYYMMDDTHHMMSSZ>-<pid>.log
```

The project component uses the same safe basename and 12-hex path identity as
automatic per-project state filenames. This remains project-derived even when
an explicit state path is supplied. The toolkit, logs, and project-log
directories are created with mode `0700`; each new log is opened atomically
with mode `0600`. Existing broader permissions are tightened where owned by the
current user; symlinked directories or log files are rejected. A timestamp and
PID collision fails closed rather than appending to an existing file.

Log creation failure stops the install before any mutating stage. Logs are not
rotated or deleted automatically in this release.

## Live subprocess execution

`SubprocessRunner` keeps its existing `run` call signature so current action
fakes and runner consumers remain compatible. Its constructor gains an optional
reporter and a command-classification policy. Captured execution remains the
default. The policy selects live tee execution only for:

- `docker pull`;
- `docker build` and `docker buildx build`;
- apt, apt-get, and dpkg mutation commands;
- direct uv and pip resolution, download, and install commands;
- Docker container commands whose executed workload is uv, pip, or
  `python -m pip`.

Docker inspect, image-ID, manifest, hash, JSON-producing, and GPU verification
commands stay captured by default because callers parse their output. Verbose
mode logs their redacted command and summarized diagnostics after parsing; it
does not indiscriminately mirror protocol output to the terminal.

Live execution uses separate concurrent readers for stdout and stderr to avoid
pipe deadlock. Each complete line is sanitized, sent to its corresponding
terminal stream subject to output mode, written to the session log, and used to
reset the heartbeat. A final unterminated fragment is emitted at EOF. Carriage
returns and unsafe control sequences are normalized to newline-oriented text.
Each stream retains at most 256 KiB in a ring buffer for return values and error
evidence; the log writer receives the full sanitized stream incrementally and
no complete command output is accumulated in memory.

BuildKit is requested in plain-progress mode whenever output is redirected so
layer and package activity remains line-oriented. Interrupts terminate the
child process using the current runner's signal behavior, drain remaining
output, and preserve exit status semantics.

The anonymous release registry is changed to use the injected runner for pulls
as well as captured inspect and identity commands. Local stable-image builds and
project Buildx builds use the same runner path, replacing inherited or fully
captured subprocess calls so their output is both visible and persisted.

## Privileged helper protocol

The privileged host helper has two distinct channels:

- stdout contains exactly one JSON result object and no progress text;
- stderr contains sanitized line-oriented host package and Docker progress plus
  helper diagnostics.

The parent starts the helper with both pipes. It drains stderr concurrently
through the reporter while retaining only a bounded tail, and captures stdout
privately for strict JSON parsing. The parent does not pass helper stdout to the
generic live-output sink. On success, any extra non-whitespace stdout before or
after the JSON object remains a protocol error. On failure, the bounded stderr
tail is included in the cause and the full sanitized stderr is available in the
session log.

Inside the helper, host mutations use a reporter sink configured for stderr
only. This includes apt, apt-get, dpkg, and Docker commands. Planning and verify
operations remain captured unless verbose diagnostics are requested. This
separation prevents package progress from corrupting the parent-child JSON
contract.

The parent passes an explicit `--progress-mode default|verbose|quiet` helper
argument. Direct helper invocation defaults to `quiet` to preserve its existing
machine-oriented behavior. The helper never creates a second log; the parent
sanitizes and persists helper stderr in the installation session log.

## Sanitization and redaction

Terminal and log output pass through one sanitizer before storage or display.
It:

- removes ANSI CSI and OSC sequences and unsafe C0/C1 controls while preserving
  newline and tab;
- converts carriage-return updates to normal lines;
- redacts URL user information;
- redacts values assigned to environment names ending in `TOKEN`, `PASSWORD`,
  `PASS`, `SECRET`, `KEY`, `CREDENTIAL`, or `AUTH`;
- redacts bearer and basic authorization values and values following known
  credential flags;
- applies the repository's existing URL and command redactors to verbose
  command rendering.

Redaction occurs before terminal output, log writes, and in-memory tail storage.
Tests use representative GitHub, package-index, Hugging Face, and authorization
tokens and assert that their literal values appear in none of those sinks.
Redaction is defense in depth; documentation still instructs users not to place
credentials directly in project dependency URLs or command arguments.

## Workflow and recovery semantics

Progress is observational. The existing order remains:

1. validate completed-stage inputs;
2. emit `SKIP` when the trusted checkpoint is reusable;
3. emit `START` and start the stage timer;
4. calculate and report any pre-dispatch disk requirement;
5. dispatch the stage;
6. apply stage output;
7. save the checkpoint;
8. emit `PASS`.

Disk shortage is reported as a stage `FAIL` before command dispatch, with the
required and available capacities. A child command failure never checkpoints
the stage. Re-running the same command revalidates and skips prior stages, then
starts at the failed stage. Installer interruption closes and flushes the log
before returning.

The progress mode, log path, timestamps, and elapsed durations are never added
to `InstallOptions` stage inputs or `InstallState`. Therefore they cannot change
stage digests. The state schema remains version 2.

The version and source revision already belong to the `BOOTSTRAP` digest, so a
toolkit upgrade must use the existing compatible-update mechanism. For a valid
same-series `0.2.x` state, adoption verifies the old `BOOTSTRAP` digest, updates
the recorded installer metadata, and replaces only that digest. Full mode keeps
its existing safety boundary of `HOST_VERIFY` or later. Container mode may adopt
after `BOOTSTRAP` is complete because it has no host-write stages. A state that
does not satisfy these rules continues to fail closed with the existing input
change or transition error. No other completed-stage digest is rewritten.

## Tests

Automated coverage includes:

- exact default, verbose, and quiet event filtering;
- parser rejection of `--verbose --quiet`;
- stage labels and positions for full and container workflows;
- plan output for new, resumed, complete, legacy, per-project, and explicit
  state selections;
- `SKIP`, `START`, checkpoint-before-`PASS`, failure, action-required, and final
  summary ordering;
- 15-second repeated heartbeats with fake clocks, activity resets, and clean
  worker shutdown;
- disk-detail rendering for pulls, local builds, and project builds;
- incremental stdout and stderr tee behavior without deadlock;
- bounded 256-KiB stream tails and 40-line/16-KiB failure evidence;
- partial-line handling, carriage-return normalization, and control-sequence
  removal;
- secret redaction in terminal output, logs, verbose commands, and failure
  tails;
- log identity, `0700` directories, `0600` files, collision refusal, symlink
  refusal, flush-on-interrupt, and quiet-mode completeness;
- streamed Docker pull and Buildx output while inspect and JSON probes remain
  captured;
- privileged helper stderr streaming with stdout JSON isolation, including
  malformed and extra-output rejection;
- fixture-backend deterministic progress suitable for CLI integration tests;
- compatible `v0.2.2` to `v0.2.3` adoption that rewrites only the verified
  `BOOTSTRAP` digest, including container resumes after `BOOTSTRAP`;
- strict rejection of states outside the compatible-update safety boundaries;
- unchanged stable manifest and image digests.

The existing focused installer suite and full test suite must pass. Ruff is run
strictly on changed Python files; repository-wide pre-existing lint findings do
not authorize unrelated cleanup.

## Documentation and release

README and installation guidance will show default output, `--verbose`,
`--quiet`, log locations and permissions, CI behavior, failure recovery, and a
complete resumed-install example. Troubleshooting will direct users to provide
the sanitized session log while still reviewing it for project-specific data.

The version changes to `0.2.3`. Release notes explicitly state that this is an
installer observability release and that stable ROCm/PyTorch images are reused
by immutable digest. Release verification must confirm that the public GitHub
tag points at the tested commit and that the existing anonymous GHCR pulls are
unchanged.
