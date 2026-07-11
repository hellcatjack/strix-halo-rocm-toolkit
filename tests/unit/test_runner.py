from __future__ import annotations

import json
import threading
from collections.abc import Mapping

import pytest

from amd_ai.installer.progress import sanitize_output
from amd_ai.runner import (
    CommandError,
    CommandObservationError,
    CommandResult,
    CommandStream,
    SubprocessRunner,
    is_live_command,
    run_observed_command,
    run_protocol_command,
)


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

    def command_finished(
        self, result: CommandResult, *, live: bool
    ) -> None:
        self.finished.append((result, live))

    @property
    def stderr_lines(self) -> list[str]:
        return [
            text
            for stream, text in self.lines
            if stream is CommandStream.STDERR
        ]


@pytest.mark.parametrize(
    "argv",
    [
        ["docker", "pull", "ghcr.io/example/image@sha256:" + "a" * 64],
        [
            "sudo",
            "-n",
            "docker",
            "--config",
            "/tmp/empty",
            "pull",
            "image",
        ],
        ["docker", "buildx", "build", "--progress=plain", "."],
        ["apt-get", "update"],
        ["apt", "install", "docker-ce"],
        ["dpkg", "--install", "package.deb"],
        ["uv", "pip", "compile", "requirements.in"],
        ["python3.12", "-m", "pip", "install", "package"],
        [
            "docker",
            "run",
            "--entrypoint",
            "/usr/local/bin/uv",
            "base",
            "pip",
            "compile",
            "requirements.in",
        ],
    ],
)
def test_live_command_policy_accepts_only_long_mutations(
    argv: list[str],
) -> None:
    assert is_live_command(argv) is True


@pytest.mark.parametrize(
    "argv",
    [
        ["docker", "image", "inspect", "image"],
        ["docker", "buildx", "imagetools", "inspect", "--raw", "image"],
        [
            "docker",
            "run",
            "--rm",
            "image",
            "container-check",
            "--json",
            "-",
        ],
        ["pip", "list", "--format=json"],
        ["sha256sum", "artifact"],
    ],
)
def test_live_command_policy_keeps_protocol_and_probe_commands_captured(
    argv: list[str],
) -> None:
    assert is_live_command(argv) is False


def test_runner_captures_stdout_without_shell():
    result = SubprocessRunner().run(["printf", "%s", "ok"])

    assert result.args == ("printf", "%s", "ok")
    assert result.stdout == "ok"
    assert result.returncode == 0


def test_runner_raises_typed_error():
    with pytest.raises(CommandError) as error:
        SubprocessRunner().run(["python3.12", "-c", "import sys; sys.exit(7)"])

    assert error.value.result.returncode == 7


def test_command_result_truncation_defaults_are_backward_compatible() -> None:
    result = CommandResult(("true",), 0, "", "")

    assert result.stdout_truncated is False
    assert result.stderr_truncated is False


def test_live_runner_emits_output_before_process_completion() -> None:
    observer = RecordingObserver()
    runner = SubprocessRunner(observer=observer, live_policy=lambda args: True)
    result: list[CommandResult] = []

    def execute() -> None:
        result.append(
            runner.run(
                [
                    "python3.12",
                    "-c",
                    "import sys,time; print('first', flush=True); "
                    "print('problem', file=sys.stderr, flush=True); "
                    "time.sleep(0.2); print('last', flush=True)",
                ]
            )
        )

    thread = threading.Thread(target=execute)
    thread.start()
    assert observer.first_output.wait(timeout=1.0)
    assert thread.is_alive()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert len(result) == 1
    stdout = [
        text
        for stream, text in observer.lines
        if stream is CommandStream.STDOUT
    ]
    stderr = [
        text
        for stream, text in observer.lines
        if stream is CommandStream.STDERR
    ]
    assert stdout == ["first\n", "last\n"]
    assert stderr == ["problem\n"]
    assert result[0].stdout == "first\nlast\n"
    assert result[0].stderr == "problem\n"
    assert observer.started[0][1] is True
    assert observer.finished[0][1] is True


def test_live_runner_bounds_stream_tails_and_partial_records() -> None:
    observer = RecordingObserver()
    result = run_observed_command(
        [
            "python3.12",
            "-c",
            "import os; os.write(1, b'x' * (400 * 1024)); "
            "os.write(2, b'y' * (400 * 1024))",
        ],
        observer=observer,
    )

    assert len(result.stdout.encode("utf-8")) <= 256 * 1024
    assert len(result.stderr.encode("utf-8")) <= 256 * 1024
    assert result.stdout_truncated is True
    assert result.stderr_truncated is True
    assert max(len(text.encode("utf-8")) for _, text in observer.lines) <= 64 * 1024


def test_live_runner_emits_final_unterminated_fragment() -> None:
    observer = RecordingObserver()

    result = run_observed_command(
        ["python3.12", "-c", "import os; os.write(1, b'final')"],
        observer=observer,
    )

    assert result.stdout == "final"
    assert observer.lines == [(CommandStream.STDOUT, "final")]


def test_observer_failure_drains_child_without_retaining_raw_output() -> None:
    class FailingObserver(RecordingObserver):
        def command_output(
            self, stream: CommandStream, text: str
        ) -> str:
            del stream, text
            raise OSError("log full")

    observer = FailingObserver()

    with pytest.raises(CommandObservationError) as error:
        run_observed_command(
            [
                "python3.12",
                "-c",
                "import os; os.write(1, b'x' * (1024 * 1024)); "
                "os.write(2, b'y' * (1024 * 1024))",
            ],
            observer=observer,
        )

    assert isinstance(error.value.__cause__, OSError)
    assert observer.lines == []


def test_protocol_runner_preserves_stdout_json_and_streams_only_stderr() -> None:
    observer = RecordingObserver()

    result = run_protocol_command(
        [
            "python3.12",
            "-c",
            "import json,sys; "
            "print('progress-one', file=sys.stderr, flush=True); "
            "print(json.dumps({'schema_version': 1}), flush=True); "
            "print('progress-two', file=sys.stderr, flush=True)",
        ],
        observer=observer,
    )

    assert json.loads(result.stdout) == {"schema_version": 1}
    assert "progress-one" not in result.stdout
    assert observer.stderr_lines == ["progress-one\n", "progress-two\n"]
    assert result.stdout_truncated is False


def test_protocol_runner_marks_oversized_stdout_as_truncated() -> None:
    result = run_protocol_command(
        [
            "python3.12",
            "-c",
            "import os; os.write(1, b'x' * (1024 * 1024 + 1))",
        ],
        observer=RecordingObserver(),
    )

    assert result.stdout_truncated is True
    assert len(result.stdout.encode("utf-8")) <= 1024 * 1024
