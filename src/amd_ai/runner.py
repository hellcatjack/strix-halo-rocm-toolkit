from __future__ import annotations

import codecs
import os
import re
import subprocess
import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol


DEFAULT_STREAM_LIMIT = 256 * 1024
MAX_RECORD_BYTES = 64 * 1024


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

    def command_finished(
        self, result: CommandResult, *, live: bool
    ) -> None: ...


class CommandError(RuntimeError):
    def __init__(self, result: CommandResult) -> None:
        super().__init__(
            f"command failed ({result.returncode}): {' '.join(result.args)}"
        )
        self.result = result


class CommandObservationError(RuntimeError):
    pass


class Runner(Protocol):
    def run(
        self,
        args: list[str],
        *,
        check: bool = True,
        input_text: str | None = None,
    ) -> CommandResult: ...


class SubprocessRunner:
    def __init__(
        self,
        *,
        observer: CommandObserver | None = None,
        live_policy: Callable[[Sequence[str]], bool] | None = None,
    ) -> None:
        self.observer = observer
        self.live_policy = live_policy or is_live_command

    def run(
        self,
        args: list[str],
        *,
        check: bool = True,
        input_text: str | None = None,
    ) -> CommandResult:
        if self.observer is not None and self.live_policy(args):
            return run_observed_command(
                args,
                observer=self.observer,
                check=check,
                input_text=input_text,
            )
        if self.observer is not None:
            self.observer.command_started(
                tuple(args),
                live=False,
                environment=os.environ,
            )
        completed = subprocess.run(
            args,
            check=False,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        result = CommandResult(
            args=tuple(args),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        if self.observer is not None:
            try:
                self.observer.command_finished(result, live=False)
            except Exception as error:
                raise CommandObservationError(
                    f"command observer failed: {error}"
                ) from error
        if check and result.returncode != 0:
            raise CommandError(result)
        return result


def is_live_command(args: Sequence[str]) -> bool:
    command = _strip_wrappers(tuple(args))
    if not command:
        return False
    executable = Path(command[0]).name.casefold()
    arguments = command[1:]
    if executable in {"apt", "apt-get"}:
        return bool(arguments) and arguments[0].casefold() in {
            "update",
            "upgrade",
            "full-upgrade",
            "dist-upgrade",
            "install",
            "remove",
            "purge",
            "autoremove",
        }
    if executable == "dpkg":
        return any(
            value.casefold()
            in {"-i", "--install", "--unpack", "--configure", "--remove", "--purge"}
            for value in arguments
        )
    if executable in {"uv", "uvx"}:
        return _uv_is_live(arguments)
    if re.fullmatch(r"pip(?:3(?:\.\d+)?)?", executable):
        return bool(arguments) and arguments[0].casefold() in {
            "install",
            "download",
            "wheel",
        }
    if re.fullmatch(r"python(?:3(?:\.\d+)?)?", executable):
        return _python_pip_is_live(arguments)
    if executable == "docker":
        return _docker_is_live(arguments)
    return False


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
    stdout_limit: int = DEFAULT_STREAM_LIMIT,
    stderr_limit: int = DEFAULT_STREAM_LIMIT,
) -> CommandResult:
    command = tuple(args)
    if observer is None and observe_stdout and observe_stderr:
        completed = subprocess.run(
            command,
            check=False,
            input=input_text,
            text=True,
            cwd=cwd,
            env=None if environment is None else dict(environment),
        )
        result = CommandResult(command, completed.returncode, "", "")
        if check and result.returncode != 0:
            raise CommandError(result)
        return result
    return _execute_observed_command(
        command,
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


class _BoundedTextBuffer:
    def __init__(self, limit: int) -> None:
        if limit <= 0:
            raise ValueError("command stream limit must be positive")
        self.limit = limit
        self._chunks: deque[str] = deque()
        self._size = 0
        self.truncated = False

    def append(self, value: str) -> None:
        encoded = value.encode("utf-8")
        if not encoded:
            return
        if len(encoded) >= self.limit:
            self._chunks.clear()
            self._chunks.append(
                encoded[-self.limit :].decode("utf-8", errors="ignore")
            )
            self._size = len(self._chunks[0].encode("utf-8"))
            self.truncated = True
            return
        self._chunks.append(value)
        self._size += len(encoded)
        while self._size > self.limit and self._chunks:
            excess = self._size - self.limit
            first = self._chunks.popleft()
            first_encoded = first.encode("utf-8")
            if len(first_encoded) <= excess:
                self._size -= len(first_encoded)
                self.truncated = True
                continue
            retained = first_encoded[excess:].decode(
                "utf-8", errors="ignore"
            )
            self._chunks.appendleft(retained)
            self._size -= len(first_encoded) - len(retained.encode("utf-8"))
            self.truncated = True
            break

    def text(self) -> str:
        return "".join(self._chunks)


@dataclass
class _ObservationState:
    error: Exception | None = None


def _execute_observed_command(
    command: tuple[str, ...],
    *,
    observer: CommandObserver | None,
    check: bool,
    input_text: str | None,
    cwd: Path | None,
    environment: Mapping[str, str] | None,
    observe_stdout: bool,
    observe_stderr: bool,
    stdout_limit: int,
    stderr_limit: int,
) -> CommandResult:
    effective_environment = dict(
        os.environ if environment is None else environment
    )
    if observer is not None:
        observer.command_started(
            command,
            live=True,
            environment=effective_environment,
        )
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=None if environment is None else dict(environment),
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout = _BoundedTextBuffer(stdout_limit)
    stderr = _BoundedTextBuffer(stderr_limit)
    observation = _ObservationState()
    observation_lock = threading.Lock()
    readers = (
        threading.Thread(
            target=_drain_stream,
            args=(
                process.stdout,
                CommandStream.STDOUT,
                observe_stdout,
                observer,
                stdout,
                observation,
                observation_lock,
            ),
            daemon=True,
        ),
        threading.Thread(
            target=_drain_stream,
            args=(
                process.stderr,
                CommandStream.STDERR,
                observe_stderr,
                observer,
                stderr,
                observation,
                observation_lock,
            ),
            daemon=True,
        ),
    )
    for reader in readers:
        reader.start()
    try:
        if input_text is not None:
            assert process.stdin is not None
            process.stdin.write(input_text.encode("utf-8"))
            process.stdin.close()
        returncode = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        for reader in readers:
            reader.join()
    result = CommandResult(
        command,
        returncode,
        stdout.text(),
        stderr.text(),
        stdout.truncated,
        stderr.truncated,
    )
    if observation.error is not None:
        raise CommandObservationError(
            f"command observer failed: {observation.error}"
        ) from observation.error
    if observer is not None:
        try:
            observer.command_finished(result, live=True)
        except Exception as error:
            raise CommandObservationError(
                f"command observer failed: {error}"
            ) from error
    if check and result.returncode != 0:
        raise CommandError(result)
    return result


def _drain_stream(
    pipe,
    stream: CommandStream,
    observed: bool,
    observer: CommandObserver | None,
    buffer: _BoundedTextBuffer,
    state: _ObservationState,
    state_lock: threading.Lock,
) -> None:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    try:
        while True:
            chunk = pipe.read1(MAX_RECORD_BYTES)
            if not chunk:
                break
            decoded = decoder.decode(chunk)
            if not observed or observer is None:
                buffer.append(decoded)
                continue
            pending += decoded
            records, pending = _complete_records(pending, final=False)
            for record in records:
                _observe_record(
                    observer,
                    stream,
                    record,
                    buffer,
                    state,
                    state_lock,
                )
        final = decoder.decode(b"", final=True)
        if not observed or observer is None:
            buffer.append(final)
            return
        pending += final
        records, pending = _complete_records(pending, final=True)
        for record in records:
            _observe_record(
                observer,
                stream,
                record,
                buffer,
                state,
                state_lock,
            )
        if pending:
            _observe_record(
                observer,
                stream,
                pending,
                buffer,
                state,
                state_lock,
            )
    finally:
        pipe.close()


def _complete_records(value: str, *, final: bool) -> tuple[list[str], str]:
    records: list[str] = []
    start = 0
    index = 0
    while index < len(value):
        character = value[index]
        if character == "\n":
            records.append(value[start : index + 1])
            start = index + 1
        elif character == "\r":
            if index + 1 == len(value) and not final:
                break
            end = index + 2 if value[index + 1 : index + 2] == "\n" else index + 1
            records.append(value[start:index] + "\n")
            start = end
            index = end - 1
        index += 1
    pending = value[start:]
    while len(pending.encode("utf-8")) > MAX_RECORD_BYTES:
        record, pending = _split_utf8_prefix(pending, MAX_RECORD_BYTES)
        records.append(record)
    if final and pending:
        records.append(pending)
        pending = ""
    return records, pending


def _split_utf8_prefix(value: str, maximum: int) -> tuple[str, str]:
    encoded = value.encode("utf-8")
    prefix = encoded[:maximum].decode("utf-8", errors="ignore")
    return prefix, value[len(prefix) :]


def _observe_record(
    observer: CommandObserver,
    stream: CommandStream,
    record: str,
    buffer: _BoundedTextBuffer,
    state: _ObservationState,
    state_lock: threading.Lock,
) -> None:
    with state_lock:
        if state.error is not None:
            return
        try:
            rendered = observer.command_output(stream, record)
        except Exception as error:
            state.error = error
            return
    buffer.append(rendered)


def _strip_wrappers(command: tuple[str, ...]) -> tuple[str, ...]:
    index = 0
    if command and Path(command[0]).name.casefold() == "sudo":
        index = 1
        value_options = {
            "-u",
            "--user",
            "-g",
            "--group",
            "-h",
            "--host",
            "-p",
            "--prompt",
            "-c",
            "--close-from",
        }
        while index < len(command):
            value = command[index]
            if value == "--":
                index += 1
                break
            if not value.startswith("-"):
                break
            option = value.split("=", 1)[0].casefold()
            index += 1
            if option in value_options and "=" not in value:
                index += 1
    if index < len(command) and Path(command[index]).name.casefold() == "env":
        index += 1
        while index < len(command):
            value = command[index]
            if value in {"-i", "--ignore-environment"}:
                index += 1
                continue
            if value in {"-u", "--unset"}:
                index += 2
                continue
            if "=" in value and not value.startswith("="):
                index += 1
                continue
            break
    return command[index:]


def _uv_is_live(arguments: tuple[str, ...]) -> bool:
    if not arguments:
        return False
    first = arguments[0].casefold()
    if first in {"sync", "lock", "add", "remove"}:
        return True
    return (
        first == "pip"
        and len(arguments) > 1
        and arguments[1].casefold()
        in {"compile", "install", "sync", "download"}
    )


def _python_pip_is_live(arguments: tuple[str, ...]) -> bool:
    return (
        len(arguments) > 2
        and arguments[0:2] == ("-m", "pip")
        and arguments[2].casefold() in {"install", "download", "wheel"}
    )


def _docker_is_live(arguments: tuple[str, ...]) -> bool:
    subcommand, remainder = _docker_subcommand(arguments)
    if subcommand in {"pull", "build"}:
        return True
    if subcommand == "buildx":
        return bool(remainder) and remainder[0].casefold() == "build"
    if subcommand == "run":
        return _docker_run_is_live(remainder)
    return False


def _docker_subcommand(
    arguments: tuple[str, ...],
) -> tuple[str | None, tuple[str, ...]]:
    value_options = {
        "--config",
        "--context",
        "-c",
        "--host",
        "-h",
        "--log-level",
        "-l",
        "--tls",
        "--tlscacert",
        "--tlscert",
        "--tlskey",
    }
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if not value.startswith("-"):
            return value.casefold(), arguments[index + 1 :]
        option = value.split("=", 1)[0].casefold()
        index += 1
        if option in value_options and "=" not in value:
            index += 1
    return None, ()


def _docker_run_is_live(arguments: tuple[str, ...]) -> bool:
    value_options = {
        "--env",
        "-e",
        "--entrypoint",
        "--mount",
        "--name",
        "--user",
        "-u",
        "--volume",
        "-v",
        "--workdir",
        "-w",
    }
    flag_options = {"--rm", "--tty", "-t", "--interactive", "-i"}
    entrypoint: str | None = None
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value == "--":
            index += 1
            break
        if not value.startswith("-"):
            break
        option, separator, inline = value.partition("=")
        normalized = option.casefold()
        index += 1
        if normalized in flag_options:
            continue
        if normalized in value_options:
            consumed = inline if separator else (
                arguments[index] if index < len(arguments) else ""
            )
            if not separator:
                index += 1
            if normalized == "--entrypoint":
                entrypoint = Path(consumed).name.casefold()
            continue
        if not separator and index < len(arguments):
            index += 1
    if index >= len(arguments):
        return False
    workload = arguments[index + 1 :]
    if entrypoint is not None:
        if entrypoint in {"uv", "uvx"}:
            return True
        if re.fullmatch(r"pip(?:3(?:\.\d+)?)?", entrypoint):
            return True
        if re.fullmatch(r"python(?:3(?:\.\d+)?)?", entrypoint):
            return _python_pip_is_live(workload)
    if not workload:
        return False
    executable = Path(workload[0]).name.casefold()
    if executable in {"uv", "uvx"}:
        return _uv_is_live(workload[1:])
    if re.fullmatch(r"pip(?:3(?:\.\d+)?)?", executable):
        return bool(workload[1:]) and workload[1].casefold() in {
            "install",
            "download",
            "wheel",
        }
    if re.fullmatch(r"python(?:3(?:\.\d+)?)?", executable):
        return _python_pip_is_live(workload[1:])
    return False
