from __future__ import annotations

import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


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
        super().__init__(f"command failed ({result.returncode}): {' '.join(result.args)}")
        self.result = result


class Runner(Protocol):
    def run(
        self,
        args: list[str],
        *,
        check: bool = True,
        input_text: str | None = None,
    ) -> CommandResult: ...


class SubprocessRunner:
    def run(
        self,
        args: list[str],
        *,
        check: bool = True,
        input_text: str | None = None,
    ) -> CommandResult:
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
        if check and result.returncode != 0:
            raise CommandError(result)
        return result
