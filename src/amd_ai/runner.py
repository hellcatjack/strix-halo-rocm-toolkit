from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


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
