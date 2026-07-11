from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from amd_ai.installer.models import InstallMode


HOME_MENU = """Strix Halo ROCm Toolkit

1. 完整工作站安装
2. 仅安装容器平台
3. 检查或修复已有安装
4. 退出"""

IMAGE_FALLBACK_MENU = """1. 使用当前源码在本机构建
2. 退出"""

STATUS_PREFIXES = frozenset(
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


class PromptError(RuntimeError):
    pass


class PromptRequired(PromptError):
    pass


class PromptRefused(PromptError):
    pass


class TerminalPrompts:
    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        is_tty: bool | None = None,
    ) -> None:
        self._input_fn = input_fn
        self._output_fn = output_fn
        self._is_tty = (
            sys.stdin.isatty() and sys.stdout.isatty()
            if is_tty is None
            else is_tty
        )

    def choose_mode(self) -> InstallMode | None:
        self._require_terminal("install mode selection")
        self._output_fn(HOME_MENU)
        choices = {
            "1": InstallMode.FULL,
            "2": InstallMode.CONTAINER,
            "3": InstallMode.DOCTOR,
            "4": None,
        }
        while True:
            answer = self._read("请选择 [1-4]: ")
            if answer is None:
                raise PromptRefused("install mode selection was refused")
            if answer in choices:
                return choices[answer]
            self._output_fn(render_status("BLOCKED", "请输入 1、2、3 或 4"))

    def choose_image_fallback(self) -> str | None:
        self._require_terminal("image fallback selection")
        self._output_fn(IMAGE_FALLBACK_MENU)
        while True:
            answer = self._read("请选择 [1-2]: ")
            if answer is None:
                raise PromptRefused("image fallback selection was refused")
            if answer == "1":
                return "build"
            if answer == "2":
                return None
            self._output_fn(render_status("BLOCKED", "请输入 1 或 2"))

    def ask_project_dir(self) -> Path:
        self._require_terminal("project directory input")
        while True:
            answer = self._read("项目目录: ")
            if answer is None:
                raise PromptRefused("project directory input was refused")
            if "\0" in answer:
                self._output_fn(
                    render_status("BLOCKED", "项目目录包含无效字符")
                )
                continue
            try:
                return Path(answer).expanduser().resolve(strict=False)
            except (OSError, RuntimeError):
                self._output_fn(
                    render_status("BLOCKED", "项目目录无法规范化")
                )

    def confirm_exact(self, word: str) -> bool:
        if not self._is_tty or not word or any(character.isspace() for character in word):
            return False
        answer = self._read(f"输入 {word} 继续: ")
        return answer == word

    def confirm_yes_no(self, question: str) -> bool:
        if not self._is_tty:
            return False
        answer = self._read(f"{question} [yes/no]: ")
        if answer is None:
            return False
        normalized = answer.casefold()
        return normalized in {"y", "yes"}

    def status(self, prefix: str, message: str) -> None:
        self._output_fn(render_status(prefix, message))

    def _require_terminal(self, operation: str) -> None:
        if not self._is_tty:
            raise PromptRefused(f"{operation} requires stdin and stdout TTYs")

    def _read(self, text: str) -> str | None:
        try:
            value = self._input_fn(text)
        except (EOFError, KeyboardInterrupt):
            return None
        value = value.strip()
        return value or None


class NonInteractivePrompts:
    def __init__(
        self, *, output_fn: Callable[[str], None] = print
    ) -> None:
        self._output_fn = output_fn

    def choose_mode(self) -> InstallMode | None:
        raise PromptRequired("install mode must be supplied non-interactively")

    def choose_image_fallback(self) -> str | None:
        raise PromptRequired(
            "image source must be supplied non-interactively"
        )

    def ask_project_dir(self) -> Path:
        raise PromptRequired(
            "project directory must be supplied non-interactively"
        )

    def confirm_exact(self, word: str) -> bool:
        raise PromptRequired(
            f"{word} authorization cannot be prompted non-interactively"
        )

    def confirm_yes_no(self, question: str) -> bool:
        raise PromptRequired(
            f"{question} authorization cannot be prompted non-interactively"
        )

    def status(self, prefix: str, message: str) -> None:
        self._output_fn(render_status(prefix, message))


def render_status(prefix: str, message: str) -> str:
    if prefix not in STATUS_PREFIXES:
        raise ValueError(f"unsupported installer status prefix: {prefix}")
    if not isinstance(message, str) or "\0" in message:
        raise ValueError("installer status message is invalid")
    return f"{prefix:<8} {message}"
