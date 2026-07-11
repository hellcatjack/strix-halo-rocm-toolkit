from __future__ import annotations

from pathlib import Path

import pytest

from amd_ai.installer.models import InstallMode
from amd_ai.installer.prompts import (
    NonInteractivePrompts,
    PromptRefused,
    PromptRequired,
    TerminalPrompts,
    render_status,
)


def test_numbered_choice_reprompts_then_returns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    answers = iter(["9", "2"])
    prompt = TerminalPrompts(input_fn=lambda text: next(answers), is_tty=True)

    assert prompt.choose_mode() == InstallMode.CONTAINER
    output = capsys.readouterr().out
    assert "Strix Halo ROCm Toolkit\n\n" in output
    assert "1. 完整工作站安装" in output
    assert "4. 退出" in output
    assert "BLOCKED" in output


def test_exact_confirmation_rejects_case_and_eof() -> None:
    assert (
        TerminalPrompts(
            input_fn=lambda text: "apply", is_tty=True
        ).confirm_exact("APPLY")
        is False
    )
    assert (
        TerminalPrompts(
            input_fn=lambda text: (_ for _ in ()).throw(EOFError),
            is_tty=True,
        ).confirm_exact("APPLY")
        is False
    )


def test_noninteractive_prompt_is_always_blocked() -> None:
    prompt = NonInteractivePrompts()

    with pytest.raises(PromptRequired):
        prompt.confirm_exact("APPLY")
    with pytest.raises(PromptRequired):
        prompt.choose_image_fallback()


def test_non_tty_and_empty_required_input_are_refused() -> None:
    with pytest.raises(PromptRefused):
        TerminalPrompts(input_fn=lambda text: "2", is_tty=False).choose_mode()
    with pytest.raises(PromptRefused):
        TerminalPrompts(input_fn=lambda text: "", is_tty=True).ask_project_dir()


def test_project_directory_is_normalized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    prompt = TerminalPrompts(input_fn=lambda text: "demo/../project", is_tty=True)

    assert prompt.ask_project_dir() == (tmp_path / "project").resolve()


def test_image_fallback_requires_explicit_number() -> None:
    answers = iter(["yes", "1"])
    prompt = TerminalPrompts(input_fn=lambda text: next(answers), is_tty=True)

    assert prompt.choose_image_fallback() == "build"


def test_status_renderer_accepts_only_approved_prefixes() -> None:
    assert render_status("PASS", "已满足") == "PASS     已满足"
    assert render_status("INFO", "状态路径") == "INFO     状态路径"
    assert render_status("WARN", "需要注意") == "WARN     需要注意"
    assert render_status("ACTION", "将执行修改") == "ACTION   将执行修改"
    assert render_status("BLOCKED", "不允许继续") == "BLOCKED  不允许继续"
    assert render_status("START", "开始") == "START    开始"
    assert render_status("WAIT", "等待") == "WAIT     等待"
    assert render_status("FAIL", "失败") == "FAIL     失败"
    assert render_status("SUMMARY", "完成") == "SUMMARY  完成"
    with pytest.raises(ValueError):
        render_status("UNKNOWN", "not approved")


def test_noninteractive_prompts_render_status_without_answering(
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt = NonInteractivePrompts()

    prompt.status("ACTION", "dry-run IMAGE_PULL_OR_BUILD")

    assert (
        capsys.readouterr().out
        == "ACTION   dry-run IMAGE_PULL_OR_BUILD\n"
    )
