from __future__ import annotations

import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

from amd_ai.installer.progress import (
    ProgressError,
    SessionLog,
    sanitize_output,
)


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
    wall_clock = lambda: datetime(
        2026, 7, 10, 14, 25, 33, tzinfo=UTC
    )
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
