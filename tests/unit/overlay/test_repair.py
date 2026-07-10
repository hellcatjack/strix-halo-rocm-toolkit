from __future__ import annotations

import os
from pathlib import Path

import pytest

from amd_ai.overlay.models import (
    OverlayPaths,
    ProtectedComponent,
    ProtectedProfile,
)
from amd_ai.overlay.repair import repair_overlay
from amd_ai.overlay.transaction import (
    TransactionError,
    build_generation,
    initialize_overlay,
    resolve_current_generation,
)
from amd_ai.runner import CommandResult


class FakeRunner:
    def run(self, args, *, environment, cwd=None):
        return CommandResult(tuple(args), 0, "", "")


class FakeGenerationBuilder:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.input_text: str | None = None
        self.lock_text: str | None = None

    def build(self, paths, *, profile, input_text, lock_text) -> object:
        self.input_text = input_text
        self.lock_text = lock_text
        if self.error is not None:
            raise self.error
        return build_generation(
            paths,
            profile=profile,
            input_text=input_text,
            lock_text=lock_text,
            runner=FakeRunner(),
            verifier=lambda path: None,
            transaction_id="20260710T120001Z-b1c2d3e4",
            acquire_lock=False,
        )


def test_overlay_repair_moves_current_generation_and_replays_lock(
    tmp_path: Path,
) -> None:
    paths, profile, current = damaged_overlay(tmp_path)
    builder = FakeGenerationBuilder()

    result = repair_overlay(
        paths,
        profile=profile,
        reason_code="TORCH.SHADOWED",
        builder=builder,
    )

    assert result.quarantine.name.endswith("-TORCH.SHADOWED")
    assert (
        result.quarantine / "generation" / "overlay.requirements.lock"
    ).is_file()
    assert builder.lock_text == ""
    assert resolve_current_generation(paths) == result.new_generation
    assert not current.exists()


def test_failed_overlay_rebuild_keeps_quarantine_and_project_blocked(
    tmp_path: Path,
) -> None:
    paths, profile, _ = damaged_overlay(tmp_path)
    builder = FakeGenerationBuilder(error=TransactionError("install failed"))

    with pytest.raises(TransactionError, match="install failed"):
        repair_overlay(
            paths,
            profile=profile,
            reason_code="TORCH.SHADOWED",
            builder=builder,
        )

    assert paths.current.is_symlink()
    assert not paths.current.exists()
    assert any(paths.quarantine.iterdir())


def test_overlay_repair_retries_existing_quarantine(
    tmp_path: Path,
) -> None:
    paths, profile, _ = damaged_overlay(tmp_path)
    with pytest.raises(TransactionError):
        repair_overlay(
            paths,
            profile=profile,
            reason_code="TORCH.SHADOWED",
            builder=FakeGenerationBuilder(
                error=TransactionError("first failed")
            ),
        )

    result = repair_overlay(
        paths,
        profile=profile,
        reason_code="TORCH.SHADOWED",
        builder=FakeGenerationBuilder(),
    )

    assert resolve_current_generation(paths) == result.new_generation


def damaged_overlay(
    tmp_path: Path,
) -> tuple[OverlayPaths, ProtectedProfile, Path]:
    project = tmp_path / "demo"
    project.mkdir()
    paths = OverlayPaths.for_project(project)
    profile = protected_profile()
    initialize_overlay(
        paths,
        profile=profile,
        transaction_id="20260710T120000Z-a1b2c3d4",
    )
    current = resolve_current_generation(paths)
    (current / "site-packages/torch.py").write_text(
        "shadow = True\n", encoding="utf-8"
    )
    assert os.readlink(paths.current).endswith(current.name)
    return paths, profile, current


def protected_profile() -> ProtectedProfile:
    return ProtectedProfile(
        "stable",
        "sha256:" + "a" * 64,
        (
            ProtectedComponent("torch", "2.9.1+rocm7.2.1.a"),
            ProtectedComponent("torchvision", "0.24.0+rocm7.2.1.b"),
            ProtectedComponent("torchaudio", "2.9.0+rocm7.2.1.c"),
            ProtectedComponent("triton", "3.5.1+rocm7.2.1.d"),
        ),
    )
