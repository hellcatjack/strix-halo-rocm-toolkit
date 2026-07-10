from __future__ import annotations

from pathlib import Path

import pytest

from amd_ai.overlay.models import (
    OverlayError,
    OverlayPaths,
    ProtectedComponent,
    ProtectedProfile,
)


def test_overlay_paths_are_project_local(tmp_path: Path) -> None:
    project = tmp_path / "demo"
    project.mkdir()

    paths = OverlayPaths.for_project(project)

    assert paths.root == project / ".amd-ai"
    assert paths.current == paths.root / "current"
    assert paths.generation("20260710T120000Z-a1b2c3d4") == (
        paths.generations / "20260710T120000Z-a1b2c3d4"
    )


def test_overlay_paths_reject_symlinked_control_root(tmp_path: Path) -> None:
    project = tmp_path / "demo"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / ".amd-ai").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OverlayError, match="symbolic link"):
        OverlayPaths.for_project(project)


def test_protected_profile_requires_all_exact_full_versions() -> None:
    profile = ProtectedProfile(
        profile_id="rocm-7.2.1-py3.12-torch-2.9.1",
        parent_config_digest="sha256:" + "a" * 64,
        components=(
            ProtectedComponent("torch", "2.9.1+rocm7.2.1.lw.gitff65f5bc"),
            ProtectedComponent("torchvision", "0.24.0+rocm7.2.1.gitb919bd0c"),
            ProtectedComponent("torchaudio", "2.9.0+rocm7.2.1.gite3c6ee2b"),
            ProtectedComponent("triton", "3.5.1+rocm7.2.1.gita272dfa8"),
        ),
    )

    assert profile.version_for("Torch_Vision") == (
        "0.24.0+rocm7.2.1.gitb919bd0c"
    )


def test_protected_profile_rejects_public_only_version() -> None:
    with pytest.raises(OverlayError, match="full local version"):
        ProtectedComponent("torch", "2.9.1")
