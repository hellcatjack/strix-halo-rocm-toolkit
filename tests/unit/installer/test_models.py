from __future__ import annotations

from pathlib import Path

import pytest

from amd_ai.installer.models import (
    CONTAINER_STAGE_ORDER,
    FULL_STAGE_ORDER,
    InstallMode,
    InstallOptions,
    InstallStage,
    InstallerModelError,
)


def test_full_and_container_stage_orders_match_approved_workflow() -> None:
    assert FULL_STAGE_ORDER == (
        InstallStage.BOOTSTRAP,
        InstallStage.HOST_PREFLIGHT,
        InstallStage.HOST_PLAN,
        InstallStage.HOST_CONFIRM,
        InstallStage.HOST_APPLY,
        InstallStage.REBOOT_PENDING,
        InstallStage.HOST_VERIFY,
        InstallStage.RELEASE_RESOLVE,
        InstallStage.IMAGE_PULL_OR_BUILD,
        InstallStage.IMAGE_VERIFY,
        InstallStage.PROJECT_INIT,
        InstallStage.PROJECT_VERIFY,
        InstallStage.COMPLETE,
    )
    assert InstallStage.HOST_APPLY not in CONTAINER_STAGE_ORDER
    assert InstallStage.CONTAINER_HOST_CHECK in CONTAINER_STAGE_ORDER


def test_noninteractive_options_require_project_and_explicit_image_source(
    tmp_path: Path,
) -> None:
    with pytest.raises(InstallerModelError):
        InstallOptions(
            mode=InstallMode.CONTAINER,
            non_interactive=True,
            project_dir=None,
            image_source=None,
        ).validate()


def test_noninteractive_full_mode_requires_exact_plan_digest(
    tmp_path: Path,
) -> None:
    with pytest.raises(InstallerModelError, match="plan"):
        InstallOptions(
            mode=InstallMode.FULL,
            non_interactive=True,
            project_dir=(tmp_path / "demo").resolve(),
            image_source="pull",
            accepted_host_plan_digest=None,
        ).validate()
