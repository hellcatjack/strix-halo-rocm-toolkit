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
        InstallStage.KERNEL_PLAN,
        InstallStage.KERNEL_CONFIRM,
        InstallStage.KERNEL_APPLY,
        InstallStage.KERNEL_REBOOT_PENDING,
        InstallStage.KERNEL_VERIFY,
        InstallStage.HOST_PLAN,
        InstallStage.HOST_CONFIRM,
        InstallStage.HOST_APPLY,
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


def test_noninteractive_full_mode_defers_missing_plan_digests_to_confirm_stages(
    tmp_path: Path,
) -> None:
    options = InstallOptions(
        mode=InstallMode.FULL,
        non_interactive=True,
        project_dir=(tmp_path / "demo").resolve(),
        image_source="pull",
        accepted_kernel_plan_digest=None,
        accepted_host_plan_digest=None,
    ).validate()

    assert options.accepted_kernel_plan_digest is None
    assert options.accepted_host_plan_digest is None


@pytest.mark.parametrize("registry", ("auto", "swr", "ghcr"))
def test_registry_choice_is_valid(registry: str, tmp_path: Path) -> None:
    options = InstallOptions(
        mode=InstallMode.CONTAINER,
        project_dir=tmp_path / "project",
        image_source="pull",
        registry=registry,
    ).validate()

    assert options.registry == registry


def test_unknown_registry_choice_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(InstallerModelError, match="registry"):
        InstallOptions(
            mode=InstallMode.CONTAINER,
            project_dir=tmp_path / "project",
            image_source="pull",
            registry="nearest",
        ).validate()


@pytest.mark.parametrize("registry", ("swr", "ghcr"))
def test_local_build_rejects_explicit_registry(
    tmp_path: Path,
    registry: str,
) -> None:
    with pytest.raises(InstallerModelError, match="does not use a registry"):
        InstallOptions(
            mode=InstallMode.CONTAINER,
            project_dir=tmp_path / "project",
            image_source="build",
            registry=registry,
        ).validate()


def test_state_path_provenance_must_be_boolean(tmp_path: Path) -> None:
    with pytest.raises(InstallerModelError, match="state path provenance"):
        InstallOptions(
            mode=InstallMode.CONTAINER,
            project_dir=tmp_path / "project",
            state_path=tmp_path / "state.json",
            state_path_explicit="no",  # type: ignore[arg-type]
        ).validate()


def test_coordination_path_is_independent_from_explicit_state_path(
    tmp_path: Path,
) -> None:
    first = InstallOptions(
        mode=InstallMode.CONTAINER,
        project_dir=tmp_path / "first-project",
        state_path=tmp_path / "one" / "state.json",
    )
    second = InstallOptions(
        mode=InstallMode.CONTAINER,
        project_dir=tmp_path / "second-project",
        state_path=tmp_path / "two" / "state.json",
    )

    assert first.coordination_state_path == second.coordination_state_path
