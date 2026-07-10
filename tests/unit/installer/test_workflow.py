from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

from amd_ai.installer.models import InstallMode, InstallOptions, InstallStage
from amd_ai.installer.state import install_lock, load_state, save_state
from amd_ai.installer.workflow import InstallerWorkflow
from tests.unit.installer.fakes import FakeInstallerActions


def workflow_options(
    tmp_path: Path,
    *,
    project_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> InstallOptions:
    return InstallOptions(
        mode=InstallMode.CONTAINER,
        project_dir=(project_dir or tmp_path / "project"),
        project_name="demo",
        image_source="pull",
        target_user="developer",
        source_root=Path.cwd(),
        stable_manifest_path=(
            manifest_path
            or Path("tests/fixtures/releases/stable.json").resolve()
        ),
        state_path=tmp_path / "install-state.json",
    )


def installer_workflow(
    tmp_path: Path,
    *,
    actions: FakeInstallerActions,
    options: InstallOptions | None = None,
) -> InstallerWorkflow:
    return InstallerWorkflow(
        options=options or workflow_options(tmp_path),
        actions=actions,
        installer_version="0.2.0",
        installer_source_revision="d" * 40,
    )


def test_container_workflow_runs_each_stage_once_and_completes(
    tmp_path: Path,
) -> None:
    actions = FakeInstallerActions.healthy()

    result = installer_workflow(tmp_path, actions=actions).run()

    assert result.exit_code == 0
    assert actions.calls == [
        "bootstrap",
        "container_host_check",
        "resolve_release",
        "pull_release",
        "verify_torch_image",
        "initialize_project",
        "verify_project",
    ]
    assert result.state is not None
    assert result.state.current_stage == InstallStage.COMPLETE


def test_resume_skips_only_stages_with_matching_input_digest(
    tmp_path: Path,
) -> None:
    actions = FakeInstallerActions.stop_after(InstallStage.IMAGE_VERIFY)
    first = installer_workflow(tmp_path, actions=actions)

    first_result = first.run()
    resumed_actions = FakeInstallerActions.healthy()
    resumed = installer_workflow(tmp_path, actions=resumed_actions).run()

    assert first_result.exit_code == 1
    assert resumed.exit_code == 0
    assert resumed_actions.calls == ["initialize_project", "verify_project"]


def test_release_manifest_change_blocks_resume_before_actions(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "stable.json"
    shutil.copyfile("tests/fixtures/releases/stable.json", manifest)
    options = workflow_options(tmp_path, manifest_path=manifest)
    first_actions = FakeInstallerActions.stop_after(InstallStage.RELEASE_RESOLVE)
    first = installer_workflow(
        tmp_path, actions=first_actions, options=options
    ).run()
    assert first.exit_code == 1
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    resumed_actions = FakeInstallerActions.healthy()
    result = installer_workflow(
        tmp_path, actions=resumed_actions, options=options
    ).run()

    assert result.exit_code == 2
    assert "inputs changed" in result.message
    assert resumed_actions.calls == []


def test_project_path_change_blocks_resume(tmp_path: Path) -> None:
    first_actions = FakeInstallerActions.stop_after(InstallStage.BOOTSTRAP)
    first = installer_workflow(tmp_path, actions=first_actions).run()
    assert first.exit_code == 1
    changed = workflow_options(tmp_path, project_dir=tmp_path / "another")

    resumed_actions = FakeInstallerActions.healthy()
    result = installer_workflow(
        tmp_path, actions=resumed_actions, options=changed
    ).run()

    assert result.exit_code == 2
    assert resumed_actions.calls == []


def test_action_failure_does_not_checkpoint_stage(tmp_path: Path) -> None:
    actions = FakeInstallerActions.healthy()
    actions.failures[InstallStage.PROJECT_INIT] = RuntimeError("build failed")

    result = installer_workflow(tmp_path, actions=actions).run()

    assert result.exit_code == 2
    assert result.state is not None
    assert InstallStage.PROJECT_INIT.value not in (
        result.state.completed_stage_input_digests
    )
    assert result.state.current_stage is InstallStage.PROJECT_INIT


def test_keyboard_interrupt_does_not_checkpoint_stage(tmp_path: Path) -> None:
    actions = FakeInstallerActions.healthy()
    actions.failures[InstallStage.IMAGE_VERIFY] = KeyboardInterrupt()

    result = installer_workflow(tmp_path, actions=actions).run()

    assert result.exit_code == 1
    assert result.state is not None
    assert InstallStage.IMAGE_VERIFY.value not in (
        result.state.completed_stage_input_digests
    )


def test_concurrent_installer_is_refused_without_actions(tmp_path: Path) -> None:
    actions = FakeInstallerActions.healthy()
    options = workflow_options(tmp_path)

    with install_lock(options.state_path):
        result = installer_workflow(
            tmp_path, actions=actions, options=options
        ).run()

    assert result.exit_code == 2
    assert actions.calls == []


def test_illegal_persisted_transition_is_blocked(tmp_path: Path) -> None:
    actions = FakeInstallerActions.healthy()
    complete = installer_workflow(tmp_path, actions=actions).run()
    assert complete.state is not None
    save_state(
        workflow_options(tmp_path).state_path,
        replace(complete.state, current_stage=InstallStage.IMAGE_VERIFY),
    )

    result = installer_workflow(
        tmp_path, actions=FakeInstallerActions.healthy()
    ).run()

    assert result.exit_code == 2
    assert "transition" in result.message


def test_image_disk_shortage_blocks_before_pull(tmp_path: Path) -> None:
    actions = FakeInstallerActions.healthy()
    actions.image_estimate = replace(
        actions.image_estimate,
        payload_bytes=20 * 1024**3,
        available_bytes=24 * 1024**3,
    )

    result = installer_workflow(tmp_path, actions=actions).run()

    assert result.exit_code == 2
    assert "required_bytes=" in result.message
    assert "pull_release" not in actions.calls


def test_project_disk_shortage_blocks_before_project_mutation(
    tmp_path: Path,
) -> None:
    actions = FakeInstallerActions.healthy()
    actions.project_estimate = replace(
        actions.project_estimate,
        payload_bytes=4 * 1024**3,
        available_bytes=8 * 1024**3,
    )

    result = installer_workflow(tmp_path, actions=actions).run()

    assert result.exit_code == 2
    assert "initialize_project" not in actions.calls
    assert result.state is not None
    assert result.state.current_stage is InstallStage.PROJECT_INIT


def test_disk_estimate_failure_is_reported_without_traceback(
    tmp_path: Path,
) -> None:
    actions = FakeInstallerActions.healthy()

    def fail_estimate(**kwargs: object):
        del kwargs
        raise RuntimeError("cannot inspect Docker root")

    actions.image_disk_estimate = fail_estimate  # type: ignore[method-assign]

    result = installer_workflow(tmp_path, actions=actions).run()

    assert result.exit_code == 2
    assert "cannot inspect Docker root" in result.message
    assert "pull_release" not in actions.calls
