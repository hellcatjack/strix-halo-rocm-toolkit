from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

from amd_ai.host.models import PlannedAction
from amd_ai.installer.actions import prepare_plan_payload
from amd_ai.installer.models import InstallMode, InstallOptions, InstallStage
from amd_ai.installer.release import (
    ReleaseAcquisitionError,
    ReleaseIdentityError,
)
from amd_ai.installer.state import (
    install_lock,
    save_state,
    stage_input_digest,
)
from amd_ai.installer.workflow import InstallerWorkflow
from tests.unit.installer.fakes import FakeInstallerActions, FakePrompts


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
    prompts: FakePrompts | None = None,
    boot_id_reader=None,
) -> InstallerWorkflow:
    kwargs = {}
    if boot_id_reader is not None:
        kwargs["boot_id_reader"] = boot_id_reader
    return InstallerWorkflow(
        options=options or workflow_options(tmp_path),
        actions=actions,
        installer_version="0.2.0",
        installer_source_revision="d" * 40,
        prompts=prompts,
        **kwargs,
    )


def full_options(
    tmp_path: Path,
    *,
    non_interactive: bool = False,
    accepted_host_plan_digest: str | None = None,
    accept_docker_group: bool = False,
) -> InstallOptions:
    return InstallOptions(
        mode=InstallMode.FULL,
        non_interactive=non_interactive,
        project_dir=tmp_path / "project",
        project_name="demo",
        image_source="pull",
        target_user="developer",
        accepted_host_plan_digest=accepted_host_plan_digest,
        accept_docker_group=accept_docker_group,
        source_root=Path.cwd(),
        stable_manifest_path=Path(
            "tests/fixtures/releases/stable.json"
        ).resolve(),
        state_path=tmp_path / "install-state.json",
    )


def noninteractive_container_options(
    tmp_path: Path, *, image_source: str
) -> InstallOptions:
    return replace(
        workflow_options(tmp_path),
        non_interactive=True,
        image_source=image_source,
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


def test_full_mode_requires_exact_apply_and_separate_docker_group_prompt(
    tmp_path: Path,
) -> None:
    prompts = FakePrompts(
        exact={"APPLY": True}, yes_no={"docker-group": False}
    )
    actions = FakeInstallerActions.host_change_requires_reboot()

    result = installer_workflow(
        tmp_path,
        actions=actions,
        options=full_options(tmp_path),
        prompts=prompts,
        boot_id_reader=lambda: "12345678-1234-4abc-8def-1234567890ab",
    ).run()

    assert result.exit_code == 1
    assert result.state is not None
    assert result.state.current_stage == InstallStage.REBOOT_PENDING
    assert actions.host_apply_include_docker_group is False
    assert any(
        prefix == "ACTION" and "HOST.CHANGE" in message
        for prefix, message in prompts.statuses
    )


def test_noninteractive_full_mode_requires_matching_plan_digest(
    tmp_path: Path,
) -> None:
    actions = FakeInstallerActions.host_change_requires_reboot()
    options = full_options(
        tmp_path,
        non_interactive=True,
        accepted_host_plan_digest="0" * 64,
    )

    result = installer_workflow(
        tmp_path,
        actions=actions,
        options=options,
        boot_id_reader=lambda: "12345678-1234-4abc-8def-1234567890ab",
    ).run()

    assert result.exit_code == 2
    assert "host plan digest" in result.message
    assert "host_apply" not in actions.calls


def test_full_mode_docker_group_authorization_reaches_apply(
    tmp_path: Path,
) -> None:
    prompts = FakePrompts(
        exact={"APPLY": True}, yes_no={"docker-group": True}
    )
    actions = FakeInstallerActions.host_change_requires_reboot()

    installer_workflow(
        tmp_path,
        actions=actions,
        options=full_options(tmp_path),
        prompts=prompts,
        boot_id_reader=lambda: "12345678-1234-4abc-8def-1234567890ab",
    ).run()

    assert actions.host_apply_include_docker_group is True


def test_same_boot_remains_pending_without_reapplying_host(
    tmp_path: Path,
) -> None:
    boot_id = "12345678-1234-4abc-8def-1234567890ab"
    first_actions = FakeInstallerActions.host_change_requires_reboot()
    first = installer_workflow(
        tmp_path,
        actions=first_actions,
        options=full_options(tmp_path),
        prompts=FakePrompts(exact={"APPLY": True}),
        boot_id_reader=lambda: boot_id,
    ).run()
    assert first.exit_code == 1

    resumed_actions = FakeInstallerActions.host_change_requires_reboot()
    resumed = installer_workflow(
        tmp_path,
        actions=resumed_actions,
        options=full_options(tmp_path),
        prompts=FakePrompts(),
        boot_id_reader=lambda: boot_id,
    ).run()

    assert resumed.exit_code == 1
    assert "host_apply" not in resumed_actions.calls
    assert "host_verify" not in resumed_actions.calls


def test_changed_boot_resumes_at_host_verify_and_completes(
    tmp_path: Path,
) -> None:
    first_boot = "12345678-1234-4abc-8def-1234567890ab"
    second_boot = "87654321-4321-4abc-8def-ba0987654321"
    first = installer_workflow(
        tmp_path,
        actions=FakeInstallerActions.host_change_requires_reboot(),
        options=full_options(tmp_path),
        prompts=FakePrompts(exact={"APPLY": True}),
        boot_id_reader=lambda: first_boot,
    ).run()
    assert first.exit_code == 1

    resumed_actions = FakeInstallerActions.host_change_requires_reboot()
    resumed = installer_workflow(
        tmp_path,
        actions=resumed_actions,
        options=full_options(tmp_path),
        prompts=FakePrompts(),
        boot_id_reader=lambda: second_boot,
    ).run()

    assert resumed.exit_code == 0
    assert "host_apply" not in resumed_actions.calls
    assert resumed_actions.calls[0] == "host_verify"


def test_release_must_support_applied_host_adapter(tmp_path: Path) -> None:
    first_boot = "12345678-1234-4abc-8def-1234567890ab"
    second_boot = "87654321-4321-4abc-8def-ba0987654321"
    first_actions = FakeInstallerActions.host_change_requires_reboot()
    first = installer_workflow(
        tmp_path,
        actions=first_actions,
        options=full_options(tmp_path),
        prompts=FakePrompts(exact={"APPLY": True}),
        boot_id_reader=lambda: first_boot,
    ).run()
    assert first.exit_code == 1

    resumed_actions = FakeInstallerActions.host_change_requires_reboot()
    resumed_actions.release = replace(
        resumed_actions.release, supported_host_adapter_ids=("other",)
    )
    result = installer_workflow(
        tmp_path,
        actions=resumed_actions,
        options=full_options(tmp_path),
        prompts=FakePrompts(),
        boot_id_reader=lambda: second_boot,
    ).run()

    assert result.exit_code == 2
    assert "adapter" in result.message
    assert "pull_release" not in resumed_actions.calls


def test_interactive_apply_refusal_never_runs_host_apply(
    tmp_path: Path,
) -> None:
    actions = FakeInstallerActions.host_change_requires_reboot()

    result = installer_workflow(
        tmp_path,
        actions=actions,
        options=full_options(tmp_path),
        prompts=FakePrompts(exact={"APPLY": False}),
        boot_id_reader=lambda: "12345678-1234-4abc-8def-1234567890ab",
    ).run()

    assert result.exit_code == 2
    assert "exact APPLY" in result.message
    assert "host_apply" not in actions.calls


def test_noninteractive_matching_digest_can_explicitly_accept_docker_group(
    tmp_path: Path,
) -> None:
    actions = FakeInstallerActions.host_change_requires_reboot()
    options = full_options(
        tmp_path,
        non_interactive=True,
        accepted_host_plan_digest=actions.host_plan_result.plan_digest,
        accept_docker_group=True,
    )

    result = installer_workflow(
        tmp_path,
        actions=actions,
        options=options,
        boot_id_reader=lambda: "12345678-1234-4abc-8def-1234567890ab",
    ).run()

    assert result.exit_code == 1
    assert actions.host_apply_include_docker_group is True


def test_changed_host_plan_after_checkpoint_requires_replanning(
    tmp_path: Path,
) -> None:
    first_actions = FakeInstallerActions.stop_after(InstallStage.HOST_PLAN)
    first = installer_workflow(
        tmp_path,
        actions=first_actions,
        options=full_options(tmp_path),
        prompts=FakePrompts(),
        boot_id_reader=lambda: "12345678-1234-4abc-8def-1234567890ab",
    ).run()
    assert first.exit_code == 1

    resumed_actions = FakeInstallerActions.host_change_requires_reboot()
    original = resumed_actions.host_plan_result
    changed_plan = replace(
        original.plan,
        actions=original.plan.actions
        + (
            PlannedAction(
                code="HOST.NEW_CHANGE",
                summary="New host change",
                argv=("true",),
                privileged=True,
            ),
        ),
    )
    resumed_actions.host_plan_result = replace(
        original,
        plan=changed_plan,
        plan_digest=stage_input_digest(prepare_plan_payload(changed_plan)),
    )

    result = installer_workflow(
        tmp_path,
        actions=resumed_actions,
        options=full_options(tmp_path),
        prompts=FakePrompts(exact={"APPLY": True}),
        boot_id_reader=lambda: "12345678-1234-4abc-8def-1234567890ab",
    ).run()

    assert result.exit_code == 2
    assert "host plan digest changed" in result.message
    assert "host_apply" not in resumed_actions.calls


def test_interactive_pull_failure_requires_explicit_build_choice(
    tmp_path: Path,
) -> None:
    actions = FakeInstallerActions.healthy()
    actions.pull_error = ReleaseAcquisitionError("registry unavailable")
    prompts = FakePrompts(image_fallback="build")

    result = installer_workflow(
        tmp_path, actions=actions, prompts=prompts
    ).run()

    assert result.exit_code == 0
    assert "build_local_images" in actions.calls
    assert result.state is not None
    assert result.state.release_id == "local"


def test_noninteractive_pull_failure_never_implicitly_builds(
    tmp_path: Path,
) -> None:
    actions = FakeInstallerActions.healthy()
    actions.pull_error = ReleaseAcquisitionError("registry unavailable")

    result = installer_workflow(
        tmp_path,
        actions=actions,
        options=noninteractive_container_options(
            tmp_path, image_source="pull"
        ),
    ).run()

    assert result.exit_code == 2
    assert "build_local_images" not in actions.calls


def test_pulled_identity_mismatch_never_offers_local_build(
    tmp_path: Path,
) -> None:
    actions = FakeInstallerActions.healthy()
    actions.pull_error = ReleaseIdentityError("config digest changed")
    prompts = FakePrompts(image_fallback="build")

    result = installer_workflow(
        tmp_path, actions=actions, prompts=prompts
    ).run()

    assert result.exit_code == 2
    assert "identity" in result.message
    assert "build_local_images" not in actions.calls


def test_noninteractive_build_source_never_attempts_pull(
    tmp_path: Path,
) -> None:
    actions = FakeInstallerActions.healthy()

    result = installer_workflow(
        tmp_path,
        actions=actions,
        options=noninteractive_container_options(
            tmp_path, image_source="build"
        ),
    ).run()

    assert result.exit_code == 0
    assert "pull_release" not in actions.calls
    assert "build_local_images" in actions.calls


def test_project_initialization_receives_selected_exact_parent_identity(
    tmp_path: Path,
) -> None:
    actions = FakeInstallerActions.healthy()

    result = installer_workflow(tmp_path, actions=actions).run()

    assert result.exit_code == 0
    assert actions.project_init_kwargs["base_image_reference"] == (
        actions.release.torch.reference
    )
    assert actions.project_init_kwargs["base_config_digest"] == (
        actions.release.torch.config_digest
    )


def test_interactive_build_fallback_rechecks_build_disk_requirement(
    tmp_path: Path,
) -> None:
    actions = FakeInstallerActions.healthy()
    actions.pull_error = ReleaseAcquisitionError("registry unavailable")
    actions.build_image_estimate = replace(
        actions.build_image_estimate,
        payload_bytes=40 * 1024**3,
        available_bytes=44 * 1024**3,
    )

    result = installer_workflow(
        tmp_path,
        actions=actions,
        prompts=FakePrompts(image_fallback="build"),
    ).run()

    assert result.exit_code == 2
    assert "image build disk space" in result.message
    assert "build_local_images" not in actions.calls
