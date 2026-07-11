from __future__ import annotations

import io
import json
import shutil
from dataclasses import replace
from pathlib import Path

from amd_ai.host.models import PlannedAction
from amd_ai.installer.actions import prepare_plan_payload
from amd_ai.installer.models import (
    CONTAINER_STAGE_ORDER,
    InstallMode,
    InstallOptions,
    InstallStage,
)
from amd_ai.installer.progress import (
    InstallerProgress,
    ProgressError,
    ProgressMode,
)
from amd_ai.installer.release import (
    ReleaseAcquisitionError,
    ReleaseIdentityError,
)
from amd_ai.installer.state import (
    installer_coordination_lock,
    install_lock,
    load_state,
    project_state_path,
    save_state,
    stage_input_digest,
)
from amd_ai.installer.workflow import InstallerWorkflow
from amd_ai.report import Finding, Report, Severity, Status
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
        coordination_state_path=tmp_path / "coordination-state.json",
    )


def installer_workflow(
    tmp_path: Path,
    *,
    actions: FakeInstallerActions,
    options: InstallOptions | None = None,
    prompts: FakePrompts | None = None,
    boot_id_reader=None,
    installer_version: str = "0.2.0",
    installer_source_revision: str = "d" * 40,
    progress: InstallerProgress | None = None,
) -> InstallerWorkflow:
    kwargs = {}
    if boot_id_reader is not None:
        kwargs["boot_id_reader"] = boot_id_reader
    return InstallerWorkflow(
        options=options or workflow_options(tmp_path),
        actions=actions,
        installer_version=installer_version,
        installer_source_revision=installer_source_revision,
        prompts=prompts,
        progress=progress,
        **kwargs,
    )


def workflow_progress(
    tmp_path: Path,
    *,
    mode: ProgressMode = ProgressMode.DEFAULT,
    process_id: int = 23,
) -> tuple[InstallerProgress, io.StringIO, io.StringIO]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    progress = InstallerProgress(
        mode=mode,
        stdout=stdout,
        stderr=stderr,
        log_root=tmp_path / "logs",
        process_id=process_id,
    )
    return progress, stdout, stderr


def host_verify_report(status: Status) -> Report:
    findings = ()
    if status is Status.UNVERIFIED:
        findings = (
            Finding(
                code="HOST.UPSTREAM_UNVERIFIED",
                severity=Severity.WARNING,
                summary="The OEM kernel is newer than the tested set",
                evidence="running kernel: 6.17.0-1028-oem",
                remediation="Run the full hardware qualification before promotion.",
            ),
        )
    elif status is Status.CHANGE_REQUIRED:
        findings = (
            Finding(
                code="GPU.PERMISSION",
                severity=Severity.WARNING,
                summary="GPU groups are missing",
                evidence="missing GID: 993",
                remediation="Start a new login session.",
            ),
        )
    return Report(
        command="host-verify",
        status=status,
        generated_at="2026-07-10T20:28:14Z",
        facts={"kernel": "6.17.0-1028-oem"},
        findings=findings,
    )


def returning_host_report(actions: FakeInstallerActions, report: Report) -> None:
    def host_verify(**kwargs: object) -> Report:
        assert kwargs == {"target_user": "developer"}
        actions.calls.append("host_verify")
        return report

    actions.host_verify = host_verify  # type: ignore[method-assign]


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
        coordination_state_path=tmp_path / "coordination-state.json",
    )


def noninteractive_container_options(
    tmp_path: Path, *, image_source: str
) -> InstallOptions:
    return replace(
        workflow_options(tmp_path),
        non_interactive=True,
        image_source=image_source,
    )


def implicit_workflow_options(
    tmp_path: Path,
    *,
    project_dir: Path,
    mode: InstallMode = InstallMode.CONTAINER,
) -> InstallOptions:
    options = (
        workflow_options(tmp_path, project_dir=project_dir)
        if mode is InstallMode.CONTAINER
        else replace(full_options(tmp_path), project_dir=project_dir)
    )
    return replace(options, state_path_explicit=False)


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


def test_resume_reports_plan_skips_start_disk_detail_and_checkpointed_pass(
    tmp_path: Path,
) -> None:
    first_actions = FakeInstallerActions.stop_after(
        InstallStage.RELEASE_RESOLVE
    )
    assert installer_workflow(
        tmp_path, actions=first_actions
    ).run().exit_code == 1
    progress, stdout, _ = workflow_progress(tmp_path)

    result = installer_workflow(
        tmp_path,
        actions=FakeInstallerActions.healthy(),
        progress=progress,
    ).run()

    assert result.exit_code == 0
    output = stdout.getvalue()
    assert "PLAN     共 8 个阶段，从 IMAGE_PULL_OR_BUILD 继续" in output
    assert "SKIP     [1/8] 安装用户运行时：已有可信检查点" in output
    assert output.index("START    [4/8]") < output.index(
        "DETAIL   缺失层=10.0 GiB"
    )
    assert output.index("DETAIL   缺失层=10.0 GiB") < output.index(
        "PASS     [4/8]"
    )
    persisted = load_state(tmp_path / "install-state.json")
    assert persisted is not None
    assert (
        InstallStage.IMAGE_PULL_OR_BUILD.value
        in persisted.completed_stage_input_digests
    )


def test_new_session_reports_pending_then_resolved_release_identity(
    tmp_path: Path,
) -> None:
    progress, stdout, _ = workflow_progress(
        tmp_path, mode=ProgressMode.VERBOSE
    )

    result = installer_workflow(
        tmp_path,
        actions=FakeInstallerActions.healthy(),
        progress=progress,
    ).run()

    assert result.exit_code == 0
    output = stdout.getvalue()
    pending = output.index("stable release=待解析")
    resolved = output.index("DETAIL   stable release=0.2.0")
    checkpointed = output.index("PASS     [3/8]")
    assert pending < resolved < checkpointed
    assert "ghcr.io/hellcatjack/strix-halo-rocm-python@sha256:" in output
    assert "sha256:" + "d" * 64 in output
    assert "ghcr.io/hellcatjack/strix-halo-rocm-pytorch@sha256:" in output
    assert "sha256:" + "7" * 64 in output


def test_disk_shortage_reports_start_detail_then_failure(
    tmp_path: Path,
) -> None:
    actions = FakeInstallerActions.healthy()
    actions.image_estimate = replace(
        actions.image_estimate,
        payload_bytes=20 * 1024**3,
        available_bytes=24 * 1024**3,
    )
    terminal = io.StringIO()
    progress = InstallerProgress(
        mode=ProgressMode.DEFAULT,
        stdout=terminal,
        stderr=terminal,
        log_root=tmp_path / "logs",
        process_id=23,
    )

    result = installer_workflow(
        tmp_path, actions=actions, progress=progress
    ).run()

    assert result.exit_code == 2
    output = terminal.getvalue()
    start = output.index("START    [4/8]")
    detail = output.index("DETAIL   缺失层=20.0 GiB")
    failure = output.index("FAIL     [4/8]")
    assert start < detail < failure
    assert "pull_release" not in actions.calls


def test_checkpointed_action_reports_pass_before_resume_instructions(
    tmp_path: Path,
) -> None:
    terminal = io.StringIO()
    progress = InstallerProgress(
        mode=ProgressMode.DEFAULT,
        stdout=terminal,
        stderr=terminal,
        log_root=tmp_path / "logs",
        process_id=23,
    )

    result = installer_workflow(
        tmp_path,
        actions=FakeInstallerActions.stop_after(
            InstallStage.RELEASE_RESOLVE
        ),
        progress=progress,
    ).run()

    assert result.exit_code == 1
    output = terminal.getvalue()
    assert output.index("PASS     [3/8]") < output.index("ACTION   [3/8]")
    assert all(token in output for token in ("STATE", "RESUME", "LOG"))
    state = load_state(tmp_path / "install-state.json")
    assert state is not None
    assert (
        InstallStage.RELEASE_RESOLVE.value
        in state.completed_stage_input_digests
    )


def test_failed_stage_reports_state_resume_and_log_without_checkpoint(
    tmp_path: Path,
) -> None:
    actions = FakeInstallerActions.healthy()
    actions.failures[InstallStage.PROJECT_INIT] = RuntimeError(
        "uv download failed"
    )
    progress, _, stderr = workflow_progress(tmp_path)

    result = installer_workflow(
        tmp_path, actions=actions, progress=progress
    ).run()

    assert result.exit_code == 2
    assert "FAIL     [6/8]" in stderr.getvalue()
    assert "CAUSE    PROJECT_INIT failed: uv download failed" in stderr.getvalue()
    assert "STATE" in stderr.getvalue()
    assert "RESUME" in stderr.getvalue()
    state = load_state(tmp_path / "install-state.json")
    assert state is not None
    assert (
        InstallStage.PROJECT_INIT.value
        not in state.completed_stage_input_digests
    )


def test_incomplete_stage_starts_before_input_probe_failure(
    tmp_path: Path,
) -> None:
    actions = FakeInstallerActions.healthy()

    def stage_inputs(stage, options, state):
        del options, state
        if stage is InstallStage.BOOTSTRAP:
            raise RuntimeError("input probe failed")
        return {}

    actions.stage_inputs = stage_inputs  # type: ignore[method-assign]
    terminal = io.StringIO()
    progress = InstallerProgress(
        mode=ProgressMode.DEFAULT,
        stdout=terminal,
        stderr=terminal,
        log_root=tmp_path / "logs",
        process_id=25,
    )

    result = installer_workflow(
        tmp_path,
        actions=actions,
        progress=progress,
    ).run()

    assert result.exit_code == 2
    output = terminal.getvalue()
    assert output.index("START    [1/8]") < output.index("FAIL     [1/8]")
    assert "input probe failed" in output
    assert actions.calls == []


def test_complete_rerun_reports_all_skips_and_summary(
    tmp_path: Path,
) -> None:
    assert installer_workflow(
        tmp_path, actions=FakeInstallerActions.healthy()
    ).run().exit_code == 0
    progress, stdout, _ = workflow_progress(tmp_path)
    actions = FakeInstallerActions.healthy()

    result = installer_workflow(
        tmp_path, actions=actions, progress=progress
    ).run()

    assert result.exit_code == 0
    assert stdout.getvalue().count("SKIP") == len(CONTAINER_STAGE_ORDER)
    assert "SUMMARY" in stdout.getvalue()
    assert actions.calls == []


def test_log_creation_failure_runs_no_stage(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    log_root = tmp_path / "unsafe-logs"
    log_root.symlink_to(target, target_is_directory=True)
    progress = InstallerProgress(
        mode=ProgressMode.DEFAULT,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        log_root=log_root,
    )
    actions = FakeInstallerActions.healthy()

    result = installer_workflow(
        tmp_path, actions=actions, progress=progress
    ).run()

    assert result.exit_code == 2
    assert "log" in result.message.lower()
    assert actions.calls == []


def test_log_close_failure_has_no_false_success_and_uses_stderr_fallback(
    tmp_path: Path,
) -> None:
    class CloseFailureProgress(InstallerProgress):
        original_close = None

        def open_session(self, project_dir: Path) -> None:
            super().open_session(project_dir)
            assert self._log is not None
            self.original_close = self._log.close

            def fail_close() -> None:
                raise ProgressError("simulated log fsync failure")

            self._log.close = fail_close  # type: ignore[method-assign]

    stdout = io.StringIO()
    stderr = io.StringIO()
    progress = CloseFailureProgress(
        mode=ProgressMode.DEFAULT,
        stdout=stdout,
        stderr=stderr,
        log_root=tmp_path / "logs",
        process_id=24,
    )

    result = installer_workflow(
        tmp_path,
        actions=FakeInstallerActions.healthy(),
        progress=progress,
    ).run()

    assert result.exit_code == 2
    assert "SUMMARY" not in stdout.getvalue()
    assert "FAIL     installer progress reporting failed" in stderr.getvalue()
    assert "simulated log fsync failure" in stderr.getvalue()
    assert progress.original_close is not None
    progress.original_close()


def test_progress_mode_does_not_change_checkpoint_digests(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    base = workflow_options(tmp_path, project_dir=project)
    default_progress, _, _ = workflow_progress(tmp_path / "default")
    quiet_progress, _, _ = workflow_progress(
        tmp_path / "quiet", mode=ProgressMode.QUIET
    )
    default_result = installer_workflow(
        tmp_path,
        actions=FakeInstallerActions.healthy(),
        options=replace(
            base, state_path=tmp_path / "default-state.json"
        ),
        progress=default_progress,
    ).run()
    quiet_result = installer_workflow(
        tmp_path,
        actions=FakeInstallerActions.healthy(),
        options=replace(base, state_path=tmp_path / "quiet-state.json"),
        progress=quiet_progress,
    ).run()

    assert default_result.state is not None
    assert quiet_result.state is not None
    assert (
        default_result.state.completed_stage_input_digests
        == quiet_result.state.completed_stage_input_digests
    )


def test_full_progress_uses_selected_workflow_positions(
    tmp_path: Path,
) -> None:
    progress, stdout, _ = workflow_progress(tmp_path)
    prompts = FakePrompts(
        exact={"APPLY": True}, yes_no={"docker-group": False}
    )

    result = installer_workflow(
        tmp_path,
        actions=FakeInstallerActions.full_no_reboot(),
        options=full_options(tmp_path),
        prompts=prompts,
        progress=progress,
    ).run()

    assert result.exit_code == 0
    output = stdout.getvalue()
    assert "共 13 个阶段" in output
    assert "[2/13] 检查宿主" in output
    assert "[7/13] 验证重启后的宿主" in output
    assert "[13/13] 完成安装" in output


def test_dry_run_progress_is_positioned_without_persisting_state(
    tmp_path: Path,
) -> None:
    progress, stdout, _ = workflow_progress(tmp_path)
    options = replace(workflow_options(tmp_path), dry_run=True)

    result = installer_workflow(
        tmp_path,
        actions=FakeInstallerActions.healthy(),
        options=options,
        progress=progress,
    ).run()

    assert result.exit_code == 0
    assert "[1/8]" in stdout.getvalue()
    assert "[8/8]" in stdout.getvalue()
    assert "START" not in stdout.getvalue()
    assert "SUMMARY" in stdout.getvalue()
    assert "no stages persisted" in stdout.getvalue()
    assert not options.state_path.exists()


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


def test_implicit_state_isolated_from_unrelated_legacy_project(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "install-state.json"
    old_project = tmp_path / "old-project"
    old_options = workflow_options(tmp_path, project_dir=old_project)
    assert old_options.state_path == legacy
    first = installer_workflow(
        tmp_path,
        actions=FakeInstallerActions.healthy(),
        options=old_options,
    ).run()
    assert first.exit_code == 0
    legacy_before = legacy.read_bytes()

    new_project = tmp_path / "video-lab"
    options = implicit_workflow_options(
        tmp_path,
        project_dir=new_project,
    )
    actions = FakeInstallerActions.healthy()
    prompts = FakePrompts()
    workflow = installer_workflow(
        tmp_path,
        actions=actions,
        options=options,
        prompts=prompts,
    )

    result = workflow.run()

    selected = project_state_path(new_project, legacy)
    assert result.exit_code == 0
    assert workflow.options.state_path == selected
    assert selected.is_file()
    assert legacy.read_bytes() == legacy_before
    assert actions.calls[0] == "bootstrap"
    assert ("INFO", f"installer state (project): {selected}") in prompts.statuses


def test_implicit_state_reuses_matching_legacy_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    first = installer_workflow(
        tmp_path,
        actions=FakeInstallerActions.healthy(),
        options=workflow_options(tmp_path, project_dir=project),
    ).run()
    assert first.exit_code == 0

    options = implicit_workflow_options(tmp_path, project_dir=project)
    actions = FakeInstallerActions.healthy()
    prompts = FakePrompts()
    workflow = installer_workflow(
        tmp_path,
        actions=actions,
        options=options,
        prompts=prompts,
    )

    result = workflow.run()

    assert result.exit_code == 0
    assert workflow.options.state_path == tmp_path / "install-state.json"
    assert actions.calls == []
    assert any(
        prefix == "INFO" and "installer state (legacy)" in message
        for prefix, message in prompts.statuses
    )


def test_implicit_state_stops_on_legacy_state_without_project_identity(
    tmp_path: Path,
) -> None:
    options = workflow_options(tmp_path)
    failing_actions = FakeInstallerActions.healthy()
    failing_actions.failures[InstallStage.BOOTSTRAP] = RuntimeError("stop")
    first = installer_workflow(
        tmp_path,
        actions=failing_actions,
        options=options,
    ).run()
    assert first.exit_code == 2
    payload = json.loads(options.state_path.read_text(encoding="utf-8"))
    payload["project_path"] = None
    options.state_path.write_text(json.dumps(payload), encoding="utf-8")

    actions = FakeInstallerActions.healthy()
    result = installer_workflow(
        tmp_path,
        actions=actions,
        options=replace(options, state_path_explicit=False),
    ).run()

    assert result.exit_code == 2
    assert "invalid install state" in result.message
    assert actions.calls == []
    assert not options.state_path.exists()
    assert len(list(tmp_path.glob("install-state.corrupt.*.json"))) == 1


def test_matching_legacy_project_still_blocks_mode_change(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    first = installer_workflow(
        tmp_path,
        actions=FakeInstallerActions.healthy(),
        options=workflow_options(tmp_path, project_dir=project),
    ).run()
    assert first.exit_code == 0

    options = implicit_workflow_options(
        tmp_path,
        project_dir=project,
        mode=InstallMode.FULL,
    )
    actions = FakeInstallerActions.host_change_requires_reboot()
    result = installer_workflow(
        tmp_path,
        actions=actions,
        options=options,
    ).run()

    assert result.exit_code == 2
    assert "mode changed" in result.message
    assert str(tmp_path / "install-state.json") in result.message
    assert actions.calls == []


def test_different_full_project_is_blocked_by_installer_coordination_lock(
    tmp_path: Path,
) -> None:
    coordination_state = tmp_path / "coordination.json"
    options = implicit_workflow_options(
        tmp_path,
        project_dir=tmp_path / "second-project",
        mode=InstallMode.FULL,
    )
    options = replace(
        options,
        state_path=tmp_path / "other-root" / "state.json",
        coordination_state_path=coordination_state,
    )
    actions = FakeInstallerActions.host_change_requires_reboot()

    with installer_coordination_lock(coordination_state):
        result = installer_workflow(
            tmp_path,
            actions=actions,
            options=options,
        ).run()

    assert result.exit_code == 2
    assert "another installer" in result.message
    assert actions.calls == []


def test_implicit_state_is_selected_after_interactive_project_prompt(
    tmp_path: Path,
) -> None:
    project = (tmp_path / "prompted-project").resolve()

    class ProjectPrompts(FakePrompts):
        def ask_project_dir(self) -> Path:
            return project

    options = replace(
        workflow_options(tmp_path),
        project_dir=None,
        state_path_explicit=False,
    )
    workflow = installer_workflow(
        tmp_path,
        actions=FakeInstallerActions.healthy(),
        options=options,
        prompts=ProjectPrompts(),
    )

    result = workflow.run()

    assert result.exit_code == 0
    assert workflow.options.project_dir == project
    assert workflow.options.state_path == project_state_path(
        project, tmp_path / "install-state.json"
    )


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


def test_same_boot_progress_names_exact_manual_reboot_resume(
    tmp_path: Path,
) -> None:
    boot_id = "12345678-1234-4abc-8def-1234567890ab"
    first = installer_workflow(
        tmp_path,
        actions=FakeInstallerActions.host_change_requires_reboot(),
        options=full_options(tmp_path),
        prompts=FakePrompts(exact={"APPLY": True}),
        boot_id_reader=lambda: boot_id,
    ).run()
    assert first.exit_code == 1
    progress, _, stderr = workflow_progress(tmp_path)
    actions = FakeInstallerActions.host_change_requires_reboot()

    resumed = installer_workflow(
        tmp_path,
        actions=actions,
        options=full_options(tmp_path),
        prompts=FakePrompts(),
        boot_id_reader=lambda: boot_id,
        progress=progress,
    ).run()

    assert resumed.exit_code == 1
    assert (
        "RESUME   sudo reboot；重启后重新执行同一条 install 命令"
        in stderr.getvalue()
    )
    assert "host_apply" not in actions.calls


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


def test_unverified_newer_oem_kernel_warns_records_and_continues(
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

    prompts = FakePrompts()
    resumed_actions = FakeInstallerActions.host_change_requires_reboot()
    returning_host_report(resumed_actions, host_verify_report(Status.UNVERIFIED))
    result = installer_workflow(
        tmp_path,
        actions=resumed_actions,
        options=full_options(tmp_path),
        prompts=prompts,
        boot_id_reader=lambda: second_boot,
    ).run()

    assert result.exit_code == 0
    assert result.state is not None
    assert result.state.host_verification_status == "unverified"
    assert result.state.host_kernel == "6.17.0-1028-oem"
    assert result.state.host_verification_findings == ("HOST.UPSTREAM_UNVERIFIED",)
    assert any(
        prefix == "WARN"
        and "6.17.0-1028-oem" in message
        and "container-check --suite stable" in message
        for prefix, message in prompts.statuses
    )
    assert "host remains unverified" in result.message
    assert "resolve_release" in resumed_actions.calls


def test_host_verify_change_required_remains_blocked(
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
    returning_host_report(resumed_actions, host_verify_report(Status.CHANGE_REQUIRED))
    result = installer_workflow(
        tmp_path,
        actions=resumed_actions,
        options=full_options(tmp_path),
        prompts=FakePrompts(),
        boot_id_reader=lambda: second_boot,
    ).run()

    assert result.exit_code == 2
    assert result.state is not None
    assert result.state.current_stage is InstallStage.HOST_VERIFY
    assert "resolve_release" not in resumed_actions.calls


def test_compatible_patch_installer_resumes_old_host_verify_state(
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
        installer_version="0.2.0",
        installer_source_revision="d" * 40,
    ).run()
    assert first.exit_code == 1

    old_actions = FakeInstallerActions.host_change_requires_reboot()
    old_actions.failures[InstallStage.HOST_VERIFY] = RuntimeError(
        "old installer rejected unverified host"
    )
    old_result = installer_workflow(
        tmp_path,
        actions=old_actions,
        options=full_options(tmp_path),
        prompts=FakePrompts(),
        boot_id_reader=lambda: second_boot,
        installer_version="0.2.0",
        installer_source_revision="d" * 40,
    ).run()
    assert old_result.exit_code == 2
    assert old_result.state is not None
    assert old_result.state.current_stage is InstallStage.HOST_VERIFY

    incompatible_actions = FakeInstallerActions.host_change_requires_reboot()
    incompatible = installer_workflow(
        tmp_path,
        actions=incompatible_actions,
        options=full_options(tmp_path),
        prompts=FakePrompts(),
        boot_id_reader=lambda: second_boot,
        installer_version="0.3.0",
        installer_source_revision="a" * 40,
    ).run()
    assert incompatible.exit_code == 2
    assert "inputs changed" in incompatible.message
    assert incompatible_actions.calls == []

    new_actions = FakeInstallerActions.host_change_requires_reboot()
    returning_host_report(new_actions, host_verify_report(Status.UNVERIFIED))
    prompts = FakePrompts()
    resumed = installer_workflow(
        tmp_path,
        actions=new_actions,
        options=full_options(tmp_path),
        prompts=prompts,
        boot_id_reader=lambda: second_boot,
        installer_version="0.2.1",
        installer_source_revision="e" * 40,
    ).run()

    assert resumed.exit_code == 0
    assert resumed.state is not None
    assert resumed.state.installer_version == "0.2.1"
    assert resumed.state.installer_source_revision == "e" * 40
    assert "host_apply" not in new_actions.calls
    assert new_actions.calls[0] == "host_verify"
    assert any(
        prefix == "WARN" and "compatible installer update" in message
        for prefix, message in prompts.statuses
    )


def test_compatible_patch_installer_adopts_container_state(
    tmp_path: Path,
) -> None:
    first = installer_workflow(
        tmp_path,
        actions=FakeInstallerActions.healthy(),
        installer_version="0.2.2",
        installer_source_revision="d" * 40,
    ).run()
    assert first.state is not None
    old_digests = dict(first.state.completed_stage_input_digests)
    resumed_actions = FakeInstallerActions.healthy()

    resumed = installer_workflow(
        tmp_path,
        actions=resumed_actions,
        installer_version="0.2.3",
        installer_source_revision="e" * 40,
    ).run()

    assert resumed.exit_code == 0
    assert resumed_actions.calls == []
    assert resumed.state is not None and resumed.state.schema_version == 2
    persisted = load_state(tmp_path / "install-state.json")
    assert persisted is not None
    assert persisted.installer_version == "0.2.3"
    assert persisted.installer_source_revision == "e" * 40
    changed = {
        name
        for name, digest in resumed.state.completed_stage_input_digests.items()
        if old_digests[name] != digest
    }
    assert changed == {InstallStage.BOOTSTRAP.value}


def test_incompatible_container_bootstrap_digest_is_rejected(
    tmp_path: Path,
) -> None:
    first = installer_workflow(
        tmp_path,
        actions=FakeInstallerActions.healthy(),
        installer_version="0.2.2",
        installer_source_revision="d" * 40,
    ).run()
    assert first.state is not None
    completed = dict(first.state.completed_stage_input_digests)
    completed[InstallStage.BOOTSTRAP.value] = "0" * 64
    save_state(
        tmp_path / "install-state.json",
        replace(first.state, completed_stage_input_digests=completed),
    )
    actions = FakeInstallerActions.healthy()

    rejected = installer_workflow(
        tmp_path,
        actions=actions,
        installer_version="0.2.3",
        installer_source_revision="e" * 40,
    ).run()

    assert rejected.exit_code == 2
    assert "inputs changed" in rejected.message
    assert actions.calls == []


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
