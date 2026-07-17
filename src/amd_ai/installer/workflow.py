from __future__ import annotations

import getpass
import hashlib
import os
import re
import shlex
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Protocol

from amd_ai.host.models import HostPlanPhase
from amd_ai.installer.actions import (
    HostPlanResult,
    LocalBuildResult,
)
from amd_ai.installer.models import (
    CONTAINER_STAGE_ORDER,
    FULL_STAGE_ORDER,
    DiskSpaceEstimate,
    InstallMode,
    InstallOptions,
    InstallStage,
    InstallState,
    InstallerModelError,
    STATE_SCHEMA_VERSION,
    StageResult,
    StableRelease,
)
from amd_ai.installer.prompts import (
    NonInteractivePrompts,
    PromptError,
    TerminalPrompts,
)
from amd_ai.installer.progress import (
    ProgressError,
    ProgressOutcome,
    ProgressReporter,
    PromptProgressAdapter,
    SessionPlan,
    StagePosition,
)
from amd_ai.installer.registry import registry_candidates
from amd_ai.installer.release import (
    ReleaseAcquisitionError,
    ReleaseIdentityError,
    VerifiedReleaseImages,
    load_stable_release,
)
from amd_ai.installer.state import (
    CorruptInstallState,
    InstallAlreadyRunning,
    InstallerStateError,
    ResumeInputChanged,
    boot_id_changed,
    installer_coordination_lock,
    install_lock,
    load_state,
    read_boot_id,
    save_state,
    select_install_state_path,
    stage_input_digest,
    validate_completed_stage,
)
from amd_ai.report import Report, Status


GIB = 1024**3
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


class InstallerActions(Protocol):
    pass


class WorkflowError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkflowResult:
    exit_code: int
    state: InstallState | None
    message: str = ""
    progress_outcome: ProgressOutcome = ProgressOutcome.FAILURE


@dataclass(frozen=True)
class HostConfirmation:
    accepted: bool
    docker_group_accepted: bool
    message: str = ""


@dataclass(frozen=True)
class DiskRequirement:
    operation: str
    source: str
    payload_label: str
    estimate: DiskSpaceEstimate
    required_bytes: int


class InstallerWorkflow:
    def __init__(
        self,
        *,
        options: InstallOptions,
        actions: InstallerActions,
        installer_version: str,
        installer_source_revision: str,
        prompts: object | None = None,
        progress: ProgressReporter | None = None,
        boot_id_reader: Callable[[], str] = read_boot_id,
    ) -> None:
        self.options = options
        self.actions = actions
        self.installer_version = installer_version
        self.installer_source_revision = installer_source_revision
        self.prompts = prompts or (
            NonInteractivePrompts()
            if options.non_interactive
            else TerminalPrompts()
        )
        status = getattr(self.prompts, "status", None)
        self.progress = (
            progress
            if progress is not None
            else PromptProgressAdapter(
                status if status is not None else lambda _prefix, _message: None
            )
        )
        self._release: StableRelease | None = None
        self._kernel_plan: HostPlanResult | None = None
        self._host_plan: HostPlanResult | None = None
        self._boot_id_reader = boot_id_reader
        self._state_source = "explicit"
        self._current_position: StagePosition | None = None

    def run(self) -> WorkflowResult:
        state: InstallState | None = None
        try:
            self._prepare_options()
            assert self.options.project_dir is not None
            self.progress.open_session(self.options.project_dir)
            self.options.validate()
            if REVISION_PATTERN.fullmatch(
                self.installer_source_revision
            ) is None:
                raise WorkflowError("installer source revision is invalid")
            with installer_coordination_lock(
                self.options.coordination_state_path
            ):
                self._select_state_path()
                with install_lock(self.options.state_path):
                    installer_update_adopted = False
                    state = load_state(self.options.state_path)
                    if state is None:
                        state = self._new_state()
                        if not self.options.dry_run:
                            save_state(self.options.state_path, state)
                    elif not self.options.dry_run:
                        previous_state = state
                        state = self._adopt_compatible_installer_update(state)
                        installer_update_adopted = state is not previous_state
                    self._validate_transition(state)
                    if installer_update_adopted:
                        save_state(self.options.state_path, state)
                    self._report_session_plan(state)
                    if self.options.dry_run:
                        result = self._run_dry(state)
                    else:
                        result = self._run_stages(state)
        except KeyboardInterrupt:
            result = WorkflowResult(
                1,
                state,
                "installation interrupted",
                ProgressOutcome.ACTION,
            )
        except ResumeInputChanged as error:
            result = WorkflowResult(
                2,
                state,
                f"{error}; state: {self.options.state_path}",
                ProgressOutcome.FAILURE,
            )
        except (
            CorruptInstallState,
            InstallAlreadyRunning,
            InstallerModelError,
            InstallerStateError,
            ProgressError,
            PromptError,
            WorkflowError,
        ) as error:
            result = WorkflowResult(
                2, state, str(error), ProgressOutcome.FAILURE
            )
        except Exception as error:
            result = WorkflowResult(
                2,
                state,
                f"installer failed: {error}",
                ProgressOutcome.FAILURE,
            )
        return self._finish_progress(result)

    def _finish_progress(self, result: WorkflowResult) -> WorkflowResult:
        progress_error: Exception | None = None
        try:
            self.progress.installation_finished(
                outcome=result.progress_outcome,
                exit_code=result.exit_code,
                message=result.message,
                state_path=self.options.state_path,
                project_dir=self.options.project_dir,
                position=self._current_position,
            )
        except Exception as error:
            progress_error = error
        try:
            self.progress.close()
        except Exception as error:
            if progress_error is None:
                progress_error = error
        if progress_error is None:
            return result
        message = f"progress reporting failed: {progress_error}"
        if result.message:
            message = f"{result.message}; {message}"
        fallback = getattr(self.progress, "fallback_failure", None)
        if fallback is not None:
            try:
                fallback(message)
            except Exception:
                pass
        return WorkflowResult(
            2,
            result.state,
            message,
            ProgressOutcome.FAILURE,
        )

    def _prepare_options(self) -> None:
        if self.options.mode is InstallMode.DOCTOR:
            raise WorkflowError("doctor mode is routed outside the install workflow")
        if self.options.project_dir is None:
            project_dir = self.prompts.ask_project_dir()
            self.options = replace(self.options, project_dir=project_dir)

    def _select_state_path(self) -> None:
        assert self.options.project_dir is not None
        selection = select_install_state_path(
            project_dir=self.options.project_dir,
            requested_path=self.options.state_path,
            explicit=self.options.state_path_explicit,
        )
        self.options = replace(self.options, state_path=selection.path)
        self._state_source = selection.source
        self._status(
            "INFO",
            f"installer state ({selection.source}): {selection.path}",
        )

    def _report_session_plan(self, state: InstallState) -> None:
        assert self.options.project_dir is not None
        order = self._stage_order()
        first_incomplete = next(
            (
                stage
                for stage in order
                if stage.value not in state.completed_stage_input_digests
            ),
            None,
        )
        self.progress.session_plan(
            SessionPlan(
                mode=self.options.mode,
                project_dir=self.options.project_dir,
                project_name=self.options.project_name,
                state_path=self.options.state_path,
                state_source=self._state_source,
                image_source=self.options.image_source or "pull",
                registry=self.options.registry,
                release_id=state.release_id,
                stages=order,
                first_incomplete=first_incomplete,
            )
        )

    def _new_state(self) -> InstallState:
        now = _utc_timestamp()
        target_user = (
            self.options.target_user
            or os.environ.get("SUDO_USER")
            or getpass.getuser()
        )
        assert self.options.project_dir is not None
        assert self.options.source_root is not None
        return InstallState(
            schema_version=STATE_SCHEMA_VERSION,
            installer_version=self.installer_version,
            mode=self.options.mode,
            target_user=target_user,
            release_id=None,
            source_revision=None,
            base_image_reference=None,
            base_manifest_digest=None,
            torch_image_reference=None,
            torch_manifest_digest=None,
            project_path=str(self.options.project_dir),
            current_stage=self._stage_order()[0],
            completed_stage_input_digests={},
            reboot_boot_id=None,
            created_at=now,
            updated_at=now,
            installer_source_revision=self.installer_source_revision,
            source_root=str(self.options.source_root),
            host_plan_digest=None,
            host_adapter_id=None,
            docker_group_accepted=False,
            base_config_digest=None,
            torch_config_digest=None,
            last_report_paths=(),
            host_verification_status=None,
            host_kernel=None,
            host_verification_findings=(),
        )

    def _run_stages(self, state: InstallState) -> WorkflowResult:
        order = self._stage_order()
        for index, stage in enumerate(order):
            position = StagePosition(stage, index + 1, len(order))
            self._current_position = position
            self.progress.stage_candidate(position)
            if stage.value in state.completed_stage_input_digests:
                inputs = self._stage_inputs(stage, state)
                validate_completed_stage(state, stage, inputs)
                if stage in {
                    InstallStage.KERNEL_VERIFY,
                    InstallStage.HOST_VERIFY,
                }:
                    state, guard_result = self._rerun_completed_guard(
                        state,
                        stage,
                        position,
                    )
                    if guard_result is not None:
                        return guard_result
                    continue
                self.progress.stage_skipped(position)
                continue
            if state.current_stage is not stage:
                raise WorkflowError(
                    f"illegal installer transition: state={state.current_stage.value}, "
                    f"next={stage.value}"
                )
            self.progress.stage_started(position)
            inputs = self._stage_inputs(stage, state)
            requirement = self._disk_requirement(stage, state)
            if requirement is not None:
                self._report_disk_requirement(requirement)
                shortage = _require_disk_space(requirement)
                if shortage is not None:
                    return WorkflowResult(
                        2,
                        state,
                        shortage,
                        ProgressOutcome.FAILURE,
                    )
            try:
                output = self._dispatch(stage, state)
            except KeyboardInterrupt:
                return WorkflowResult(
                    1,
                    state,
                    f"{stage.value} interrupted",
                    ProgressOutcome.ACTION,
                )
            except Exception as error:
                return WorkflowResult(
                    2,
                    state,
                    f"{stage.value} failed: {error}",
                    ProgressOutcome.FAILURE,
                )

            outcome = self._stage_result(stage, output)
            if outcome.blocked:
                if isinstance(output, Report):
                    state = self._apply_output(state, stage, output, outcome)
                state = self._record_reports(state, outcome)
                save_state(self.options.state_path, state)
                return WorkflowResult(
                    2,
                    state,
                    outcome.message or f"{stage.value} is blocked",
                    ProgressOutcome.BLOCKED,
                )
            if (
                stage is InstallStage.KERNEL_REBOOT_PENDING
                and outcome.action_required
            ):
                state = self._record_reports(state, outcome)
                save_state(self.options.state_path, state)
                return WorkflowResult(
                    1,
                    state,
                    outcome.message or "manual reboot is required",
                    ProgressOutcome.ACTION,
                )
            state = self._apply_output(state, stage, output, outcome)
            if stage is InstallStage.RELEASE_RESOLVE:
                self._report_release_identity(output)
            state = self._checkpoint(
                state,
                stage=stage,
                inputs=inputs,
                next_stage=(order[index + 1] if index + 1 < len(order) else stage),
            )
            save_state(self.options.state_path, state)
            self.progress.stage_passed(position)
            if (
                stage is InstallStage.HOST_VERIFY
                and isinstance(output, Report)
                and output.status is Status.UNVERIFIED
            ):
                self._status("WARN", outcome.message)
            if outcome.action_required:
                return WorkflowResult(
                    1,
                    state,
                    outcome.message or f"{stage.value} requires operator action",
                    ProgressOutcome.ACTION,
                )
        message = "installation complete"
        if state.host_verification_status == Status.UNVERIFIED.value:
            message += (
                f"; host remains unverified on {state.host_kernel}; "
                "run the displayed full qualification before release promotion"
            )
        return WorkflowResult(
            0, state, message, ProgressOutcome.SUCCESS
        )

    def _rerun_completed_guard(
        self,
        state: InstallState,
        stage: InstallStage,
        position: StagePosition,
    ) -> tuple[InstallState, WorkflowResult | None]:
        self.progress.stage_started(position)
        try:
            output = self._dispatch(stage, state)
        except KeyboardInterrupt:
            return state, WorkflowResult(
                1,
                state,
                f"{stage.value} interrupted",
                ProgressOutcome.ACTION,
            )
        except Exception as error:
            return state, WorkflowResult(
                2,
                state,
                f"{stage.value} failed: {error}",
                ProgressOutcome.FAILURE,
            )

        outcome = self._stage_result(stage, output)
        if outcome.blocked:
            if isinstance(output, Report):
                state = self._apply_output(state, stage, output, outcome)
            state = self._record_reports(state, outcome)
            save_state(self.options.state_path, state)
            return state, WorkflowResult(
                2,
                state,
                outcome.message or f"{stage.value} is blocked",
                ProgressOutcome.BLOCKED,
            )
        state = self._apply_output(state, stage, output, outcome)
        state = self._record_reports(state, outcome)
        save_state(self.options.state_path, state)
        self.progress.stage_passed(position)
        if (
            stage is InstallStage.HOST_VERIFY
            and isinstance(output, Report)
            and output.status is Status.UNVERIFIED
        ):
            self._status("WARN", outcome.message)
        return state, None

    def _run_dry(self, state: InstallState) -> WorkflowResult:
        mutating = {
            InstallStage.KERNEL_APPLY,
            InstallStage.HOST_APPLY,
            InstallStage.IMAGE_PULL_OR_BUILD,
            InstallStage.IMAGE_VERIFY,
            InstallStage.PROJECT_INIT,
            InstallStage.PROJECT_VERIFY,
        }
        order = self._stage_order()
        for index, stage in enumerate(order):
            position = StagePosition(stage, index + 1, len(order))
            self._current_position = position
            self.progress.stage_candidate(position)
            prefix = "ACTION" if stage in mutating else "PASS"
            self._status(
                prefix,
                f"[{position.index}/{position.total}] dry-run {stage.value}",
            )
        return WorkflowResult(
            0,
            state,
            "dry-run complete; no stages persisted",
            ProgressOutcome.SUCCESS,
        )

    def _dispatch(self, stage: InstallStage, state: InstallState) -> object:
        assert self.options.source_root is not None
        assert self.options.project_dir is not None
        if stage is InstallStage.BOOTSTRAP:
            return self.actions.bootstrap(
                source_root=self.options.source_root,
                revision=self.installer_source_revision,
            )
        if stage is InstallStage.CONTAINER_HOST_CHECK:
            return self.actions.container_host_check()
        if stage is InstallStage.HOST_PREFLIGHT:
            return self.actions.host_preflight(target_user=state.target_user)
        if stage is InstallStage.KERNEL_PLAN:
            if state.target_user is None:
                raise WorkflowError("host target user is unavailable")
            self._kernel_plan = self.actions.host_plan(
                target_user=state.target_user,
                phase=HostPlanPhase.KERNEL,
            )
            return self._kernel_plan
        if stage is InstallStage.KERNEL_CONFIRM:
            return self._confirm_kernel_plan(state)
        if stage is InstallStage.KERNEL_APPLY:
            return self.actions.host_apply(
                self._resolved_kernel_plan(state),
                include_docker_group=False,
            )
        if stage is InstallStage.KERNEL_REBOOT_PENDING:
            return self._reboot_pending(
                state.kernel_reboot_boot_id,
                message=(
                    "manual reboot into the OEM 6.17 kernel is required; "
                    "reboot and rerun the same install command"
                ),
            )
        if stage is InstallStage.KERNEL_VERIFY:
            if state.target_user is None:
                raise WorkflowError("host target user is unavailable")
            return self.actions.kernel_verify(
                target_user=state.target_user,
                display_manager_was_loaded=(
                    state.display_manager_was_loaded
                ),
                display_manager_was_active=state.display_manager_was_active,
            )
        if stage is InstallStage.HOST_PLAN:
            if state.target_user is None:
                raise WorkflowError("host target user is unavailable")
            self._host_plan = self.actions.host_plan(
                target_user=state.target_user,
                phase=HostPlanPhase.TUNING,
            )
            return self._host_plan
        if stage is InstallStage.HOST_CONFIRM:
            return self._confirm_host_plan(state)
        if stage is InstallStage.HOST_APPLY:
            return self.actions.host_apply(
                self._resolved_host_plan(state),
                include_docker_group=state.docker_group_accepted,
            )
        if stage is InstallStage.HOST_VERIFY:
            if state.target_user is None:
                raise WorkflowError("host target user is unavailable")
            return self.actions.host_verify(target_user=state.target_user)
        if stage is InstallStage.RELEASE_RESOLVE:
            self._release = self.actions.resolve_release(
                self.options.manifest_path
            )
            return self._release
        if stage is InstallStage.IMAGE_PULL_OR_BUILD:
            release = self._resolved_release(state)
            source = self.options.image_source or "pull"
            if source == "pull":
                try:
                    return self._pull_release_candidates(release)
                except ReleaseIdentityError as error:
                    raise WorkflowError(
                        "release image identity verification failed; "
                        f"local fallback is forbidden: {error}"
                    ) from error
                except ReleaseAcquisitionError as error:
                    if self.options.non_interactive:
                        raise WorkflowError(
                            f"exact release pull failed: {error}"
                        ) from error
                    self.progress.pause_heartbeat()
                    try:
                        fallback = self.prompts.choose_image_fallback()
                    finally:
                        self.progress.resume_heartbeat()
                    if fallback != "build":
                        raise WorkflowError(
                            f"exact release pull failed and local build was refused: {error}"
                        ) from error
                    self._status(
                        "WARN",
                        "exact release pull failed; using explicit local build",
                    )
                    requirement = self._image_disk_requirement(
                        release,
                        image_source="build",
                    )
                    self._report_disk_requirement(requirement)
                    shortage = _require_disk_space(requirement)
                    if shortage is not None:
                        raise WorkflowError(shortage)
                    return self.actions.build_local_images(
                        source_root=self.options.source_root,
                        installer_source_revision=self.installer_source_revision,
                    )
            return self.actions.build_local_images(
                source_root=self.options.source_root,
                installer_source_revision=self.installer_source_revision,
            )
        if stage is InstallStage.IMAGE_VERIFY:
            if state.torch_image_reference is None:
                raise WorkflowError("Torch image identity is unavailable")
            return self.actions.verify_torch_image(
                state.torch_image_reference
            )
        if stage is InstallStage.PROJECT_INIT:
            return self.actions.initialize_project(
                project_dir=self.options.project_dir,
                project_name=self.options.project_name,
                base_profile="stable",
                base_image_reference=state.torch_image_reference,
                base_config_digest=state.torch_config_digest,
                target_user=state.target_user,
            )
        if stage is InstallStage.PROJECT_VERIFY:
            return self.actions.verify_project(
                project_dir=self.options.project_dir,
                manifest_path=self.options.manifest_path,
                qualified=state.release_id != "local",
                target_user=state.target_user,
            )
        if stage is InstallStage.COMPLETE:
            return StageResult()
        raise WorkflowError(f"stage is not implemented: {stage.value}")

    def _pull_release_candidates(
        self,
        release: StableRelease,
    ) -> VerifiedReleaseImages:
        failures: list[str] = []
        candidates = registry_candidates(release, self.options.registry)
        for index, candidate in enumerate(candidates):
            self.progress.detail(
                f"当前仓库={candidate.label}，来源=公开匿名镜像"
            )
            try:
                verified = self.actions.pull_release(candidate.release)
            except ReleaseAcquisitionError as error:
                failures.append(f"{candidate.label}: {error}")
                if index + 1 < len(candidates):
                    next_candidate = candidates[index + 1]
                    self._status(
                        "WARN",
                        f"{candidate.label} 获取失败，正在回退 "
                        f"{next_candidate.label}：{error}",
                    )
                continue
            if not isinstance(verified, VerifiedReleaseImages):
                raise WorkflowError(
                    "release pull returned invalid verified identities"
                )
            self.progress.detail(
                f"已采用仓库={candidate.label}，"
                f"base={verified.base.reference}，"
                f"torch={verified.torch.reference}"
            )
            return verified
        raise ReleaseAcquisitionError(
            "all configured registries failed: " + "; ".join(failures)
        )

    def _stage_inputs(
        self, stage: InstallStage, state: InstallState
    ) -> dict[str, object]:
        assert self.options.project_dir is not None
        assert self.options.source_root is not None
        values: dict[str, object] = {
            "stage": stage.value,
            "mode": self.options.mode.value,
        }
        if stage is InstallStage.BOOTSTRAP:
            values.update(
                {
                    "installer_version": self.installer_version,
                    "installer_source_revision": self.installer_source_revision,
                    "source_root": str(self.options.source_root),
                    "project_path": str(self.options.project_dir),
                    "project_name": self.options.project_name,
                    "target_user": self.options.target_user,
                }
            )
        elif stage in {
            InstallStage.HOST_PREFLIGHT,
            InstallStage.CONTAINER_HOST_CHECK,
        }:
            values.update(
                {
                    "target_user": state.target_user,
                }
            )
        elif stage is InstallStage.KERNEL_PLAN:
            values.update(
                {
                    "target_user": state.target_user,
                    "phase": HostPlanPhase.KERNEL.value,
                }
            )
        elif stage is InstallStage.KERNEL_CONFIRM:
            values.update(
                {
                    "kernel_plan_digest": state.kernel_plan_digest,
                    "non_interactive": self.options.non_interactive,
                    "accepted_kernel_plan_digest": (
                        self.options.accepted_kernel_plan_digest
                        if self.options.non_interactive
                        else None
                    ),
                }
            )
        elif stage is InstallStage.KERNEL_APPLY:
            values.update(
                {
                    "kernel_plan_digest": state.kernel_plan_digest,
                    "host_adapter_id": state.host_adapter_id,
                }
            )
        elif stage in {
            InstallStage.KERNEL_REBOOT_PENDING,
            InstallStage.KERNEL_VERIFY,
        }:
            values.update(
                {
                    "kernel_plan_digest": state.kernel_plan_digest,
                    "host_adapter_id": state.host_adapter_id,
                    "kernel_reboot_boot_id": state.kernel_reboot_boot_id,
                    "recovery_kernel": state.recovery_kernel,
                    "display_manager_was_loaded": (
                        state.display_manager_was_loaded
                    ),
                    "display_manager_was_active": (
                        state.display_manager_was_active
                    ),
                }
            )
        elif stage is InstallStage.HOST_PLAN:
            values.update(
                {
                    "target_user": state.target_user,
                    "phase": HostPlanPhase.TUNING.value,
                    "kernel_verification_status": (
                        state.kernel_verification_status
                    ),
                    "kernel_kernel": state.kernel_kernel,
                }
            )
        elif stage is InstallStage.HOST_CONFIRM:
            values.update(
                {
                    "host_plan_digest": state.host_plan_digest,
                    "non_interactive": self.options.non_interactive,
                    "accepted_host_plan_digest": (
                        self.options.accepted_host_plan_digest
                        if self.options.non_interactive
                        else None
                    ),
                    "accept_docker_group": (
                        self.options.accept_docker_group
                        if self.options.non_interactive
                        else None
                    ),
                }
            )
        elif stage is InstallStage.HOST_APPLY:
            values.update(
                {
                    "host_plan_digest": state.host_plan_digest,
                    "host_adapter_id": state.host_adapter_id,
                    "docker_group_accepted": state.docker_group_accepted,
                }
            )
        elif stage is InstallStage.HOST_VERIFY:
            values.update(
                {
                    "host_plan_digest": state.host_plan_digest,
                    "host_adapter_id": state.host_adapter_id,
                }
            )
        elif stage is InstallStage.RELEASE_RESOLVE:
            values.update(
                {
                    "manifest_path": str(self.options.manifest_path),
                    "manifest_file_digest": _file_digest(
                        self.options.manifest_path
                    ),
                    "host_plan_digest": state.host_plan_digest,
                }
            )
        elif stage is InstallStage.IMAGE_PULL_OR_BUILD:
            release = load_stable_release(self.options.manifest_path)
            values.update(
                {
                    "release_id": release.release_id,
                    "source_revision": release.source_revision,
                    "base_reference": release.base.reference,
                    "base_manifest_digest": release.base.manifest_digest,
                    "base_config_digest": release.base.config_digest,
                    "torch_reference": release.torch.reference,
                    "torch_manifest_digest": release.torch.manifest_digest,
                    "torch_config_digest": release.torch.config_digest,
                    "image_source": self.options.image_source or "pull",
                }
            )
        elif stage is InstallStage.IMAGE_VERIFY:
            values.update(
                {
                    "torch_reference": state.torch_image_reference,
                    "torch_manifest_digest": state.torch_manifest_digest,
                    "torch_config_digest": state.torch_config_digest,
                }
            )
        elif stage is InstallStage.PROJECT_INIT:
            values.update(
                {
                    "project_path": str(self.options.project_dir),
                    "project_name": self.options.project_name,
                    "torch_reference": state.torch_image_reference,
                    "torch_manifest_digest": state.torch_manifest_digest,
                    "torch_config_digest": state.torch_config_digest,
                }
            )
        elif stage in (InstallStage.PROJECT_VERIFY, InstallStage.COMPLETE):
            values.update(
                {
                    "project_path": str(self.options.project_dir),
                    "project_config_digest": _file_digest(
                        self.options.project_dir / "amd-ai-project.toml"
                    ),
                    "torch_manifest_digest": state.torch_manifest_digest,
                }
            )
        hook = getattr(self.actions, "stage_inputs", None)
        if hook is not None:
            values["observed"] = hook(stage, self.options, state)
        return values

    def _stage_result(
        self, stage: InstallStage, output: object
    ) -> StageResult:
        if isinstance(output, StageResult):
            return output
        if isinstance(output, HostConfirmation):
            return StageResult(
                blocked=not output.accepted,
                message=output.message,
            )
        if isinstance(output, Report):
            if stage is InstallStage.HOST_PREFLIGHT:
                blocked = output.status is Status.BLOCKED
            elif stage is InstallStage.KERNEL_VERIFY:
                blocked = output.status is not Status.PASS
            elif stage is InstallStage.HOST_VERIFY:
                blocked = output.status not in (
                    Status.PASS,
                    Status.UNVERIFIED,
                )
            else:
                blocked = False
            if (
                stage is InstallStage.HOST_VERIFY
                and output.status is Status.UNVERIFIED
            ):
                message = self._unverified_host_message(output)
            elif stage is InstallStage.KERNEL_VERIFY and blocked:
                message = (
                    "kernel verification failed; reboot and select the retained "
                    "recovery kernel under Advanced options for Ubuntu, then "
                    "inspect the current-boot amdgpu log; "
                    + _report_finding_message(output)
                )
            elif blocked:
                message = (
                    f"{output.command} returned {output.status.value}; "
                    + _report_finding_message(output)
                )
            else:
                message = ""
            return StageResult(
                facts={"report": output.to_dict()},
                blocked=blocked,
                message=message,
            )
        hook = getattr(self.actions, "stage_result", None)
        if hook is None:
            return StageResult()
        result = hook(stage, output)
        if not isinstance(result, StageResult):
            raise WorkflowError("action stage_result returned an invalid value")
        return result

    def _apply_output(
        self,
        state: InstallState,
        stage: InstallStage,
        output: object,
        outcome: StageResult,
    ) -> InstallState:
        changes: dict[str, object] = {}
        if stage is InstallStage.BOOTSTRAP:
            assert self.options.source_root is not None
            changes.update(
                {
                    "installer_version": self.installer_version,
                    "installer_source_revision": self.installer_source_revision,
                    "source_root": str(self.options.source_root),
                }
            )
        elif stage is InstallStage.KERNEL_PLAN:
            if not isinstance(output, HostPlanResult):
                raise WorkflowError("kernel plan action returned an invalid value")
            if output.plan.phase is not HostPlanPhase.KERNEL:
                raise WorkflowError("kernel plan action returned the wrong phase")
            changes.update(
                {
                    "kernel_plan_digest": output.plan_digest,
                    "host_adapter_id": output.adapter_id,
                }
            )
        elif stage is InstallStage.KERNEL_APPLY:
            plan = self._resolved_kernel_plan(state)
            changes.update(
                {
                    "kernel_reboot_boot_id": (
                        self._boot_id_reader()
                        if plan.plan.reboot_required
                        else None
                    ),
                    "recovery_kernel": plan.running_kernel,
                    "display_manager_was_loaded": (
                        plan.display_manager_loaded
                    ),
                    "display_manager_was_active": (
                        plan.display_manager_active
                    ),
                }
            )
        elif stage is InstallStage.KERNEL_VERIFY and isinstance(output, Report):
            kernel = _report_kernel(output, "kernel verification")
            changes.update(
                {
                    "kernel_verification_status": output.status.value,
                    "kernel_kernel": kernel,
                    "kernel_verification_findings": tuple(
                        finding.code for finding in output.findings
                    ),
                }
            )
        elif stage is InstallStage.HOST_PLAN:
            if not isinstance(output, HostPlanResult):
                raise WorkflowError("host plan action returned an invalid value")
            if output.plan.phase is not HostPlanPhase.TUNING:
                raise WorkflowError("host plan action returned the wrong phase")
            if state.host_adapter_id not in (None, output.adapter_id):
                raise WorkflowError("host adapter changed between plan phases")
            changes.update(
                {
                    "host_plan_digest": output.plan_digest,
                    "host_adapter_id": output.adapter_id,
                }
            )
        elif stage is InstallStage.HOST_CONFIRM:
            if not isinstance(output, HostConfirmation):
                raise WorkflowError("host confirmation returned an invalid value")
            changes["docker_group_accepted"] = (
                output.docker_group_accepted
            )
        elif stage is InstallStage.HOST_VERIFY and isinstance(output, Report):
            kernel = _report_kernel(output, "host verification")
            changes.update(
                {
                    "host_verification_status": output.status.value,
                    "host_kernel": kernel,
                    "host_verification_findings": tuple(
                        finding.code for finding in output.findings
                    ),
                }
            )
        elif stage is InstallStage.RELEASE_RESOLVE:
            if not isinstance(output, StableRelease):
                raise WorkflowError("release action returned an invalid value")
            if (
                self.options.mode is InstallMode.FULL
                and state.host_adapter_id
                not in output.supported_host_adapter_ids
            ):
                raise WorkflowError(
                    "stable release does not support the applied host adapter"
                )
            changes.update(
                {
                    "release_id": output.release_id,
                    "source_revision": output.source_revision,
                    "base_image_reference": output.base.reference,
                    "base_manifest_digest": output.base.manifest_digest,
                    "base_config_digest": output.base.config_digest,
                    "torch_image_reference": output.torch.reference,
                    "torch_manifest_digest": output.torch.manifest_digest,
                    "torch_config_digest": output.torch.config_digest,
                }
            )
        elif stage is InstallStage.IMAGE_PULL_OR_BUILD and isinstance(
            output, VerifiedReleaseImages
        ):
            changes.update(
                {
                    "base_image_reference": output.base.reference,
                    "base_manifest_digest": _reference_digest(
                        output.base.reference
                    ),
                    "base_config_digest": output.base.config_digest,
                    "torch_image_reference": output.torch.reference,
                    "torch_manifest_digest": _reference_digest(
                        output.torch.reference
                    ),
                    "torch_config_digest": output.torch.config_digest,
                }
            )
        elif stage is InstallStage.IMAGE_PULL_OR_BUILD and isinstance(
            output, LocalBuildResult
        ):
            changes.update(
                {
                    "release_id": "local",
                    "base_image_reference": output.base_reference,
                    "base_manifest_digest": output.base_reference,
                    "base_config_digest": output.base_config_digest,
                    "torch_image_reference": output.torch_reference,
                    "torch_manifest_digest": output.torch_reference,
                    "torch_config_digest": output.torch_config_digest,
                }
            )
        reports = tuple(dict.fromkeys((*state.last_report_paths, *outcome.report_paths)))
        return replace(state, last_report_paths=reports, **changes)

    def _adopt_compatible_installer_update(self, state: InstallState) -> InstallState:
        assert self.options.source_root is not None
        current_root = str(self.options.source_root)
        if (
            state.installer_version == self.installer_version
            and state.installer_source_revision == self.installer_source_revision
            and state.source_root == current_root
        ):
            return state
        if not self._can_adopt_installer_update(state):
            return state
        expected = state.completed_stage_input_digests.get(InstallStage.BOOTSTRAP.value)
        if expected is None:
            return state
        current_inputs = self._stage_inputs(InstallStage.BOOTSTRAP, state)
        previous_inputs = dict(current_inputs)
        previous_inputs.update(
            {
                "installer_version": state.installer_version,
                "installer_source_revision": state.installer_source_revision,
                "source_root": state.source_root,
            }
        )
        if stage_input_digest(previous_inputs) != expected:
            return state
        completed = dict(state.completed_stage_input_digests)
        completed[InstallStage.BOOTSTRAP.value] = stage_input_digest(current_inputs)
        updated = replace(
            state,
            installer_version=self.installer_version,
            installer_source_revision=self.installer_source_revision,
            source_root=current_root,
            completed_stage_input_digests=completed,
            updated_at=_utc_timestamp(),
        )
        self._status(
            "WARN",
            "resuming after compatible installer update "
            f"{state.installer_version}@{state.installer_source_revision[:12]} "
            f"-> {self.installer_version}@{self.installer_source_revision[:12]}",
        )
        return updated

    def _can_adopt_installer_update(self, state: InstallState) -> bool:
        previous_series = _installer_series(state.installer_version)
        current_series = _installer_series(self.installer_version)
        same_series = (
            previous_series is not None and previous_series == current_series
        )
        if state.mode is InstallMode.FULL:
            diagnostic_patch = (
                state.installer_version == "0.3.0"
                and self.installer_version == "0.3.1"
            )
            start_stage = (
                InstallStage.KERNEL_VERIFY
                if diagnostic_patch
                else InstallStage.HOST_VERIFY
            )
            start = FULL_STAGE_ORDER.index(start_stage)
            compatible_stage = state.current_stage in FULL_STAGE_ORDER[start:]
            compatible_version = same_series
        elif state.mode is InstallMode.CONTAINER:
            compatible_stage = (
                InstallStage.BOOTSTRAP.value
                in state.completed_stage_input_digests
                and state.current_stage in CONTAINER_STAGE_ORDER[1:]
            )
            compatible_version = same_series or (
                previous_series == (0, 2) and current_series == (0, 3)
            )
        else:
            return False
        return compatible_stage and compatible_version

    def _unverified_host_message(self, report: Report) -> str:
        assert self.options.source_root is not None
        kernel = report.facts.get("kernel")
        if (
            not isinstance(kernel, str)
            or re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z.+_-]{0,127}", kernel) is None
        ):
            raise WorkflowError("host verification report has no valid kernel identity")
        codes = ",".join(finding.code for finding in report.findings)
        root = self.options.source_root
        report_path = root / "reports" / f"qualification-{kernel}.json"
        qualification = shlex.join(
            (
                str(root / "bin/container-check"),
                "--suite",
                "stable",
                "--profile",
                str(root / "profiles/qualification/stable.toml"),
                "--json",
                str(report_path),
            )
        )
        return (
            f"HOST_VERIFY unverified on {kernel} ({codes}); "
            "post-reboot host checks passed, so installation will continue; "
            "IMAGE_VERIFY will still enforce the gfx1151 GPU runtime. "
            "Optional full qualification after image acquisition and before "
            "release promotion: "
            f"sudo -v && {qualification}"
        )

    def _record_reports(
        self, state: InstallState, outcome: StageResult
    ) -> InstallState:
        reports = tuple(
            dict.fromkeys((*state.last_report_paths, *outcome.report_paths))
        )
        return replace(
            state,
            last_report_paths=reports,
            updated_at=_utc_timestamp(),
        )

    def _checkpoint(
        self,
        state: InstallState,
        *,
        stage: InstallStage,
        inputs: object,
        next_stage: InstallStage,
    ) -> InstallState:
        completed = dict(state.completed_stage_input_digests)
        completed[stage.value] = stage_input_digest(inputs)
        return replace(
            state,
            current_stage=next_stage,
            completed_stage_input_digests=completed,
            updated_at=_utc_timestamp(),
        )

    def _validate_transition(self, state: InstallState) -> None:
        if state.mode is not self.options.mode:
            raise WorkflowError(
                "illegal installer transition: mode changed; "
                f"state: {self.options.state_path}"
            )
        order = self._stage_order()
        allowed = {stage.value for stage in order}
        unknown = set(state.completed_stage_input_digests).difference(allowed)
        if unknown:
            raise WorkflowError(
                "illegal installer transition: completed stages belong to another mode"
            )
        seen_incomplete = False
        for stage in order:
            complete = stage.value in state.completed_stage_input_digests
            if seen_incomplete and complete:
                raise WorkflowError(
                    "illegal installer transition: completed stages are not a prefix"
                )
            if not complete:
                seen_incomplete = True
        first_incomplete = next(
            (
                stage
                for stage in order
                if stage.value not in state.completed_stage_input_digests
            ),
            InstallStage.COMPLETE,
        )
        if state.current_stage is not first_incomplete:
            raise WorkflowError(
                "illegal installer transition: current stage does not match checkpoint"
            )

    def _disk_requirement(
        self, stage: InstallStage, state: InstallState
    ) -> DiskRequirement | None:
        if stage is InstallStage.IMAGE_PULL_OR_BUILD:
            return self._image_disk_requirement(
                self._resolved_release(state),
                image_source=self.options.image_source or "pull",
            )
        if stage is InstallStage.PROJECT_INIT:
            hook = getattr(self.actions, "project_disk_estimate", None)
            if hook is None:
                raise WorkflowError("project disk estimate is unavailable")
            estimate = hook(project_dir=self.options.project_dir)
            _validate_disk_estimate(estimate, "project generation")
            return DiskRequirement(
                operation="project generation",
                source="项目文件系统",
                payload_label="项目数据",
                estimate=estimate,
                required_bytes=estimate.payload_bytes * 2 + GIB,
            )
        return None

    def _image_disk_requirement(
        self,
        release: StableRelease,
        *,
        image_source: str,
    ) -> DiskRequirement:
        hook = getattr(self.actions, "image_disk_estimate", None)
        if hook is None:
            raise WorkflowError("image disk estimate is unavailable")
        estimate = hook(
            release=release,
            image_source=image_source,
            registry=self.options.registry,
        )
        operation = (
            "image build" if image_source == "build" else "image acquisition"
        )
        _validate_disk_estimate(estimate, operation)
        return DiskRequirement(
            operation=operation,
            source=(
                "本地源码构建"
                if image_source == "build"
                else {
                    "auto": "公开镜像（华为 SWR 优先，GHCR 回退）",
                    "swr": "公开镜像（仅华为 SWR）",
                    "ghcr": "公开镜像（仅 GHCR）",
                }[self.options.registry]
            ),
            payload_label=("构建估算" if image_source == "build" else "缺失层"),
            estimate=estimate,
            required_bytes=estimate.payload_bytes + 5 * GIB,
        )

    def _report_disk_requirement(
        self, requirement: DiskRequirement
    ) -> None:
        estimate = requirement.estimate
        self.progress.detail(
            f"{requirement.payload_label}="
            f"{_format_gib(estimate.payload_bytes)} GiB，"
            f"需要={_format_gib(requirement.required_bytes)} GiB，"
            f"可用={_format_gib(estimate.available_bytes)} GiB，"
            f"位置={estimate.location}，来源={requirement.source}"
        )
        self.progress.debug(
            f"disk operation={requirement.operation} "
            f"payload_bytes={estimate.payload_bytes} "
            f"required_bytes={requirement.required_bytes} "
            f"available_bytes={estimate.available_bytes} "
            f"location={estimate.location} source={requirement.source}"
        )

    def _report_release_identity(self, output: object) -> None:
        if not isinstance(output, StableRelease):
            raise WorkflowError("release action returned an invalid value")
        self.progress.detail(f"stable release={output.release_id}")
        self.progress.debug(
            f"base reference={output.base.reference} "
            f"manifest_digest={output.base.manifest_digest} "
            f"config_digest={output.base.config_digest}"
        )
        self.progress.debug(
            f"torch reference={output.torch.reference} "
            f"manifest_digest={output.torch.manifest_digest} "
            f"config_digest={output.torch.config_digest}"
        )

    def _resolved_release(self, state: InstallState) -> StableRelease:
        if self._release is None:
            self._release = load_stable_release(self.options.manifest_path)
        if (
            state.release_id != self._release.release_id
            or state.source_revision != self._release.source_revision
        ):
            raise WorkflowError("persisted release identity does not match manifest")
        return self._release

    def _resolved_host_plan(self, state: InstallState) -> HostPlanResult:
        if state.target_user is None or state.host_plan_digest is None:
            raise WorkflowError("persisted host plan identity is unavailable")
        if self._host_plan is None:
            self._host_plan = self.actions.host_plan(
                target_user=state.target_user,
                phase=HostPlanPhase.TUNING,
            )
        if (
            self._host_plan.plan_digest != state.host_plan_digest
            or self._host_plan.adapter_id != state.host_adapter_id
        ):
            raise WorkflowError(
                "host plan digest changed after authorization; replan is required"
            )
        return self._host_plan

    def _resolved_kernel_plan(self, state: InstallState) -> HostPlanResult:
        if state.target_user is None or state.kernel_plan_digest is None:
            raise WorkflowError("persisted kernel plan identity is unavailable")
        if self._kernel_plan is None:
            self._kernel_plan = self.actions.host_plan(
                target_user=state.target_user,
                phase=HostPlanPhase.KERNEL,
            )
        if (
            self._kernel_plan.plan_digest != state.kernel_plan_digest
            or self._kernel_plan.adapter_id != state.host_adapter_id
            or self._kernel_plan.plan.phase is not HostPlanPhase.KERNEL
        ):
            raise WorkflowError(
                "kernel plan digest changed after authorization; replan is required"
            )
        return self._kernel_plan

    def _confirm_kernel_plan(self, state: InstallState) -> HostConfirmation:
        kernel_plan = self._resolved_kernel_plan(state)
        for action in kernel_plan.plan.actions:
            self._status("ACTION", f"{action.code}: {action.summary}")
        self._status(
            "ACTION",
            "Recovery kernel retained: "
            f"{kernel_plan.running_kernel}. If the desktop fails, reboot and "
            "select it under Advanced options for Ubuntu.",
        )
        if self.options.non_interactive:
            if self.options.accepted_kernel_plan_digest != (
                kernel_plan.plan_digest
            ):
                return HostConfirmation(
                    False,
                    False,
                    "kernel plan requires --accept-kernel-plan-digest "
                    f"{kernel_plan.plan_digest}",
                )
            return HostConfirmation(True, False)
        self.progress.pause_heartbeat()
        try:
            accepted = self.prompts.confirm_exact("INSTALL-KERNEL")
        finally:
            self.progress.resume_heartbeat()
        if not accepted:
            return HostConfirmation(
                False,
                False,
                "kernel plan confirmation refused; exact INSTALL-KERNEL is required",
            )
        return HostConfirmation(True, False)

    def _confirm_host_plan(self, state: InstallState) -> HostConfirmation:
        host_plan = self._resolved_host_plan(state)
        for action in host_plan.plan.actions:
            self._status("ACTION", f"{action.code}: {action.summary}")
        if self.options.non_interactive:
            if (
                self.options.accepted_host_plan_digest
                != host_plan.plan_digest
            ):
                return HostConfirmation(
                    False,
                    False,
                    "host platform plan requires --accept-host-plan-digest "
                    f"{host_plan.plan_digest}",
                )
            return HostConfirmation(
                True,
                self.options.accept_docker_group,
            )
        self.progress.pause_heartbeat()
        try:
            accepted = self.prompts.confirm_exact("APPLY")
        finally:
            self.progress.resume_heartbeat()
        if not accepted:
            return HostConfirmation(
                False,
                False,
                "host plan confirmation refused; exact APPLY is required",
            )
        docker_group = False
        if host_plan.plan.target_user != "root":
            self.progress.pause_heartbeat()
            try:
                docker_group = self.prompts.confirm_yes_no("docker-group")
            finally:
                self.progress.resume_heartbeat()
        return HostConfirmation(True, docker_group)

    def _reboot_pending(
        self,
        previous_boot_id: str | None,
        *,
        message: str,
    ) -> StageResult:
        if previous_boot_id is None:
            return StageResult()
        current = self._boot_id_reader()
        if boot_id_changed(previous_boot_id, current_boot_id=current):
            return StageResult()
        return StageResult(action_required=True, message=message)

    def _stage_order(self) -> tuple[InstallStage, ...]:
        if self.options.mode is InstallMode.FULL:
            return FULL_STAGE_ORDER
        if self.options.mode is InstallMode.CONTAINER:
            return CONTAINER_STAGE_ORDER
        raise WorkflowError("doctor mode has no install stage order")

    def _status(self, prefix: str, message: str) -> None:
        self.progress.status(prefix, message)


def _validate_disk_estimate(
    estimate: object, operation: str
) -> None:
    if not isinstance(estimate, DiskSpaceEstimate):
        raise WorkflowError(f"{operation} returned an invalid disk estimate")


def _report_kernel(report: Report, label: str) -> str:
    kernel = report.facts.get("kernel")
    if not isinstance(kernel, str) or re.fullmatch(
        r"[0-9A-Za-z][0-9A-Za-z.+_-]{0,127}", kernel
    ) is None:
        raise WorkflowError(f"{label} report has no valid kernel identity")
    return kernel


def _reference_digest(reference: str) -> str:
    _, separator, digest = reference.rpartition("@")
    if separator != "@" or not digest:
        raise WorkflowError(
            f"verified image reference has no manifest digest: {reference}"
        )
    return digest


def _report_finding_message(report: Report) -> str:
    if not report.findings:
        return "no diagnostic findings were returned"
    return " | ".join(
        f"{finding.code}: {finding.summary}; action: {finding.remediation}"
        for finding in report.findings
    )


def _require_disk_space(requirement: DiskRequirement) -> str | None:
    estimate = requirement.estimate
    if estimate.available_bytes > requirement.required_bytes:
        return None
    return (
        f"{requirement.operation} disk space is insufficient at "
        f"{estimate.location}: required_bytes={requirement.required_bytes}, "
        f"available_bytes={estimate.available_bytes}"
    )


def _format_gib(byte_count: int) -> str:
    return f"{byte_count / GIB:.1f}"


def _file_digest(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise WorkflowError(f"cannot hash stage input {path}: {error}") from error
    return hashlib.sha256(data).hexdigest()


def _installer_series(version: str) -> tuple[int, int] | None:
    match = re.fullmatch(
        r"(\d+)\.(\d+)\.\d+(?:[+.-][0-9A-Za-z.-]+)?",
        version,
    )
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _utc_timestamp() -> str:
    return (
        datetime.now(UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
