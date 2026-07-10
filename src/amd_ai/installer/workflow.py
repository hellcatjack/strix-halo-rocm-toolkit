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
from amd_ai.installer.release import (
    ReleaseAcquisitionError,
    ReleaseIdentityError,
    load_stable_release,
)
from amd_ai.installer.state import (
    CorruptInstallState,
    InstallAlreadyRunning,
    InstallerStateError,
    boot_id_changed,
    install_lock,
    load_state,
    read_boot_id,
    save_state,
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


@dataclass(frozen=True)
class HostConfirmation:
    accepted: bool
    docker_group_accepted: bool
    message: str = ""


class InstallerWorkflow:
    def __init__(
        self,
        *,
        options: InstallOptions,
        actions: InstallerActions,
        installer_version: str,
        installer_source_revision: str,
        prompts: object | None = None,
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
        self._release: StableRelease | None = None
        self._host_plan: HostPlanResult | None = None
        self._boot_id_reader = boot_id_reader

    def run(self) -> WorkflowResult:
        state: InstallState | None = None
        try:
            self._prepare_options()
            self.options.validate()
            if REVISION_PATTERN.fullmatch(
                self.installer_source_revision
            ) is None:
                raise WorkflowError("installer source revision is invalid")
            with install_lock(self.options.state_path):
                state = load_state(self.options.state_path)
                if state is None:
                    state = self._new_state()
                    if not self.options.dry_run:
                        save_state(self.options.state_path, state)
                elif not self.options.dry_run:
                    state = self._adopt_compatible_installer_update(state)
                self._validate_transition(state)
                if self.options.dry_run:
                    return self._run_dry(state)
                return self._run_stages(state)
        except KeyboardInterrupt:
            return WorkflowResult(1, state, "installation interrupted")
        except (
            CorruptInstallState,
            InstallAlreadyRunning,
            InstallerModelError,
            InstallerStateError,
            PromptError,
            WorkflowError,
        ) as error:
            return WorkflowResult(2, state, str(error))
        except Exception as error:
            return WorkflowResult(2, state, f"installer failed: {error}")

    def _prepare_options(self) -> None:
        if self.options.mode is InstallMode.DOCTOR:
            raise WorkflowError("doctor mode is routed outside the install workflow")
        if self.options.project_dir is None:
            project_dir = self.prompts.ask_project_dir()
            self.options = replace(self.options, project_dir=project_dir)

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
            inputs = self._stage_inputs(stage, state)
            if validate_completed_stage(state, stage, inputs):
                continue
            if state.current_stage is not stage:
                raise WorkflowError(
                    f"illegal installer transition: state={state.current_stage.value}, "
                    f"next={stage.value}"
                )
            shortage = self._disk_shortage(stage, state)
            if shortage is not None:
                return WorkflowResult(2, state, shortage)
            try:
                output = self._dispatch(stage, state)
            except KeyboardInterrupt:
                return WorkflowResult(1, state, f"{stage.value} interrupted")
            except Exception as error:
                return WorkflowResult(
                    2, state, f"{stage.value} failed: {error}"
                )

            outcome = self._stage_result(stage, output)
            if outcome.blocked:
                state = self._record_reports(state, outcome)
                save_state(self.options.state_path, state)
                return WorkflowResult(
                    2,
                    state,
                    outcome.message or f"{stage.value} is blocked",
                )
            if (
                stage is InstallStage.REBOOT_PENDING
                and outcome.action_required
            ):
                state = self._record_reports(state, outcome)
                save_state(self.options.state_path, state)
                return WorkflowResult(
                    1,
                    state,
                    outcome.message or "manual reboot is required",
                )
            state = self._apply_output(state, stage, output, outcome)
            state = self._checkpoint(
                state,
                stage=stage,
                inputs=inputs,
                next_stage=(order[index + 1] if index + 1 < len(order) else stage),
            )
            save_state(self.options.state_path, state)
            if (
                stage is InstallStage.HOST_VERIFY
                and isinstance(output, Report)
                and output.status is Status.UNVERIFIED
            ):
                self._status("WARN", outcome.message)
            else:
                self._status("PASS", stage.value)
            if outcome.action_required:
                return WorkflowResult(
                    1,
                    state,
                    outcome.message or f"{stage.value} requires operator action",
                )
        message = "installation complete"
        if state.host_verification_status == Status.UNVERIFIED.value:
            message += (
                f"; host remains unverified on {state.host_kernel}; "
                "run the displayed full qualification before release promotion"
            )
        return WorkflowResult(0, state, message)

    def _run_dry(self, state: InstallState) -> WorkflowResult:
        mutating = {
            InstallStage.HOST_APPLY,
            InstallStage.IMAGE_PULL_OR_BUILD,
            InstallStage.IMAGE_VERIFY,
            InstallStage.PROJECT_INIT,
            InstallStage.PROJECT_VERIFY,
        }
        for stage in self._stage_order():
            prefix = "ACTION" if stage in mutating else "PASS"
            self._status(prefix, f"dry-run {stage.value}")
        return WorkflowResult(0, state, "dry-run complete; no stages persisted")

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
        if stage is InstallStage.HOST_PLAN:
            if state.target_user is None:
                raise WorkflowError("host target user is unavailable")
            self._host_plan = self.actions.host_plan(
                target_user=state.target_user
            )
            return self._host_plan
        if stage is InstallStage.HOST_CONFIRM:
            return self._confirm_host_plan(state)
        if stage is InstallStage.HOST_APPLY:
            return self.actions.host_apply(
                self._resolved_host_plan(state),
                include_docker_group=state.docker_group_accepted,
            )
        if stage is InstallStage.REBOOT_PENDING:
            if state.reboot_boot_id is None:
                return StageResult()
            current = self._boot_id_reader()
            if not boot_id_changed(
                state.reboot_boot_id, current_boot_id=current
            ):
                return StageResult(
                    action_required=True,
                    message=(
                        "manual reboot is required; rerun the installer after reboot"
                    ),
                )
            return StageResult()
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
                    return self.actions.pull_release(release)
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
                    fallback = self.prompts.choose_image_fallback()
                    if fallback != "build":
                        raise WorkflowError(
                            f"exact release pull failed and local build was refused: {error}"
                        ) from error
                    self._status(
                        "WARN",
                        "exact release pull failed; using explicit local build",
                    )
                    estimate = self.actions.image_disk_estimate(
                        release=release, image_source="build"
                    )
                    shortage = _require_disk_space(
                        estimate,
                        required_bytes=estimate.payload_bytes + 5 * GIB,
                        operation="image build",
                    )
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
            InstallStage.HOST_PLAN,
            InstallStage.CONTAINER_HOST_CHECK,
        }:
            values.update(
                {
                    "target_user": state.target_user,
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
        elif stage in {
            InstallStage.REBOOT_PENDING,
            InstallStage.HOST_VERIFY,
        }:
            values.update(
                {
                    "host_plan_digest": state.host_plan_digest,
                    "host_adapter_id": state.host_adapter_id,
                    "reboot_boot_id": state.reboot_boot_id,
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
            values.update(
                {
                    "release_id": state.release_id,
                    "source_revision": state.source_revision,
                    "base_reference": state.base_image_reference,
                    "base_manifest_digest": state.base_manifest_digest,
                    "base_config_digest": state.base_config_digest,
                    "torch_reference": state.torch_image_reference,
                    "torch_manifest_digest": state.torch_manifest_digest,
                    "torch_config_digest": state.torch_config_digest,
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
            elif blocked:
                message = f"{output.command} returned {output.status.value}"
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
        if stage is InstallStage.HOST_PLAN:
            if not isinstance(output, HostPlanResult):
                raise WorkflowError("host plan action returned an invalid value")
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
        elif stage is InstallStage.HOST_APPLY:
            plan = self._resolved_host_plan(state).plan
            changes["reboot_boot_id"] = (
                self._boot_id_reader() if plan.reboot_required else None
            )
        elif stage is InstallStage.HOST_VERIFY and isinstance(output, Report):
            kernel = output.facts.get("kernel")
            if not isinstance(kernel, str) or not kernel:
                raise WorkflowError("host verification report has no kernel identity")
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
            output, LocalBuildResult
        ):
            changes.update(
                {
                    "release_id": "local",
                    "base_image_reference": output.base_reference,
                    "base_manifest_digest": output.base_config_digest,
                    "base_config_digest": output.base_config_digest,
                    "torch_image_reference": output.torch_reference,
                    "torch_manifest_digest": output.torch_config_digest,
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
        if state.mode is not InstallMode.FULL:
            return False
        start = FULL_STAGE_ORDER.index(InstallStage.HOST_VERIFY)
        if state.current_stage not in FULL_STAGE_ORDER[start:]:
            return False
        previous_series = _installer_series(state.installer_version)
        return previous_series is not None and previous_series == _installer_series(
            self.installer_version
        )

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
            raise WorkflowError("illegal installer transition: mode changed")
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

    def _disk_shortage(
        self, stage: InstallStage, state: InstallState
    ) -> str | None:
        if stage is InstallStage.IMAGE_PULL_OR_BUILD:
            hook = getattr(self.actions, "image_disk_estimate", None)
            if hook is None:
                raise WorkflowError("image disk estimate is unavailable")
            estimate = hook(
                release=self._resolved_release(state),
                image_source=self.options.image_source or "pull",
            )
            return _require_disk_space(
                estimate,
                required_bytes=estimate.payload_bytes + 5 * GIB,
                operation="image acquisition",
            )
        if stage is InstallStage.PROJECT_INIT:
            hook = getattr(self.actions, "project_disk_estimate", None)
            if hook is None:
                raise WorkflowError("project disk estimate is unavailable")
            estimate = hook(project_dir=self.options.project_dir)
            return _require_disk_space(
                estimate,
                required_bytes=estimate.payload_bytes * 2 + GIB,
                operation="project generation",
            )
        return None

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
                target_user=state.target_user
            )
        if (
            self._host_plan.plan_digest != state.host_plan_digest
            or self._host_plan.adapter_id != state.host_adapter_id
        ):
            raise WorkflowError(
                "host plan digest changed after authorization; replan is required"
            )
        return self._host_plan

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
                    "accepted host plan digest does not match the current plan",
                )
            return HostConfirmation(
                True,
                self.options.accept_docker_group,
            )
        if not self.prompts.confirm_exact("APPLY"):
            return HostConfirmation(
                False,
                False,
                "host plan confirmation refused; exact APPLY is required",
            )
        docker_group = False
        if host_plan.plan.target_user != "root":
            docker_group = self.prompts.confirm_yes_no("docker-group")
        return HostConfirmation(True, docker_group)

    def _stage_order(self) -> tuple[InstallStage, ...]:
        if self.options.mode is InstallMode.FULL:
            return FULL_STAGE_ORDER
        if self.options.mode is InstallMode.CONTAINER:
            return CONTAINER_STAGE_ORDER
        raise WorkflowError("doctor mode has no install stage order")

    def _status(self, prefix: str, message: str) -> None:
        status = getattr(self.prompts, "status", None)
        if status is not None:
            status(prefix, message)


def _require_disk_space(
    estimate: DiskSpaceEstimate,
    *,
    required_bytes: int,
    operation: str,
) -> str | None:
    if not isinstance(estimate, DiskSpaceEstimate):
        raise WorkflowError(f"{operation} returned an invalid disk estimate")
    if estimate.available_bytes > required_bytes:
        return None
    return (
        f"{operation} disk space is insufficient at {estimate.location}: "
        f"required_bytes={required_bytes}, available_bytes={estimate.available_bytes}"
    )


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
