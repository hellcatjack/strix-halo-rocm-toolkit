# Read-Only GTT/TTM Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish toolkit v0.3.0 with an OEM 6.17 desktop-safe kernel checkpoint while permanently removing every host GTT/TTM mutation and second TTM reboot.

**Architecture:** Keep existing TTM and kernel-command-line facts in `HostSnapshot` as read-only diagnostics. Remove TTM actions from host planning/apply and remove TTM enforcement from final verification; then finish the phase-aware installer with one kernel reboot checkpoint followed by non-rebooting Docker/Buildx/group preparation.

**Tech Stack:** Python 3.12, dataclasses, StrEnum, Ubuntu APT/systemd, Docker Engine/Buildx, pytest 8.4, uv, Bash, Markdown.

---

## File Map

| Area | Files | Responsibility |
|---|---|---|
| Host planning | `src/amd_ai/host/prepare.py`, `src/amd_ai/host/adapters/*.py` | Generate kernel and platform plans without memory or TTM write inputs |
| Privileged apply | `src/amd_ai/host/apply.py` | Execute only audited APT, Docker, group, and backup actions |
| Read-only verification | `src/amd_ai/host/verify.py`, `src/amd_ai/host/probe.py`, `src/amd_ai/host/policy.py` | Retain observed TTM facts but never derive a required host limit |
| Installer protocol | `src/amd_ai/installer/actions.py`, `src/amd_ai/installer/privileged.py` | Carry explicit kernel/platform phases without `memory_gib` |
| Installer workflow | `src/amd_ai/installer/models.py`, `workflow.py`, `progress.py`, `state.py` | Persist one kernel reboot checkpoint and non-rebooting platform apply |
| User CLI | `src/amd_ai/cli.py`, `install.sh` | Expose phase approvals and reject removed memory tuning arguments |
| Fixtures/tests | `tests/unit/host`, `tests/unit/installer`, `tests/cli`, `tests/fixtures` | Prove no TTM write or second reboot remains |
| Documentation/version | `README.md`, `docs/*.md`, `src/amd_ai/__init__.py` | Publish v0.3.0 behavior and read-only diagnostic contract |

## Task 1: Remove Host TTM Planning and Apply

**Files:**
- Modify: `src/amd_ai/host/adapters/base.py`
- Modify: `src/amd_ai/host/adapters/ubuntu_2404.py`
- Modify: `src/amd_ai/host/prepare.py`
- Modify: `src/amd_ai/host/apply.py`
- Test: `tests/unit/host/test_prepare.py`
- Test: `tests/unit/host/test_apply.py`

- [ ] **Step 1: Add planner tests that prohibit every TTM mutation**

Add this test to `tests/unit/host/test_prepare.py`:

```python
def test_platform_plan_is_read_only_for_gtt_and_never_reboots():
    plan = create_prepare_plan(
        healthy_snapshot(ttm_pages_limit=1),
        target_user="customer",
        phase=HostPlanPhase.TUNING,
    )
    codes = action_codes(plan)
    assert not any(code.startswith("TTM.") for code in codes)
    assert "HOST.REBOOT" not in codes
    assert plan.reboot_required is False
    assert "amd-ttm" not in " ".join(
        argument for action in plan.actions for argument in action.argv
    )
```

- [ ] **Step 2: Run the planner test and verify the existing TTM actions fail it**

Run:

```bash
uv run pytest tests/unit/host/test_prepare.py::test_platform_plan_is_read_only_for_gtt_and_never_reboots -q
```

Expected: fail because `TTM.INSTALL_AMD_DEBUG_TOOLS`, `TTM.SET_AI_MAX`, and `HOST.REBOOT` are present.

- [ ] **Step 3: Remove memory inputs and TTM actions from the planner**

Use these signatures throughout the adapter boundary:

```python
def create_prepare_plan(
    snapshot: HostSnapshot,
    *,
    target_user: str,
    phase: HostPlanPhase | None = None,
) -> PreparePlan:
    ...


def create_tuning_prepare_plan(
    snapshot: HostSnapshot,
    target_user: str,
) -> PreparePlan:
    ...
```

The tuning plan ends after Docker/Buildx and GPU group actions and returns:

```python
return PreparePlan(
    phase=HostPlanPhase.TUNING,
    supported=True,
    target_user=target_user,
    actions=tuple(actions),
    reboot_required=False,
)
```

Remove `python3-pip` and `pipx` from `APT.INSTALL_HOST_TOOLS` because the planner no longer installs `amd-debug-tools`.

- [ ] **Step 4: Delete internal TTM apply dispatch and write helpers**

Remove these symbols and their imports/constants from `host/apply.py`:

```text
AMD_DEBUG_TOOLS_VERSION
AMD_DEBUG_TOOLS_WHEEL
AMD_DEBUG_TOOLS_SHA256
ttm_input_text
_install_amd_debug_tools
_set_ttm
_eligible_ttm_memory_fallback
_write_ttm_fallback
_read_ttm_pages
```

Keep TTM paths in `BACKUP_FILES`; reading an existing configuration is diagnostic evidence and does not mutate it.

- [ ] **Step 5: Replace mutation tests with an unknown-action refusal test**

Add:

```python
@pytest.mark.parametrize("code", ["TTM.SET_AI_MAX", "TTM.INSTALL_AMD_DEBUG_TOOLS"])
def test_removed_ttm_actions_are_never_dispatchable(tmp_path, code):
    action = PlannedAction(code=code, summary="removed", argv=(), privileged=True)
    with pytest.raises(ApplyError, match="unknown internal action"):
        execute_plan(
            action_plan(backup_action(), action),
            FakeRunner.backup_only(),
            effective_uid=0,
            confirmed=True,
            snapshot=healthy_snapshot(),
            root=tmp_path / "root",
            backup_destination=tmp_path / "backups",
        )
```

- [ ] **Step 6: Run host planning/apply tests**

```bash
uv run pytest tests/unit/host/test_prepare.py tests/unit/host/test_apply.py -q
```

Expected: all pass and `rg -n 'TTM\.|amd-ttm|ttm.conf' src/amd_ai/host/prepare.py src/amd_ai/host/apply.py` finds no write path.

- [ ] **Step 7: Commit**

```bash
git add src/amd_ai/host/adapters src/amd_ai/host/prepare.py \
  src/amd_ai/host/apply.py tests/unit/host
git commit -m "refactor: remove host GTT and TTM mutation"
```

## Task 2: Make Final Host Verification TTM-Neutral

**Files:**
- Modify: `src/amd_ai/host/verify.py`
- Test: `tests/unit/host/test_verify.py`

- [ ] **Step 1: Add a failing test for mismatched and missing live limits**

```python
@pytest.mark.parametrize("live_limit", [None, 0, 1, 8183158])
def test_final_verification_treats_ttm_limit_as_diagnostic_only(live_limit):
    report = evaluate_post_reboot(
        healthy_snapshot(
            kernel="6.17.0-1028-oem",
            ttm_pages_limit=live_limit,
        )
    )
    assert report.status is Status.PASS
    assert report.facts["ttm_pages_limit"] == live_limit
    assert "HOST.TTM_MISMATCH" not in finding_codes(report)
    assert "HOST.MEMORY_CONFLICT" not in finding_codes(report)
```

- [ ] **Step 2: Run and observe the current mismatch blocker**

```bash
uv run pytest tests/unit/host/test_verify.py::test_final_verification_treats_ttm_limit_as_diagnostic_only -q
```

Expected: fail with `reboot-required` and `HOST.TTM_MISMATCH`.

- [ ] **Step 3: Remove TTM target computation from final verification**

Delete `MemoryConflict` and `compute_ttm_plan` imports and the full TTM computation block from `evaluate_post_reboot()`. Preserve `preflight.facts`, which already contains the observed `ttm_pages_limit`.

- [ ] **Step 4: Run host verification and kernel log tests**

```bash
uv run pytest tests/unit/host/test_verify.py \
  tests/unit/qualification/test_kernel_log.py -q
```

Expected: all pass; kernel, display, and GPU log failures remain blocking.

- [ ] **Step 5: Commit**

```bash
git add src/amd_ai/host/verify.py tests/unit/host/test_verify.py
git commit -m "refactor: make TTM facts diagnostic only"
```

## Task 3: Remove Memory Tuning from CLI and Privileged Protocol

**Files:**
- Modify: `src/amd_ai/cli.py`
- Modify: `src/amd_ai/installer/actions.py`
- Modify: `src/amd_ai/installer/privileged.py`
- Test: `tests/cli/test_host_commands.py`
- Test: `tests/unit/installer/test_actions.py`
- Test: `tests/unit/installer/test_privileged.py`

- [ ] **Step 1: Add parser rejection tests**

```python
def test_host_prepare_rejects_removed_memory_gib_option():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["host-prepare", "plan", "--target-user", "developer", "--memory-gib", "128"]
        )


def test_privileged_helper_rejects_removed_memory_gib_option():
    with pytest.raises(SystemExit):
        privileged.build_parser().parse_args(
            ["--target-user", "developer", "--phase", "tuning", "--memory-gib", "128", "plan"]
        )
```

- [ ] **Step 2: Run parser tests and verify both parsers still accept the option**

```bash
uv run pytest tests/cli/test_host_commands.py \
  tests/unit/installer/test_privileged.py -q
```

Expected: new parser rejection tests fail.

- [ ] **Step 3: Remove `memory_gib` from public and privileged APIs**

Use this complete action signature:

```python
def host_plan(
    self,
    *,
    target_user: str,
    phase: HostPlanPhase,
) -> HostPlanResult:
    ...
```

Remove `--memory-gib` from both parsers, from sudo command construction, from strict argument checks, and from every call to `create_prepare_plan()`.

- [ ] **Step 4: Run protocol regressions**

```bash
uv run pytest tests/unit/installer/test_actions.py \
  tests/unit/installer/test_privileged.py tests/cli/test_host_commands.py -q
```

Expected: all pass and phase/digest drift remains blocked before mutation.

- [ ] **Step 5: Commit**

```bash
git add src/amd_ai/cli.py src/amd_ai/installer/actions.py \
  src/amd_ai/installer/privileged.py tests/cli/test_host_commands.py \
  tests/unit/installer/test_actions.py tests/unit/installer/test_privileged.py
git commit -m "refactor: remove memory tuning interfaces"
```

## Task 4: Finish the Single-Reboot Installer Workflow

**Files:**
- Modify: `src/amd_ai/installer/models.py`
- Modify: `src/amd_ai/installer/workflow.py`
- Modify: `src/amd_ai/installer/progress.py`
- Modify: `src/amd_ai/installer/state.py`
- Modify: `tests/unit/installer/fakes.py`
- Test: `tests/unit/installer/test_models.py`
- Test: `tests/unit/installer/test_workflow.py`
- Test: `tests/unit/installer/test_progress.py`

- [ ] **Step 1: Set the exact full-stage order without the legacy second reboot**

```python
FULL_STAGE_ORDER = (
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
```

Remove `InstallStage.REBOOT_PENDING` and its progress label. Keep the schema field `reboot_boot_id` only for v1/v2 migration compatibility and require new workflows to leave it `None`.

- [ ] **Step 2: Add ordered checkpoint tests**

```python
def test_old_kernel_stops_before_platform_plan_until_kernel_reboot(tmp_path):
    actions = FakeInstallerActions.host_change_requires_kernel_reboot()
    first = installer_workflow(
        tmp_path,
        actions=actions,
        options=full_options(tmp_path),
        prompts=FakePrompts(exact={"INSTALL-KERNEL": True}),
        boot_id_reader=lambda: FIRST_BOOT_ID,
    ).run()
    assert first.exit_code == 1
    assert first.state.current_stage is InstallStage.KERNEL_REBOOT_PENDING
    assert "host_plan" not in actions.calls


def test_platform_apply_never_creates_second_reboot_checkpoint(tmp_path):
    actions = FakeInstallerActions.full_no_kernel_reboot()
    result = installer_workflow(
        tmp_path,
        actions=actions,
        options=full_options(tmp_path),
        prompts=FakePrompts(exact={"INSTALL-KERNEL": True, "APPLY": True}),
    ).run()
    assert result.exit_code == 0
    assert result.state.reboot_boot_id is None
    assert "host_verify" in actions.calls
    assert actions.calls.index("host_verify") < actions.calls.index("pull_release")
```

- [ ] **Step 3: Implement independent approvals and recovery persistence**

Retain `accepted_kernel_plan_digest` and `accepted_host_plan_digest`. Persist these kernel apply facts only:

```python
changes.update(
    {
        "kernel_reboot_boot_id": (
            self._boot_id_reader() if plan.plan.reboot_required else None
        ),
        "recovery_kernel": plan.running_kernel,
        "display_manager_was_active": plan.display_manager_active,
    }
)
```

Platform apply never writes `reboot_boot_id`. `HOST_VERIFY` follows `HOST_APPLY` immediately.

- [ ] **Step 4: Keep update adoption deliberate**

Allow cross-minor `0.2.x -> 0.3.x` adoption only for migrated container mode with a reconstructable Bootstrap digest. Full-mode schema-2 states restart at Bootstrap, preserve immutable image references, and never reuse a host approval.

- [ ] **Step 5: Run workflow, state, and progress tests**

```bash
uv run pytest tests/unit/installer/test_models.py \
  tests/unit/installer/test_state.py tests/unit/installer/test_workflow.py \
  tests/unit/installer/test_progress.py -q
```

Expected: all pass with 17 full stages and 8 unchanged container stages.

- [ ] **Step 6: Commit**

```bash
git add src/amd_ai/installer/models.py src/amd_ai/installer/workflow.py \
  src/amd_ai/installer/progress.py src/amd_ai/installer/state.py \
  tests/unit/installer
git commit -m "feat: use a single kernel reboot checkpoint"
```

## Task 5: Integrate Fixtures, CLI, and Active Documentation

**Files:**
- Modify: `src/amd_ai/installer/fixture.py`
- Modify: `tests/fixtures/installer/*.json`
- Modify: `tests/fixtures/installer/README.md`
- Modify: `tests/cli/test_installer_resume.py`
- Modify: `README.md`
- Modify: `docs/install.md`
- Modify: `docs/host-operations.md`
- Modify: `docs/project-workflow.md`
- Modify: `tests/container/test_image_contract.py`
- Modify: `src/amd_ai/__init__.py`

- [ ] **Step 1: Make fixture plans phase-specific**

`FixtureInstallerActions.host_plan()` must accept `phase`, return a matching `PreparePlan.phase`, and expose `running_kernel` plus `display_manager_active`. `kernel_verify()` records a separate call and returns a `host-kernel-verify` report. No fixture scenario contains a TTM reboot.

- [ ] **Step 2: Add end-to-end fixture assertions**

```python
assert "kernel_apply" in calls
assert calls.index("kernel_verify") < calls.index("host_plan")
assert "REBOOT_PENDING" not in calls
assert calls.index("host_verify") < calls.index("pull_release")
```

Run:

```bash
uv run pytest tests/cli/test_installer_resume.py -q
```

Expected: full mode resumes once after the kernel boot and completes without a second reboot.

- [ ] **Step 3: Rewrite active documentation around read-only diagnostics**

Active docs must state all of the following verbatim or equivalently:

```text
The toolkit does not install amd-debug-tools, invoke amd-ttm, write ttm.conf,
set ttm.pages_limit or amdgpu.gttsize, or request a GTT/TTM reboot.
Observed GTT/TTM values are diagnostic facts only.
```

Remove `--memory-gib` examples, TTM rollback instructions for tool-created files, the second reboot description, and any claim that host verification requires a computed TTM target. Keep BIOS UMA guidance as manual firmware advice.

- [ ] **Step 4: Set toolkit version 0.3.0 without rebuilding stable images**

Set:

```python
__version__ = "0.3.0"
```

Keep `profiles/releases/stable.json` at release ID `0.2.0` and preserve its immutable ROCm 7.2.1/PyTorch 2.9.1 image digests.

- [ ] **Step 5: Run CLI and documentation contracts**

```bash
uv run pytest tests/cli tests/container/test_image_contract.py -q
```

Expected: all pass; quick start documents one kernel reboot and read-only GTT/TTM facts.

- [ ] **Step 6: Commit**

```bash
git add src/amd_ai/installer/fixture.py src/amd_ai/__init__.py \
  tests/fixtures/installer tests/cli README.md docs \
  tests/container/test_image_contract.py
git commit -m "docs: publish the read-only GTT v0.3 workflow"
```

## Task 6: Full Verification and OEM 6.17 Qualification

**Files:**
- Modify after hardware pass: `profiles/host/tested-kernels.json`
- Modify after hardware pass: `profiles/qualification/stable.toml` only if the existing profile requires a recorded kernel update

- [ ] **Step 1: Prove no host write surface remains**

```bash
rg -n 'TTM\.|amd-ttm|options (ttm|amdttm)|update-initramfs' src/amd_ai
rg -n -- '--memory-gib' src tests README.md docs
```

Expected: the first command finds only read-only log classification or no matches; the second finds no active interface/documentation match.

- [ ] **Step 2: Run the complete non-hardware suite**

```bash
uv run pytest -m 'not hardware' -q
```

Expected: all tests pass.

- [ ] **Step 3: Run shell and repository integrity checks**

```bash
bash -n install.sh bin/* images/common/*
git diff --check
git status --short
```

Expected: shell parsing and whitespace checks pass; only intended files are changed.

- [ ] **Step 4: Run hardware qualification on the current 6.17 OEM boot**

```bash
sudo -v
uv run strix-halo-rocm host-preflight --json reports/host-preflight-6.17.json
uv run strix-halo-rocm host-verify --json reports/host-verify-6.17.json
sudo -n ./bin/container-check \
  --suite stable \
  --profile profiles/qualification/stable.toml \
  --json reports/qualification-6.17.json
```

Expected: Radeon `1002:1586`, inbox `amdgpu`, `/dev/kfd`, render node, gfx1151 ROCm agent, stable PyTorch GPU execution, stress suite, and current-boot log gate all pass without any GTT/TTM write.

- [ ] **Step 5: Promote only the exact qualified kernel**

If and only if Step 4 passes on `6.17.0-1028-oem`, update:

```json
{
  "schema_version": 1,
  "kernels": ["6.17.0-1028-oem"],
  "source": "local hardware qualification report"
}
```

Then rerun host policy, verification, and full non-hardware tests.

- [ ] **Step 6: Request code review and verify the branch**

Use `superpowers:requesting-code-review`, address findings, then use `superpowers:verification-before-completion`. Record exact commands and results before claiming completion.

- [ ] **Step 7: Commit qualification evidence metadata**

```bash
git add profiles/host/tested-kernels.json profiles/qualification/stable.toml
git commit -m "test: qualify the OEM 6.17 host baseline"
```

- [ ] **Step 8: Finish the development branch**

Use `superpowers:finishing-a-development-branch` to choose integration, run the final merge verification, and only then push the branch/tag requested by the maintainer.
