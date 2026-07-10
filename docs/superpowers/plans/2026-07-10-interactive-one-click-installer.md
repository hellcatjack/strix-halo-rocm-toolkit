# Interactive One-Click Installer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide one local `install.sh` entry and one installed `strix-halo-rocm` command that guide users through a resumable full-workstation or container-only deployment without bypassing host, image, release, project, or GPU safety gates.

**Architecture:** A standard-library state machine persists canonical stage-input digests and resumes only when facts still match. An injected action service calls the existing host/image/project modules and the new release/doctor APIs; prompts are isolated from policy so TTY, EOF, and non-interactive refusal are testable. The bootstrap installs a versioned user-local runtime and launcher, while local image build fallback remains tied to a verified source checkout.

**Tech Stack:** Bash bootstrap, Python 3.12 standard library, argparse, JSON, `fcntl`, `hashlib`, pytest, existing Docker/host/project APIs.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `install.sh` | Auditable local bootstrap and argv forwarding |
| `bin/strix-halo-rocm` | Repository wrapper for the unified CLI |
| `src/amd_ai/installer/models.py` | Install mode, stage, options, persisted state and stage result records |
| `src/amd_ai/installer/state.py` | Canonical digests, atomic state, install lock, corruption preservation, boot ID |
| `src/amd_ai/installer/prompts.py` | Numbered choices, exact confirmation, TTY/EOF/non-interactive behavior |
| `src/amd_ai/installer/bootstrap.py` | Versioned `~/.local/share` runtime and `~/.local/bin` launcher installation |
| `src/amd_ai/installer/actions.py` | Production adapter over existing host/image/project/release APIs |
| `src/amd_ai/installer/workflow.py` | Full/container stage machine, resume and exit status |
| `src/amd_ai/cli.py` | Thin unified `install`, `project`, `doctor`, `repair`, and `release` routing |
| `tests/unit/installer/` | State, prompts, bootstrap, actions, and workflow tests |
| `tests/cli/test_installer_commands.py` | Bootstrap/unified CLI parsing and dispatch |
| `tests/fixtures/installer/` | Healthy, host-change, reboot, blocked and corrupt-state fixtures |
| `README.md` | Fast path and platform boundary |
| `docs/install.md` | Full/container/non-interactive operator guide |
| `docs/protected-pip.md` | Supported commands, rejected flags and promotion workflow |
| `docs/doctor-repair.md` | Diagnostic codes, exact repair and evidence retention |
| `docs/release-chain.md` | GHCR digest, config ID, SBOM and qualification relationship |

### Task 1: Define Install Modes, Stages, Options, and State

**Files:**
- Modify: `src/amd_ai/installer/models.py`
- Create: `tests/unit/installer/test_models.py`

- [ ] **Step 1: Write failing stage-order tests**

```python
import pytest

from amd_ai.installer.models import (
    FULL_STAGE_ORDER,
    CONTAINER_STAGE_ORDER,
    InstallMode,
    InstallOptions,
    InstallStage,
    InstallerModelError,
)


def test_full_and_container_stage_orders_match_approved_workflow():
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


def test_noninteractive_options_require_project_and_explicit_image_source(tmp_path):
    with pytest.raises(InstallerModelError):
        InstallOptions(
            mode=InstallMode.CONTAINER,
            non_interactive=True,
            project_dir=None,
            image_source=None,
        ).validate()
```

- [ ] **Step 2: Run model tests and observe missing definitions**

Run: `uv run pytest tests/unit/installer/test_models.py -q`

Expected: import failures for installer model definitions.

- [ ] **Step 3: Add exact enums and records**

Define:

```python
class InstallMode(StrEnum):
    FULL = "full"
    CONTAINER = "container"
    DOCTOR = "doctor"


class InstallStage(StrEnum):
    BOOTSTRAP = "BOOTSTRAP"
    HOST_PREFLIGHT = "HOST_PREFLIGHT"
    HOST_PLAN = "HOST_PLAN"
    HOST_CONFIRM = "HOST_CONFIRM"
    HOST_APPLY = "HOST_APPLY"
    REBOOT_PENDING = "REBOOT_PENDING"
    HOST_VERIFY = "HOST_VERIFY"
    CONTAINER_HOST_CHECK = "CONTAINER_HOST_CHECK"
    RELEASE_RESOLVE = "RELEASE_RESOLVE"
    IMAGE_PULL_OR_BUILD = "IMAGE_PULL_OR_BUILD"
    IMAGE_VERIFY = "IMAGE_VERIFY"
    PROJECT_INIT = "PROJECT_INIT"
    PROJECT_VERIFY = "PROJECT_VERIFY"
    COMPLETE = "COMPLETE"
```

`InstallOptions` fields are mode, non-interactive flag, dry-run flag, project directory, project name, image source (`pull` or `build`), target user, accepted host-plan digest, Docker-group acceptance, stable manifest path, source root, and state path. Validate normalized absolute paths, Python 3.12, explicit non-interactive project/image values, and full-mode host-plan authorization. Experimental profiles are not an install option. Dry-run may install/update the user-local launcher and collect read-only facts, but it cannot apply host actions, pull/build/tag images, create projects, or advance persistent completion stages.

`InstallState` contains every field approved in the specification plus `installer_source_revision`, `source_root`, `host_plan_digest`, and `last_report_paths`. The approved `source_revision` field is the source revision bound by the stable release manifest; `installer_source_revision` identifies the bootstrap checkout/runtime and may be one later manifest-only commit. Completed stage digests are a mapping from stage value to lowercase SHA-256. Reject secrets and unknown keys while loading.

- [ ] **Step 4: Run model tests**

Run: `uv run pytest tests/unit/installer/test_models.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit installer models**

```bash
git add src/amd_ai/installer/models.py tests/unit/installer/test_models.py
git commit -m "feat: define installer modes and stages"
```

### Task 2: Persist Atomic State and Stage Input Digests

**Files:**
- Create: `src/amd_ai/installer/state.py`
- Create: `tests/unit/installer/test_state.py`

- [ ] **Step 1: Write failing canonical digest and atomic-state tests**

```python
def test_stage_digest_is_canonical_across_mapping_order():
    left = stage_input_digest({"mode": "container", "facts": {"b": 2, "a": 1}})
    right = stage_input_digest({"facts": {"a": 1, "b": 2}, "mode": "container"})

    assert left == right
    assert len(left) == 64


def test_state_round_trip_uses_atomic_replace(tmp_path, monkeypatch):
    path = tmp_path / "install-state.json"
    calls = []
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda source, target: calls.append((Path(source), Path(target))) or real_replace(source, target))

    save_state(path, install_state())

    assert load_state(path) == install_state()
    assert calls[-1][1] == path


def test_corrupt_state_is_preserved_before_replanning(tmp_path):
    path = tmp_path / "install-state.json"
    path.write_text("not-json", encoding="utf-8")

    with pytest.raises(CorruptInstallState) as error:
        load_state(path)

    assert error.value.preserved_path.read_text(encoding="utf-8") == "not-json"
```

- [ ] **Step 2: Run state tests and confirm missing module failure**

Run: `uv run pytest tests/unit/installer/test_state.py -q`

Expected: collection fails because `amd_ai.installer.state` is missing.

- [ ] **Step 3: Implement canonical JSON digest and atomic persistence**

Use:

```python
def stage_input_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
```

Write state to a mode-`0600` temporary file in the same directory, flush, fsync, replace, and fsync the directory. Hold `install-state.lock` with nonblocking exclusive `fcntl.flock` for the complete workflow. Preserve malformed state as `install-state.corrupt.<UTC>.json` with `os.replace` and return no guessed completion state.

- [ ] **Step 4: Implement boot-ID and resume validation**

Read `/proc/sys/kernel/random/boot_id`, require UUID syntax, and store it when entering `REBOOT_PENDING`. Resumption may leave that stage only when the current boot ID differs. `validate_completed_stage(state, stage, current_inputs)` recomputes the input digest and raises `ResumeInputChanged` on mismatch; it never silently deletes a completed stage.

- [ ] **Step 5: Run state tests**

Run: `uv run pytest tests/unit/installer/test_state.py -q`

Expected: atomic write, lock contention, corruption preservation, unchanged resume, changed release digest, and boot-ID tests pass.

- [ ] **Step 6: Commit installer state support**

```bash
git add src/amd_ai/installer/state.py tests/unit/installer/test_state.py
git commit -m "feat: persist resumable installer state"
```

### Task 3: Isolate Interactive Prompts and Non-Interactive Refusal

**Files:**
- Create: `src/amd_ai/installer/prompts.py`
- Create: `tests/unit/installer/test_prompts.py`

- [ ] **Step 1: Write failing prompt behavior tests**

```python
def test_numbered_choice_reprompts_then_returns(monkeypatch, capsys):
    answers = iter(["9", "2"])
    prompt = TerminalPrompts(input_fn=lambda text: next(answers), is_tty=True)

    assert prompt.choose_mode() == InstallMode.CONTAINER
    assert "1. 完整工作站安装" in capsys.readouterr().out


def test_exact_confirmation_rejects_case_and_eof():
    assert TerminalPrompts(input_fn=lambda text: "apply", is_tty=True).confirm_exact("APPLY") is False
    assert TerminalPrompts(input_fn=lambda text: (_ for _ in ()).throw(EOFError), is_tty=True).confirm_exact("APPLY") is False


def test_noninteractive_prompt_is_always_blocked():
    prompt = NonInteractivePrompts()

    with pytest.raises(PromptRequired):
        prompt.confirm_exact("APPLY")
```

- [ ] **Step 2: Run prompt tests and observe failure**

Run: `uv run pytest tests/unit/installer/test_prompts.py -q`

Expected: collection fails because prompts module is missing.

- [ ] **Step 3: Implement numbered terminal UI and exact confirmations**

The home menu is exactly:

```text
Strix Halo ROCm Toolkit

1. 完整工作站安装
2. 仅安装容器平台
3. 检查或修复已有安装
4. 退出
```

Expose `choose_mode()`, `choose_image_fallback()`, `ask_project_dir()`, `confirm_exact(word)`, and `confirm_yes_no()`. Terminal prompts require both stdin and stdout TTY. EOF, `KeyboardInterrupt`, empty input, and non-TTY return a typed refusal instead of accepting defaults. Status rendering uses only `PASS`, `WARN`, `ACTION`, and `BLOCKED` prefixes.

- [ ] **Step 4: Run prompt tests**

Run: `uv run pytest tests/unit/installer/test_prompts.py -q`

Expected: all prompt tests pass.

- [ ] **Step 5: Commit prompt isolation**

```bash
git add src/amd_ai/installer/prompts.py tests/unit/installer/test_prompts.py
git commit -m "feat: add deterministic installer prompts"
```

### Task 4: Install a Versioned User-Local Runtime and Launcher

**Files:**
- Create: `src/amd_ai/installer/bootstrap.py`
- Create: `tests/unit/installer/test_bootstrap.py`

- [ ] **Step 1: Write failing bootstrap installation tests**

```python
def test_install_runtime_copies_required_payload_and_switches_current(tmp_path):
    source = toolkit_fixture(tmp_path / "source")
    home = tmp_path / "home"

    result = install_user_runtime(
        source_root=source,
        home=home,
        version="0.2.0",
        installer_source_revision="a" * 40,
    )

    assert (result.runtime / "src/amd_ai/cli.py").is_file()
    assert (result.runtime / "profiles/torch/stable.env").is_file()
    assert os.readlink(home / ".local/share/strix-halo-rocm-toolkit/current") == "releases/0.2.0-" + "a" * 12
    assert os.access(home / ".local/bin/strix-halo-rocm", os.X_OK)
```

Add tests for a symlinked destination, partial copy, failed current switch, and launcher content.

- [ ] **Step 2: Run bootstrap tests and confirm failure**

Run: `uv run pytest tests/unit/installer/test_bootstrap.py -q`

Expected: collection fails because bootstrap module is missing.

- [ ] **Step 3: Implement the versioned runtime payload**

Copy only these repository paths into a private staging directory:

```python
RUNTIME_PATHS = (
    "src/amd_ai",
    "profiles",
    "templates",
    "images",
    "bin",
    "pyproject.toml",
)
```

Reject symlinks in source payload and destination control directories. Name the release directory `<version>-<installer_source_revision[:12]>`, fsync copied files, atomically switch a relative `current` symlink, then install a mode-`0755` launcher at `~/.local/bin/strix-halo-rocm`.

Launcher content is exactly:

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="${HOME}/.local/share/strix-halo-rocm-toolkit/current"
export AMD_AI_TOOLKIT_ROOT="${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python3.12 -m amd_ai.cli "$@"
```

Store source checkout path and installer source revision separately in install state for local builds. The copied runtime is sufficient for pull, doctor, repair, project initialization, and documentation lookup; local build is allowed only when the recorded source root is a clean checkout at `installer_source_revision`. After release resolution, persist the manifest's qualified image source revision in `source_revision`; never overwrite one identity with the other.

Add `main(argv=None) -> int` and a `if __name__ == "__main__": raise SystemExit(main())` guard. It parses `--source-root`, installs the user runtime, then forwards the remaining install arguments to the workflow through the same Python process.

- [ ] **Step 4: Run bootstrap tests**

Run: `uv run pytest tests/unit/installer/test_bootstrap.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit runtime bootstrap**

```bash
git add src/amd_ai/installer/bootstrap.py tests/unit/installer/test_bootstrap.py
git commit -m "feat: install versioned user-local toolkit runtime"
```

### Task 5: Define the Workflow Action Boundary

**Files:**
- Create: `src/amd_ai/installer/actions.py`
- Create: `tests/unit/installer/test_actions.py`
- Modify: `src/amd_ai/host/prepare.py`
- Modify: `tests/unit/host/test_prepare.py`

- [ ] **Step 1: Write failing action-adapter tests**

```python
def test_host_plan_uses_existing_probe_and_prepare_policy(monkeypatch):
    calls = []
    monkeypatch.setattr(actions, "HostProbe", fake_probe(calls))
    monkeypatch.setattr(actions, "create_prepare_plan", fake_prepare(calls))

    result = ProductionInstallerActions().host_plan(target_user="developer")

    assert calls == ["HostProbe.collect", "create_prepare_plan"]
    assert result.plan_digest == stage_input_digest(result.plan.to_dict())


def test_release_pull_calls_exact_release_api(monkeypatch, release):
    captured = {}
    monkeypatch.setattr(actions, "pull_and_verify_release", lambda value, docker: captured.update(release=value) or verified_images())

    ProductionInstallerActions().pull_release(release)

    assert captured["release"] == release
```

- [ ] **Step 2: Run actions tests and confirm missing adapter failure**

Run: `uv run pytest tests/unit/installer/test_actions.py -q`

Expected: collection fails because actions module is missing.

- [ ] **Step 3: Expose Docker-group planning as a reusable host function**

Move the pure addition of the optional Docker group action out of private CLI prompt code into `with_docker_group_action(plan: PreparePlan) -> PreparePlan` in `host/prepare.py`. It returns a new plan and never prompts or applies. Existing `host-prepare` behavior calls it only after its own user confirmation.

- [ ] **Step 4: Implement production actions by composition**

`ProductionInstallerActions` exposes:

```text
bootstrap
host_preflight
host_plan
host_apply
host_verify
container_host_check
resolve_release
pull_release
build_local_images
verify_torch_image
initialize_project
verify_project
doctor
```

Each method calls existing public policy/build/check functions and returns typed facts/reports; it does not invoke `amd_ai.cli`, parse printed text, or duplicate host/image/project rules. `container_host_check` is read-only and checks Docker daemon, host support policy, `/dev/kfd`, all render nodes and actual GIDs. GPU tensor verification remains `verify_torch_image` after image acquisition.

Local builds verify the recorded source root with `git rev-parse HEAD`, `git status --porcelain`, required build files, and state `installer_source_revision` before calling existing serial image builders.

- [ ] **Step 5: Run action and host tests**

Run:

```bash
uv run pytest tests/unit/installer/test_actions.py tests/unit/host/test_prepare.py \
  tests/cli/test_host_commands.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit action adapters**

```bash
git add src/amd_ai/installer/actions.py src/amd_ai/host/prepare.py \
  tests/unit/installer/test_actions.py tests/unit/host/test_prepare.py
git commit -m "refactor: expose installer action boundary"
```

### Task 6: Implement the Generic Resumable Stage Engine

**Files:**
- Create: `src/amd_ai/installer/workflow.py`
- Create: `tests/unit/installer/fakes.py`
- Create: `tests/unit/installer/test_workflow.py`

- [ ] **Step 1: Write failing transition and resume tests**

```python
def test_container_workflow_runs_each_stage_once_and_completes(tmp_path):
    actions = FakeInstallerActions.healthy()
    workflow = installer_workflow(tmp_path, actions=actions, mode=InstallMode.CONTAINER)

    result = workflow.run()

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
    assert result.state.current_stage == InstallStage.COMPLETE


def test_resume_skips_only_stages_with_matching_input_digest(tmp_path):
    actions = FakeInstallerActions.stop_after(InstallStage.IMAGE_VERIFY)
    first = installer_workflow(tmp_path, actions=actions, mode=InstallMode.CONTAINER)
    first.run()
    resumed_actions = FakeInstallerActions.healthy()

    installer_workflow(tmp_path, actions=resumed_actions, mode=InstallMode.CONTAINER).run()

    assert resumed_actions.calls == ["initialize_project", "verify_project"]
```

Add illegal transition, changed release digest, changed project path, Ctrl+C, action failure, and concurrent installer tests.

- [ ] **Step 2: Run workflow tests and observe failure**

Run: `uv run pytest tests/unit/installer/test_workflow.py -q`

Expected: collection fails because workflow module is missing.

- [ ] **Step 3: Implement stage dispatch and checkpoints**

Use the mode's locked stage tuple. Before a stage, compute its canonical inputs from options plus immutable outputs of prior stages. If a completed digest exists, validate and skip; otherwise execute once. Persist returned facts and report paths, mark the stage digest complete, set the next stage, and atomically save after each successful stage.

On `KeyboardInterrupt`, action exit 1/2, or exception, do not add a completed digest. Preserve the latest report path and return 1 for action/reboot required or 2 for blocked/refused. Never catch `SystemExit` from a child CLI because actions do not call the CLI.

- [ ] **Step 4: Add disk-space and state-lock preconditions**

Before image acquisition, require free bytes at the Docker data root to exceed the action service's reported missing-layer estimate plus 5 GiB. Before a generation build, require free project filesystem bytes to exceed resolved wheel bytes times two plus 1 GiB. A failed estimate blocks before pull/build and records exact required/available bytes.

- [ ] **Step 5: Run workflow tests**

Run: `uv run pytest tests/unit/installer/test_workflow.py -q`

Expected: all generic workflow tests pass.

- [ ] **Step 6: Commit the stage engine**

```bash
git add src/amd_ai/installer/workflow.py tests/unit/installer
git commit -m "feat: add resumable installer stage engine"
```

### Task 7: Implement Full Workstation and Reboot Semantics

**Files:**
- Modify: `src/amd_ai/installer/workflow.py`
- Modify: `tests/unit/installer/test_workflow.py`
- Create: `tests/fixtures/installer/full-host-change.json`
- Create: `tests/fixtures/installer/full-rebooted.json`

- [ ] **Step 1: Write failing full-mode authorization tests**

```python
def test_full_mode_requires_exact_apply_and_separate_docker_group_prompt(tmp_path):
    prompts = FakePrompts(exact={"APPLY": True}, yes_no={"docker-group": False})
    actions = FakeInstallerActions.host_change_requires_reboot()

    result = installer_workflow(tmp_path, actions=actions, prompts=prompts, mode=InstallMode.FULL).run()

    assert result.exit_code == 1
    assert result.state.current_stage == InstallStage.REBOOT_PENDING
    assert actions.host_apply_plan.includes_docker_group is False


def test_noninteractive_full_mode_requires_matching_plan_digest(tmp_path):
    actions = FakeInstallerActions.host_change_requires_reboot()
    options = noninteractive_full_options(accepted_host_plan_digest="0" * 64)

    result = installer_workflow(tmp_path, actions=actions, options=options).run()

    assert result.exit_code == 2
    assert "host plan digest" in result.message
    assert "host_apply" not in actions.calls
```

- [ ] **Step 2: Run full workflow tests and observe failures**

Run: `uv run pytest tests/unit/installer/test_workflow.py -q`

Expected: full-mode confirmation and reboot tests fail.

- [ ] **Step 3: Implement host plan, confirmation, apply, and reboot boundary**

Full mode permits apply only when the selected formal HostAdapter is `ubuntu-24.04` on AMD64. Show every `PreparePlan` action. Interactive mode requires exact `APPLY`, then separately asks whether to add the target user to Docker group. Non-interactive mode requires `--accept-host-plan-digest` equal to the displayed canonical digest and `--accept-docker-group` when that action is requested. At `RELEASE_RESOLVE`, require the manifest's supported adapter IDs to contain the already applied adapter; a mismatch blocks image deployment and is never silently substituted.

After apply, if the plan requires reboot, store the current boot ID, enter `REBOOT_PENDING`, print the manual reboot instruction, and exit 1. Do not invoke `reboot`. On the next run, unchanged boot ID remains action-required; changed boot ID advances to `HOST_VERIFY`. Host verify must pass before release resolution.

- [ ] **Step 4: Run full workflow tests**

Run: `uv run pytest tests/unit/installer/test_workflow.py -q`

Expected: exact confirmation, refusal, Docker group, boot ID, changed plan, supported adapter, and resumed host verify tests pass.

- [ ] **Step 5: Commit full workstation workflow**

```bash
git add src/amd_ai/installer/workflow.py tests/unit/installer/test_workflow.py \
  tests/fixtures/installer
git commit -m "feat: orchestrate full workstation installation"
```

### Task 8: Implement Container Mode, Pull Preference, and Build Fallback

**Files:**
- Modify: `src/amd_ai/installer/workflow.py`
- Modify: `tests/unit/installer/test_workflow.py`
- Create: `tests/fixtures/installer/container-healthy.json`
- Create: `tests/fixtures/installer/container-blocked.json`

- [ ] **Step 1: Write failing container-mode image selection tests**

```python
def test_container_mode_prefers_anonymous_exact_pull(tmp_path):
    actions = FakeInstallerActions.healthy()

    installer_workflow(tmp_path, actions=actions, mode=InstallMode.CONTAINER).run()

    assert actions.image_calls == [
        ("pull", actions.release.base.reference),
        ("pull", actions.release.torch.reference),
    ]


def test_interactive_pull_failure_requires_explicit_build_choice(tmp_path):
    actions = FakeInstallerActions.pull_fails()
    prompts = FakePrompts(image_fallback="build")

    result = installer_workflow(tmp_path, actions=actions, prompts=prompts, mode=InstallMode.CONTAINER).run()

    assert result.exit_code == 0
    assert "build_local_images" in actions.calls


def test_noninteractive_pull_failure_never_implicitly_builds(tmp_path):
    actions = FakeInstallerActions.pull_fails()
    options = noninteractive_container_options(image_source="pull")

    result = installer_workflow(tmp_path, actions=actions, options=options).run()

    assert result.exit_code == 2
    assert "build_local_images" not in actions.calls
```

- [ ] **Step 2: Run container workflow tests**

Run: `uv run pytest tests/unit/installer/test_workflow.py -q`

Expected: image selection tests fail.

- [ ] **Step 3: Implement read-only host check and exact pull preference**

Container mode never calls host plan/apply. It blocks when Docker daemon, `/dev/kfd`, render node, actual GID access, host policy, or AMD GPU identity is missing. Resolve the strict stable manifest, then pull both exact references and verify identity. A valid manifest with identity mismatch is blocked and cannot fall back to a local build under the same release ID.

Network/missing-release failure may offer local build interactively. Non-interactive mode follows only `--image-source pull` or `--image-source build`. Build mode requires a clean recorded source checkout and produces an unqualified local candidate; the installer must run metadata and GPU checks and label resulting state as local, not rewrite the stable manifest.

- [ ] **Step 4: Initialize and verify the selected project**

Require a normalized project directory. If absent, call existing `initialize_project`; if present, require its strict config and matching selected parent digest. Build/reuse the project image, initialize its empty overlay, run a managed read-only startup check, and require the gfx1151 synchronized operation. Do not generate ComfyUI, Hugging Face, model, input, or output mounts.

- [ ] **Step 5: Run container workflow tests**

Run: `uv run pytest tests/unit/installer/test_workflow.py -q`

Expected: pull preference, valid-manifest mismatch, interactive fallback, non-interactive refusal, blocked host, existing project, and project verification tests pass.

- [ ] **Step 6: Commit container workflow**

```bash
git add src/amd_ai/installer/workflow.py tests/unit/installer/test_workflow.py \
  tests/fixtures/installer
git commit -m "feat: orchestrate container platform installation"
```

### Task 9: Add Local Bootstrap Script and Unified CLI Routing

**Files:**
- Create: `install.sh`
- Create: `bin/strix-halo-rocm`
- Modify: `src/amd_ai/cli.py`
- Create: `tests/cli/test_installer_commands.py`
- Modify: `tests/test_version.py`
- Modify: `src/amd_ai/__init__.py`

- [ ] **Step 1: Write failing bootstrap and parser tests**

```python
def test_install_script_is_local_auditable_and_does_not_pipe_remote_shell():
    text = Path("install.sh").read_text(encoding="utf-8")

    assert "python3.12 -m amd_ai.installer.bootstrap" in text
    assert "curl |" not in text
    assert "wget |" not in text
    assert "eval " not in text


def test_install_noninteractive_arguments_parse():
    args = cli.build_parser().parse_args(
        [
            "install",
            "--mode",
            "container",
            "--non-interactive",
            "--project-dir",
            "/srv/comfy-lab",
            "--image-source",
            "pull",
        ]
    )

    assert args.command == "install"
    assert args.mode == "container"


def test_unified_project_command_maps_to_existing_handler(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "_project_run", lambda args: captured.update(project=args.project) or 0)

    assert cli.main(["project", "run", "/srv/demo", "--dry-run"]) == 0
    assert captured["project"] == Path("/srv/demo")
```

- [ ] **Step 2: Run CLI tests and verify missing entries**

Run: `uv run pytest tests/cli/test_installer_commands.py tests/test_version.py -q`

Expected: missing script and unknown command failures.

- [ ] **Step 3: Create the auditable Bash bootstrap**

`install.sh` must:

```bash
#!/usr/bin/env bash
set -euo pipefail
[[ -f "${BASH_SOURCE[0]}" ]] || { echo "install.sh must run from a local file" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${ROOT}/pyproject.toml" && -d "${ROOT}/src/amd_ai" ]] || {
  echo "incomplete toolkit checkout: ${ROOT}" >&2
  exit 2
}
command -v python3.12 >/dev/null || { echo "python3.12 is required" >&2; exit 2; }
NON_INTERACTIVE=0
for argument in "$@"; do
  [[ "${argument}" == "--non-interactive" ]] && NON_INTERACTIVE=1
done
if [[ "${NON_INTERACTIVE}" -eq 0 ]]; then
  [[ -t 0 && -t 1 ]] || { echo "interactive install requires a terminal" >&2; exit 2; }
fi
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python3.12 -m amd_ai.installer.bootstrap --source-root "${ROOT}" "$@"
```

Create executable `bin/strix-halo-rocm` using the repository `_dispatch` wrapper pattern.

- [ ] **Step 4: Add unified routing without breaking legacy commands**

Add `install` arguments from Tasks 1 and 7-8, including `--dry-run`. Add nested `project init|lock|run` parsers that reuse existing handler argument definitions and functions. Keep `project-init`, `project-lock`, and `project-run` aliases passing their existing tests. Existing top-level `doctor`, `repair`, and `release` remain the implementations used by the unified executable.

Interactive `strix-halo-rocm install` without `--mode` shows the home menu. Choice 3 dispatches doctor and then offers repair only when the doctor report is repairable; it still requires `REPAIR` confirmation.

- [ ] **Step 5: Bump the implementation version**

Set:

```python
__version__ = "0.2.0"
```

Update version tests and no other dependency metadata.

- [ ] **Step 6: Run CLI and legacy command tests**

Run:

```bash
chmod +x install.sh bin/strix-halo-rocm
uv run pytest tests/cli tests/test_version.py -q
```

Expected: all new and legacy CLI tests pass.

- [ ] **Step 7: Commit bootstrap and unified command**

```bash
git add install.sh bin/strix-halo-rocm src/amd_ai/cli.py \
  src/amd_ai/__init__.py tests/cli/test_installer_commands.py tests/test_version.py
git commit -m "feat: add interactive installer entrypoint"
```

### Task 10: Add End-to-End Installer Fixtures and Resume Tests

**Files:**
- Create: `tests/fixtures/installer/README.md`
- Create: `tests/cli/test_installer_resume.py`

- [ ] **Step 1: Test complete container mode with injected fixture facts**

Run `install.sh` through an environment-selected fixture backend, never the real host apply path. Assert status order, exact digest pull references, project creation, final state, installed launcher, and exit 0.

- [ ] **Step 2: Test full mode across a simulated reboot**

First invocation must stop at `REBOOT_PENDING` with exit 1 and boot ID A. Second invocation with boot ID A must remain exit 1. Third invocation with boot ID B must start at host verify, complete image/project stages, and exit 0 without repeating host apply.

- [ ] **Step 3: Test corruption and changed inputs**

Cover malformed state preservation, stable manifest digest change, project path change, source revision change, incomplete prior action, EOF at confirmation, and Ctrl+C. Assert no changed-input run advances or mutates the host/project.

- [ ] **Step 4: Run end-to-end fixture tests**

Run:

```bash
uv run pytest tests/cli/test_installer_resume.py -q
```

Expected: all fixture workflows pass without requiring sudo, Docker, or hardware.

- [ ] **Step 5: Commit resume integration tests**

```bash
git add tests/fixtures/installer tests/cli/test_installer_resume.py
git commit -m "test: cover installer resume and reboot flow"
```

### Task 11: Write Operator Documentation

**Files:**
- Modify: `README.md`
- Create: `docs/install.md`
- Create: `docs/protected-pip.md`
- Create: `docs/doctor-repair.md`
- Create: `docs/release-chain.md`
- Modify: `tests/container/test_image_contract.py`

- [ ] **Step 1: Add documentation contract tests**

Assert each file exists and includes exact command anchors:

```python
required = {
    "README.md": ["./install.sh", "ROCm 7.2.1", "PyTorch 2.9.1"],
    "docs/install.md": ["--mode container", "--non-interactive", "REBOOT_PENDING"],
    "docs/protected-pip.md": ["pip install", "--target", "overlay.requirements.lock"],
    "docs/doctor-repair.md": ["TORCH.SHADOWED", "quarantine", "docker system prune"],
    "docs/release-chain.md": ["manifest digest", "config digest", "anonymous"],
}
```

- [ ] **Step 2: Run documentation contract tests**

Run: `uv run pytest tests/container/test_image_contract.py -q`

Expected: failures for missing documentation anchors.

- [ ] **Step 3: Document install modes and safety boundaries**

README starts with clone-a-fixed-release then local `./install.sh`; it does not recommend `curl | bash`. State Ubuntu 24.04 AMD64 and gfx1151 formal support, container-only behavior, sudo boundaries, no automatic reboot, no ComfyUI/model/cache defaults, and exact GHCR digest pulls. Preserve the existing custom complete Torch-profile route as an explicit experimental workflow using `bin/image-build rocm-pytorch --profile ... --allow-experimental`; the installer never silently selects or promotes that image.

`docs/install.md` covers interactive full mode, interactive container mode, every non-interactive flag, reboot resume, state path, exit codes, local-build fallback, explicit writable mounts, disk-space failures, and source checkout retention.

- [ ] **Step 4: Document pip, doctor/repair, and release identity**

`docs/protected-pip.md` contains a table for install, `-r`, local source, local wheel, exact-commit Git, uninstall, list/show/check/freeze; a separate rejected table for user/target/prefix/root/editable, mutable or unnamed VCS, and protected uninstall; exact parent-satisfaction examples; project-private persistence; and the workflow to move tested roots into project `requirements.in`, run `project lock`, rebuild, then uninstall the overlay roots.

`docs/doctor-repair.md` lists every stable code and disposition, exact actions, `REPAIR`/`--yes`, quarantine evidence, offline replay, immutable parent/project rebuild, and the explicit statement that repair never runs `docker system prune` or force-reinstalls Torch.

`docs/release-chain.md` distinguishes source revision, qualification profile/report digest, SPDX digest, registry manifest digest, config ID, parent digest, project image ID, and anonymous pull verification. State that large verified images are published from the trusted qualified host after hardware tests; public-repository self-hosted runners are not enabled for untrusted pull-request code, and standard GitHub-hosted runner disk is not assumed sufficient.

- [ ] **Step 5: Run documentation and link checks**

Run:

```bash
uv run pytest tests/container/test_image_contract.py -q
rg -n 'curl[^\n]*\|[^\n]*(bash|sh)' README.md docs install.sh
```

Expected: tests pass and ripgrep prints no remote-pipe installation command.

- [ ] **Step 6: Commit operator documentation**

```bash
git add README.md docs tests/container/test_image_contract.py
git commit -m "docs: add installer and protected pip operations"
```

### Task 12: Install Locally and Run Final Non-Hardware Gates

**Files:**
- Modify only files implicated by a failing check

- [ ] **Step 1: Run the complete test suite without hardware**

Run:

```bash
uv run pytest -m 'not hardware' -q
git diff --check
```

Expected: zero failures and no whitespace errors.

- [ ] **Step 2: Exercise the real user-local bootstrap without host changes**

Run:

```bash
./install.sh --mode container --non-interactive \
  --project-dir /tmp/strix-halo-installer-smoke \
  --image-source build \
  --dry-run
"${HOME}/.local/bin/strix-halo-rocm" --version
```

Expected: dry-run prints the exact stages/actions without applying or building, installs the launcher, and version output is `amd-ai 0.2.0`.

- [ ] **Step 3: Verify legacy wrappers remain executable**

Run:

```bash
for command in host-preflight host-prepare host-verify image-build \
  container-check project-init project-lock project-run gpu-release \
  strix-halo-rocm; do
  test -x "bin/${command}"
done
```

Expected: exit 0.

- [ ] **Step 4: Scan for secrets and unsafe bootstrap behavior**

Run:

```bash
rg -n 'ghp_|password=|token=|curl[^\n]*\|[^\n]*(bash|sh)|eval ' \
  install.sh bin src profiles templates docs README.md
```

Expected: no credential literals, remote shell pipelines, or eval execution. Documentation may use uppercase placeholder environment names such as `HF_TOKEN` without values.

- [ ] **Step 5: Record the clean implementation checkpoint**

```bash
git status --short
git log --oneline -20
```

Expected: clean feature worktree. Continue with final image rebuild, hardware qualification, GHCR publication, anonymous pull, and stable manifest commit in the index plan.
