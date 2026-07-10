# Per-project Installer State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make an omitted `--state-path` select a deterministic per-project state while preserving compatible legacy global-state resumes.

**Architecture:** Add read-only state identity inspection and deterministic path selection to `installer/state.py`. Mark CLI-provided state paths as explicit in `InstallOptions`, then let `InstallerWorkflow` select and report the effective path after the project directory is known but before locking or loading state. Existing state decoding, checkpoint validation, and explicit-path behavior remain authoritative.

**Tech Stack:** Python 3.12, standard-library `dataclasses`, `hashlib`, `json`, `pathlib`, pytest 8.4, shell-level fixture integration tests.

---

## Task 1: State selection primitives

**Files:**

- Modify: `src/amd_ai/installer/state.py`
- Test: `tests/unit/installer/test_state.py`

- [x] **Step 1: Write failing deterministic path and selection tests**

Add imports for `project_state_path` and `select_install_state_path`, then add tests equivalent to:

```python
def test_project_state_path_is_stable_and_separates_equal_basenames(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "state" / "install-state.json"
    first = project_state_path(tmp_path / "one" / "video-lab", legacy)
    second = project_state_path(tmp_path / "two" / "video-lab", legacy)

    assert first == project_state_path(
        (tmp_path / "one" / "." / "video-lab").resolve(), legacy
    )
    assert first.parent == legacy.parent / "projects"
    assert first.name.startswith("video-lab-")
    assert first.suffix == ".json"
    assert first != second


def test_unrelated_legacy_state_selects_new_project_state(tmp_path: Path) -> None:
    legacy = tmp_path / "install-state.json"
    save_state(legacy, install_state(tmp_path, project_path=str(tmp_path / "old")))

    selection = select_install_state_path(
        project_dir=(tmp_path / "new").resolve(),
        requested_path=legacy,
        explicit=False,
    )

    assert selection.source == "project"
    assert selection.path == project_state_path(tmp_path / "new", legacy)
    assert legacy.is_file()


def test_matching_legacy_state_is_reused_even_when_mode_will_conflict(
    tmp_path: Path,
) -> None:
    project = (tmp_path / "project").resolve()
    legacy = tmp_path / "install-state.json"
    save_state(legacy, install_state(tmp_path, project_path=str(project)))

    selection = select_install_state_path(
        project_dir=project,
        requested_path=legacy,
        explicit=False,
    )

    assert selection.source == "legacy"
    assert selection.path == legacy.resolve()


def test_explicit_state_path_always_wins(tmp_path: Path) -> None:
    explicit = tmp_path / "operator.json"
    selection = select_install_state_path(
        project_dir=(tmp_path / "project").resolve(),
        requested_path=explicit,
        explicit=True,
    )
    assert selection.source == "explicit"
    assert selection.path == explicit.resolve()
```

Also cover an existing project state winning over legacy state, unsafe basename sanitization, and malformed legacy JSON selecting `source="legacy"` without changing the malformed file.

- [x] **Step 2: Run the new tests and verify RED**

Run:

```bash
PYTHONPATH=src /app/imgMaker/.venv/bin/python -m pytest \
  tests/unit/installer/test_state.py -q
```

Expected: collection fails because `project_state_path` and `select_install_state_path` do not exist.

- [x] **Step 3: Implement read-only identity inspection and path selection**

In `src/amd_ai/installer/state.py`, add a frozen selection result:

```python
@dataclass(frozen=True)
class StatePathSelection:
    path: Path
    source: str
```

Extract the non-mutating JSON-to-`InstallState` logic currently embedded in
`load_state` into a private `_decode_state(raw: str) -> InstallState`. Keep
`load_state` responsible for preserving corrupt state on failure.

Add deterministic path generation:

```python
def project_state_path(project_dir: Path, legacy_path: Path) -> Path:
    project = Path(project_dir).resolve(strict=False)
    legacy = Path(legacy_path).resolve(strict=False)
    readable = re.sub(r"[^A-Za-z0-9._-]+", "-", project.name)
    readable = readable.strip(".-_")[:48] or "project"
    identity = hashlib.sha256(str(project).encode("utf-8")).hexdigest()[:12]
    return legacy.parent / "projects" / f"{readable}-{identity}.json"
```

Add a private read-only legacy identity probe. A missing file is distinct from
an existing but invalid file. Implement selection in this order: explicit,
existing project state, matching valid legacy, unrelated valid legacy, invalid
legacy, no legacy. Return sources `explicit`, `project`, or `legacy` exactly.

- [x] **Step 4: Run state tests and verify GREEN**

Run:

```bash
PYTHONPATH=src /app/imgMaker/.venv/bin/python -m pytest \
  tests/unit/installer/test_state.py -q
```

Expected: all state tests pass and malformed-state preservation tests remain unchanged.

- [x] **Step 5: Commit the state-selection primitive**

```bash
git add src/amd_ai/installer/state.py tests/unit/installer/test_state.py
git commit -m "feat: select installer state by project"
```

## Task 2: Wire implicit selection into CLI and workflow

**Files:**

- Modify: `src/amd_ai/installer/models.py`
- Modify: `src/amd_ai/cli.py`
- Modify: `src/amd_ai/installer/prompts.py`
- Modify: `src/amd_ai/installer/workflow.py`
- Modify: `tests/cli/test_installer_commands.py`
- Modify: `tests/cli/test_installer_resume.py`
- Modify: `tests/unit/installer/test_models.py`
- Modify: `tests/unit/installer/test_prompts.py`
- Modify: `tests/unit/installer/test_workflow.py`

- [x] **Step 1: Write failing CLI and workflow tests**

In `tests/cli/test_installer_commands.py`, capture the workflow options and
assert omitted and explicit state-path provenance:

```python
assert captured["options"].state_path_explicit is False
```

for an invocation without `--state-path`, and:

```python
assert captured["options"].state_path_explicit is True
```

for the existing explicit invocation.

In `tests/unit/installer/test_workflow.py`, add a helper that constructs
`InstallOptions` with `state_path_explicit=False`. Add tests that:

1. save a completed unrelated legacy state and verify a new container project
   uses `project_state_path(...)` and runs from `BOOTSTRAP`;
2. save a matching legacy state and verify it is resumed;
3. save a matching legacy state with another mode and verify `mode changed`
   includes the selected legacy path;
4. omit `project_dir`, return it from a prompt fake, and verify selection occurs
   after the prompt;
5. verify an `INFO` status contains the source and selected absolute path.

Extend `tests/cli/test_installer_resume.py` so `run_install` accepts
`state: Path | None` and only emits `--state-path` when non-`None`. Add a fixture
test that completes project A into the legacy default path, then installs
project B without `--state-path` and verifies both state files and projects
remain present.

- [x] **Step 2: Run focused tests and verify RED**

Run:

```bash
PYTHONPATH=src /app/imgMaker/.venv/bin/python -m pytest \
  tests/cli/test_installer_commands.py \
  tests/cli/test_installer_resume.py \
  tests/unit/installer/test_models.py \
  tests/unit/installer/test_workflow.py -q
```

Expected: failures show missing `state_path_explicit` and the workflow still loading the unrelated legacy state.

- [x] **Step 3: Mark explicit CLI state paths**

Add this frozen option field in `InstallOptions` with a strict boolean check:

```python
state_path_explicit: bool = True
```

The default is `True` so internal callers and existing tests that directly
construct an option with a chosen path preserve current semantics. In
`_install_command`, pass:

```python
state_path_explicit=args.state_path is not None
```

- [x] **Step 4: Select and report state after project prompting**

Import `ResumeInputChanged` and `select_install_state_path` into the workflow.
After `_prepare_options` has obtained `project_dir`, call selection when
`state_path_explicit` is false, replace the option's effective `state_path`, and
emit:

```text
INFO installer state (<source>): <absolute path>
```

For explicit paths, emit the same status with source `explicit`. Keep selection
before `install_lock`. Catch `ResumeInputChanged` separately so its message
includes the selected state path. Include that path in the existing mode-change
transition error as well. Add `INFO` to the prompt renderer's approved status
prefixes and retain rejection of unknown prefixes.

- [x] **Step 5: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all focused tests pass, including the
real `install.sh` fixture regression for two projects.

- [x] **Step 6: Commit CLI and workflow integration**

```bash
git add src/amd_ai/installer/models.py src/amd_ai/cli.py \
  src/amd_ai/installer/prompts.py src/amd_ai/installer/workflow.py \
  tests/cli/test_installer_commands.py \
  tests/cli/test_installer_resume.py tests/unit/installer/test_models.py \
  tests/unit/installer/test_prompts.py tests/unit/installer/test_workflow.py
git commit -m "fix: isolate implicit installer state per project"
```

## Task 3: Version and user documentation

**Files:**

- Modify: `src/amd_ai/__init__.py`
- Modify: `tests/test_version.py`
- Modify: `README.md`
- Modify: `docs/install.md`

- [ ] **Step 1: Write the failing version assertion**

Change `tests/test_version.py` to require `0.2.2` from both the module and CLI.

- [ ] **Step 2: Run the version test and verify RED**

```bash
PYTHONPATH=src /app/imgMaker/.venv/bin/python -m pytest tests/test_version.py -q
```

Expected: failure reports current version `0.2.1`.

- [ ] **Step 3: Update version and documentation**

Set `__version__ = "0.2.2"`. Update README clone examples and toolkit-version
banner to `v0.2.2`, while retaining stable image release ID `0.2.0`.

Document this default command as sufficient for a second project:

```bash
./install.sh --mode container \
  --project-dir /app/test/video-lab \
  --project-name video-lab \
  --image-source pull
```

Document the deterministic state directory, legacy matching behavior, selected
state `INFO` line, and explicit `--state-path` override. Do not recommend
deleting the legacy state.

- [ ] **Step 4: Verify tests and Markdown**

```bash
PYTHONPATH=src /app/imgMaker/.venv/bin/python -m pytest tests/test_version.py -q
npx --yes markdownlint-cli@0.45.0 README.md docs/install.md --disable MD013
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 5: Commit release-facing changes**

```bash
git add src/amd_ai/__init__.py tests/test_version.py README.md docs/install.md
git commit -m "docs: publish per-project state workflow"
```

## Task 4: Verification, production regression, and release

**Files:**

- Verify only; no planned source edits

- [ ] **Step 1: Run static and complete automated verification**

```bash
uvx --from ruff==0.12.3 ruff check src tests
PYTHONPATH=src /app/imgMaker/.venv/bin/python -m pytest -m 'not hardware' -q
npx --yes markdownlint-cli@0.45.0 README.md docs/install.md \
  docs/superpowers/specs/2026-07-10-per-project-installer-state-design.md \
  docs/superpowers/plans/2026-07-10-per-project-installer-state.md \
  --disable MD013
git diff --check
```

Expected: Ruff, Markdown, and diff checks pass; the non-hardware suite has no failures.

- [ ] **Step 2: Verify stable image identities are unchanged**

```bash
PYTHONPATH=src bin/strix-halo-rocm release verify \
  --manifest profiles/releases/stable.json
```

Expected exact manifests:

```text
sha256:e9991f97f578156c8620fbb587d2d34504eb632f165cc5597deaadaa3e692a12
sha256:dc0bb217474cfd4f602423bd3bf4fe8714b03e900cf3c6b4417b99e622ebcf8b
```

- [ ] **Step 3: Exercise the user's legacy state without mutation**

Record the SHA-256 of
`~/.local/state/strix-halo-rocm-toolkit/install-state.json`, copy it and the
user command environment into a temporary HOME/project fixture, then run the
new installer with container mode, a different project, and no `--state-path`.
Verify the selected state is under `projects/`, the copied legacy digest is
unchanged, and no full-host action is called.

- [ ] **Step 4: Review and publish**

Review `git diff v0.2.1...HEAD`, fast-forward `main`, create annotated tag
`v0.2.2`, and atomically push `main` plus the tag. Verify anonymous HTTPS refs
and raw `__version__` content.

- [ ] **Step 5: Run the user's original command on `v0.2.2`**

Update `/app/test/strix-halo-rocm-toolkit` to detached tag `v0.2.2`, then run:

```bash
./install.sh --mode container \
  --project-dir /app/test/video-lab \
  --project-name video-lab
```

Verify it reports a per-project state path, completes without a mode-change or
`BOOTSTRAP` digest error, passes exact-image GPU verification and project
doctor, and leaves the old `/app/test/rocmToolkit` state digest unchanged.
