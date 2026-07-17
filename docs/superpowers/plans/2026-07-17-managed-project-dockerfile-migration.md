# Managed Project Dockerfile Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically migrate an untouched legacy project Dockerfile away from the external Docker Hub frontend while preserving every user-modified Dockerfile.

**Architecture:** A project-layer helper compares exact file bytes through an allowlisted SHA-256 digest and performs an atomic same-directory replacement with the current toolkit template. The production installer invokes it only for an existing, already-validated project before calculating the build fingerprint.

**Tech Stack:** Python 3.12 standard library, Dockerfile templates, pytest, existing installer action framework.

---

### Task 1: Implement Exact Legacy Template Migration

**Files:**
- Create: `tests/fixtures/project/legacy-Dockerfile-0.3.2`
- Modify: `tests/unit/project/test_init.py`
- Modify: `src/amd_ai/project/init.py`

- [ ] **Step 1: Add the legacy fixture and failing migration tests**

Create `tests/fixtures/project/legacy-Dockerfile-0.3.2` with the exact project
Dockerfile content from the parent of commit `8ab9edc`. Its SHA-256 must be:

```text
9a347d016b40564cc950dc3aa76a798ab9792ba33c8aa82a40697b6660227373
```

Add tests to `tests/unit/project/test_init.py` that import
`migrate_legacy_project_dockerfile` and verify:

```python
def test_untouched_legacy_dockerfile_is_atomically_migrated(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    target = project / "Dockerfile"
    target.write_bytes(LEGACY_DOCKERFILE.read_bytes())
    target.chmod(0o640)

    changed = migrate_legacy_project_dockerfile(project)

    assert changed is True
    assert target.read_bytes() == Path(
        "templates/project/Dockerfile"
    ).read_bytes()
    assert target.stat().st_mode & 0o777 == 0o640
    assert "# syntax=" not in target.read_text(encoding="utf-8")
    assert "--mount=" not in target.read_text(encoding="utf-8")
```

Add separate tests asserting:

- the current template returns `False` and remains byte-identical;
- appending one byte to the legacy fixture returns `False` and remains
  byte-identical;
- a symlink at `project/Dockerfile` raises `ProjectInitError` containing
  `symlink`.

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/project/test_init.py -q
```

Expected: collection fails because
`migrate_legacy_project_dockerfile` does not exist.

- [ ] **Step 3: Implement the migration helper**

In `src/amd_ai/project/init.py`:

- import `hashlib`, `stat`, and `tempfile`;
- define:

```python
LEGACY_PROJECT_DOCKERFILE_DIGESTS = frozenset(
    {"9a347d016b40564cc950dc3aa76a798ab9792ba33c8aa82a40697b6660227373"}
)
```

- implement:

```python
def migrate_legacy_project_dockerfile(
    destination: Path,
    *,
    template_root: Path = TEMPLATE_ROOT,
) -> bool:
```

The function must:

1. Resolve the destination and select `Dockerfile` under both roots.
2. Return `False` when the target does not exist.
3. Reject a target or template symlink.
4. Read both files as bytes and return `False` when already equal.
5. Return `False` for a target digest outside the legacy allowlist.
6. Create a temporary file in the project directory with `tempfile.mkstemp`.
7. Preserve the target mode and, when needed, owner/group.
8. Write, flush, and fsync the current template.
9. Replace the target with `os.replace`, then fsync the project directory.
10. Remove the temporary file on every failure and raise
    `ProjectInitError("cannot migrate project Dockerfile: ...")`.

- [ ] **Step 4: Run focused project tests**

Run:

```bash
uv run pytest tests/unit/project/test_init.py tests/container/test_project_dockerfile.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/amd_ai/project/init.py tests/unit/project/test_init.py \
  tests/fixtures/project/legacy-Dockerfile-0.3.2
git commit -m "fix: migrate untouched legacy project Dockerfiles"
```

### Task 2: Invoke Migration During Installer Resume

**Files:**
- Modify: `src/amd_ai/installer/actions.py`
- Modify: `tests/unit/installer/test_actions.py`

- [ ] **Step 1: Write the failing installer boundary test**

Add a test that creates an existing `amd-ai-project.toml`, monkeypatches the
existing parent validation/build dependencies, and records calls from:

```python
monkeypatch.setattr(
    actions,
    "migrate_legacy_project_dockerfile",
    lambda project_dir: calls.append(("migrate", project_dir)) or True,
)
```

Assert the migration call occurs after selected-parent validation and before
`build_or_reuse_project`. Also assert a newly created project does not invoke
the migration helper.

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/unit/installer/test_actions.py -q
```

Expected: FAIL because the installer action does not invoke the migration.

- [ ] **Step 3: Wire the helper into the existing-project path**

Import `migrate_legacy_project_dockerfile` from `amd_ai.project.init`. In
`ProductionInstallerActions.initialize_project`, invoke it after loading and
validating the existing project config and before `build_or_reuse_project`.
Do not invoke it after `create_project`, because a new project already receives
the current template.

- [ ] **Step 4: Run focused installer and project tests**

Run:

```bash
uv run pytest tests/unit/installer/test_actions.py \
  tests/unit/project/test_init.py tests/unit/project/test_build.py \
  tests/container/test_project_dockerfile.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/amd_ai/installer/actions.py tests/unit/installer/test_actions.py
git commit -m "fix: migrate project template before resumed builds"
```

### Task 3: Document, Verify, And Publish

**Files:**
- Modify: `README.md`
- Modify: `docs/releases/v0.3.3.md`

- [ ] **Step 1: Update recovery documentation**

Change the Docker Hub frontend troubleshooting entry to state that current
`main` automatically migrates an untouched legacy generated Dockerfile during
`PROJECT_INIT`; customized Dockerfiles remain unchanged and require an explicit
manual edit.

Add the same migration boundary to `docs/releases/v0.3.3.md`.

- [ ] **Step 2: Run full local verification**

Run:

```bash
uv run pytest -m 'not hardware' -q
git diff --check
```

Expected: all non-hardware tests pass with one hardware test deselected. No
test performs a real SWR, GHCR, Docker Hub, GitHub, or CodeArts download.

- [ ] **Step 3: Commit documentation**

```bash
git add README.md docs/releases/v0.3.3.md
git commit -m "docs: explain safe legacy project migration"
```

- [ ] **Step 4: Integrate and push both code remotes**

Fast-forward `main`, verify the focused regression on the merged result, then
push:

```bash
git -C /app/imgMaker merge --ff-only fix/project-template-migration
git -C /app/imgMaker push origin main
git -C /app/imgMaker push china main
```

Expected: GitHub and CodeArts `main` point to the same documentation commit.
Do not create `v0.3.3` until China-host installation acceptance passes.
