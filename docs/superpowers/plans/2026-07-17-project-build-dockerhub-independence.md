# Project Build Docker Hub Independence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent end-user project image builds from contacting Docker Hub for the `docker/dockerfile:1.7` frontend.

**Architecture:** Keep the immutable SWR/GHCR release images and project build command unchanged. Make the project Dockerfile compatible with BuildKit's bundled frontend by removing the external syntax directive and cache mount while preserving locked dependency installation and Torch integrity verification.

**Tech Stack:** Dockerfile, Docker Buildx/BuildKit, uv, pytest.

---

### Task 1: Make The Project Template Self-Contained

**Files:**
- Modify: `tests/container/test_project_dockerfile.py`
- Modify: `templates/project/Dockerfile`

- [ ] **Step 1: Write the failing template contract**

Replace the old cache-mount assertion in
`test_project_install_cannot_sync_or_replace_parent` with:

```python
assert not text.startswith("# syntax=")
assert "docker.io/docker/dockerfile" not in text
assert "--mount=" not in text
assert "RUN if [ -s /opt/amd-ai/project-locks/requirements.lock ]; then" in text
```

Keep the existing assertions for `uv pip install`, Torch constraints,
`--require-hashes`, and `torch-manifest.py verify`.

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/container/test_project_dockerfile.py::test_project_install_cannot_sync_or_replace_parent -q
```

Expected: FAIL because the template still starts with
`# syntax=docker/dockerfile:1.7` and contains `RUN --mount=type=cache`.

- [ ] **Step 3: Remove the external frontend dependency**

Change `templates/project/Dockerfile` from:

```dockerfile
# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE
```

to:

```dockerfile
ARG BASE_IMAGE
```

Change the dependency instruction from:

```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    if [ -s /opt/amd-ai/project-locks/requirements.lock ]; then \
```

to:

```dockerfile
RUN if [ -s /opt/amd-ai/project-locks/requirements.lock ]; then \
```

Do not change the locked uv arguments, Torch constraints, manifest
verification, user, labels, entrypoint, or parent image argument.

- [ ] **Step 4: Run focused verification**

Run:

```bash
uv run pytest tests/container/test_project_dockerfile.py tests/unit/project -q
```

Expected: all focused tests pass without contacting Docker Hub or SWR.

- [ ] **Step 5: Commit**

```bash
git add templates/project/Dockerfile tests/container/test_project_dockerfile.py
git commit -m "fix: avoid Docker Hub frontend for project builds"
```

### Task 2: Document Recovery And Publish

**Files:**
- Modify: `README.md`
- Modify: `docs/releases/v0.3.3.md`

- [ ] **Step 1: Document the resolved failure**

Add a README troubleshooting entry for:

```text
failed to resolve source metadata for docker.io/docker/dockerfile:1.7
```

State that current `main` no longer requires this frontend for project builds
and that rerunning the identical installer command resumes at `PROJECT_INIT`
without redownloading the verified SWR images.

Add the same compatibility note to `docs/releases/v0.3.3.md`.

- [ ] **Step 2: Run regression verification**

Run:

```bash
uv run pytest -m 'not hardware' -q
git diff --check
```

Expected: all non-hardware tests pass, one hardware test remains deselected,
and the diff check is silent. Tests must not perform a real SWR or Docker Hub
pull.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/releases/v0.3.3.md
git commit -m "docs: explain project build frontend recovery"
```

- [ ] **Step 4: Integrate and publish**

Fast-forward `main` from the primary worktree and push:

```bash
git -C /app/imgMaker merge --ff-only fix/project-build-no-dockerhub
git -C /app/imgMaker push origin main
```

Expected: `origin/main` points to the documentation commit. Do not create the
final `v0.3.3` tag until China-host installation acceptance is complete.
