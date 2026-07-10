# Protected Python Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow ordinary `pip` and `pip3` commands inside managed project containers while keeping the verified four-component Torch stack immutable and storing all other Python packages in a project-private, replayable, atomic overlay.

**Architecture:** A protected pip frontend resolves against the installed `/opt/venv` so compatible Torch requirements are externally satisfied, constrains all four protected distributions to their exact full versions, materializes every non-protected result as a hashed project-local wheel, and installs a complete lock into a fresh generation. The generation is validated before a relative `current` symlink is atomically switched. Query commands run pip with the effective overlay path; direct base pip remains unable to mutate the read-only parent.

**Tech Stack:** Python 3.12, pip 24.0 `--dry-run --report` schema v1, `pip._vendor.packaging` PEP 440/508 and wheel filename parsers pinned by the parent image, `fcntl.flock`, `hashlib`, `json`, `pathlib`, pytest, Docker.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `src/amd_ai/overlay/__init__.py` | Public constants and overlay exception exports |
| `src/amd_ai/overlay/models.py` | Immutable paths, profile, artifact, lock, state, and request models |
| `src/amd_ai/overlay/policy.py` | pip command parsing, forbidden option checks, protected-name/source policy |
| `src/amd_ai/overlay/requirements.py` | Recursive requirements-file loading, PEP 508 parsing, local source/wheel canonicalization |
| `src/amd_ai/overlay/resolver.py` | Parent-constrained pip report, wheel materialization, protected-result rejection |
| `src/amd_ai/overlay/lock.py` | Deterministic requirements lock rendering/parsing and SHA-256 validation |
| `src/amd_ai/overlay/transaction.py` | Project lock, generation build, fsync, atomic symlink and mirror updates |
| `src/amd_ai/overlay/verify.py` | Structural overlay validation and effective dependency check hook |
| `src/amd_ai/overlay/cli.py` | Protected pip install/uninstall/list/show/check/freeze behavior |
| `images/common/protected-pip` | Container executable forwarding argv to the overlay CLI |
| `images/rocm-pytorch/Dockerfile` | Embed protected pip and expose it before `/opt/venv/bin` |
| `templates/project/.dockerignore` | Preserve the existing `.amd-ai` build-context exclusion |
| `tests/unit/overlay/` | Pure policy, resolver, lock, and transaction tests |
| `tests/cli/test_overlay_commands.py` | CLI dispatch and exit-code tests |
| `tests/container/test_readonly_overlay.py` | Real image install, persistence, conflict, and interruption tests |

`overlay.requirements.lock` is pip-compatible text containing only project-local wheel URLs and SHA-256 hashes. `overlay-state.json` binds the generation, input digest, lock digest, parent profile ID, parent config digest, and creation time.

### Task 1: Define Overlay Paths, Protected Profile, and State Models

**Files:**
- Create: `src/amd_ai/overlay/__init__.py`
- Create: `src/amd_ai/overlay/models.py`
- Create: `tests/unit/overlay/__init__.py`
- Create: `tests/unit/overlay/test_models.py`

- [ ] **Step 1: Write failing path and model tests**

```python
from pathlib import Path

import pytest

from amd_ai.overlay.models import (
    OverlayError,
    OverlayPaths,
    ProtectedComponent,
    ProtectedProfile,
)


def test_overlay_paths_are_project_local_and_current_is_relative(tmp_path):
    project = tmp_path / "demo"
    project.mkdir()
    paths = OverlayPaths.for_project(project)

    assert paths.root == project / ".amd-ai"
    assert paths.current == paths.root / "current"
    assert paths.generation("20260710T120000Z-a1b2c3d4") == (
        paths.generations / "20260710T120000Z-a1b2c3d4"
    )


def test_overlay_paths_reject_symlinked_control_root(tmp_path):
    project = tmp_path / "demo"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / ".amd-ai").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OverlayError, match="symbolic link"):
        OverlayPaths.for_project(project)


def test_protected_profile_requires_all_exact_full_versions():
    profile = ProtectedProfile(
        profile_id="rocm-7.2.1-py3.12-torch-2.9.1",
        parent_config_digest="sha256:" + "a" * 64,
        components=(
            ProtectedComponent("torch", "2.9.1+rocm7.2.1.lw.gitff65f5bc"),
            ProtectedComponent("torchvision", "0.24.0+rocm7.2.1.gitb919bd0c"),
            ProtectedComponent("torchaudio", "2.9.0+rocm7.2.1.gitd0d0d0d"),
            ProtectedComponent("triton", "3.5.1+rocm7.2.1.gita272dfa8"),
        ),
    )

    assert profile.version_for("Torch_Vision") == (
        "0.24.0+rocm7.2.1.gitb919bd0c"
    )
```

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run: `uv run pytest tests/unit/overlay/test_models.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'amd_ai.overlay'`.

- [ ] **Step 3: Implement the immutable models and path boundary**

Create `src/amd_ai/overlay/models.py` with these public definitions:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


PROTECTED_DISTRIBUTIONS = frozenset(
    {"torch", "torchvision", "torchaudio", "triton"}
)
GENERATION_PATTERN = re.compile(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}")
DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


class OverlayError(RuntimeError):
    pass


def canonicalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class OverlayPaths:
    project: Path
    root: Path
    inputs: Path
    lock: Path
    state: Path
    transaction_lock: Path
    generations: Path
    current: Path
    quarantine: Path
    artifacts: Path
    logs: Path

    @classmethod
    def for_project(cls, project: Path) -> "OverlayPaths":
        resolved = project.resolve()
        if not resolved.is_dir():
            raise OverlayError(f"project directory does not exist: {resolved}")
        root = resolved / ".amd-ai"
        if root.is_symlink():
            raise OverlayError(f"overlay root must not be a symbolic link: {root}")
        return cls(
            project=resolved,
            root=root,
            inputs=root / "overlay.requirements.in",
            lock=root / "overlay.requirements.lock",
            state=root / "overlay-state.json",
            transaction_lock=root / "transaction.lock",
            generations=root / "generations",
            current=root / "current",
            quarantine=root / "quarantine",
            artifacts=root / "artifacts" / "sha256",
            logs=root / "logs",
        )

    def generation(self, transaction_id: str) -> Path:
        if GENERATION_PATTERN.fullmatch(transaction_id) is None:
            raise OverlayError(f"invalid transaction ID: {transaction_id!r}")
        return self.generations / transaction_id


@dataclass(frozen=True)
class ProtectedComponent:
    name: str
    version: str

    def __post_init__(self) -> None:
        canonical = canonicalize_name(self.name)
        if canonical not in PROTECTED_DISTRIBUTIONS:
            raise OverlayError(f"unknown protected distribution: {self.name}")
        if not self.version or self.version.split("+", 1)[0] == self.version:
            raise OverlayError(
                f"protected distribution requires a full local version: {self.name}"
            )
        object.__setattr__(self, "name", canonical)


@dataclass(frozen=True)
class ProtectedProfile:
    profile_id: str
    parent_config_digest: str
    components: tuple[ProtectedComponent, ...]

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise OverlayError("protected profile ID is empty")
        if DIGEST_PATTERN.fullmatch(self.parent_config_digest) is None:
            raise OverlayError("protected parent config digest is invalid")
        names = tuple(component.name for component in self.components)
        if set(names) != PROTECTED_DISTRIBUTIONS or len(names) != 4:
            raise OverlayError("protected profile must contain each component once")

    def version_for(self, name: str) -> str:
        canonical = canonicalize_name(name)
        for component in self.components:
            if component.name == canonical:
                return component.version
        raise OverlayError(f"distribution is not protected: {name}")
```

Export `OverlayError`, `OverlayPaths`, `PROTECTED_DISTRIBUTIONS`, and `ProtectedProfile` from `src/amd_ai/overlay/__init__.py`.

- [ ] **Step 4: Run model tests**

Run: `uv run pytest tests/unit/overlay/test_models.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit the model boundary**

```bash
git add src/amd_ai/overlay tests/unit/overlay
git commit -m "feat: define protected overlay model boundary"
```

### Task 2: Parse the Supported pip Command Surface

**Files:**
- Create: `src/amd_ai/overlay/policy.py`
- Create: `tests/unit/overlay/test_policy.py`

- [ ] **Step 1: Write failing command-policy tests**

```python
import pytest

from amd_ai.overlay.policy import PipPolicyError, parse_pip_request


@pytest.mark.parametrize(
    "argv",
    [
        ["install", "--user", "requests"],
        ["install", "--target=/tmp/site", "requests"],
        ["install", "--prefix", "/tmp/prefix", "requests"],
        ["install", "--root", "/tmp/root", "requests"],
        ["install", "-e", "."],
        ["install", "git+https://github.com/pallets/flask.git"],
    ],
)
def test_install_rejects_non_transactional_targets(argv):
    with pytest.raises(PipPolicyError):
        parse_pip_request(argv)


def test_install_keeps_requirements_and_secret_free_index_options():
    request = parse_pip_request(
        ["install", "--index-url", "https://packages.example/simple", "-r", "deps.txt"]
    )

    assert request.command == "install"
    assert request.requirements_files == ("deps.txt",)
    assert request.requirements == ()
    assert request.resolver_options == (
        "--index-url",
        "https://packages.example/simple",
    )


def test_query_and_uninstall_commands_have_narrow_grammar():
    assert parse_pip_request(["show", "torch"]).names == ("torch",)
    assert parse_pip_request(["freeze"]).command == "freeze"
    assert parse_pip_request(["uninstall", "-y", "requests"]).assume_yes is True
```

- [ ] **Step 2: Confirm the tests fail for the missing parser**

Run: `uv run pytest tests/unit/overlay/test_policy.py -q`

Expected: collection fails because `amd_ai.overlay.policy` does not exist.

- [ ] **Step 3: Implement a deterministic argv parser**

Create `PipRequest`, `PipPolicyError`, option tables, and `parse_pip_request()` in `src/amd_ai/overlay/policy.py`. Use argv iteration, never shell parsing. The public request model is:

```python
@dataclass(frozen=True)
class PipRequest:
    command: str
    requirements: tuple[str, ...] = ()
    requirements_files: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    resolver_options: tuple[str, ...] = ()
    upgrade: bool = False
    assume_yes: bool = False
    query_options: tuple[str, ...] = ()
```

The parser must enforce these exact rules:

```python
FORBIDDEN_OPTIONS = frozenset(
    {
        "--user",
        "--target",
        "--prefix",
        "--root",
        "--editable",
        "-e",
        "--break-system-packages",
        "--ignore-installed",
        "--force-reinstall",
        "--python",
    }
)
VALUE_OPTIONS = frozenset(
    {
        "--index-url",
        "--extra-index-url",
        "--index",
        "--find-links",
    }
)
FLAG_OPTIONS = frozenset({"--no-index", "--pre"})
QUERY_COMMANDS = frozenset({"list", "show", "check", "freeze"})
```

Reject option abbreviations, NUL/newline characters, unknown subcommands, unknown install options, credential-bearing or non-HTTPS index URLs, bare VCS URLs without a PEP 508 distribution name, non-Git VCS schemes, and Git URLs without an exact 40-character lowercase commit. Credentialed private indexes must be supplied through pip environment variables or `.netrc`, so no secret is written to overlay inputs or logs. A valid `name @ git+https://host/repository.git@<40-hex-commit>` is passed to requirement inspection and materialized as a wheel before locking.

- [ ] **Step 4: Run policy tests**

Run: `uv run pytest tests/unit/overlay/test_policy.py -q`

Expected: all policy tests pass.

- [ ] **Step 5: Commit the pip policy**

```bash
git add src/amd_ai/overlay/policy.py tests/unit/overlay/test_policy.py
git commit -m "feat: enforce protected pip command policy"
```

### Task 3: Expand Requirements and Block Direct Protected Sources

**Files:**
- Create: `src/amd_ai/overlay/requirements.py`
- Create: `tests/unit/overlay/test_requirements.py`

- [ ] **Step 1: Write failing protected-requirement tests**

```python
from pathlib import Path

import pytest

from amd_ai.overlay.models import ProtectedComponent, ProtectedProfile
from amd_ai.overlay.requirements import RequirementPolicyError, inspect_requirements


@pytest.fixture
def profile():
    return ProtectedProfile(
        "stable",
        "sha256:" + "a" * 64,
        (
            ProtectedComponent("torch", "2.9.1+rocm7.2.1.build1"),
            ProtectedComponent("torchvision", "0.24.0+rocm7.2.1.build1"),
            ProtectedComponent("torchaudio", "2.9.0+rocm7.2.1.build1"),
            ProtectedComponent("triton", "3.5.1+rocm7.2.1.build1"),
        ),
    )


def test_bare_and_compatible_protected_requirements_are_external(profile, tmp_path):
    result = inspect_requirements(
        ("torch", "Torch_Vision>=0.24", "requests==2.32.5"),
        (),
        project=tmp_path,
        profile=profile,
    )

    assert result.external == ("torch", "torchvision")
    assert result.resolver_inputs == ("requests==2.32.5",)


@pytest.mark.parametrize(
    "requirement",
    [
        "torch==2.8.0",
        "torch @ https://download.example/torch-2.9.1-cp312.whl",
        "./torch-2.9.1-py3-none-any.whl",
        "triton @ file:///workspace/triton-3.5.1.whl",
    ],
)
def test_incompatible_or_source_bound_protected_requirement_is_blocked(
    requirement, profile, tmp_path
):
    with pytest.raises(RequirementPolicyError):
        inspect_requirements(
            (requirement,), (), project=tmp_path, profile=profile
        )


def test_nested_requirements_are_bounded_to_project(profile, tmp_path):
    (tmp_path / "nested.txt").write_text("torch>=2.9\nrequests\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text(
        "-r nested.txt\n", encoding="utf-8"
    )

    result = inspect_requirements(
        (), ("requirements.txt",), project=tmp_path, profile=profile
    )

    assert result.external == ("torch",)
    assert result.resolver_inputs == ("requests",)


def test_exact_commit_git_is_materialized_but_mutable_ref_is_rejected(profile, tmp_path):
    commit = "a" * 40
    result = inspect_requirements(
        (f"demo @ git+https://github.com/example/demo.git@{commit}",),
        (),
        project=tmp_path,
        profile=profile,
    )

    assert result.vcs_inputs == (
        f"demo @ git+https://github.com/example/demo.git@{commit}",
    )
    with pytest.raises(RequirementPolicyError, match="exact commit"):
        inspect_requirements(
            ("demo @ git+https://github.com/example/demo.git@main",),
            (),
            project=tmp_path,
            profile=profile,
        )
```

- [ ] **Step 2: Verify the requirements tests fail**

Run: `uv run pytest tests/unit/overlay/test_requirements.py -q`

Expected: collection fails because `amd_ai.overlay.requirements` is missing.

- [ ] **Step 3: Implement structured PEP 508 and wheel-name inspection**

Use these pinned parsers from the parent pip distribution:

```python
from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
from pip._vendor.packaging.utils import (
    InvalidWheelFilename,
    canonicalize_name,
    parse_wheel_filename,
)
from pip._vendor.packaging.version import Version
```

Implement:

```python
@dataclass(frozen=True)
class InspectedRequirements:
    resolver_inputs: tuple[str, ...]
    external: tuple[str, ...]
    local_inputs: tuple[Path, ...]
    vcs_inputs: tuple[str, ...]
```

Implement `inspect_requirements(requirements, requirements_files, *, project, profile) -> InspectedRequirements`. Its recursion rules are exact: resolve requirement files relative to the including file, call `resolve(strict=True)`, require each file to remain under the project root, reject symlinks leaving the project, reject include cycles, join backslash continuations, ignore blank/comment-only lines, recurse only for `-r` and `--requirement`, and pass validated index-option lines to pip without treating them as package names.

For every parsed protected `Requirement`, reject direct URLs and accept it only when its marker applies and its specifier contains `Version(profile.version_for(name))`. Remove accepted protected requirements from resolver inputs and report their canonical names in sorted order. For local `.whl` paths, use `parse_wheel_filename`; reject a protected canonical name before any resolver command runs. For a local source directory, require it to remain under the project and defer its wheel name check to Task 4. For VCS, accept only an explicit non-protected name bound to `git+https` and one exact 40-character lowercase commit; reject branches, tags, abbreviated commits, URL credentials, and mutable refs.

- [ ] **Step 4: Run requirements tests**

Run: `uv run pytest tests/unit/overlay/test_requirements.py -q`

Expected: all tests pass, including separator, extras, nested include, URL, local wheel, exact-commit Git, mutable Git ref, and credential-bearing nested index variants.

- [ ] **Step 5: Commit requirement inspection**

```bash
git add src/amd_ai/overlay/requirements.py tests/unit/overlay/test_requirements.py
git commit -m "feat: inspect protected requirements structurally"
```

### Task 4: Resolve Against the Verified Parent and Materialize Wheels

**Files:**
- Create: `src/amd_ai/overlay/resolver.py`
- Create: `tests/unit/overlay/fakes.py`
- Create: `tests/unit/overlay/test_resolver.py`

- [ ] **Step 1: Write failing resolver-report tests**

```python
import json
from pathlib import Path

import pytest

from amd_ai.overlay.models import ProtectedComponent, ProtectedProfile
from amd_ai.overlay.resolver import ResolverError, parse_pip_report


def test_report_accepts_hashed_nonprotected_wheel():
    payload = {
        "version": "1",
        "pip_version": "24.0",
        "install": [
            {
                "download_info": {
                    "url": "https://files.pythonhosted.org/x/requests.whl",
                    "archive_info": {"hashes": {"sha256": "b" * 64}},
                },
                "requested": True,
                "metadata": {"name": "requests", "version": "2.32.5"},
            }
        ],
    }

    items = parse_pip_report(json.dumps(payload))

    assert items[0].name == "requests"
    assert items[0].sha256 == "b" * 64
    assert items[0].requested is True


def test_report_blocks_transitive_protected_distribution():
    payload = {
        "version": "1",
        "pip_version": "24.0",
        "install": [
            {
                "download_info": {
                    "url": "https://example/torch.whl",
                    "archive_info": {"hashes": {"sha256": "c" * 64}},
                },
                "requested": False,
                "metadata": {"name": "Torch", "version": "2.8.0"},
            }
        ],
    }

    with pytest.raises(ResolverError, match="protected distribution"):
        parse_pip_report(json.dumps(payload))
```

- [ ] **Step 2: Run resolver tests and verify failure**

Run: `uv run pytest tests/unit/overlay/test_resolver.py -q`

Expected: collection fails because `amd_ai.overlay.resolver` is missing.

- [ ] **Step 3: Implement the resolver command and strict report parser**

Define these immutable records:

```python
@dataclass(frozen=True)
class ReportItem:
    name: str
    version: str
    url: str
    sha256: str
    requested: bool


@dataclass(frozen=True)
class WheelArtifact:
    name: str
    version: str
    sha256: str
    path: Path
    requested: bool
```

Build the resolver argv exactly from arrays:

```python
def resolver_argv(
    *,
    input_path: Path,
    constraints_path: Path,
    report_path: Path,
    resolver_options: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        "/opt/venv/bin/python",
        "-m",
        "pip",
        "install",
        "--dry-run",
        "--disable-pip-version-check",
        "--report",
        str(report_path),
        "--constraint",
        str(constraints_path),
        *resolver_options,
        "--requirement",
        str(input_path),
    )
```

Run it with a copied environment where `PYTHONPATH=/opt/amd-ai/src`, `PYTHONNOUSERSITE=1`, `PIP_DISABLE_PIP_VERSION_CHECK=1`, and `PIP_CACHE_DIR` points to the current transaction's private staging directory. This deliberately hides the current overlay, so every project package is resolved from the complete input while parent packages remain visible as installed. Remove the transaction cache on every success or failure; a resolver that inspects an incompatible protected candidate cannot leave a large Torch wheel in the persistent project cache.

Write four exact full-version constraints from `ProtectedProfile`. `parse_pip_report()` must require schema `1`, a list of install records, canonical non-protected names, nonempty versions, HTTPS/file URLs, and one lowercase SHA-256 per artifact. A protected report item is always an error, including a version equal to the parent.

- [ ] **Step 4: Implement wheel materialization and local-source builds**

For each report item, write a private one-line temporary requirement containing its URL and hash, then run:

```python
(
    "/opt/venv/bin/python",
    "-m",
    "pip",
    "download",
    "--disable-pip-version-check",
    "--no-deps",
    "--require-hashes",
    "--dest",
    str(download_dir),
    "--requirement",
    str(single_requirement),
)
```

If the downloaded artifact is not a wheel, build it with:

```python
(
    "/opt/venv/bin/python",
    "-m",
    "pip",
    "wheel",
    "--disable-pip-version-check",
    "--no-deps",
    "--wheel-dir",
    str(wheel_dir),
    str(downloaded_artifact),
)
```

Local source directories use the same wheel command directly. Exact-commit Git inputs are built with the full PEP 508 requirement as one argv item; inspect pip's `direct_url.json` in the built wheel metadata and require the recorded commit ID to equal the requested commit before accepting it. Add each local/VCS wheel to the temporary resolver input so its transitive dependencies are resolved against the verified parent. Validate the final wheel filename with `parse_wheel_filename`, require its canonical name/version to equal the report item, hash its bytes, and atomically place it at `.amd-ai/artifacts/sha256/<hash>/<filename>` with mode `0444`. Reject every protected wheel name. Logs receive redacted argv and output; temporary requirement files and build directories are removed in `finally`.

- [ ] **Step 5: Run resolver tests**

Run: `uv run pytest tests/unit/overlay/test_resolver.py -q`

Expected: report schema, protected transitive result, missing hash, URL credentials, wheel mismatch, local source, and command environment tests all pass.

- [ ] **Step 6: Commit resolver and materializer**

```bash
git add src/amd_ai/overlay/resolver.py tests/unit/overlay
git commit -m "feat: resolve overlay wheels against verified parent"
```

### Task 5: Render and Validate the Replayable Overlay Lock

**Files:**
- Create: `src/amd_ai/overlay/lock.py`
- Create: `tests/unit/overlay/test_lock.py`

- [ ] **Step 1: Write failing lock round-trip tests**

```python
import hashlib
from pathlib import Path

import pytest

from amd_ai.overlay.lock import LockError, parse_lock, render_lock
from amd_ai.overlay.resolver import WheelArtifact


def test_lock_is_sorted_hashed_and_project_local(tmp_path):
    digest = hashlib.sha256(b"wheel").hexdigest()
    artifact = (
        tmp_path
        / ".amd-ai/artifacts/sha256"
        / digest
        / "requests-2.32.5-py3-none-any.whl"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"wheel")
    wheel = WheelArtifact("requests", "2.32.5", digest, artifact, True)

    text = render_lock((wheel,), project=tmp_path)

    assert text == (
        "requests @ file:///workspace/.amd-ai/artifacts/sha256/"
        + digest
        + "/requests-2.32.5-py3-none-any.whl \\"
        + "\n    --hash=sha256:"
        + digest
        + "\n"
    )
    assert parse_lock(text)[0].name == "requests"


def test_lock_rejects_protected_distribution():
    text = "torch @ file:///workspace/.amd-ai/artifacts/sha256/" + "a" * 64
    text += "/torch-2.9.1-py3-none-any.whl --hash=sha256:" + "a" * 64 + "\n"

    with pytest.raises(LockError, match="protected"):
        parse_lock(text)
```

- [ ] **Step 2: Verify lock tests fail**

Run: `uv run pytest tests/unit/overlay/test_lock.py -q`

Expected: collection fails because `amd_ai.overlay.lock` is missing.

- [ ] **Step 3: Implement deterministic lock rendering and parsing**

Use only generated direct wheel lines. Require canonical-name sort order, one entry per distribution, a `/workspace/.amd-ai/artifacts/sha256/<digest>/<wheel>` URL, matching path digest and `--hash`, a valid wheel filename, exact metadata name/version, and no protected names. Reject index directives, includes, editable lines, VCS URLs, HTTP URLs, absolute host paths, duplicate packages, duplicate hashes, and line continuations that do not end in exactly one hash line.

Expose:

```python
@dataclass(frozen=True)
class LockedWheel:
    name: str
    version: str
    sha256: str
    container_path: str


def lock_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

Implement `render_lock(artifacts, *, project) -> str` and `parse_lock(text) -> tuple[LockedWheel, ...]` with the strict rules above. Verify actual artifact bytes before accepting a lock for installation or repair.

- [ ] **Step 4: Run lock tests**

Run: `uv run pytest tests/unit/overlay/test_lock.py -q`

Expected: all lock round-trip and corruption tests pass.

- [ ] **Step 5: Commit lock support**

```bash
git add src/amd_ai/overlay/lock.py tests/unit/overlay/test_lock.py
git commit -m "feat: add replayable overlay wheel lock"
```

### Task 6: Build and Atomically Switch Generations

**Files:**
- Create: `src/amd_ai/overlay/transaction.py`
- Create: `tests/unit/overlay/test_transaction.py`

- [ ] **Step 1: Write failing atomicity tests**

```python
import os
from pathlib import Path

import pytest

from amd_ai.overlay.models import OverlayPaths
from amd_ai.overlay.transaction import (
    TransactionError,
    activate_generation,
    resolve_current_generation,
)


def test_activation_uses_relative_symlink_inside_control_root(tmp_path):
    project = tmp_path / "demo"
    project.mkdir()
    paths = OverlayPaths.for_project(project)
    generation = paths.generation("20260710T120000Z-a1b2c3d4")
    (generation / "site-packages").mkdir(parents=True)

    activate_generation(paths, generation)

    assert paths.current.is_symlink()
    assert os.readlink(paths.current) == "generations/20260710T120000Z-a1b2c3d4"
    assert resolve_current_generation(paths) == generation


def test_activation_failure_keeps_previous_generation(tmp_path, monkeypatch):
    project = tmp_path / "demo"
    project.mkdir()
    paths = OverlayPaths.for_project(project)
    old = paths.generation("20260710T120000Z-a1b2c3d4")
    new = paths.generation("20260710T120001Z-b1c2d3e4")
    (old / "site-packages").mkdir(parents=True)
    (new / "site-packages").mkdir(parents=True)
    activate_generation(paths, old)
    monkeypatch.setattr(os, "replace", lambda source, target: (_ for _ in ()).throw(OSError("stop")))

    with pytest.raises(TransactionError, match="activate"):
        activate_generation(paths, new)

    assert os.readlink(paths.current) == "generations/20260710T120000Z-a1b2c3d4"
```

- [ ] **Step 2: Verify transaction tests fail**

Run: `uv run pytest tests/unit/overlay/test_transaction.py -q`

Expected: collection fails because `amd_ai.overlay.transaction` is missing.

- [ ] **Step 3: Implement lock acquisition and generation layout**

Use `fcntl.flock(fd, LOCK_EX | LOCK_NB)` on `.amd-ai/transaction.lock`; report the owning PID written after lock acquisition. Create `.amd-ai`, `generations`, `artifacts`, `logs`, and `quarantine` as real mode-`0700` directories owned by the runtime UID. Reject symlinks at every control path.

Generate IDs with UTC `YYYYMMDDTHHMMSSZ-` plus the first eight hex digits of `secrets.token_hex(4)`. Each new generation starts as a new directory and contains an empty `site-packages` directory plus private transaction log.

- [ ] **Step 4: Implement installation, validation hook, and atomic activation**

Install the lock into the empty target with:

```python
def install_argv(lock_path: Path, site_packages: Path) -> tuple[str, ...]:
    return (
        "/opt/venv/bin/python",
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-index",
        "--no-deps",
        "--no-compile",
        "--require-hashes",
        "--target",
        str(site_packages),
        "--requirement",
        str(lock_path),
    )
```

Write the generation input, lock, and state with temporary files, `flush`, `os.fsync`, and `os.replace`. Fsync the generation directory before activation. `activate_generation()` must create `.current.<transaction-id>.tmp` with target `generations/<transaction-id>`, validate that both link and resolved target remain under `.amd-ai`, then call `os.replace(temp_link, current)` and fsync `.amd-ai`.

After activation, atomically mirror the generation metadata to the root paths. If mirror update fails, return a warning state recoverable from `current`; never roll back a valid current link to a partially known target.

Add `mark_generation_healthy(paths, generation_id)` and bounded garbage collection. A successful startup writes a fsynced health marker, retains only the current and immediately previous healthy generation, and removes only artifact hashes unreferenced by those two validated locks. Reject every deletion candidate that is a symlink or resolves outside the selected project's `.amd-ai` root.

- [ ] **Step 5: Run transaction tests**

Run: `uv run pytest tests/unit/overlay/test_transaction.py -q`

Expected: lock contention, path traversal, existing external symlink, install failure, validation failure, activation failure, mirror recovery, and successful switch tests pass.

- [ ] **Step 6: Commit transaction support**

```bash
git add src/amd_ai/overlay/transaction.py tests/unit/overlay/test_transaction.py
git commit -m "feat: make overlay generations transactional"
```

### Task 7: Add Structural Verification and the Protected pip CLI

**Files:**
- Create: `src/amd_ai/overlay/verify.py`
- Create: `src/amd_ai/overlay/cli.py`
- Create: `tests/unit/overlay/test_verify.py`
- Create: `tests/cli/test_overlay_commands.py`

- [ ] **Step 1: Write failing verification and CLI tests**

```python
from pathlib import Path

import pytest

from amd_ai.overlay.verify import OverlayVerificationError, scan_protected_entries


@pytest.mark.parametrize(
    "name",
    [
        "torch",
        "torch.py",
        "Torch-2.9.1.dist-info",
        "torchvision-0.24.0.egg-info",
        "triton",
    ],
)
def test_structural_scan_blocks_protected_shadow(name, tmp_path):
    path = tmp_path / name
    path.mkdir() if "." not in name else path.write_text("", encoding="utf-8")

    with pytest.raises(OverlayVerificationError, match="protected"):
        scan_protected_entries(tmp_path)
```

CLI tests must inject fake resolver, wheel materializer, transaction, and query runner. Assert:

```python
assert overlay_cli.main(["install", "torch"]) == 0
assert "already satisfied by verified parent" in capsys.readouterr().out
assert overlay_cli.main(["uninstall", "torch", "-y"]) == 2
assert overlay_cli.main(["install", "torch==2.8.0"]) == 2
assert overlay_cli.main(["list", "--format=json"]) == 0
```

- [ ] **Step 2: Run verification and CLI tests**

Run: `uv run pytest tests/unit/overlay/test_verify.py tests/cli/test_overlay_commands.py -q`

Expected: collection fails for missing modules.

- [ ] **Step 3: Implement structural and dependency verification**

`scan_protected_entries(site_packages)` must canonicalize top-level package names and every `.dist-info`/`.egg-info` prefix. It rejects `torch.py`, protected package directories, and protected metadata directories.

Run effective pip dependency checking with:

```python
(
    "/opt/venv/bin/python",
    "-m",
    "pip",
    "check",
    "--disable-pip-version-check",
)
```

Set `PYTHONPATH=<candidate site-packages>:/opt/amd-ai/src` and `PYTHONNOUSERSITE=1`. A nonzero result rejects the generation. The deeper protected import identity check is added by the managed runtime plan and registered through a `verify_effective_stack(profile, site_packages, runtime=False)` hook called here before activation.

- [ ] **Step 4: Implement install, uninstall, and query orchestration**

`amd_ai.overlay.cli.main(argv)` must:

1. Require `/workspace/amd-ai-project.toml` and derive `OverlayPaths` from `/workspace`.
2. Load exact full component versions from `/opt/amd-ai/torch-manifest.json`, the profile ID from `/opt/amd-ai/profile.env`, and the exact parent config digest from `AMD_AI_PARENT_CONFIG_DIGEST` supplied by managed project-run.
3. Parse argv with Task 2 policy.
4. Before every mutation, verify the base Torch manifest and current effective stack; a damaged current stack is blocked with a repair instruction before any resolver or artifact command runs.
5. For `install`, lock the project, inspect all requirements, report compatible protected names, combine canonical current top-level roots with requested roots, resolve without current overlay, materialize wheels, render a complete lock, and call the transaction builder.
6. For `uninstall`, reject protected names, require each name to be a current top-level root, remove it, then resolve and build a complete new generation.
7. For `list`, `show`, `check`, and `freeze`, execute `/opt/venv/bin/python -m pip <command>` with `PYTHONPATH=/workspace/.amd-ai/current/site-packages:/opt/amd-ai/src` and pass only the query options accepted by policy.
8. Return `0` for success, `1` for an interrupted or action-required state, and `2` for policy, resolution, lock, or verification failure.

If an install request contains only compatible protected requirements and no non-protected change, validate the parent/effective stack, print `already satisfied by verified parent`, and leave `current` unchanged.

All subprocess logs go to `.amd-ai/logs/<transaction-id>.log` with URL userinfo and values for environment names containing `TOKEN`, `SECRET`, `PASSWORD`, or `KEY` replaced by `<redacted>`.

- [ ] **Step 5: Run overlay unit and CLI tests**

Run: `uv run pytest tests/unit/overlay tests/cli/test_overlay_commands.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit the protected pip CLI**

```bash
git add src/amd_ai/overlay tests/unit/overlay tests/cli/test_overlay_commands.py
git commit -m "feat: expose transactional protected pip"
```

### Task 8: Embed Protected pip and Configure Managed Project Paths

**Files:**
- Create: `images/common/protected-pip`
- Modify: `images/rocm-pytorch/Dockerfile`
- Modify: `src/amd_ai/project/run.py`
- Modify: `tests/container/test_rocm_pytorch_dockerfile.py`
- Modify: `tests/unit/project/test_run.py`

- [ ] **Step 1: Write failing image and run-argv assertions**

Add assertions that the PyTorch Dockerfile copies the executable, creates both names, and puts the protected directory first in `PATH`:

```python
assert "COPY images/common/protected-pip /opt/amd-ai/bin/pip" in dockerfile
assert "ln -s pip /opt/amd-ai/bin/pip3" in dockerfile
assert 'PATH="/opt/amd-ai/bin:/opt/venv/bin:/opt/rocm/bin:' in dockerfile
assert 'assert pip.__version__ == "24.0"' in dockerfile
assert ".amd-ai" in Path("templates/project/.dockerignore").read_text(encoding="utf-8")
```

Extend `test_normal_run_is_unprivileged_and_uses_private_ipc`:

```python
assert "PYTHONNOUSERSITE=1" in argv
assert "PYTHONDONTWRITEBYTECODE=1" in argv
assert "AMD_AI_OVERLAY=/workspace/.amd-ai/current/site-packages" in argv
assert "PYTHONPATH=/workspace/.amd-ai/current/site-packages:/opt/amd-ai/src" in argv
```

- [ ] **Step 2: Confirm focused tests fail**

Run:

```bash
uv run pytest tests/container/test_rocm_pytorch_dockerfile.py \
  tests/unit/project/test_run.py::test_normal_run_is_unprivileged_and_uses_private_ipc -q
```

Expected: assertions fail because the wrapper and environment are absent.

- [ ] **Step 3: Add the executable wrapper and image wiring**

Create `images/common/protected-pip`:

```bash
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/opt/amd-ai/src${PYTHONPATH:+:${PYTHONPATH}}"
exec /opt/venv/bin/python -m amd_ai.overlay.cli "$@"
```

In `images/rocm-pytorch/Dockerfile`, create `/opt/amd-ai/bin`, copy the wrapper as `pip`, chmod it `0755`, create a relative `pip3 -> pip` symlink, and prepend `/opt/amd-ai/bin` to the inherited `PATH`. During the build, import the three pinned `pip._vendor.packaging` APIs and assert pip version `24.0`; this makes a future base-pip change fail the image contract instead of silently changing requirement semantics. Do not replace or delete `/opt/venv/bin/pip`; it remains useful as a read-only failure boundary and resolver implementation.

- [ ] **Step 4: Add exact overlay environment to managed run argv**

In `build_run_argv()`, add these reserved values before user-configured environment:

```python
argv.extend(
    (
        "--env",
        "PYTHONNOUSERSITE=1",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "AMD_AI_OVERLAY=/workspace/.amd-ai/current/site-packages",
        "--env",
        "PYTHONPATH=/workspace/.amd-ai/current/site-packages:/opt/amd-ai/src",
    )
)
```

Keep these names reserved in project config so project TOML cannot override them.

- [ ] **Step 5: Run focused tests and wrapper mode check**

Run:

```bash
chmod +x images/common/protected-pip
uv run pytest tests/container/test_rocm_pytorch_dockerfile.py \
  tests/unit/project/test_run.py -q
```

Expected: all focused tests pass and Git records mode `100755` for the wrapper.

- [ ] **Step 6: Commit image integration**

```bash
git add images/common/protected-pip images/rocm-pytorch/Dockerfile \
  src/amd_ai/project/run.py tests/container/test_rocm_pytorch_dockerfile.py \
  tests/unit/project/test_run.py
git commit -m "feat: embed protected pip in managed projects"
```

### Task 9: Add Real Container Integration Coverage

**Files:**
- Create: `tests/container/test_readonly_overlay.py`

- [ ] **Step 1: Add a container marker test fixture and persistence test**

The test must create a temporary project through `bin/project-init`, build it, run two containers with the same project bind, and use a small fixed package:

```python
@pytest.mark.container
def test_plain_pip_install_persists_without_copying_torch(project_factory):
    project = project_factory("overlay-persist")
    first = project.run("pip", "install", "six==1.17.0")
    assert first.returncode == 0
    second = project.run(
        "python",
        "-c",
        "import six,torch; print(six.__version__); print(torch.__version__)",
    )
    assert second.returncode == 0
    assert "1.17.0" in second.stdout
    assert not any(
        name.startswith(("torch-", "torchvision-", "torchaudio-", "triton-"))
        for name in project.overlay_entries()
    )
```

Implement `project_factory` in the same test module using exact `subprocess.run([...])` argv, a temporary directory, and the local verified image. Do not invoke a shell.

- [ ] **Step 2: Add conflict, forbidden flag, and rollback tests**

Add real tests for:

```text
pip install torch
pip install 'torch==2.8.0'
pip install --target /tmp/site six
pip uninstall -y torch
pip install -r requirements-with-compatible-torch.txt
resolver process terminated before activation
invalid wheel hash during repair replay
```

Assert the compatible cases return 0 without a protected overlay entry, forbidden cases return 2, and failed transactions leave `os.readlink(.amd-ai/current)` unchanged.

- [ ] **Step 3: Run unit tests before the image build**

Run:

```bash
uv run pytest tests/unit/overlay tests/cli/test_overlay_commands.py \
  tests/unit/project/test_run.py tests/container/test_rocm_pytorch_dockerfile.py -q
```

Expected: zero failures.

- [ ] **Step 4: Build the modified parent serially and run container tests**

Run:

```bash
bin/image-build rocm-pytorch --profile profiles/torch/stable.env
uv run pytest -m container tests/container/test_readonly_overlay.py -q
```

Expected: the cached build completes, every overlay integration test passes, and `sudo docker image inspect` shows the current Git revision label. This build is developmental only; final verified publication occurs after all plans and hardware qualification.

- [ ] **Step 5: Commit integration tests**

```bash
git add tests/container/test_readonly_overlay.py
git commit -m "test: cover protected overlay in containers"
```

### Task 10: Run Overlay Regression and Storage Checks

**Files:**
- Modify only files implicated by a failing check

- [ ] **Step 1: Run the complete non-hardware suite**

Run: `uv run pytest -m 'not hardware' -q`

Expected: zero failures.

- [ ] **Step 2: Verify no protected artifacts or duplicate Torch files exist**

Run:

```bash
uv run pytest -m container \
  tests/container/test_readonly_overlay.py::test_plain_pip_install_persists_without_copying_torch -q
sudo -n docker run --rm \
  --entrypoint /opt/venv/bin/python \
  rocm-pytorch:7.2.1-py3.12-torch2.9.1 \
  -c 'import importlib.metadata as m; print(m.distribution("torch").locate_file(""))'
```

Expected: the integration test proves the project overlay has no protected package or metadata entry, and the parent distribution path is under `/opt/venv/lib/python3.12/site-packages`.

- [ ] **Step 3: Verify generation retention is bounded**

Install two successive versions of a small package, start one healthy managed container, and assert only the current and immediately previous successful generations remain. Artifact garbage collection may remove only hashes not referenced by either retained lock; it must never inspect or delete outside the selected project's `.amd-ai` directory.

- [ ] **Step 4: Record the plan completion checkpoint**

```bash
git status --short
git log --oneline -10
```

Expected: the worktree is clean and the overlay commits are present in task order.
