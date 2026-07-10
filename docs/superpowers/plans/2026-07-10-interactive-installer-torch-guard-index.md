# Interactive Installer and Torch Guard Implementation Plan Index

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the approved interactive installer, immutable GHCR release chain, project-private transactional Python overlay, protected Torch identity checks, and rebuild-only repair workflow for the Strix Halo ROCm platform.

**Architecture:** Four bounded plans extend the existing host, image, project, and qualification layers instead of replacing them. The protected overlay lands first because it changes the runtime image contract; release identity follows, doctor/repair composes both, and the installer is the final orchestration layer. A final integration gate rebuilds and qualifies the exact source revision before publishing immutable GHCR digests and generating `profiles/releases/stable.json`.

**Tech Stack:** Python 3.12 standard library, pip 24.0 report schema v1, pip vendored PEP 440/508 parser, pytest 8.4.1, Bash, Docker Engine/BuildKit, GHCR, Ubuntu 24.04, ROCm 7.2.1, PyTorch 2.9.1, gfx1151.

---

## Repository Precondition

Implementation must run in a dedicated worktree. From `/app/imgMaker`, run once:

```bash
git status --short --branch
git worktree add .worktrees/interactive-installer-torch-guard \
  -b feature/interactive-installer-torch-guard main
cd .worktrees/interactive-installer-torch-guard
git rev-parse --show-toplevel
```

Expected: the original tree is clean, the new branch starts at the approved design and plan commit, and `git rev-parse` prints `/app/imgMaker/.worktrees/interactive-installer-torch-guard`.

Do not run image builds in parallel. The local images are about 25.4 GB and the host has limited free space; every plan uses exact image IDs and forbids broad Docker pruning.

## Plan Boundaries

| Order | Plan | Independently testable result | Depends on |
| --- | --- | --- | --- |
| 1 | [Protected Python overlay](./2026-07-10-protected-python-overlay.md) | Plain `pip`/`pip3` writes non-protected packages to an atomic project generation while verified Torch remains external and immutable | Existing verified parent and project templates |
| 2 | [Stable release image delivery](./2026-07-10-stable-release-image-delivery.md) | Strict release schema, exact-digest pull/verification, publication manifest generator, and anonymous-pull gate | Existing image build and qualification reports |
| 3 | [Managed runtime, doctor, and repair](./2026-07-10-managed-runtime-doctor-repair.md) | Read-only managed containers, effective Torch checks, classified diagnostics, quarantine, and exact-ID rebuild repair | Plans 1-2 |
| 4 | [Interactive one-click installer](./2026-07-10-interactive-one-click-installer.md) | Resumable full/container workflows, local bootstrap, unified CLI, project setup, and operator documentation | Plans 1-3 |

The public stable manifest is generated only after all four plans are merged into the feature branch, the final image revision is rebuilt, and the gfx1151 qualification gate passes. A digest from an older image or a locally guessed digest is never committed as stable.

## Locked Cross-Plan Interfaces

### Command surface

```text
./install.sh
bin/strix-halo-rocm

strix-halo-rocm install
strix-halo-rocm doctor [PROJECT] [--json PATH]
strix-halo-rocm repair PROJECT [--yes]
strix-halo-rocm project init NAME [existing project-init options]
strix-halo-rocm project lock PROJECT
strix-halo-rocm project run PROJECT [existing project-run options]
strix-halo-rocm release verify [--manifest PATH]
strix-halo-rocm release publish --release-id VERSION --qualification REPORT --sbom SPDX
```

The existing `bin/host-*`, `bin/image-build`, `bin/container-check`, and `bin/project-*` wrappers remain supported.

### Protected stack

```python
PROTECTED_DISTRIBUTIONS = frozenset(
    {"torch", "torchvision", "torchaudio", "triton"}
)
BASE_SITE_PACKAGES = Path("/opt/venv/lib/python3.12/site-packages")
OVERLAY_ROOT = Path("/workspace/.amd-ai")
OVERLAY_SITE_PACKAGES = Path(
    "/workspace/.amd-ai/current/site-packages"
)
```

The exact full versions come from `/opt/amd-ai/torch-manifest.json`; the public versions in `profile.env` are insufficient. `pip install torch` succeeds only by reporting the verified parent as already satisfied. Any resolver report or completed generation containing a protected distribution is blocked.

### Overlay generation contract

Every generation contains the replay material as well as installed files:

```text
.amd-ai/generations/<transaction-id>/
  overlay.requirements.in
  overlay.requirements.lock
  overlay-state.json
  site-packages/
```

`current` is a relative symlink to `generations/<transaction-id>`. The root-level input, lock, and state files are atomic mirrors for operator visibility; the generation files are the recovery source of truth. A transaction builds from an empty target, validates it, switches `current` with `os.replace`, then updates the mirrors. A crash before the switch leaves the old environment active; a crash after the switch is recoverable from the generation metadata.

### Release API

```python
@dataclass(frozen=True)
class ReleaseImage:
    image: str
    manifest_digest: str
    config_digest: str
    artifact_digests: Mapping[str, str]

    @property
    def reference(self) -> str:
        return f"{self.image}@{self.manifest_digest}"


@dataclass(frozen=True)
class StableRelease:
    schema_version: int
    release_id: str
    source_repository: str
    source_revision: str
    qualification_profile_digest: str
    qualification_report_digest: str
    sbom_digest: str
    gpu_arch: str
    supported_host_adapter_ids: tuple[str, ...]
    rocm_version: str
    python_version: str
    torch_version: str
    torch_profile_id: str
    torch_profile_digest: str
    base: ReleaseImage
    torch: ReleaseImage
    published_at: str
```

`manifest_digest` is an OCI registry manifest digest. `config_digest` is the Docker image/config ID. Code and tests must not treat them as interchangeable.

### Diagnostic and repair API

```python
class DiagnosticDisposition(StrEnum):
    PASS = "pass"
    WARNING = "warning"
    REPAIRABLE = "repairable"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class Diagnostic:
    code: str
    disposition: DiagnosticDisposition
    summary: str
    evidence: str
    remediation: str


@dataclass(frozen=True)
class RepairAction:
    kind: str
    exact_target: str
    reason_code: str
```

Repair plans may name only a generation path inside one project, an exact local `sha256:` image ID, or an exact `image@sha256:` registry reference. Wildcards, broad prune commands, and mutable tags are rejected.

### Installer state

The canonical state path is:

```text
~/.local/state/strix-halo-rocm-toolkit/install-state.json
```

The stage order is locked to the approved specification. Each completed stage stores a SHA-256 digest of its canonical JSON inputs. Resumption recomputes that digest and blocks on mismatch. The state file contains no tokens, package-index credentials, full environment dump, or Docker auth data.

`source_revision` records the stable manifest's qualified image source. `installer_source_revision` records the local bootstrap/runtime checkout; these may differ by the final manifest-only commit and must never be compared as if they were the same identity.

## Execution Sequence

- [ ] **Step 1: Verify the approved specification**

Run:

```bash
sha256sum docs/superpowers/specs/2026-07-10-interactive-installer-torch-guard-design.md
```

Expected: `1d801ba0237788a42644b456a7c6b80ccb46dc89bcc281c098adbdf9bc949c1f`. If the digest differs, review the specification diff and update all affected child plans before implementation.

- [ ] **Step 2: Execute the protected Python overlay plan**

Follow every checkbox in `docs/superpowers/plans/2026-07-10-protected-python-overlay.md`.

Expected evidence:

```text
uv run pytest tests/unit/overlay tests/cli/test_overlay_commands.py -q
all tests passed
```

- [ ] **Step 3: Execute the stable release image delivery plan**

Follow every checkbox in `docs/superpowers/plans/2026-07-10-stable-release-image-delivery.md` through its local and fixture-based verification tasks. Defer its final registry publication task until Step 7 below.

Expected evidence:

```text
uv run pytest tests/unit/installer/test_release.py tests/cli/test_release_commands.py -q
all tests passed
```

- [ ] **Step 4: Execute the managed runtime, doctor, and repair plan**

Follow every checkbox in `docs/superpowers/plans/2026-07-10-managed-runtime-doctor-repair.md`.

Expected evidence:

```text
uv run pytest tests/unit/doctor tests/unit/overlay tests/unit/project \
  tests/cli/test_doctor_commands.py -q
all tests passed
```

- [ ] **Step 5: Execute the interactive installer plan**

Follow every checkbox in `docs/superpowers/plans/2026-07-10-interactive-one-click-installer.md`.

Expected evidence:

```text
uv run pytest tests/unit/installer tests/cli/test_installer_commands.py -q
all tests passed
```

- [ ] **Step 6: Run all non-hardware gates before rebuilding images**

Run:

```bash
uv run pytest -m 'not hardware' -q
git diff --check
git status --short
```

Expected: zero failures, no whitespace errors, and only intentional implementation files are listed. Commit any remaining intentional documentation or fixture changes before building so the image revision is a clean commit.

- [ ] **Step 7: Rebuild the exact release revision**

Run serially:

```bash
test -z "$(git status --porcelain)"
bin/image-build rocm-python
bin/image-build rocm-pytorch --profile profiles/torch/stable.env
bin/container-check --mode torch --metadata-only \
  --image rocm-pytorch:7.2.1-py3.12-torch2.9.1 \
  --json reports/release-candidate-metadata.json
```

Expected: both builds exit 0, image labels contain the current 40-character `git rev-parse HEAD`, the embedded protected-pip files are present, and the metadata report has status `pass`.

- [ ] **Step 8: Re-run target hardware qualification**

Run on the Ryzen AI Max+ 395 host:

```bash
bin/container-check --suite stable \
  --profile profiles/qualification/stable.toml \
  --json reports/qualification.json
bin/gpu-release \
  --qualification reports/qualification.json \
  --image rocm-pytorch:7.2.1-py3.12-torch2.9.1 \
  --output-dir reports/releases
```

Expected: all eight qualification checks pass, `gpu_arch` is `gfx1151`, the stress duration is at least 300 seconds, and the release report binds the current source revision and current image ID.

- [ ] **Step 9: Push exact images and make both GHCR packages public**

Set a release ID from the project version and run the implemented publisher:

```bash
RELEASE_ID="$(uv run python -c 'import amd_ai; print(amd_ai.__version__)')"
RELEASE_REPORT="$(ls -1t reports/releases/*-gfx1151.json | head -1)"
SBOM_REPORT="${RELEASE_REPORT%.json}.spdx.json"
bin/strix-halo-rocm release publish \
  --release-id "${RELEASE_ID}" \
  --qualification "${RELEASE_REPORT}" \
  --sbom "${SBOM_REPORT}" \
  --output profiles/releases/stable.json \
  --push-only
```

Expected: the command pushes both package names, reads registry-assigned manifest digests, validates config IDs, labels and compressed layer limits, prints both exact references, does not write `stable.json`, and exits 0. In GitHub package settings, set `strix-halo-rocm-python` and `strix-halo-rocm-pytorch` visibility to public before continuing.

- [ ] **Step 10: Verify anonymous pulls and generate the stable manifest**

Run:

```bash
AUTHLESS_DOCKER_CONFIG="$(mktemp -d)"
TORCH_REFERENCE="$(uv run python -c 'from pathlib import Path; from amd_ai.image.publish import observe_pushed_release; print(observe_pushed_release(Path("reports/publish-candidate.json")).torch.reference)')"
DOCKER_CONFIG="${AUTHLESS_DOCKER_CONFIG}" \
  docker pull "${TORCH_REFERENCE}"
rm -rf "${AUTHLESS_DOCKER_CONFIG}"
bin/strix-halo-rocm release publish \
  --release-id "${RELEASE_ID}" \
  --qualification "${RELEASE_REPORT}" \
  --sbom "${SBOM_REPORT}" \
  --output profiles/releases/stable.json
```

Expected: the explicit pull exits 0 without credentials. The second publisher invocation anonymously pulls and verifies both exact references, then atomically writes a strict `stable.json`; any authorization or identity failure leaves the output absent or byte-for-byte unchanged.

- [ ] **Step 11: Run end-to-end managed project tests**

Run:

```bash
uv run pytest -m container tests/container/test_readonly_overlay.py \
  tests/container/test_repair_flow.py -q
./install.sh --mode container --non-interactive \
  --project-dir "$(mktemp -d)/installer-smoke" \
  --image-source pull
```

Expected: ordinary package installation persists across two containers, protected package attempts are rejected, shadow corruption blocks startup, repair restores the last lock, and the non-interactive installer completes from anonymous exact-digest pulls.

- [ ] **Step 12: Commit the verified release manifest and tag the release candidate**

```bash
git add profiles/releases/stable.json
git commit -m "release: publish verified Strix Halo image digests"
test -z "$(git status --porcelain)"
git tag -a "v$(uv run python -c 'import amd_ai; print(amd_ai.__version__)')-rc1" \
  -m "Interactive installer and protected overlay release candidate"
git show --no-patch --oneline --decorate HEAD
```

Expected: the manifest commit contains registry-observed digests and the annotated tag points to that commit. Pushing the branch, tags, or release is a separate user-authorized operation.

## Specification Coverage

| Approved requirement | Owning plan/task |
| --- | --- |
| Plain in-container pip with per-project persistence | Overlay Tasks 2-8 |
| No duplicate Torch in overlays | Overlay Tasks 1, 4, 6-8 |
| Exact full protected versions and transitive conflicts | Overlay Tasks 3-6 |
| Local source/wheel hashing and unsupported flag rejection | Overlay Tasks 2-5 |
| Atomic generation, lock, interruption safety | Overlay Tasks 6-8 |
| Read-only root, bounded tmpfs/shm, explicit writable mounts | Runtime Tasks 1-3 |
| Effective import identity and shadow detection | Runtime Tasks 4-5 |
| Diagnostic codes and secret-safe evidence | Runtime Tasks 6-7 |
| Quarantine and exact immutable rebuild repair | Runtime Tasks 8-10 |
| OCI manifest/config digest distinction | Release Tasks 1-4 |
| Anonymous GHCR pull and local build fallback | Release Tasks 5-8; Installer Tasks 7-8 |
| Two installer modes and resumable reboot boundary | Installer Tasks 1-7 |
| Non-interactive refusal of implicit approvals | Installer Tasks 3, 6-8 |
| Unified CLI and backward-compatible wrappers | Installer Tasks 8-9 |
| README, operator guides, pip matrix, doctor codes | Installer Tasks 10-11 |
| Final gfx1151 qualification and stable publication | Index Steps 6-12 |
