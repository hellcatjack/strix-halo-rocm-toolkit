# Stable Release Image Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make verified ROCm and PyTorch images discoverable, pullable without authentication, and deployable only through a strict stable manifest that binds registry manifest digests, Docker config IDs, source revision, qualification evidence, SBOM, labels, and embedded locks.

**Architecture:** A strict standard-library parser distinguishes OCI manifest identity from Docker image/config identity. Pull and verification always use `image@sha256:...`, inspect RepoDigests and labels, then hash embedded artifacts in disposable containers. Publication tags the already-qualified local images, pushes them to separate public GHCR packages, validates registry layer sizes and anonymous pulls, and only then atomically writes `profiles/releases/stable.json` from observed registry data.

**Tech Stack:** Python 3.12 standard library, Docker Engine/BuildKit, OCI image manifests, GHCR, pytest, existing qualification JSON/SPDX reports.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `src/amd_ai/installer/__init__.py` | Installer package marker |
| `src/amd_ai/installer/models.py` | `ReleaseImage` and `StableRelease` immutable records |
| `src/amd_ai/installer/release.py` | Strict manifest parser, exact pull, local identity and embedded-artifact verification |
| `src/amd_ai/image/publish.py` | GHCR tag/push, registry digest discovery, layer gate, anonymous pull, manifest generation |
| `tests/fixtures/releases/stable.json` | Deterministic valid release fixture with synthetic digests |
| `tests/unit/installer/test_release.py` | Manifest and exact-pull tests |
| `tests/unit/image/test_publish.py` | Publication, digest and anonymous-pull tests |
| `tests/cli/test_release_commands.py` | `release verify` and `release publish` dispatch |
| `profiles/releases/stable.json` | Generated only by the final qualified publication gate |

The parser accepts no mutable fallback tags. Publication may use a release tag for discovery, but deployment identity and the committed stable manifest use only the registry-observed manifest digest.

### Task 1: Define the Strict Stable Release Schema

**Files:**
- Create: `src/amd_ai/installer/__init__.py`
- Create: `src/amd_ai/installer/models.py`
- Create: `src/amd_ai/installer/release.py`
- Create: `tests/fixtures/releases/stable.json`
- Create: `tests/unit/installer/__init__.py`
- Create: `tests/unit/installer/test_release.py`

- [ ] **Step 1: Create the deterministic release fixture**

Write `tests/fixtures/releases/stable.json` with this complete shape; use 64 repeated hex characters for synthetic digests and a 40-character revision:

```json
{
  "base": {
    "artifact_digests": {
      "rocm_keyring": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "rocm_packages_lock": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    },
    "config_digest": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    "image": "ghcr.io/hellcatjack/strix-halo-rocm-python",
    "manifest_digest": "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
  },
  "gpu_arch": "gfx1151",
  "published_at": "2026-07-10T12:00:00Z",
  "python_version": "3.12",
  "qualification_profile_digest": "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
  "qualification_report_digest": "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
  "release_id": "0.2.0",
  "rocm_version": "7.2.1",
  "sbom_digest": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
  "schema_version": 1,
  "source_repository": "https://github.com/hellcatjack/strix-halo-rocm-toolkit",
  "source_revision": "2222222222222222222222222222222222222222",
  "supported_host_adapter_ids": ["ubuntu-24.04"],
  "torch": {
    "artifact_digests": {
      "profile": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
      "requirements_lock": "sha256:4444444444444444444444444444444444444444444444444444444444444444",
      "torch_manifest": "sha256:5555555555555555555555555555555555555555555555555555555555555555"
    },
    "config_digest": "sha256:6666666666666666666666666666666666666666666666666666666666666666",
    "image": "ghcr.io/hellcatjack/strix-halo-rocm-pytorch",
    "manifest_digest": "sha256:7777777777777777777777777777777777777777777777777777777777777777"
  },
  "torch_profile_digest": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
  "torch_profile_id": "rocm-7.2.1-py3.12-torch-2.9.1",
  "torch_version": "2.9.1"
}
```

- [ ] **Step 2: Write failing schema tests**

```python
import json
from pathlib import Path

import pytest

from amd_ai.installer.release import ReleaseError, load_stable_release


FIXTURE = Path("tests/fixtures/releases/stable.json")


def test_valid_release_distinguishes_manifest_and_config_digest():
    release = load_stable_release(FIXTURE)

    assert release.torch.reference.startswith(
        "ghcr.io/hellcatjack/strix-halo-rocm-pytorch@sha256:"
    )
    assert release.torch.manifest_digest != release.torch.config_digest
    assert release.supported_host_adapter_ids == ("ubuntu-24.04",)


@pytest.mark.parametrize("mutation", ["missing", "unknown", "bad-digest", "mutable-image"])
def test_release_schema_rejects_ambiguous_payload(tmp_path, mutation):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if mutation == "missing":
        del payload["source_revision"]
    elif mutation == "unknown":
        payload["latest"] = True
    elif mutation == "bad-digest":
        payload["torch"]["manifest_digest"] = payload["torch"]["config_digest"]
    else:
        payload["torch"]["image"] += ":latest"
    path = tmp_path / "release.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReleaseError):
        load_stable_release(path)
```

- [ ] **Step 3: Run tests and confirm missing implementation**

Run: `uv run pytest tests/unit/installer/test_release.py -q`

Expected: collection fails because `amd_ai.installer` does not exist.

- [ ] **Step 4: Implement release records and strict parser**

Define in `src/amd_ai/installer/models.py`:

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

`load_stable_release()` must reject duplicate JSON keys through an `object_pairs_hook`, unknown or missing keys at every level, booleans where integers are expected, non-semantic release IDs, non-UTC timestamps, any digest outside `sha256:<64 lowercase hex>`, equal manifest/config digests, duplicate adapters, any architecture other than `gfx1151`, any ROCm/Python/Torch version outside `7.2.1`/`3.12`/`2.9.1`, non-GHCR image names, image tags, image digests embedded in the `image` field, and a torch profile digest unequal to `torch.artifact_digests.profile`.

- [ ] **Step 5: Run schema tests**

Run: `uv run pytest tests/unit/installer/test_release.py -q`

Expected: all schema tests pass.

- [ ] **Step 6: Commit the release schema**

```bash
git add src/amd_ai/installer tests/fixtures/releases tests/unit/installer
git commit -m "feat: define strict stable release manifest"
```

### Task 2: Verify Exact Local Image Identity and Embedded Artifacts

**Files:**
- Modify: `src/amd_ai/installer/release.py`
- Modify: `tests/unit/installer/test_release.py`
- Create: `tests/unit/installer/fakes.py`

- [ ] **Step 1: Write failing image verification tests**

Create a fake Docker interface that records `pull`, `inspect`, and `run_sha256` calls. Add:

```python
def test_verify_release_image_requires_repo_digest_config_labels_and_artifacts(release):
    docker = FakeReleaseDocker.for_release(release)

    identity = verify_release_image(release, release.torch, kind="torch", docker=docker)

    assert identity.config_digest == release.torch.config_digest
    assert identity.repo_digests == (release.torch.reference,)
    assert docker.hash_calls == [
        (release.torch.reference, "/opt/amd-ai/profile.env"),
        (release.torch.reference, "/opt/amd-ai/profile.requirements.lock"),
        (release.torch.reference, "/opt/amd-ai/torch-manifest.json"),
    ]


def test_verify_release_image_rejects_friendly_tag_drift(release):
    docker = FakeReleaseDocker.for_release(release)
    docker.records[release.torch.reference]["RepoDigests"] = [
        release.torch.image + "@sha256:" + "9" * 64
    ]

    with pytest.raises(ReleaseError, match="RepoDigest"):
        verify_release_image(release, release.torch, kind="torch", docker=docker)
```

- [ ] **Step 2: Run focused tests and confirm missing verifier failure**

Run: `uv run pytest tests/unit/installer/test_release.py -q`

Expected: tests fail because `verify_release_image` is absent.

- [ ] **Step 3: Implement the Docker protocol and identity verifier**

Define:

```python
class ReleaseDocker(Protocol):
    def pull(self, reference: str) -> None:
        pass

    def inspect(self, reference: str) -> Mapping[str, object]:
        pass

    def hash_file(self, reference: str, path: str) -> str:
        pass


@dataclass(frozen=True)
class VerifiedImageIdentity:
    reference: str
    config_digest: str
    repo_digests: tuple[str, ...]
    labels: Mapping[str, str]
```

`verify_release_image()` validates exact config digest, inclusion of the exact RepoDigest, and labels:

```python
expected_labels = {
    "org.opencontainers.image.source": release.source_repository,
    "org.opencontainers.image.revision": release.source_revision,
    "org.amd-ai.rocm.version": release.rocm_version,
    "org.amd-ai.python.version": release.python_version,
}
if kind == "torch":
    expected_labels.update(
        {
            "org.amd-ai.profile.id": release.torch_profile_id,
            "org.amd-ai.profile.status": "verified",
            "org.amd-ai.torch.version": release.torch_version,
        }
    )
```

Hash these exact paths and compare `sha256:<hex>` values:

```python
BASE_ARTIFACT_PATHS = {
    "rocm_keyring": "/etc/apt/keyrings/rocm.gpg",
    "rocm_packages_lock": "/opt/amd-ai/locks/rocm-packages.lock",
}
TORCH_ARTIFACT_PATHS = {
    "profile": "/opt/amd-ai/profile.env",
    "requirements_lock": "/opt/amd-ai/profile.requirements.lock",
    "torch_manifest": "/opt/amd-ai/torch-manifest.json",
}
```

- [ ] **Step 4: Run verifier tests**

Run: `uv run pytest tests/unit/installer/test_release.py -q`

Expected: config, RepoDigest, label, embedded hash, and image-kind mismatch tests pass.

- [ ] **Step 5: Commit image identity verification**

```bash
git add src/amd_ai/installer/release.py tests/unit/installer
git commit -m "feat: verify stable image identity chain"
```

### Task 3: Pull Both Images by Exact Digest

**Files:**
- Modify: `src/amd_ai/installer/release.py`
- Modify: `tests/unit/installer/test_release.py`

- [ ] **Step 1: Write failing exact-pull tests**

```python
def test_pull_release_uses_only_manifest_digest_references(release):
    docker = FakeReleaseDocker.for_release(release)

    result = pull_and_verify_release(release, docker=docker)

    assert docker.pull_calls == [release.base.reference, release.torch.reference]
    assert result.base.config_digest == release.base.config_digest
    assert result.torch.config_digest == release.torch.config_digest


def test_pull_failure_does_not_fall_back_to_mutable_tag(release):
    docker = FakeReleaseDocker.for_release(release)
    docker.pull_error = ReleaseError("network stopped")

    with pytest.raises(ReleaseError, match="network stopped"):
        pull_and_verify_release(release, docker=docker)

    assert all("@sha256:" in value for value in docker.pull_calls)
```

- [ ] **Step 2: Confirm the new tests fail**

Run: `uv run pytest tests/unit/installer/test_release.py -q`

Expected: failure because `pull_and_verify_release` is absent.

- [ ] **Step 3: Implement exact pull sequencing**

Define `VerifiedReleaseImages(base, torch)` and implement `pull_and_verify_release()`. Pull base first, verify it completely, then pull and verify torch. On any failure, do not retag, remove, rebuild, or pull another reference. Return identities only after both pass. A retry calls the same exact references and relies on Docker's content-addressed partial-layer cache.

- [ ] **Step 4: Run release tests**

Run: `uv run pytest tests/unit/installer/test_release.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit exact pull support**

```bash
git add src/amd_ai/installer/release.py tests/unit/installer/test_release.py
git commit -m "feat: pull verified images by exact digest"
```

### Task 4: Bind Normal Image Builds to the Public Source Repository

**Files:**
- Modify: `src/amd_ai/image/build.py`
- Modify: `tests/unit/image/test_build.py`
- Modify: `tests/cli/test_image_commands.py`

- [ ] **Step 1: Write failing source-label tests**

Add:

```python
def test_normal_builds_use_public_source_repository():
    assert IMAGE_SOURCE == (
        "https://github.com/hellcatjack/strix-halo-rocm-toolkit"
    )

    argv = build_rocm_python_argv(
        ubuntu_base="ubuntu@sha256:" + "a" * 64,
        uv_image="ghcr.io/astral-sh/uv@sha256:" + "b" * 64,
        revision="c" * 40,
    )

    assert f"IMAGE_SOURCE={IMAGE_SOURCE}" in argv
```

- [ ] **Step 2: Run focused image build tests**

Run: `uv run pytest tests/unit/image/test_build.py tests/cli/test_image_commands.py -q`

Expected: source assertions fail because normal builds currently label images as `local`.

- [ ] **Step 3: Make source identity a repository constant**

Add:

```python
IMAGE_SOURCE = "https://github.com/hellcatjack/strix-halo-rocm-toolkit"
```

Use it as the default `image_source` in both argv builders and pass it from `build_rocm_python()` and `build_rocm_pytorch()`. Keep explicit injection in unit tests, but remove production `"local"` values. The build revision remains the exact clean `git rev-parse HEAD`; existing dirty-worktree rejection remains unchanged.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/unit/image/test_build.py tests/cli/test_image_commands.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit source label enforcement**

```bash
git add src/amd_ai/image/build.py tests/unit/image/test_build.py \
  tests/cli/test_image_commands.py
git commit -m "fix: bind image labels to public source"
```

### Task 5: Generate a Candidate Manifest from Qualified Local Evidence

**Files:**
- Create: `src/amd_ai/image/publish.py`
- Create: `tests/unit/image/test_publish.py`

- [ ] **Step 1: Write failing evidence-validation tests**

```python
def test_candidate_requires_matching_revision_image_and_evidence(tmp_path):
    qualification = write_qualification_report(
        tmp_path,
        revision="a" * 40,
        image_id="sha256:" + "b" * 64,
        gpu_arch="gfx1151",
        status="verified",
    )
    sbom = tmp_path / "release.spdx.json"
    sbom.write_text('{"spdxVersion":"SPDX-2.3"}\n', encoding="utf-8")

    candidate = validate_publish_inputs(
        release_id="0.2.0",
        qualification_path=qualification,
        sbom_path=sbom,
        current_revision="a" * 40,
        torch_image_id="sha256:" + "b" * 64,
    )

    assert candidate.gpu_arch == "gfx1151"
    assert candidate.qualification_digest.startswith("sha256:")
    assert candidate.sbom_digest.startswith("sha256:")
```

Add negative cases for stale revision, local image ID mismatch, non-verified status, wrong architecture, missing eight required checks, qualification digest mismatch, and SBOM digest mismatch.

- [ ] **Step 2: Verify publish tests fail**

Run: `uv run pytest tests/unit/image/test_publish.py -q`

Expected: collection fails because `amd_ai.image.publish` is missing.

- [ ] **Step 3: Implement candidate evidence validation**

Parse the existing release report strictly. Require:

```python
REQUIRED_QUALIFICATION_CHECKS = frozenset(
    {
        "rocm",
        "torch-fp16",
        "hip",
        "torch-extension",
        "triton",
        "repeated-start",
        "stress",
        "kernel-log",
    }
)
```

Re-hash the qualification JSON, SPDX JSON, `profiles/qualification/stable.toml`, `profiles/torch/stable.env`, base ROCm lock, and embedded artifact source files. Require clean current revision equality, exact local image ID equality, verified profile, `gfx1151`, and source-repository labels before constructing `PublishCandidate`.

- [ ] **Step 4: Run candidate tests**

Run: `uv run pytest tests/unit/image/test_publish.py -q`

Expected: all candidate tests pass.

- [ ] **Step 5: Commit candidate validation**

```bash
git add src/amd_ai/image/publish.py tests/unit/image/test_publish.py
git commit -m "feat: validate release publication evidence"
```

### Task 6: Push GHCR Images and Discover Registry Identity

**Files:**
- Modify: `src/amd_ai/image/publish.py`
- Modify: `tests/unit/image/test_publish.py`

- [ ] **Step 1: Write failing publication sequence tests**

```python
def test_publish_tags_pushes_and_observes_each_registry_digest(candidate, tmp_path):
    registry = FakeRegistry.for_candidate(candidate)

    observed = publish_images(candidate, registry=registry)

    assert registry.calls == [
        ("tag", candidate.base_local_id, "ghcr.io/hellcatjack/strix-halo-rocm-python:0.2.0"),
        ("push", "ghcr.io/hellcatjack/strix-halo-rocm-python:0.2.0"),
        ("observe", "ghcr.io/hellcatjack/strix-halo-rocm-python:0.2.0"),
        ("tag", candidate.torch_local_id, "ghcr.io/hellcatjack/strix-halo-rocm-pytorch:0.2.0"),
        ("push", "ghcr.io/hellcatjack/strix-halo-rocm-pytorch:0.2.0"),
        ("observe", "ghcr.io/hellcatjack/strix-halo-rocm-pytorch:0.2.0"),
    ]
    assert observed.torch.manifest_digest.startswith("sha256:")

    report = tmp_path / "publish-candidate.json"
    write_observed_release(report, observed)
    assert observe_pushed_release(report) == observed
```

- [ ] **Step 2: Run focused publication tests**

Run: `uv run pytest tests/unit/image/test_publish.py -q`

Expected: failure because `publish_images` is absent.

- [ ] **Step 3: Implement exact tag, push, and digest observation**

Use separate package names:

```python
BASE_PACKAGE = "ghcr.io/hellcatjack/strix-halo-rocm-python"
TORCH_PACKAGE = "ghcr.io/hellcatjack/strix-halo-rocm-pytorch"
MAX_GHCR_LAYER_BYTES = 10_000_000_000
```

Validate release IDs before using them as tags. Tag only the exact local image IDs validated in Task 5, push base then torch, pull each release tag, and derive the matching `package@sha256:` entry from Docker's RepoDigests. Inspect config ID and require it equals the local ID. Never parse a digest from mutable tag text or synthesize it by hashing command output.

Atomically serialize the observed release IDs, exact references, config IDs, artifact digests, source revision, qualification/SBOM digests, and release ID to `reports/publish-candidate.json`. Implement `observe_pushed_release(path)` as a strict loader so a later anonymous-pull invocation cannot substitute another tag or digest. This report contains no credentials and is not the stable manifest.

Read the raw registry manifest through `docker buildx imagetools inspect --raw <exact-reference>`, parse its descriptor sizes, and reject any layer above `MAX_GHCR_LAYER_BYTES`. Reject manifest lists containing a platform other than `linux/amd64` or more than one platform.

- [ ] **Step 4: Run publication tests**

Run: `uv run pytest tests/unit/image/test_publish.py -q`

Expected: push failure, missing RepoDigest, config mismatch, oversized layer, wrong platform, and happy-path tests pass.

- [ ] **Step 5: Commit registry publication**

```bash
git add src/amd_ai/image/publish.py tests/unit/image/test_publish.py
git commit -m "feat: publish exact GHCR image identities"
```

### Task 7: Gate Stable Manifest Writing on Anonymous Pull

**Files:**
- Modify: `src/amd_ai/image/publish.py`
- Modify: `tests/unit/image/test_publish.py`

- [ ] **Step 1: Write failing anonymous-pull and atomic-write tests**

```python
def test_manifest_is_written_only_after_two_authless_pulls(candidate, tmp_path):
    registry = FakeRegistry.for_candidate(candidate)
    output = tmp_path / "stable.json"

    release = publish_stable_release(candidate, registry=registry, output=output)

    assert registry.authless_pull_calls == [release.base.reference, release.torch.reference]
    assert output.is_file()
    assert load_stable_release(output) == release


def test_failed_authless_pull_leaves_existing_manifest_unchanged(candidate, tmp_path):
    output = tmp_path / "stable.json"
    output.write_text("known-good\n", encoding="utf-8")
    registry = FakeRegistry.for_candidate(candidate)
    registry.authless_error = PublishError("denied")

    with pytest.raises(PublishError, match="denied"):
        publish_stable_release(candidate, registry=registry, output=output)

    assert output.read_text(encoding="utf-8") == "known-good\n"
```

- [ ] **Step 2: Confirm tests fail for missing final publisher**

Run: `uv run pytest tests/unit/image/test_publish.py -q`

Expected: failure because `publish_stable_release` is absent.

- [ ] **Step 3: Implement clean-auth anonymous pull and atomic manifest write**

For each exact reference, create a temporary empty Docker config directory, set `DOCKER_CONFIG` only for the pull subprocess, and run `docker pull image@sha256:...`. Do not copy host credentials, credential helpers, or config files into that directory. Remove it in `finally`.

After both pulls and `verify_release_image()` pass, serialize `StableRelease` with sorted keys and indentation, write a mode-`0644` temporary file beside the public output, `flush`, `fsync`, `os.replace`, and fsync the parent directory. A failure before `os.replace` must leave any prior stable manifest byte-for-byte unchanged.

- [ ] **Step 4: Run publisher tests**

Run: `uv run pytest tests/unit/image/test_publish.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit stable publication gate**

```bash
git add src/amd_ai/image/publish.py tests/unit/image/test_publish.py
git commit -m "feat: require anonymous pull before stable manifest"
```

### Task 8: Add Release CLI Commands

**Files:**
- Modify: `src/amd_ai/cli.py`
- Create: `tests/cli/test_release_commands.py`

- [ ] **Step 1: Write failing CLI dispatch tests**

```python
def test_release_verify_loads_and_checks_manifest(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "verify_release_command", lambda **kw: captured.update(kw) or 0)

    code = cli.main(
        ["release", "verify", "--manifest", "tests/fixtures/releases/stable.json"]
    )

    assert code == 0
    assert captured["manifest_path"] == Path("tests/fixtures/releases/stable.json")


def test_release_publish_requires_all_evidence_paths():
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args(["release", "publish", "--release-id", "0.2.0"])

    assert error.value.code == 2
```

- [ ] **Step 2: Run CLI tests and confirm parser failure**

Run: `uv run pytest tests/cli/test_release_commands.py -q`

Expected: parser rejects the unknown `release` command.

- [ ] **Step 3: Add release command parsing and thin dispatch**

Add subcommands:

```text
release verify --manifest PATH
release publish --release-id VERSION --qualification PATH --sbom PATH --output PATH [--dry-run | --push-only] [--publish-report PATH]
```

Defaults are `profiles/releases/stable.json` for verify/output and `reports/publish-candidate.json` for the publication observation report. `--dry-run` validates local evidence without tags, pushes, authless pulls, or output writes. `--push-only` validates, tags, pushes, observes and writes only the publication report; it does not perform anonymous pulls or write the stable manifest. With neither flag, publication requires the observation to match current evidence, performs both anonymous pulls, and writes stable output. The CLI catches `ReleaseError` and `PublishError`, prints one redacted diagnostic to stderr, and returns 2. It contains no Docker verification or publication policy itself.

- [ ] **Step 4: Run release CLI and unit tests**

Run:

```bash
uv run pytest tests/unit/installer/test_release.py \
  tests/unit/image/test_publish.py tests/cli/test_release_commands.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit release commands**

```bash
git add src/amd_ai/cli.py tests/cli/test_release_commands.py
git commit -m "feat: expose stable release commands"
```

### Task 9: Verify Release Tooling Without Publishing

**Files:**
- Modify only files implicated by a failing check

- [ ] **Step 1: Run all release and image tests**

Run:

```bash
uv run pytest tests/unit/installer/test_release.py tests/unit/image \
  tests/cli/test_image_commands.py tests/cli/test_release_commands.py -q
```

Expected: zero failures.

- [ ] **Step 2: Verify the synthetic manifest through the real CLI**

Run:

```bash
uv run python -m amd_ai.cli release verify \
  --manifest tests/fixtures/releases/stable.json
```

Expected: schema parsing succeeds and image verification reports the synthetic images as missing with exit 2; absence of local synthetic images is the correct behavior.

- [ ] **Step 3: Exercise publication validation against current evidence in dry-run mode**

Run the no-push evidence gate:

```bash
RELEASE_REPORT="$(ls -1t reports/releases/*-gfx1151.json | head -1)"
SBOM_REPORT="${RELEASE_REPORT%.json}.spdx.json"
uv run python -m amd_ai.cli release publish \
  --release-id 0.2.0 \
  --qualification "${RELEASE_REPORT}" \
  --sbom "${SBOM_REPORT}" \
  --output /tmp/strix-halo-stable.json \
  --dry-run
```

Expected before the final rebuild: exit 2 with precise evidence that the report revision/image labels are stale or source is `local`. Dry-run must not push, retag, or write the output.

- [ ] **Step 4: Run the non-hardware regression suite**

Run: `uv run pytest -m 'not hardware' -q`

Expected: zero failures.

- [ ] **Step 5: Record a clean release-tooling checkpoint**

```bash
git status --short
git log --oneline -10
```

Expected: clean worktree. Actual GHCR publication remains deferred to the index plan after all image-affecting changes and hardware qualification.
