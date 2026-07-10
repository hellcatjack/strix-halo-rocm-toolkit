from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from amd_ai.image.build import BuildError, Docker
from amd_ai.image.profile import ProfileError, load_profile
from amd_ai.project.config import IMAGE_ID_PATTERN
from amd_ai.qualification.models import REQUIRED_CHECKS


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
QUALIFICATION_PROFILE = REPOSITORY_ROOT / "profiles/qualification/stable.toml"
TORCH_PROFILE = REPOSITORY_ROOT / "profiles/torch/stable.env"
DESIGN_DOCUMENT = (
    REPOSITORY_ROOT
    / "docs/superpowers/specs/2026-07-09-amd-ryzen-ai-max-395-rocm-container-design.md"
)
ROCM_PACKAGE_LOCK = REPOSITORY_ROOT / "profiles/rocm/7.2.1-packages.lock"
SBOM_TOOL = REPOSITORY_ROOT / "tools/generate-sbom.py"
VERIFIED_TAG = "rocm-pytorch:7.2.1-py3.12-torch2.9.1-gfx1151-verified"
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_REVISION_PATTERN = re.compile(r"[0-9a-f]{40,64}")


class ReleaseBlocked(RuntimeError):
    pass


@dataclass(frozen=True)
class ReleaseInputs:
    qualification: Mapping[str, object]
    qualification_digest: str
    profile_digest: str
    design_digest: str
    image_reference: str
    image_id: str
    repo_digest: str | None
    image_labels: Mapping[str, str]
    wheel_hashes: Mapping[str, str]
    rocm_package_lock_digest: str
    embedded_rocm_package_lock_digest: str
    torch_profile_digest: str
    embedded_torch_profile_digest: str
    sbom_digest: str
    git_revision: str
    git_clean: bool
    generated_at: str


@dataclass(frozen=True)
class ReleaseRecord:
    status: str
    generated_at: str
    profile_id: str
    gpu_arch: str
    qualification_digest: str
    profile_digest: str
    design_digest: str
    image_reference: str
    image_id: str
    repo_digest: str | None
    image_labels: Mapping[str, str]
    wheel_hashes: Mapping[str, str]
    rocm_package_lock_digest: str
    torch_profile_digest: str
    sbom_digest: str
    git_revision: str
    verified_tag: str
    qualification_file: str | None = None
    sbom_file: str | None = None
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "generated_at": self.generated_at,
            "profile_id": self.profile_id,
            "gpu_arch": self.gpu_arch,
            "qualification_digest": self.qualification_digest,
            "profile_digest": self.profile_digest,
            "design_digest": self.design_digest,
            "image_reference": self.image_reference,
            "image_id": self.image_id,
            "repo_digest": self.repo_digest,
            "image_labels": dict(self.image_labels),
            "wheel_hashes": dict(self.wheel_hashes),
            "rocm_package_lock_digest": self.rocm_package_lock_digest,
            "torch_profile_digest": self.torch_profile_digest,
            "sbom_digest": self.sbom_digest,
            "git_revision": self.git_revision,
            "verified_tag": self.verified_tag,
            "qualification_file": self.qualification_file,
            "sbom_file": self.sbom_file,
        }


def verify_release(inputs: ReleaseInputs) -> ReleaseRecord:
    qualification = inputs.qualification
    required = qualification.get("required_checks")
    if not isinstance(required, list) or tuple(required) != REQUIRED_CHECKS:
        raise ReleaseBlocked("qualification required-check set is not the stable profile")
    raw_results = qualification.get("results")
    if not isinstance(raw_results, list):
        raise ReleaseBlocked("qualification results are missing")
    names = [
        result.get("name")
        for result in raw_results
        if isinstance(result, dict)
    ]
    counts = Counter(names)
    by_name = {
        str(result.get("name")): result
        for result in raw_results
        if isinstance(result, dict) and isinstance(result.get("name"), str)
    }
    for name in REQUIRED_CHECKS:
        if counts[name] != 1 or by_name.get(name, {}).get("passed") is not True:
            raise ReleaseBlocked(f"required qualification check failed: {name}")
    if qualification.get("status") != "pass":
        raise ReleaseBlocked("qualification report status is not pass")
    if qualification.get("profile_id") != "stable-gfx1151":
        raise ReleaseBlocked("qualification profile ID is not stable-gfx1151")
    if qualification.get("profile_digest") != inputs.profile_digest:
        raise ReleaseBlocked("qualification profile digest does not match")
    if qualification.get("image") != inputs.image_reference:
        raise ReleaseBlocked("qualification image reference does not match")
    if qualification.get("image_id") != inputs.image_id:
        raise ReleaseBlocked("qualification image ID does not match release image ID")
    if qualification.get("gpu_arch") != "gfx1151":
        raise ReleaseBlocked("qualification did not record gfx1151")

    for label, expected in {
        "org.amd-ai.profile.id": "rocm-7.2.1-py3.12-torch-2.9.1",
        "org.amd-ai.profile.status": "verified",
        "org.amd-ai.rocm.version": "7.2.1",
        "org.amd-ai.torch.version": "2.9.1",
    }.items():
        if inputs.image_labels.get(label) != expected:
            raise ReleaseBlocked(
                f"image label {label} is not the required verified value"
            )
    if IMAGE_ID_PATTERN.fullmatch(inputs.image_id) is None:
        raise ReleaseBlocked("qualified image has no immutable local image ID")
    if inputs.repo_digest is not None and re.search(
        r"@sha256:[0-9a-f]{64}$", inputs.repo_digest
    ) is None:
        raise ReleaseBlocked("registry repository digest is invalid")

    expected_wheels = {"torch", "torchvision", "torchaudio", "triton"}
    if set(inputs.wheel_hashes) != expected_wheels:
        raise ReleaseBlocked("exactly four primary wheel hashes are required")
    for name, digest in inputs.wheel_hashes.items():
        _require_digest(digest, f"{name} wheel")
    _require_digest(inputs.qualification_digest, "qualification report")
    _require_digest(inputs.profile_digest, "qualification profile")
    _require_digest(inputs.design_digest, "approved design")
    _require_digest(inputs.rocm_package_lock_digest, "ROCm package lock")
    _require_digest(
        inputs.embedded_rocm_package_lock_digest,
        "embedded ROCm package lock",
    )
    _require_digest(inputs.torch_profile_digest, "Torch profile")
    _require_digest(inputs.embedded_torch_profile_digest, "embedded Torch profile")
    if (
        inputs.embedded_rocm_package_lock_digest
        != inputs.rocm_package_lock_digest
    ):
        raise ReleaseBlocked("embedded ROCm package lock does not match approved lock")
    if inputs.embedded_torch_profile_digest != inputs.torch_profile_digest:
        raise ReleaseBlocked("embedded Torch profile does not match approved profile")
    _require_digest(inputs.sbom_digest, "SPDX document")
    if GIT_REVISION_PATTERN.fullmatch(inputs.git_revision) is None:
        raise ReleaseBlocked("Git revision is invalid")
    if not inputs.git_clean:
        raise ReleaseBlocked("Git tracked files are modified")
    if not isinstance(inputs.generated_at, str) or not inputs.generated_at.endswith("Z"):
        raise ReleaseBlocked("release timestamp must be UTC")

    return ReleaseRecord(
        status="verified",
        generated_at=inputs.generated_at,
        profile_id="stable-gfx1151",
        gpu_arch="gfx1151",
        qualification_digest=inputs.qualification_digest,
        profile_digest=inputs.profile_digest,
        design_digest=inputs.design_digest,
        image_reference=inputs.image_reference,
        image_id=inputs.image_id,
        repo_digest=inputs.repo_digest,
        image_labels=MappingProxyType(dict(inputs.image_labels)),
        wheel_hashes=MappingProxyType(dict(inputs.wheel_hashes)),
        rocm_package_lock_digest=inputs.rocm_package_lock_digest,
        torch_profile_digest=inputs.torch_profile_digest,
        sbom_digest=inputs.sbom_digest,
        git_revision=inputs.git_revision,
        verified_tag=VERIFIED_TAG,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="amd-ai-release")
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/releases"))
    args = parser.parse_args(argv)
    try:
        release_path, sbom_path = create_release(
            qualification_path=args.qualification,
            image=args.image,
            output_dir=args.output_dir,
        )
    except (BuildError, OSError, ProfileError, ReleaseBlocked, ValueError) as error:
        print(f"amd-ai-release: {error}", file=sys.stderr)
        return 2
    print(f"release report: {release_path}")
    print(f"SPDX SBOM: {sbom_path}")
    return 0


def create_release(
    *,
    qualification_path: Path,
    image: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    qualification_bytes = qualification_path.read_bytes()
    qualification = json.loads(qualification_bytes)
    if not isinstance(qualification, dict):
        raise ReleaseBlocked("qualification report must be a JSON object")
    generated_at = qualification.get("generated_at")
    if not isinstance(generated_at, str):
        raise ReleaseBlocked("qualification report has no generation timestamp")

    docker = Docker.detect()
    image_id, repo_digest, labels = _inspect_image(docker, image)
    sbom_document, sbom_bytes = _generate_sbom(
        docker,
        runtime_image=image_id,
        name=image,
        created=generated_at,
    )
    if sbom_document.get("spdxVersion") != "SPDX-2.3":
        raise ReleaseBlocked("generated inventory is not SPDX 2.3")
    profile = load_profile(TORCH_PROFILE, allow_verified=True)
    wheel_hashes = {
        name: wheel.sha256 for name, wheel in profile.wheels.items()
    }
    git_revision, git_clean = _git_state()
    embedded_rocm_lock = _read_image_file(
        docker,
        image_id,
        "/opt/amd-ai/locks/rocm-packages.lock",
    )
    embedded_torch_profile = _read_image_file(
        docker,
        image_id,
        "/opt/amd-ai/profile.env",
    )
    inputs = ReleaseInputs(
        qualification=qualification,
        qualification_digest=hashlib.sha256(qualification_bytes).hexdigest(),
        profile_digest=_hash_file(QUALIFICATION_PROFILE),
        design_digest=_hash_file(DESIGN_DOCUMENT),
        image_reference=image,
        image_id=image_id,
        repo_digest=repo_digest,
        image_labels=labels,
        wheel_hashes=wheel_hashes,
        rocm_package_lock_digest=_hash_file(ROCM_PACKAGE_LOCK),
        embedded_rocm_package_lock_digest=hashlib.sha256(
            embedded_rocm_lock
        ).hexdigest(),
        torch_profile_digest=_hash_file(TORCH_PROFILE),
        embedded_torch_profile_digest=hashlib.sha256(
            embedded_torch_profile
        ).hexdigest(),
        sbom_digest=hashlib.sha256(sbom_bytes).hexdigest(),
        git_revision=git_revision,
        git_clean=git_clean,
        generated_at=generated_at,
    )
    record = verify_release(inputs)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _release_stem(generated_at)
    release_path = output_dir / f"{stem}-gfx1151.json"
    sbom_path = output_dir / f"{stem}-gfx1151.spdx.json"
    record = replace(
        record,
        qualification_file=str(qualification_path),
        sbom_file=sbom_path.name,
    )
    publish_release_artifacts(
        docker=docker,
        record=record,
        sbom_bytes=sbom_bytes,
        release_path=release_path,
        sbom_path=sbom_path,
    )
    return release_path, sbom_path


def publish_release_artifacts(
    *,
    docker: Docker,
    record: ReleaseRecord,
    sbom_bytes: bytes,
    release_path: Path,
    sbom_path: Path,
) -> None:
    if release_path.exists() or sbom_path.exists():
        raise ReleaseBlocked("release artifact path already exists")
    release_temporary = release_path.with_name(f".{release_path.name}.tmp")
    sbom_temporary = sbom_path.with_name(f".{sbom_path.name}.tmp")
    release_rendered = (
        json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    previous_tag_id: str | None = None
    tag_changed = False
    sbom_published = False
    try:
        sbom_temporary.write_bytes(sbom_bytes)
        release_temporary.write_text(release_rendered, encoding="utf-8")
        previous_tag_id = _tag_verified_image(
            docker,
            record.image_id,
            record.verified_tag,
        )
        tag_changed = previous_tag_id != record.image_id
        os.replace(sbom_temporary, sbom_path)
        sbom_published = True
        os.replace(release_temporary, release_path)
    except Exception as error:
        if tag_changed:
            try:
                _restore_verified_tag(
                    docker,
                    tag=record.verified_tag,
                    previous_image_id=previous_tag_id,
                )
            except ReleaseBlocked as rollback_error:
                raise ReleaseBlocked(
                    "release artifact failure also failed to restore verified tag"
                ) from rollback_error
        if sbom_published:
            sbom_path.unlink(missing_ok=True)
        release_path.unlink(missing_ok=True)
        raise error
    finally:
        sbom_temporary.unlink(missing_ok=True)
        release_temporary.unlink(missing_ok=True)


def _inspect_image(
    docker: Docker,
    image: str,
) -> tuple[str, str | None, Mapping[str, str]]:
    result = docker.capture(("image", "inspect", image), check=False)
    if result.returncode != 0:
        raise ReleaseBlocked(
            f"cannot inspect qualified image: {result.stderr.strip()}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ReleaseBlocked("cannot parse qualified image metadata") from error
    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], dict)
    ):
        raise ReleaseBlocked("qualified image metadata is unexpected")
    record = payload[0]
    image_id = record.get("Id")
    if not isinstance(image_id, str):
        raise ReleaseBlocked("qualified image has no local image ID")
    config = record.get("Config")
    if not isinstance(config, dict) or not isinstance(config.get("Labels"), dict):
        raise ReleaseBlocked("qualified image labels are missing")
    labels = {
        str(key): str(value)
        for key, value in config["Labels"].items()
    }
    raw_repo_digests = record.get("RepoDigests") or []
    repo_digests = sorted(
        value for value in raw_repo_digests if isinstance(value, str)
    )
    return image_id, repo_digests[0] if repo_digests else None, labels


def _generate_sbom(
    docker: Docker,
    *,
    runtime_image: str,
    name: str,
    created: str,
) -> tuple[dict[str, object], bytes]:
    mount = (
        f"type=bind,src={SBOM_TOOL},"
        "dst=/opt/amd-ai/generate-sbom.py,readonly"
    )
    result = docker.capture(
        (
            "run",
            "--rm",
            "--mount",
            mount,
            "--entrypoint",
            "/opt/venv/bin/python",
            runtime_image,
            "/opt/amd-ai/generate-sbom.py",
            "--name",
            name,
            "--created",
            created,
            "--output",
            "-",
        ),
        check=False,
    )
    if result.returncode != 0:
        evidence = result.stderr.strip() or result.stdout.strip()
        raise ReleaseBlocked(f"SPDX generation failed: {evidence}")
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ReleaseBlocked("generated SPDX output is invalid JSON") from error
    if not isinstance(document, dict):
        raise ReleaseBlocked("generated SPDX output is not an object")
    rendered = (
        json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    return document, rendered


def _read_image_file(docker: Docker, image_id: str, path: str) -> bytes:
    result = docker.capture(
        (
            "run",
            "--rm",
            "--entrypoint",
            "/bin/cat",
            image_id,
            path,
        ),
        check=False,
    )
    if result.returncode != 0:
        evidence = result.stderr.strip() or result.stdout.strip()
        raise ReleaseBlocked(f"cannot read embedded build input {path}: {evidence}")
    return result.stdout.encode("utf-8")


def _git_state() -> tuple[str, bool]:
    revision = _completed(("git", "rev-parse", "HEAD"))
    if revision.returncode != 0:
        raise ReleaseBlocked("cannot read Git revision")
    status = _completed(git_status_argv())
    if status.returncode != 0:
        raise ReleaseBlocked("cannot read Git tracked-file status")
    return revision.stdout.strip(), not status.stdout.strip()


def git_status_argv() -> tuple[str, ...]:
    return ("git", "status", "--porcelain", "--untracked-files=all")


def _tag_verified_image(docker: Docker, image_id: str, tag: str) -> str | None:
    previous_image_id = docker.image_id(tag, required=False)
    if previous_image_id == image_id:
        return previous_image_id
    tagged = docker.capture(("image", "tag", image_id, tag), check=False)
    if tagged.returncode != 0:
        raise ReleaseBlocked(
            f"cannot tag verified image: {tagged.stderr.strip()}"
        )
    resolved = docker.image_id(tag)
    if resolved != image_id:
        raise ReleaseBlocked(
            "verified tag did not resolve to the qualified image ID"
        )
    return previous_image_id


def _restore_verified_tag(
    docker: Docker,
    *,
    tag: str,
    previous_image_id: str | None,
) -> None:
    if previous_image_id is None:
        result = docker.capture(("image", "rm", tag), check=False)
    else:
        result = docker.capture(
            ("image", "tag", previous_image_id, tag),
            check=False,
        )
    if result.returncode != 0:
        raise ReleaseBlocked("cannot restore the previous verified tag")
    if docker.image_id(tag, required=False) != previous_image_id:
        raise ReleaseBlocked("verified tag rollback did not restore prior state")


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_digest(value: str, label: str) -> None:
    if DIGEST_PATTERN.fullmatch(value) is None:
        raise ReleaseBlocked(f"{label} SHA-256 digest is invalid")


def _release_stem(generated_at: str) -> str:
    stem = re.sub(r"[^0-9TZ]", "", generated_at)
    if not stem:
        raise ReleaseBlocked("release timestamp cannot form a filename")
    return stem


def _completed(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=False,
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


if __name__ == "__main__":
    raise SystemExit(main())
