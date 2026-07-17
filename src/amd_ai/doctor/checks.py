from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from amd_ai.doctor.models import (
    Diagnostic,
    DiagnosticDisposition,
    DoctorReport,
)
from amd_ai.image.build import Docker
from amd_ai.image.publish import AnonymousDockerRegistry, PublishError
from amd_ai.installer.models import (
    ReleaseImage,
    StableRelease,
    default_state_path,
)
from amd_ai.installer.registry import (
    RegistryCandidate,
    registry_candidates,
)
from amd_ai.installer.release import (
    ReleaseError,
    load_stable_release,
    verify_release_image,
)
from amd_ai.installer.state import (
    InstallerStateError,
    load_state_readonly,
    project_state_path,
)
from amd_ai.overlay.lock import lock_digest, parse_lock, validate_lock_artifacts
from amd_ai.overlay.models import OverlayPaths
from amd_ai.overlay.transaction import (
    TransactionError,
    load_generation_state,
    resolve_current_generation,
)
from amd_ai.overlay.verify import OverlayVerificationError, scan_protected_entries
from amd_ai.project.build import build_context_fingerprint
from amd_ai.project.config import ConfigError, ProjectConfig, load_project_config
from amd_ai.project.runtime import RuntimePolicyError, discover_gpu_access


BASE_FRIENDLY_TAG = "rocm-python:7.2.1-py3.12"
TORCH_FRIENDLY_TAG = "rocm-pytorch:7.2.1-py3.12-torch2.9.1"
DOCTOR_TMPFS = "/tmp:rw,nosuid,nodev,size=1g,mode=1777"


@dataclass(frozen=True)
class DoctorImageInspection:
    config_digest: str
    labels: Mapping[str, str]
    repo_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", MappingProxyType(dict(self.labels)))
        object.__setattr__(self, "repo_digests", tuple(self.repo_digests))


@dataclass(frozen=True)
class ProjectImageInspection:
    image_id: str
    fingerprint: str
    changed: bool
    base_manifest_error: str | None


class DoctorBackend(Protocol):
    def inspect_image(self, reference: str) -> DoctorImageInspection | None:
        pass

    def inspect_friendly(self, reference: str) -> str | None:
        pass

    def verify_image(
        self, release: StableRelease, image: ReleaseImage, kind: str
    ) -> str | None:
        pass

    def host_preflight(self) -> str | None:
        pass

    def gpu_runtime(self, reference: str) -> str | None:
        pass

    def inspect_project(self, config: ProjectConfig) -> ProjectImageInspection | None:
        pass

    def verify_effective_overlay(
        self, config: ProjectConfig, site_packages: Path
    ) -> str | None:
        pass


def doctor_platform(
    *,
    manifest_path: Path,
    backend: DoctorBackend,
    registry: str = "auto",
    preferred_references: Mapping[str, str] | None = None,
) -> DoctorReport:
    diagnostics: list[Diagnostic] = []
    facts: dict[str, object] = {"manifest": str(manifest_path)}
    try:
        release = load_stable_release(manifest_path)
    except (OSError, ReleaseError) as error:
        diagnostics.append(
            _diagnostic(
                "RELEASE.INVALID",
                DiagnosticDisposition.BLOCKED,
                "Stable release manifest is invalid",
                str(error),
                "Install or select a valid signed stable release manifest.",
            )
        )
        return DoctorReport.create(
            project=None,
            diagnostics=tuple(diagnostics),
            facts=facts,
            environment=os.environ,
        )

    facts["release_id"] = release.release_id
    candidates = registry_candidates(release, registry)
    selected_images: dict[str, tuple[StableRelease, ReleaseImage]] = {}
    torch_static_valid = True
    for kind in ("base", "torch"):
        kind_candidates = _prefer_registry_reference(
            candidates,
            kind=kind,
            preferred_reference=(
                None
                if preferred_references is None
                else preferred_references.get(kind)
            ),
        )
        selected_release = kind_candidates[0].release
        selected_image = getattr(selected_release, kind)
        inspection = None
        for candidate in kind_candidates:
            candidate_image = getattr(candidate.release, kind)
            observed = backend.inspect_image(candidate_image.reference)
            if observed is not None:
                selected_release = candidate.release
                selected_image = candidate_image
                inspection = observed
                break
        selected_images[kind] = (selected_release, selected_image)
        facts[f"{kind}_reference"] = selected_image.reference
        facts[f"{kind}_config_digest"] = selected_image.config_digest
        if inspection is None:
            diagnostics.append(
                _diagnostic(
                    "IMAGE.PARENT_MISSING",
                    DiagnosticDisposition.REPAIRABLE,
                    f"Exact {kind} parent image is missing",
                    selected_image.reference,
                    "Pull and verify the exact release digest.",
                )
            )
            if kind == "torch":
                torch_static_valid = False
            continue
        if (
            inspection.config_digest
            not in {
                selected_image.config_digest,
                selected_image.manifest_digest,
            }
            or selected_image.reference not in inspection.repo_digests
        ):
            diagnostics.append(
                _diagnostic(
                    "IMAGE.DIGEST_DRIFT",
                    DiagnosticDisposition.REPAIRABLE,
                    f"Exact {kind} image identity drifted",
                    f"expected={selected_image.config_digest}, "
                    f"actual={inspection.config_digest}",
                    "Restore the friendly tag only after exact digest verification.",
                )
            )
            if kind == "torch":
                torch_static_valid = False
        verification_error = backend.verify_image(
            selected_release,
            selected_image,
            kind,
        )
        if verification_error:
            code = "TORCH.BASE_CHANGED" if kind == "torch" else "IMAGE.DIGEST_DRIFT"
            diagnostics.append(
                _diagnostic(
                    code,
                    DiagnosticDisposition.REPAIRABLE,
                    f"Embedded {kind} image identity changed",
                    verification_error,
                    "Restore the exact verified parent image.",
                )
            )
            if kind == "torch":
                torch_static_valid = False

    for tag, image in (
        (BASE_FRIENDLY_TAG, selected_images["base"][1]),
        (TORCH_FRIENDLY_TAG, selected_images["torch"][1]),
    ):
        friendly_id = backend.inspect_friendly(tag)
        if friendly_id is not None and friendly_id not in {
            image.config_digest,
            image.manifest_digest,
        }:
            diagnostics.append(
                _diagnostic(
                    "IMAGE.DIGEST_DRIFT",
                    DiagnosticDisposition.REPAIRABLE,
                    "Friendly image tag points at another config",
                    f"{tag}={friendly_id}, expected={image.config_digest}",
                    "Retag only the verified exact config ID.",
                )
            )

    host_error = backend.host_preflight()
    if host_error:
        diagnostics.append(
            _diagnostic(
                "HOST.PREFLIGHT_FAILED",
                DiagnosticDisposition.BLOCKED,
                "Host preflight failed",
                host_error,
                "Repair the host driver and device permissions first.",
            )
        )
    if torch_static_valid:
        gpu_error = backend.gpu_runtime(
            selected_images["torch"][1].reference
        )
        if gpu_error:
            diagnostics.append(
                _diagnostic(
                    "GPU.RUNTIME_FAILED",
                    DiagnosticDisposition.BLOCKED,
                    "gfx1151 runtime operation failed",
                    gpu_error,
                    "Inspect ROCm and kernel evidence before retrying.",
                )
            )
    return DoctorReport.create(
        project=None,
        diagnostics=tuple(diagnostics),
        facts=facts,
        environment=os.environ,
    )


def doctor_project(
    *,
    config_path: Path,
    manifest_path: Path,
    backend: DoctorBackend,
    registry: str = "auto",
    preferred_references: Mapping[str, str] | None = None,
) -> DoctorReport:
    platform = doctor_platform(
        manifest_path=manifest_path,
        backend=backend,
        registry=registry,
        preferred_references=preferred_references,
    )
    diagnostics = list(platform.diagnostics)
    facts = dict(platform.facts)
    try:
        release = load_stable_release(manifest_path)
    except (OSError, ReleaseError):
        return DoctorReport.create(
            project=config_path.parent,
            diagnostics=tuple(diagnostics),
            facts=facts,
            environment=os.environ,
        )
    try:
        config = load_project_config(config_path)
    except (OSError, ConfigError) as error:
        diagnostics.append(
            _diagnostic(
                "PROJECT.CONFIG_INVALID",
                DiagnosticDisposition.BLOCKED,
                "Project configuration is invalid",
                str(error),
                "Restore a valid amd-ai-project.toml.",
            )
        )
        return DoctorReport.create(
            project=config_path.parent,
            diagnostics=tuple(diagnostics),
            facts=facts,
            environment=os.environ,
        )

    facts["project_config"] = str(config.path)
    if (
        config.base_digest != release.torch.config_digest
        or config.base_manifest_digest
        not in {None, release.torch.manifest_digest}
        or config.base_image
        not in {
            release.torch.config_digest,
            release.torch.manifest_digest,
        }
    ):
        diagnostics.append(
            _diagnostic(
                "IMAGE.DIGEST_DRIFT",
                DiagnosticDisposition.REPAIRABLE,
                "Project parent digest differs from stable release",
                f"configured={config.base_digest}, expected={release.torch.config_digest}",
                "Rebind and rebuild against the exact stable parent config.",
            )
        )
    project_image = backend.inspect_project(config)
    if project_image is None or project_image.changed:
        image_id = None if project_image is None else project_image.image_id
        facts["project_image_id"] = image_id
        diagnostics.append(
            _diagnostic(
                "IMAGE.PROJECT_CHANGED",
                DiagnosticDisposition.REPAIRABLE,
                "Derived project image changed or is missing",
                image_id or config.image,
                "Remove only the exact changed image ID and rebuild the project.",
            )
        )
    else:
        facts.update(
            {
                "project_image_id": project_image.image_id,
                "project_fingerprint": project_image.fingerprint,
            }
        )
        if project_image.base_manifest_error:
            diagnostics.append(
                _diagnostic(
                    "TORCH.BASE_CHANGED",
                    DiagnosticDisposition.REPAIRABLE,
                    "Project image changed protected Torch files",
                    project_image.base_manifest_error,
                    "Rebuild from the exact verified parent.",
                )
            )

    paths = OverlayPaths.for_project(config.path.parent)
    _check_overlay(
        paths=paths,
        config=config,
        release=release,
        backend=backend,
        diagnostics=diagnostics,
        facts=facts,
    )
    return DoctorReport.create(
        project=config.path.parent,
        diagnostics=tuple(diagnostics),
        facts=facts,
        environment=os.environ,
    )


def run_doctor(
    project: Path | None,
    manifest: Path,
    *,
    backend: DoctorBackend | None = None,
    registry: str = "auto",
    state_path: Path | None = None,
) -> DoctorReport:
    selected = backend or SubprocessDoctorBackend.detect()
    if project is None:
        return doctor_platform(
            manifest_path=manifest,
            backend=selected,
            registry=registry,
        )
    config_path = project if project.name == "amd-ai-project.toml" else project / "amd-ai-project.toml"
    preferred_references = _recorded_project_references(
        config_path.parent,
        manifest,
        state_path=state_path,
    )
    return doctor_project(
        config_path=config_path,
        manifest_path=manifest,
        backend=selected,
        registry=registry,
        preferred_references=preferred_references,
    )


def _prefer_registry_reference(
    candidates: tuple[RegistryCandidate, ...],
    *,
    kind: str,
    preferred_reference: str | None,
) -> tuple[RegistryCandidate, ...]:
    if preferred_reference is None:
        return candidates
    preferred = tuple(
        candidate
        for candidate in candidates
        if getattr(candidate.release, kind).reference == preferred_reference
    )
    if not preferred:
        return candidates
    return preferred + tuple(
        candidate
        for candidate in candidates
        if getattr(candidate.release, kind).reference != preferred_reference
    )


def _recorded_project_references(
    project: Path,
    manifest: Path,
    *,
    state_path: Path | None = None,
) -> Mapping[str, str]:
    try:
        release = load_stable_release(manifest)
    except (OSError, ReleaseError):
        return MappingProxyType({})
    legacy = default_state_path()
    paths = (
        (Path(state_path).resolve(strict=False),)
        if state_path is not None
        else (project_state_path(project, legacy), legacy)
    )
    for path in paths:
        try:
            state = load_state_readonly(path)
        except InstallerStateError:
            continue
        if (
            state is None
            or state.project_path != str(project.resolve())
            or state.release_id != release.release_id
            or state.source_revision != release.source_revision
            or state.base_manifest_digest != release.base.manifest_digest
            or state.base_config_digest != release.base.config_digest
            or state.torch_manifest_digest != release.torch.manifest_digest
            or state.torch_config_digest != release.torch.config_digest
        ):
            continue
        for candidate in registry_candidates(release):
            if (
                state.base_image_reference
                == candidate.release.base.reference
                and state.torch_image_reference
                == candidate.release.torch.reference
            ):
                return MappingProxyType(
                    {
                        "base": candidate.release.base.reference,
                        "torch": candidate.release.torch.reference,
                    }
                )
    return MappingProxyType({})


def _check_overlay(
    *,
    paths: OverlayPaths,
    config: ProjectConfig,
    release: StableRelease,
    backend: DoctorBackend,
    diagnostics: list[Diagnostic],
    facts: dict[str, object],
) -> None:
    try:
        current = resolve_current_generation(paths)
        state = load_generation_state(current)
        input_text = _read_text(current / "overlay.requirements.in")
        lock_text = _read_text(current / "overlay.requirements.lock")
        if state.generation_id != current.name:
            raise TransactionError("generation ID differs from state")
        if (
            state.profile_id != release.torch_profile_id
            or state.parent_config_digest != config.base_digest
        ):
            raise TransactionError("generation belongs to another parent")
        if hashlib.sha256(input_text.encode()).hexdigest() != state.input_digest:
            raise TransactionError("overlay input digest changed")
        if lock_digest(lock_text) != state.lock_digest:
            raise TransactionError("overlay lock digest changed")
        validate_lock_artifacts(parse_lock(lock_text), project=paths.project)
        site_packages = current / "site-packages"
        scan_protected_entries(site_packages)
        facts.update(
            {
                "current_generation": str(current),
                "last_valid_lock": str(current / "overlay.requirements.lock"),
            }
        )
    except (OSError, TransactionError, ValueError, OverlayVerificationError) as error:
        diagnostics.append(
            _diagnostic(
                "OVERLAY.LOCK_INVALID",
                DiagnosticDisposition.REPAIRABLE,
                "Active overlay generation is invalid",
                str(error),
                "Quarantine and replay the last valid hash lock.",
            )
        )
        return

    incomplete = []
    for generation in paths.generations.iterdir():
        if generation == current or not generation.is_dir():
            continue
        if not (generation / "overlay-state.json").is_file():
            incomplete.append(str(generation))
    if incomplete:
        diagnostics.append(
            _diagnostic(
                "OVERLAY.TRANSACTION_INCOMPLETE",
                DiagnosticDisposition.WARNING,
                "An unreferenced overlay transaction is incomplete",
                ", ".join(sorted(incomplete)),
                "Inspect and remove it only after confirming current health.",
            )
        )
    effective_error = backend.verify_effective_overlay(config, site_packages)
    if effective_error:
        diagnostics.append(
            _diagnostic(
                "TORCH.SHADOWED",
                DiagnosticDisposition.REPAIRABLE,
                "Project overlay shadows the protected Torch stack",
                effective_error,
                "Quarantine and replay the verified overlay lock.",
            )
        )


class SubprocessDoctorBackend:
    def __init__(self, docker_prefix: tuple[str, ...]) -> None:
        self.docker_prefix = docker_prefix
        self.registry = AnonymousDockerRegistry(docker_prefix)

    @classmethod
    def detect(cls) -> SubprocessDoctorBackend:
        return cls(Docker.detect().prefix)

    def inspect_image(self, reference: str) -> DoctorImageInspection | None:
        record = self._inspect(reference)
        if record is None:
            return None
        return _image_inspection(record)

    def inspect_friendly(self, reference: str) -> str | None:
        record = self._inspect(reference)
        if record is None:
            return None
        value = record.get("Id")
        return value if isinstance(value, str) else None

    def verify_image(
        self, release: StableRelease, image: ReleaseImage, kind: str
    ) -> str | None:
        try:
            verify_release_image(
                release,
                image,
                kind="base" if kind == "base" else "torch",
                docker=self.registry,
            )
        except (ReleaseError, PublishError) as error:
            return str(error)
        return None

    def host_preflight(self) -> str | None:
        missing = [
            str(path)
            for path in (Path("/dev/kfd"), Path("/dev/dri"))
            if not path.exists()
        ]
        return "missing GPU devices: " + ", ".join(missing) if missing else None

    def gpu_runtime(self, reference: str) -> str | None:
        try:
            access = discover_gpu_access()
        except RuntimePolicyError as error:
            return str(error)
        args = [
            "run",
            "--rm",
            "--read-only",
            "--ipc=private",
            "--shm-size=1g",
            *_readonly_probe_arguments(),
        ]
        for device in access.devices:
            args.extend(("--device", str(device)))
        for group_id in access.group_ids:
            args.extend(("--group-add", str(group_id)))
        args.extend(
            (reference, "container-check", "--mode", "torch", "--runtime")
        )
        result = self._completed(tuple(args))
        return None if result.returncode == 0 else _evidence(result)

    def inspect_project(self, config: ProjectConfig) -> ProjectImageInspection | None:
        record = self._inspect(config.image)
        if record is None:
            return None
        image_id = record.get("Id")
        raw_config = record.get("Config")
        labels = raw_config.get("Labels") if isinstance(raw_config, dict) else None
        if not isinstance(image_id, str) or not isinstance(labels, dict):
            return None
        expected = build_context_fingerprint(config.path.parent)
        changed = (
            labels.get("org.amd-ai.base.digest") != config.base_digest
            or labels.get("org.amd-ai.project.fingerprint") != expected
        )
        manifest = self._completed(
            (
                "run",
                "--rm",
                "--entrypoint",
                "/opt/venv/bin/python",
                config.image,
                "/opt/amd-ai/torch-manifest.py",
                "verify",
                "/opt/amd-ai/torch-manifest.json",
            )
        )
        return ProjectImageInspection(
            image_id=image_id,
            fingerprint=expected,
            changed=changed,
            base_manifest_error=(
                None if manifest.returncode == 0 else _evidence(manifest)
            ),
        )

    def verify_effective_overlay(
        self, config: ProjectConfig, site_packages: Path
    ) -> str | None:
        args = (
            "run",
            "--rm",
            "--read-only",
            *_readonly_probe_arguments(),
            "--entrypoint",
            "container-check",
            "--env",
            f"AMD_AI_OVERLAY=/workspace/.amd-ai/current/site-packages",
            "--env",
            f"AMD_AI_PARENT_CONFIG_DIGEST={config.base_digest}",
            "--env",
            "PYTHONPATH=/workspace/.amd-ai/current/site-packages:/opt/amd-ai/src",
            "--mount",
            f"type=bind,src={config.path.parent},dst=/workspace,readonly",
            config.image,
            "--mode",
            "torch",
            "--metadata-only",
        )
        result = self._completed(args)
        return None if result.returncode == 0 else _evidence(result)

    def _inspect(self, reference: str) -> Mapping[str, object] | None:
        result = self._completed(("image", "inspect", reference))
        if result.returncode != 0:
            return None
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None
        if (
            not isinstance(payload, list)
            or len(payload) != 1
            or not isinstance(payload[0], dict)
        ):
            return None
        return payload[0]

    def _completed(self, args: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            (*self.docker_prefix, *args),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


def _image_inspection(record: Mapping[str, object]) -> DoctorImageInspection:
    config_digest = record.get("Id")
    raw_config = record.get("Config")
    labels = raw_config.get("Labels") if isinstance(raw_config, Mapping) else None
    repo_digests = record.get("RepoDigests")
    if (
        not isinstance(config_digest, str)
        or not isinstance(labels, Mapping)
        or not isinstance(repo_digests, list)
    ):
        raise ValueError("image inspection is incomplete")
    return DoctorImageInspection(
        config_digest,
        {str(name): str(value) for name, value in labels.items()},
        tuple(str(value) for value in repo_digests),
    )


def _readonly_probe_arguments() -> tuple[str, ...]:
    return (
        "--tmpfs",
        DOCTOR_TMPFS,
        "--env",
        "HOME=/tmp/amd-ai-home",
        "--env",
        "PYTHONNOUSERSITE=1",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
    )


def _diagnostic(
    code: str,
    disposition: DiagnosticDisposition,
    summary: str,
    evidence: str,
    remediation: str,
) -> Diagnostic:
    return Diagnostic(code, disposition, summary, evidence, remediation)


def _read_text(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise TransactionError(f"overlay metadata is not a regular file: {path}")
    return path.read_text(encoding="utf-8")


def _evidence(result: subprocess.CompletedProcess[str]) -> str:
    return result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
