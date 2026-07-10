from __future__ import annotations

import json
import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from amd_ai.overlay.lock import parse_lock, validate_lock_artifacts
from amd_ai.overlay.models import (
    PROTECTED_DISTRIBUTIONS,
    OverlayPaths,
    ProtectedComponent,
    ProtectedProfile,
    canonicalize_protected_name,
)
from amd_ai.overlay.resolver import ProcessRunner
from amd_ai.overlay.transaction import (
    load_generation_state,
    resolve_current_generation,
)


METADATA_NAME_PATTERN = re.compile(r"(?P<name>.+?)-[0-9].*")


class OverlayVerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerifiedGeneration:
    generation: Path
    input_text: str
    lock_text: str


def verify_base_manifest(
    *,
    runner: ProcessRunner,
    base_environment: Mapping[str, str] | None = None,
) -> None:
    environment = dict(os.environ if base_environment is None else base_environment)
    environment["PYTHONNOUSERSITE"] = "1"
    argv = (
        "/opt/venv/bin/python",
        "/opt/amd-ai/torch-manifest.py",
        "verify",
        "/opt/amd-ai/torch-manifest.json",
    )
    result = runner.run(
        list(argv),
        environment=environment,
        cwd=Path("/workspace"),
    )
    if result.returncode != 0:
        evidence = result.stderr.strip() or result.stdout.strip() or "no output"
        raise OverlayVerificationError(
            f"protected Torch base manifest check failed: {evidence}"
        )


def scan_protected_entries(site_packages: Path) -> None:
    if site_packages.is_symlink():
        raise OverlayVerificationError(
            f"overlay site-packages must not be a symlink: {site_packages}"
        )
    if not site_packages.is_dir():
        raise OverlayVerificationError(
            f"overlay site-packages is not a directory: {site_packages}"
        )
    for entry in site_packages.iterdir():
        candidate = _entry_distribution_name(entry.name)
        protected = canonicalize_protected_name(candidate)
        if protected in PROTECTED_DISTRIBUTIONS:
            raise OverlayVerificationError(
                f"overlay contains protected distribution entry: {entry.name}"
            )


def load_protected_profile(
    *,
    manifest_path: Path = Path("/opt/amd-ai/torch-manifest.json"),
    profile_path: Path = Path("/opt/amd-ai/profile.env"),
    parent_config_digest: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> ProtectedProfile:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OverlayVerificationError(
            f"cannot read protected Torch manifest: {error}"
        ) from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise OverlayVerificationError("protected Torch manifest schema is invalid")
    packages = manifest.get("packages")
    if not isinstance(packages, list):
        raise OverlayVerificationError("protected Torch manifest packages are invalid")
    components: list[ProtectedComponent] = []
    for package in packages:
        if not isinstance(package, dict):
            raise OverlayVerificationError(
                "protected Torch manifest package is invalid"
            )
        name = package.get("name")
        version = package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise OverlayVerificationError(
                "protected Torch manifest identity is invalid"
            )
        protected = canonicalize_protected_name(name)
        if protected in PROTECTED_DISTRIBUTIONS:
            try:
                components.append(ProtectedComponent(protected, version))
            except Exception as error:
                raise OverlayVerificationError(
                    f"protected Torch manifest version is invalid: {name}={version}"
                ) from error
    if len(components) != 4:
        raise OverlayVerificationError(
            "protected Torch manifest does not contain exactly four components"
        )

    profile_id = _read_profile_id(profile_path)
    values = os.environ if environment is None else environment
    digest = parent_config_digest or values.get("AMD_AI_PARENT_CONFIG_DIGEST")
    if not digest:
        raise OverlayVerificationError(
            "AMD_AI_PARENT_CONFIG_DIGEST is required for protected pip"
        )
    try:
        return ProtectedProfile(profile_id, digest, tuple(components))
    except Exception as error:
        raise OverlayVerificationError(
            f"protected profile identity is invalid: {error}"
        ) from error


def verify_overlay_dependencies(
    site_packages: Path,
    *,
    runner: ProcessRunner,
    base_environment: Mapping[str, str] | None = None,
) -> None:
    environment = dict(os.environ if base_environment is None else base_environment)
    environment.update(
        {
            "PYTHONPATH": f"{site_packages}:/opt/amd-ai/src",
            "PYTHONNOUSERSITE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        }
    )
    argv = (
        "/opt/venv/bin/python",
        "-m",
        "pip",
        "check",
        "--disable-pip-version-check",
    )
    result = runner.run(
        list(argv),
        environment=environment,
        cwd=Path("/workspace"),
    )
    if result.returncode != 0:
        evidence = result.stderr.strip() or result.stdout.strip() or "no output"
        raise OverlayVerificationError(
            f"overlay dependency check failed: {evidence}"
        )


def verify_candidate_overlay(
    site_packages: Path,
    *,
    runner: ProcessRunner,
    base_environment: Mapping[str, str] | None = None,
) -> None:
    scan_protected_entries(site_packages)
    verify_overlay_dependencies(
        site_packages,
        runner=runner,
        base_environment=base_environment,
    )


def verify_current_generation(
    paths: OverlayPaths,
    *,
    profile: ProtectedProfile,
    runner: ProcessRunner,
    base_environment: Mapping[str, str] | None = None,
) -> VerifiedGeneration:
    try:
        generation = resolve_current_generation(paths)
        state = load_generation_state(generation)
        if state.generation_id != generation.name:
            raise OverlayVerificationError(
                "current overlay generation identity does not match state"
            )
        if (
            state.profile_id != profile.profile_id
            or state.parent_config_digest != profile.parent_config_digest
        ):
            raise OverlayVerificationError(
                "current overlay belongs to a different protected parent"
            )
        input_text = _read_regular_text(
            generation / "overlay.requirements.in"
        )
        lock_text = _read_regular_text(
            generation / "overlay.requirements.lock"
        )
        if _text_digest(input_text) != state.input_digest:
            raise OverlayVerificationError(
                "current overlay input digest does not match state"
            )
        if _text_digest(lock_text) != state.lock_digest:
            raise OverlayVerificationError(
                "current overlay lock digest does not match state"
            )
        validate_lock_artifacts(parse_lock(lock_text), project=paths.project)
        verify_candidate_overlay(
            generation / "site-packages",
            runner=runner,
            base_environment=base_environment,
        )
    except Exception as error:
        if isinstance(error, KeyboardInterrupt):
            raise
        raise OverlayVerificationError(
            "current overlay verification failed; run amd-ai doctor and repair: "
            f"{error}"
        ) from error
    return VerifiedGeneration(generation, input_text, lock_text)


def _entry_distribution_name(filename: str) -> str:
    lowered = filename.lower()
    if lowered.endswith(".py"):
        return filename[:-3]
    for suffix in (".dist-info", ".egg-info"):
        if lowered.endswith(suffix):
            stem = filename[: -len(suffix)]
            match = METADATA_NAME_PATTERN.fullmatch(stem)
            return match.group("name") if match is not None else stem
    return filename


def _read_regular_text(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise OverlayVerificationError(
            f"overlay metadata is not a regular file: {path}"
        )
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise OverlayVerificationError(
            f"cannot read overlay metadata {path}: {error}"
        ) from error


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_profile_id(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise OverlayVerificationError(
            f"cannot read protected profile: {error}"
        ) from error
    profile_ids: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if not separator or not name or not value:
            raise OverlayVerificationError(
                f"invalid protected profile line: {raw_line}"
            )
        if name == "PROFILE_ID":
            profile_ids.append(value)
    if len(profile_ids) != 1:
        raise OverlayVerificationError(
            "protected profile must contain one PROFILE_ID"
        )
    return profile_ids[0]
