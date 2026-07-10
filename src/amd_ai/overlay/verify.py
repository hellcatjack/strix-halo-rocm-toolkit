from __future__ import annotations

import json
import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

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


@dataclass(frozen=True)
class EffectiveProbeResult:
    components: Mapping[str, Mapping[str, str]]
    torch_hip_version: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "components",
            MappingProxyType(
                {
                    name: MappingProxyType(dict(component))
                    for name, component in self.components.items()
                }
            ),
        )


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
    profile: ProtectedProfile | None = None,
    base_environment: Mapping[str, str] | None = None,
) -> None:
    scan_protected_entries(site_packages)
    verify_overlay_dependencies(
        site_packages,
        runner=runner,
        base_environment=base_environment,
    )
    if profile is not None:
        verify_effective_stack(
            profile,
            site_packages,
            runner=runner,
            base_environment=base_environment,
        )


def verify_effective_stack(
    profile: ProtectedProfile,
    site_packages: Path,
    *,
    runner: ProcessRunner,
    base_environment: Mapping[str, str] | None = None,
) -> EffectiveProbeResult:
    scan_protected_entries(site_packages)
    environment = dict(os.environ if base_environment is None else base_environment)
    environment.update(
        {
            "PYTHONPATH": f"{site_packages}:/opt/amd-ai/src",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    command = (
        "/opt/venv/bin/python",
        "-m",
        "amd_ai.overlay.effective_probe",
    )
    result = runner.run(
        list(command),
        environment=environment,
        cwd=Path("/workspace"),
    )
    if result.returncode != 0:
        evidence = result.stderr.strip() or result.stdout.strip() or "no output"
        raise OverlayVerificationError(
            f"effective protected import probe failed: {evidence}"
        )
    try:
        payload = json.loads(result.stdout, object_pairs_hook=_unique_json_object)
    except json.JSONDecodeError as error:
        raise OverlayVerificationError(
            f"cannot parse effective protected import probe: {error}"
        ) from error
    return validate_effective_probe(payload, profile=profile)


def validate_effective_probe(
    payload: object,
    *,
    profile: ProtectedProfile,
    base_root: Path = Path("/opt/venv"),
) -> EffectiveProbeResult:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "components",
        "torch_hip_version",
    }:
        raise OverlayVerificationError("effective protected probe schema is invalid")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise OverlayVerificationError("effective protected probe version is invalid")
    components = payload["components"]
    if not isinstance(components, dict) or set(components) != PROTECTED_DISTRIBUTIONS:
        raise OverlayVerificationError(
            "effective protected probe component set is invalid"
        )
    parsed: dict[str, dict[str, str]] = {}
    for name in sorted(PROTECTED_DISTRIBUTIONS):
        component = components[name]
        if not isinstance(component, dict):
            raise OverlayVerificationError(
                f"effective protected component is invalid: {name}"
            )
        if "error" in component:
            raise OverlayVerificationError(
                f"effective protected import failed for {name}: {component['error']}"
            )
        if set(component) != {"distribution_path", "module_path", "version"}:
            raise OverlayVerificationError(
                f"effective protected component schema is invalid: {name}"
            )
        values = {
            field: component[field]
            for field in ("distribution_path", "module_path", "version")
        }
        if any(not isinstance(value, str) or not value for value in values.values()):
            raise OverlayVerificationError(
                f"effective protected component identity is invalid: {name}"
            )
        if values["version"] != profile.version_for(name):
            raise OverlayVerificationError(
                f"effective protected version differs for {name}: {values['version']}"
            )
        for field in ("distribution_path", "module_path"):
            _require_path_below_base(
                values[field],
                base_root=base_root,
                label=f"{name} {field}",
            )
        parsed[name] = values
    hip = payload["torch_hip_version"]
    if not isinstance(hip, str) or re.fullmatch(
        r"7\.2\.(?:1|53211)(?:-[0-9A-Za-z.-]+)?", hip
    ) is None:
        raise OverlayVerificationError(
            f"effective Torch HIP version is not ROCm 7.2.1: {hip}"
        )
    return EffectiveProbeResult(parsed, hip)


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
            profile=profile,
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


def _require_path_below_base(
    value: str, *, base_root: Path, label: str
) -> None:
    path = Path(value)
    if not path.is_absolute() or "\0" in value:
        raise OverlayVerificationError(f"effective {label} path is invalid")
    resolved_base = base_root.resolve(strict=False)
    resolved = path.resolve(strict=False)
    if resolved != resolved_base and not resolved.is_relative_to(resolved_base):
        raise OverlayVerificationError(
            f"effective {label} is outside verified base: {resolved}"
        )


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    values: dict[str, object] = {}
    for name, value in pairs:
        if name in values:
            raise OverlayVerificationError(
                f"duplicate key in effective protected probe: {name}"
            )
        values[name] = value
    return values


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
