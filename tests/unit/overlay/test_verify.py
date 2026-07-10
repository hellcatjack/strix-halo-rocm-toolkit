from __future__ import annotations

import json
from pathlib import Path

import pytest

from amd_ai.overlay.verify import (
    EffectiveProbeResult,
    OverlayVerificationError,
    load_protected_profile,
    scan_protected_entries,
    verify_base_manifest,
    verify_current_generation,
    validate_effective_probe,
    verify_effective_stack,
    verify_overlay_dependencies,
)
from amd_ai.overlay.models import (
    OverlayPaths,
    ProtectedComponent,
    ProtectedProfile,
)
from amd_ai.overlay.transaction import initialize_overlay, resolve_current_generation
from amd_ai.runner import CommandResult


@pytest.mark.parametrize(
    "name",
    [
        "torch",
        "torch.py",
        "Torch-2.9.1.dist-info",
        "torch_vision-0.24.0.egg-info",
        "triton",
    ],
)
def test_structural_scan_blocks_protected_shadow(
    name: str, tmp_path: Path
) -> None:
    path = tmp_path / name
    if name.endswith((".dist-info", ".egg-info")) or "." not in name:
        path.mkdir()
    else:
        path.write_text("", encoding="utf-8")

    with pytest.raises(OverlayVerificationError, match="protected"):
        scan_protected_entries(tmp_path)


def test_structural_scan_allows_unrelated_packages(tmp_path: Path) -> None:
    (tmp_path / "requests").mkdir()
    (tmp_path / "requests-2.32.5.dist-info").mkdir()

    scan_protected_entries(tmp_path)


def test_load_profile_uses_full_manifest_versions(tmp_path: Path) -> None:
    manifest = tmp_path / "torch-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "packages": [
                    {"name": "torch", "version": "2.9.1+rocm7.2.1.a", "files": []},
                    {"name": "torchvision", "version": "0.24.0+rocm7.2.1.b", "files": []},
                    {"name": "torchaudio", "version": "2.9.0+rocm7.2.1.c", "files": []},
                    {"name": "triton", "version": "3.5.1+rocm7.2.1.d", "files": []},
                ],
            }
        ),
        encoding="utf-8",
    )
    profile = tmp_path / "profile.env"
    profile.write_text(
        "PROFILE_ID=rocm-7.2.1-py3.12-torch-2.9.1\n",
        encoding="utf-8",
    )

    loaded = load_protected_profile(
        manifest_path=manifest,
        profile_path=profile,
        parent_config_digest="sha256:" + "a" * 64,
    )

    assert loaded.profile_id == "rocm-7.2.1-py3.12-torch-2.9.1"
    assert loaded.version_for("torch") == "2.9.1+rocm7.2.1.a"


class CheckRunner:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode
        self.calls: list[tuple[tuple[str, ...], dict[str, str], Path | None]] = []

    def run(
        self,
        args: list[str],
        *,
        environment: dict[str, str],
        cwd: Path | None = None,
    ) -> CommandResult:
        self.calls.append((tuple(args), environment, cwd))
        return CommandResult(tuple(args), self.returncode, "", "broken dependency")


def test_dependency_check_uses_candidate_before_base(tmp_path: Path) -> None:
    runner = CheckRunner(returncode=0)

    verify_overlay_dependencies(tmp_path, runner=runner)

    command, environment, cwd = runner.calls[0]
    assert command == (
        "/opt/venv/bin/python",
        "-m",
        "pip",
        "check",
        "--disable-pip-version-check",
    )
    assert environment["PYTHONPATH"] == f"{tmp_path}:/opt/amd-ai/src"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert cwd == Path("/workspace")


def test_dependency_check_failure_blocks_generation(tmp_path: Path) -> None:
    with pytest.raises(OverlayVerificationError, match="dependency check"):
        verify_overlay_dependencies(tmp_path, runner=CheckRunner(returncode=1))


def test_base_manifest_uses_immutable_parent_files() -> None:
    runner = CheckRunner(returncode=0)

    verify_base_manifest(runner=runner)

    command, environment, cwd = runner.calls[0]
    assert command == (
        "/opt/venv/bin/python",
        "/opt/amd-ai/torch-manifest.py",
        "verify",
        "/opt/amd-ai/torch-manifest.json",
    )
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert cwd == Path("/workspace")


def test_current_generation_rejects_tampered_input_digest(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    paths = OverlayPaths.for_project(project)
    profile = _profile()
    initialize_overlay(
        paths,
        profile=profile,
        transaction_id="20260710T120000Z-a1b2c3d4",
    )
    generation = resolve_current_generation(paths)
    (generation / "overlay.requirements.in").write_text(
        "tampered==1\n", encoding="utf-8"
    )

    with pytest.raises(OverlayVerificationError, match="repair"):
        verify_current_generation(
            paths,
            profile=profile,
            runner=CheckRunner(returncode=0),
        )


def test_current_generation_returns_verified_metadata(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    paths = OverlayPaths.for_project(project)
    profile = _profile()
    initialize_overlay(
        paths,
        profile=profile,
        transaction_id="20260710T120000Z-a1b2c3d4",
    )

    current = verify_current_generation(
        paths,
        profile=profile,
        runner=ProbeRunner(_valid_probe(profile)),
    )

    assert current.generation.name == "20260710T120000Z-a1b2c3d4"
    assert current.input_text == ""
    assert current.lock_text == ""


def test_effective_identity_requires_base_distribution_and_module_paths() -> None:
    profile = _profile()
    payload = _valid_probe(profile)

    result = validate_effective_probe(payload, profile=profile)

    assert isinstance(result, EffectiveProbeResult)
    assert result.torch_hip_version == "7.2.1"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "distribution_path",
            "/workspace/.amd-ai/current/site-packages/torch.dist-info",
        ),
        ("module_path", "/workspace/torch.py"),
        ("version", "2.9.1"),
    ),
)
def test_effective_identity_rejects_shadow_or_public_only_version(
    field: str, value: str
) -> None:
    profile = _profile()
    payload = _valid_probe(profile)
    payload["components"]["torch"][field] = value

    with pytest.raises(OverlayVerificationError):
        validate_effective_probe(payload, profile=profile)


@pytest.mark.parametrize("hip", ("6.4", "7.2.4", None))
def test_effective_identity_rejects_wrong_hip_version(hip: object) -> None:
    profile = _profile()
    payload = _valid_probe(profile)
    payload["torch_hip_version"] = hip

    with pytest.raises(OverlayVerificationError, match="HIP"):
        validate_effective_probe(payload, profile=profile)


def test_effective_identity_accepts_verified_rocm_build_version() -> None:
    profile = _profile()
    payload = _valid_probe(profile)
    payload["torch_hip_version"] = "7.2.53211-e1a6bc5663"

    result = validate_effective_probe(payload, profile=profile)

    assert result.torch_hip_version == "7.2.53211-e1a6bc5663"


def test_effective_probe_runs_with_candidate_first(tmp_path: Path) -> None:
    profile = _profile()
    runner = ProbeRunner(_valid_probe(profile))

    result = verify_effective_stack(
        profile,
        tmp_path,
        runner=runner,
    )

    command, environment, cwd = runner.calls[0]
    assert result.torch_hip_version == "7.2.1"
    assert command == (
        "/opt/venv/bin/python",
        "-m",
        "amd_ai.overlay.effective_probe",
    )
    assert environment["PYTHONPATH"] == f"{tmp_path}:/opt/amd-ai/src"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert cwd == Path("/workspace")


def _profile() -> ProtectedProfile:
    return ProtectedProfile(
        "stable",
        "sha256:" + "a" * 64,
        (
            ProtectedComponent("torch", "2.9.1+rocm7.2.1.a"),
            ProtectedComponent("torchvision", "0.24.0+rocm7.2.1.b"),
            ProtectedComponent("torchaudio", "2.9.0+rocm7.2.1.c"),
            ProtectedComponent("triton", "3.5.1+rocm7.2.1.d"),
        ),
    )


def _valid_probe(profile: ProtectedProfile) -> dict:
    return {
        "schema_version": 1,
        "components": {
            name: {
                "distribution_path": (
                    f"/opt/venv/lib/python3.12/site-packages/{name}-x.dist-info"
                ),
                "module_path": (
                    f"/opt/venv/lib/python3.12/site-packages/{name}/__init__.py"
                ),
                "version": profile.version_for(name),
            }
            for name in ("torch", "torchvision", "torchaudio", "triton")
        },
        "torch_hip_version": "7.2.1",
    }


class ProbeRunner(CheckRunner):
    def __init__(self, payload: dict) -> None:
        super().__init__(returncode=0)
        self.payload = payload

    def run(
        self,
        args: list[str],
        *,
        environment: dict[str, str],
        cwd: Path | None = None,
    ) -> CommandResult:
        self.calls.append((tuple(args), environment, cwd))
        return CommandResult(tuple(args), 0, json.dumps(self.payload), "")
