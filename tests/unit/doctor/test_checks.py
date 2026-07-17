from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from amd_ai.doctor import checks
from amd_ai.doctor.checks import (
    SubprocessDoctorBackend,
    doctor_platform,
    doctor_project,
)
from amd_ai.installer.registry import registry_candidates
from amd_ai.installer.release import load_stable_release
from amd_ai.overlay.models import (
    OverlayPaths,
    ProtectedComponent,
    ProtectedProfile,
)
from amd_ai.overlay.transaction import initialize_overlay, resolve_current_generation
from amd_ai.project.runtime import GpuAccess
from tests.unit.doctor.fakes import FakeDoctorBackend
from tests.unit.project.fakes import project_config


FIXTURE = Path("tests/fixtures/releases/stable.json")


@pytest.mark.parametrize(
    ("state", "expected_code"),
    (
        ("invalid_release", "RELEASE.INVALID"),
        ("missing_parent", "IMAGE.PARENT_MISSING"),
        ("tag_drift", "IMAGE.DIGEST_DRIFT"),
        ("base_manifest", "TORCH.BASE_CHANGED"),
        ("gpu", "GPU.RUNTIME_FAILED"),
    ),
)
def test_platform_classification_is_stable_and_read_only(
    tmp_path: Path, state: str, expected_code: str
) -> None:
    release = load_stable_release(FIXTURE)
    backend = FakeDoctorBackend(release)
    manifest = FIXTURE
    if state == "invalid_release":
        manifest = tmp_path / "invalid.json"
        manifest.write_text("{}\n", encoding="utf-8")
    elif state == "missing_parent":
        del backend.images[release.torch.reference]
    elif state == "tag_drift":
        backend.friendly[
            "rocm-pytorch:7.2.1-py3.12-torch2.9.1"
        ] = "sha256:" + "9" * 64
    elif state == "base_manifest":
        backend.verify_errors["torch"] = "manifest changed"
    elif state == "gpu":
        backend.gpu_error = "gfx1151 operation failed"

    report = doctor_platform(manifest_path=manifest, backend=backend)

    assert expected_code in {item.code for item in report.diagnostics}
    assert backend.mutations == []


def test_platform_accepts_containerd_manifest_ids_after_release_verification() -> None:
    release = load_stable_release(FIXTURE)
    backend = FakeDoctorBackend(release)
    backend.images[release.base.reference] = replace(
        backend.images[release.base.reference],
        config_digest=release.base.manifest_digest,
    )
    backend.images[release.torch.reference] = replace(
        backend.images[release.torch.reference],
        config_digest=release.torch.manifest_digest,
    )
    backend.friendly["rocm-python:7.2.1-py3.12"] = release.base.manifest_digest
    backend.friendly[
        "rocm-pytorch:7.2.1-py3.12-torch2.9.1"
    ] = release.torch.manifest_digest

    report = doctor_platform(manifest_path=FIXTURE, backend=backend)

    assert "IMAGE.DIGEST_DRIFT" not in {
        item.code for item in report.diagnostics
    }


def test_platform_accepts_verified_swr_parent_images() -> None:
    release = load_stable_release(FIXTURE)
    swr = registry_candidates(release, "swr")[0].release
    backend = FakeDoctorBackend(swr)

    report = doctor_platform(
        manifest_path=FIXTURE,
        backend=backend,
        registry="auto",
    )

    assert report.status == "pass"
    assert report.facts["base_reference"] == swr.base.reference
    assert report.facts["torch_reference"] == swr.torch.reference


def test_platform_uses_canonical_parent_when_swr_is_absent() -> None:
    release = load_stable_release(FIXTURE)
    backend = FakeDoctorBackend(release)

    report = doctor_platform(
        manifest_path=FIXTURE,
        backend=backend,
        registry="auto",
    )

    assert report.status == "pass"
    assert report.facts["torch_reference"] == release.torch.reference


def test_doctor_release_descriptor_lookup_uses_anonymous_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = SubprocessDoctorBackend(("sudo", "-n", "docker"))
    reference = "ghcr.io/example/image@sha256:" + "a" * 64
    expected = "sha256:" + "b" * 64
    calls: list[str] = []
    monkeypatch.setattr(
        backend.registry,
        "authless_manifest_config_digest",
        lambda value: calls.append(value) or expected,
    )

    observed = backend.registry.manifest_config_digest(reference)

    assert observed == expected
    assert calls == [reference]


@pytest.mark.parametrize(
    ("state", "expected_code"),
    (
        ("changed_project", "IMAGE.PROJECT_CHANGED"),
        ("overlay_shadow", "TORCH.SHADOWED"),
        ("lock_invalid", "OVERLAY.LOCK_INVALID"),
        ("incomplete", "OVERLAY.TRANSACTION_INCOMPLETE"),
    ),
)
def test_project_classification_is_stable_and_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    expected_code: str,
) -> None:
    release = load_stable_release(FIXTURE)
    backend = FakeDoctorBackend(release)
    config = replace(
        project_config(tmp_path / "demo"),
        base_image=release.torch.config_digest,
        base_digest=release.torch.config_digest,
    )
    monkeypatch.setattr(checks, "load_project_config", lambda path: config)
    paths = OverlayPaths.for_project(config.path.parent)
    initialize_overlay(
        paths,
        profile=_profile(release.torch.config_digest),
        transaction_id="20260710T120000Z-a1b2c3d4",
    )
    if state == "changed_project":
        backend.project = replace(backend.project, changed=True)
    elif state == "overlay_shadow":
        backend.overlay_error = "torch imported from overlay"
    elif state == "lock_invalid":
        generation = resolve_current_generation(paths)
        (generation / "overlay.requirements.in").write_text(
            "changed==1\n", encoding="utf-8"
        )
    else:
        (paths.generations / "20260710T120001Z-b1c2d3e4").mkdir()

    report = doctor_project(
        config_path=config.path,
        manifest_path=FIXTURE,
        backend=backend,
    )

    assert expected_code in {item.code for item in report.diagnostics}
    assert backend.mutations == []


def test_subprocess_gpu_probe_supplies_tmp_home_and_device_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    access = GpuAccess(
        devices=(Path("/dev/kfd"), Path("/dev/dri")),
        render_nodes=(Path("/dev/dri/renderD128"),),
        group_ids=(993,),
    )
    monkeypatch.setattr(checks, "discover_gpu_access", lambda: access)

    def completed(argv, **kwargs):
        del kwargs
        calls.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(checks.subprocess, "run", completed)

    error = SubprocessDoctorBackend(("sudo", "-n", "docker")).gpu_runtime(
        "ghcr.io/example/torch@sha256:" + "a" * 64
    )

    assert error is None
    command = calls[0]
    assert ("--group-add", "993") == command[
        command.index("--group-add") : command.index("--group-add") + 2
    ]
    assert "/tmp:rw,nosuid,nodev,size=1g,mode=1777" in command
    assert "HOME=/tmp/amd-ai-home" in command


def test_subprocess_overlay_probe_supplies_writable_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ...]] = []

    def completed(argv, **kwargs):
        del kwargs
        calls.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(checks.subprocess, "run", completed)
    config = project_config(tmp_path / "demo")

    error = SubprocessDoctorBackend(
        ("sudo", "-n", "docker")
    ).verify_effective_overlay(
        config, config.path.parent / ".amd-ai/current/site-packages"
    )

    assert error is None
    command = calls[0]
    assert "/tmp:rw,nosuid,nodev,size=1g,mode=1777" in command
    assert "HOME=/tmp/amd-ai-home" in command


def _profile(parent_digest: str) -> ProtectedProfile:
    return ProtectedProfile(
        "rocm-7.2.1-py3.12-torch-2.9.1",
        parent_digest,
        (
            ProtectedComponent("torch", "2.9.1+rocm7.2.1.a"),
            ProtectedComponent("torchvision", "0.24.0+rocm7.2.1.b"),
            ProtectedComponent("torchaudio", "2.9.0+rocm7.2.1.c"),
            ProtectedComponent("triton", "3.5.1+rocm7.2.1.d"),
        ),
    )
