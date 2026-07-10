from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from amd_ai.doctor import checks
from amd_ai.doctor.checks import doctor_platform, doctor_project
from amd_ai.installer.release import load_stable_release
from amd_ai.overlay.models import (
    OverlayPaths,
    ProtectedComponent,
    ProtectedProfile,
)
from amd_ai.overlay.transaction import initialize_overlay, resolve_current_generation
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
