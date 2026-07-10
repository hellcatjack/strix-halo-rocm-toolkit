from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from amd_ai.installer.models import InstallMode, InstallStage, InstallState
from amd_ai.installer.state import (
    CorruptInstallState,
    InstallAlreadyRunning,
    ResumeInputChanged,
    boot_id_changed,
    install_lock,
    load_state,
    read_boot_id,
    save_state,
    stage_input_digest,
    validate_completed_stage,
)


def install_state(tmp_path: Path, **changes: object) -> InstallState:
    values: dict[str, object] = {
        "schema_version": 1,
        "installer_version": "0.2.0",
        "mode": InstallMode.CONTAINER,
        "target_user": "developer",
        "release_id": "1.0.0",
        "source_revision": "a" * 40,
        "base_image_reference": "ghcr.io/example/base@sha256:" + "b" * 64,
        "base_manifest_digest": "sha256:" + "b" * 64,
        "torch_image_reference": "ghcr.io/example/torch@sha256:" + "c" * 64,
        "torch_manifest_digest": "sha256:" + "c" * 64,
        "project_path": str((tmp_path / "project").resolve()),
        "current_stage": InstallStage.IMAGE_VERIFY,
        "completed_stage_input_digests": {
            InstallStage.RELEASE_RESOLVE.value: stage_input_digest(
                {"release": "1.0.0"}
            )
        },
        "reboot_boot_id": None,
        "created_at": "2026-07-10T10:00:00Z",
        "updated_at": "2026-07-10T10:01:00Z",
        "installer_source_revision": "d" * 40,
        "source_root": str(tmp_path.resolve()),
        "host_plan_digest": None,
        "last_report_paths": (str((tmp_path / "report.json").resolve()),),
    }
    values.update(changes)
    return InstallState(**values)  # type: ignore[arg-type]


def test_stage_digest_is_canonical_across_mapping_order() -> None:
    left = stage_input_digest(
        {"mode": "container", "facts": {"b": 2, "a": 1}}
    )
    right = stage_input_digest(
        {"facts": {"a": 1, "b": 2}, "mode": "container"}
    )

    assert left == right
    assert len(left) == 64


def test_state_round_trip_uses_atomic_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "install-state.json"
    calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def record_replace(source: str | Path, target: str | Path) -> None:
        calls.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", record_replace)
    expected = install_state(tmp_path)

    save_state(path, expected)

    assert load_state(path) == expected
    assert calls[-1][1] == path
    assert path.stat().st_mode & 0o777 == 0o600


def test_corrupt_state_is_preserved_before_replanning(tmp_path: Path) -> None:
    path = tmp_path / "install-state.json"
    path.write_text("not-json", encoding="utf-8")

    with pytest.raises(CorruptInstallState) as error:
        load_state(path)

    assert error.value.preserved_path.read_text(encoding="utf-8") == "not-json"
    assert not path.exists()


def test_unknown_or_sensitive_state_key_is_preserved_as_corrupt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "install-state.json"
    save_state(path, install_state(tmp_path))
    payload = json.loads(path.read_text(encoding="ascii"))
    payload["access_token"] = "must-not-load"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CorruptInstallState, match="unknown"):
        load_state(path)


def test_install_lock_refuses_a_concurrent_workflow(tmp_path: Path) -> None:
    path = tmp_path / "install-state.json"

    with install_lock(path):
        with pytest.raises(InstallAlreadyRunning):
            with install_lock(path):
                pytest.fail("second lock must not be acquired")


def test_completed_stage_accepts_same_inputs_and_rejects_changed_inputs(
    tmp_path: Path,
) -> None:
    state = install_state(tmp_path)

    assert (
        validate_completed_stage(
            state,
            InstallStage.RELEASE_RESOLVE,
            {"release": "1.0.0"},
        )
        is True
    )
    with pytest.raises(ResumeInputChanged) as error:
        validate_completed_stage(
            state,
            InstallStage.RELEASE_RESOLVE,
            {"release": "1.0.1"},
        )

    assert error.value.stage is InstallStage.RELEASE_RESOLVE


def test_missing_completed_stage_returns_false(tmp_path: Path) -> None:
    assert (
        validate_completed_stage(
            install_state(tmp_path),
            InstallStage.IMAGE_VERIFY,
            {"image": "unchanged"},
        )
        is False
    )


def test_boot_id_requires_canonical_uuid_and_detects_reboot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "boot_id"
    first = "12345678-1234-4abc-8def-1234567890ab"
    second = "87654321-4321-4abc-8def-ba0987654321"
    path.write_text(first + "\n", encoding="ascii")

    assert read_boot_id(path) == first
    assert boot_id_changed(first, current_boot_id=first) is False
    assert boot_id_changed(first, current_boot_id=second) is True

    path.write_text("not-a-uuid\n", encoding="ascii")
    with pytest.raises(ValueError, match="boot ID"):
        read_boot_id(path)
