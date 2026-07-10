from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from amd_ai.installer.fixture import fixture_host_plan_digest
from amd_ai.installer.models import (
    FULL_STAGE_ORDER,
    InstallMode,
    InstallStage,
    InstallState,
)
from amd_ai.installer.state import project_state_path, save_state


ROOT = Path.cwd()
MANIFEST = (ROOT / "tests/fixtures/releases/stable.json").resolve()
INSTALL = ROOT / "install.sh"
BOOT_A = "12345678-1234-4abc-8def-1234567890ab"
BOOT_B = "87654321-4321-4abc-8def-ba0987654321"


def fixture_environment(home: Path, fixture: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "HOME": str(home),
            "AMD_AI_INSTALLER_ENABLE_FIXTURES": "1",
            "AMD_AI_INSTALLER_FIXTURE_ROOT": str(fixture),
        }
    )
    return environment


def make_fixture(tmp_path: Path, scenario_name: str) -> Path:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    source = ROOT / "tests/fixtures/installer" / scenario_name
    (fixture / "scenario.json").write_text(
        source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (fixture / "boot_id").write_text(BOOT_A + "\n", encoding="ascii")
    return fixture


def run_install(
    *,
    home: Path,
    fixture: Path,
    state: Path | None,
    project: Path,
    mode: str,
    extra: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    arguments = [
            str(INSTALL),
            "--mode",
            mode,
            "--non-interactive",
            "--project-dir",
            str(project),
            "--project-name",
            "fixture-project",
            "--image-source",
            "pull",
            "--manifest",
            str(MANIFEST),
            "--target-user",
            "developer",
    ]
    if state is not None:
        arguments.extend(("--state-path", str(state)))
    arguments.extend(extra)
    return subprocess.run(
        arguments,
        cwd=ROOT,
        env=fixture_environment(home, fixture),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_real_install_script_completes_fixture_container_mode(
    tmp_path: Path,
) -> None:
    fixture = make_fixture(tmp_path, "container-healthy.json")
    home = tmp_path / "home"
    state = tmp_path / "state.json"
    project = tmp_path / "project"

    result = run_install(
        home=home,
        fixture=fixture,
        state=state,
        project=project,
        mode="container",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(state.read_text(encoding="utf-8"))
    assert payload["current_stage"] == "COMPLETE"
    assert (project / "amd-ai-project.toml").is_file()
    assert os.access(home / ".local/bin/strix-halo-rocm", os.X_OK)
    calls = (fixture / "calls.log").read_text(encoding="utf-8").splitlines()
    assert calls == [
        "bootstrap",
        "container_host_check",
        "resolve_release",
        "pull_release",
        "verify_torch_image",
        "initialize_project",
        "verify_project",
    ]


def test_full_fixture_resumes_only_after_boot_id_changes(
    tmp_path: Path,
) -> None:
    fixture = make_fixture(tmp_path, "full-host-change.json")
    home = tmp_path / "home"
    state = tmp_path / "state.json"
    project = tmp_path / "project"
    digest = fixture_host_plan_digest("developer")
    extra = ("--accept-host-plan-digest", digest)

    first = run_install(
        home=home,
        fixture=fixture,
        state=state,
        project=project,
        mode="full",
        extra=extra,
    )
    second = run_install(
        home=home,
        fixture=fixture,
        state=state,
        project=project,
        mode="full",
        extra=extra,
    )
    (fixture / "boot_id").write_text(BOOT_B + "\n", encoding="ascii")
    third = run_install(
        home=home,
        fixture=fixture,
        state=state,
        project=project,
        mode="full",
        extra=extra,
    )

    assert first.returncode == 1
    assert second.returncode == 1
    assert third.returncode == 0, third.stderr
    calls = (fixture / "calls.log").read_text(encoding="utf-8").splitlines()
    assert calls.count("host_apply") == 1
    assert calls.count("host_verify") == 1
    assert calls.index("host_apply") < calls.index("host_verify")


def test_real_install_preserves_corrupt_state_without_replaying_actions(
    tmp_path: Path,
) -> None:
    fixture = make_fixture(tmp_path, "container-healthy.json")
    home = tmp_path / "home"
    state = tmp_path / "state.json"
    project = tmp_path / "project"
    first = run_install(
        home=home,
        fixture=fixture,
        state=state,
        project=project,
        mode="container",
    )
    assert first.returncode == 0
    calls_before = (fixture / "calls.log").read_text(encoding="utf-8")
    state.write_text("not-json", encoding="utf-8")

    resumed = run_install(
        home=home,
        fixture=fixture,
        state=state,
        project=project,
        mode="container",
    )

    assert resumed.returncode == 2
    assert not state.exists()
    assert len(list(tmp_path.glob("state.corrupt.*.json"))) == 1
    assert (fixture / "calls.log").read_text(encoding="utf-8") == calls_before


def test_real_install_blocks_changed_project_path_before_actions(
    tmp_path: Path,
) -> None:
    fixture = make_fixture(tmp_path, "container-healthy.json")
    home = tmp_path / "home"
    state = tmp_path / "state.json"
    first = run_install(
        home=home,
        fixture=fixture,
        state=state,
        project=tmp_path / "project-a",
        mode="container",
    )
    assert first.returncode == 0
    calls_before = (fixture / "calls.log").read_text(encoding="utf-8")

    resumed = run_install(
        home=home,
        fixture=fixture,
        state=state,
        project=tmp_path / "project-b",
        mode="container",
    )

    assert resumed.returncode == 2
    assert "inputs changed" in resumed.stderr
    assert (fixture / "calls.log").read_text(encoding="utf-8") == calls_before


def test_real_install_implicitly_isolates_second_project_from_legacy_state(
    tmp_path: Path,
) -> None:
    fixture = make_fixture(tmp_path, "container-healthy.json")
    home = tmp_path / "home"
    legacy = (
        home
        / ".local/state/strix-halo-rocm-toolkit/install-state.json"
    )
    first_project = tmp_path / "first-project"
    second_project = tmp_path / "video-lab"
    save_state(
        legacy,
        InstallState(
            schema_version=2,
            installer_version="0.2.1",
            mode=InstallMode.FULL,
            target_user="developer",
            release_id=None,
            source_revision=None,
            base_image_reference=None,
            base_manifest_digest=None,
            torch_image_reference=None,
            torch_manifest_digest=None,
            project_path=str(first_project.resolve()),
            current_stage=InstallStage.COMPLETE,
            completed_stage_input_digests={
                stage.value: "0" * 64 for stage in FULL_STAGE_ORDER
            },
            reboot_boot_id=None,
            created_at="2026-07-10T10:00:00Z",
            updated_at="2026-07-10T10:01:00Z",
            installer_source_revision="a" * 40,
            source_root=str(ROOT.resolve()),
        ),
    )
    legacy_before = legacy.read_bytes()

    second = run_install(
        home=home,
        fixture=fixture,
        state=None,
        project=second_project,
        mode="container",
    )

    selected = project_state_path(second_project, legacy)
    assert second.returncode == 0, second.stderr
    assert f"installer state (project): {selected}" in second.stdout
    assert legacy.read_bytes() == legacy_before
    assert selected.is_file()
    assert (second_project / "amd-ai-project.toml").is_file()
    calls = (fixture / "calls.log").read_text(encoding="utf-8").splitlines()
    assert calls == [
        "bootstrap",
        "container_host_check",
        "resolve_release",
        "pull_release",
        "verify_torch_image",
        "initialize_project",
        "verify_project",
    ]
