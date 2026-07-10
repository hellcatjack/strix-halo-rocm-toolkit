from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from amd_ai import cli


def test_install_script_is_local_auditable_and_does_not_pipe_remote_shell() -> None:
    text = Path("install.sh").read_text(encoding="utf-8")

    assert "python3.12 -m amd_ai.installer.bootstrap" in text
    assert "curl |" not in text
    assert "wget |" not in text
    assert "eval " not in text


def test_install_noninteractive_arguments_parse() -> None:
    args = cli.build_parser().parse_args(
        [
            "install",
            "--mode",
            "container",
            "--non-interactive",
            "--project-dir",
            "/srv/comfy-lab",
            "--image-source",
            "pull",
        ]
    )

    assert args.command == "install"
    assert args.mode == "container"
    assert args.project_dir == Path("/srv/comfy-lab")


def test_unified_project_command_maps_to_existing_handler(monkeypatch) -> None:
    captured: dict[str, Path] = {}
    monkeypatch.setattr(
        cli,
        "_project_run",
        lambda args: captured.update(project=args.project) or 0,
    )

    assert cli.main(["project", "run", "/srv/demo", "--dry-run"]) == 0
    assert captured["project"] == Path("/srv/demo")


def test_repository_unified_wrapper_dispatches_without_command_rewrite() -> None:
    path = Path("bin/strix-halo-rocm")
    text = path.read_text(encoding="utf-8")

    assert os.access(path, os.X_OK)
    assert 'exec "$(dirname "$0")/_dispatch" "$@"' in text


def test_install_dispatch_constructs_workflow_once(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    class Result:
        exit_code = 0
        message = "complete"

    class Workflow:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def run(self):
            return Result()

    monkeypatch.setenv("AMD_AI_INSTALLER_SOURCE_REVISION", "a" * 40)
    monkeypatch.setattr(cli, "InstallerWorkflow", Workflow)
    monkeypatch.setattr(
        cli, "ProductionInstallerActions", lambda **kwargs: "actions"
    )

    code = cli.main(
        [
            "install",
            "--mode",
            "container",
            "--project-dir",
            str(tmp_path / "project"),
            "--image-source",
            "pull",
            "--source-root",
            str(Path.cwd()),
            "--state-path",
            str(tmp_path / "state.json"),
        ]
    )

    assert code == 0
    assert captured["actions"] == "actions"
    assert captured["installer_source_revision"] == "a" * 40
    assert captured["options"].state_path_explicit is True


def test_install_dispatch_marks_omitted_state_path_as_implicit(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    class Result:
        exit_code = 0
        message = "complete"

    class Workflow:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def run(self):
            return Result()

    monkeypatch.setenv("AMD_AI_INSTALLER_SOURCE_REVISION", "a" * 40)
    monkeypatch.setattr(
        cli.Docker,
        "detect",
        lambda: SimpleNamespace(prefix=("docker",)),
    )
    monkeypatch.setattr(cli, "InstallerWorkflow", Workflow)
    monkeypatch.setattr(
        cli, "ProductionInstallerActions", lambda **kwargs: "actions"
    )

    code = cli.main(
        [
            "install",
            "--mode",
            "container",
            "--project-dir",
            str(tmp_path / "project"),
            "--image-source",
            "pull",
            "--source-root",
            str(Path.cwd()),
        ]
    )

    assert code == 0
    assert captured["options"].state_path_explicit is False


def test_install_dispatch_uses_detected_sudo_docker_prefix(
    tmp_path: Path, monkeypatch
) -> None:
    action_arguments: dict[str, object] = {}

    class Result:
        exit_code = 0
        message = ""

    class Workflow:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def run(self):
            return Result()

    monkeypatch.setenv("AMD_AI_INSTALLER_SOURCE_REVISION", "a" * 40)
    monkeypatch.setattr(
        cli.Docker,
        "detect",
        lambda: SimpleNamespace(prefix=("sudo", "-n", "docker")),
    )
    monkeypatch.setattr(cli, "InstallerWorkflow", Workflow)
    monkeypatch.setattr(
        cli,
        "ProductionInstallerActions",
        lambda **kwargs: action_arguments.update(kwargs) or "actions",
    )

    code = cli.main(
        [
            "install",
            "--mode",
            "container",
            "--project-dir",
            str(tmp_path / "project"),
            "--image-source",
            "pull",
            "--source-root",
            str(Path.cwd()),
            "--state-path",
            str(tmp_path / "state.json"),
        ]
    )

    assert code == 0
    assert action_arguments["docker_prefix"] == ("sudo", "-n", "docker")
