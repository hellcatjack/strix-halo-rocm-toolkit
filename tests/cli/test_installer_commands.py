from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from amd_ai import cli
from amd_ai.installer.progress import ProgressMode


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


def test_install_progress_modes_are_mutually_exclusive() -> None:
    verbose = cli.build_parser().parse_args(["install", "--verbose"])
    quiet = cli.build_parser().parse_args(["install", "--quiet"])

    assert verbose.verbose is True and verbose.quiet is False
    assert quiet.quiet is True and quiet.verbose is False
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["install", "--verbose", "--quiet"]
        )


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


@pytest.mark.parametrize(
    ("flag", "expected_mode"),
    [
        ("--verbose", ProgressMode.VERBOSE),
        ("--quiet", ProgressMode.QUIET),
    ],
)
def test_install_dispatch_shares_one_progress_reporter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    expected_mode: ProgressMode,
) -> None:
    captured: dict[str, object] = {}

    class Progress:
        def __init__(self, *, mode: ProgressMode) -> None:
            self.mode = mode
            captured["progress"] = self

    class Runner:
        def __init__(self, *, observer) -> None:
            self.observer = observer
            captured["runner"] = self

    class Actions:
        def __init__(self, **kwargs: object) -> None:
            captured["action_arguments"] = kwargs

    class Result:
        exit_code = 0
        message = ""

    class Workflow:
        def __init__(self, **kwargs: object) -> None:
            captured["workflow_arguments"] = kwargs

        def run(self):
            return Result()

    monkeypatch.setenv("AMD_AI_INSTALLER_SOURCE_REVISION", "a" * 40)
    monkeypatch.setattr(cli, "InstallerProgress", Progress, raising=False)
    monkeypatch.setattr(cli, "SubprocessRunner", Runner)
    monkeypatch.setattr(cli, "ProductionInstallerActions", Actions)
    monkeypatch.setattr(cli, "InstallerWorkflow", Workflow)
    monkeypatch.setattr(
        cli.Docker,
        "detect",
        lambda: SimpleNamespace(prefix=("docker",)),
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
            flag,
        ]
    )

    assert code == 0
    progress = captured["progress"]
    assert progress.mode is expected_mode
    runner = captured["runner"]
    assert runner.observer is progress
    action_arguments = captured["action_arguments"]
    assert action_arguments["runner"] is runner
    assert action_arguments["command_observer"] is progress
    assert action_arguments["progress_mode"] is expected_mode
    workflow_arguments = captured["workflow_arguments"]
    assert workflow_arguments["progress"] is progress


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
