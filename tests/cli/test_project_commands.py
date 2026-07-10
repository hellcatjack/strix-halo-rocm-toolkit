from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from amd_ai import cli
from amd_ai.project.run import ProjectImageMetadata
from amd_ai.runner import CommandResult
from tests.unit.project.fakes import FakeRunner, project_config, runtime_access


class FakeDocker:
    prefix = ("docker",)


def test_project_init_creates_digest_pinned_directory(tmp_path, monkeypatch):
    image = "rocm-pytorch:7.2.1-py3.12-torch2.9.1"
    inspect = ("docker", "image", "inspect", "--format", "{{.Id}}", image)
    image_id = "sha256:" + "a" * 64
    runner = FakeRunner(
        {inspect: CommandResult(inspect, 0, image_id + "\n", "")}
    )
    monkeypatch.setattr(cli.Docker, "detect", classmethod(lambda cls: FakeDocker()))
    monkeypatch.setattr(cli, "SubprocessRunner", lambda: runner)
    destination = tmp_path / "demo"

    code = cli.main(
        ["project-init", "demo", "--directory", str(destination)]
    )

    assert code == 0
    config = (destination / "amd-ai-project.toml").read_text(encoding="utf-8")
    assert f'base_image = "{image_id}"' in config
    assert (destination / "requirements.lock").read_text(encoding="utf-8") == ""


def test_project_run_dry_run_validates_and_prints_redacted_command(
    tmp_path,
    monkeypatch,
    capsys,
):
    config = project_config(
        tmp_path / "demo",
        environment=(("API_TOKEN", "secret with spaces"),),
    )
    captured = {}
    _patch_project_run(monkeypatch, config, captured)

    code = cli.main(["project-run", str(config.path.parent), "--dry-run"])

    output = capsys.readouterr().out
    assert code == 0
    assert "docker run" in output
    assert "--device /dev/kfd" in output
    assert "--ipc=private" in output
    assert "--privileged" not in output
    assert "secret with spaces" not in output
    assert "API_TOKEN=<redacted>" in output
    assert captured["build"] == {"force": False, "no_build": False}


def test_project_run_forwards_force_debug_and_shm_override(
    tmp_path,
    monkeypatch,
    capsys,
):
    config = project_config(tmp_path / "demo")
    captured = {}
    _patch_project_run(monkeypatch, config, captured)

    code = cli.main(
        [
            "project-run",
            str(config.path),
            "--build",
            "--debug",
            "--shm-size-gib",
            "12",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert captured["build"] == {"force": True, "no_build": False}
    assert "--shm-size 12g" in output
    assert "--cap-add SYS_PTRACE" in output
    assert "seccomp=unconfined" in output


def test_project_run_build_flags_are_mutually_exclusive():
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args(
            ["project-run", "demo", "--build", "--no-build"]
        )

    assert error.value.code == 2


def test_project_lock_updates_only_the_selected_project(tmp_path, monkeypatch):
    config = project_config(tmp_path / "demo")
    captured = {}
    monkeypatch.setattr(cli, "load_project_config", lambda path: config)
    monkeypatch.setattr(cli.Docker, "detect", classmethod(lambda cls: FakeDocker()))
    runner = FakeRunner()
    monkeypatch.setattr(cli, "SubprocessRunner", lambda: runner)
    monkeypatch.setattr(cli, "_runtime_identity", lambda: (1000, 1000))

    def fake_lock(**kwargs):
        captured.update(kwargs)
        return kwargs["project_dir"] / "requirements.lock"

    monkeypatch.setattr(cli, "lock_project_dependencies_in_container", fake_lock)

    code = cli.main(["project-lock", str(config.path.parent)])

    assert code == 0
    assert captured["project_dir"] == config.path.parent
    assert captured["base_image"] == config.base_image
    assert captured["docker_prefix"] == ("docker",)
    assert captured["runner"] is runner


def test_project_run_returns_live_container_exit_code(tmp_path, monkeypatch):
    config = project_config(tmp_path / "demo")
    captured = {}
    _patch_project_run(monkeypatch, config, captured)
    monkeypatch.setattr(cli, "_run_live", lambda argv: 23)

    code = cli.main(["project-run", str(config.path.parent), "--no-build"])

    assert code == 23
    assert captured["build"] == {"force": False, "no_build": True}


def test_project_command_wrappers_are_executable_and_dispatch_expected_command():
    for name in ("project-init", "project-lock", "project-run"):
        path = Path("bin") / name
        assert path.is_file()
        assert os.access(path, os.X_OK)
        assert f'_dispatch" {name} "$@"' in path.read_text(encoding="utf-8")


def _patch_project_run(monkeypatch, config, captured):
    monkeypatch.setattr(cli.Docker, "detect", classmethod(lambda cls: FakeDocker()))
    monkeypatch.setattr(cli, "SubprocessRunner", FakeRunner)
    monkeypatch.setattr(cli, "load_project_config", lambda path: config)

    def fake_build(**kwargs):
        captured["build"] = {
            "force": kwargs["force"],
            "no_build": kwargs["no_build"],
        }
        return SimpleNamespace(built=False)

    monkeypatch.setattr(cli, "build_or_reuse_project", fake_build)
    monkeypatch.setattr(
        cli,
        "inspect_project_image",
        lambda config, runner, docker_prefix: ProjectImageMetadata(
            image_id="sha256:" + "b" * 64,
            profile_id="rocm-7.2.1-py3.12-torch-2.9.1",
            profile_status="verified",
            rocm_version="7.2.1",
            python_version="3.12.11",
            torch_version="2.9.1",
            base_digest=config.base_digest,
            fingerprint="f" * 64,
        ),
    )
    monkeypatch.setattr(cli, "discover_gpu_access", runtime_access)
    monkeypatch.setattr(cli, "read_mem_total_kib", lambda: 131015488)
    monkeypatch.setattr(cli, "ensure_project_home", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_runtime_identity", lambda: (1000, 1000))
