import os
from pathlib import Path
from types import SimpleNamespace

from amd_ai import cli
from tests.unit.project.fakes import FakeRunner, runtime_access


def test_image_build_parser_defaults_to_repository_stable_profile():
    args = cli.build_parser().parse_args(["image-build", "rocm-pytorch"])

    assert args.profile == Path("profiles/torch/stable.env")
    assert args.allow_experimental is False


def test_prune_defaults_to_preview_without_mutating(monkeypatch):
    captured = {}

    def fake_prune(**kwargs):
        captured.update(kwargs)
        return ()

    monkeypatch.setattr(cli, "prune_images", fake_prune)

    code = cli.main(["image-build", "prune"])

    assert code == 0
    assert captured["apply"] is False
    assert captured["older_than_hours"] == 168


def test_container_check_wrapper_forwards_image_mode_and_metadata(monkeypatch):
    captured = {}

    def fake_check(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "run_image_check", fake_check)

    code = cli.main(
        [
            "container-check",
            "--image",
            "rocm-pytorch:test",
            "--mode",
            "torch",
            "--metadata-only",
            "--json",
            "-",
        ]
    )

    assert code == 0
    assert captured["image"] == "rocm-pytorch:test"
    assert captured["mode"] == "torch"
    assert captured["metadata_only"] is True
    assert captured["runtime"] is False


def test_container_check_stable_suite_forwards_profile_and_report_path(
    tmp_path,
    monkeypatch,
):
    captured = {}

    class FakeDocker:
        prefix = ("docker",)

    monkeypatch.setattr(
        cli.Docker,
        "detect",
        classmethod(lambda cls: FakeDocker()),
    )
    runner = FakeRunner()
    monkeypatch.setattr(cli, "SubprocessRunner", lambda: runner)
    monkeypatch.setattr(cli, "discover_gpu_access", runtime_access)
    monkeypatch.setattr(cli, "_runtime_identity", lambda: (1000, 1000))

    def fake_run_profile(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(status="pass", to_dict=lambda: {"status": "pass"})

    monkeypatch.setattr(cli, "run_qualification_profile", fake_run_profile)
    output = tmp_path / "qualification.json"

    code = cli.main(
        [
            "container-check",
            "--suite",
            "stable",
            "--profile",
            "profiles/qualification/stable.toml",
            "--json",
            str(output),
        ]
    )

    assert code == 0
    assert captured["profile_path"] == Path("profiles/qualification/stable.toml")
    assert captured["output_path"] == output
    assert captured["runner"] is runner
    assert captured["docker_prefix"] == ("docker",)
    assert captured["gids"] == (109, 110)


def test_gpu_release_wrapper_forwards_qualification_and_image(tmp_path, monkeypatch):
    captured = {}

    def fake_release(argv):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(cli, "qualification_release_main", fake_release)
    qualification = tmp_path / "qualification.json"
    output = tmp_path / "releases"

    code = cli.main(
        [
            "gpu-release",
            "--qualification",
            str(qualification),
            "--image",
            "rocm-pytorch:stable",
            "--output-dir",
            str(output),
        ]
    )

    assert code == 0
    assert captured["argv"] == [
        "--qualification",
        str(qualification),
        "--image",
        "rocm-pytorch:stable",
        "--output-dir",
        str(output),
    ]


def test_image_command_wrappers_are_executable_and_dispatch_expected_command():
    for name in ("image-build", "container-check", "gpu-release"):
        path = Path("bin") / name
        assert path.is_file()
        assert os.access(path, os.X_OK)
        assert f'_dispatch" {name} "$@"' in path.read_text(encoding="utf-8")
