import os
from pathlib import Path

from amd_ai import cli


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


def test_image_command_wrappers_are_executable_and_dispatch_expected_command():
    for name in ("image-build", "container-check"):
        path = Path("bin") / name
        assert path.is_file()
        assert os.access(path, os.X_OK)
        assert f'_dispatch" {name} "$@"' in path.read_text(encoding="utf-8")
