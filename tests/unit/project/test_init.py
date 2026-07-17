import os

import pytest

from amd_ai.project.init import ProjectInitError, initialize_project
from tests.unit.host.fakes import FakeRunner


def test_scaffold_is_pinned_and_has_no_implicit_shared_storage(tmp_path):
    image = "rocm-pytorch:7.2.1-py3.12-torch2.9.1"
    digest = "sha256:" + "a" * 64
    runner = FakeRunner.image_digest(image, digest)

    project = initialize_project(
        name="video-lab",
        destination=tmp_path / "video-lab",
        base_profile="stable",
        runner=runner,
    )

    config = (project / "amd-ai-project.toml").read_text(encoding="utf-8")
    assert 'name = "video-lab"' in config
    assert 'image = "video-lab:runtime"' in config
    assert f'base_image = "{digest}"' in config
    assert f'base_digest = "{digest}"' in config
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in project.iterdir()
        if path.is_file()
    )
    assert "HF_HOME" not in combined
    assert "ComfyUI" not in combined
    assert "[[mounts]]" not in config
    assert not (project / "models").exists()
    assert not (project / "input").exists()
    assert not (project / "output").exists()
    assert os.access(project / "project-entrypoint", os.X_OK)


def test_scaffold_keeps_containerd_local_id_separate_from_config_digest(tmp_path):
    image = "rocm-pytorch:7.2.1-py3.12-torch2.9.1"
    manifest_digest = "sha256:" + "a" * 64
    config_digest = "sha256:" + "b" * 64
    runner = FakeRunner.image_digest(image, manifest_digest)

    project = initialize_project(
        name="video-lab",
        destination=tmp_path / "video-lab",
        base_profile="stable",
        base_config_digest=config_digest,
        base_manifest_digest=manifest_digest,
        runner=runner,
    )

    config = (project / "amd-ai-project.toml").read_text(encoding="utf-8")
    assert f'base_image = "{manifest_digest}"' in config
    assert f'base_manifest_digest = "{manifest_digest}"' in config
    assert f'base_digest = "{config_digest}"' in config


def test_scaffold_rejects_unrelated_local_manifest_and_config(tmp_path):
    image = "rocm-pytorch:7.2.1-py3.12-torch2.9.1"
    runner = FakeRunner.image_digest(image, "sha256:" + "c" * 64)

    with pytest.raises(ProjectInitError, match="identity"):
        initialize_project(
            name="video-lab",
            destination=tmp_path / "video-lab",
            base_profile="stable",
            base_config_digest="sha256:" + "b" * 64,
            base_manifest_digest="sha256:" + "a" * 64,
            runner=runner,
        )


def test_nonempty_destination_is_refused(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "user-file").write_text("keep", encoding="utf-8")
    runner = FakeRunner.image_digest(
        "rocm-pytorch:7.2.1-py3.12-torch2.9.1",
        "sha256:" + "a" * 64,
    )

    with pytest.raises(ProjectInitError, match="not empty"):
        initialize_project(
            name="demo",
            destination=destination,
            base_profile="stable",
            runner=runner,
        )

    assert (destination / "user-file").read_text(encoding="utf-8") == "keep"
