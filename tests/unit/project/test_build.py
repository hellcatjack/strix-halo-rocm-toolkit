from pathlib import Path

import pytest

from amd_ai.project.build import (
    ParentImageMetadata,
    ProjectBuildError,
    build_context_fingerprint,
    project_manifest_argv,
    project_build_argv,
    project_parent_alias,
    remove_exact_project_image,
    validate_dockerignore,
    validate_project_image_contract,
)
from amd_ai.runner import CommandResult
from tests.unit.project.fakes import FakeRunner


def test_fingerprint_changes_with_lock_not_ignored_models(tmp_path):
    lock = tmp_path / "requirements.lock"
    lock.write_text("alpha==1.0 --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")
    (tmp_path / "models").mkdir()
    model = tmp_path / "models/model.bin"
    model.write_bytes(b"large")

    first = build_context_fingerprint(tmp_path)
    model.write_bytes(b"changed")
    assert build_context_fingerprint(tmp_path) == first
    lock.write_text("alpha==1.1 --hash=sha256:" + "b" * 64 + "\n", encoding="utf-8")
    assert build_context_fingerprint(tmp_path) != first


def test_fingerprint_tracks_regular_source_mode_and_content(tmp_path):
    source = tmp_path / "app.py"
    source.write_text("print('one')\n", encoding="utf-8")
    first = build_context_fingerprint(tmp_path)
    source.chmod(0o755)
    second = build_context_fingerprint(tmp_path)
    source.write_text("print('two')\n", encoding="utf-8")

    assert second != first
    assert build_context_fingerprint(tmp_path) != second


def test_build_uses_content_addressed_parent_alias_and_labels():
    parent = "sha256:" + "a" * 64
    argv = project_build_argv(
        context=Path("projects/demo"),
        image="demo:runtime",
        base_image=parent,
        base_digest=parent,
        profile_id="rocm-7.2.1-py3.12-torch-2.9.1",
        profile_status="verified",
        fingerprint="f" * 64,
    )

    alias = "amd-ai-local/project-base:" + "a" * 64
    assert project_parent_alias(parent) == alias
    assert f"BASE_IMAGE={alias}" in argv
    assert "PROFILE_STATUS=verified" in argv
    assert "org.amd-ai.project.fingerprint=" + "f" * 64 in argv
    assert "org.amd-ai.base.digest=" + parent in argv
    assert "--load" in argv


def test_dockerignore_must_preserve_mandatory_storage_exclusions(tmp_path):
    dockerignore = tmp_path / ".dockerignore"
    dockerignore.write_text(
        ".git\n.venv\n.cache\n.amd-ai\nmodels\ninput\noutput\nreports\n"
        "__pycache__\n*.pyc\n",
        encoding="utf-8",
    )
    validate_dockerignore(tmp_path)

    dockerignore.write_text(dockerignore.read_text() + "!.venv\n", encoding="utf-8")
    with pytest.raises(ProjectBuildError, match="negation"):
        validate_dockerignore(tmp_path)
    dockerignore.unlink()
    with pytest.raises(ProjectBuildError, match="missing"):
        validate_dockerignore(tmp_path)


def test_project_image_cannot_override_experimental_parent_contract():
    parent = ParentImageMetadata(
        image_id="sha256:" + "a" * 64,
        profile_id="custom",
        profile_status="experimental",
        rocm_version="7.2.1",
        python_version="3.12",
        torch_version="2.9.1",
        layers=("layer-a", "layer-b"),
    )
    record = {
        "RootFS": {"Layers": ["layer-a", "layer-b", "project-layer"]},
        "Config": {
            "User": "1000:1000",
            "WorkingDir": "/workspace",
            "Entrypoint": ["/usr/local/bin/project-entrypoint"],
            "Env": [
                "AMD_AI_PROFILE_ID=custom",
                "AMD_AI_PROFILE_STATUS=verified",
            ],
            "Labels": {
                "org.amd-ai.profile.id": "custom",
                "org.amd-ai.profile.status": "verified",
                "org.amd-ai.rocm.version": "7.2.1",
                "org.amd-ai.python.version": "3.12",
                "org.amd-ai.torch.version": "2.9.1",
            },
        },
    }

    with pytest.raises(ProjectBuildError, match="profile status"):
        validate_project_image_contract(record, parent)


def test_project_image_must_reuse_parent_layers_and_manifest_command():
    parent_id = "sha256:" + "a" * 64
    argv = project_manifest_argv(
        "demo:runtime",
        docker_prefix=("sudo", "-n", "docker"),
    )

    assert argv[:5] == ("sudo", "-n", "docker", "run", "--rm")
    assert "/opt/amd-ai/torch-manifest.py" in argv
    assert "verify" in argv
    assert parent_id not in argv


def test_project_image_removal_accepts_only_exact_id() -> None:
    image_id = "sha256:" + "a" * 64
    command = ("docker", "image", "rm", image_id)
    runner = FakeRunner(
        {command: CommandResult(command, 0, image_id + "\n", "")}
    )

    remove_exact_project_image(image_id, runner=runner)

    assert runner.calls == [command]
    with pytest.raises(ProjectBuildError):
        remove_exact_project_image("demo:latest", runner=runner)
