from pathlib import Path

import pytest

from amd_ai.project.dependencies import (
    DependencyError,
    lock_project_dependencies,
    project_lock_argv,
    render_torch_constraints,
    validate_project_lock,
)


def test_constraints_pin_complete_verified_stack():
    text = render_torch_constraints("profiles/torch/stable.requirements.lock")

    assert text.splitlines() == [
        "torch==2.9.1",
        "torchvision==0.24.0",
        "torchaudio==2.9.0",
        "triton==3.5.1",
    ]


def test_empty_project_input_produces_an_empty_lock(tmp_path):
    (tmp_path / "requirements.in").write_text(
        "# Project dependencies only.\n", encoding="utf-8"
    )
    (tmp_path / "torch-constraints.txt").write_text(
        render_torch_constraints("profiles/torch/stable.requirements.lock"),
        encoding="utf-8",
    )

    lock_project_dependencies(tmp_path)

    assert (tmp_path / "requirements.lock").read_text(encoding="utf-8") == ""


def test_project_lock_uses_parent_container_without_gpu_or_persistent_cache(tmp_path):
    parent = "sha256:" + "a" * 64

    argv = project_lock_argv(
        project_dir=tmp_path,
        base_image=parent,
        uid=1000,
        gid=1000,
        docker_prefix=("sudo", "-n", "docker"),
    )

    assert argv[:5] == ("sudo", "-n", "docker", "run", "--rm")
    assert "1000:1000" in argv
    assert f"type=bind,src={tmp_path},dst=/workspace" in argv
    assert "UV_CACHE_DIR=/tmp/uv-cache" in argv
    assert "/usr/local/bin/uv" in argv
    assert parent in argv
    assert "/dev/kfd" not in argv and "/dev/dri" not in argv
    assert not any("HF_HOME" in value for value in argv)


@pytest.mark.parametrize(
    "lock_text",
    [
        "torch==2.8.0 --hash=sha256:" + "a" * 64 + "\n",
        "torch @ https://example.com/torch.whl --hash=sha256:" + "a" * 64 + "\n",
    ],
)
def test_project_lock_rejects_replacement_or_direct_torch(lock_text):
    constraints = render_torch_constraints(
        Path("profiles/torch/stable.requirements.lock")
    )

    with pytest.raises(DependencyError):
        validate_project_lock(lock_text, constraints)
