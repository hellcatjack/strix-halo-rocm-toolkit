from __future__ import annotations

import json
import os
import stat

import pytest

from amd_ai.project.run import (
    ProjectRunError,
    UnverifiedImage,
    build_run_argv,
    ensure_project_home,
    inspect_project_image,
    redact_run_argv,
    require_profile_allowed,
)
from amd_ai.runner import CommandResult
from tests.unit.project.fakes import FakeRunner, project_config, runtime_access


def test_experimental_image_requires_exact_explicit_environment():
    with pytest.raises(UnverifiedImage):
        require_profile_allowed("experimental", {})
    with pytest.raises(UnverifiedImage):
        require_profile_allowed("experimental", {"ALLOW_UNVERIFIED": "true"})

    require_profile_allowed("experimental", {"ALLOW_UNVERIFIED": "1"})
    require_profile_allowed("verified", {})


def test_normal_run_is_unprivileged_and_uses_private_ipc(tmp_path):
    config = project_config(tmp_path)
    argv = build_run_argv(
        config=config,
        access=runtime_access(),
        uid=1000,
        gid=1000,
        shm_gib=16,
        environment={},
        terminal=False,
    )

    assert argv[:3] == ("docker", "run", "--rm")
    assert "--privileged" not in argv
    assert "--ipc=host" not in argv
    assert "--ipc=private" in argv
    assert "SYS_PTRACE" not in argv
    assert "--tty" not in argv and "--interactive" not in argv
    assert tuple(argv[argv.index("--shm-size") : argv.index("--shm-size") + 2]) == (
        "--shm-size",
        "16g",
    )
    assert ("--device", "/dev/kfd") == tuple(
        argv[argv.index("--device") : argv.index("--device") + 2]
    )
    assert argv.count("--device") == 2
    assert "109" in argv and "110" in argv
    assert "1000:1000" in argv
    assert "HOME=/workspace/.amd-ai/home" in argv
    assert f"type=bind,src={tmp_path},dst=/workspace" in argv
    assert not any("HF_HOME" in value or "HF_HUB_CACHE" in value for value in argv)
    assert argv[-3:] == (config.image, "python", "app.py")


def test_debug_adds_only_ptrace_and_seccomp(tmp_path):
    argv = build_run_argv(
        config=project_config(tmp_path, debug=True),
        access=runtime_access(),
        uid=1000,
        gid=1000,
        shm_gib=16,
        environment={},
        terminal=True,
    )

    assert "SYS_PTRACE" in argv
    assert "seccomp=unconfined" in argv
    assert "--tty" in argv and "--interactive" in argv
    assert "--privileged" not in argv
    assert argv.count("--cap-add") == 1
    assert argv.count("--security-opt") == 1


def test_explicit_environment_is_passed_and_secrets_are_redacted(tmp_path):
    config = project_config(
        tmp_path,
        environment=(
            ("API_TOKEN", "secret with spaces"),
            ("HF_HOME", "/workspace/cache/huggingface"),
        ),
    )
    argv = build_run_argv(
        config=config,
        access=runtime_access(),
        uid=1000,
        gid=1000,
        shm_gib=8,
        environment={"ALLOW_UNVERIFIED": "1", "IGNORED_HOST_VALUE": "x"},
        terminal=False,
    )

    assert "API_TOKEN=secret with spaces" in argv
    assert "HF_HOME=/workspace/cache/huggingface" in argv
    assert "ALLOW_UNVERIFIED=1" in argv
    assert not any("IGNORED_HOST_VALUE" in value for value in argv)
    redacted = redact_run_argv(argv)
    assert "API_TOKEN=<redacted>" in redacted
    assert "API_TOKEN=secret with spaces" not in redacted
    assert "HF_HOME=/workspace/cache/huggingface" in redacted


def test_image_inspection_requires_inherited_and_project_labels(tmp_path):
    config = project_config(tmp_path)
    inspect_args = ("docker", "image", "inspect", config.image)
    labels = {
        "org.amd-ai.profile.id": "rocm-7.2.1-py3.12-torch-2.9.1",
        "org.amd-ai.rocm.version": "7.2.1",
        "org.amd-ai.python.version": "3.12.11",
        "org.amd-ai.torch.version": "2.9.1",
        "org.amd-ai.base.digest": config.base_digest,
        "org.amd-ai.project.fingerprint": "f" * 64,
    }
    payload = [{"Id": "sha256:" + "b" * 64, "Config": {"Labels": labels}}]
    runner = FakeRunner(
        {
            inspect_args: CommandResult(
                inspect_args,
                0,
                json.dumps(payload),
                "",
            )
        }
    )

    metadata = inspect_project_image(config, runner)

    assert metadata.profile_status == "experimental"
    assert metadata.profile_id == "rocm-7.2.1-py3.12-torch-2.9.1"
    assert metadata.base_digest == config.base_digest
    assert metadata.fingerprint == "f" * 64


def test_image_inspection_rejects_wrong_base_digest(tmp_path):
    config = project_config(tmp_path)
    inspect_args = ("docker", "image", "inspect", config.image)
    labels = {
        "org.amd-ai.profile.id": "stable",
        "org.amd-ai.profile.status": "verified",
        "org.amd-ai.rocm.version": "7.2.1",
        "org.amd-ai.python.version": "3.12.11",
        "org.amd-ai.torch.version": "2.9.1",
        "org.amd-ai.base.digest": "sha256:" + "c" * 64,
        "org.amd-ai.project.fingerprint": "f" * 64,
    }
    payload = [{"Id": "sha256:" + "b" * 64, "Config": {"Labels": labels}}]
    runner = FakeRunner(
        {inspect_args: CommandResult(inspect_args, 0, json.dumps(payload), "")}
    )

    with pytest.raises(ProjectRunError, match="base digest"):
        inspect_project_image(config, runner)


def test_project_home_is_private_and_owned_by_runtime_user(tmp_path):
    config = project_config(tmp_path)

    home = ensure_project_home(
        config.path.parent,
        uid=os.getuid(),
        gid=os.getgid(),
    )

    metadata = home.stat()
    assert home == tmp_path / ".amd-ai/home"
    assert stat.S_IMODE(metadata.st_mode) == 0o700
    assert (metadata.st_uid, metadata.st_gid) == (os.getuid(), os.getgid())


def test_project_home_rejects_symlinked_control_directory(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / ".amd-ai").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ProjectRunError, match="symbolic link"):
        ensure_project_home(project, uid=os.getuid(), gid=os.getgid())
