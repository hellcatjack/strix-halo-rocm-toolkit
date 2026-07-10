from pathlib import Path

import pytest

from amd_ai.project.config import ConfigError, load_project_config


def test_default_template_has_no_mounts_or_cache_policy():
    config = load_project_config(Path("templates/project/amd-ai-project.toml"))

    assert config.name == "demo"
    assert config.base_profile == "stable"
    assert config.mounts == ()
    assert config.environment == ()
    assert config.command == ("bash",)
    assert config.shm_size_gib is None


def test_reserved_mount_target_is_rejected(tmp_path):
    path = write_config(
        tmp_path,
        '[[mounts]]\nsource="/data/a"\ntarget="/opt/venv"\nread_only=true\n',
    )

    with pytest.raises(ConfigError, match="reserved"):
        load_project_config(path)


def test_duplicate_or_non_normalized_mount_target_is_rejected(tmp_path):
    duplicate = write_config(
        tmp_path,
        '[[mounts]]\nsource="a"\ntarget="/models"\nread_only=true\n'
        '[[mounts]]\nsource="b"\ntarget="/models"\nread_only=false\n',
    )
    with pytest.raises(ConfigError, match="duplicate"):
        load_project_config(duplicate)

    non_normalized = write_config(
        tmp_path,
        '[[mounts]]\nsource="a"\ntarget="/data/../models"\nread_only=true\n',
    )
    with pytest.raises(ConfigError, match="normalized"):
        load_project_config(non_normalized)


def test_explicit_environment_and_relative_mount_are_preserved(tmp_path):
    path = write_config(
        tmp_path,
        '[[mounts]]\nsource="models"\ntarget="/models"\nread_only=true\n'
        '[environment]\nHF_HOME="/workspace/.cache/huggingface"\nA_FLAG="1"\n',
    )

    config = load_project_config(path)

    assert config.mounts[0].source == (tmp_path / "models").resolve()
    assert config.mounts[0].read_only is True
    assert config.environment == (
        ("A_FLAG", "1"),
        ("HF_HOME", "/workspace/.cache/huggingface"),
    )


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ('[project]\nunknown="value"\n', "unknown"),
        ('[environment]\nPATH="/tmp"\n', "reserved"),
    ],
)
def test_unknown_project_key_or_reserved_environment_is_rejected(
    tmp_path, extra, message
):
    if extra.startswith("[project]"):
        path = tmp_path / "amd-ai-project.toml"
        path.write_text(base_config() + 'unknown="value"\n', encoding="utf-8")
    else:
        path = write_config(tmp_path, extra)

    with pytest.raises(ConfigError, match=message):
        load_project_config(path)


def test_parent_ids_must_match(tmp_path):
    text = base_config().replace("b" * 64, "c" * 64, 1)
    path = tmp_path / "amd-ai-project.toml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="must match"):
        load_project_config(path)


def base_config():
    digest = "b" * 64
    return (
        "[project]\n"
        'name="demo"\n'
        'base_profile="stable"\n'
        'image="demo:runtime"\n'
        f'base_image="sha256:{digest}"\n'
        f'base_digest="sha256:{digest}"\n'
        'command=["bash"]\n'
        "debug=false\n"
    )


def write_config(tmp_path, extra):
    path = tmp_path / "amd-ai-project.toml"
    path.write_text(base_config() + extra, encoding="utf-8")
    return path
