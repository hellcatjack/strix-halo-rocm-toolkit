from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

from amd_ai.image.build import BuildError, Docker
from amd_ai.project.build import build_or_reuse_project
from amd_ai.project.config import ProjectConfig, load_project_config
from amd_ai.project.dependencies import validate_project_lock
from amd_ai.project.init import STABLE_IMAGE, initialize_project
from amd_ai.runner import SubprocessRunner


FIXTURES = Path("tests/fixtures/projects")


@dataclass(frozen=True)
class BuiltProjects:
    docker_prefix: tuple[str, ...]
    parent: dict[str, object]
    projects: tuple[tuple[str, ProjectConfig, dict[str, object]], ...]


@pytest.fixture(scope="module")
def built_projects(tmp_path_factory) -> BuiltProjects:
    try:
        docker = Docker.detect()
    except BuildError:
        pytest.skip("Docker daemon or buildx is unavailable to the user and sudo -n")
    runner = SubprocessRunner()
    parent_result = runner.run(
        [*docker.prefix, "image", "inspect", STABLE_IMAGE],
        check=False,
    )
    assert parent_result.returncode == 0, (
        f"required stable parent image is missing: {parent_result.stderr}"
    )
    parent = json.loads(parent_result.stdout)[0]
    root = tmp_path_factory.mktemp("project-layers")
    built: list[tuple[str, ProjectConfig, dict[str, object]]] = []
    configs: list[ProjectConfig] = []
    run_id = uuid.uuid4().hex[:8]
    try:
        for fixture_name in ("alpha", "beta"):
            name = f"layer-{fixture_name}-{os.getpid()}-{run_id}"
            destination = root / fixture_name
            initialize_project(
                name=name,
                destination=destination,
                base_profile="stable",
                runner=runner,
                docker_prefix=docker.prefix,
            )
            fixture = FIXTURES / fixture_name
            input_text = (fixture / "requirements.in").read_text(encoding="utf-8")
            assert "torch" not in input_text.lower()
            constraints = (destination / "torch-constraints.txt").read_text(
                encoding="utf-8"
            )
            assert (
                fixture / "torch-constraints.txt"
            ).read_text(encoding="utf-8") == constraints
            lock_text = (fixture / "requirements.lock").read_text(encoding="utf-8")
            validate_project_lock(lock_text, constraints)
            shutil.copyfile(fixture / "requirements.in", destination / "requirements.in")
            shutil.copyfile(
                fixture / "requirements.lock",
                destination / "requirements.lock",
            )
            config = load_project_config(destination / "amd-ai-project.toml")
            configs.append(config)
            build_or_reuse_project(
                config=config,
                runner=runner,
                force=True,
                no_build=False,
                docker_prefix=docker.prefix,
            )
            inspect = runner.run(
                [*docker.prefix, "image", "inspect", config.image],
                check=False,
            )
            assert inspect.returncode == 0, inspect.stderr
            built.append((fixture_name, config, json.loads(inspect.stdout)[0]))
        yield BuiltProjects(docker.prefix, parent, tuple(built))
    finally:
        for config in configs:
            subprocess.run(
                (*docker.prefix, "image", "rm", "--force", config.image),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )


@pytest.mark.container
def test_projects_reuse_the_complete_parent_layer_prefix(built_projects):
    parent_layers = built_projects.parent["RootFS"]["Layers"]
    assert parent_layers
    for _, config, image in built_projects.projects:
        layers = image["RootFS"]["Layers"]
        assert layers[: len(parent_layers)] == parent_layers, config.image
        labels = image["Config"]["Labels"]
        assert labels["org.amd-ai.base.digest"] == config.base_digest
        assert labels["org.amd-ai.profile.status"] == "verified"


@pytest.mark.container
def test_projects_keep_the_parent_torch_distribution_unchanged(built_projects):
    parent_metadata = _torch_metadata(
        built_projects.docker_prefix,
        STABLE_IMAGE,
    )
    for fixture_name, config, _ in built_projects.projects:
        verify = _completed(
            built_projects.docker_prefix,
            (
                "run",
                "--rm",
                "--entrypoint",
                "/opt/venv/bin/python",
                config.image,
                "/opt/amd-ai/torch-manifest.py",
                "verify",
                "/opt/amd-ai/torch-manifest.json",
            ),
        )
        assert verify.returncode == 0, verify.stderr or verify.stdout
        extra_name = {"alpha": "safetensors", "beta": "einops"}[fixture_name]
        package = _completed(
            built_projects.docker_prefix,
            (
                "run",
                "--rm",
                "--entrypoint",
                "/opt/venv/bin/python",
                config.image,
                "-c",
                "import importlib.metadata as m, json; "
                f"print(json.dumps({{'torch': m.version('torch'), "
                f"'extra': m.version('{extra_name}'), "
                "'root': str(m.distribution('torch').locate_file(''))}))",
            ),
        )
        assert package.returncode == 0, package.stderr or package.stdout
        metadata = json.loads(package.stdout)
        assert metadata["torch"] == parent_metadata["torch"]
        assert metadata["root"] == parent_metadata["root"]
        assert metadata["extra"] == {"alpha": "0.5.3", "beta": "0.8.1"}[
            fixture_name
        ]


def _torch_metadata(prefix: tuple[str, ...], image: str) -> dict[str, str]:
    result = _completed(
        prefix,
        (
            "run",
            "--rm",
            "--entrypoint",
            "/opt/venv/bin/python",
            image,
            "-c",
            "import importlib.metadata as m, json; "
            "print(json.dumps({'torch': m.version('torch'), "
            "'root': str(m.distribution('torch').locate_file(''))}))",
        ),
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


def _completed(
    prefix: tuple[str, ...],
    args: tuple[str, ...],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (*prefix, *args),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
