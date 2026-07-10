from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from amd_ai.image.build import BuildError, Docker
from amd_ai.project.build import build_or_reuse_project
from amd_ai.project.config import ProjectConfig, load_project_config
from amd_ai.project.init import STABLE_IMAGE
from amd_ai.runner import SubprocessRunner


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_PREFIXES = ("torch-", "torchvision-", "torchaudio-", "triton-")
pytestmark = pytest.mark.container


@dataclass(frozen=True)
class OverlayProject:
    path: Path
    config: ProjectConfig
    docker_prefix: tuple[str, ...]

    def run(self, *command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self._docker_argv(command),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def start_detached(self, name: str, *command: str) -> str:
        argv = list(self._docker_argv(command))
        argv[argv.index("run") + 1 : argv.index("run") + 2] = [
            "--detach",
            "--name",
            name,
        ]
        result = subprocess.run(
            argv,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert result.returncode == 0, result.stderr or result.stdout
        return result.stdout.strip()

    def current_target(self) -> str:
        return os.readlink(self.path / ".amd-ai/current")

    def overlay_entries(self) -> tuple[str, ...]:
        site_packages = self.path / ".amd-ai/current/site-packages"
        return tuple(sorted(path.name for path in site_packages.iterdir()))

    def _docker_argv(self, command: tuple[str, ...]) -> tuple[str, ...]:
        return (
            *self.docker_prefix,
            "run",
            "--rm",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--workdir",
            "/workspace",
            "--entrypoint",
            "/usr/bin/env",
            "--add-host",
            "host.docker.internal:host-gateway",
            "--env",
            "HOME=/workspace/.amd-ai/home",
            "--env",
            "PYTHONNOUSERSITE=1",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "AMD_AI_OVERLAY=/workspace/.amd-ai/current/site-packages",
            "--env",
            "PYTHONPATH=/workspace/.amd-ai/current/site-packages:/opt/amd-ai/src",
            "--env",
            f"AMD_AI_PARENT_CONFIG_DIGEST={self.config.base_digest}",
            "--mount",
            f"type=bind,src={self.path},dst=/workspace",
            self.config.image,
            *command,
        )


@pytest.fixture(scope="module")
def project_factory(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[str], OverlayProject]:
    try:
        docker = Docker.detect()
    except BuildError:
        pytest.skip("Docker daemon or buildx is unavailable")
    inspect = subprocess.run(
        (*docker.prefix, "image", "inspect", STABLE_IMAGE),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if inspect.returncode != 0:
        pytest.skip(f"required stable image is missing: {inspect.stderr}")

    root = tmp_path_factory.mktemp("readonly-overlay")
    runner = SubprocessRunner()
    projects: list[OverlayProject] = []
    counter = 0

    def create(label: str) -> OverlayProject:
        nonlocal counter
        counter += 1
        name = f"overlay-{label}-{os.getpid()}-{counter}"
        destination = root / name
        result = subprocess.run(
            (
                str(REPOSITORY_ROOT / "bin/project-init"),
                name,
                "--directory",
                str(destination),
            ),
            cwd=REPOSITORY_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert result.returncode == 0, result.stderr or result.stdout
        config = load_project_config(destination / "amd-ai-project.toml")
        build_or_reuse_project(
            config=config,
            runner=runner,
            force=True,
            no_build=False,
            docker_prefix=docker.prefix,
        )
        project = OverlayProject(destination, config, docker.prefix)
        projects.append(project)
        return project

    try:
        yield create
    finally:
        for project in projects:
            subprocess.run(
                (*docker.prefix, "image", "rm", "--force", project.config.image),
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )


def test_plain_pip_install_persists_without_copying_torch(project_factory) -> None:
    project = project_factory("persist")

    first = project.run("pip", "install", "six==1.17.0")
    second = project.run(
        "python",
        "-c",
        "import six, torch; print(six.__version__); print(torch.__version__)",
    )

    assert first.returncode == 0, first.stderr or first.stdout
    assert second.returncode == 0, second.stderr or second.stdout
    assert "1.17.0" in second.stdout
    assert not any(
        _is_protected_entry(name)
        for name in project.overlay_entries()
    )


def test_compatible_torch_requests_use_only_the_parent(project_factory) -> None:
    project = project_factory("compatible")
    requirements = project.path / "compatible.txt"
    requirements.write_text("torch>=2.9\n", encoding="utf-8")

    direct = project.run("pip", "install", "torch")
    from_file = project.run("pip", "install", "-r", "compatible.txt")

    assert direct.returncode == 0, direct.stderr or direct.stdout
    assert from_file.returncode == 0, from_file.stderr or from_file.stdout
    assert "already satisfied by verified parent" in direct.stdout
    assert not any(
        _is_protected_entry(name)
        for name in project.overlay_entries()
    )


@pytest.mark.parametrize(
    "command",
    (
        ("pip", "install", "torch==2.8.0"),
        ("pip", "install", "--target", "/tmp/site", "six"),
        ("pip", "uninstall", "-y", "torch"),
    ),
)
def test_protected_or_escaping_pip_operations_return_two(
    project_factory, command: tuple[str, ...]
) -> None:
    project = project_factory("forbidden-" + command[1])

    result = project.run(*command)

    assert result.returncode == 2, result.stderr or result.stdout


def test_killed_resolver_does_not_activate_partial_generation(
    project_factory,
) -> None:
    project = project_factory("interrupt")
    initialized = project.run("pip", "install", "torch")
    assert initialized.returncode == 0, initialized.stderr or initialized.stdout
    current_before = project.current_target()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("0.0.0.0", 0))
    listener.listen()
    stop = threading.Event()
    thread = threading.Thread(
        target=_hold_tls_connections,
        args=(listener, stop),
        daemon=True,
    )
    thread.start()
    container_name = "overlay-interrupt-" + uuid.uuid4().hex[:10]
    try:
        project.start_detached(
            container_name,
            "pip",
            "install",
            "amd-ai-never-resolves",
            "--index-url",
            f"https://host.docker.internal:{listener.getsockname()[1]}/simple",
        )
        transaction_root = project.path / ".amd-ai/transactions"
        assert _wait_for(lambda: transaction_root.is_dir() and any(transaction_root.iterdir()))
        killed = subprocess.run(
            (*project.docker_prefix, "kill", container_name),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert killed.returncode == 0, killed.stderr or killed.stdout
    finally:
        stop.set()
        listener.close()
        thread.join(timeout=2)
        subprocess.run(
            (*project.docker_prefix, "rm", "--force", container_name),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    assert project.current_target() == current_before


def test_changed_artifact_hash_blocks_next_transaction(project_factory) -> None:
    project = project_factory("hash")
    installed = project.run("pip", "install", "six==1.17.0")
    assert installed.returncode == 0, installed.stderr or installed.stdout
    current_before = project.current_target()
    wheels = tuple((project.path / ".amd-ai/artifacts/sha256").glob("*/*.whl"))
    assert wheels
    wheels[0].chmod(0o644)
    wheels[0].write_bytes(b"changed")

    result = project.run("pip", "install", "idna==3.10")

    assert result.returncode == 2, result.stderr or result.stdout
    assert "repair" in result.stderr
    assert project.current_target() == current_before


def test_healthy_generation_retention_is_bounded(project_factory) -> None:
    project = project_factory("retention")

    for command in (
        ("pip", "install", "torch"),
        _mark_healthy_command(),
        ("pip", "install", "six==1.16.0"),
        _mark_healthy_command(),
        ("pip", "install", "six==1.17.0"),
        _mark_healthy_command(),
    ):
        result = project.run(*command)
        assert result.returncode == 0, result.stderr or result.stdout

    generations = tuple(
        path
        for path in (project.path / ".amd-ai/generations").iterdir()
        if path.is_dir()
    )
    artifacts = tuple(
        path
        for path in (project.path / ".amd-ai/artifacts/sha256").iterdir()
        if path.is_dir()
    )
    assert len(generations) == 2
    assert len(artifacts) == 2
    assert (project.path / ".amd-ai/current").resolve() in generations


def _hold_tls_connections(listener: socket.socket, stop: threading.Event) -> None:
    listener.settimeout(0.2)
    connections: list[socket.socket] = []
    try:
        while not stop.is_set():
            try:
                connection, _ = listener.accept()
            except (TimeoutError, OSError):
                continue
            connections.append(connection)
    finally:
        for connection in connections:
            connection.close()


def _wait_for(predicate: Callable[[], bool], timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _mark_healthy_command() -> tuple[str, ...]:
    return (
        "python",
        "-c",
        "from pathlib import Path; "
        "from amd_ai.overlay.models import OverlayPaths; "
        "from amd_ai.overlay.transaction import mark_generation_healthy; "
        "mark_generation_healthy(OverlayPaths.for_project(Path('/workspace')))",
    )


def _is_protected_entry(name: str) -> bool:
    lowered = name.lower()
    return lowered in {"torch", "torchvision", "torchaudio", "triton"} or (
        lowered.startswith(PROTECTED_PREFIXES)
    )
