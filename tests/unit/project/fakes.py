from __future__ import annotations

from pathlib import Path

from amd_ai.project.config import MountConfig, ProjectConfig
from amd_ai.project.runtime import GpuAccess
from amd_ai.runner import CommandResult


class FakeRunner:
    def __init__(
        self,
        responses: dict[tuple[str, ...], CommandResult] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.calls: list[tuple[str, ...]] = []

    def run(
        self,
        args: list[str],
        *,
        check: bool = True,
        input_text: str | None = None,
    ) -> CommandResult:
        key = tuple(args)
        self.calls.append(key)
        if key not in self.responses:
            raise AssertionError(f"unregistered command: {key}")
        result = self.responses[key]
        if check and result.returncode != 0:
            raise AssertionError(f"fake command failed: {key}")
        return result


def project_config(
    directory: Path,
    *,
    debug: bool = False,
    mounts: tuple[MountConfig, ...] = (),
    environment: tuple[tuple[str, str], ...] = (),
) -> ProjectConfig:
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "amd-ai-project.toml"
    config_path.write_text("# test fixture\n", encoding="utf-8")
    parent = "sha256:" + "a" * 64
    return ProjectConfig(
        path=config_path,
        name="demo",
        base_profile="stable",
        image="amd-ai-project/demo:runtime",
        base_image=parent,
        base_manifest_digest=parent,
        base_digest=parent,
        command=("python", "app.py"),
        debug=debug,
        shm_size_gib=None,
        mounts=mounts,
        environment=environment,
    )


def runtime_access() -> GpuAccess:
    return GpuAccess(
        devices=(Path("/dev/kfd"), Path("/dev/dri")),
        render_nodes=(Path("/dev/dri/renderD128"),),
        group_ids=(109, 110),
    )
