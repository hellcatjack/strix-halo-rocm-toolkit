from __future__ import annotations

import os
from pathlib import Path

import pytest

from amd_ai.image.build import Docker
from amd_ai.project.runtime import discover_gpu_access
from amd_ai.qualification.run import run_profile
from amd_ai.runner import SubprocessRunner


@pytest.mark.hardware
def test_stable_gfx1151_release_suite():
    if not Path("/dev/kfd").exists():
        pytest.skip("target GPU device /dev/kfd is absent")

    docker = Docker.detect()
    access = discover_gpu_access()
    uid, gid = _host_identity()
    output = Path(
        os.environ.get(
            "AMD_AI_QUALIFICATION_REPORT",
            "reports/qualification.json",
        )
    )
    report = run_profile(
        profile_path=Path("profiles/qualification/stable.toml"),
        output_path=output,
        runner=SubprocessRunner(),
        docker_prefix=docker.prefix,
        gids=access.group_ids,
        uid=uid,
        gid=gid,
    )

    failures = [
        f"{result.name}: {result.evidence}"
        for result in report.results
        if not result.passed
    ]
    assert report.status == "pass", "\n".join(failures)
    assert tuple(result.name for result in report.results) == report.required_checks
    assert report.gpu_arch == "gfx1151"


def _host_identity() -> tuple[int, int]:
    if os.geteuid() == 0 and "SUDO_UID" in os.environ and "SUDO_GID" in os.environ:
        return int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"])
    return os.getuid(), os.getgid()
