from pathlib import Path

from amd_ai.container.check import run_rocm_check
from amd_ai.runner import CommandResult
from tests.unit.host.fakes import FakeRunner


def rocm_root(tmp_path):
    root = tmp_path / "root"
    version = root / "opt/rocm/.info/version"
    version.parent.mkdir(parents=True)
    version.write_text("7.2.1\n", encoding="utf-8")
    return root


def hipcc_runner():
    args = ("/opt/rocm/bin/hipcc", "--version")
    return FakeRunner(
        {
            args: CommandResult(
                args,
                0,
                "HIP version: 7.2.53211\n",
                "",
            )
        }
    )


def test_rocm_metadata_only_does_not_require_host_devices(tmp_path):
    runner = hipcc_runner()

    report = run_rocm_check(
        root=rocm_root(tmp_path),
        runner=runner,
        metadata_only=True,
    )

    assert report.status.value == "pass"
    assert report.facts["rocm_version"] == "7.2.1"
    assert runner.calls == [("/opt/rocm/bin/hipcc", "--version")]


def test_rocm_runtime_check_blocks_when_devices_are_not_mapped(tmp_path):
    runner = hipcc_runner()

    report = run_rocm_check(
        root=rocm_root(tmp_path),
        runner=runner,
        metadata_only=False,
    )

    assert report.status.value == "blocked"
    codes = {finding.code for finding in report.findings}
    assert {"GPU.KFD_MISSING", "GPU.RENDER_MISSING"} <= codes

