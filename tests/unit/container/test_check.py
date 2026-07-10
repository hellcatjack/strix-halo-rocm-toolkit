from pathlib import Path

from amd_ai.container import check
from amd_ai.container.check import public_version, run_rocm_check, run_torch_check
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


def write_torch_profile(root, versions=None):
    versions = versions or {
        "TORCH": "2.9.1",
        "TORCHVISION": "0.24.0",
        "TORCHAUDIO": "2.9.0",
        "TRITON": "3.5.1",
    }
    lines = [
        "PROFILE_ID=test-profile",
        "PROFILE_STATUS=experimental",
        "ROCM_VERSION=7.2.1",
        "PYTHON_ABI=cp312",
        "PLATFORM=linux/amd64",
    ]
    for prefix, version in versions.items():
        lines.extend(
            (
                f"{prefix}_VERSION={version}",
                f"{prefix}_URL=https://example.com/{prefix.lower()}.whl",
                f"{prefix}_SHA256={'1' * 64}",
            )
        )
    profile = root / "opt/amd-ai/profile.env"
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def test_public_version_retains_full_local_version_as_separate_evidence():
    assert public_version("2.9.1+rocm7.2.1.lw.gitff65f5bc") == "2.9.1"


def test_torch_metadata_checks_four_versions_hip_and_manifest(tmp_path, monkeypatch):
    root = rocm_root(tmp_path)
    write_torch_profile(root)
    manifest = root / "opt/amd-ai/torch-manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")

    class Version:
        hip = "7.2.1"

    class Module:
        def __init__(self, version):
            self.__version__ = version

    modules = {
        "torch": Module("2.9.1+rocm7.2.1.local"),
        "torchvision": Module("0.24.0+rocm7.2.1.local"),
        "torchaudio": Module("2.9.0+rocm7.2.1.local"),
        "triton": Module("3.5.1+rocm7.2.1.local"),
    }
    modules["torch"].version = Version()
    monkeypatch.setattr(check.importlib, "import_module", lambda name: modules[name])
    runner = hipcc_runner()
    verify_args = (
        check.sys.executable,
        str(root / "opt/amd-ai/torch-manifest.py"),
        "verify",
        str(manifest),
    )
    runner.responses[verify_args] = CommandResult(verify_args, 0, "", "")

    report = run_torch_check(
        root=root,
        runner=runner,
        metadata_only=True,
        runtime=False,
    )

    assert report.status.value == "pass"
    assert report.facts["torch"]["public_version"] == "2.9.1"
    assert report.facts["torch"]["full_version"].endswith(".local")
    assert report.facts["torch_hip_version"] == "7.2.1"


def test_torch_versions_are_loaded_from_image_profile(tmp_path, monkeypatch):
    root = rocm_root(tmp_path)
    versions = {
        "TORCH": "2.10.0",
        "TORCHVISION": "0.25.0",
        "TORCHAUDIO": "2.10.0",
        "TRITON": "3.6.0",
    }
    write_torch_profile(root, versions)
    manifest = root / "opt/amd-ai/torch-manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    class Version:
        hip = "7.2.1"

    class Module:
        def __init__(self, version):
            self.__version__ = f"{version}+rocm7.2.1.local"

    modules = {
        "torch": Module(versions["TORCH"]),
        "torchvision": Module(versions["TORCHVISION"]),
        "torchaudio": Module(versions["TORCHAUDIO"]),
        "triton": Module(versions["TRITON"]),
    }
    modules["torch"].version = Version()
    monkeypatch.setattr(check.importlib, "import_module", lambda name: modules[name])
    runner = hipcc_runner()
    verify_args = (
        check.sys.executable,
        str(root / "opt/amd-ai/torch-manifest.py"),
        "verify",
        str(manifest),
    )
    runner.responses[verify_args] = CommandResult(verify_args, 0, "", "")

    report = run_torch_check(
        root=root,
        runner=runner,
        metadata_only=True,
        runtime=False,
    )

    assert report.status.value == "pass"
    assert report.facts["torch"]["public_version"] == "2.10.0"
    assert report.facts["torch_profile"] == {
        "id": "test-profile",
        "status": "experimental",
    }
