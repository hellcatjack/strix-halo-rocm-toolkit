from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from amd_ai.overlay.models import ProtectedComponent, ProtectedProfile
from amd_ai.overlay.resolver import (
    ResolverError,
    ReportItem,
    direct_wheel_requirements,
    materialize_report,
    parse_pip_report,
    prepare_direct_wheels,
    render_constraints,
    resolve_and_materialize,
    resolve_report,
    resolver_argv,
    store_wheel,
)
from amd_ai.runner import CommandResult
from amd_ai.overlay.requirements import InspectedRequirements


@pytest.fixture
def profile() -> ProtectedProfile:
    return ProtectedProfile(
        "stable",
        "sha256:" + "a" * 64,
        (
            ProtectedComponent("torch", "2.9.1+rocm7.2.1.build1"),
            ProtectedComponent("torchvision", "0.24.0+rocm7.2.1.build1"),
            ProtectedComponent("torchaudio", "2.9.0+rocm7.2.1.build1"),
            ProtectedComponent("triton", "3.5.1+rocm7.2.1.build1"),
        ),
    )


def test_report_accepts_hashed_nonprotected_wheel() -> None:
    payload = {
        "version": "1",
        "pip_version": "24.0",
        "install": [
            {
                "download_info": {
                    "url": "https://files.pythonhosted.org/x/requests.whl",
                    "archive_info": {"hashes": {"sha256": "b" * 64}},
                },
                "requested": True,
                "metadata": {"name": "requests", "version": "2.32.5"},
            }
        ],
        "environment": {"python_version": "3.12"},
    }

    items = parse_pip_report(json.dumps(payload))

    assert items == (
        ReportItem(
            name="requests",
            version="2.32.5",
            url="https://files.pythonhosted.org/x/requests.whl",
            sha256="b" * 64,
            requested=True,
        ),
    )


def test_report_blocks_transitive_protected_distribution() -> None:
    payload = {
        "version": "1",
        "pip_version": "24.0",
        "install": [
            {
                "download_info": {
                    "url": "https://example/torch.whl",
                    "archive_info": {"hashes": {"sha256": "c" * 64}},
                },
                "requested": False,
                "metadata": {"name": "Torch", "version": "2.8.0"},
            }
        ],
    }

    with pytest.raises(ResolverError, match="protected distribution"):
        parse_pip_report(json.dumps(payload))


@pytest.mark.parametrize(
    "mutation",
    ["schema", "missing-hash", "credential-url", "duplicate"],
)
def test_report_rejects_ambiguous_artifacts(mutation: str) -> None:
    item = {
        "download_info": {
            "url": "https://example/demo.whl",
            "archive_info": {"hashes": {"sha256": "d" * 64}},
        },
        "requested": True,
        "metadata": {"name": "demo", "version": "1.0"},
    }
    payload: dict[str, object] = {
        "version": "1",
        "pip_version": "24.0",
        "install": [item],
    }
    if mutation == "schema":
        payload["version"] = "2"
    elif mutation == "missing-hash":
        item["download_info"]["archive_info"] = {"hashes": {}}
    elif mutation == "credential-url":
        item["download_info"]["url"] = "https://user:secret@example/demo.whl"
    else:
        payload["install"] = [item, dict(item)]

    with pytest.raises(ResolverError):
        parse_pip_report(json.dumps(payload))


def test_constraints_pin_all_protected_full_versions(profile: ProtectedProfile) -> None:
    assert render_constraints(profile) == (
        "torch==2.9.1+rocm7.2.1.build1\n"
        "torchaudio==2.9.0+rocm7.2.1.build1\n"
        "torchvision==0.24.0+rocm7.2.1.build1\n"
        "triton==3.5.1+rocm7.2.1.build1\n"
    )


def test_resolver_argv_uses_report_and_constraints(tmp_path: Path) -> None:
    argv = resolver_argv(
        input_path=tmp_path / "requirements.in",
        constraints_path=tmp_path / "constraints.txt",
        report_path=tmp_path / "report.json",
        resolver_options=("--index-url", "https://packages.example/simple"),
    )

    assert argv == (
        "/opt/venv/bin/python",
        "-m",
        "pip",
        "install",
        "--dry-run",
        "--disable-pip-version-check",
        "--report",
        str(tmp_path / "report.json"),
        "--constraint",
        str(tmp_path / "constraints.txt"),
        "--index-url",
        "https://packages.example/simple",
        "--requirement",
        str(tmp_path / "requirements.in"),
    )


def test_store_wheel_hashes_and_uses_content_addressed_path(tmp_path: Path) -> None:
    source = tmp_path / "download" / "demo-1.0-py3-none-any.whl"
    source.parent.mkdir()
    source.write_bytes(b"wheel bytes")
    digest = hashlib.sha256(b"wheel bytes").hexdigest()

    artifact = store_wheel(
        source,
        artifacts_root=tmp_path / "artifacts" / "sha256",
        expected_name="demo",
        expected_version="1.0",
        requested=True,
    )

    assert artifact.sha256 == digest
    assert artifact.path == (
        tmp_path
        / "artifacts"
        / "sha256"
        / digest
        / "demo-1.0-py3-none-any.whl"
    )
    assert artifact.path.read_bytes() == b"wheel bytes"
    assert artifact.path.stat().st_mode & 0o777 == 0o444


def test_store_wheel_rejects_name_or_version_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "demo-1.0-py3-none-any.whl"
    source.write_bytes(b"wheel")

    with pytest.raises(ResolverError, match="identity"):
        store_wheel(
            source,
            artifacts_root=tmp_path / "artifacts",
            expected_name="other",
            expected_version="1.0",
            requested=False,
        )


class ReportRunner:
    def __init__(self, report: dict[str, object], *, returncode: int = 0) -> None:
        self.report = report
        self.returncode = returncode
        self.calls: list[tuple[tuple[str, ...], dict[str, str], Path | None]] = []

    def run(
        self,
        args: list[str],
        *,
        environment: dict[str, str],
        cwd: Path | None = None,
    ) -> CommandResult:
        self.calls.append((tuple(args), environment, cwd))
        report_path = Path(args[args.index("--report") + 1])
        if self.returncode == 0:
            report_path.write_text(json.dumps(self.report), encoding="utf-8")
        return CommandResult(tuple(args), self.returncode, "resolver out", "resolver err")


def test_resolve_report_hides_current_overlay_and_removes_private_cache(
    profile: ProtectedProfile, tmp_path: Path
) -> None:
    report = {"version": "1", "pip_version": "24.0", "install": []}
    runner = ReportRunner(report)
    transaction = tmp_path / "transaction"

    items = resolve_report(
        input_lines=("requests==2.32.5",),
        profile=profile,
        transaction_dir=transaction,
        resolver_options=(),
        runner=runner,
        base_environment={"PYTHONPATH": "/untrusted/current"},
    )

    assert items == ()
    args, environment, cwd = runner.calls[0]
    assert args[0:4] == ("/opt/venv/bin/python", "-m", "pip", "install")
    assert environment["PYTHONPATH"] == "/opt/amd-ai/src"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PIP_CACHE_DIR"].endswith("/pip-cache")
    assert not Path(environment["PIP_CACHE_DIR"]).exists()
    assert cwd == Path("/workspace")


def test_resolve_report_failure_does_not_leave_cache(
    profile: ProtectedProfile, tmp_path: Path
) -> None:
    runner = ReportRunner({}, returncode=1)
    transaction = tmp_path / "transaction"

    with pytest.raises(ResolverError, match="pip resolution failed"):
        resolve_report(
            input_lines=("requests",),
            profile=profile,
            transaction_dir=transaction,
            resolver_options=(),
            runner=runner,
        )

    assert not (transaction / "pip-cache").exists()


class DownloadRunner:
    def __init__(self, artifact_name: str, artifact_bytes: bytes) -> None:
        self.artifact_name = artifact_name
        self.artifact_bytes = artifact_bytes
        self.calls: list[tuple[str, ...]] = []

    def run(
        self,
        args: list[str],
        *,
        environment: dict[str, str],
        cwd: Path | None = None,
    ) -> CommandResult:
        command = tuple(args)
        self.calls.append(command)
        destination = Path(args[args.index("--dest") + 1])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / self.artifact_name).write_bytes(self.artifact_bytes)
        return CommandResult(command, 0, "downloaded", "")


def test_materialize_report_downloads_and_stores_exact_wheel(tmp_path: Path) -> None:
    wheel_bytes = b"resolved wheel"
    item = ReportItem(
        name="demo",
        version="1.0",
        url="https://packages.example/demo-1.0-py3-none-any.whl",
        sha256=hashlib.sha256(wheel_bytes).hexdigest(),
        requested=True,
    )
    runner = DownloadRunner("demo-1.0-py3-none-any.whl", wheel_bytes)

    artifacts = materialize_report(
        (item,),
        artifacts_root=tmp_path / "artifacts",
        transaction_dir=tmp_path / "transaction",
        runner=runner,
    )

    assert artifacts[0].name == "demo"
    assert artifacts[0].requested is True
    assert len(runner.calls) == 1
    assert "--require-hashes" in runner.calls[0]
    assert not (tmp_path / "transaction" / "downloads").exists()


class WheelBuildRunner:
    def __init__(self, wheel_name: str) -> None:
        self.wheel_name = wheel_name
        self.calls: list[tuple[str, ...]] = []

    def run(
        self,
        args: list[str],
        *,
        environment: dict[str, str],
        cwd: Path | None = None,
    ) -> CommandResult:
        command = tuple(args)
        self.calls.append(command)
        wheel_dir = Path(args[args.index("--wheel-dir") + 1])
        wheel_dir.mkdir(parents=True, exist_ok=True)
        (wheel_dir / self.wheel_name).write_bytes(b"built wheel")
        return CommandResult(command, 0, "built", "")


def test_prepare_direct_wheels_builds_local_source_and_exact_git(
    tmp_path: Path,
) -> None:
    source = tmp_path / "project" / "source"
    source.mkdir(parents=True)
    commit = "a" * 40
    vcs = f"demo @ git+https://github.com/example/demo.git@{commit}"
    inspected = InspectedRequirements((), (), (source,), (vcs,))
    runner = WheelBuildRunner("demo-1.0-py3-none-any.whl")

    wheels = prepare_direct_wheels(
        inspected,
        transaction_dir=tmp_path / "transaction",
        runner=runner,
    )

    assert len(wheels) == 2
    assert str(source) in runner.calls[0]
    assert vcs in runner.calls[1]
    assert direct_wheel_requirements(wheels) == (
        f"demo @ {wheels[0].as_uri()}",
        f"demo @ {wheels[1].as_uri()}",
    )


def test_prepare_direct_wheels_blocks_source_that_builds_protected_wheel(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    inspected = InspectedRequirements((), (), (source,), ())
    runner = WheelBuildRunner("torch-2.9.1-py3-none-any.whl")

    with pytest.raises(ResolverError, match="protected"):
        prepare_direct_wheels(
            inspected,
            transaction_dir=tmp_path / "transaction",
            runner=runner,
        )


class PipelineRunner:
    def __init__(self, wheel: Path) -> None:
        self.wheel = wheel
        self.calls: list[tuple[str, ...]] = []

    def run(
        self,
        args: list[str],
        *,
        environment: dict[str, str],
        cwd: Path | None = None,
    ) -> CommandResult:
        command = tuple(args)
        self.calls.append(command)
        report_path = Path(args[args.index("--report") + 1])
        report_path.write_text(
            json.dumps(
                {
                    "version": "1",
                    "pip_version": "24.0",
                    "install": [
                        {
                            "download_info": {
                                "url": self.wheel.as_uri(),
                                "archive_info": {
                                    "hashes": {
                                        "sha256": hashlib.sha256(
                                            self.wheel.read_bytes()
                                        ).hexdigest()
                                    }
                                },
                            },
                            "requested": True,
                            "metadata": {"name": "demo", "version": "1.0"},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return CommandResult(command, 0, "resolved", "")


def test_resolve_and_materialize_builds_complete_artifact_set(
    profile: ProtectedProfile, tmp_path: Path
) -> None:
    wheel = tmp_path / "project" / "demo-1.0-py3-none-any.whl"
    wheel.parent.mkdir()
    wheel.write_bytes(b"wheel")
    inspected = InspectedRequirements((), ("torch",), (wheel,), ())
    runner = PipelineRunner(wheel)

    artifacts = resolve_and_materialize(
        inspected,
        profile=profile,
        artifacts_root=tmp_path / "artifacts",
        transaction_dir=tmp_path / "transaction",
        resolver_options=(),
        runner=runner,
    )

    assert tuple(artifact.name for artifact in artifacts) == ("demo",)
    assert len(runner.calls) == 1
