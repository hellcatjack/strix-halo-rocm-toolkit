from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from amd_ai.overlay import cli as overlay_cli
from amd_ai.overlay.models import ProtectedComponent, ProtectedProfile
from amd_ai.overlay.resolver import WheelArtifact
from amd_ai.overlay.transaction import resolve_current_generation
from amd_ai.runner import CommandResult


class RecordingRunner:
    def __init__(
        self,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.calls: list[
            tuple[tuple[str, ...], dict[str, str], Path | None]
        ] = []

    def run(
        self,
        args: list[str],
        *,
        environment: dict[str, str],
        cwd: Path | None = None,
    ) -> CommandResult:
        command = tuple(args)
        self.calls.append((command, dict(environment), cwd))
        return CommandResult(
            command, self.returncode, self.stdout, self.stderr
        )


@pytest.fixture
def profile() -> ProtectedProfile:
    return ProtectedProfile(
        "stable",
        "sha256:" + "a" * 64,
        (
            ProtectedComponent("torch", "2.9.1+rocm7.2.1.build1"),
            ProtectedComponent(
                "torchvision", "0.24.0+rocm7.2.1.build1"
            ),
            ProtectedComponent("torchaudio", "2.9.0+rocm7.2.1.build1"),
            ProtectedComponent("triton", "3.5.1+rocm7.2.1.build1"),
        ),
    )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    path = tmp_path / "project"
    path.mkdir()
    (path / "amd-ai-project.toml").write_text(
        "schema_version = 1\n", encoding="utf-8"
    )
    return path


@pytest.fixture
def configured_cli(
    monkeypatch: pytest.MonkeyPatch, profile: ProtectedProfile
) -> None:
    monkeypatch.setattr(
        overlay_cli,
        "load_protected_profile",
        lambda **kwargs: profile,
    )


def test_install_compatible_torch_uses_verified_parent_without_resolution(
    project: Path,
    configured_cli: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_resolver(*args, **kwargs):
        raise AssertionError("resolver must not run for protected-only request")

    monkeypatch.setattr(
        overlay_cli, "resolve_and_materialize", unexpected_resolver
    )

    code = overlay_cli.main(
        ["install", "torch"], project=project, runner=RecordingRunner()
    )

    assert code == 0
    assert "already satisfied by verified parent" in capsys.readouterr().out


def test_install_conflicting_torch_version_is_rejected(
    project: Path,
    configured_cli: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = overlay_cli.main(
        ["install", "torch==2.8.0"],
        project=project,
        runner=RecordingRunner(),
    )

    assert code == 2
    assert "conflicts with verified parent" in capsys.readouterr().err


def test_uninstall_protected_torch_is_rejected(
    project: Path,
    configured_cli: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = overlay_cli.main(
        ["uninstall", "torch", "-y"],
        project=project,
        runner=RecordingRunner(),
    )

    assert code == 2
    assert "protected" in capsys.readouterr().err


def test_query_runs_against_effective_overlay(
    project: Path, configured_cli: None
) -> None:
    runner = RecordingRunner()

    code = overlay_cli.main(
        ["list", "--format=json"], project=project, runner=runner
    )

    command, environment, cwd = runner.calls[-1]
    assert code == 0
    assert command == (
        "/opt/venv/bin/python",
        "-m",
        "pip",
        "list",
        "--format",
        "json",
    )
    assert environment["PYTHONPATH"] == (
        f"{project}/.amd-ai/current/site-packages:/opt/amd-ai/src"
    )
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert cwd == project


def test_install_builds_complete_nonprotected_generation(
    project: Path,
    configured_cli: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(project, "demo_pkg-1.0-py3-none-any.whl", requested=True)
    captured = {}

    def fake_resolve(inspected, **kwargs):
        captured["inspected"] = inspected
        captured["transaction_dir"] = kwargs["transaction_dir"]
        return (artifact,)

    monkeypatch.setattr(overlay_cli, "resolve_and_materialize", fake_resolve)

    code = overlay_cli.main(
        ["install", "demo-pkg==1.0"],
        project=project,
        runner=RecordingRunner(),
    )

    generation = resolve_current_generation(
        overlay_cli.OverlayPaths.for_project(project)
    )
    input_text = (generation / "overlay.requirements.in").read_text(
        encoding="utf-8"
    )
    lock_text = (generation / "overlay.requirements.lock").read_text(
        encoding="utf-8"
    )
    assert code == 0
    assert captured["inspected"].resolver_inputs == ("demo-pkg==1.0",)
    assert captured["transaction_dir"].is_relative_to(project / ".amd-ai")
    assert "demo-pkg @ file://" in input_text
    assert "demo-pkg @ file:///workspace/.amd-ai/artifacts/" in lock_text
    assert "torch" not in lock_text


def test_tampered_current_metadata_blocks_resolution(
    project: Path,
    configured_cli: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = RecordingRunner()
    assert overlay_cli.main(
        ["install", "torch"], project=project, runner=runner
    ) == 0
    paths = overlay_cli.OverlayPaths.for_project(project)
    generation = resolve_current_generation(paths)
    (generation / "overlay.requirements.in").write_text(
        "tampered==1\n", encoding="utf-8"
    )

    def unexpected_resolver(*args, **kwargs):
        raise AssertionError("resolver ran after metadata corruption")

    monkeypatch.setattr(
        overlay_cli, "resolve_and_materialize", unexpected_resolver
    )

    code = overlay_cli.main(
        ["install", "demo"], project=project, runner=runner
    )

    assert code == 2
    assert "repair" in capsys.readouterr().err


def test_install_preserves_existing_top_level_root(
    project: Path,
    configured_cli: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alpha = _artifact(project, "alpha-1.0-py3-none-any.whl", requested=True)
    beta = _artifact(project, "beta-2.0-py3-none-any.whl", requested=True)
    calls = []

    def fake_resolve(inspected, **kwargs):
        calls.append(inspected)
        return (alpha,) if len(calls) == 1 else (alpha, beta)

    monkeypatch.setattr(overlay_cli, "resolve_and_materialize", fake_resolve)

    assert overlay_cli.main(
        ["install", "alpha"], project=project, runner=RecordingRunner()
    ) == 0
    assert overlay_cli.main(
        ["install", "beta"], project=project, runner=RecordingRunner()
    ) == 0

    assert calls[1].resolver_inputs == ("beta",)
    assert calls[1].local_inputs == (alpha.path,)
    generation = resolve_current_generation(
        overlay_cli.OverlayPaths.for_project(project)
    )
    roots = (generation / "overlay.requirements.in").read_text(
        encoding="utf-8"
    )
    assert "alpha @ file://" in roots
    assert "beta @ file://" in roots


def test_uninstall_removes_top_level_root_and_dependencies(
    project: Path,
    configured_cli: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(project, "alpha-1.0-py3-none-any.whl", requested=True)
    results = iter(((artifact,), ()))
    monkeypatch.setattr(
        overlay_cli,
        "resolve_and_materialize",
        lambda *args, **kwargs: next(results),
    )
    runner = RecordingRunner()
    assert overlay_cli.main(
        ["install", "alpha"], project=project, runner=runner
    ) == 0

    code = overlay_cli.main(
        ["uninstall", "alpha", "-y"], project=project, runner=runner
    )

    generation = resolve_current_generation(
        overlay_cli.OverlayPaths.for_project(project)
    )
    assert code == 0
    assert (generation / "overlay.requirements.in").read_text(
        encoding="utf-8"
    ) == ""
    assert (generation / "overlay.requirements.lock").read_text(
        encoding="utf-8"
    ) == ""


def test_missing_project_marker_is_rejected(
    tmp_path: Path,
    configured_cli: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = overlay_cli.main(
        ["list"], project=tmp_path, runner=RecordingRunner()
    )

    assert code == 2
    assert "amd-ai-project.toml" in capsys.readouterr().err


def test_logged_runner_redacts_sensitive_values_and_url_userinfo(
    tmp_path: Path,
) -> None:
    delegate = RecordingRunner(
        stdout="server echoed top-secret\n",
        stderr="https://person:password@example.test/error\n",
    )
    log_path = tmp_path / "transaction.log"
    runner = overlay_cli.LoggedProcessRunner(delegate, log_path)

    runner.run(
        ["pip", "https://person:password@example.test/simple"],
        environment={"API_TOKEN": "top-secret", "SAFE": "visible"},
        cwd=tmp_path,
    )

    text = log_path.read_text(encoding="utf-8")
    assert "password" not in text
    assert "top-secret" not in text
    assert "https://<redacted>@example.test/simple" in text
    assert "API_TOKEN=<redacted>" in text
    assert "SAFE=visible" in text


def _artifact(project: Path, filename: str, *, requested: bool) -> WheelArtifact:
    content = b"wheel"
    digest = hashlib.sha256(content).hexdigest()
    directory = project / ".amd-ai" / "artifacts" / "sha256" / digest
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_bytes(content)
    path.chmod(0o444)
    name = filename.split("-", 1)[0].replace("_", "-")
    version = filename.split("-", 2)[1]
    return WheelArtifact(name, version, digest, path, requested)
