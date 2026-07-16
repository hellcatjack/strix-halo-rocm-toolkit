import io
import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from amd_ai.image import build
from amd_ai.image.build import (
    IMAGE_SOURCE,
    LocalImage,
    build_rocm_python_argv,
    build_torch_argv,
    driver_supports_attestations,
    default_project_roots,
    immutable_parent_alias,
    materialize_profile_context,
    project_base_image_ids,
    select_prunable_images,
)
from amd_ai.image.profile import load_profile
from amd_ai.installer.progress import InstallerProgress, ProgressMode
from amd_ai.runner import CommandResult, CommandStream


def test_docker_detection_does_not_probe_buildx(monkeypatch):
    calls: list[tuple[str, ...]] = []

    def fake_completed(args, *, check=True):
        del check
        command = tuple(args)
        calls.append(command)
        assert command == (
            "docker",
            "info",
            "--format",
            "{{.ServerVersion}}",
        )
        return subprocess.CompletedProcess(command, 0, "29.1.3\n", "")

    monkeypatch.setattr(build, "_completed", fake_completed)

    docker = build.Docker.detect()

    assert docker.prefix == ("docker",)
    assert docker.server_version == "29.1.3"
    assert calls == [
        ("docker", "info", "--format", "{{.ServerVersion}}")
    ]


def test_require_buildx_reports_healthy_daemon(monkeypatch):
    docker = build.Docker(("docker",), "29.1.3")
    missing = subprocess.CompletedProcess(
        ("docker", "buildx", "version"),
        1,
        "",
        "docker: unknown command: docker buildx",
    )
    monkeypatch.setattr(
        docker,
        "capture",
        lambda args, check=False: missing,
    )

    with pytest.raises(build.BuildError, match="29.1.3.*host repair"):
        docker.require_buildx()


def test_build_argv_uses_content_addressed_local_parent_and_named_context():
    profile = load_profile("profiles/torch/stable.env", allow_verified=True)
    parent = "sha256:" + "a" * 64

    argv = build_torch_argv(
        profile=profile,
        parent=parent,
        wheelhouse=Path(".cache/wheels/rocm-7.2.1-py3.12-torch-2.9.1"),
        revision="deadbeef",
    )

    alias = "amd-ai-local/rocm-python:" + "a" * 64
    assert immutable_parent_alias(parent) == alias
    assert f"ROCM_PYTHON_BASE={alias}" in argv
    assert "wheels=.cache/wheels/rocm-7.2.1-py3.12-torch-2.9.1" in argv
    assert (
        "profile-context=.cache/profile-context/"
        "rocm-7.2.1-py3.12-torch-2.9.1" in argv
    )
    assert "--provenance=mode=max" in argv
    assert "--sbom=true" in argv
    assert "--load" in argv
    assert argv.count("--progress=plain") == 1
    assert not any("repo.radeon.com" in argument for argument in argv)


def test_classic_store_build_disables_unsupported_image_attestations():
    profile = load_profile("profiles/torch/stable.env", allow_verified=True)

    argv = build_torch_argv(
        profile=profile,
        parent="sha256:" + "a" * 64,
        wheelhouse=Path("wheelhouse"),
        revision="deadbeef",
        attestations=False,
    )

    assert "--provenance=false" in argv
    assert "--sbom=false" in argv
    assert driver_supports_attestations(
        '[["driver-type","io.containerd.snapshotter.v1"]]'
    )
    assert not driver_supports_attestations(
        '[["Backing Filesystem","extfs"],["Supports d_type","true"]]'
    )


def test_base_build_passes_only_pinned_images_and_source_metadata():
    argv = build_rocm_python_argv(
        ubuntu_base="ubuntu@sha256:" + "a" * 64,
        uv_image="ghcr.io/astral-sh/uv@sha256:" + "b" * 64,
        revision="deadbeef",
        image_source="local",
    )

    assert "UBUNTU_BASE=ubuntu@sha256:" + "a" * 64 in argv
    assert "UV_IMAGE=ghcr.io/astral-sh/uv@sha256:" + "b" * 64 in argv
    assert "IMAGE_SOURCE=local" in argv
    assert argv.count("--progress=plain") == 1


def test_normal_builds_use_public_source_repository():
    assert IMAGE_SOURCE == (
        "https://github.com/hellcatjack/strix-halo-rocm-toolkit"
    )

    argv = build_rocm_python_argv(
        ubuntu_base="ubuntu@sha256:" + "a" * 64,
        uv_image="ghcr.io/astral-sh/uv@sha256:" + "b" * 64,
        revision="c" * 40,
    )

    assert f"IMAGE_SOURCE={IMAGE_SOURCE}" in argv


def test_run_live_streams_output_to_command_observer(tmp_path: Path) -> None:
    class RecordingObserver:
        def __init__(self) -> None:
            self.lines: list[tuple[CommandStream, str]] = []

        def command_started(
            self, args, *, live, environment=None
        ) -> None:
            del args, live, environment

        def command_output(
            self, stream: CommandStream, text: str
        ) -> str:
            self.lines.append((stream, text))
            return text

        def command_finished(
            self, result: CommandResult, *, live: bool
        ) -> None:
            del result, live

    observer = RecordingObserver()

    build._run_live(
        [sys.executable, "-c", "print('build-visible')"],
        cwd=tmp_path,
        observer=observer,
    )

    assert observer.lines == [
        (CommandStream.STDOUT, "build-visible\n")
    ]


def test_locked_wheel_progress_exposes_filename_without_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stable = Path("profiles/torch/stable.env").resolve()
    profile = replace(
        load_profile(stable, allow_verified=True),
        profile_id="experimental-progress",
        status="experimental",
    )

    class StopDownload(RuntimeError):
        pass

    class RecordingObserver:
        def __init__(self) -> None:
            self.output: list[str] = []

        def command_output(
            self, stream: CommandStream, text: str
        ) -> str:
            assert stream is CommandStream.STDOUT
            self.output.append(text)
            return text

    def stop_after_progress(
        url, destination, expected_sha256, progress=None
    ):
        del url, destination, expected_sha256
        assert progress is not None
        progress(1024**3, 2 * 1024**3)
        raise StopDownload

    observer = RecordingObserver()
    monkeypatch.setattr(build, "download", stop_after_progress)

    with pytest.raises(StopDownload):
        build._prepare_profile_artifacts(
            profile=profile,
            profile_path=stable,
            repo_root=tmp_path,
            stable=tmp_path / "not-stable.env",
            observer=observer,
        )

    output = "".join(observer.output)
    assert "下载 torch-2.9.1" in output
    assert "1.00 GiB/2.00 GiB" in output
    assert "https://" not in output


def test_materialized_context_contains_only_profile_and_matching_lock(tmp_path):
    profile = tmp_path / "custom.env"
    requirements = tmp_path / "custom.requirements.lock"
    profile.write_text("PROFILE_ID=custom\n", encoding="utf-8")
    requirements.write_text("torch==custom\n", encoding="utf-8")
    destination = tmp_path / "context"

    materialize_profile_context(profile, requirements, destination)

    assert sorted(path.name for path in destination.iterdir()) == [
        "profile.env",
        "requirements.lock",
    ]
    assert (destination / "profile.env").read_text() == "PROFILE_ID=custom\n"
    assert (destination / "requirements.lock").read_text() == "torch==custom\n"


def test_missing_wheelhouse_is_reported_as_a_build_error(tmp_path):
    profile = load_profile("profiles/torch/stable.env", allow_verified=True)

    with pytest.raises(build.BuildError, match="wheelhouse"):
        build._validate_profile_artifacts(
            profile,
            tmp_path / "missing",
            Path("profiles/torch/stable.requirements.lock"),
        )


def test_torch_build_requires_buildx_before_preparing_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class StopAfterBuildx(RuntimeError):
        pass

    class FakeDocker:
        prefix = ("docker",)

        def require_buildx(self) -> str:
            events.append("buildx")
            return "buildx 0.30.1"

    def stop_artifact_preparation(**kwargs):
        del kwargs
        events.append("artifacts")
        raise StopAfterBuildx

    monkeypatch.setattr(
        build.Docker,
        "detect",
        classmethod(lambda cls: FakeDocker()),
    )
    monkeypatch.setattr(build, "_prepare_profile_artifacts", stop_artifact_preparation)

    with pytest.raises(StopAfterBuildx):
        build.build_rocm_pytorch(
            profile_path=Path("profiles/torch/stable.env"),
            allow_experimental=False,
        )

    assert events == ["buildx", "artifacts"]


def test_build_metadata_returns_the_exact_config_digest(tmp_path):
    digest = "sha256:" + "a" * 64
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "containerimage.config.digest": digest,
                "buildx.build.provenance": {
                    "buildType": "https://mobyproject.org/buildkit@v1",
                    "builder": {},
                    "invocation": {},
                    "materials": [],
                },
            }
        ),
        encoding="utf-8",
    )

    assert build._validate_build_metadata(metadata) == digest


def test_project_base_ids_are_protected_from_prune(tmp_path):
    project = tmp_path / "video" / "amd-ai-project.toml"
    project.parent.mkdir()
    digest = "sha256:" + "b" * 64
    project.write_text(
        "[project]\n"
        'name = "video"\n'
        f'base_image = "{digest}"\n'
        f'base_digest = "{digest}"\n',
        encoding="utf-8",
    )

    assert project_base_image_ids((tmp_path,)) == frozenset({digest})


def test_prune_selection_excludes_protected_running_or_recent_images():
    now = datetime(2026, 7, 10, tzinfo=UTC)
    old = now - timedelta(days=30)
    recent = now - timedelta(hours=2)
    images = (
        LocalImage("sha256:" + "a" * 64, 100, old, {"org.amd-ai.profile.id": "old"}),
        LocalImage("sha256:" + "b" * 64, 200, old, {"org.amd-ai.profile.id": "used"}),
        LocalImage(
            "sha256:" + "c" * 64,
            300,
            recent,
            {"org.amd-ai.project.fingerprint": "recent"},
        ),
        LocalImage("sha256:" + "d" * 64, 400, old, {}),
    )

    selected = select_prunable_images(
        images,
        protected_ids={"sha256:" + "b" * 64},
        cutoff=now - timedelta(hours=168),
    )

    assert [image.image_id for image in selected] == ["sha256:" + "a" * 64]


def test_default_prune_roots_include_cli_default_project_location(tmp_path):
    repo_root = tmp_path / "repo"
    current_dir = tmp_path / "operator"

    assert default_project_roots(
        repo_root=repo_root,
        current_dir=current_dir,
    ) == (current_dir.resolve(), (repo_root / "projects").resolve())


def test_image_check_rejects_option_like_image_before_docker_detection(monkeypatch):
    def fail_detect(cls):
        raise AssertionError("Docker detection must not run")

    monkeypatch.setattr(build.Docker, "detect", classmethod(fail_detect))

    with pytest.raises(build.BuildError, match="image"):
        build.run_image_check(
            image="--privileged",
            mode="torch",
            metadata_only=True,
            runtime=False,
            json_path=None,
        )


@pytest.mark.parametrize(
    "progress_mode", (ProgressMode.DEFAULT, ProgressMode.QUIET)
)
def test_captured_image_check_keeps_json_out_of_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    progress_mode: ProgressMode,
) -> None:
    class FakeDocker:
        prefix = ("docker",)

    class CapturedRunner:
        def __init__(self, *, observer) -> None:
            self.observer = observer

        def run(self, args, *, check=True, input_text=None):
            del check, input_text
            command = tuple(args)
            self.observer.command_started(
                command,
                live=False,
                environment={"HF_TOKEN": "private-value"},
            )
            result = CommandResult(
                command,
                0,
                '{"status":"pass","HF_TOKEN":"private-value"}\n',
                "probe warning\n",
            )
            self.observer.command_finished(result, live=False)
            return result

    monkeypatch.setattr(
        build.Docker, "detect", classmethod(lambda cls: FakeDocker())
    )
    monkeypatch.setattr(build, "SubprocessRunner", CapturedRunner)
    stdout = io.StringIO()
    stderr = io.StringIO()
    reporter = InstallerProgress(
        mode=progress_mode,
        stdout=stdout,
        stderr=stderr,
        log_root=tmp_path / "logs",
        process_id=91,
    )
    reporter.open_session(tmp_path / "project")

    returncode = build.run_image_check(
        image="rocm-pytorch:test",
        mode="torch",
        metadata_only=False,
        runtime=True,
        json_path="-",
        observer=reporter,
    )
    log_path = reporter.log_path
    reporter.close()

    assert returncode == 0
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == ""
    assert log_path is not None
    log = log_path.read_text(encoding="utf-8")
    assert "[captured]" in log
    assert "stdout_bytes=" in log
    assert '"status":"pass"' not in log
    assert "probe warning" not in log
    assert "private-value" not in log


def test_prune_preview_never_issues_a_mutating_docker_command(tmp_path, monkeypatch):
    old = "2026-01-01T00:00:00Z"
    candidate = "sha256:" + "a" * 64
    project_base = "sha256:" + "b" * 64
    project = tmp_path / "project" / "amd-ai-project.toml"
    project.parent.mkdir()
    project.write_text(
        "[project]\n"
        f'base_image = "{project_base}"\n'
        f'base_digest = "{project_base}"\n',
        encoding="utf-8",
    )

    class FakeDocker:
        live_calls = []

        def image_id(self, reference, *, required=True):
            return None

        def capture(self, args, *, check=True):
            if args == ("ps", "--quiet"):
                return type("Result", (), {"stdout": ""})()
            if args == ("image", "ls", "--quiet", "--no-trunc"):
                return type(
                    "Result", (), {"stdout": f"{candidate}\n{project_base}\n"}
                )()
            if args[:2] == ("image", "inspect"):
                payload = [
                    {
                        "Id": candidate,
                        "Created": old,
                        "Size": 100,
                        "Config": {"Labels": {"org.amd-ai.profile.id": "old"}},
                    },
                    {
                        "Id": project_base,
                        "Created": old,
                        "Size": 200,
                        "Config": {"Labels": {"org.amd-ai.profile.id": "used"}},
                    },
                ]
                return type("Result", (), {"stdout": json.dumps(payload)})()
            raise AssertionError(args)

        def live(self, args, *, cwd=None):
            self.live_calls.append(args)

    fake = FakeDocker()
    monkeypatch.setattr(build.Docker, "detect", classmethod(lambda cls: fake))

    selected = build.prune_images(
        apply=False,
        older_than_hours=168,
        project_roots=(tmp_path,),
        repo_root=tmp_path,
    )

    assert [image.image_id for image in selected] == [candidate]
    assert fake.live_calls == []
