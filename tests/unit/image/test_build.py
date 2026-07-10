import json
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
