from __future__ import annotations

from pathlib import Path

import pytest

from amd_ai.overlay.models import ProtectedComponent, ProtectedProfile
from amd_ai.overlay.requirements import (
    RequirementPolicyError,
    inspect_requirements,
)


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


def test_bare_and_compatible_protected_requirements_are_external(
    profile: ProtectedProfile, tmp_path: Path
) -> None:
    result = inspect_requirements(
        ("torch", "Torch_Vision>=0.24", "requests==2.32.5"),
        (),
        project=tmp_path,
        profile=profile,
    )

    assert result.external == ("torch", "torchvision")
    assert result.resolver_inputs == ("requests==2.32.5",)


@pytest.mark.parametrize(
    "requirement",
    [
        "torch==2.8.0",
        "torch @ https://download.example/torch-2.9.1-cp312.whl",
        "triton @ file:///workspace/triton-3.5.1-py3-none-any.whl",
    ],
)
def test_incompatible_or_source_bound_protected_requirement_is_blocked(
    requirement: str, profile: ProtectedProfile, tmp_path: Path
) -> None:
    with pytest.raises(RequirementPolicyError):
        inspect_requirements(
            (requirement,), (), project=tmp_path, profile=profile
        )


def test_protected_local_wheel_name_is_blocked(
    profile: ProtectedProfile, tmp_path: Path
) -> None:
    wheel = tmp_path / "Torch_Vision-0.24.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")

    with pytest.raises(RequirementPolicyError, match="protected"):
        inspect_requirements(
            (str(wheel),), (), project=tmp_path, profile=profile
        )


def test_nested_requirements_are_bounded_to_project(
    profile: ProtectedProfile, tmp_path: Path
) -> None:
    (tmp_path / "nested.txt").write_text(
        "torch>=2.9\nrequests\n", encoding="utf-8"
    )
    (tmp_path / "requirements.txt").write_text(
        "-r nested.txt\n", encoding="utf-8"
    )

    result = inspect_requirements(
        (), ("requirements.txt",), project=tmp_path, profile=profile
    )

    assert result.external == ("torch",)
    assert result.resolver_inputs == ("requests",)


def test_requirements_include_cannot_escape_project(
    profile: ProtectedProfile, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("requests\n", encoding="utf-8")
    (project / "requirements.txt").write_text(
        "-r ../outside.txt\n", encoding="utf-8"
    )

    with pytest.raises(RequirementPolicyError, match="outside project"):
        inspect_requirements(
            (), ("requirements.txt",), project=project, profile=profile
        )


def test_exact_commit_git_is_materialized_but_mutable_ref_is_rejected(
    profile: ProtectedProfile, tmp_path: Path
) -> None:
    commit = "a" * 40
    requirement = (
        f"demo @ git+https://github.com/example/demo.git@{commit}"
    )

    result = inspect_requirements(
        (requirement,), (), project=tmp_path, profile=profile
    )

    assert result.vcs_inputs == (requirement,)
    with pytest.raises(RequirementPolicyError, match="exact commit"):
        inspect_requirements(
            ("demo @ git+https://github.com/example/demo.git@main",),
            (),
            project=tmp_path,
            profile=profile,
        )


def test_local_source_and_wheel_are_returned_for_materialization(
    profile: ProtectedProfile, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
    wheel = tmp_path / "demo-1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")

    result = inspect_requirements(
        (str(source), str(wheel)), (), project=tmp_path, profile=profile
    )

    assert result.local_inputs == (source, wheel)


def test_nested_index_option_rejects_credentials(
    profile: ProtectedProfile, tmp_path: Path
) -> None:
    (tmp_path / "requirements.txt").write_text(
        "--index-url https://user:secret@packages.example/simple\nrequests\n",
        encoding="utf-8",
    )

    with pytest.raises(RequirementPolicyError, match="credentials"):
        inspect_requirements(
            (), ("requirements.txt",), project=tmp_path, profile=profile
        )


def test_nested_equals_index_option_is_preserved_without_secrets(
    profile: ProtectedProfile, tmp_path: Path
) -> None:
    (tmp_path / "requirements.txt").write_text(
        "--index-url=https://packages.example/simple\nrequests\n",
        encoding="utf-8",
    )

    result = inspect_requirements(
        (), ("requirements.txt",), project=tmp_path, profile=profile
    )

    assert result.resolver_inputs == (
        "--index-url https://packages.example/simple",
        "requests",
    )
