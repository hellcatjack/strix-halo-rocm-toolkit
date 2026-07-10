from __future__ import annotations

import pytest
from types import SimpleNamespace

from amd_ai.qualification.release import (
    ReleaseBlocked,
    git_status_argv,
    publish_release_artifacts,
    verify_release,
)
from tests.unit.qualification.fakes import passing_release_inputs


def test_release_requires_digest_locks_sbom_and_all_checks():
    result = verify_release(passing_release_inputs())

    assert result.status == "verified"
    assert result.image_id.startswith("sha256:")
    assert set(result.wheel_hashes) == {
        "torch",
        "torchvision",
        "torchaudio",
        "triton",
    }
    with pytest.raises(ReleaseBlocked, match="kernel-log"):
        verify_release(passing_release_inputs(failed_check="kernel-log"))


def test_release_rejects_dirty_git_or_experimental_image():
    with pytest.raises(ReleaseBlocked, match="Git"):
        verify_release(passing_release_inputs(git_clean=False))
    with pytest.raises(ReleaseBlocked, match="verified"):
        verify_release(passing_release_inputs(profile_status="experimental"))


def test_release_binds_qualification_and_embedded_locks_to_image():
    with pytest.raises(ReleaseBlocked, match="image ID"):
        verify_release(
            passing_release_inputs(
                qualification_image_id="sha256:" + "a" * 64,
            )
        )
    with pytest.raises(ReleaseBlocked, match="embedded"):
        verify_release(passing_release_inputs(embedded_locks_match=False))


def test_git_cleanliness_includes_untracked_files():
    assert git_status_argv() == (
        "git",
        "status",
        "--porcelain",
        "--untracked-files=all",
    )


def test_artifact_publish_failure_rolls_back_new_verified_tag(
    tmp_path,
    monkeypatch,
):
    record = verify_release(passing_release_inputs())
    docker = TagDocker()
    monkeypatch.setattr(
        "amd_ai.qualification.release.os.replace",
        lambda source, destination: (_ for _ in ()).throw(OSError("disk failure")),
    )

    with pytest.raises(OSError, match="disk failure"):
        publish_release_artifacts(
            docker=docker,
            record=record,
            sbom_bytes=b"{}\n",
            release_path=tmp_path / "release.json",
            sbom_path=tmp_path / "release.spdx.json",
        )

    assert docker.current_tag_id is None
    assert any(call[:2] == ("image", "rm") for call in docker.calls)


class TagDocker:
    def __init__(self):
        self.current_tag_id = None
        self.calls = []

    def image_id(self, reference, *, required=True):
        return self.current_tag_id

    def capture(self, args, *, check=True):
        self.calls.append(args)
        if args[:2] == ("image", "tag"):
            self.current_tag_id = args[2]
        elif args[:2] == ("image", "rm"):
            self.current_tag_id = None
        return SimpleNamespace(returncode=0, stdout="", stderr="")
