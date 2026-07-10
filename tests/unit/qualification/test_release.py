from __future__ import annotations

import pytest

from amd_ai.qualification.release import ReleaseBlocked, verify_release
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
