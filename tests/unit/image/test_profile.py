from pathlib import Path

import pytest

from amd_ai.image.profile import ProfileError, load_profile


def valid_profile_text(*, status="experimental"):
    return "\n".join(
        [
            "PROFILE_ID=custom" if status == "experimental" else "PROFILE_ID=stable-test",
            f"PROFILE_STATUS={status}",
            "ROCM_VERSION=7.2.1",
            "PYTHON_ABI=cp312",
            "PLATFORM=linux/amd64",
            "TORCH_VERSION=2.9.1",
            "TORCH_URL=https://repo.radeon.com/torch.whl",
            f"TORCH_SHA256={'a' * 64}",
            "TORCHVISION_VERSION=0.24.0",
            "TORCHVISION_URL=https://repo.radeon.com/vision.whl",
            f"TORCHVISION_SHA256={'b' * 64}",
            "TORCHAUDIO_VERSION=2.9.0",
            "TORCHAUDIO_URL=https://repo.radeon.com/audio.whl",
            f"TORCHAUDIO_SHA256={'c' * 64}",
            "TRITON_VERSION=3.5.1",
            "TRITON_URL=https://repo.radeon.com/triton.whl",
            f"TRITON_SHA256={'d' * 64}",
        ]
    ) + "\n"


def replace_key(source, key, replacement):
    lines = [line for line in source.splitlines() if not line.startswith(f"{key}=")]
    return "\n".join([*lines, replacement]) + "\n"


def test_complete_verified_profile_preserves_component_order(tmp_path):
    profile_file = tmp_path / "stable.env"
    profile_file.write_text(valid_profile_text(status="verified"), encoding="utf-8")

    profile = load_profile(profile_file, allow_verified=True)

    assert profile.profile_id == "stable-test"
    assert profile.status == "verified"
    assert tuple(profile.wheels) == ("torch", "torchvision", "torchaudio", "triton")
    assert profile.wheels["torch"].version == "2.9.1"


@pytest.mark.parametrize(
    "bad_line",
    [
        "TORCH_URL=http://example.com/torch.whl",
        "TORCH_URL=https://token@example.com/torch.whl",
        "TORCH_URL=https://example.com/torch.whl?token=secret",
        "TORCH_URL=https://example.com/torch.whl#fragment",
        "TORCH_URL=$(curl attacker)",
        "TORCH_URL=${UNTRUSTED}",
        "TORCH_URL=`curl attacker`",
        "TORCH_SHA256=short",
        f"TORCH_SHA256={'0' * 64}",
        "EXTRA_KEY=value",
        "ROCM_VERSION=7.2.4",
        "PYTHON_ABI=cp313",
        "PLATFORM=linux/arm64",
    ],
)
def test_profile_rejects_unsafe_unknown_or_wrong_base_values(tmp_path, bad_line):
    key = bad_line.split("=", 1)[0]
    path = tmp_path / "bad.env"
    path.write_text(
        replace_key(valid_profile_text(), key, bad_line),
        encoding="utf-8",
    )

    with pytest.raises(ProfileError):
        load_profile(path, allow_verified=False)


def test_profile_rejects_duplicates_and_missing_keys(tmp_path):
    duplicate = tmp_path / "duplicate.env"
    duplicate.write_text(
        valid_profile_text() + "TORCH_VERSION=9.9.9\n",
        encoding="utf-8",
    )
    missing = tmp_path / "missing.env"
    missing.write_text(
        "\n".join(
            line
            for line in valid_profile_text().splitlines()
            if not line.startswith("TRITON_SHA256=")
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ProfileError, match="duplicate"):
        load_profile(duplicate, allow_verified=False)
    with pytest.raises(ProfileError, match="missing"):
        load_profile(missing, allow_verified=False)


def test_custom_path_cannot_claim_verified_status(tmp_path):
    path = tmp_path / "custom.env"
    path.write_text(valid_profile_text(status="verified"), encoding="utf-8")

    with pytest.raises(ProfileError, match="verified"):
        load_profile(path, allow_verified=False)


def test_repository_custom_example_is_deliberately_not_buildable():
    with pytest.raises(ProfileError, match="all-zero"):
        load_profile("profiles/torch/custom.example.env", allow_verified=False)

