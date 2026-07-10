from pathlib import Path


def test_torch_image_uses_named_wheel_context_and_manifest():
    text = Path("images/rocm-pytorch/Dockerfile").read_text(encoding="utf-8")

    assert text.startswith("# syntax=docker/dockerfile:1.7")
    assert "FROM ${ROCM_PYTHON_BASE}" in text
    assert "--mount=from=wheels" in text
    assert "COPY --from=profile-context" in text
    assert "--require-hashes" in text
    assert "torch-manifest.py create" in text
    assert "ARG PROFILE_STATUS" in text
    assert "pip cache" not in text


def test_torch_image_labels_are_profile_driven_and_no_wheel_is_copied():
    text = Path("images/rocm-pytorch/Dockerfile").read_text(encoding="utf-8")

    assert 'org.amd-ai.profile.status="${PROFILE_STATUS}"' in text
    assert 'org.amd-ai.profile.id="${PROFILE_ID}"' in text
    assert 'org.amd-ai.torch.version="${TORCH_VERSION}"' in text
    assert "PROFILE_STATUS=verified" not in text
    assert "COPY" not in "\n".join(
        line for line in text.splitlines() if ".whl" in line
    )
    assert "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL" not in text
    assert text.index("ARG VCS_REVISION=unknown") > text.index("uv pip install")
    assert text.index("ARG IMAGE_SOURCE=unknown") > text.index("uv pip install")
