from pathlib import Path


def test_project_install_cannot_sync_or_replace_parent():
    text = Path("templates/project/Dockerfile").read_text(encoding="utf-8")

    assert "uv pip install" in text
    assert "--constraint /opt/amd-ai/project-locks/torch-constraints.txt" in text
    assert "torch-manifest.py verify" in text
    assert "AMD_AI_PROFILE_STATUS" in text
    assert "uv pip sync" not in text
    assert "pip uninstall" not in text
    assert "--mount=type=cache,target=/root/.cache/uv,sharing=locked" in text
