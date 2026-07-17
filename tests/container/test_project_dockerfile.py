from pathlib import Path


def test_project_install_cannot_sync_or_replace_parent():
    text = Path("templates/project/Dockerfile").read_text(encoding="utf-8")

    assert "uv pip install" in text
    assert "--constraint /opt/amd-ai/project-locks/torch-constraints.txt" in text
    assert "--require-hashes" in text
    assert "torch-manifest.py verify" in text
    assert "AMD_AI_PROFILE_STATUS" in text
    assert "uv pip sync" not in text
    assert "pip uninstall" not in text
    assert not text.startswith("# syntax=")
    assert "docker.io/docker/dockerfile" not in text
    assert "--mount=" not in text
    assert (
        "RUN if [ -s /opt/amd-ai/project-locks/requirements.lock ]; then"
        in text
    )


def test_project_entrypoint_checks_and_marks_overlay_under_startup_lock():
    text = Path("templates/project/project-entrypoint").read_text(
        encoding="utf-8"
    )

    assert "fcntl.flock" in text
    assert "fcntl.LOCK_EX | fcntl.LOCK_NB" in text
    assert "OVERLAY.TRANSACTION_INCOMPLETE" in text
    assert "mark_generation_healthy" in text
    assert "acquire_lock=False" in text
    assert text.index("fcntl.LOCK_UN") < text.index("os.execvp")
