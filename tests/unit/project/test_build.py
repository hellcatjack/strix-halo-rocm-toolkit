from pathlib import Path

from amd_ai.project.build import (
    build_context_fingerprint,
    project_build_argv,
    project_parent_alias,
)


def test_fingerprint_changes_with_lock_not_ignored_models(tmp_path):
    lock = tmp_path / "requirements.lock"
    lock.write_text("alpha==1.0 --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")
    (tmp_path / "models").mkdir()
    model = tmp_path / "models/model.bin"
    model.write_bytes(b"large")

    first = build_context_fingerprint(tmp_path)
    model.write_bytes(b"changed")
    assert build_context_fingerprint(tmp_path) == first
    lock.write_text("alpha==1.1 --hash=sha256:" + "b" * 64 + "\n", encoding="utf-8")
    assert build_context_fingerprint(tmp_path) != first


def test_fingerprint_tracks_regular_source_mode_and_content(tmp_path):
    source = tmp_path / "app.py"
    source.write_text("print('one')\n", encoding="utf-8")
    first = build_context_fingerprint(tmp_path)
    source.chmod(0o755)
    second = build_context_fingerprint(tmp_path)
    source.write_text("print('two')\n", encoding="utf-8")

    assert second != first
    assert build_context_fingerprint(tmp_path) != second


def test_build_uses_content_addressed_parent_alias_and_labels():
    parent = "sha256:" + "a" * 64
    argv = project_build_argv(
        context=Path("projects/demo"),
        image="demo:runtime",
        base_image=parent,
        base_digest=parent,
        profile_id="rocm-7.2.1-py3.12-torch-2.9.1",
        profile_status="verified",
        fingerprint="f" * 64,
    )

    alias = "amd-ai-local/project-base:" + "a" * 64
    assert project_parent_alias(parent) == alias
    assert f"BASE_IMAGE={alias}" in argv
    assert "PROFILE_STATUS=verified" in argv
    assert "org.amd-ai.project.fingerprint=" + "f" * 64 in argv
    assert "org.amd-ai.base.digest=" + parent in argv
    assert "--load" in argv
