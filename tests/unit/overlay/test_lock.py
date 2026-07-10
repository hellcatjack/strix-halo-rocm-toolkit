from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from amd_ai.overlay.lock import (
    LockError,
    lock_digest,
    parse_lock,
    render_lock,
    validate_lock_artifacts,
)
from amd_ai.overlay.resolver import WheelArtifact


def artifact(
    project: Path,
    *,
    name: str = "requests",
    version: str = "2.32.5",
    content: bytes = b"wheel",
    requested: bool = True,
) -> WheelArtifact:
    digest = hashlib.sha256(content).hexdigest()
    filename = f"{name}-{version}-py3-none-any.whl"
    path = project / ".amd-ai/artifacts/sha256" / digest / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return WheelArtifact(name, version, digest, path, requested)


def test_lock_is_sorted_hashed_and_project_local(tmp_path: Path) -> None:
    wheel = artifact(tmp_path)

    text = render_lock((wheel,), project=tmp_path)

    assert text == (
        "requests @ file:///workspace/.amd-ai/artifacts/sha256/"
        + wheel.sha256
        + "/requests-2.32.5-py3-none-any.whl \\\n"
        + "    --hash=sha256:"
        + wheel.sha256
        + "\n"
    )
    locked = parse_lock(text)
    assert locked[0].name == "requests"
    assert locked[0].version == "2.32.5"
    assert validate_lock_artifacts(locked, project=tmp_path) == (
        wheel.path,
    )
    assert lock_digest(text) == hashlib.sha256(text.encode()).hexdigest()


def test_lock_rejects_protected_distribution() -> None:
    digest = "a" * 64
    text = (
        "torch @ file:///workspace/.amd-ai/artifacts/sha256/"
        f"{digest}/torch-2.9.1-py3-none-any.whl \\\n"
        f"    --hash=sha256:{digest}\n"
    )

    with pytest.raises(LockError, match="protected"):
        parse_lock(text)


def test_render_rejects_artifact_outside_project(tmp_path: Path) -> None:
    outside = tmp_path.parent / "demo-1.0-py3-none-any.whl"
    outside.write_bytes(b"wheel")
    digest = hashlib.sha256(b"wheel").hexdigest()
    wheel = WheelArtifact("demo", "1.0", digest, outside, True)

    with pytest.raises(LockError, match="artifact store"):
        render_lock((wheel,), project=tmp_path)


def test_lock_rejects_duplicate_or_unsorted_packages(tmp_path: Path) -> None:
    first = artifact(tmp_path, name="zeta", version="1.0", content=b"z")
    second = artifact(tmp_path, name="alpha", version="1.0", content=b"a")
    text = render_lock((first, second), project=tmp_path)
    blocks = text.splitlines(keepends=True)
    unsorted = "".join(blocks[2:] + blocks[:2])

    with pytest.raises(LockError, match="sorted"):
        parse_lock(unsorted)
    with pytest.raises(LockError, match="duplicate"):
        render_lock((first, first), project=tmp_path)


def test_artifact_validation_detects_changed_bytes(tmp_path: Path) -> None:
    wheel = artifact(tmp_path)
    locked = parse_lock(render_lock((wheel,), project=tmp_path))
    wheel.path.chmod(0o644)
    wheel.path.write_bytes(b"changed")

    with pytest.raises(LockError, match="hash"):
        validate_lock_artifacts(locked, project=tmp_path)


def test_render_rejects_symlinked_artifact_store(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    control = tmp_path / ".amd-ai"
    control.mkdir()
    (control / "artifacts").symlink_to(outside, target_is_directory=True)
    content = b"wheel"
    digest = hashlib.sha256(content).hexdigest()
    wheel_path = outside / "sha256" / digest / "demo-1.0-py3-none-any.whl"
    wheel_path.parent.mkdir(parents=True)
    wheel_path.write_bytes(content)
    wheel = WheelArtifact("demo", "1.0", digest, wheel_path, True)

    with pytest.raises(LockError, match="symlink"):
        render_lock((wheel,), project=tmp_path)
