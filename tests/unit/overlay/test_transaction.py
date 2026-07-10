from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from amd_ai.overlay.models import (
    OverlayPaths,
    ProtectedComponent,
    ProtectedProfile,
)
from amd_ai.overlay.transaction import (
    TransactionError,
    activate_generation,
    build_generation,
    initialize_overlay,
    install_argv,
    mark_generation_healthy,
    overlay_transaction,
    resolve_current_generation,
)
from amd_ai.runner import CommandResult


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


def make_paths(tmp_path: Path) -> OverlayPaths:
    project = tmp_path / "demo"
    project.mkdir()
    return OverlayPaths.for_project(project)


def make_generation(paths: OverlayPaths, transaction_id: str) -> Path:
    generation = paths.generation(transaction_id)
    (generation / "site-packages").mkdir(parents=True)
    return generation


def test_activation_uses_relative_symlink_inside_control_root(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    generation = make_generation(paths, "20260710T120000Z-a1b2c3d4")

    activate_generation(paths, generation)

    assert paths.current.is_symlink()
    assert os.readlink(paths.current) == "generations/20260710T120000Z-a1b2c3d4"
    assert resolve_current_generation(paths) == generation


def test_activation_failure_keeps_previous_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = make_paths(tmp_path)
    old = make_generation(paths, "20260710T120000Z-a1b2c3d4")
    new = make_generation(paths, "20260710T120001Z-b1c2d3e4")
    activate_generation(paths, old)

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("stop")

    monkeypatch.setattr("amd_ai.overlay.transaction.os.replace", fail_replace)

    with pytest.raises(TransactionError, match="activate"):
        activate_generation(paths, new)

    assert os.readlink(paths.current) == "generations/20260710T120000Z-a1b2c3d4"


def test_transaction_lock_rejects_concurrent_writer(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)

    with overlay_transaction(paths):
        with pytest.raises(TransactionError, match="already in progress"):
            with overlay_transaction(paths):
                pass


def test_generation_can_build_while_caller_holds_transaction_lock(
    tmp_path: Path, profile: ProtectedProfile
) -> None:
    paths = make_paths(tmp_path)

    with overlay_transaction(paths):
        state = build_generation(
            paths,
            profile=profile,
            input_text="",
            lock_text="",
            runner=FakeRunner(returncode=0),
            verifier=lambda path: None,
            transaction_id="20260710T120000Z-a1b2c3d4",
            acquire_lock=False,
        )

    assert resolve_current_generation(paths).name == state.generation_id


def test_overlay_can_initialize_while_caller_holds_transaction_lock(
    tmp_path: Path, profile: ProtectedProfile
) -> None:
    paths = make_paths(tmp_path)

    with overlay_transaction(paths):
        state = initialize_overlay(
            paths,
            profile=profile,
            transaction_id="20260710T120000Z-a1b2c3d4",
            acquire_lock=False,
        )

    assert resolve_current_generation(paths).name == state.generation_id


def test_initialize_overlay_creates_one_valid_empty_generation(
    tmp_path: Path, profile: ProtectedProfile
) -> None:
    paths = make_paths(tmp_path)

    state = initialize_overlay(
        paths,
        profile=profile,
        transaction_id="20260710T120000Z-a1b2c3d4",
    )

    assert state.lock_digest == hashlib.sha256(b"").hexdigest()
    assert resolve_current_generation(paths).name == state.generation_id
    assert (resolve_current_generation(paths) / "site-packages").is_dir()
    assert initialize_overlay(paths, profile=profile) == state


class FakeRunner:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode
        self.calls: list[tuple[str, ...]] = []

    def run(
        self,
        args: list[str],
        *,
        environment: dict[str, str],
        cwd: Path | None = None,
    ) -> CommandResult:
        command = tuple(args)
        self.calls.append(command)
        return CommandResult(command, self.returncode, "out", "failed")


def test_generation_install_failure_leaves_current_unchanged(
    tmp_path: Path, profile: ProtectedProfile
) -> None:
    paths = make_paths(tmp_path)
    old = initialize_overlay(
        paths,
        profile=profile,
        transaction_id="20260710T120000Z-a1b2c3d4",
    )
    runner = FakeRunner(returncode=1)

    with pytest.raises(TransactionError, match="install failed"):
        build_generation(
            paths,
            profile=profile,
            input_text="demo==1.0\n",
            lock_text="invalid but nonempty\n",
            runner=runner,
            verifier=lambda path: None,
            transaction_id="20260710T120001Z-b1c2d3e4",
            validate_artifacts=False,
        )

    assert resolve_current_generation(paths).name == old.generation_id


def test_generation_verification_failure_leaves_current_unchanged(
    tmp_path: Path, profile: ProtectedProfile
) -> None:
    paths = make_paths(tmp_path)
    old = initialize_overlay(
        paths,
        profile=profile,
        transaction_id="20260710T120000Z-a1b2c3d4",
    )

    def fail_verify(path: Path) -> None:
        raise RuntimeError("shadow")

    with pytest.raises(TransactionError, match="verification failed"):
        build_generation(
            paths,
            profile=profile,
            input_text="",
            lock_text="",
            runner=FakeRunner(returncode=0),
            verifier=fail_verify,
            transaction_id="20260710T120001Z-b1c2d3e4",
        )

    assert resolve_current_generation(paths).name == old.generation_id


def test_install_argv_targets_only_candidate_site_packages(tmp_path: Path) -> None:
    assert install_argv(tmp_path / "lock", tmp_path / "site") == (
        "/opt/venv/bin/python",
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-index",
        "--no-deps",
        "--no-compile",
        "--require-hashes",
        "--target",
        str(tmp_path / "site"),
        "--requirement",
        str(tmp_path / "lock"),
    )


def test_healthy_generation_retention_keeps_current_and_previous(
    tmp_path: Path, profile: ProtectedProfile
) -> None:
    paths = make_paths(tmp_path)
    first = initialize_overlay(
        paths,
        profile=profile,
        transaction_id="20260710T120000Z-a1b2c3d4",
    )
    assert mark_generation_healthy(paths).generation_id == first.generation_id
    second = build_generation(
        paths,
        profile=profile,
        input_text="",
        lock_text="",
        runner=FakeRunner(returncode=0),
        verifier=lambda path: None,
        transaction_id="20260710T120001Z-b1c2d3e4",
    )
    mark_generation_healthy(paths)
    third = build_generation(
        paths,
        profile=profile,
        input_text="",
        lock_text="",
        runner=FakeRunner(returncode=0),
        verifier=lambda path: None,
        transaction_id="20260710T120002Z-c1d2e3f4",
    )

    state = mark_generation_healthy(paths)

    assert state.generation_id == third.generation_id
    assert state.healthy is True
    retained = {path.name for path in paths.generations.iterdir() if path.is_dir()}
    assert retained == {second.generation_id, third.generation_id}


def test_mirror_failure_keeps_valid_new_generation_active(
    tmp_path: Path,
    profile: ProtectedProfile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = make_paths(tmp_path)
    initialize_overlay(
        paths,
        profile=profile,
        transaction_id="20260710T120000Z-a1b2c3d4",
    )

    def fail_mirror(paths: OverlayPaths, generation: Path) -> None:
        raise TransactionError("mirror disk error")

    monkeypatch.setattr(
        "amd_ai.overlay.transaction._mirror_metadata", fail_mirror
    )

    state = build_generation(
        paths,
        profile=profile,
        input_text="",
        lock_text="",
        runner=FakeRunner(returncode=0),
        verifier=lambda path: None,
        transaction_id="20260710T120001Z-b1c2d3e4",
    )

    generation = resolve_current_generation(paths)
    assert generation.name == state.generation_id
    assert (generation / "mirror-warning.txt").is_file()
