from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import shutil
import stat
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from amd_ai.overlay.lock import (
    LockError,
    lock_digest,
    parse_lock,
    validate_lock_artifacts,
)
from amd_ai.overlay.models import (
    GENERATION_PATTERN,
    OverlayError,
    OverlayPaths,
    OverlayState,
    ProtectedProfile,
)
from amd_ai.overlay.resolver import ProcessRunner


class TransactionError(RuntimeError):
    pass


def install_argv(lock_path: Path, site_packages: Path) -> tuple[str, ...]:
    return (
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
        str(site_packages),
        "--requirement",
        str(lock_path),
    )


@contextmanager
def overlay_transaction(paths: OverlayPaths) -> Iterator[None]:
    ensure_layout(paths)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(paths.transaction_lock, flags, 0o600)
    except OSError as error:
        raise TransactionError(
            f"cannot open overlay transaction lock: {error}"
        ) from error
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise TransactionError(
                "another overlay transaction is already in progress"
            ) from error
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def ensure_layout(paths: OverlayPaths) -> None:
    if paths.root.is_symlink():
        raise TransactionError(
            f"overlay control path must not be a symlink: {paths.root}"
        )
    for path in (
        paths.root,
        paths.generations,
        paths.artifacts.parent,
        paths.artifacts,
        paths.quarantine,
        paths.logs,
    ):
        if path.is_symlink():
            raise TransactionError(
                f"overlay control path must not be a symlink: {path}"
            )
        try:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            path.chmod(0o700)
            metadata = path.stat()
        except OSError as error:
            raise TransactionError(f"cannot prepare overlay path {path}: {error}") from error
        if not stat.S_ISDIR(metadata.st_mode):
            raise TransactionError(f"overlay control path is not a directory: {path}")


def initialize_overlay(
    paths: OverlayPaths,
    *,
    profile: ProtectedProfile,
    transaction_id: str | None = None,
) -> OverlayState:
    with overlay_transaction(paths):
        if paths.current.is_symlink():
            try:
                generation = resolve_current_generation(paths)
            except TransactionError:
                raise
            state = load_generation_state(generation)
            _require_state_profile(state, profile)
            return state
        if paths.current.exists():
            raise TransactionError("overlay current path is not a symbolic link")
        existing = tuple(
            path
            for path in paths.generations.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
        if existing:
            raise TransactionError(
                "overlay has generations but no current link; run doctor and repair"
            )
        generation_id = transaction_id or new_transaction_id()
        generation = _create_generation(paths, generation_id)
        input_text = ""
        lock_text = ""
        state = _state_for(
            generation_id,
            input_text=input_text,
            lock_text=lock_text,
            profile=profile,
        )
        _write_generation_metadata(generation, input_text, lock_text, state)
        activate_generation(paths, generation)
        _mirror_metadata(paths, generation)
        return state


def build_generation(
    paths: OverlayPaths,
    *,
    profile: ProtectedProfile,
    input_text: str,
    lock_text: str,
    runner: ProcessRunner,
    verifier: Callable[[Path], None],
    transaction_id: str | None = None,
    validate_artifacts: bool = True,
    base_environment: Mapping[str, str] | None = None,
) -> OverlayState:
    with overlay_transaction(paths):
        generation_id = transaction_id or new_transaction_id()
        generation = _create_generation(paths, generation_id)
        site_packages = generation / "site-packages"
        lock_path = generation / "overlay.requirements.lock"
        _atomic_write(lock_path, lock_text, mode=0o600)
        _atomic_write(
            generation / "overlay.requirements.in", input_text, mode=0o600
        )
        if validate_artifacts:
            try:
                locked = parse_lock(lock_text)
                validate_lock_artifacts(locked, project=paths.project)
            except LockError as error:
                _write_failure(generation, f"lock validation failed: {error}")
                raise TransactionError(f"overlay lock validation failed: {error}") from error
        if lock_text:
            environment = dict(
                os.environ if base_environment is None else base_environment
            )
            environment.update(
                {
                    "PYTHONPATH": f"{site_packages}:/opt/amd-ai/src",
                    "PYTHONNOUSERSITE": "1",
                    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                }
            )
            result = runner.run(
                list(install_argv(lock_path, site_packages)),
                environment=environment,
                cwd=Path("/workspace"),
            )
            if result.returncode != 0:
                evidence = result.stderr.strip() or result.stdout.strip() or "no output"
                _write_failure(generation, f"install failed: {evidence}")
                raise TransactionError(f"overlay install failed: {evidence}")
        try:
            verifier(site_packages)
        except Exception as error:
            _write_failure(generation, f"verification failed: {error}")
            raise TransactionError(
                f"overlay verification failed: {error}"
            ) from error
        state = _state_for(
            generation_id,
            input_text=input_text,
            lock_text=lock_text,
            profile=profile,
        )
        _atomic_write(
            generation / "overlay-state.json",
            _state_json(state),
            mode=0o600,
        )
        _fsync_directory(generation)
        activate_generation(paths, generation)
        try:
            _mirror_metadata(paths, generation)
        except TransactionError as error:
            _atomic_write(
                generation / "mirror-warning.txt",
                f"{error}\n",
                mode=0o600,
            )
        return state


def activate_generation(paths: OverlayPaths, generation: Path) -> None:
    ensure_layout(paths)
    try:
        resolved_generation = generation.resolve(strict=True)
        generations_root = paths.generations.resolve(strict=True)
    except OSError as error:
        raise TransactionError(f"cannot inspect generation: {error}") from error
    if (
        resolved_generation.parent != generations_root
        or GENERATION_PATTERN.fullmatch(resolved_generation.name) is None
        or not (resolved_generation / "site-packages").is_dir()
    ):
        raise TransactionError(
            f"generation is outside the overlay boundary: {generation}"
        )
    relative_target = PurePosixPath("generations") / resolved_generation.name
    temporary = paths.root / f".current.{resolved_generation.name}.tmp"
    if temporary.exists() or temporary.is_symlink():
        if not temporary.is_symlink():
            raise TransactionError(f"temporary current path is unsafe: {temporary}")
        temporary.unlink()
    try:
        temporary.symlink_to(relative_target)
        os.replace(temporary, paths.current)
        _fsync_directory(paths.root)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise TransactionError(f"cannot activate generation: {error}") from error


def resolve_current_generation(paths: OverlayPaths) -> Path:
    if not paths.current.is_symlink():
        raise TransactionError("overlay current is not a symbolic link")
    try:
        raw_target = os.readlink(paths.current)
    except OSError as error:
        raise TransactionError(f"cannot read overlay current: {error}") from error
    target = PurePosixPath(raw_target)
    if target.is_absolute() or target.parts[:1] != ("generations",) or len(target.parts) != 2:
        raise TransactionError("overlay current target is outside generations")
    if GENERATION_PATTERN.fullmatch(target.name) is None:
        raise TransactionError("overlay current target has an invalid generation ID")
    try:
        generation = (paths.root / target).resolve(strict=True)
        generations_root = paths.generations.resolve(strict=True)
    except OSError as error:
        raise TransactionError(f"overlay current target is missing: {error}") from error
    if generation.parent != generations_root:
        raise TransactionError("overlay current target escaped generations")
    if not (generation / "site-packages").is_dir():
        raise TransactionError("overlay current generation has no site-packages")
    return generation


def load_generation_state(generation: Path) -> OverlayState:
    path = generation / "overlay-state.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TransactionError(f"cannot read overlay state: {error}") from error
    expected = {
        "schema_version",
        "generation_id",
        "input_digest",
        "lock_digest",
        "profile_id",
        "parent_config_digest",
        "created_at",
        "healthy",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise TransactionError("overlay state keys are invalid")
    try:
        return OverlayState(**payload)
    except (OverlayError, TypeError) as error:
        raise TransactionError(f"overlay state is invalid: {error}") from error


def mark_generation_healthy(paths: OverlayPaths) -> OverlayState:
    with overlay_transaction(paths):
        current = resolve_current_generation(paths)
        state = load_generation_state(current)
        if not state.healthy:
            state = replace(state, healthy=True)
            _atomic_write(
                current / "overlay-state.json",
                _state_json(state),
                mode=0o600,
            )
            _mirror_metadata(paths, current)

        healthy: list[tuple[OverlayState, Path]] = []
        for generation in paths.generations.iterdir():
            if generation.is_symlink():
                raise TransactionError(
                    f"generation must not be a symlink: {generation}"
                )
            if not generation.is_dir() or GENERATION_PATTERN.fullmatch(generation.name) is None:
                continue
            try:
                candidate_state = load_generation_state(generation)
            except TransactionError:
                continue
            if candidate_state.healthy:
                healthy.append((candidate_state, generation))
        healthy.sort(key=lambda item: item[0].generation_id)
        keep = {item[0].generation_id for item in healthy[-2:]}
        for candidate_state, generation in healthy:
            if candidate_state.generation_id in keep:
                continue
            resolved = generation.resolve(strict=True)
            if resolved.parent != paths.generations.resolve(strict=True):
                raise TransactionError(
                    f"generation cleanup target escaped boundary: {generation}"
                )
            shutil.rmtree(resolved)
        _garbage_collect_artifacts(paths, keep)
        return state


def new_transaction_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{secrets.token_hex(4)}"


def _create_generation(paths: OverlayPaths, generation_id: str) -> Path:
    try:
        generation = paths.generation(generation_id)
    except OverlayError as error:
        raise TransactionError(str(error)) from error
    if generation.exists() or generation.is_symlink():
        raise TransactionError(f"generation already exists: {generation_id}")
    try:
        generation.mkdir(mode=0o700)
        (generation / "site-packages").mkdir(mode=0o700)
    except OSError as error:
        raise TransactionError(f"cannot create generation: {error}") from error
    return generation


def _write_generation_metadata(
    generation: Path,
    input_text: str,
    lock_text: str,
    state: OverlayState,
) -> None:
    _atomic_write(
        generation / "overlay.requirements.in", input_text, mode=0o600
    )
    _atomic_write(
        generation / "overlay.requirements.lock", lock_text, mode=0o600
    )
    _atomic_write(
        generation / "overlay-state.json", _state_json(state), mode=0o600
    )
    _fsync_directory(generation)


def _mirror_metadata(paths: OverlayPaths, generation: Path) -> None:
    for source_name, destination in (
        ("overlay.requirements.in", paths.inputs),
        ("overlay.requirements.lock", paths.lock),
        ("overlay-state.json", paths.state),
    ):
        try:
            content = (generation / source_name).read_text(encoding="utf-8")
        except OSError as error:
            raise TransactionError(
                f"cannot read generation metadata for mirror: {error}"
            ) from error
        _atomic_write(destination, content, mode=0o600)


def _state_for(
    generation_id: str,
    *,
    input_text: str,
    lock_text: str,
    profile: ProtectedProfile,
) -> OverlayState:
    return OverlayState(
        generation_id=generation_id,
        input_digest=hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
        lock_digest=lock_digest(lock_text),
        profile_id=profile.profile_id,
        parent_config_digest=profile.parent_config_digest,
        created_at=datetime.now(UTC).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
    )


def _require_state_profile(
    state: OverlayState, profile: ProtectedProfile
) -> None:
    if (
        state.profile_id != profile.profile_id
        or state.parent_config_digest != profile.parent_config_digest
    ):
        raise TransactionError(
            "overlay generation belongs to another protected parent profile"
        )


def _garbage_collect_artifacts(
    paths: OverlayPaths, retained_generation_ids: set[str]
) -> None:
    referenced: set[str] = set()
    for generation_id in retained_generation_ids:
        generation = paths.generation(generation_id)
        try:
            lock_text = (generation / "overlay.requirements.lock").read_text(
                encoding="utf-8"
            )
            locked = parse_lock(lock_text)
        except (OSError, LockError) as error:
            raise TransactionError(
                f"cannot validate retained generation lock: {error}"
            ) from error
        referenced.update(wheel.sha256 for wheel in locked)
    for candidate in paths.artifacts.iterdir():
        if candidate.name in referenced:
            continue
        if candidate.is_symlink():
            raise TransactionError(
                f"artifact cleanup target must not be a symlink: {candidate}"
            )
        if not candidate.is_dir() or len(candidate.name) != 64 or any(
            character not in "0123456789abcdef" for character in candidate.name
        ):
            continue
        resolved = candidate.resolve(strict=True)
        if resolved.parent != paths.artifacts.resolve(strict=True):
            raise TransactionError(
                f"artifact cleanup target escaped boundary: {candidate}"
            )
        shutil.rmtree(resolved)


def _state_json(state: OverlayState) -> str:
    return json.dumps(asdict(state), indent=2, sort_keys=True) + "\n"


def _write_failure(generation: Path, message: str) -> None:
    _atomic_write(generation / "transaction-failure.txt", message + "\n", mode=0o600)


def _atomic_write(path: Path, content: str, *, mode: int) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            mode,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
