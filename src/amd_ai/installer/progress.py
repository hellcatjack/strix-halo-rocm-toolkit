from __future__ import annotations

import os
import re
import threading
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TextIO

from amd_ai.installer.state import project_identity_key


CSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
OSC_PATTERN = re.compile(r"\x1b\].*?(?:\x07|\x1b\\)", re.DOTALL)
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
URL_USERINFO_PATTERN = re.compile(
    r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)[^\s/@]+@"
)
SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r"(?P<name>\b[A-Za-z_][A-Za-z0-9_]*"
    r"(?:TOKEN|PASSWORD|PASS|SECRET|KEY|CREDENTIAL|AUTH))=[^\s]+",
    re.IGNORECASE,
)
AUTHORIZATION_PATTERN = re.compile(
    r"(?P<label>\bAuthorization\s*:\s*(?:Bearer|Basic)\s+)[^\s]+",
    re.IGNORECASE,
)
CREDENTIAL_FLAG_PATTERN = re.compile(
    r"(?P<flag>--(?:token|password|secret|key|credential|auth|index-url)"
    r"(?:=|\s+))[^\s]+",
    re.IGNORECASE,
)


class ProgressError(RuntimeError):
    pass


class ProgressMode(StrEnum):
    DEFAULT = "default"
    VERBOSE = "verbose"
    QUIET = "quiet"


def default_log_root() -> Path:
    return Path.home() / ".local/state/strix-halo-rocm-toolkit/logs"


def sanitize_output(
    value: str, *, secret_values: Iterable[str] = ()
) -> str:
    rendered = value.replace("\r\n", "\n").replace("\r", "\n")
    rendered = OSC_PATTERN.sub("", rendered)
    rendered = CSI_PATTERN.sub("", rendered)
    rendered = CONTROL_PATTERN.sub("", rendered)
    secrets = {item for item in secret_values if item}
    for secret in sorted(secrets, key=len, reverse=True):
        rendered = rendered.replace(secret, "<redacted>")
    rendered = URL_USERINFO_PATTERN.sub(
        r"\g<scheme><redacted>@", rendered
    )
    rendered = SENSITIVE_ASSIGNMENT_PATTERN.sub(
        r"\g<name>=<redacted>", rendered
    )
    rendered = AUTHORIZATION_PATTERN.sub(
        r"\g<label><redacted>", rendered
    )
    return CREDENTIAL_FLAG_PATTERN.sub(r"\g<flag><redacted>", rendered)


class SessionLog:
    def __init__(
        self,
        *,
        path: Path,
        stream: TextIO,
        wall_clock: Callable[[], datetime],
    ) -> None:
        self.path = path
        self._stream = stream
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._closed = False

    @classmethod
    def open(
        cls,
        *,
        project_dir: Path,
        log_root: Path,
        wall_clock: Callable[[], datetime],
        process_id: int,
    ) -> SessionLog:
        root = Path(log_root).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        _ensure_private_directory(root.parent, parents=True)
        _ensure_private_directory(root)
        project_root = root / project_identity_key(project_dir)
        _ensure_private_directory(project_root)

        now = _utc_now(wall_clock)
        timestamp = now.strftime("%Y%m%dT%H%M%SZ")
        path = project_root / f"install-{timestamp}-{process_id}.log"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError as error:
            raise ProgressError(
                f"installer log already exists: {path}"
            ) from error
        except OSError as error:
            raise ProgressError(
                f"cannot create installer log {path}: {error}"
            ) from error
        try:
            os.fchmod(descriptor, 0o600)
            stream = os.fdopen(
                descriptor,
                "w",
                encoding="utf-8",
                newline="\n",
            )
        except Exception:
            os.close(descriptor)
            path.unlink(missing_ok=True)
            raise
        return cls(path=path, stream=stream, wall_clock=wall_clock)

    def write(self, stream: str, kind: str, text: str) -> None:
        rendered = sanitize_output(text)
        with self._lock:
            if self._closed:
                raise ProgressError("installer log is closed")
            timestamp = _utc_now(self._wall_clock).isoformat(
                timespec="milliseconds"
            ).replace("+00:00", "Z")
            lines = rendered.split("\n")
            if lines and lines[-1] == "":
                lines.pop()
            if not lines:
                lines = [""]
            try:
                for line in lines:
                    self._stream.write(
                        f"{timestamp} {stream} {kind} {line}\n"
                    )
                self._stream.flush()
            except OSError as error:
                raise ProgressError(
                    f"cannot write installer log {self.path}: {error}"
                ) from error

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._stream.flush()
                os.fsync(self._stream.fileno())
            except OSError as error:
                raise ProgressError(
                    f"cannot flush installer log {self.path}: {error}"
                ) from error
            finally:
                self._stream.close()
                self._closed = True


def _ensure_private_directory(path: Path, *, parents: bool = False) -> None:
    if os.path.lexists(path):
        if path.is_symlink():
            raise ProgressError(
                f"installer log directory is a symlink: {path}"
            )
        if not path.is_dir():
            raise ProgressError(
                f"installer log control path is not a directory: {path}"
            )
    else:
        try:
            path.mkdir(parents=parents, mode=0o700)
        except OSError as error:
            raise ProgressError(
                f"cannot create installer log directory {path}: {error}"
            ) from error
    try:
        metadata = path.stat(follow_symlinks=False)
        if metadata.st_uid != os.geteuid():
            raise ProgressError(
                f"installer log directory is not owned by current user: {path}"
            )
        path.chmod(0o700, follow_symlinks=False)
    except OSError as error:
        raise ProgressError(
            f"cannot secure installer log directory {path}: {error}"
        ) from error


def _utc_now(wall_clock: Callable[[], datetime]) -> datetime:
    value = wall_clock()
    if value.tzinfo is None:
        raise ProgressError("installer log clock must be timezone-aware")
    return value.astimezone(UTC)
