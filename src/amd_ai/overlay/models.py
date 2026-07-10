from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


PROTECTED_DISTRIBUTIONS = frozenset(
    {"torch", "torchvision", "torchaudio", "triton"}
)
GENERATION_PATTERN = re.compile(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}")
DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


class OverlayError(RuntimeError):
    pass


def canonicalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def canonicalize_protected_name(name: str) -> str:
    canonical = canonicalize_name(name)
    collapsed = canonical.replace("-", "")
    for protected in PROTECTED_DISTRIBUTIONS:
        if protected.replace("-", "") == collapsed:
            return protected
    return canonical


@dataclass(frozen=True)
class OverlayPaths:
    project: Path
    root: Path
    inputs: Path
    lock: Path
    state: Path
    transaction_lock: Path
    generations: Path
    current: Path
    quarantine: Path
    artifacts: Path
    logs: Path

    @classmethod
    def for_project(cls, project: Path) -> OverlayPaths:
        resolved = project.resolve()
        if not resolved.is_dir():
            raise OverlayError(f"project directory does not exist: {resolved}")
        root = resolved / ".amd-ai"
        if root.is_symlink():
            raise OverlayError(
                f"overlay root must not be a symbolic link: {root}"
            )
        return cls(
            project=resolved,
            root=root,
            inputs=root / "overlay.requirements.in",
            lock=root / "overlay.requirements.lock",
            state=root / "overlay-state.json",
            transaction_lock=root / "transaction.lock",
            generations=root / "generations",
            current=root / "current",
            quarantine=root / "quarantine",
            artifacts=root / "artifacts" / "sha256",
            logs=root / "logs",
        )

    def generation(self, transaction_id: str) -> Path:
        if GENERATION_PATTERN.fullmatch(transaction_id) is None:
            raise OverlayError(f"invalid transaction ID: {transaction_id!r}")
        return self.generations / transaction_id


@dataclass(frozen=True)
class ProtectedComponent:
    name: str
    version: str

    def __post_init__(self) -> None:
        canonical = canonicalize_protected_name(self.name)
        if canonical not in PROTECTED_DISTRIBUTIONS:
            raise OverlayError(f"unknown protected distribution: {self.name}")
        if not self.version or "+" not in self.version:
            raise OverlayError(
                f"protected distribution requires a full local version: {self.name}"
            )
        object.__setattr__(self, "name", canonical)


@dataclass(frozen=True)
class ProtectedProfile:
    profile_id: str
    parent_config_digest: str
    components: tuple[ProtectedComponent, ...]

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise OverlayError("protected profile ID is empty")
        if DIGEST_PATTERN.fullmatch(self.parent_config_digest) is None:
            raise OverlayError("protected parent config digest is invalid")
        names = tuple(component.name for component in self.components)
        if set(names) != PROTECTED_DISTRIBUTIONS or len(names) != 4:
            raise OverlayError(
                "protected profile must contain each component once"
            )

    def version_for(self, name: str) -> str:
        canonical = canonicalize_protected_name(name)
        for component in self.components:
            if component.name == canonical:
                return component.version
        raise OverlayError(f"distribution is not protected: {name}")


@dataclass(frozen=True)
class OverlayState:
    generation_id: str
    input_digest: str
    lock_digest: str
    profile_id: str
    parent_config_digest: str
    created_at: str
    healthy: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        if GENERATION_PATTERN.fullmatch(self.generation_id) is None:
            raise OverlayError("overlay state generation ID is invalid")
        for label, value in (
            ("input", self.input_digest),
            ("lock", self.lock_digest),
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise OverlayError(f"overlay state {label} digest is invalid")
        if not self.profile_id:
            raise OverlayError("overlay state profile ID is empty")
        if DIGEST_PATTERN.fullmatch(self.parent_config_digest) is None:
            raise OverlayError("overlay state parent config digest is invalid")
        if not self.created_at.endswith("Z"):
            raise OverlayError("overlay state timestamp is not UTC")
