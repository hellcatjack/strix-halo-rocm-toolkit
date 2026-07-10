from __future__ import annotations

import json
import math
import re
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


REQUIRED_CHECKS = (
    "rocm",
    "torch-fp16",
    "hip",
    "torch-extension",
    "triton",
    "repeated-start",
    "stress",
    "kernel-log",
)
PROFILE_KEYS = frozenset(
    {
        "profile_id",
        "image",
        "rocm_version",
        "torch_version",
        "gpu_arch",
        "stress_seconds",
        "repeated_starts",
        "required_checks",
    }
)
IDENTIFIER_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
GPU_ARCH_PATTERN = re.compile(r"gfx[0-9a-f]+")


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class QualificationProfile:
    profile_id: str
    image: str
    rocm_version: str
    torch_version: str
    gpu_arch: str
    stress_seconds: int
    repeated_starts: int
    required_checks: tuple[str, ...]


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    duration_seconds: float
    details: Mapping[str, object]
    evidence: str

    def __post_init__(self) -> None:
        if not self.name or any(character.isspace() for character in self.name):
            raise ValueError("check name must be nonempty and contain no whitespace")
        if not isinstance(self.passed, bool):
            raise ValueError("check passed value must be a boolean")
        if (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or not math.isfinite(float(self.duration_seconds))
            or self.duration_seconds < 0
        ):
            raise ValueError("check duration must be a finite nonnegative number")
        if not isinstance(self.details, Mapping):
            raise ValueError("check details must be a mapping")
        if not isinstance(self.evidence, str):
            raise ValueError("check evidence must be a string")
        object.__setattr__(self, "duration_seconds", float(self.duration_seconds))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "duration_seconds": self.duration_seconds,
            "details": dict(self.details),
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class QualificationReport:
    profile_id: str
    status: str
    results: tuple[CheckResult, ...]
    required_checks: tuple[str, ...]
    generated_at: str | None = None
    profile_digest: str | None = None
    image: str | None = None
    image_id: str | None = None
    gpu_arch: str | None = None
    schema_version: int = 1

    @classmethod
    def from_results(
        cls,
        *,
        profile_id: str,
        results: Sequence[CheckResult],
        required_checks: Sequence[str],
        generated_at: str | None = None,
        profile_digest: str | None = None,
        image: str | None = None,
        image_id: str | None = None,
        gpu_arch: str | None = None,
    ) -> QualificationReport:
        result_tuple = tuple(results)
        required_tuple = tuple(required_checks)
        counts = Counter(result.name for result in result_tuple)
        by_name = {result.name: result for result in result_tuple}
        passed = all(
            counts[name] == 1 and by_name[name].passed for name in required_tuple
        )
        return cls(
            profile_id=profile_id,
            status="pass" if passed else "blocked",
            results=result_tuple,
            required_checks=required_tuple,
            generated_at=generated_at,
            profile_digest=profile_digest,
            image=image,
            image_id=image_id,
            gpu_arch=gpu_arch,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "status": self.status,
            "required_checks": list(self.required_checks),
            "results": [result.to_dict() for result in self.results],
            "generated_at": self.generated_at,
            "profile_digest": self.profile_digest,
            "image": self.image,
            "image_id": self.image_id,
            "gpu_arch": self.gpu_arch,
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def load_profile(path: Path) -> QualificationProfile:
    try:
        with path.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ProfileError(f"cannot read qualification profile {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ProfileError("qualification profile must be a TOML table")
    unknown = sorted(set(payload).difference(PROFILE_KEYS))
    missing = sorted(PROFILE_KEYS.difference(payload))
    if unknown:
        raise ProfileError("unknown qualification profile keys: " + ", ".join(unknown))
    if missing:
        raise ProfileError("missing qualification profile keys: " + ", ".join(missing))

    profile_id = _string(payload, "profile_id")
    if IDENTIFIER_PATTERN.fullmatch(profile_id) is None:
        raise ProfileError(f"invalid profile_id: {profile_id!r}")
    image = _string(payload, "image")
    if image.startswith("-") or any(character.isspace() for character in image):
        raise ProfileError("qualification image is invalid")
    rocm_version = _version(payload, "rocm_version")
    torch_version = _version(payload, "torch_version")
    gpu_arch = _string(payload, "gpu_arch")
    if GPU_ARCH_PATTERN.fullmatch(gpu_arch) is None:
        raise ProfileError(f"invalid gpu_arch: {gpu_arch!r}")
    stress_seconds = _bounded_integer(payload, "stress_seconds", 60, 3600)
    repeated_starts = _bounded_integer(payload, "repeated_starts", 2, 20)
    raw_checks = payload["required_checks"]
    if (
        not isinstance(raw_checks, list)
        or any(not isinstance(name, str) for name in raw_checks)
        or tuple(raw_checks) != REQUIRED_CHECKS
    ):
        raise ProfileError(
            "required_checks must contain every stable check exactly once in order"
        )
    return QualificationProfile(
        profile_id=profile_id,
        image=image,
        rocm_version=rocm_version,
        torch_version=torch_version,
        gpu_arch=gpu_arch,
        stress_seconds=stress_seconds,
        repeated_starts=repeated_starts,
        required_checks=tuple(raw_checks),
    )


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value or "\0" in value:
        raise ProfileError(f"{key} must be a nonempty string")
    return value


def _version(payload: Mapping[str, object], key: str) -> str:
    value = _string(payload, key)
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)+(?:[a-zA-Z0-9.+-]*)?", value) is None:
        raise ProfileError(f"invalid {key}: {value!r}")
    return value


def _bounded_integer(
    payload: Mapping[str, object],
    key: str,
    minimum: int,
    maximum: int,
) -> int:
    value = payload[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ProfileError(f"{key} must be from {minimum} through {maximum}")
    return value
