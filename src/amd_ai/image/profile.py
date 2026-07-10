from __future__ import annotations

import argparse
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit


BASE_KEYS = (
    "PROFILE_ID",
    "PROFILE_STATUS",
    "ROCM_VERSION",
    "PYTHON_ABI",
    "PLATFORM",
)
COMPONENTS = (
    ("torch", "TORCH"),
    ("torchvision", "TORCHVISION"),
    ("torchaudio", "TORCHAUDIO"),
    ("triton", "TRITON"),
)
WHEEL_KEYS = tuple(
    f"{prefix}_{field}"
    for _, prefix in COMPONENTS
    for field in ("VERSION", "URL", "SHA256")
)
REQUIRED_KEYS = BASE_KEYS + WHEEL_KEYS
REQUIRED_KEY_SET = frozenset(REQUIRED_KEYS)


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class WheelSpec:
    name: str
    version: str
    url: str
    sha256: str


@dataclass(frozen=True)
class TorchProfile:
    profile_id: str
    status: str
    rocm_version: str
    python_abi: str
    platform: str
    wheels: Mapping[str, WheelSpec]


def load_profile(
    path: str | Path,
    *,
    allow_verified: bool,
) -> TorchProfile:
    profile_path = Path(path)
    values = _parse_values(profile_path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_KEYS if key not in values]
    if missing:
        raise ProfileError("missing profile keys: " + ", ".join(missing))

    profile_id = values["PROFILE_ID"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", profile_id):
        raise ProfileError(f"invalid PROFILE_ID: {profile_id!r}")

    status = values["PROFILE_STATUS"]
    if status not in {"verified", "experimental"}:
        raise ProfileError(f"invalid PROFILE_STATUS: {status!r}")
    if status == "verified" and not allow_verified:
        raise ProfileError("verified status is restricted to the repository stable profile")

    required_base = {
        "ROCM_VERSION": "7.2.1",
        "PYTHON_ABI": "cp312",
        "PLATFORM": "linux/amd64",
    }
    for key, expected in required_base.items():
        if values[key] != expected:
            raise ProfileError(f"{key} must be {expected!r} for this image family")

    wheels: dict[str, WheelSpec] = {}
    for name, prefix in COMPONENTS:
        version = values[f"{prefix}_VERSION"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_-]*", version):
            raise ProfileError(f"invalid {prefix}_VERSION: {version!r}")
        url = values[f"{prefix}_URL"]
        _validate_url(prefix, url)
        digest = values[f"{prefix}_SHA256"]
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ProfileError(f"invalid {prefix}_SHA256")
        if digest == "0" * 64:
            raise ProfileError(f"all-zero {prefix}_SHA256 must be replaced")
        wheels[name] = WheelSpec(name, version, url, digest)

    return TorchProfile(
        profile_id=profile_id,
        status=status,
        rocm_version=values["ROCM_VERSION"],
        python_abi=values["PYTHON_ABI"],
        platform=values["PLATFORM"],
        wheels=MappingProxyType(wheels),
    )


def _parse_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if any(token in line for token in ("`", "$(", "${")):
            raise ProfileError(f"shell substitution is forbidden on line {line_number}")
        key, separator, value = line.partition("=")
        if not separator or not key or not value:
            raise ProfileError(f"invalid KEY=VALUE entry on line {line_number}")
        if key != key.strip() or any(character.isspace() for character in key):
            raise ProfileError(f"invalid key whitespace on line {line_number}")
        if key not in REQUIRED_KEY_SET:
            raise ProfileError(f"unknown profile key: {key}")
        if key in values:
            raise ProfileError(f"duplicate profile key: {key}")
        values[key] = value
    return values


def _validate_url(prefix: str, url: str) -> None:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise ProfileError(f"invalid {prefix}_URL") from error
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith(".whl")
        or any(character.isspace() for character in url)
    ):
        raise ProfileError(f"unsafe {prefix}_URL")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m amd_ai.image.profile")
    parser.add_argument("profile", type=Path)
    args = parser.parse_args(argv)
    stable = Path(__file__).resolve().parents[3] / "profiles/torch/stable.env"
    allow_verified = args.profile.resolve() == stable.resolve()
    profile = load_profile(args.profile, allow_verified=allow_verified)
    print(f"{profile.status} {profile.profile_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

