from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from urllib.parse import unquote, urlsplit

from amd_ai.image.profile import COMPONENTS, ProfileError, load_profile


CHUNK_SIZE = 8 * 1024 * 1024
PROGRESS_INTERVAL_BYTES = 64 * 1024**2
AMD_WHEEL_INDEX = "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/"
MANIFEST_NAME = "wheelhouse-manifest.json"
CHECKSUM_NAME = "wheelhouse.sha256"


class LockError(RuntimeError):
    pass


class DownloadError(LockError):
    pass


def hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(
    url: str,
    destination: Path,
    expected_sha256: str | None = None,
    progress: Callable[[int, int | None], None] | None = None,
) -> str:
    if destination.is_file() and expected_sha256 is not None:
        if hash_file(destination) == expected_sha256:
            return expected_sha256
        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "amd-ai-lock/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open(
            "wb"
        ) as output:
            total = _response_content_length(response)
            downloaded = 0
            next_progress = PROGRESS_INTERVAL_BYTES
            last_progress = -1
            if progress is not None:
                progress(0, total)
                last_progress = 0
            while chunk := response.read(CHUNK_SIZE):
                output.write(chunk)
                downloaded += len(chunk)
                if progress is not None and downloaded >= next_progress:
                    progress(downloaded, total)
                    last_progress = downloaded
                    while downloaded >= next_progress:
                        next_progress += PROGRESS_INTERVAL_BYTES
            if progress is not None and downloaded != last_progress:
                progress(downloaded, total)
            output.flush()
            os.fsync(output.fileno())
        digest = hash_file(partial)
        if expected_sha256 is not None and digest != expected_sha256:
            raise DownloadError(
                f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {digest}"
            )
        os.replace(partial, destination)
        _fsync_directory(destination.parent)
        return digest
    except (OSError, urllib.error.URLError) as error:
        raise DownloadError(f"download failed for {url}: {error}") from error
    finally:
        partial.unlink(missing_ok=True)


def _response_content_length(response: object) -> int | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Content-Length")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def render_verified_profile(source_text: str, digests: Mapping[str, str]) -> str:
    prefix_to_name = {prefix: name for name, prefix in COMPONENTS}
    inserted: set[str] = set()
    rendered: list[str] = []
    for line in source_text.splitlines():
        key, separator, _ = line.partition("=")
        if separator and key.endswith("_SHA256"):
            raise LockError("source profile must not contain SHA256 keys")
        rendered.append(line)
        if not separator or not key.endswith("_URL"):
            continue
        prefix = key.removesuffix("_URL")
        name = prefix_to_name.get(prefix)
        if name is None:
            continue
        if name not in digests:
            raise LockError(f"missing digest for {name}")
        rendered.append(f"{prefix}_SHA256={digests[name]}")
        inserted.add(name)
    missing = [name for name, _ in COMPONENTS if name not in inserted]
    if missing:
        raise LockError("source profile is missing wheel URLs: " + ", ".join(missing))
    return "\n".join(rendered) + "\n"


def write_wheelhouse_manifest(wheelhouse: Path) -> Path:
    records: list[dict[str, object]] = []
    for path in _artifact_paths(wheelhouse):
        records.append(
            {
                "filename": path.name,
                "size": path.stat().st_size,
                "sha256": hash_file(path),
            }
        )
    if not records:
        raise LockError(f"wheelhouse contains no artifacts: {wheelhouse}")
    manifest = wheelhouse / MANIFEST_NAME
    _write_text_atomic(
        manifest,
        json.dumps({"schema_version": 1, "files": records}, indent=2, sort_keys=True)
        + "\n",
    )
    checksum_lines = [
        f"{record['sha256']}  {record['filename']}" for record in records
    ]
    _write_text_atomic(
        wheelhouse / CHECKSUM_NAME,
        "\n".join(checksum_lines) + "\n",
    )
    return manifest


def validate_wheelhouse_manifest(wheelhouse: Path) -> tuple[str, ...]:
    manifest = wheelhouse / MANIFEST_NAME
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        return (f"invalid manifest: {error}",)
    if payload.get("schema_version") != 1 or not isinstance(payload.get("files"), list):
        return ("invalid manifest schema",)

    errors: list[str] = []
    expected_names: set[str] = set()
    for record in payload["files"]:
        if not isinstance(record, dict) or not isinstance(record.get("filename"), str):
            errors.append("invalid manifest record")
            continue
        name = record["filename"]
        if name in expected_names or Path(name).name != name:
            errors.append(f"invalid filename: {name}")
            continue
        expected_names.add(name)
        path = wheelhouse / name
        if not path.is_file():
            errors.append(f"missing: {name}")
            continue
        if path.stat().st_size != record.get("size") or hash_file(path) != record.get(
            "sha256"
        ):
            errors.append(f"changed: {name}")

    actual_names = {path.name for path in _artifact_paths(wheelhouse)}
    errors.extend(f"untracked: {name}" for name in sorted(actual_names - expected_names))
    return tuple(errors)


def parse_package_lock(text: str) -> tuple[tuple[str, str], ...]:
    packages: list[tuple[str, str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line:
            continue
        match = re.fullmatch(
            r"([a-z0-9][a-z0-9+.-]*)=([^\s=]+)",
            raw_line,
        )
        if match is None:
            raise LockError(f"invalid package lock line {line_number}: {raw_line!r}")
        packages.append((match.group(1), match.group(2)))
    if not packages:
        raise LockError("package lock is empty")
    names = [name for name, _ in packages]
    if names != sorted(names):
        raise LockError("package lock is not sorted")
    if len(names) != len(set(names)):
        raise LockError("package lock contains duplicate names")
    return tuple(packages)


def lock_wheels(*, sources: Path, profile_path: Path, wheelhouse: Path) -> None:
    source_text = sources.read_text(encoding="utf-8")
    source_values = _parse_source_values(source_text)
    existing_hashes = _matching_existing_hashes(profile_path, source_values)
    digests: dict[str, str] = {}
    for name, prefix in COMPONENTS:
        url = source_values[f"{prefix}_URL"]
        filename = _filename_from_url(url)
        print(f"locking {name}: {filename}", file=sys.stderr, flush=True)
        digests[name] = download(
            url,
            wheelhouse / filename,
            existing_hashes.get(name),
        )
        size = (wheelhouse / filename).stat().st_size
        print(
            f"locked {name}: {size} bytes sha256={digests[name][:16]}...",
            file=sys.stderr,
            flush=True,
        )

    rendered = render_verified_profile(source_text, digests)
    _write_text_atomic(profile_path, rendered)
    try:
        load_profile(profile_path, allow_verified=True)
    except ProfileError as error:
        raise LockError(f"generated profile is invalid: {error}") from error

    lock_input = Path(".cache/locks/stable.in")
    lock_input.parent.mkdir(parents=True, exist_ok=True)
    requirements = [
        f"{name}=={source_values[f'{prefix}_VERSION']}"
        for name, prefix in COMPONENTS
    ]
    _write_text_atomic(lock_input, "\n".join(requirements) + "\n")
    requirements_lock = profile_path.with_name("stable.requirements.lock")
    _run(
        (
            "uv",
            "pip",
            "compile",
            "--python-version",
            "3.12",
            "--python-platform",
            "x86_64-unknown-linux-gnu",
            "--find-links",
            AMD_WHEEL_INDEX,
            "--generate-hashes",
            "--output-file",
            str(requirements_lock),
            str(lock_input),
        )
    )
    print(f"compiled dependency lock: {requirements_lock}", file=sys.stderr, flush=True)
    lock_text = requirements_lock.read_text(encoding="utf-8")
    _require_amd_primary_versions(lock_text, source_values)
    _run(
        (
            "python3",
            "-m",
            "pip",
            "download",
            "--require-hashes",
            "--dest",
            str(wheelhouse),
            "--requirement",
            str(requirements_lock),
            "--find-links",
            AMD_WHEEL_INDEX,
        )
    )
    write_wheelhouse_manifest(wheelhouse)
    print(f"wrote wheelhouse manifest: {wheelhouse / MANIFEST_NAME}", file=sys.stderr)


def _parse_source_values(text: str) -> dict[str, str]:
    expected = {
        "PROFILE_ID",
        "PROFILE_STATUS",
        "ROCM_VERSION",
        "PYTHON_ABI",
        "PLATFORM",
        *(
            f"{prefix}_{field}"
            for _, prefix in COMPONENTS
            for field in ("VERSION", "URL")
        ),
    }
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if any(token in line for token in ("`", "$(", "${")):
            raise LockError(f"shell substitution on source line {line_number}")
        key, separator, value = line.partition("=")
        if not separator or key not in expected or not value:
            raise LockError(f"invalid source key on line {line_number}: {key}")
        if key in values:
            raise LockError(f"duplicate source key: {key}")
        values[key] = value
    missing = sorted(expected.difference(values))
    if missing:
        raise LockError("missing source keys: " + ", ".join(missing))
    if values["PROFILE_STATUS"] != "verified":
        raise LockError("stable source profile must be verified")
    if (
        values["ROCM_VERSION"] != "7.2.1"
        or values["PYTHON_ABI"] != "cp312"
        or values["PLATFORM"] != "linux/amd64"
    ):
        raise LockError("source profile does not match the locked image family")
    return values


def _matching_existing_hashes(
    profile_path: Path,
    source_values: Mapping[str, str],
) -> dict[str, str]:
    if not profile_path.is_file():
        return {}
    try:
        profile = load_profile(profile_path, allow_verified=True)
    except (OSError, ProfileError):
        return {}
    hashes: dict[str, str] = {}
    for name, prefix in COMPONENTS:
        wheel = profile.wheels[name]
        if (
            wheel.url == source_values[f"{prefix}_URL"]
            and wheel.version == source_values[f"{prefix}_VERSION"]
        ):
            hashes[name] = wheel.sha256
    return hashes


def _require_amd_primary_versions(
    lock_text: str,
    source_values: Mapping[str, str],
) -> None:
    for name, prefix in COMPONENTS:
        match = re.search(rf"^{re.escape(name)}==([^\s\\]+)", lock_text, re.MULTILINE)
        expected = source_values[f"{prefix}_VERSION"] + "+rocm7.2.1"
        if match is None or not match.group(1).startswith(expected):
            resolved = match.group(1) if match else "<missing>"
            raise LockError(f"{name} resolved to non-AMD version {resolved}")


def _filename_from_url(url: str) -> str:
    filename = unquote(Path(urlsplit(url).path).name)
    if not filename.endswith(".whl") or Path(filename).name != filename:
        raise LockError(f"URL does not identify a wheel: {url}")
    return filename


def _artifact_paths(wheelhouse: Path) -> list[Path]:
    excluded = {MANIFEST_NAME, CHECKSUM_NAME}
    return sorted(
        (
            path
            for path in wheelhouse.iterdir()
            if path.is_file() and path.name not in excluded and path.suffix != ".part"
        ),
        key=lambda path: path.name,
    )


def _run(argv: tuple[str, ...]) -> None:
    completed = subprocess.run(
        argv,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        evidence = completed.stderr.strip() or completed.stdout.strip()
        raise LockError(f"command failed ({completed.returncode}): {' '.join(argv)}: {evidence}")


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lock-wheels")
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True, dest="profile_path")
    parser.add_argument("--wheelhouse", type=Path, required=True)
    args = parser.parse_args(argv)
    lock_wheels(
        sources=args.sources,
        profile_path=args.profile_path,
        wheelhouse=args.wheelhouse,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
