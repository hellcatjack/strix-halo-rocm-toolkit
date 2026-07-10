#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


PROTECTED_DISTRIBUTIONS = ("torch", "torchvision", "torchaudio", "triton")


def hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_manifest(
    distributions: Mapping[str, Iterable[Path]],
    path: Path,
    *,
    versions: Mapping[str, str] | None = None,
) -> None:
    package_records: list[dict[str, object]] = []
    versions = versions or {}
    for name in sorted(distributions):
        files = sorted(
            {Path(file).resolve() for file in distributions[name] if Path(file).is_file()},
            key=str,
        )
        if not files:
            raise ValueError(f"distribution has no regular files: {name}")
        root = Path(os.path.commonpath([str(file.parent) for file in files]))
        file_records = [
            {
                "path": str(file.relative_to(root)),
                "size": file.stat().st_size,
                "sha256": hash_file(file),
            }
            for file in files
        ]
        package_records.append(
            {
                "name": name,
                "version": versions.get(name),
                "root": str(root),
                "files": file_records,
            }
        )
    payload = {"schema_version": 1, "packages": package_records}
    _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def verify_manifest(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("packages"), list
    ):
        return ["invalid manifest schema"]

    errors: list[str] = []
    for package in payload["packages"]:
        name = package["name"]
        expected_version = package.get("version")
        if expected_version is not None:
            try:
                actual_version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                actual_version = "<missing>"
            if actual_version != expected_version:
                errors.append(
                    f"unexpected version: {name} expected {expected_version}, "
                    f"got {actual_version}"
                )

        root = Path(package["root"])
        for record in package["files"]:
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                errors.append(f"invalid path: {relative}")
                continue
            file = root / relative
            if not file.is_file():
                errors.append(f"missing: {file}")
            elif file.stat().st_size != record["size"] or hash_file(file) != record[
                "sha256"
            ]:
                errors.append(f"changed: {file}")
    return errors


def collect_installed_distributions() -> tuple[dict[str, list[Path]], dict[str, str]]:
    files_by_name: dict[str, list[Path]] = {}
    versions: dict[str, str] = {}
    for name in PROTECTED_DISTRIBUTIONS:
        distribution = importlib.metadata.distribution(name)
        files = distribution.files
        if files is None:
            raise RuntimeError(f"distribution has no file metadata: {name}")
        located = [
            Path(distribution.locate_file(file)).resolve()
            for file in files
            if Path(distribution.locate_file(file)).is_file()
        ]
        files_by_name[name] = located
        versions[name] = distribution.version
    return files_by_name, versions


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="torch-manifest.py")
    parser.add_argument("command", choices=("create", "verify"))
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    if args.command == "create":
        distributions, versions = collect_installed_distributions()
        write_manifest(distributions, args.manifest, versions=versions)
        return 0
    errors = verify_manifest(args.manifest)
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

