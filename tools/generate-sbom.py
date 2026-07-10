#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path


def build_spdx(
    *,
    name: str,
    namespace: str,
    os_packages: Iterable[tuple[str, str]],
    python_packages: Iterable[tuple[str, str]],
    created: str,
) -> dict[str, object]:
    packages = sorted(
        {
            (package_namespace, package_name.strip(), version.strip())
            for package_namespace, values in (
                ("os", os_packages),
                ("python", python_packages),
            )
            for package_name, version in values
            if package_name.strip() and version.strip()
        }
    )
    package_records = [
        _package_record(package_namespace, package_name, version)
        for package_namespace, package_name, version in packages
    ]
    relationships = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": package["SPDXID"],
        }
        for package in package_records
    ]
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": name,
        "documentNamespace": namespace,
        "creationInfo": {
            "created": created,
            "creators": ["Tool: amd-ai-container-platform"],
        },
        "packages": package_records,
        "relationships": relationships,
    }


def collect_os_packages() -> tuple[tuple[str, str], ...]:
    completed = subprocess.run(
        (
            "dpkg-query",
            "-W",
            "-f=${binary:Package}\\t${Version}\\n",
        ),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        evidence = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"dpkg-query failed: {evidence}")
    packages = []
    for line in completed.stdout.splitlines():
        name, separator, version = line.partition("\t")
        if separator and name and version:
            packages.append((name, version))
    return tuple(packages)


def collect_python_packages() -> tuple[tuple[str, str], ...]:
    packages = set()
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name", "")).strip()
        version = str(distribution.version).strip()
        if name and version:
            packages.add((name, version))
    return tuple(sorted(packages))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--created", required=True)
    parser.add_argument("--namespace")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    namespace = args.namespace or _default_namespace(args.name, args.created)
    document = build_spdx(
        name=args.name,
        namespace=namespace,
        os_packages=collect_os_packages(),
        python_packages=collect_python_packages(),
        created=args.created,
    )
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output == "-":
        print(rendered, end="")
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    return 0


def _package_record(
    package_namespace: str,
    name: str,
    version: str,
) -> dict[str, object]:
    identity = f"{package_namespace}\0{name}\0{version}".encode("utf-8")
    identifier = hashlib.sha256(identity).hexdigest()[:24]
    return {
        "name": name,
        "SPDXID": f"SPDXRef-Package-{identifier}",
        "versionInfo": version,
        "downloadLocation": "NOASSERTION",
        "filesAnalyzed": False,
        "comment": f"package namespace: {package_namespace}",
    }


def _default_namespace(name: str, created: str) -> str:
    digest = hashlib.sha256(f"{name}\0{created}".encode("utf-8")).hexdigest()
    return f"https://amd-ai.invalid/spdx/{digest}"


if __name__ == "__main__":
    raise SystemExit(main())
