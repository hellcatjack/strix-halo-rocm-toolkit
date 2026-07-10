from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from amd_ai.image.lock import LockError, download, hash_file, parse_package_lock


UBUNTU_TAG = "ubuntu:24.04"
UV_TAG = "ghcr.io/astral-sh/uv:0.11.28"
UV_VERSION = "0.11.28"
ROCM_KEY_URL = "https://repo.radeon.com/rocm/rocm.gpg.key"


RESOLVER_SCRIPT = r"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update >&2
apt-get install -y --no-install-recommends ca-certificates python3-minimal >&2
cat >/etc/apt/sources.list.d/amd-rocm.list <<'EOF'
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2.1 noble main
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/graphics/7.2.1/ubuntu noble main
EOF
cat >/etc/apt/preferences.d/rocm-pin-600 <<'EOF'
Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 600
EOF
apt-get update >&2
apt-get install -y --no-install-recommends rocm-hip-sdk rocm-ml-sdk >&2
echo AMD_AI_LOCK_BEGIN
python3 - <<'PY'
import re
import subprocess

installed = []
output = subprocess.run(
    ["dpkg-query", "-W", "-f=${binary:Package}\\t${Version}\\n"],
    check=True,
    text=True,
    stdout=subprocess.PIPE,
).stdout
for line in output.splitlines():
    name, version = line.split("\t", 1)
    name = name.split(":", 1)[0]
    policy = subprocess.run(
        ["apt-cache", "policy", name],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    active = False
    from_amd = False
    for policy_line in policy.splitlines():
        stripped = policy_line.strip()
        if stripped.startswith("*** "):
            fields = stripped.split()
            active = len(fields) > 1 and fields[1] == version
            continue
        indentation = len(policy_line) - len(policy_line.lstrip(" "))
        if active and indentation <= 5 and stripped:
            active = False
        if active and "repo.radeon.com" in policy_line:
            from_amd = True
            break
    if from_amd:
        installed.append((name, version))

for name, version in sorted(set(installed)):
    print(f"{name}={version}")
PY
echo AMD_AI_LOCK_END
"""


class DockerClient:
    def __init__(self, prefix: tuple[str, ...]) -> None:
        self.prefix = prefix

    @classmethod
    def detect(cls) -> DockerClient:
        for prefix in (("docker",), ("sudo", "-n", "docker")):
            completed = subprocess.run(
                (*prefix, "info", "--format", "{{.ServerVersion}}"),
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if completed.returncode == 0:
                return cls(prefix)
        raise LockError("Docker daemon is unavailable to the user and sudo -n")

    def run(
        self,
        args: tuple[str, ...],
        *,
        input_text: str | None = None,
        inherit_stderr: bool = False,
    ) -> str:
        completed = subprocess.run(
            (*self.prefix, *args),
            check=False,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=None if inherit_stderr else subprocess.PIPE,
        )
        if completed.returncode != 0:
            evidence = "" if inherit_stderr else completed.stderr.strip()
            raise LockError(
                f"Docker command failed ({completed.returncode}): "
                f"{' '.join(args)}: {evidence}"
            )
        return completed.stdout


def lock_rocm_packages(*, rocm_version: str, ubuntu: str) -> None:
    if rocm_version != "7.2.1" or ubuntu != "noble":
        raise LockError("this resolver is locked to ROCm 7.2.1 on Ubuntu Noble")
    docker = DockerClient.detect()
    print(f"using Docker command: {' '.join(docker.prefix)}", file=sys.stderr)

    print(f"pulling {UBUNTU_TAG} for linux/amd64", file=sys.stderr, flush=True)
    docker.run(("pull", "--platform", "linux/amd64", UBUNTU_TAG))
    ubuntu_digest = _repo_digest(docker, UBUNTU_TAG, "ubuntu")
    print(f"resolved Ubuntu: {ubuntu_digest}", file=sys.stderr, flush=True)

    print(f"pulling {UV_TAG} for linux/amd64", file=sys.stderr, flush=True)
    docker.run(("pull", "--platform", "linux/amd64", UV_TAG))
    uv_digest = _repo_digest(docker, UV_TAG, "ghcr.io/astral-sh/uv")
    print(f"resolved uv: {uv_digest}", file=sys.stderr, flush=True)
    _write_text_atomic(
        Path("profiles/base-images.lock"),
        f"UBUNTU_24_04={ubuntu_digest}\n"
        f"UV_IMAGE={uv_digest}\n"
        f"UV_VERSION={UV_VERSION}\n",
    )

    rocm_dir = Path("profiles/rocm")
    rocm_dir.mkdir(parents=True, exist_ok=True)
    armored_key = rocm_dir / "rocm.gpg.key"
    dearmored_temp = rocm_dir / ".rocm.gpg.tmp"
    print("downloading and dearmoring the ROCm repository key", file=sys.stderr)
    download(ROCM_KEY_URL, armored_key)
    try:
        _run_host(
            (
                "gpg",
                "--batch",
                "--yes",
                "--dearmor",
                "--output",
                str(dearmored_temp),
                str(armored_key),
            )
        )
        if not dearmored_temp.is_file() or dearmored_temp.stat().st_size == 0:
            raise LockError("gpg produced an empty ROCm keyring")
        rocm_key = rocm_dir / "rocm.gpg"
        os.replace(dearmored_temp, rocm_key)
        digest = hash_file(rocm_key)
        _write_text_atomic(
            rocm_dir / "rocm.gpg.sha256",
            f"{digest}  rocm.gpg\n",
        )
    finally:
        armored_key.unlink(missing_ok=True)
        dearmored_temp.unlink(missing_ok=True)

    print("resolving installed AMD packages in an ephemeral container", file=sys.stderr)
    resolver_output = docker.run(
        build_resolver_argv(
            ubuntu_digest=ubuntu_digest,
            key_path=Path("profiles/rocm/rocm.gpg"),
        ),
        input_text=RESOLVER_SCRIPT,
        inherit_stderr=True,
    )
    package_text = _extract_package_lock(resolver_output)
    packages = parse_package_lock(package_text)
    if not any(name == "rocm-core" for name, _ in packages):
        raise LockError("resolved package set does not contain rocm-core")
    lock_path = rocm_dir / "7.2.1-packages.lock"
    _write_text_atomic(
        lock_path,
        "".join(f"{name}={version}\n" for name, version in packages),
    )
    print(f"locked {len(packages)} AMD packages in {lock_path}", file=sys.stderr)


def build_resolver_argv(*, ubuntu_digest: str, key_path: Path) -> tuple[str, ...]:
    return (
        "run",
        "--rm",
        "--interactive",
        "--platform",
        "linux/amd64",
        "--mount",
        f"type=bind,src={key_path.resolve()},"
        "dst=/etc/apt/keyrings/rocm.gpg,readonly",
        ubuntu_digest,
        "bash",
        "-s",
    )


def _repo_digest(
    docker: DockerClient,
    image: str,
    repository: str,
) -> str:
    output = docker.run(("image", "inspect", "--format", "{{json .RepoDigests}}", image))
    try:
        digests = json.loads(output)
    except json.JSONDecodeError as error:
        raise LockError(f"invalid RepoDigests for {image}") from error
    prefix = repository + "@sha256:"
    matches = sorted(digest for digest in digests if digest.startswith(prefix))
    if len(matches) != 1 or not re.fullmatch(
        rf"{re.escape(repository)}@sha256:[0-9a-f]{{64}}",
        matches[0],
    ):
        raise LockError(f"could not resolve one immutable digest for {image}: {digests}")
    return matches[0]


def _extract_package_lock(output: str) -> str:
    begin = "AMD_AI_LOCK_BEGIN\n"
    end = "AMD_AI_LOCK_END"
    if begin not in output or end not in output:
        raise LockError("resolver output does not contain package lock markers")
    body = output.split(begin, 1)[1].split(end, 1)[0]
    return body.strip() + "\n"


def _run_host(argv: tuple[str, ...]) -> None:
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
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lock-rocm-packages")
    parser.add_argument("--rocm-version", required=True)
    parser.add_argument("--ubuntu", required=True)
    args = parser.parse_args(argv)
    lock_rocm_packages(rocm_version=args.rocm_version, ubuntu=args.ubuntu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
