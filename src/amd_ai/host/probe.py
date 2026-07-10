from __future__ import annotations

import json
import os
from pathlib import Path

from amd_ai.host.models import AptSourceFile, HostSnapshot, InstalledPackage
from amd_ai.host.parsers import (
    parse_apt_policy_origin,
    parse_cmdline,
    parse_dmi_memory_bytes,
    parse_dpkg_packages,
    parse_lspci_gpu,
    parse_meminfo,
    parse_os_release,
    parse_vram_mib,
)
from amd_ai.runner import CommandResult, Runner


PACKAGE_PREFIXES = (
    "rocm",
    "hip-",
    "hsa-",
    "hsakmt",
    "comgr",
    "miopen",
    "rocblas",
    "rocfft",
    "rocrand",
    "rocsolver",
    "rocsparse",
    "rccl",
)


class FixtureRunner:
    def __init__(self, responses: dict[tuple[str, ...], CommandResult]) -> None:
        self.responses = responses

    @classmethod
    def from_root(cls, root: str | Path) -> FixtureRunner:
        path = Path(root) / "commands.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        responses: dict[tuple[str, ...], CommandResult] = {}
        for joined_args, record in payload.items():
            args = tuple(joined_args.split("\0"))
            responses[args] = CommandResult(
                args,
                int(record["returncode"]),
                str(record["stdout"]),
                str(record["stderr"]),
            )
        return cls(responses)

    def run(
        self,
        args: list[str],
        *,
        check: bool = True,
        input_text: str | None = None,
    ) -> CommandResult:
        key = tuple(args)
        if key not in self.responses:
            raise AssertionError(f"unregistered fixture command: {key}")
        result = self.responses[key]
        if check and result.returncode != 0:
            raise RuntimeError(f"fixture command failed: {key}")
        return result


def load_fixture_device_gids(root: str | Path) -> dict[str, int]:
    path = Path(root) / "devices.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid fixture device map: {path}")
    return {str(device): int(gid) for device, gid in payload.items()}


class HostProbe:
    def __init__(
        self,
        *,
        root: str | Path,
        runner: Runner,
        device_gids: dict[str, int] | None = None,
        current_group_ids: tuple[int, ...] | None = None,
    ) -> None:
        self.root = Path(root)
        self.runner = runner
        self._device_gids = device_gids
        self._current_group_ids = current_group_ids

    def collect(self) -> HostSnapshot:
        os_release = parse_os_release(self._read("etc/os-release"))
        meminfo = parse_meminfo(self._read("proc/meminfo"))
        cmdline = parse_cmdline(self._read("proc/cmdline"))
        architecture = self._run(["uname", "-m"]).stdout.strip()
        kernel = self._run(["uname", "-r"]).stdout.strip()
        gpu_result = self._run(["lspci", "-Dnnk", "-d", "1002:1586"])
        dmi_result = self._run(["dmidecode", "--type", "memory"])
        package_result = self._run(
            ["dpkg-query", "-W", "-f=${binary:Package}\\t${Version}\\n"]
        )
        dkms_result = self._run(["dkms", "status"])
        docker_result = self._run(
            ["docker", "version", "--format", "{{.Server.Version}}"]
        )
        dmesg_result = self._run(["dmesg", "--color=never"])
        page_size_result = self._run(["getconf", "PAGESIZE"])

        packages = tuple(
            InstalledPackage(name, version, self._package_origin(name))
            for name, version in parse_dpkg_packages(package_result.stdout)
        )
        ttm_text = self._read("sys/module/ttm/parameters/pages_limit", required=False)
        return HostSnapshot(
            os_id=os_release.get("ID", ""),
            os_version=os_release.get("VERSION_ID", ""),
            architecture=architecture,
            kernel=kernel,
            gpu=parse_lspci_gpu(gpu_result.stdout),
            mem_total_kib=meminfo.get("MemTotal", 0),
            swap_total_kib=meminfo.get("SwapTotal", 0),
            page_size=self._parse_int(page_size_result.stdout),
            kernel_args=cmdline,
            ttm_pages_limit=self._parse_optional_int(ttm_text),
            dmi_memory_bytes=parse_dmi_memory_bytes(dmi_result.stdout),
            device_gids=self._collect_device_gids(),
            current_group_ids=self._current_group_ids
            if self._current_group_ids is not None
            else tuple(os.getgroups()),
            packages=packages,
            apt_sources=self._read_apt_sources(),
            dkms_status=dkms_result.stdout.strip(),
            docker_version=docker_result.stdout.strip() if docker_result.returncode == 0 else None,
            dmesg=dmesg_result.stdout,
            dedicated_vram_mib=parse_vram_mib(dmesg_result.stdout),
        )

    def _run(self, args: list[str]) -> CommandResult:
        try:
            return self.runner.run(args, check=False)
        except OSError as error:
            return CommandResult(tuple(args), 127, "", str(error))

    def _package_origin(self, name: str) -> str | None:
        if name != "amdgpu-dkms" and not name.startswith(PACKAGE_PREFIXES):
            return None
        result = self._run(["apt-cache", "policy", name])
        return parse_apt_policy_origin(result.stdout)

    def _read(self, relative_path: str, *, required: bool = True) -> str:
        path = self.root / relative_path
        try:
            return path.read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError):
            if required:
                raise
            return ""

    def _read_apt_sources(self) -> tuple[AptSourceFile, ...]:
        paths = [self.root / "etc/apt/sources.list"]
        source_dir = self.root / "etc/apt/sources.list.d"
        if source_dir.is_dir():
            paths.extend(sorted(path for path in source_dir.iterdir() if path.is_file()))
        records: list[AptSourceFile] = []
        for path in paths:
            try:
                content = path.read_text(encoding="utf-8")
            except (FileNotFoundError, PermissionError, UnicodeDecodeError):
                continue
            records.append(AptSourceFile(path=str(path.relative_to(self.root)), content=content))
        return tuple(records)

    def _collect_device_gids(self) -> dict[str, int]:
        if self._device_gids is not None:
            return dict(self._device_gids)
        candidates = [self.root / "dev/kfd"]
        dri = self.root / "dev/dri"
        if dri.is_dir():
            candidates.extend(sorted(dri.glob("renderD*")))
        result: dict[str, int] = {}
        for path in candidates:
            try:
                gid = path.stat().st_gid
            except (FileNotFoundError, PermissionError):
                continue
            canonical = "/" + str(path.relative_to(self.root))
            result[canonical] = gid
        return result

    @staticmethod
    def _parse_int(value: str) -> int:
        try:
            return int(value.strip())
        except ValueError:
            return 0

    @staticmethod
    def _parse_optional_int(value: str) -> int | None:
        try:
            return int(value.strip())
        except ValueError:
            return None
