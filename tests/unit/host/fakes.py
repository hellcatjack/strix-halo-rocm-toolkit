from __future__ import annotations

from dataclasses import replace

from amd_ai.host.models import HostSnapshot
from amd_ai.runner import CommandResult


class FakeRunner:
    def __init__(
        self,
        responses: dict[tuple[str, ...], CommandResult] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.calls: list[tuple[str, ...]] = []

    @classmethod
    def healthy_target(cls, *, with_rocm64: bool = False) -> "FakeRunner":
        package_output = "linux-firmware\t20240318.git3b128b60-0ubuntu2\n"
        if with_rocm64:
            package_output += (
                "rocm-core:amd64\t6.4.43483-1\n"
                "hip-runtime-amd:amd64\t6.4.43483-1\n"
                "amdgpu-dkms\t6.12.12.60402-1\n"
                "dkms\t3.0.11-1ubuntu13\n"
                "zfs-dkms\t2.2.2-0ubuntu9\n"
            )
        values = {
            ("uname", "-m"): (0, "x86_64\n", ""),
            ("uname", "-r"): (0, "6.17.0-1025-oem\n", ""),
            ("uname", "-a"): (
                0,
                "Linux fixture 6.17.0-1025-oem #25-Ubuntu SMP x86_64 GNU/Linux\n",
                "",
            ),
            ("lspci", "-Dnnk", "-d", "1002:1586"): (
                0,
                "0000:c5:00.0 VGA compatible controller [0300]: AMD Device [1002:1586]\n"
                "\tKernel driver in use: amdgpu\n",
                "",
            ),
            ("dmidecode", "--type", "memory"): (0, "Size: 64 GB\nSize: 64 GB\n", ""),
            ("dpkg-query", "-W", "-f=${binary:Package}\\t${Version}\\n"): (
                0,
                package_output,
                "",
            ),
            ("dkms", "status"): (0, "", ""),
            ("docker", "version", "--format", "{{.Server.Version}}"): (0, "27.5.1\n", ""),
            ("docker", "version"): (0, "Docker version 27.5.1\n", ""),
            ("dmesg", "--color=never"): (
                0,
                "amdgpu: 512M of VRAM memory ready\n",
                "",
            ),
            ("getconf", "PAGESIZE"): (0, "4096\n", ""),
        }
        if with_rocm64:
            for package in ("rocm-core", "hip-runtime-amd"):
                values[("apt-cache", "policy", package)] = (
                    0,
                    " 500 https://repo.radeon.com/rocm/apt/6.4 noble/main amd64 Packages\n",
                    "",
                )
            values[("apt-cache", "policy", "amdgpu-dkms")] = (
                0,
                " 500 https://repo.radeon.com/amdgpu/6.4/ubuntu noble/main amd64 Packages\n",
                "",
            )
        responses = {
            args: CommandResult(args, returncode, stdout, stderr)
            for args, (returncode, stdout, stderr) in values.items()
        }
        return cls(responses)

    @classmethod
    def backup_only(cls) -> "FakeRunner":
        healthy = cls.healthy_target()
        backup_commands = {
            ("dpkg-query", "-W", "-f=${binary:Package}\\t${Version}\\n"),
            ("dkms", "status"),
            ("uname", "-a"),
            ("lspci", "-Dnnk", "-d", "1002:1586"),
            ("docker", "version"),
        }
        return cls(
            {
                args: result
                for args, result in healthy.responses.items()
                if args in backup_commands
            }
        )

    @classmethod
    def image_digest(cls, image: str, digest: str) -> "FakeRunner":
        args = ("docker", "image", "inspect", "--format", "{{.Id}}", image)
        return cls({args: CommandResult(args, 0, digest + "\n", "")})

    def run(
        self,
        args: list[str],
        *,
        check: bool = True,
        input_text: str | None = None,
    ) -> CommandResult:
        key = tuple(args)
        self.calls.append(key)
        if key not in self.responses:
            raise AssertionError(f"unregistered command: {key}")
        result = self.responses[key]
        if check and result.returncode != 0:
            raise AssertionError(f"fake command failed: {key}")
        return result


def healthy_snapshot(**changes: object) -> HostSnapshot:
    from amd_ai.host.probe import HostProbe

    snapshot = HostProbe(
        root="tests/fixtures/host/healthy",
        runner=FakeRunner.healthy_target(with_rocm64=bool(changes.pop("with_rocm64", False))),
        device_gids={"/dev/kfd": 109, "/dev/dri/renderD128": 110},
        current_group_ids=(109, 110),
    ).collect()
    return replace(snapshot, **changes)
